(* The performance kernels, checked against plain reference implementations,
   plus the guarantees the fast paths rely on: gradients of the fused ops,
   gradient-buffer ownership, and results that don't depend on the thread
   count. Run with: dune exec test/test_kernels.exe

   Parallel.min_parallel_work is set to 0 for these tests, so even tiny
   shapes are split into chunks across threads. That way the parallel code
   paths are exercised and checked, not only the sequential fallback. *)

open Flockml
open OUnit2
open Helpers

let () = Parallel.min_parallel_work := 0

(* ------------------------------------------------------------------------ *)
(* Reference implementations: as plain as possible                           *)
(* ------------------------------------------------------------------------ *)

(** Strides of a contiguous row-major shape. *)
let strides shape =
  let n = Array.length shape in
  let s = Array.make n 1 in
  for i = n - 2 downto 0 do
    s.(i) <- s.(i + 1) * shape.(i + 1)
  done;
  s

(** Multi-index of flat position [i] in [shape]. *)
let unravel shape i =
  let idx = Array.make (Array.length shape) 0 and r = ref i in
  for ax = Array.length shape - 1 downto 0 do
    idx.(ax) <- !r mod shape.(ax);
    r := !r / shape.(ax)
  done;
  idx

(** Flat offset, in a tensor of shape [src], of the element that big-shape
    multi-index [idx] reads under broadcasting. *)
let source src idx =
  let k = Array.length idx - Array.length src and st = strides src in
  let o = ref 0 in
  Array.iteri (fun s d -> if d <> 1 then o := !o + (idx.(s + k) * st.(s))) src;
  !o

let ref_binop f a b =
  let shape = Tensor.broadcast_shapes (Tensor.shape a) (Tensor.shape b) in
  let ad = Tensor.to_array a and bd = Tensor.to_array b in
  let n = Array.fold_left ( * ) 1 shape in
  t shape
    (Array.init n (fun i ->
         let idx = unravel shape i in
         f ad.(source (Tensor.shape a) idx) bd.(source (Tensor.shape b) idx)))

let ref_sum_to_shape x target =
  let shape = Tensor.shape x and xd = Tensor.to_array x in
  let out = Array.make (Array.fold_left ( * ) 1 target) 0.0 in
  Array.iteri
    (fun i v ->
      let o = source target (unravel shape i) in
      out.(o) <- out.(o) +. v)
    xd;
  t target out

(** [alpha * op(a) op(b) + beta * c], straight from the definition. *)
let ref_gemm ~ta ~tb ~alpha ~beta a b c =
  let sa = Tensor.shape a and sb = Tensor.shape b in
  let m = if ta then sa.(1) else sa.(0) and k = if ta then sa.(0) else sa.(1) in
  let n = if tb then sb.(0) else sb.(1) in
  let at i p =
    if ta then Tensor.get a [| p; i |] else Tensor.get a [| i; p |]
  in
  let bt p j =
    if tb then Tensor.get b [| j; p |] else Tensor.get b [| p; j |]
  in
  t [| m; n |]
    (Array.init (m * n) (fun ij ->
         let i = ij / n and j = ij mod n in
         let s = ref 0.0 in
         for p = 0 to k - 1 do
           s := !s +. (at i p *. bt p j)
         done;
         (alpha *. !s) +. (beta *. Tensor.get c [| i; j |])))

let ref_im2col x ~kh ~kw ~stride ~padding =
  let s = Tensor.shape x in
  let n = s.(0) and h = s.(1) and w = s.(2) and c = s.(3) in
  let oh = ((h + (2 * padding) - kh) / stride) + 1
  and ow = ((w + (2 * padding) - kw) / stride) + 1 in
  let cols = kh * kw * c in
  t
    [| n * oh * ow; cols |]
    (Array.init
       (n * oh * ow * cols)
       (fun i ->
         let row = i / cols and col = i mod cols in
         let b = row / (oh * ow) and oy = row / ow mod oh and ox = row mod ow in
         let ky = col / (kw * c) and kx = col / c mod kw and ch = col mod c in
         let y = (oy * stride) + ky - padding
         and xx = (ox * stride) + kx - padding in
         if y < 0 || y >= h || xx < 0 || xx >= w then 0.0
         else Tensor.get x [| b; y; xx; ch |]))

let ref_max_pool x ~size ~stride =
  let s = Tensor.shape x in
  let n = s.(0) and h = s.(1) and w = s.(2) and c = s.(3) in
  let oh = ((h - size) / stride) + 1 and ow = ((w - size) / stride) + 1 in
  t [| n; oh; ow; c |]
    (Array.init
       (n * oh * ow * c)
       (fun o ->
         let idx = unravel [| n; oh; ow; c |] o in
         let best = ref Float.neg_infinity in
         for ky = 0 to size - 1 do
           for kx = 0 to size - 1 do
             let v =
               Tensor.get x
                 [|
                   idx.(0);
                   (idx.(1) * stride) + ky;
                   (idx.(2) * stride) + kx;
                   idx.(3);
                 |]
             in
             if v > !best then best := v
           done
         done;
         !best))

(** Bit-for-bit equality (NaN-free inputs). *)
let assert_identical ?(msg = "bit-identical") a b =
  assert_shape ~msg (Tensor.shape a) b;
  assert_bool msg (Tensor.to_array a = Tensor.to_array b)

(** Equal up to rounding in the last bit or so. The fused optimizer kernels do
    the same arithmetic as the unfused version, but ocamlopt's arm64 backend may
    fuse a different multiply-add pair, which rounds differently. *)
let assert_last_bits ?msg a b = assert_tensor ~rtol:1e-14 ~atol:1e-15 ?msg a b

(* ------------------------------------------------------------------------ *)
(* Kernels against the references                                           *)
(* ------------------------------------------------------------------------ *)

(** Shapes chosen to hit edge tiles (not multiples of 4 or 64), k longer than
    one 256-wide block, vectors, and the split-k path (k >= 4096 with a tiny
    result). *)
let gemm_shapes =
  [
    (1, 1, 1);
    (3, 5, 2);
    (4, 4, 4);
    (7, 13, 9);
    (65, 70, 67);
    (5, 300, 6);
    (64, 784, 10);
    (9, 5000, 8);
    (2, 4100, 3);
    (130, 9, 8);
  ]

let gemm_tests =
  List.concat_map
    (fun (m, k, n) ->
      List.map
        (fun (ta, tb) ->
          let name =
            Printf.sprintf "gemm %dx%dx%d%s%s" m k n
              (if ta then " a^T" else "")
              (if tb then " b^T" else "")
          in
          name >:: fun _ ->
          let a = rand_tensor (if ta then [| k; m |] else [| m; k |]) in
          let b = rand_tensor (if tb then [| n; k |] else [| k; n |]) in
          let c0 = rand_tensor [| m; n |] in
          List.iter
            (fun (alpha, beta) ->
              let c = Tensor.copy c0 in
              Tensor.gemm_ ~trans_a:ta ~trans_b:tb ~alpha ~beta a b ~c;
              assert_tensor ~rtol:1e-9 ~atol:1e-9
                ~msg:(Printf.sprintf "alpha %g beta %g" alpha beta)
                (ref_gemm ~ta ~tb ~alpha ~beta a b c0)
                c)
            [ (1.0, 0.0); (1.0, 1.0); (-0.5, 2.0) ])
        [ (false, false); (true, false); (false, true); (true, true) ])
    gemm_shapes

let gemm_errors =
  "gemm errors"
  >::: [
         ( "wrong output shape" >:: fun _ ->
           assert_invalid_arg (fun () ->
               Tensor.gemm_
                 (range [| 2; 3 |])
                 (range [| 3; 4 |])
                 ~c:(Tensor.zeros [| 2; 3 |])) );
         ( "inner dimensions" >:: fun _ ->
           assert_invalid_arg (fun () ->
               Tensor.gemm_ ~trans_a:true
                 (range [| 2; 3 |])
                 (range [| 3; 4 |])
                 ~c:(Tensor.zeros [| 3; 4 |])) );
         ( "c aliasing an input" >:: fun _ ->
           let a = range [| 3; 3 |] in
           assert_invalid_arg (fun () -> Tensor.gemm_ a (range [| 3; 3 |]) ~c:a)
         );
       ]

let broadcast_pairs =
  [
    ([| 7; 5 |], [| 7; 5 |]);
    ([| 7; 5 |], [| 5 |]);
    ([| 7; 5 |], [| 7; 1 |]);
    ([| 1; 5 |], [| 7; 1 |]);
    ([| 3; 1; 4 |], [| 2; 1 |]);
    ([| 2; 3; 4; 5 |], [| 3; 1; 5 |]);
    ([||], [| 4; 2 |]);
    ([| 300; 8 |], [| 8 |]);
  ]

let shape_str s =
  "[" ^ String.concat ";" (Array.to_list (Array.map string_of_int s)) ^ "]"

let elementwise_tests =
  List.concat_map
    (fun (sa, sb) ->
      let a = rand_tensor sa and b = rand_tensor ~lo:0.5 ~hi:2.0 sb in
      let name op =
        Printf.sprintf "%s %s %s" op (shape_str sa) (shape_str sb)
      in
      [
        ( name "add" >:: fun _ ->
          assert_tensor (ref_binop ( +. ) a b) (Tensor.add a b) );
        ( name "sub" >:: fun _ ->
          assert_tensor (ref_binop ( -. ) b a) (Tensor.sub b a) );
        ( name "mul" >:: fun _ ->
          assert_tensor (ref_binop ( *. ) a b) (Tensor.mul a b) );
        ( name "div" >:: fun _ ->
          assert_tensor (ref_binop ( /. ) a b) (Tensor.div a b) );
        ( name "map2" >:: fun _ ->
          let f x y = (x *. 2.) -. y in
          assert_tensor (ref_binop f a b) (Tensor.map2 f a b) );
        ( name "broadcast_to + sum_to_shape" >:: fun _ ->
          let big = Tensor.broadcast_shapes sa sb in
          let x = rand_tensor big in
          assert_tensor
            (ref_binop (fun _ v -> v) x b)
            (Tensor.broadcast_to b big);
          assert_tensor ~rtol:1e-9 (ref_sum_to_shape x sb)
            (Tensor.sum_to_shape x sb) );
      ])
    broadcast_pairs

let unary_tests =
  let x = rand_tensor [| 1001 |]
  and pos = rand_tensor ~lo:0.1 ~hi:3.0 [| 1001 |] in
  let check name f g input =
    name >:: fun _ -> assert_tensor (Tensor.map g input) (f input)
  in
  [
    check "neg" Tensor.neg Float.neg x;
    check "scale" (Tensor.scale 2.5) (fun v -> 2.5 *. v) x;
    check "add_scalar" (Tensor.add_scalar 1.5) (fun v -> v +. 1.5) x;
    check "exp" Tensor.exp Float.exp x;
    check "log" Tensor.log Float.log pos;
    check "sqrt" Tensor.sqrt Float.sqrt pos;
    check "pow" (fun v -> Tensor.pow v 1.7) (fun v -> Float.pow v 1.7) pos;
    check "relu" Tensor.relu (fun v -> if v > 0. then v else 0.) x;
    ( "relu gives +0 for negatives and keeps NaN" >:: fun _ ->
      let r =
        Tensor.to_array (Tensor.relu (t [| 3 |] [| -2.; Float.nan; 0. |]))
      in
      assert_bool "+0" (1. /. r.(0) = Float.infinity);
      assert_bool "nan" (Float.is_nan r.(1)) );
    ( "relu_backward" >:: fun _ ->
      let g = rand_tensor [| 1001 |] in
      assert_tensor
        (Tensor.map2 (fun xi gi -> if xi > 0. then gi else 0.) x g)
        (Tensor.relu_backward ~input:x ~grad:g) );
  ]

let reduction_tests =
  let x = rand_tensor [| 3; 4; 5 |] in
  List.concat_map
    (fun axis ->
      [
        ( Printf.sprintf "sum_axis %d" axis >:: fun _ ->
          let shape = Tensor.shape x in
          let target =
            Array.mapi (fun i d -> if i = axis then 1 else d) shape
          in
          assert_tensor
            (ref_sum_to_shape x target)
            (Tensor.sum_axis ~keepdims:true x axis) );
        ( Printf.sprintf "max_axis %d" axis >:: fun _ ->
          let r = Tensor.max_axis x axis and idx = Tensor.argmax_axis x axis in
          let rd = Tensor.to_array r in
          let s = Tensor.shape x in
          Array.iteri
            (fun j m ->
              let outer = Array.fold_left ( * ) 1 (Array.sub s 0 axis) in
              let inner = Array.length rd / outer in
              let o = j / inner and i = j mod inner in
              let at k =
                (Tensor.to_array x).((o * s.(axis) * inner) + (k * inner) + i)
              in
              assert_float m (at idx.(j)))
            rd );
      ])
    [ 0; 1; 2 ]

let conv_tests =
  List.concat_map
    (fun (shape, k, stride, padding) ->
      let x = rand_tensor shape in
      let name what =
        Printf.sprintf "%s %s k%d s%d p%d" what (shape_str shape) k stride
          padding
      in
      [
        ( name "im2col" >:: fun _ ->
          assert_tensor
            (ref_im2col x ~kh:k ~kw:k ~stride ~padding)
            (Tensor.im2col x ~kh:k ~kw:k ~stride ~padding) );
        ( name "col2im adjoint" >:: fun _ ->
          let cols = Tensor.im2col x ~kh:k ~kw:k ~stride ~padding in
          let y = rand_tensor (Tensor.shape cols) in
          let back =
            Tensor.col2im y ~input_shape:shape ~kh:k ~kw:k ~stride ~padding
          in
          let dot a b =
            let a = Tensor.to_array a and b = Tensor.to_array b in
            let s = ref 0. in
            Array.iteri (fun i v -> s := !s +. (v *. b.(i))) a;
            !s
          in
          assert_float ~eps:1e-9 (dot cols y) (dot x back) );
      ])
    [
      ([| 2; 5; 5; 1 |], 3, 1, 1);
      ([| 2; 6; 7; 3 |], 3, 2, 1);
      ([| 1; 4; 4; 2 |], 2, 2, 0);
      ([| 3; 7; 5; 2 |], 3, 1, 0);
      ([| 1; 3; 3; 1 |], 5, 1, 2);
    ]
  @ List.map
      (fun (shape, size, stride) ->
        Printf.sprintf "max_pool2d %s size %d stride %d" (shape_str shape) size
          stride
        >:: fun _ ->
        let x = rand_tensor shape in
        let out, argmax = Tensor.max_pool2d x ~size ~stride in
        assert_tensor (ref_max_pool x ~size ~stride) out;
        let xd = Tensor.to_array x in
        Array.iteri
          (fun o i -> assert_float (Tensor.to_array out).(o) xd.(i))
          argmax)
      [
        ([| 2; 4; 4; 3 |], 2, 2);
        ([| 1; 7; 5; 2 |], 3, 2);
        ([| 2; 5; 5; 1 |], 2, 1);
      ]
  @ [
      ( "conv geometry rejects a window larger than the input" >:: fun _ ->
        assert_invalid_arg (fun () ->
            Tensor.max_pool2d (rand_tensor [| 1; 1; 1; 1 |]) ~size:2 ~stride:2)
      );
    ]

let softmax_tests =
  let logits = rand_tensor ~lo:(-3.) ~hi:3. [| 6; 5 |] in
  let labels = [| 0; 4; 2; 2; 1; 3 |] in
  [
    ( "log_softmax matches the composed formula" >:: fun _ ->
      let m = Tensor.max_axis ~keepdims:true logits (-1) in
      let shifted = Tensor.sub logits m in
      let lse =
        Tensor.log (Tensor.sum_axis ~keepdims:true (Tensor.exp shifted) (-1))
      in
      assert_tensor (Tensor.sub shifted lse) (Tensor.log_softmax logits) );
    ( "softmax_cross_entropy value" >:: fun _ ->
      let loss, _ = Tensor.softmax_cross_entropy logits labels in
      let lp = Tensor.log_softmax logits in
      let expected =
        -.Array.fold_left ( +. ) 0.
            (Array.mapi (fun r l -> Tensor.get lp [| r; l |]) labels)
        /. 6.
      in
      assert_float ~eps:1e-12 expected loss );
    ( "softmax_cross_entropy rejects bad labels" >:: fun _ ->
      assert_invalid_arg (fun () ->
          Tensor.softmax_cross_entropy logits [| 0; 1 |]);
      assert_invalid_arg (fun () ->
          Tensor.softmax_cross_entropy logits [| 0; 4; 2; 2; 1; 5 |]) );
    gradcheck "Var.softmax_cross_entropy" [ logits ]
      (unary (fun l -> Var.softmax_cross_entropy l labels));
    gradcheck "Var.linear"
      [ rand_tensor [| 5; 3 |]; rand_tensor [| 3; 4 |]; rand_tensor [| 4 |] ]
      (ternary Var.linear);
    gradcheck "Var.linear with a [1; n] bias"
      [ rand_tensor [| 2; 3 |]; rand_tensor [| 3; 2 |]; rand_tensor [| 1; 2 |] ]
      (ternary Var.linear);
    ( "Var.linear = add (matmul x w) b" >:: fun _ ->
      let x = rand_tensor [| 70; 30 |] and w = rand_tensor [| 30; 9 |] in
      let b = rand_tensor [| 9 |] in
      assert_tensor ~rtol:1e-12
        (Tensor.add (Tensor.matmul x w) b)
        (Var.value (Var.linear (Var.const x) (Var.const w) (Var.const b))) );
  ]

(* ------------------------------------------------------------------------ *)
(* Optimizer kernels: identical to the step-by-step formulas                 *)
(* ------------------------------------------------------------------------ *)

let optimizer_tests =
  let shape = [| 3; 7 |] in
  [
    ( "adam_update_ = unfused Adam" >:: fun _ ->
      let p = rand_tensor shape and g = rand_tensor shape in
      let m = rand_tensor shape and v = rand_tensor ~lo:0. ~hi:1. shape in
      let lr = 0.01 and b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and wd = 0.1 in
      let bc1 = 1. -. (b1 ** 3.) and bc2 = 1. -. (b2 ** 3.) in
      (* the old implementation, step by step *)
      let p' = Tensor.copy p and m' = Tensor.copy m and v' = Tensor.copy v in
      let g' = Tensor.add g (Tensor.scale wd p') in
      Tensor.scale_ b1 m';
      Tensor.axpy_ ~alpha:(1. -. b1) ~x:g' ~y:m';
      Tensor.scale_ b2 v';
      Tensor.axpy_ ~alpha:(1. -. b2) ~x:(Tensor.mul g' g') ~y:v';
      let u =
        Tensor.map2
          (fun mi vi -> mi /. bc1 /. (Float.sqrt (vi /. bc2) +. eps))
          m' v'
      in
      Tensor.axpy_ ~alpha:(-.lr) ~x:u ~y:p';
      Tensor.adam_update_ ~lr ~beta1:b1 ~beta2:b2 ~eps ~weight_decay:wd ~bc1
        ~bc2 ~param:p ~grad:g ~m ~v;
      assert_last_bits ~msg:"param" p' p;
      assert_last_bits ~msg:"m" m' m;
      assert_last_bits ~msg:"v" v' v );
    ( "sgd_update_ = unfused SGD with momentum" >:: fun _ ->
      let p = rand_tensor shape
      and g = rand_tensor shape
      and vel = rand_tensor shape in
      let p' = Tensor.copy p and vel' = Tensor.copy vel in
      let g' = Tensor.add g (Tensor.scale 0.01 p') in
      Tensor.scale_ 0.9 vel';
      Tensor.add_ vel' g';
      Tensor.axpy_ ~alpha:(-0.1) ~x:vel' ~y:p';
      Tensor.sgd_update_ ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01 ~param:p
        ~grad:g ~velocity:vel;
      assert_last_bits ~msg:"param" p' p;
      assert_last_bits ~msg:"velocity" vel' vel );
    ( "take_rows_ = take_rows" >:: fun _ ->
      let x = rand_tensor [| 10; 3; 2 |] and idx = [| 4; 0; 9 |] in
      let dst = Tensor.zeros [| 3; 3; 2 |] in
      Tensor.take_rows_ x idx ~dst;
      assert_identical (Tensor.take_rows x idx) dst;
      assert_invalid_arg (fun () ->
          Tensor.take_rows_ x idx ~dst:(Tensor.zeros [| 2; 3; 2 |])) );
  ]

(* ------------------------------------------------------------------------ *)
(* Gradient buffers                                                          *)
(* ------------------------------------------------------------------------ *)

let autograd_tests =
  [
    ( "a gradient passed through add is not corrupted by a second backward"
    >:: fun _ ->
      (* y = x + c, loss = sum y. Interior gradients accumulate across
         backward calls too (as they always have in this engine): the second
         call makes loss's gradient 2, y's 1 + 2 = 3, and x's 1 + 3 = 4.
         After the first call x shares y's gradient buffer. If the second
         call updated that buffer in place, x would silently become 3 before
         adding its own contribution, ending at 6. *)
      let x = Var.param (t [| 2 |] [| 1.; 2. |]) in
      let y = Var.add x (Var.const (t [| 2 |] [| 5.; 5. |])) in
      let loss = Var.sum y in
      Var.backward loss;
      assert_tensor (t [| 2 |] [| 1.; 1. |]) (Var.grad_exn x);
      Var.backward loss;
      assert_tensor (t [| 2 |] [| 4.; 4. |]) (Var.grad_exn x) );
    ( "zero_grad then backward overwrites the reused buffer" >:: fun _ ->
      let w = Var.param (rand_tensor [| 3; 2 |]) in
      let x = Var.const (rand_tensor [| 4; 3 |]) in
      let step () = Var.backward (Var.sum (Var.matmul x w)) in
      step ();
      let first = Tensor.copy (Var.grad_exn w) in
      Var.zero_grad w;
      assert_bool "cleared" (Var.grad w = None);
      step ();
      assert_tensor first (Var.grad_exn w);
      step ();
      assert_tensor (Tensor.scale 2. first) (Var.grad_exn w) );
    ( "no_grad records nothing" >:: fun _ ->
      let w = Var.param (rand_tensor [| 3 |]) in
      let y = Var.no_grad (fun () -> Var.mul w w) in
      assert_bool "no gradient" (not (Var.requires_grad y));
      assert_equal [] (Var.parents y);
      let z = Var.mul w w in
      assert_bool "recording again" (Var.requires_grad z) );
    ( "no_grad restores recording after an exception" >:: fun _ ->
      (try Var.no_grad (fun () -> failwith "boom") with Failure _ -> ());
      let w = Var.param (rand_tensor [| 3 |]) in
      assert_bool "recording" (Var.requires_grad (Var.mul w w)) );
  ]

(* ------------------------------------------------------------------------ *)
(* Thread-count independence                                                 *)
(* ------------------------------------------------------------------------ *)

(** Run [f] at several thread counts and require bit-identical results. *)
let same_at_every_thread_count name f =
  name >:: fun _ ->
  let saved = Parallel.num_threads () in
  Fun.protect
    ~finally:(fun () -> Parallel.set_num_threads saved)
    (fun () ->
      Parallel.set_num_threads 1;
      let reference = f () in
      List.iter
        (fun n ->
          Parallel.set_num_threads n;
          List.iter2
            (fun a b ->
              assert_identical ~msg:(Printf.sprintf "%d threads" n) a b)
            reference (f ()))
        [ 2; 3; 8 ])

(** A few training steps of a small CNN from a fixed seed: every parameter
    afterwards. Exercises every kernel, forward and backward. *)
let train_cnn () =
  let rng = Random.State.make [| 7 |] in
  let model =
    Layer.sequential
      [
        Layer.conv2d ~rng ~padding:1 ~in_channels:2 ~out_channels:4 ~kernel:3 ();
        Layer.relu;
        Layer.max_pool2d 2;
        Layer.flatten;
        Layer.linear ~rng ~in_dim:(4 * 4 * 4) ~out_dim:5 ();
      ]
  in
  let opt = Optim.adam ~lr:1e-2 (Layer.params model) in
  for step = 1 to 3 do
    let x = Tensor.randn ~rng:(Random.State.make [| step |]) [| 6; 8; 8; 2 |] in
    Optim.zero_grad opt;
    let logits = Layer.forward ~train:true model (Var.const x) in
    Var.backward (Loss.cross_entropy ~logits ~labels:[| 0; 1; 2; 3; 4; 0 |]);
    Optim.step opt
  done;
  List.map (fun p -> Tensor.copy (Var.value p)) (Layer.params model)

let determinism_tests =
  let a = rand_tensor [| 130; 300 |] and b = rand_tensor [| 300; 70 |] in
  let tall = rand_tensor [| 5000; 9 |] and g = rand_tensor [| 5000; 8 |] in
  let big = rand_tensor [| 70000; 8 |] in
  let img = rand_tensor [| 6; 9; 9; 3 |] in
  [
    same_at_every_thread_count "matmul" (fun () -> [ Tensor.matmul a b ]);
    same_at_every_thread_count "split-k gemm" (fun () ->
        let c = Tensor.zeros [| 9; 8 |] in
        Tensor.gemm_ ~trans_a:true tall g ~c;
        [ c ]);
    same_at_every_thread_count "sum_to_shape over chunks" (fun () ->
        [ Tensor.sum_to_shape big [| 8 |]; Tensor.sum_axis big 0 ]);
    same_at_every_thread_count "convolution kernels" (fun () ->
        let cols = Tensor.im2col img ~kh:3 ~kw:3 ~stride:1 ~padding:1 in
        let pooled, _ = Tensor.max_pool2d img ~size:2 ~stride:2 in
        [
          cols;
          Tensor.col2im cols ~input_shape:[| 6; 9; 9; 3 |] ~kh:3 ~kw:3 ~stride:1
            ~padding:1;
          pooled;
        ]);
    same_at_every_thread_count "CNN training steps" train_cnn;
  ]

let () =
  run_test_tt_main
    ("kernels"
    >::: [
           "gemm" >::: gemm_tests;
           gemm_errors;
           "elementwise and broadcasting" >::: elementwise_tests;
           "unary" >::: unary_tests;
           "reductions" >::: reduction_tests;
           "convolution" >::: conv_tests;
           "softmax and fused ops" >::: softmax_tests;
           "optimizer kernels" >::: optimizer_tests;
           "autograd buffers" >::: autograd_tests;
           "thread-count independence" >::: determinism_tests;
         ])

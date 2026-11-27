(* The BLAS backend (flockml.blas): every test suite, run on Tensor_blas.F64,
   plus cross-backend agreement. Every op, and a few whole training steps,
   are computed on the default backend and on Tensor_blas.F64 / F32 from the
   same inputs, and the results must agree within the backend's tolerance.
   Gradient checks run in float64 only. Run with:
     dune exec test/test_backends.exe

   As in test_kernels, Parallel.min_parallel_work is 0, so the parallel paths
   of every kernel are exercised. *)

open Flockml
open OUnit2

let () = Parallel.min_parallel_work := 0

(* Building the suites already runs kernels (their inputs are tensors). That
   must not start worker domains: OUnit forks its test runners, and a process
   can't fork once it has spawned a domain. So the suites are built on one
   thread, and the thread count is restored before the tests run. *)
let threads = Parallel.num_threads ()
let () = Parallel.set_num_threads 1

module F64 = Flockml.Make (Tensor_blas.F64)
module F32 = Flockml.Make (Tensor_blas.F32)

(** Every suite, on the float64 BLAS backend. *)
let suites_f64 =
  let module T = Flockml_tests.Tensor_tests.Make (F64) in
  let module A = Flockml_tests.Autograd_tests.Make (F64) in
  let module N = Flockml_tests.Nn_tests.Make (F64) in
  let module O = Flockml_tests.Optim_tests.Make (F64) in
  let module C = Flockml_tests.Conv_tests.Make (F64) in
  let module K = Flockml_tests.Kernel_tests.Make (F64) in
  let module P = Flockml_tests.Numpy_tests.Make (F64) in
  let module Th = Flockml_tests.Torch_tests.Make (F64) in
  let module L = Flockml_tests.Layers_tests.Make (F64) in
  "suites on Tensor_blas.F64"
  >::: [
         T.suite;
         A.suite;
         N.suite;
         O.suite;
         C.suite;
         K.suite;
         P.suite;
         Th.suite;
         L.suite;
       ]

(** How close a backend must be to the default one. *)
module type TOLERANCE = sig
  val name : string
  val rtol : float
  val atol : float

  val exact_argmax : bool
  (** Whether argmax results must be identical. Rounding to float32 can turn two
      nearly equal values into a tie, so float32 doesn't require it. *)
end

(** Run each op on the default backend and on [B] from the same inputs, and
    compare. Inputs are drawn from a fixed seed with the default backend, then
    copied into [B]. *)
module Agree (B : Flockml.S) (Tol : TOLERANCE) = struct
  let rng = Random.State.make [| 99 |]

  let rand ?(lo = -1.0) ?(hi = 1.0) shape =
    Tensor.rand_uniform ~rng ~lo ~hi shape

  let into (x : Tensor.t) =
    B.Tensor.of_array (Tensor.shape x) (Tensor.to_array x)

  let back (x : B.Tensor.t) =
    Tensor.of_array (B.Tensor.shape x) (B.Tensor.to_array x)

  let max_diff a b =
    let a = Tensor.to_array a and b = Tensor.to_array b in
    let d = ref 0.0 in
    Array.iteri (fun i x -> d := Float.max !d (Float.abs (x -. b.(i)))) a;
    !d

  (** Fail unless [actual] (from [B]) is within tolerance of [expected]. *)
  let close ?(scale = 1.0) ~msg expected actual =
    let actual = back actual in
    if
      Tensor.shape expected <> Tensor.shape actual
      || not
           (Tensor.allclose ~rtol:(Tol.rtol *. scale) ~atol:(Tol.atol *. scale)
              actual expected)
    then
      assert_failure
        (Printf.sprintf
           "%s: %s differs from the default backend (max |diff| %g)" msg
           Tol.name (max_diff expected actual))

  let case name f = name >:: fun _ -> f ~msg:name

  let elementwise =
    let pairs =
      [
        ([| 7; 5 |], [| 7; 5 |]);
        ([| 7; 5 |], [| 5 |]);
        ([| 7; 5 |], [| 7; 1 |]);
        ([| 3; 1; 4 |], [| 2; 1 |]);
        ([| 300; 8 |], [| 8 |]);
        ([||], [| 4; 2 |]);
      ]
    in
    List.concat_map
      (fun (sa, sb) ->
        let a = rand sa and b = rand ~lo:0.5 ~hi:2.0 sb in
        let big = Tensor.broadcast_shapes sa sb in
        let x = rand big in
        let name op =
          Printf.sprintf "%s %s %s" op
            (Layout.shape_to_string sa)
            (Layout.shape_to_string sb)
        in
        [
          case (name "add") (fun ~msg ->
              close ~msg (Tensor.add a b) (B.Tensor.add (into a) (into b)));
          case (name "sub") (fun ~msg ->
              close ~msg (Tensor.sub b a) (B.Tensor.sub (into b) (into a)));
          case (name "mul") (fun ~msg ->
              close ~msg (Tensor.mul a b) (B.Tensor.mul (into a) (into b)));
          case (name "div") (fun ~msg ->
              close ~msg (Tensor.div a b) (B.Tensor.div (into a) (into b)));
          case (name "broadcast_to") (fun ~msg ->
              close ~msg
                (Tensor.broadcast_to b big)
                (B.Tensor.broadcast_to (into b) big));
          case (name "sum_to_shape") (fun ~msg ->
              close ~msg (Tensor.sum_to_shape x sb)
                (B.Tensor.sum_to_shape (into x) sb));
        ])
      pairs

  let unary =
    let x = rand [| 1001 |] and pos = rand ~lo:0.1 ~hi:3.0 [| 1001 |] in
    let g = rand [| 1001 |] in
    [
      case "neg" (fun ~msg -> close ~msg (Tensor.neg x) (B.Tensor.neg (into x)));
      case "scale" (fun ~msg ->
          close ~msg (Tensor.scale 2.5 x) (B.Tensor.scale 2.5 (into x)));
      case "add_scalar" (fun ~msg ->
          close ~msg (Tensor.add_scalar 1.5 x)
            (B.Tensor.add_scalar 1.5 (into x)));
      case "exp" (fun ~msg -> close ~msg (Tensor.exp x) (B.Tensor.exp (into x)));
      case "log" (fun ~msg ->
          close ~msg (Tensor.log pos) (B.Tensor.log (into pos)));
      case "sqrt" (fun ~msg ->
          close ~msg (Tensor.sqrt pos) (B.Tensor.sqrt (into pos)));
      case "pow" (fun ~msg ->
          close ~msg (Tensor.pow pos 1.7) (B.Tensor.pow (into pos) 1.7));
      case "relu" (fun ~msg ->
          close ~msg (Tensor.relu x) (B.Tensor.relu (into x)));
      case "relu_backward" (fun ~msg ->
          close ~msg
            (Tensor.relu_backward ~input:x ~grad:g)
            (B.Tensor.relu_backward ~input:(into x) ~grad:(into g)));
      case "map / map2" (fun ~msg ->
          close ~msg
            (Tensor.map2 (fun u v -> (u *. 2.) -. v) x g)
            (B.Tensor.map2 (fun u v -> (u *. 2.) -. v) (into x) (into g));
          close ~msg (Tensor.map Float.cos x) (B.Tensor.map Float.cos (into x)));
    ]

  let reductions =
    let x = rand [| 3; 4; 5 |] and big = rand [| 70000; 8 |] in
    [
      case "sum, mean" (fun ~msg ->
          close ~msg (Tensor.sum big) (B.Tensor.sum (into big)) ~scale:10.;
          close ~msg (Tensor.mean x) (B.Tensor.mean (into x)));
      case "sum_axis 0 of [70000; 8]" (fun ~msg ->
          close ~msg ~scale:10. (Tensor.sum_axis big 0)
            (B.Tensor.sum_axis (into big) 0));
    ]
    @ List.concat_map
        (fun axis ->
          [
            case (Printf.sprintf "sum_axis %d" axis) (fun ~msg ->
                close ~msg
                  (Tensor.sum_axis ~keepdims:true x axis)
                  (B.Tensor.sum_axis ~keepdims:true (into x) axis));
            case (Printf.sprintf "max_axis / argmax_axis %d" axis) (fun ~msg ->
                close ~msg (Tensor.max_axis x axis)
                  (B.Tensor.max_axis (into x) axis);
                if Tol.exact_argmax then
                  assert_equal ~msg
                    (Tensor.argmax_axis x axis)
                    (B.Tensor.argmax_axis (into x) axis));
          ])
        [ 0; 1; 2 ]

  let linear_algebra =
    List.concat_map
      (fun (m, k, n) ->
        List.map
          (fun (ta, tb) ->
            let a = rand (if ta then [| k; m |] else [| m; k |]) in
            let b = rand (if tb then [| n; k |] else [| k; n |]) in
            let c = rand [| m; n |] in
            case
              (Printf.sprintf "gemm_ %dx%dx%d%s%s" m k n
                 (if ta then " a^T" else "")
                 (if tb then " b^T" else ""))
              (fun ~msg ->
                let expected = Tensor.copy c and actual = into c in
                Tensor.gemm_ ~trans_a:ta ~trans_b:tb ~alpha:(-0.5) ~beta:2.0 a b
                  ~c:expected;
                B.Tensor.gemm_ ~trans_a:ta ~trans_b:tb ~alpha:(-0.5) ~beta:2.0
                  (into a) (into b) ~c:actual;
                close ~msg ~scale:10. expected actual))
          [ (false, false); (true, false); (false, true); (true, true) ])
      [ (1, 1, 1); (7, 13, 9); (65, 300, 67); (9, 5000, 8) ]
    @
    let x = rand [| 37; 23 |] in
    [
      case "matmul" (fun ~msg ->
          let w = rand [| 23; 11 |] in
          close ~msg (Tensor.matmul x w) (B.Tensor.matmul (into x) (into w)));
      case "transpose" (fun ~msg ->
          close ~msg (Tensor.transpose x) (B.Tensor.transpose (into x)));
      case "reshape" (fun ~msg ->
          close ~msg
            (Tensor.reshape x [| -1; 37 |])
            (B.Tensor.reshape (into x) [| -1; 37 |]));
      case "take_rows" (fun ~msg ->
          close ~msg
            (Tensor.take_rows x [| 36; 0; 5; 5 |])
            (B.Tensor.take_rows (into x) [| 36; 0; 5; 5 |]));
      case "one_hot" (fun ~msg ->
          close ~msg
            (Tensor.one_hot ~num_classes:4 [| 3; 0; 2 |])
            (B.Tensor.one_hot ~num_classes:4 [| 3; 0; 2 |]));
    ]

  let softmax =
    let logits = rand ~lo:(-3.) ~hi:3. [| 6; 5 |] and g = rand [| 6; 5 |] in
    let labels = [| 0; 4; 2; 2; 1; 3 |] in
    [
      case "log_softmax and backward" (fun ~msg ->
          let y = Tensor.log_softmax logits in
          let by = B.Tensor.log_softmax (into logits) in
          close ~msg y by;
          close ~msg
            (Tensor.log_softmax_backward ~output:y ~grad:g)
            (B.Tensor.log_softmax_backward ~output:by ~grad:(into g)));
      case "softmax_cross_entropy and backward" (fun ~msg ->
          let loss, lp = Tensor.softmax_cross_entropy logits labels in
          let bloss, blp =
            B.Tensor.softmax_cross_entropy (into logits) labels
          in
          close ~msg (Tensor.scalar loss) (B.Tensor.scalar bloss);
          close ~msg lp blp;
          close ~msg
            (Tensor.softmax_cross_entropy_backward ~log_probs:lp ~labels
               ~scale:0.3)
            (B.Tensor.softmax_cross_entropy_backward ~log_probs:blp ~labels
               ~scale:0.3));
    ]

  let in_place =
    let shape = [| 3; 7 |] in
    let p = rand shape and g = rand shape and m = rand shape in
    let v = rand ~lo:0. ~hi:1. shape in
    [
      case "add_, axpy_, scale_, fill_, blit_" (fun ~msg ->
          let y = Tensor.copy p and by = into p in
          Tensor.add_ y g;
          B.Tensor.add_ by (into g);
          Tensor.axpy_ ~alpha:0.7 ~x:m ~y;
          B.Tensor.axpy_ ~alpha:0.7 ~x:(into m) ~y:by;
          Tensor.scale_ 1.3 y;
          B.Tensor.scale_ 1.3 by;
          close ~msg y by;
          B.Tensor.blit_ ~src:(into g) ~dst:by;
          close ~msg g by;
          B.Tensor.fill_ by 4.0;
          close ~msg (Tensor.full shape 4.0) by);
      case "adam_update_" (fun ~msg ->
          let p' = Tensor.copy p
          and m' = Tensor.copy m
          and v' = Tensor.copy v in
          let bp = into p and bm = into m and bv = into v in
          Tensor.adam_update_ ~lr:0.01 ~beta1:0.9 ~beta2:0.999 ~eps:1e-8
            ~weight_decay:0.1 ~bc1:0.271 ~bc2:0.003 ~param:p' ~grad:g ~m:m'
            ~v:v';
          B.Tensor.adam_update_ ~lr:0.01 ~beta1:0.9 ~beta2:0.999 ~eps:1e-8
            ~weight_decay:0.1 ~bc1:0.271 ~bc2:0.003 ~param:bp ~grad:(into g)
            ~m:bm ~v:bv;
          close ~msg p' bp;
          close ~msg m' bm;
          close ~msg v' bv);
      case "sgd_update_" (fun ~msg ->
          let p' = Tensor.copy p and vel = Tensor.copy v in
          let bp = into p and bvel = into v in
          Tensor.sgd_update_ ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01 ~param:p'
            ~grad:g ~velocity:vel;
          B.Tensor.sgd_update_ ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01
            ~param:bp ~grad:(into g) ~velocity:bvel;
          close ~msg p' bp;
          close ~msg vel bvel);
      case "adamw_update_" (fun ~msg ->
          let p' = Tensor.copy p
          and m' = Tensor.copy m
          and v' = Tensor.copy v in
          let bp = into p and bm = into m and bv = into v in
          Tensor.adamw_update_ ~lr:0.01 ~beta1:0.9 ~beta2:0.999 ~eps:1e-8
            ~weight_decay:0.1 ~bc1:0.271 ~bc2:0.003 ~param:p' ~grad:g ~m:m'
            ~v:v';
          B.Tensor.adamw_update_ ~lr:0.01 ~beta1:0.9 ~beta2:0.999 ~eps:1e-8
            ~weight_decay:0.1 ~bc1:0.271 ~bc2:0.003 ~param:bp ~grad:(into g)
            ~m:bm ~v:bv;
          close ~msg p' bp;
          close ~msg m' bm;
          close ~msg v' bv);
      case "sgd_nesterov_update_" (fun ~msg ->
          let p' = Tensor.copy p and vel = Tensor.copy v in
          let bp = into p and bvel = into v in
          Tensor.sgd_nesterov_update_ ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01
            ~param:p' ~grad:g ~velocity:vel;
          B.Tensor.sgd_nesterov_update_ ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01
            ~param:bp ~grad:(into g) ~velocity:bvel;
          close ~msg p' bp;
          close ~msg vel bvel);
      case "rmsprop_update_" (fun ~msg ->
          List.iter
            (fun (momentum, centered) ->
              let p' = Tensor.copy p and sq = Tensor.add_scalar 2. v in
              let ga = Tensor.copy m and buf = Tensor.copy g in
              let bp = into p and bsq = into sq and bga = into ga in
              let bbuf = into buf in
              Tensor.rmsprop_update_ ~lr:0.01 ~alpha:0.9 ~eps:1e-8
                ~weight_decay:0.01 ~momentum ~centered ~param:p' ~grad:g
                ~square_avg:sq ~grad_avg:ga ~momentum_buf:buf;
              B.Tensor.rmsprop_update_ ~lr:0.01 ~alpha:0.9 ~eps:1e-8
                ~weight_decay:0.01 ~momentum ~centered ~param:bp ~grad:(into g)
                ~square_avg:bsq ~grad_avg:bga ~momentum_buf:bbuf;
              List.iter2 (close ~msg) [ p'; sq; ga; buf ] [ bp; bsq; bga; bbuf ])
            [ (0., false); (0.5, true) ]);
      case "adagrad_update_" (fun ~msg ->
          let p' = Tensor.copy p and sum = Tensor.copy v in
          let bp = into p and bsum = into v in
          Tensor.adagrad_update_ ~lr:0.1 ~eps:1e-10 ~weight_decay:0.01 ~param:p'
            ~grad:g ~sum;
          B.Tensor.adagrad_update_ ~lr:0.1 ~eps:1e-10 ~weight_decay:0.01
            ~param:bp ~grad:(into g) ~sum:bsum;
          close ~msg p' bp;
          close ~msg sum bsum);
    ]

  let convolution =
    List.concat_map
      (fun (shape, k, stride, padding) ->
        let x = rand shape in
        let name what =
          Printf.sprintf "%s %s k%d s%d p%d" what
            (Layout.shape_to_string shape)
            k stride padding
        in
        [
          case (name "im2col / col2im") (fun ~msg ->
              let cols = Tensor.im2col x ~kh:k ~kw:k ~stride ~padding in
              let bcols =
                B.Tensor.im2col (into x) ~kh:k ~kw:k ~stride ~padding
              in
              close ~msg cols bcols;
              close ~msg
                (Tensor.col2im cols ~input_shape:shape ~kh:k ~kw:k ~stride
                   ~padding)
                (B.Tensor.col2im bcols ~input_shape:shape ~kh:k ~kw:k ~stride
                   ~padding));
        ])
      [
        ([| 2; 5; 5; 1 |], 3, 1, 1);
        ([| 2; 6; 7; 3 |], 3, 2, 1);
        ([| 3; 7; 5; 2 |], 3, 1, 0);
        ([| 1; 3; 3; 1 |], 5, 1, 2);
      ]
    @ List.map
        (fun (shape, size, stride) ->
          let x = rand shape in
          case
            (Printf.sprintf "max_pool2d and backward %s size %d"
               (Layout.shape_to_string shape)
               size)
            (fun ~msg ->
              let out, argmax = Tensor.max_pool2d x ~size ~stride in
              let bout, bargmax = B.Tensor.max_pool2d (into x) ~size ~stride in
              close ~msg out bout;
              if Tol.exact_argmax then assert_equal ~msg argmax bargmax;
              let g = rand (Tensor.shape out) in
              close ~msg
                (Tensor.max_pool2d_backward ~grad_out:g ~argmax
                   ~input_shape:shape)
                (B.Tensor.max_pool2d_backward ~grad_out:(into g) ~argmax
                   ~input_shape:shape)))
        [ ([| 2; 4; 4; 3 |], 2, 2); ([| 1; 7; 5; 2 |], 3, 2) ]

  (** A few Adam steps of a small CNN and of an MLP, from the same seed on both
      backends: the losses at every step and the final parameters must agree.
      The parameters are compared at a looser tolerance: Adam divides by
      [sqrt v], which magnifies tiny differences in near-zero gradients. *)
  let training =
    let train (type v t)
        (module S : Flockml.S with type Var.t = v and type Tensor.t = t) kind =
      let rng = Random.State.make [| 7 |] in
      let model =
        match kind with
        | `Cnn ->
            S.Layer.sequential
              [
                S.Layer.conv2d ~rng ~padding:1 ~in_channels:2 ~out_channels:4
                  ~kernel:3 ();
                S.Layer.relu;
                S.Layer.max_pool2d 2;
                S.Layer.flatten;
                S.Layer.linear ~rng ~in_dim:(4 * 4 * 4) ~out_dim:5 ();
              ]
        | `Mlp ->
            S.Layer.sequential
              [
                S.Layer.linear ~rng ~in_dim:20 ~out_dim:16 ();
                S.Layer.relu;
                S.Layer.linear ~rng ~in_dim:16 ~out_dim:5 ();
              ]
      in
      let opt = S.Optim.adam ~lr:1e-2 (S.Layer.params model) in
      let losses =
        List.init 4 (fun step ->
            let shape =
              match kind with `Cnn -> [| 6; 8; 8; 2 |] | `Mlp -> [| 6; 20 |]
            in
            let x = S.Tensor.randn ~rng:(Random.State.make [| step |]) shape in
            S.Optim.zero_grad opt;
            let logits = S.Layer.forward ~train:true model (S.Var.const x) in
            let loss = S.Loss.cross_entropy logits [| 0; 1; 2; 3; 4; 0 |] in
            S.Var.backward loss;
            S.Optim.step opt;
            S.Tensor.item (S.Var.value loss))
      in
      let params =
        List.map
          (fun p ->
            let v = S.Var.value p in
            Tensor.of_array (S.Tensor.shape v) (S.Tensor.to_array v))
          (S.Layer.params model)
      in
      (losses, params)
    in
    List.map
      (fun kind ->
        case
          (match kind with
          | `Cnn -> "CNN training steps"
          | `Mlp -> "MLP training steps")
          (fun ~msg ->
            let losses, params = train (module Default) kind in
            let blosses, bparams = train (module B) kind in
            List.iter2
              (fun l bl ->
                close ~msg ~scale:10. (Tensor.scalar l) (B.Tensor.scalar bl))
              losses blosses;
            List.iter2
              (fun p bp -> close ~msg ~scale:100. p (into bp))
              params bparams))
      [ `Cnn; `Mlp ]

  (** A few steps of two models made of the Phase 9 layers, from the same seed
      on both backends (as {!training}): a CNN with dilated and grouped
      convolutions, batch norm and average pooling, trained with Nesterov SGD,
      and a causal-attention / layer-norm / GELU / GRU stack trained with AdamW
      and label smoothing. *)
  let layers_training =
    let train (type v t)
        (module S : Flockml.S with type Var.t = v and type Tensor.t = t) kind =
      let rng = Random.State.make [| 11 |] in
      let model, opt, shape, labels =
        match kind with
        | `Cnn ->
            let m =
              S.Layer.sequential
                [
                  S.Layer.conv2d ~rng ~padding:2 ~dilation:2 ~in_channels:2
                    ~out_channels:4 ~kernel:3 ();
                  S.Layer.batch_norm 4;
                  S.Layer.relu;
                  S.Layer.conv2d ~rng ~groups:2 ~stride:2 ~padding:1
                    ~in_channels:4 ~out_channels:4 ~kernel:3 ();
                  S.Layer.avg_pool2d 2;
                  S.Layer.flatten;
                  S.Layer.linear ~rng ~in_dim:16 ~out_dim:5 ();
                ]
            in
            ( m,
              S.Optim.sgd ~lr:0.05 ~momentum:0.9 ~nesterov:true
                (S.Layer.params m),
              [| 6; 8; 8; 2 |],
              [| 0; 1; 2; 3; 4; 0 |] )
        | `Seq ->
            let m =
              S.Layer.sequential
                [
                  S.Layer.residual
                    (S.Layer.sequential
                       [
                         S.Layer.layer_norm [| 8 |];
                         S.Layer.multi_head_attention ~rng ~causal:true ~dim:8
                           ~heads:2 ();
                       ]);
                  S.Layer.gelu ();
                  S.Layer.gru ~rng ~input:8 ~hidden:5 ();
                ]
            in
            ( m,
              S.Optim.adamw ~lr:1e-2 (S.Layer.params m),
              [| 2; 5; 8 |],
              Array.init 10 (fun i -> i mod 5) )
      in
      let losses =
        List.init 3 (fun step ->
            let x = S.Tensor.randn ~rng:(Random.State.make [| step |]) shape in
            S.Optim.zero_grad opt;
            let logits = S.Layer.forward ~train:true model (S.Var.const x) in
            let loss =
              S.Loss.cross_entropy ~label_smoothing:0.1 logits labels
            in
            S.Var.backward loss;
            S.Optim.step opt;
            S.Tensor.item (S.Var.value loss))
      in
      let state =
        List.map
          (fun (_, v) ->
            Tensor.of_array (S.Tensor.shape v) (S.Tensor.to_array v))
          (S.Layer.state_dict model)
      in
      (losses, state)
    in
    List.map
      (fun (name, kind) ->
        case name (fun ~msg ->
            let losses, state = train (module Default) kind in
            let blosses, bstate = train (module B) kind in
            List.iter2
              (fun l bl ->
                close ~msg ~scale:10. (Tensor.scalar l) (B.Tensor.scalar bl))
              losses blosses;
            List.iter2
              (fun p bp -> close ~msg ~scale:100. p (into bp))
              state bstate))
      [ ("CNN of the new layers", `Cnn); ("sequence model", `Seq) ]

  (** The NumPy-level API (Phase 7): every op from the same inputs. *)
  let numpy_api =
    let x = rand [| 4; 6; 5 |] and pos = rand ~lo:0.1 ~hi:3.0 [| 4; 6; 5 |] in
    let y = rand [| 6; 5 |] and m = rand [| 6; 1 |] in
    let ints = Tensor.randint ~rng ~lo:0 ~hi:3 [| 6; 5 |] in
    let mask = Tensor.gt m (Tensor.scalar 0.) in
    let idx = [| 5; 0; 0; 3; 2 |] in
    let gidx = Array.init (4 * 2 * 5) (fun i -> i * 7 mod 6) in
    let unary name f g =
      case name (fun ~msg -> close ~msg (f x) (g (into x)))
    in
    let on_pos name f g =
      case name (fun ~msg -> close ~msg (f pos) (g (into pos)))
    in
    let same name f g = case name (fun ~msg -> close ~msg (f ()) (g ())) in
    [
      unary "abs" Tensor.abs B.Tensor.abs;
      unary "sign" Tensor.sign B.Tensor.sign;
      unary "square" Tensor.square B.Tensor.square;
      on_pos "rsqrt" Tensor.rsqrt B.Tensor.rsqrt;
      on_pos "log1p" Tensor.log1p B.Tensor.log1p;
      unary "expm1" Tensor.expm1 B.Tensor.expm1;
      unary "sin" Tensor.sin B.Tensor.sin;
      unary "cos" Tensor.cos B.Tensor.cos;
      unary "tanh" Tensor.tanh B.Tensor.tanh;
      unary "sigmoid" Tensor.sigmoid B.Tensor.sigmoid;
      unary "erf" Tensor.erf B.Tensor.erf;
      unary "clamp"
        (Tensor.clamp ~min:(-0.3) ~max:0.4)
        (B.Tensor.clamp ~min:(-0.3) ~max:0.4);
      case "minimum, maximum, comparisons" (fun ~msg ->
          close ~msg (Tensor.minimum x y) (B.Tensor.minimum (into x) (into y));
          close ~msg (Tensor.maximum x m) (B.Tensor.maximum (into x) (into m));
          List.iter2
            (fun f g -> close ~msg (f ints m) (g (into ints) (into m)))
            [ Tensor.eq; Tensor.ne; Tensor.lt; Tensor.le; Tensor.gt; Tensor.ge ]
            [
              B.Tensor.eq;
              B.Tensor.ne;
              B.Tensor.lt;
              B.Tensor.le;
              B.Tensor.gt;
              B.Tensor.ge;
            ]);
      case "where, masked_fill" (fun ~msg ->
          close ~msg (Tensor.where mask x y)
            (B.Tensor.where (into mask) (into x) (into y));
          close ~msg
            (Tensor.masked_fill x mask 3.)
            (B.Tensor.masked_fill (into x) (into mask) 3.));
      case "min_axis, argmin_axis, max, min" (fun ~msg ->
          close ~msg (Tensor.min_axis x 1) (B.Tensor.min_axis (into x) 1);
          if Tol.exact_argmax then
            assert_equal ~msg (Tensor.argmin_axis x 2)
              (B.Tensor.argmin_axis (into x) 2);
          close ~msg (Tensor.max x) (B.Tensor.max (into x));
          close ~msg (Tensor.min x) (B.Tensor.min (into x)));
      case "mean_axis, sum_axes, prod_axis, cumsum" (fun ~msg ->
          close ~msg (Tensor.mean_axis x 0) (B.Tensor.mean_axis (into x) 0);
          close ~msg
            (Tensor.sum_axes x [ 0; 2 ])
            (B.Tensor.sum_axes (into x) [ 0; 2 ]);
          close ~msg (Tensor.prod_axis pos 2) (B.Tensor.prod_axis (into pos) 2);
          close ~msg (Tensor.cumsum x 1) (B.Tensor.cumsum (into x) 1));
      case "logsumexp, var, std" (fun ~msg ->
          close ~msg (Tensor.logsumexp x 1) (B.Tensor.logsumexp (into x) 1);
          close ~msg
            (Tensor.var ~correction:1 x 2)
            (B.Tensor.var ~correction:1 (into x) 2);
          close ~msg (Tensor.std x 0) (B.Tensor.std (into x) 0));
      case "permute, slice, flip, pad" (fun ~msg ->
          close ~msg
            (Tensor.permute x [| 2; 0; 1 |])
            (B.Tensor.permute (into x) [| 2; 0; 1 |]);
          let r = [ Tensor.Step (1, 4, 2); Index 3; Range (-3, 5) ] in
          close ~msg (Tensor.slice x r) (B.Tensor.slice (into x) r);
          close ~msg (Tensor.flip x [ 0; 2 ]) (B.Tensor.flip (into x) [ 0; 2 ]);
          close ~msg
            (Tensor.pad ~value:2. x [| (1, 0); (0, 2); (3, 1) |])
            (B.Tensor.pad ~value:2. (into x) [| (1, 0); (0, 2); (3, 1) |]));
      case "set_slice_" (fun ~msg ->
          let src = Tensor.slice y [ Step (0, 6, 3) ] in
          let a = Tensor.copy x and b = into x in
          Tensor.set_slice_ a [ All; Step (0, 6, 3) ] src;
          B.Tensor.set_slice_ b [ All; Step (0, 6, 3) ] (into src);
          close ~msg a b);
      case "concat, stack, split, chunk" (fun ~msg ->
          close ~msg
            (Tensor.concat ~axis:1 [ x; x ])
            (B.Tensor.concat ~axis:1 [ into x; into x ]);
          close ~msg
            (Tensor.stack ~axis:0 [ y; y ])
            (B.Tensor.stack ~axis:0 [ into y; into y ]);
          List.iter2 (close ~msg)
            (Tensor.split x ~axis:1 [ 1; 5 ])
            (B.Tensor.split (into x) ~axis:1 [ 1; 5 ]);
          List.iter2 (close ~msg) (Tensor.chunk x ~axis:2 2)
            (B.Tensor.chunk (into x) ~axis:2 2));
      case "tile, repeat" (fun ~msg ->
          close ~msg
            (Tensor.tile y [| 2; 1; 3 |])
            (B.Tensor.tile (into y) [| 2; 1; 3 |]);
          close ~msg
            (Tensor.repeat y ~axis:0 2)
            (B.Tensor.repeat (into y) ~axis:0 2));
      case "index_select, index_add, gather, scatter_add" (fun ~msg ->
          close ~msg
            (Tensor.index_select x ~axis:1 idx)
            (B.Tensor.index_select (into x) ~axis:1 idx);
          let src = Tensor.index_select x ~axis:1 idx in
          close ~msg
            (Tensor.index_add x ~axis:1 idx src)
            (B.Tensor.index_add (into x) ~axis:1 idx (into src));
          close ~msg
            (Tensor.gather x ~axis:1 gidx)
            (B.Tensor.gather (into x) ~axis:1 gidx);
          let gsrc = Tensor.gather x ~axis:1 gidx in
          close ~msg
            (Tensor.scatter_add x ~axis:1 gidx gsrc)
            (B.Tensor.scatter_add (into x) ~axis:1 gidx (into gsrc)));
      case "batched matmul" (fun ~msg ->
          let a = rand [| 3; 1; 4; 7 |] and b = rand [| 5; 7; 6 |] in
          close ~msg ~scale:10. (Tensor.matmul a b)
            (B.Tensor.matmul (into a) (into b));
          let w = rand [| 7; 6 |] in
          close ~msg ~scale:10. (Tensor.matmul a w)
            (B.Tensor.matmul (into a) (into w));
          let c = rand [| 5; 4; 7 |] in
          close ~msg ~scale:10.
            (Tensor.bmm ~trans_a:true ~trans_b:true b c)
            (B.Tensor.bmm ~trans_a:true ~trans_b:true (into b) (into c));
          let v = rand [| 7 |] in
          close ~msg ~scale:10. (Tensor.matmul a v)
            (B.Tensor.matmul (into a) (into v)));
      case "dot, outer, norm" (fun ~msg ->
          let v = rand [| 9 |] and u = rand [| 9 |] in
          close ~msg (Tensor.dot v u) (B.Tensor.dot (into v) (into u));
          close ~msg (Tensor.outer v u) (B.Tensor.outer (into v) (into u));
          List.iter
            (fun p -> close ~msg (Tensor.norm ~p x) (B.Tensor.norm ~p (into x)))
            [ 1.; 2.; 3.; Float.infinity ]);
      same "arange, linspace, eye"
        (fun () ->
          Tensor.concat ~axis:0
            [
              Tensor.arange ~start:1. ~step:0.25 3.;
              Tensor.linspace 0. 1. 5;
              Tensor.reshape (Tensor.eye 3) [| 9 |];
            ])
        (fun () ->
          B.Tensor.concat ~axis:0
            [
              B.Tensor.arange ~start:1. ~step:0.25 3.;
              B.Tensor.linspace 0. 1. 5;
              B.Tensor.reshape (B.Tensor.eye 3) [| 9 |];
            ]);
      same "bernoulli, randint"
        (fun () ->
          Tensor.add
            (Tensor.bernoulli ~rng:(Random.State.make [| 3 |]) ~p:0.4 [| 50 |])
            (Tensor.randint
               ~rng:(Random.State.make [| 4 |])
               ~lo:0 ~hi:9 [| 50 |]))
        (fun () ->
          B.Tensor.add
            (B.Tensor.bernoulli
               ~rng:(Random.State.make [| 3 |])
               ~p:0.4 [| 50 |])
            (B.Tensor.randint
               ~rng:(Random.State.make [| 4 |])
               ~lo:0 ~hi:9 [| 50 |]));
    ]

  (** The gradients of a small causal-attention block, on both backends. *)
  let attention =
    let grads (type v t)
        (module S : Flockml.S with type Var.t = v and type Tensor.t = t) =
      let qkv =
        List.map
          (fun s ->
            S.Tensor.randn ~rng:(Random.State.make [| s |]) [| 2; 5; 4 |])
          [ 1; 2; 3 ]
      in
      let ps = List.map S.Var.param qkv in
      let q, k, v =
        match ps with [ q; k; v ] -> (q, k, v) | _ -> assert false
      in
      let causal =
        S.Tensor.init [| 5; 5 |] (fun i -> if i.(1) > i.(0) then 1. else 0.)
      in
      let scores = S.Var.scale 0.5 (S.Var.matmul q (S.Var.swapaxes k 1 2)) in
      let att = S.Var.softmax (S.Var.masked_fill scores causal (-1e9)) in
      let out = S.Var.matmul att v in
      S.Var.backward (S.Var.sum (S.Var.mul out out));
      List.map
        (fun p ->
          let g = S.Var.grad_exn p in
          Tensor.of_array (S.Tensor.shape g) (S.Tensor.to_array g))
        ps
    in
    [
      case "causal attention gradients" (fun ~msg ->
          List.iter2
            (fun a b -> close ~msg ~scale:10. a (into b))
            (grads (module Default))
            (grads (module B)));
    ]

  let suite =
    Tol.name
    >::: [
           "NumPy-level API" >::: numpy_api;
           "attention" >::: attention;
           "elementwise and broadcasting" >::: elementwise;
           "unary" >::: unary;
           "reductions" >::: reductions;
           "linear algebra" >::: linear_algebra;
           "softmax" >::: softmax;
           "in-place and optimizer kernels" >::: in_place;
           "convolution" >::: convolution;
           "training" >::: training;
           "training with the Phase 9 layers" >::: layers_training;
         ]
end

module Agree_f64 =
  Agree
    (F64)
    (struct
      let name = "Tensor_blas.F64"
      let rtol = 1e-11
      let atol = 1e-12
      let exact_argmax = true
    end)

module Agree_f32 =
  Agree
    (F32)
    (struct
      let name = "Tensor_blas.F32"
      let rtol = 1e-5
      let atol = 1e-5
      let exact_argmax = false
    end)

let info =
  "backend info" >:: fun _ ->
  Printf.printf "flockml.blas uses BLAS: %s\n%!" Tensor_blas.blas;
  assert_equal ~printer:Fun.id "float64" Tensor_blas.F64.dtype;
  assert_equal ~printer:Fun.id "float32" Tensor_blas.F32.dtype

let () =
  Parallel.set_num_threads threads;
  run_test_tt_main
    ("backends"
    >::: [
           info;
           suites_f64;
           "agreement with the default backend"
           >::: [ Agree_f64.suite; Agree_f32.suite ];
         ])

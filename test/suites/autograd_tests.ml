(* Autograd: graph mechanics and a gradient check for every differentiable op.
   Run with: dune exec test/test_autograd.exe (default backend) *)

open OUnit2

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let mechanics =
    "mechanics"
    >::: [
           ( "const vs param" >:: fun _ ->
             assert_bool "const"
               (not (Var.requires_grad (Var.const (range [| 2 |]))));
             assert_bool "param" (Var.requires_grad (Var.param (range [| 2 |])))
           );
           ( "d/dx sum(x * x) = 2x" >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 2.; 3. |]) in
             Var.backward (Var.sum (Var.mul x x));
             assert_tensor (t [| 2 |] [| 4.; 6. |]) (Var.grad_exn x) );
           ( "node used twice accumulates" >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 2.; 3. |]) in
             let y = Var.mul x x in
             Var.backward (Var.sum (Var.add y y));
             assert_tensor (t [| 2 |] [| 8.; 12. |]) (Var.grad_exn x) );
           ( "grads accumulate across backward calls until zero_grad"
           >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 3. |]) in
             Var.backward (Var.sum (Var.scale 2.0 x));
             Var.backward (Var.sum (Var.scale 2.0 x));
             assert_tensor (t [| 1 |] [| 4. |]) (Var.grad_exn x);
             Var.zero_grad x;
             assert_bool "cleared" (Var.grad x = None) );
           ( "constants get no gradient" >:: fun _ ->
             let x = Var.param (range [| 3 |])
             and c = Var.const (range [| 3 |]) in
             Var.backward (Var.sum (Var.mul x c));
             assert_bool "const grad is None" (Var.grad c = None);
             assert_tensor (range [| 3 |]) (Var.grad_exn x) );
           ( "backward needs a scalar" >:: fun _ ->
             let x = Var.param (range [| 3 |]) in
             assert_invalid_arg (fun () -> Var.backward (Var.scale 2.0 x)) );
           ( "log_softmax rows exponentiate to probabilities" >:: fun _ ->
             let lp =
               Var.value
                 (Var.log_softmax
                    (Var.const (rand_tensor ~lo:(-5.) ~hi:5. [| 4; 6 |])))
             in
             assert_tensor (Tensor.ones [| 4 |])
               (Tensor.sum_axis (Tensor.exp lp) 1) );
           ( "log_softmax is stable for huge logits" >:: fun _ ->
             let lp =
               Var.value
                 (Var.log_softmax (Var.const (t [| 1; 2 |] [| 1000.; 0. |])))
             in
             assert_tensor (t [| 1; 2 |] [| 0.; -1000. |]) lp );
         ]

  (** Shorthand for {!Helpers.rand_tensor}: uniform samples in [[-1, 1)]. *)
  let r = rand_tensor

  (** Strictly positive samples, for ops undefined at or below 0 (log, div). *)
  let pos shape = rand_tensor ~lo:0.5 ~hi:2.0 shape

  let gradchecks =
    "gradcheck"
    >::: [
           gradcheck "add" [ r [| 3; 4 |]; r [| 3; 4 |] ] (binary Var.add);
           gradcheck "add broadcast row"
             [ r [| 3; 4 |]; r [| 4 |] ]
             (binary Var.add);
           gradcheck "add broadcast both"
             [ r [| 3; 1 |]; r [| 1; 4 |] ]
             (binary Var.add);
           gradcheck "sub broadcast"
             [ r [| 3; 4 |]; r [| 4 |] ]
             (binary Var.sub);
           gradcheck "mul broadcast"
             [ r [| 3; 4 |]; r [| 3; 1 |] ]
             (binary Var.mul);
           gradcheck "div broadcast"
             [ r [| 3; 4 |]; pos [| 4 |] ]
             (binary Var.div);
           gradcheck "neg" [ r [| 5 |] ] (unary Var.neg);
           gradcheck "scale" [ r [| 5 |] ] (unary (Var.scale 2.5));
           gradcheck "pow 3" [ r [| 5 |] ] (unary (fun a -> Var.pow a 3.0));
           gradcheck "pow 0.5" [ pos [| 5 |] ] (unary (fun a -> Var.pow a 0.5));
           gradcheck "exp" [ r [| 5 |] ] (unary Var.exp);
           gradcheck "log" [ pos [| 5 |] ] (unary Var.log);
           gradcheck "matmul" [ r [| 3; 4 |]; r [| 4; 2 |] ] (binary Var.matmul);
           gradcheck "transpose" [ r [| 3; 4 |] ] (unary Var.transpose);
           gradcheck "reshape"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.reshape a [| 2; 6 |]));
           gradcheck "sum" [ r [| 3; 4 |] ] (unary Var.sum);
           gradcheck "mean" [ r [| 3; 4 |] ] (unary Var.mean);
           gradcheck "sum_axis 0"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.sum_axis a 0));
           gradcheck "sum_axis 1"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.sum_axis a 1));
           gradcheck "sum_axis -1 keepdims"
             [ r [| 2; 3; 4 |] ]
             (unary (fun a -> Var.sum_axis ~keepdims:true a (-1)));
           gradcheck "relu" [ r [| 10 |] ] (unary Var.relu);
           gradcheck "sigmoid"
             [ r ~lo:(-5.) ~hi:5. [| 10 |] ]
             (unary Var.sigmoid);
           gradcheck "tanh" [ r ~lo:(-3.) ~hi:3. [| 10 |] ] (unary Var.tanh);
           gradcheck "log_softmax"
             [ r ~lo:(-3.) ~hi:3. [| 4; 5 |] ]
             (unary Var.log_softmax);
           gradcheck "two-layer MLP"
             [ r [| 5; 3 |]; r [| 3; 4 |]; r [| 4 |] ]
             (ternary (fun x w b ->
                  Var.tanh (Var.relu (Var.add (Var.matmul x w) b))));
           gradcheck "infix"
             [ r [| 4 |]; pos [| 4 |] ]
             (binary (fun a b -> Var.Infix.((a * b) - (a / b) + a)));
         ]

  (** The gradient of every op added with the NumPy-level API. *)
  let more_gradchecks =
    let mask = t [| 3; 1 |] [| 1.; 0.; 1. |] in
    let sum_list = function
      | [] -> invalid_arg "sum_list"
      | x :: rest -> List.fold_left Var.add (Var.sum x) (List.map Var.sum rest)
    in
    (* The pieces weighted differently, so each one's gradient is checked. *)
    let weigh pieces =
      sum_list
        (List.mapi (fun i p -> Var.scale (float_of_int (i + 1)) p) pieces)
    in
    "gradcheck (more ops)"
    >::: [
           gradcheck "sqrt" [ pos [| 5 |] ] (unary Var.sqrt);
           gradcheck "abs" [ r [| 6 |] ] (unary Var.abs);
           gradcheck "sign (zero gradient)" [ r [| 6 |] ] (unary Var.sign);
           gradcheck "square" [ r [| 6 |] ] (unary Var.square);
           gradcheck "rsqrt" [ pos [| 5 |] ] (unary Var.rsqrt);
           gradcheck "log1p" [ pos [| 5 |] ] (unary Var.log1p);
           gradcheck "expm1" [ r [| 5 |] ] (unary Var.expm1);
           gradcheck "sin" [ r ~lo:(-3.) ~hi:3. [| 5 |] ] (unary Var.sin);
           gradcheck "cos" [ r ~lo:(-3.) ~hi:3. [| 5 |] ] (unary Var.cos);
           gradcheck "erf" [ r ~lo:(-2.) ~hi:2. [| 5 |] ] (unary Var.erf);
           gradcheck "clamp"
             [ r [| 12 |] ]
             (unary (Var.clamp ~min:(-0.5) ~max:0.5));
           gradcheck "clamp min only"
             [ r [| 12 |] ]
             (unary (Var.clamp ~min:0.1));
           gradcheck "minimum broadcast"
             [ r [| 3; 4 |]; r [| 4 |] ]
             (binary Var.minimum);
           gradcheck "maximum broadcast"
             [ r [| 3; 1 |]; r [| 1; 4 |] ]
             (binary Var.maximum);
           gradcheck "where broadcast"
             [ r [| 3; 4 |]; r [| 4 |] ]
             (binary (Var.where mask));
           gradcheck "masked_fill"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.masked_fill a mask 5.));
           gradcheck "max_axis"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.max_axis a 1));
           gradcheck "min_axis keepdims"
             [ r [| 3; 4; 2 |] ]
             (unary (fun a -> Var.min_axis ~keepdims:true a 0));
           gradcheck "mean_axis"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.mean_axis a (-1)));
           gradcheck "sum_axes"
             [ r [| 2; 3; 4 |] ]
             (unary (fun a -> Var.sum_axes a [ 0; 2 ]));
           gradcheck "mean_axes keepdims"
             [ r [| 2; 3; 4 |] ]
             (unary (fun a -> Var.mean_axes ~keepdims:true a [ 1; -1 ]));
           gradcheck "var" [ r [| 3; 5 |] ] (unary (fun a -> Var.var a 1));
           gradcheck "var correction 1"
             [ r [| 4; 3 |] ]
             (unary (fun a -> Var.var ~correction:1 a 0));
           gradcheck "std"
             [ r [| 3; 5 |] ]
             (unary (fun a -> Var.std ~keepdims:true a 1));
           gradcheck "softmax"
             [ r ~lo:(-3.) ~hi:3. [| 3; 5 |] ]
             (unary Var.softmax);
           gradcheck "logsumexp"
             [ r ~lo:(-3.) ~hi:3. [| 3; 5 |] ]
             (unary (fun a -> Var.logsumexp a 0));
           gradcheck "logsumexp keepdims"
             [ r ~lo:(-3.) ~hi:3. [| 3; 5 |] ]
             (unary (fun a -> Var.logsumexp ~keepdims:true a 1));
           gradcheck "cumsum" [ r [| 3; 4 |] ] (unary (fun a -> Var.cumsum a 1));
           gradcheck "permute"
             [ r [| 2; 3; 4 |] ]
             (unary (fun a -> Var.permute a [| 2; 0; 1 |]));
           gradcheck "swapaxes"
             [ r [| 2; 3; 4 |] ]
             (unary (fun a -> Var.swapaxes a 0 2));
           gradcheck "squeeze / unsqueeze / flatten"
             [ r [| 1; 3; 4 |] ]
             (unary (fun a -> Var.flatten (Var.unsqueeze (Var.squeeze a) 1)));
           gradcheck "broadcast_to"
             [ r [| 3; 1 |] ]
             (unary (fun a -> Var.broadcast_to a [| 2; 3; 4 |]));
           gradcheck "slice"
             [ r [| 4; 5; 3 |] ]
             (unary (fun a ->
                  Var.slice a [ Range (1, 4); Step (0, 5, 2); Index (-1) ]));
           gradcheck "narrow"
             [ r [| 4; 5 |] ]
             (unary (fun a -> Var.narrow a ~axis:1 ~start:1 ~len:3));
           gradcheck "pad"
             [ r [| 2; 3 |] ]
             (unary (fun a -> Var.pad ~value:1. a [| (1, 0); (2, 1) |]));
           gradcheck "flip"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.flip a [ 0; 1 ]));
           gradcheck "concat"
             [ r [| 2; 3 |]; r [| 2; 1 |]; r [| 2; 4 |] ]
             (ternary (fun a b c -> Var.concat ~axis:1 [ a; b; c ]));
           gradcheck "stack"
             [ r [| 2; 3 |]; r [| 2; 3 |] ]
             (binary (fun a b -> Var.stack ~axis:2 [ a; b ]));
           gradcheck "split"
             [ r [| 2; 5 |] ]
             (unary (fun a -> weigh (Var.split a ~axis:1 [ 2; 0; 3 ])));
           gradcheck "chunk"
             [ r [| 7; 2 |] ]
             (unary (fun a -> weigh (Var.chunk a ~axis:0 3)));
           gradcheck "tile"
             [ r [| 2; 3 |] ]
             (unary (fun a -> Var.tile a [| 2; 1; 2 |]));
           gradcheck "repeat"
             [ r [| 2; 3 |] ]
             (unary (fun a -> Var.repeat a ~axis:0 3));
           gradcheck "index_select with repeats"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.index_select a ~axis:1 [| 2; 0; 2; 3 |]));
           gradcheck "gather"
             [ r [| 3; 4 |] ]
             (unary (fun a -> Var.gather a ~axis:1 [| 0; 3; 2; 2; 1; 0 |]));
           gradcheck "batched matmul, broadcasting"
             [ r [| 2; 1; 3; 4 |]; r [| 5; 4; 2 |] ]
             (binary Var.matmul);
           gradcheck "matmul 3-D by 2-D"
             [ r [| 2; 3; 4 |]; r [| 4; 5 |] ]
             (binary Var.matmul);
           gradcheck "matmul 2-D by 3-D"
             [ r [| 3; 4 |]; r [| 2; 4; 5 |] ]
             (binary Var.matmul);
           gradcheck "matmul 1-D by 2-D"
             [ r [| 4 |]; r [| 4; 3 |] ]
             (binary Var.matmul);
           gradcheck "matmul 3-D by 1-D"
             [ r [| 2; 3; 4 |]; r [| 4 |] ]
             (binary Var.matmul);
           gradcheck "matmul 1-D by 1-D"
             [ r [| 4 |]; r [| 4 |] ]
             (binary Var.matmul);
           gradcheck "dot" [ r [| 4 |]; r [| 4 |] ] (binary Var.dot);
           gradcheck "outer" [ r [| 4 |]; r [| 3 |] ] (binary Var.outer);
           gradcheck "norm" [ r [| 3; 4 |] ] (unary Var.norm);
           gradcheck "norm 1" [ r [| 3; 4 |] ] (unary (Var.norm ~p:1.));
           gradcheck "norm 3" [ r [| 3; 4 |] ] (unary (Var.norm ~p:3.));
           gradcheck "norm infinity"
             [ r [| 3; 4 |] ]
             (unary (Var.norm ~p:Float.infinity));
           gradcheck "attention-shaped graph"
             [ r [| 2; 3; 4 |]; r [| 2; 3; 4 |]; r [| 2; 3; 4 |] ]
             (ternary (fun q k v ->
                  let scores =
                    Var.scale 0.5 (Var.matmul q (Var.swapaxes k 1 2))
                  in
                  let causal =
                    Tensor.sub
                      (Tensor.ones [| 3; 3 |])
                      (Tensor.init [| 3; 3 |] (fun i ->
                           if i.(1) <= i.(0) then 1. else 0.))
                  in
                  let scores = Var.masked_fill scores causal (-1e9) in
                  Var.matmul (Var.softmax scores) v));
         ]

  (** GELU as a custom op, with its exact derivative (the example in
      Autograd_intf). *)
  let gelu x =
    let v = Var.value x in
    let cdf =
      Tensor.scale 0.5
        (Tensor.add_scalar 1.
           (Tensor.erf (Tensor.scale (1. /. Float.sqrt 2.) v)))
    in
    let pdf =
      Tensor.scale
        (1. /. Float.sqrt (2. *. Float.pi))
        (Tensor.exp (Tensor.scale (-0.5) (Tensor.square v)))
    in
    Var.custom ~name:"gelu" ~inputs:[ x ] ~value:(Tensor.mul v cdf)
      ~backward:(fun g ->
        [ Some (Tensor.mul g (Tensor.add cdf (Tensor.mul v pdf))) ])

  let raises_failure f =
    match f () with
    | _ -> assert_failure "expected Failure"
    | exception Failure _ -> ()

  let utilities =
    "custom ops and gradient utilities"
    >::: [
           gradcheck "custom gelu" [ r ~lo:(-3.) ~hi:3. [| 8 |] ] (unary gelu);
           gradcheck "custom op inside a graph"
             [ r [| 3; 4 |]; r [| 4; 2 |] ]
             (binary (fun x w -> Var.tanh (Var.matmul (gelu x) w)));
           ( "custom: gelu value and name" >:: fun _ ->
             let y = gelu (Var.const (t [| 3 |] [| -1.; 0.; 2. |])) in
             assert_tensor ~atol:1e-6
               (t [| 3 |] [| -0.158655; 0.; 1.954500 |])
               (Var.value y);
             assert_equal ~printer:Fun.id "gelu" (Var.op_name y) );
           ( "custom: a gradient shared with the upstream one is not mutated"
           >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 1.; 2. |]) in
             let y =
               Var.custom ~name:"identity" ~inputs:[ x ] ~value:(Var.value x)
                 ~backward:(fun g -> [ Some g ])
             in
             let z = Var.scale 3. y in
             Var.backward (Var.add (Var.sum z) (Var.sum x));
             assert_tensor (t [| 2 |] [| 4.; 4. |]) (Var.grad_exn x);
             assert_tensor (t [| 2 |] [| 3.; 3. |]) (Var.grad_exn y) );
           ( "custom: one gradient for two inputs, and None" >:: fun _ ->
             let a = Var.param (range [| 2 |])
             and b = Var.param (range [| 2 |]) in
             let c = Var.param (range [| 2 |]) in
             let y =
               Var.custom ~name:"add3" ~inputs:[ a; b; c ]
                 ~value:(Tensor.add (Var.value a) (Var.value b))
                 ~backward:(fun g -> [ Some g; Some g; None ])
             in
             Var.backward (Var.sum (Var.mul y (Var.const (range [| 2 |]))));
             assert_tensor (range [| 2 |]) (Var.grad_exn a);
             assert_tensor (range [| 2 |]) (Var.grad_exn b);
             assert_bool "no gradient for c" (Var.grad c = None) );
           ( "custom: wrong gradients are rejected" >:: fun _ ->
             let x = Var.param (range [| 2 |]) in
             let bad backward =
               Var.custom ~name:"bad" ~inputs:[ x ] ~value:(Var.value x)
                 ~backward
             in
             assert_invalid_arg (fun () ->
                 Var.backward (Var.sum (bad (fun g -> [ Some g; Some g ]))));
             assert_invalid_arg (fun () ->
                 Var.backward
                   (Var.sum (bad (fun _ -> [ Some (Tensor.zeros [| 3 |]) ]))))
           );
           ( "maximum splits ties; max_axis picks the first maximum" >:: fun _ ->
             let a = Var.param (t [| 2 |] [| 1.; 2. |])
             and b = Var.param (t [| 2 |] [| 1.; 3. |]) in
             Var.backward (Var.sum (Var.maximum a b));
             assert_tensor (t [| 2 |] [| 0.5; 0. |]) (Var.grad_exn a);
             assert_tensor (t [| 2 |] [| 0.5; 1. |]) (Var.grad_exn b);
             let x = Var.param (t [| 3 |] [| 2.; 2.; 1. |]) in
             Var.backward (Var.max_axis x 0);
             assert_tensor (t [| 3 |] [| 1.; 0.; 0. |]) (Var.grad_exn x) );
           ( "set_requires_grad freezes a parameter" >:: fun _ ->
             let w = Var.param (range [| 3 |])
             and b = Var.param (range [| 3 |]) in
             Var.set_requires_grad w false;
             Var.backward (Var.sum (Var.mul w b));
             assert_bool "frozen" (Var.grad w = None);
             assert_tensor (range [| 3 |]) (Var.grad_exn b);
             let opt = Optim.sgd ~lr:1. [ w; b ] in
             Optim.step opt;
             assert_tensor (range [| 3 |]) (Var.value w);
             Var.set_requires_grad w true;
             assert_bool "unfrozen" (Var.requires_grad (Var.mul w b));
             assert_invalid_arg (fun () ->
                 Var.set_requires_grad (Var.mul w b) false) );
           ( "grad_norm and clip_grad_norm" >:: fun _ ->
             let a = Var.param (t [| 1 |] [| 0. |])
             and b = Var.param (t [| 1 |] [| 0. |]) in
             let c = Var.param (t [| 1 |] [| 0. |]) in
             Var.backward
               (Var.add (Var.sum (Var.scale 3. a)) (Var.sum (Var.scale 4. b)));
             assert_float 5. (Var.grad_norm [ a; b; c ]);
             assert_float 5. (Var.clip_grad_norm ~max_norm:10. [ a; b; c ]);
             assert_tensor (t [| 1 |] [| 3. |]) (Var.grad_exn a);
             assert_float 5. (Var.clip_grad_norm ~max_norm:1. [ a; b; c ]);
             assert_tensor ~atol:1e-6 (t [| 1 |] [| 0.6 |]) (Var.grad_exn a);
             assert_tensor ~atol:1e-6 (t [| 1 |] [| 0.8 |]) (Var.grad_exn b);
             assert_bool "c untouched" (Var.grad c = None) );
           ( "clip_grad_value, including a shared gradient" >:: fun _ ->
             let x = Var.param (t [| 3 |] [| 0.; 0.; 0. |]) in
             let y = Var.add x (Var.const (Tensor.zeros [| 3 |])) in
             Var.backward
               (Var.sum (Var.mul y (Var.const (t [| 3 |] [| -5.; 0.5; 2. |]))));
             let upstream = Tensor.copy (Var.grad_exn y) in
             Var.clip_grad_value 1. [ x ];
             assert_tensor (t [| 3 |] [| -1.; 0.5; 1. |]) (Var.grad_exn x);
             assert_tensor upstream (Var.grad_exn y) );
           ( "detect_anomaly finds a NaN forward value" >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 1.; -1. |]) in
             ignore (Var.log x);
             raises_failure (fun () -> Var.detect_anomaly (fun () -> Var.log x));
             ignore (Var.log x) );
           ( "detect_anomaly finds an infinite gradient" >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 0.; 4. |]) in
             Var.backward (Var.sum (Var.sqrt x));
             Var.zero_grad x;
             raises_failure (fun () ->
                 Var.detect_anomaly (fun () ->
                     Var.backward (Var.sum (Var.sqrt x)))) );
           ( "value_and_grad" >:: fun _ ->
             let w = Var.param (t [| 3 |] [| 1.; 2.; 3. |]) in
             let f x =
               Var.add (Var.sum (Var.mul x x)) (Var.sum (Var.mul x w))
             in
             let x = t [| 3 |] [| 1.; -1.; 0.5 |] in
             let v, g = Var.value_and_grad f x in
             assert_float (2.25 +. 0.5) v;
             assert_tensor (t [| 3 |] [| 3.; 0.; 4. |]) g;
             assert_bool "the parameter got no gradient" (Var.grad w = None);
             assert_tensor g (Var.no_grad (fun () -> Var.gradient f x));
             assert_tensor (Tensor.zeros [| 3 |])
               (Var.gradient (fun _ -> Var.sum w) x);
             assert_invalid_arg (fun () -> Var.gradient (fun x -> x) x);
             (* recording and normal backward still work afterwards *)
             Var.backward (Var.sum (Var.mul w w));
             assert_tensor (t [| 3 |] [| 2.; 4.; 6. |]) (Var.grad_exn w) );
         ]

  let suite =
    "autograd" >::: [ mechanics; gradchecks; more_gradchecks; utilities ]
end

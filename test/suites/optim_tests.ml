(* Optimizers, checked against hand-computed updates.
   Run with: dune exec test/test_optim.exe (default backend) *)

open OUnit2

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  (** [sum ((x - target)²)]: gradient [2 (x - target)], minimum at [target]. *)
  let quadratic ?(target = 0.0) x =
    Var.sum (Var.pow (Var.sub x (Var.scalar target)) 2.0)

  (** Take [steps] optimizer steps on {!quadratic}. *)
  let run opt x ~steps ~target =
    for _ = 1 to steps do
      Optim.zero_grad opt;
      Var.backward (quadratic ~target x);
      Optim.step opt
    done

  let tests =
    "optim"
    >::: [
           ( "sgd single step" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt = Optim.sgd ~lr:0.1 [ x ] in
             run opt x ~steps:1 ~target:0.0;
             (* grad = 2x = 2, x = 1 - 0.1 * 2 *)
             assert_tensor (t [| 1 |] [| 0.8 |]) (Var.value x) );
           ( "sgd momentum, two steps" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt = Optim.sgd ~momentum:0.9 ~lr:0.1 [ x ] in
             run opt x ~steps:2 ~target:0.0;
             (* v1 = 2, x1 = 0.8; g2 = 1.6, v2 = 0.9 * 2 + 1.6 = 3.4, x2 = 0.46 *)
             assert_tensor (t [| 1 |] [| 0.46 |]) (Var.value x) );
           ( "sgd weight decay" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt = Optim.sgd ~weight_decay:0.5 ~lr:0.1 [ x ] in
             Var.backward (Var.sum (Var.scale 0.0 x));
             Optim.step opt;
             (* g = 0 + 0.5 * 1, x = 1 - 0.1 * 0.5 *)
             assert_tensor (t [| 1 |] [| 0.95 |]) (Var.value x) );
           ( "sgd converges" >:: fun _ ->
             let x = Var.param (Tensor.zeros [| 3 |]) in
             run (Optim.sgd ~lr:0.1 [ x ]) x ~steps:100 ~target:3.0;
             assert_tensor ~atol:1e-3 (Tensor.full [| 3 |] 3.0) (Var.value x) );
           ( "adam first step moves each param by lr" >:: fun _ ->
             let x = Var.param (t [| 2 |] [| 1.0; -2.0 |]) in
             run (Optim.adam ~lr:0.1 [ x ]) x ~steps:1 ~target:0.0;
             assert_tensor ~atol:1e-6 (t [| 2 |] [| 0.9; -1.9 |]) (Var.value x)
           );
           ( "adam converges" >:: fun _ ->
             let x = Var.param (Tensor.zeros [| 3 |]) in
             run (Optim.adam ~lr:0.1 [ x ]) x ~steps:1000 ~target:3.0;
             assert_tensor ~atol:1e-2 (Tensor.full [| 3 |] 3.0) (Var.value x) );
           ( "params without gradients are left alone" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |])
             and unused = Var.param (t [| 1 |] [| 5.0 |]) in
             let opt = Optim.adam ~lr:0.1 [ x; unused ] in
             run opt x ~steps:3 ~target:0.0;
             assert_tensor (t [| 1 |] [| 5.0 |]) (Var.value unused) );
           ( "adamw decays the parameter, not the gradient" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 2.0 |]) in
             let opt = Optim.adamw ~lr:0.1 ~weight_decay:0.5 [ x ] in
             Var.backward (Var.sum (Var.scale 0.0 x));
             Optim.step opt;
             (* g = 0, so the Adam step is 0: x = 2 (1 - 0.1 * 0.5) *)
             assert_tensor (t [| 1 |] [| 1.9 |]) (Var.value x) );
           ( "nesterov, two steps" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt = Optim.sgd ~momentum:0.9 ~nesterov:true ~lr:0.1 [ x ] in
             run opt x ~steps:2 ~target:0.0;
             (* g1 = 2, v1 = 2, x1 = 1 - 0.1 (2 + 1.8) = 0.62;
                g2 = 1.24, v2 = 3.04, x2 = 0.62 - 0.1 (1.24 + 2.736) *)
             assert_tensor
               (t [| 1 |] [| 0.62 -. (0.1 *. (1.24 +. 2.736)) |])
               (Var.value x);
             assert_invalid_arg (fun () ->
                 Optim.sgd ~nesterov:true ~lr:0.1 [ x ]) );
           ( "rmsprop and adagrad converge" >:: fun _ ->
             List.iter
               (fun make ->
                 let x = Var.param (Tensor.zeros [| 3 |]) in
                 run (make [ x ]) x ~steps:500 ~target:3.0;
                 assert_tensor ~atol:2e-2 (Tensor.full [| 3 |] 3.0)
                   (Var.value x))
               [
                 (fun ps -> Optim.rmsprop ~lr:0.01 ps);
                 (fun ps ->
                   Optim.rmsprop ~lr:0.01 ~momentum:0.5 ~centered:true ps);
                 (fun ps -> Optim.adagrad ~lr:0.5 ps);
               ] );
           ( "parameter groups" >:: fun _ ->
             let a = Var.param (t [| 1 |] [| 1.0 |])
             and b = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt =
               Optim.sgd ~lr:0.1 ~groups:[ Optim.group ~lr:0.01 [ b ] ] [ a ]
             in
             run opt a ~steps:1 ~target:0.0;
             Optim.zero_grad opt;
             Var.backward (quadratic b);
             Optim.step opt;
             assert_tensor (t [| 1 |] [| 0.8 |]) (Var.value a);
             assert_tensor (t [| 1 |] [| 0.98 |]) (Var.value b);
             Optim.set_lr opt 0.5;
             List.iter2
               (fun e a -> assert_float ~eps:1e-15 e a)
               [ 0.5; 0.05 ] (Optim.group_lrs opt);
             assert_equal [ a; b ] (Optim.params opt);
             assert_invalid_arg (fun () ->
                 Optim.adam ~lr:0.1 ~groups:[ Optim.group [ a ] ] [ a ]) );
           ( "state_dict round trip" >:: fun _ ->
             let make () =
               let x = Var.param (t [| 2 |] [| 1.0; -2.0 |]) in
               (x, Optim.adam ~lr:0.1 [ x ])
             in
             let x, opt = make () in
             run opt x ~steps:3 ~target:0.0;
             assert_equal ~printer:(String.concat " ")
               [ "lr"; "0.step"; "0.exp_avg"; "0.exp_avg_sq" ]
               (List.map fst (Optim.state_dict opt));
             let y, opt' = make () in
             Tensor.blit_ ~src:(Var.value x) ~dst:(Var.value y);
             Optim.load_state_dict opt' (Optim.state_dict opt);
             run opt x ~steps:2 ~target:0.0;
             run opt' y ~steps:2 ~target:0.0;
             assert_tensor ~rtol:0. ~atol:0. (Var.value x) (Var.value y);
             let _, sgd = make () in
             assert_raises
               (Failure "Optim.load_state_dict: no entry 0.momentum_buffer")
               (fun () ->
                 Optim.load_state_dict
                   (Optim.sgd ~momentum:0.9 ~lr:0.1 (Optim.params sgd))
                   (Optim.state_dict opt)) );
           ( "gradient accumulation = one big batch" >:: fun _ ->
             (* two half batches, each loss halved, add up to the full batch's
                mean gradient *)
             let data = range [| 4; 2 |] in
             let w = Var.param (t [| 2; 1 |] [| 0.5; -0.25 |]) in
             let loss rows =
               Loss.mse
                 (Var.matmul (Var.const (Tensor.take_rows data rows)) w)
                 (Var.const (Tensor.ones [| Array.length rows; 1 |]))
             in
             Var.backward (loss [| 0; 1; 2; 3 |]);
             let full = Tensor.copy (Var.grad_exn w) in
             Var.zero_grad w;
             Var.backward (Var.scale 0.5 (loss [| 0; 1 |]));
             Var.backward (Var.scale 0.5 (loss [| 2; 3 |]));
             assert_tensor ~atol:1e-12 full (Var.grad_exn w) );
           ( "zero_grad and set_lr" >:: fun _ ->
             let x = Var.param (t [| 1 |] [| 1.0 |]) in
             let opt = Optim.sgd ~lr:0.1 [ x ] in
             Var.backward (quadratic x);
             Optim.zero_grad opt;
             assert_bool "grad cleared" (Var.grad x = None);
             Optim.set_lr opt 0.01;
             assert_float 0.01 (Optim.lr opt) );
         ]

  let suite = tests
end

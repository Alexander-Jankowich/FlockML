(* Phase 4: optimizers. Run with: dune exec test/test_optim.exe *)

open Flockml
open OUnit2
open Helpers

(* loss = sum((x - target)^2) *)
let quadratic ?(target = 0.0) x =
  Var.sum (Var.pow (Var.sub x (Var.scalar target)) 2.0)

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
           assert_tensor ~atol:1e-6 (t [| 2 |] [| 0.9; -1.9 |]) (Var.value x) );
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
         ( "zero_grad and set_lr" >:: fun _ ->
           let x = Var.param (t [| 1 |] [| 1.0 |]) in
           let opt = Optim.sgd ~lr:0.1 [ x ] in
           Var.backward (quadratic x);
           Optim.zero_grad opt;
           assert_bool "grad cleared" (Var.grad x = None);
           Optim.set_lr opt 0.01;
           assert_float 0.01 (Optim.lr opt) );
       ]

let () = run_test_tt_main tests

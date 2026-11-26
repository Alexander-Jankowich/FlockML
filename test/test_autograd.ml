(* Phase 2: autograd. Run with: dune exec test/test_autograd.exe *)

open Flockml
open OUnit2
open Helpers

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
         ( "grads accumulate across backward calls until zero_grad" >:: fun _ ->
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

let r = rand_tensor
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
         gradcheck "sub broadcast" [ r [| 3; 4 |]; r [| 4 |] ] (binary Var.sub);
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
         gradcheck "sigmoid" [ r ~lo:(-5.) ~hi:5. [| 10 |] ] (unary Var.sigmoid);
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

let () = run_test_tt_main ("autograd" >::: [ mechanics; gradchecks ])

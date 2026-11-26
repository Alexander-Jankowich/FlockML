(** The smallest end-to-end test of the whole stack: learn XOR with a 2-8-1
   MLP. Linear models can't solve XOR, so success here means tensors,
   autograd, layers, the loss and the optimizer all work together.

   Run with: dune exec bin/xor.exe
   Expected: loss drops below ~0.01 and predictions approach 0 1 1 0. *)

open Flockml

let () =
  let rng = Random.State.make [| 42 |] in
  let x =
    Var.const (Tensor.of_array [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |])
  in
  let y = Var.const (Tensor.of_array [| 4; 1 |] [| 0.; 1.; 1.; 0. |]) in
  let model =
    Layer.sequential
      [
        Layer.linear ~rng ~in_dim:2 ~out_dim:8 ();
        Layer.tanh;
        Layer.linear ~rng ~in_dim:8 ~out_dim:1 ();
        Layer.sigmoid;
      ]
  in
  print_endline (Layer.to_string model);
  let opt = Optim.adam ~lr:0.05 (Layer.params model) in
  for epoch = 1 to 1000 do
    Optim.zero_grad opt;
    let loss = Loss.mse (Layer.forward model x) y in
    Var.backward loss;
    Optim.step opt;
    if epoch mod 100 = 0 then
      Printf.printf "epoch %4d  loss %.6f\n%!" epoch
        (Tensor.item (Var.value loss));
    if epoch = 1 then Graphviz.save_dot "xor_graph.dot" loss
  done;
  Format.printf "predictions:@.%a@." Tensor.pp
    (Var.value (Layer.forward model x))

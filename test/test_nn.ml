(* Neural-network building blocks: init, losses, layers, data, metrics.
   Run with: dune exec test/test_nn.exe *)

open Flockml
open OUnit2
open Helpers

let init =
  "init"
  >::: [
         ( "xavier_uniform bounds" >:: fun _ ->
           let a = Float.sqrt (6.0 /. (30.0 +. 50.0)) in
           Tensor.to_array
             (Init.xavier_uniform ~rng ~fan_in:30 ~fan_out:50 [| 30; 50 |])
           |> Array.iter (fun v ->
               assert_bool "within bound" (Float.abs v <= a)) );
         ( "he_normal std" >:: fun _ ->
           let x =
             Tensor.to_array (Init.he_normal ~rng ~fan_in:100 [| 100; 200 |])
           in
           let n = float_of_int (Array.length x) in
           let var =
             Array.fold_left (fun acc v -> acc +. (v *. v)) 0.0 x /. n
           in
           assert_float ~eps:0.01 (Float.sqrt 0.02) (Float.sqrt var) );
       ]

let losses =
  "losses"
  >::: [
         ( "mse value" >:: fun _ ->
           let l =
             Loss.mse
               (Var.const (range [| 3 |]))
               (Var.const (Tensor.ones [| 3 |]))
           in
           assert_float (5.0 /. 3.0) (Tensor.item (Var.value l)) );
         ( "mse gradient" >:: fun _ ->
           let p = Var.param (range [| 3 |]) in
           Var.backward (Loss.mse p (Var.const (Tensor.ones [| 3 |])));
           assert_tensor
             (t [| 3 |] [| 0.; 2. /. 3.; 4. /. 3. |])
             (Var.grad_exn p) );
         ( "cross_entropy of uniform logits is log(classes)" >:: fun _ ->
           let l =
             Loss.cross_entropy
               ~logits:(Var.const (Tensor.zeros [| 2; 10 |]))
               ~labels:[| 3; 7 |]
           in
           assert_float (Float.log 10.0) (Tensor.item (Var.value l)) );
         ( "cross_entropy confident and correct" >:: fun _ ->
           let l =
             Loss.cross_entropy
               ~logits:(Var.const (t [| 1; 3 |] [| 10.; 0.; 0. |]))
               ~labels:[| 0 |]
           in
           assert_float ~eps:1e-9
             (Float.log (1.0 +. (2.0 *. Float.exp (-10.0))))
             (Tensor.item (Var.value l)) );
         ( "cross_entropy gradient = softmax - onehot" >:: fun _ ->
           let logits = Var.param (Tensor.zeros [| 1; 3 |]) in
           Var.backward (Loss.cross_entropy ~logits ~labels:[| 0 |]);
           assert_tensor
             (t [| 1; 3 |] [| -2. /. 3.; 1. /. 3.; 1. /. 3. |])
             (Var.grad_exn logits) );
         ( "binary_cross_entropy" >:: fun _ ->
           let l =
             Loss.binary_cross_entropy
               (Var.const (t [| 2 |] [| 0.9; 0.2 |]))
               (Var.const (t [| 2 |] [| 1.; 0. |]))
           in
           assert_float ~eps:1e-5
             (-.(Float.log 0.9 +. Float.log 0.8) /. 2.0)
             (Tensor.item (Var.value l)) );
       ]

let layers =
  "layers"
  >::: [
         ( "linear computes x @ w + b" >:: fun _ ->
           let layer =
             Layer.Linear
               {
                 w = Var.param (range [| 2; 2 |]);
                 b = Var.param (t [| 2 |] [| 10.; 20. |]);
               }
           in
           let y = Layer.forward layer (Var.const (Tensor.ones [| 1; 2 |])) in
           assert_tensor (t [| 1; 2 |] [| 14.; 26. |]) (Var.value y) );
         ( "linear shapes and parameter count" >:: fun _ ->
           let layer = Layer.linear ~rng ~in_dim:4 ~out_dim:3 () in
           assert_shape [| 5; 3 |]
             (Var.value
                (Layer.forward layer (Var.const (rand_tensor [| 5; 4 |]))));
           assert_equal ~printer:string_of_int 15 (Layer.num_params layer) );
         ( "sequential params" >:: fun _ ->
           let model =
             Layer.sequential
               [
                 Layer.linear ~rng ~in_dim:4 ~out_dim:8 ();
                 Layer.relu;
                 Layer.linear ~rng ~in_dim:8 ~out_dim:2 ();
               ]
           in
           assert_equal ~printer:string_of_int 4
             (List.length (Layer.params model));
           assert_equal ~printer:string_of_int 58 (Layer.num_params model);
           assert_shape [| 3; 2 |]
             (Var.value
                (Layer.forward model (Var.const (rand_tensor [| 3; 4 |])))) );
         ( "relu layer" >:: fun _ ->
           let y =
             Layer.forward Layer.relu (Var.const (t [| 3 |] [| -1.; 0.5; 2. |]))
           in
           assert_tensor (t [| 3 |] [| 0.; 0.5; 2. |]) (Var.value y) );
         ( "flatten" >:: fun _ ->
           assert_shape [| 2; 12 |]
             (Var.value
                (Layer.forward Layer.flatten (Var.const (range [| 2; 3; 4 |]))))
         );
         ( "dropout is identity in eval mode" >:: fun _ ->
           let x = rand_tensor [| 100 |] in
           assert_tensor x
             (Var.value (Layer.forward (Layer.dropout 0.5) (Var.const x))) );
         ( "dropout in train mode" >:: fun _ ->
           let y =
             Layer.forward ~train:true ~rng (Layer.dropout 0.25)
               (Var.const (Tensor.ones [| 10000 |]))
             |> Var.value |> Tensor.to_array
           in
           let zeros =
             Array.fold_left (fun n v -> if v = 0.0 then n + 1 else n) 0 y
           in
           Array.iter
             (fun v ->
               assert_bool "0 or 1/(1-p)"
                 (v = 0.0 || Float.abs (v -. (1. /. 0.75)) < 1e-9))
             y;
           assert_float ~eps:0.03 0.25 (float_of_int zeros /. 10000.0) );
       ]

let data_and_metrics =
  let d =
    { Dataset.inputs = range [| 10; 2 |]; labels = Array.init 10 Fun.id }
  in
  "data and metrics"
  >::: [
         ( "shuffled_indices is a permutation" >:: fun _ ->
           let p = Dataset.shuffled_indices rng 100 in
           let sorted = Array.copy p in
           Array.sort compare sorted;
           assert_equal (Array.init 100 Fun.id) sorted );
         ( "batches without shuffle" >:: fun _ ->
           let bs = List.of_seq (Dataset.batches ~batch_size:4 d) in
           assert_equal ~printer:string_of_int 3 (List.length bs);
           assert_equal
             [ [| 0; 1; 2; 3 |]; [| 4; 5; 6; 7 |]; [| 8; 9 |] ]
             (List.map snd bs);
           assert_tensor
             (t [| 2; 2 |] [| 17.; 18.; 19.; 20. |])
             (fst (List.nth bs 2)) );
         ( "batches with shuffle cover everything once" >:: fun _ ->
           let labels =
             Dataset.batches ~shuffle:true ~rng ~batch_size:3 d
             |> Seq.map snd |> List.of_seq |> Array.concat
           in
           Array.sort compare labels;
           assert_equal (Array.init 10 Fun.id) labels );
         ( "split" >:: fun _ ->
           let a, b = Dataset.split ~rng ~ratio:0.8 d in
           assert_equal 8 (Dataset.length a);
           assert_equal 2 (Dataset.length b) );
         ( "accuracy" >:: fun _ ->
           assert_float 0.75
             (Metrics.accuracy ~predictions:[| 0; 1; 2; 2 |]
                ~labels:[| 0; 1; 1; 2 |]) );
         ( "confusion_matrix" >:: fun _ ->
           assert_equal
             [| [| 1; 0; 0 |]; [| 0; 1; 1 |]; [| 0; 0; 1 |] |]
             (Metrics.confusion_matrix ~num_classes:3
                ~predictions:[| 0; 1; 2; 2 |] ~labels:[| 0; 1; 1; 2 |]) );
       ]

let () = run_test_tt_main ("nn" >::: [ init; losses; layers; data_and_metrics ])

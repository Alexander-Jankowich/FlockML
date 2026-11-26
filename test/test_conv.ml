(* Convolution, pooling and graph export.
   Run with: dune exec test/test_conv.exe *)

open Flockml
open OUnit2
open Helpers

(** The inner product of two same-sized tensors, for the adjoint test
    [<im2col x, y> = <x, col2im y>]. *)
let dot a b =
  let a = Tensor.to_array a and b = Tensor.to_array b in
  let s = ref 0.0 in
  Array.iteri (fun i x -> s := !s +. (x *. b.(i))) a;
  !s

let tensor_level =
  "tensor level"
  >::: [
         ( "im2col 3x3 image, 2x2 kernel" >:: fun _ ->
           assert_tensor
             (t [| 4; 4 |]
                [|
                  1.; 2.; 4.; 5.; 2.; 3.; 5.; 6.; 4.; 5.; 7.; 8.; 5.; 6.; 8.; 9.;
                |])
             (Tensor.im2col
                (range [| 1; 3; 3; 1 |])
                ~kh:2 ~kw:2 ~stride:1 ~padding:0) );
         ( "im2col with padding" >:: fun _ ->
           let cols =
             Tensor.im2col
               (range [| 1; 3; 3; 1 |])
               ~kh:3 ~kw:3 ~stride:1 ~padding:1
           in
           assert_shape [| 9; 9 |] cols;
           assert_tensor
             (t [| 1; 9 |] [| 0.; 0.; 0.; 0.; 1.; 2.; 0.; 4.; 5. |])
             (Tensor.take_rows cols [| 0 |]);
           assert_tensor (range [| 1; 9 |]) (Tensor.take_rows cols [| 4 |]) );
         ( "im2col column order is (ky, kx, channel)" >:: fun _ ->
           assert_tensor
             (range [| 1; 8 |])
             (Tensor.im2col
                (range [| 1; 2; 2; 2 |])
                ~kh:2 ~kw:2 ~stride:1 ~padding:0) );
         ( "col2im is the adjoint of im2col" >:: fun _ ->
           let input_shape = [| 2; 5; 5; 3 |] in
           let x = rand_tensor input_shape in
           let cols = Tensor.im2col x ~kh:3 ~kw:3 ~stride:2 ~padding:1 in
           let y = rand_tensor (Tensor.shape cols) in
           let back =
             Tensor.col2im y ~input_shape ~kh:3 ~kw:3 ~stride:2 ~padding:1
           in
           assert_float ~eps:1e-9 (dot cols y) (dot x back) );
         ( "max_pool2d" >:: fun _ ->
           let out, argmax =
             Tensor.max_pool2d (range [| 1; 4; 4; 1 |]) ~size:2 ~stride:2
           in
           assert_tensor (t [| 1; 2; 2; 1 |] [| 6.; 8.; 14.; 16. |]) out;
           assert_equal [| 5; 7; 13; 15 |] argmax );
       ]

(** Shorthand for {!Helpers.rand_tensor}. *)
let r = rand_tensor

let gradchecks =
  "gradcheck"
  >::: [
         gradcheck "im2col"
           [ r [| 2; 4; 4; 3 |] ]
           (unary (fun x -> Var.im2col x ~kh:3 ~kw:3 ~stride:1 ~padding:1));
         gradcheck "max_pool2d"
           [ r [| 2; 4; 4; 3 |] ]
           (unary (fun x -> Var.max_pool2d x ~size:2 ~stride:2));
         gradcheck "conv2d layer"
           [ r [| 2; 5; 5; 2 |]; r [| 18; 3 |]; r [| 3 |] ]
           (ternary (fun x w b ->
                Layer.forward
                  (Layer.Conv2d { w; b; kernel = 3; stride = 1; padding = 1 })
                  x));
       ]

let layers =
  "layers"
  >::: [
         ( "conv2d output shapes" >:: fun _ ->
           let x = Var.const (r [| 2; 8; 8; 3 |]) in
           let conv stride =
             Layer.conv2d ~rng ~stride ~padding:1 ~in_channels:3 ~out_channels:4
               ~kernel:3 ()
           in
           assert_shape [| 2; 8; 8; 4 |] (Var.value (Layer.forward (conv 1) x));
           assert_shape [| 2; 4; 4; 4 |] (Var.value (Layer.forward (conv 2) x))
         );
         ( "graphviz export" >:: fun _ ->
           let x = Var.param (r [| 2; 3 |]) and w = Var.param (r [| 3; 1 |]) in
           let dot = Graphviz.to_dot (Var.sum (Var.matmul x w)) in
           let contains sub =
             let n = String.length sub in
             let rec go i =
               i + n <= String.length dot
               && (String.sub dot i n = sub || go (i + 1))
             in
             go 0
           in
           assert_bool "digraph" (contains "digraph");
           assert_bool "matmul node" (contains "matmul") );
       ]

let () = run_test_tt_main ("conv" >::: [ tensor_level; gradchecks; layers ])

(* Tensor backend: construction, broadcasting, reductions, linear algebra and
   in-place ops. Run with: dune exec test/test_tensor.exe (default backend) *)

open OUnit2

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let construction =
    "construction"
    >::: [
           ( "of_array rejects wrong length" >:: fun _ ->
             assert_invalid_arg (fun () -> t [| 2; 3 |] [| 1.; 2. |]) );
           ( "scalar has empty shape" >:: fun _ ->
             assert_shape [||] (Tensor.scalar 3.0);
             assert_float 3.0 (Tensor.item (Tensor.scalar 3.0)) );
           ( "get" >:: fun _ ->
             let x = range [| 2; 3 |] in
             assert_float 6.0 (Tensor.get x [| 1; 2 |]);
             assert_float 2.0 (Tensor.get x [| 0; 1 |]) );
           ( "get out of bounds" >:: fun _ ->
             let x = range [| 2; 3 |] in
             assert_invalid_arg (fun () -> Tensor.get x [| 2; 0 |]);
             assert_invalid_arg (fun () -> Tensor.get x [| 0 |]) );
           ( "set" >:: fun _ ->
             let x = Tensor.zeros [| 2; 2 |] in
             Tensor.set x [| 1; 0 |] 7.0;
             assert_tensor (t [| 2; 2 |] [| 0.; 0.; 7.; 0. |]) x );
         ]

  let random =
    let stats x =
      let a = Tensor.to_array x in
      let n = float_of_int (Array.length a) in
      let mean = Array.fold_left ( +. ) 0.0 a /. n in
      let var =
        Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.0)) 0.0 a /. n
      in
      (mean, Float.sqrt var)
    in
    "random"
    >::: [
           ( "rand_uniform range and mean" >:: fun _ ->
             let x = Tensor.rand_uniform ~rng ~lo:2.0 ~hi:4.0 [| 20000 |] in
             Array.iter
               (fun v -> assert_bool "in [2, 4)" (v >= 2.0 && v < 4.0))
               (Tensor.to_array x);
             assert_float ~eps:0.05 3.0 (fst (stats x)) );
           ( "randn mean and std" >:: fun _ ->
             let m, s = stats (Tensor.randn ~rng [| 20000 |]) in
             assert_float ~eps:0.05 0.0 m;
             assert_float ~eps:0.05 1.0 s );
           ( "randn custom mean and std" >:: fun _ ->
             let m, s =
               stats (Tensor.randn ~rng ~mean:5.0 ~std:2.0 [| 20000 |])
             in
             assert_float ~eps:0.1 5.0 m;
             assert_float ~eps:0.1 2.0 s );
         ]

  let elementwise =
    "elementwise"
    >::: [
           ( "map" >:: fun _ ->
             assert_tensor
               (t [| 3 |] [| 2.; 4.; 6. |])
               (Tensor.map (fun x -> 2.0 *. x) (range [| 3 |])) );
           ( "add same shape" >:: fun _ ->
             assert_tensor
               (t [| 3 |] [| 11.; 22.; 33. |])
               (Tensor.add (range [| 3 |]) (t [| 3 |] [| 10.; 20.; 30. |])) );
           ( "exp / log roundtrip" >:: fun _ ->
             let x = range [| 4 |] in
             assert_tensor x (Tensor.log (Tensor.exp x)) );
         ]

  let broadcasting =
    "broadcasting"
    >::: [
           ( "broadcast_shapes" >:: fun _ ->
             let show s =
               String.concat ";" (Array.to_list (Array.map string_of_int s))
             in
             let check a b expected =
               assert_equal ~printer:show expected (Tensor.broadcast_shapes a b)
             in
             check [| 2; 3 |] [| 3 |] [| 2; 3 |];
             check [| 4; 1 |] [| 1; 5 |] [| 4; 5 |];
             check [||] [| 2; 2 |] [| 2; 2 |];
             check [| 3; 1; 2 |] [| 4; 1 |] [| 3; 4; 2 |] );
           ( "broadcast_shapes incompatible" >:: fun _ ->
             assert_invalid_arg (fun () ->
                 Tensor.broadcast_shapes [| 2; 3 |] [| 4 |]) );
           ( "add row vector" >:: fun _ ->
             assert_tensor
               (t [| 2; 3 |] [| 11.; 22.; 33.; 14.; 25.; 36. |])
               (Tensor.add (range [| 2; 3 |]) (t [| 3 |] [| 10.; 20.; 30. |]))
           );
           ( "mul column vector" >:: fun _ ->
             assert_tensor
               (t [| 2; 3 |] [| 2.; 4.; 6.; 12.; 15.; 18. |])
               (Tensor.mul (range [| 2; 3 |]) (t [| 2; 1 |] [| 2.; 3. |])) );
           ( "add scalar tensor" >:: fun _ ->
             assert_tensor
               (t [| 2; 2 |] [| 2.; 3.; 4.; 5. |])
               (Tensor.add (range [| 2; 2 |]) (Tensor.scalar 1.0)) );
           ( "broadcast_to" >:: fun _ ->
             assert_tensor
               (t [| 2; 3 |] [| 1.; 2.; 3.; 1.; 2.; 3. |])
               (Tensor.broadcast_to (range [| 3 |]) [| 2; 3 |]) );
           ( "sum_to_shape" >:: fun _ ->
             let x = range [| 2; 3 |] in
             assert_tensor
               (t [| 3 |] [| 5.; 7.; 9. |])
               (Tensor.sum_to_shape x [| 3 |]);
             assert_tensor
               (t [| 2; 1 |] [| 6.; 15. |])
               (Tensor.sum_to_shape x [| 2; 1 |]);
             assert_tensor (Tensor.scalar 21.0) (Tensor.sum_to_shape x [||]);
             assert_tensor x (Tensor.sum_to_shape x [| 2; 3 |]) );
         ]

  let reductions =
    let m = t [| 2; 3 |] [| 1.; 5.; 2.; 7.; 0.; 3. |] in
    "reductions"
    >::: [
           ( "sum and mean" >:: fun _ ->
             assert_tensor (Tensor.scalar 21.0) (Tensor.sum (range [| 2; 3 |]));
             assert_tensor (Tensor.scalar 3.5) (Tensor.mean (range [| 2; 3 |]))
           );
           ( "sum_axis 2-D" >:: fun _ ->
             let x = range [| 2; 3 |] in
             assert_tensor (t [| 3 |] [| 5.; 7.; 9. |]) (Tensor.sum_axis x 0);
             assert_tensor (t [| 2 |] [| 6.; 15. |]) (Tensor.sum_axis x 1);
             assert_tensor (t [| 2 |] [| 6.; 15. |]) (Tensor.sum_axis x (-1));
             assert_tensor
               (t [| 2; 1 |] [| 6.; 15. |])
               (Tensor.sum_axis ~keepdims:true x 1) );
           ( "sum_axis 3-D middle axis" >:: fun _ ->
             assert_tensor
               (t [| 2; 2 |] [| 4.; 6.; 12.; 14. |])
               (Tensor.sum_axis (range [| 2; 2; 2 |]) 1) );
           ( "max_axis" >:: fun _ ->
             assert_tensor (t [| 2 |] [| 5.; 7. |]) (Tensor.max_axis m 1);
             assert_tensor (t [| 3 |] [| 7.; 5.; 3. |]) (Tensor.max_axis m 0) );
           ( "argmax_axis" >:: fun _ ->
             let show a =
               String.concat ";" (Array.to_list (Array.map string_of_int a))
             in
             assert_equal ~printer:show [| 1; 0 |] (Tensor.argmax_axis m 1);
             assert_equal ~printer:show [| 1; 0; 1 |] (Tensor.argmax_axis m 0)
           );
         ]

  let linalg =
    "linear algebra and shapes"
    >::: [
           ( "matmul" >:: fun _ ->
             let a = range [| 2; 3 |] in
             let b = t [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
             assert_tensor
               (t [| 2; 2 |] [| 58.; 64.; 139.; 154. |])
               (Tensor.matmul a b) );
           ( "matmul non-square" >:: fun _ ->
             assert_shape [| 4; 5 |]
               (Tensor.matmul (rand_tensor [| 4; 7 |]) (rand_tensor [| 7; 5 |]))
           );
           ( "matmul shape mismatch" >:: fun _ ->
             assert_invalid_arg (fun () ->
                 Tensor.matmul (range [| 2; 3 |]) (range [| 2; 3 |])) );
           ( "transpose" >:: fun _ ->
             assert_tensor
               (t [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |])
               (Tensor.transpose (range [| 2; 3 |])) );
           ( "reshape" >:: fun _ ->
             let r = Tensor.reshape (range [| 2; 3 |]) [| 3; 2 |] in
             assert_tensor (range [| 3; 2 |]) r;
             assert_invalid_arg (fun () ->
                 Tensor.reshape (range [| 2; 3 |]) [| 4 |]) );
           ( "take_rows" >:: fun _ ->
             assert_tensor
               (t [| 2; 2 |] [| 5.; 6.; 1.; 2. |])
               (Tensor.take_rows (range [| 3; 2 |]) [| 2; 0 |]) );
           ( "one_hot" >:: fun _ ->
             assert_tensor
               (t [| 2; 3 |] [| 0.; 0.; 1.; 1.; 0.; 0. |])
               (Tensor.one_hot ~num_classes:3 [| 2; 0 |]) );
         ]

  let in_place =
    "in-place"
    >::: [
           ( "axpy_" >:: fun _ ->
             let y = Tensor.ones [| 3 |] in
             Tensor.axpy_ ~alpha:2.0 ~x:(range [| 3 |]) ~y;
             assert_tensor (t [| 3 |] [| 3.; 5.; 7. |]) y );
           ( "add_ / scale_ / fill_ / blit_" >:: fun _ ->
             let y = range [| 3 |] in
             Tensor.add_ y (Tensor.ones [| 3 |]);
             assert_tensor (t [| 3 |] [| 2.; 3.; 4. |]) y;
             Tensor.scale_ 0.5 y;
             assert_tensor (t [| 3 |] [| 1.; 1.5; 2. |]) y;
             Tensor.fill_ y 9.0;
             assert_tensor (Tensor.full [| 3 |] 9.0) y;
             Tensor.blit_ ~src:(range [| 3 |]) ~dst:y;
             assert_tensor (range [| 3 |]) y );
           ( "in-place shape mismatch" >:: fun _ ->
             assert_invalid_arg (fun () ->
                 Tensor.add_ (Tensor.ones [| 3 |]) (Tensor.ones [| 2 |])) );
         ]

  let suite =
    "tensor"
    >::: [
           construction;
           random;
           elementwise;
           broadcasting;
           reductions;
           linalg;
           in_place;
         ]
end

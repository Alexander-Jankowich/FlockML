(* The NumPy-level tensor API (Phase 7), checked against NumPy itself: the
   expected outputs are .npy files written by test/fixtures/gen.py and
   committed, so running the tests needs no Python. Each case repeats the
   parameters gen.py used. Then edge cases (empty and size-1 axes, negative
   axes, NaN), the .npy reader and writer, and the global generator.
   Run with: dune exec test/test_numpy.exe (default backend) *)

open OUnit2

(** Where the fixtures are: dune copies them next to the test executables, and
    [dune exec] runs from the repository root. *)
let dir =
  lazy
    (match
       List.find_opt Sys.file_exists
         [ "fixtures"; "test/fixtures"; "../test/fixtures" ]
     with
    | Some d -> d
    | None -> failwith "numpy_tests: test/fixtures not found")

let path name = Filename.concat (Lazy.force dir) (name ^ ".npy")

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let fx name = Npy.load (path name)
  let fxi name = snd (Npy.load_ints (path name))

  (** Compare [f ()] with the fixture [name]. *)
  let check ?(tol = 1e-10) ?expected name f =
    name >:: fun _ ->
    let expected = fx (Option.value expected ~default:name) in
    assert_tensor ~rtol:tol ~atol:tol ~msg:name expected (f ())

  let x = lazy (fx "x")
  and pos = lazy (fx "pos")
  and small = lazy (fx "small")
  and x3 = lazy (fx "x3")
  and row = lazy (fx "row")
  and m2 = lazy (fx "m2")
  and v4 = lazy (fx "v4")

  let ( !! ) = Lazy.force

  let elementwise =
    [
      check "abs" (fun () -> Tensor.abs !!x);
      check "sign" (fun () -> Tensor.sign !!x);
      check "square" (fun () -> Tensor.square !!x);
      check "rsqrt" (fun () -> Tensor.rsqrt !!pos);
      check "log1p" (fun () -> Tensor.log1p !!small);
      check "expm1" (fun () -> Tensor.expm1 !!x);
      check "sin" (fun () -> Tensor.sin !!x);
      check "cos" (fun () -> Tensor.cos !!x);
      check "tanh" (fun () -> Tensor.tanh !!x);
      check "sigmoid" (fun () -> Tensor.sigmoid !!x);
      check "erf" (fun () -> Tensor.erf !!x);
      check "clamp" (fun () -> Tensor.clamp ~min:(-1.0) ~max:0.5 !!x);
      check "clamp_min" (fun () -> Tensor.clamp ~min:0.25 !!x);
      check "minimum" (fun () -> Tensor.minimum !!x !!row);
      check "maximum" (fun () -> Tensor.maximum !!x !!row);
    ]
    @ List.map
        (fun (name, f) -> check name (fun () -> f (fx "ia") (fx "ib")))
        [
          ("eq", Tensor.eq);
          ("ne", Tensor.ne);
          ("lt", Tensor.lt);
          ("le", Tensor.le);
          ("gt", Tensor.gt);
          ("ge", Tensor.ge);
        ]
    @ [
        check "where" (fun () ->
            Tensor.where (fx "mask") !!m2
              (Tensor.narrow !!row ~axis:0 ~start:0 ~len:4));
        check "masked_fill" (fun () -> Tensor.masked_fill !!m2 (fx "mask") 9.0);
      ]

  let reductions =
    [
      check "max" (fun () -> Tensor.max !!x3);
      check "min" (fun () -> Tensor.min !!x3);
      check "mean_axis1" (fun () -> Tensor.mean_axis !!x3 1);
      check "min_axis2" (fun () -> Tensor.min_axis !!x3 (-1));
      ( "argmin_axis0" >:: fun _ ->
        assert_equal (fxi "argmin_axis0") (Tensor.argmin_axis !!x3 0) );
      check "prod_axis1" (fun () -> Tensor.prod_axis (fx "near1") 1);
      check "cumsum_axis1" (fun () -> Tensor.cumsum !!x3 1);
      check "logsumexp_axis2" (fun () -> Tensor.logsumexp !!x3 2);
      check "var_axis0" (fun () -> Tensor.var !!x3 0);
      check "var1_axis2" (fun () -> Tensor.var ~correction:1 !!x3 (-1));
      check "std_axis1" (fun () -> Tensor.std !!x3 1);
      check "sum_axes02" (fun () -> Tensor.sum_axes !!x3 [ 2; 0; -1 ]);
      check "mean_axes12_keep" (fun () ->
          Tensor.mean_axes ~keepdims:true !!x3 [ 1; 2 ]);
    ]

  let shapes =
    [
      check "permute201" (fun () -> Tensor.permute !!x3 [| 2; 0; 1 |]);
      check "swapaxes02" (fun () -> Tensor.swapaxes !!x3 0 (-1));
      check "concat1" (fun () ->
          Tensor.concat ~axis:1
            [ fx "c325"; Tensor.slice !!x3 [ All; Range (0, 4) ] ]);
      check "stack1" (fun () ->
          Tensor.stack ~axis:1 [ !!m2; Tensor.scale 2. !!m2 ]);
      ( "split2" >:: fun _ ->
        match Tensor.split !!x3 ~axis:2 [ 2; 3 ] with
        | [ a; b ] ->
            assert_tensor (fx "split2_a") a;
            assert_tensor (fx "split2_b") b
        | _ -> assert_failure "two pieces" );
      check "slice_mixed" (fun () ->
          Tensor.slice !!x3 [ Range (1, 3); Step (0, 4, 2); Index (-1) ]);
      check "slice_clip" (fun () ->
          Tensor.slice !!x3 [ Range (-10, 100); Range (1, 99) ]);
      check "flip02" (fun () -> Tensor.flip !!x3 [ 0; -1 ]);
      check "pad" (fun () -> Tensor.pad ~value:7.0 !!m2 [| (1, 2); (0, 3) |]);
      check "tile" (fun () ->
          Tensor.tile
            (Tensor.slice !!m2 [ Range (0, 2); Range (0, 3) ])
            [| 2; 1; 2 |]);
      check "repeat1" (fun () ->
          Tensor.repeat
            (Tensor.slice !!m2 [ Range (0, 2); Range (0, 3) ])
            ~axis:1 3);
      check "index_select1" (fun () ->
          Tensor.index_select !!x3 ~axis:1 (fxi "idx"));
      check "gather1" (fun () -> Tensor.gather !!m2 ~axis:1 (fxi "gather_idx"));
      check "scatter_add1" (fun () ->
          Tensor.scatter_add !!m2 ~axis:1 (fxi "gather_idx") (fx "scatter_src"));
      check "index_add1" (fun () ->
          Tensor.index_add !!x3 ~axis:1 (fxi "idx") (fx "index_src"));
      check "set_slice" (fun () ->
          let t = Tensor.zeros [| 4; 5 |] in
          Tensor.set_slice_ t [ Step (1, 4, 2); Range (1, 3) ] (fx "set_src");
          t);
      check "set_slice_bcast" (fun () ->
          let t = Tensor.zeros [| 4; 5 |] in
          Tensor.set_slice_ t [ All; Index 2 ] !!v4;
          t);
    ]

  let linear_algebra =
    [
      check "matmul_3d_2d" (fun () -> Tensor.matmul (fx "a234") (fx "b45"));
      check "matmul_bcast" (fun () -> Tensor.matmul (fx "a2134") (fx "b542"));
      check "matmul_1d_2d" (fun () -> Tensor.matmul !!v4 (fx "b45"));
      check "matmul_2d_1d" (fun () ->
          Tensor.matmul (Tensor.transpose (fx "b45")) !!v4);
      check "matmul_1d_1d" (fun () -> Tensor.matmul !!v4 (fx "w4"));
      check "bmm_trans_a" (fun () ->
          Tensor.bmm ~trans_a:true (fx "a243") (fx "b245"));
      check "bmm_trans_b" (fun () ->
          Tensor.bmm ~trans_b:true (fx "a243") (fx "b253"));
      check "dot" (fun () -> Tensor.dot !!v4 (fx "w4"));
      check "outer" (fun () -> Tensor.outer !!v4 (fx "v3"));
      check "norm2" (fun () -> Tensor.norm !!x3);
      check "norm1" (fun () -> Tensor.norm ~p:1. !!x3);
      check "norminf" (fun () -> Tensor.norm ~p:Float.infinity !!x3);
      check "norm3" (fun () -> Tensor.norm ~p:3. !!x3);
    ]

  let construction =
    [
      check "arange_step" (fun () -> Tensor.arange ~start:0.5 ~step:0.3 2.0);
      check "arange5" (fun () -> Tensor.arange 5.0);
      check "arange_down" (fun () -> Tensor.arange ~start:5.0 ~step:(-1.5) 0.0);
      check "linspace" (fun () -> Tensor.linspace (-1.0) 2.0 7);
      check "eye" (fun () -> Tensor.eye 4);
      ( "init, zeros_like, full_like" >:: fun _ ->
        let x =
          Tensor.init [| 2; 3 |] (fun i -> float_of_int ((10 * i.(0)) + i.(1)))
        in
        assert_tensor (t [| 2; 3 |] [| 0.; 1.; 2.; 10.; 11.; 12. |]) x;
        assert_tensor (Tensor.zeros [| 2; 3 |]) (Tensor.zeros_like x);
        assert_tensor (Tensor.full [| 2; 3 |] 1.5) (Tensor.full_like x 1.5);
        assert_tensor (Tensor.ones [| 2; 3 |]) (Tensor.ones_like x) );
      ( "bernoulli and randint" >:: fun _ ->
        let b = Tensor.to_array (Tensor.bernoulli ~rng ~p:0.3 [| 20000 |]) in
        Array.iter (fun v -> assert_bool "0 or 1" (v = 0. || v = 1.)) b;
        assert_float ~eps:0.02 0.3 (Array.fold_left ( +. ) 0. b /. 20000.);
        let r =
          Tensor.to_array (Tensor.randint ~rng ~lo:(-2) ~hi:3 [| 5000 |])
        in
        Array.iter
          (fun v ->
            assert_bool "integer in range"
              (Float.is_integer v && v >= -2. && v < 3.))
          r;
        List.iter
          (fun k ->
            assert_bool "every value drawn" (Array.mem (float_of_int k) r))
          [ -2; -1; 0; 1; 2 ];
        assert_invalid_arg (fun () -> Tensor.bernoulli ~p:1.5 [| 2 |]);
        assert_invalid_arg (fun () -> Tensor.randint ~lo:2 ~hi:2 [| 2 |]) );
    ]

  (** Hand-checked results on awkward shapes, NaN, and the errors. *)
  let edge_cases =
    let empty = Tensor.zeros [| 0; 3 |] in
    let nan = Float.nan in
    let is_nan x = Float.is_nan (Tensor.item x) in
    [
      ( "empty axes" >:: fun _ ->
        assert_shape [| 3; 0 |] (Tensor.permute empty [| 1; 0 |]);
        assert_shape [| 0; 3 |] (Tensor.flip empty [ 0; 1 ]);
        assert_shape [| 2; 3 |] (Tensor.pad empty [| (1, 1); (0, 0) |]);
        assert_shape [| 0; 3 |] (Tensor.slice empty [ Range (2, 5) ]);
        assert_shape [| 4; 3 |]
          (Tensor.concat ~axis:0 [ empty; Tensor.ones [| 4; 3 |] ]);
        assert_shape [| 3 |] (Tensor.min_axis empty 0);
        assert_equal Float.infinity
          (Tensor.item (Tensor.min (Tensor.min_axis empty 0)));
        assert_equal Float.neg_infinity (Tensor.item (Tensor.max empty));
        assert_shape [| 0; 3 |] (Tensor.cumsum empty 0);
        assert_shape [| 0; 2 |] (Tensor.index_select empty ~axis:1 [| 2; 0 |]);
        assert_shape [| 2; 0; 3 |]
          (Tensor.matmul (Tensor.zeros [| 2; 0; 4 |]) (Tensor.ones [| 4; 3 |]));
        assert_tensor
          (Tensor.zeros [| 2; 3 |])
          (Tensor.matmul (Tensor.zeros [| 2; 0 |]) (Tensor.zeros [| 0; 3 |]));
        assert_shape [| 0 |] (Tensor.arange ~start:3. 1.) );
      ( "size-1 axes and scalars" >:: fun _ ->
        let x = range [| 1; 3; 1 |] in
        assert_shape [| 3 |] (Tensor.squeeze x);
        assert_shape [| 3; 1 |] (Tensor.squeeze ~axis:0 x);
        assert_invalid_arg (fun () -> Tensor.squeeze ~axis:1 x);
        assert_shape [| 1; 3; 1; 1 |] (Tensor.unsqueeze x (-1));
        assert_shape [| 3 |] (Tensor.flatten x);
        assert_shape [| 1; 3 |] (Tensor.flatten ~first:1 x);
        assert_shape [| 1 |] (Tensor.flatten (Tensor.scalar 2.));
        assert_tensor (Tensor.scalar 2.) (Tensor.slice (Tensor.scalar 2.) []);
        assert_tensor (Tensor.scalar 6.) (Tensor.sum_axes (range [| 3 |]) [ 0 ]);
        assert_tensor (range [| 1; 3; 1 |]) (Tensor.sum_axes x []);
        assert_tensor (Tensor.scalar 3.) (Tensor.max (range [| 1; 3; 1 |])) );
      ( "odd sizes cross the parallel chunks" >:: fun _ ->
        let x = rand_tensor [| 7; 131; 3 |] in
        let p = Tensor.permute x [| 1; 2; 0 |] in
        assert_tensor x (Tensor.permute p [| 2; 0; 1 |]);
        let parts = Tensor.chunk x ~axis:1 4 in
        assert_equal ~printer:string_of_int 4 (List.length parts);
        assert_shape [| 7; 32; 3 |] (List.nth parts 3);
        assert_tensor x (Tensor.concat ~axis:1 parts);
        assert_tensor x (Tensor.flip (Tensor.flip x [ 1 ]) [ 1 ]);
        assert_tensor x
          (Tensor.slice
             (Tensor.pad x [| (0, 0); (2, 5); (1, 0) |])
             [ All; Range (2, 133); Range (1, 4) ]) );
      ( "chunk like torch.chunk" >:: fun _ ->
        let sizes x n =
          List.map (fun c -> (Tensor.shape c).(0)) (Tensor.chunk x ~axis:0 n)
        in
        assert_equal [ 3; 3; 1 ] (sizes (range [| 7 |]) 3);
        assert_equal [ 1; 1 ] (sizes (range [| 2 |]) 5);
        assert_equal [ 2; 2; 2 ] (sizes (range [| 6 |]) 3) );
      ( "slicing rules" >:: fun _ ->
        let x = range [| 5 |] in
        assert_tensor
          (t [| 2 |] [| 4.; 5. |])
          (Tensor.slice x [ Range (-2, 5) ]);
        assert_tensor (t [| 0 |] [||]) (Tensor.slice x [ Range (3, 1) ]);
        assert_tensor
          (t [| 2 |] [| 2.; 5. |])
          (Tensor.slice x [ Step (1, 5, 3) ]);
        assert_tensor (Tensor.scalar 5.) (Tensor.slice x [ Index (-1) ]);
        assert_invalid_arg (fun () -> Tensor.slice x [ Index 5 ]);
        assert_invalid_arg (fun () -> Tensor.slice x [ Step (0, 5, 0) ]);
        assert_invalid_arg (fun () -> Tensor.slice x [ All; All ]);
        assert_invalid_arg (fun () -> Tensor.narrow x ~axis:0 ~start:3 ~len:3);
        let y = Tensor.copy x in
        Tensor.set_slice_ y [ Range (0, 3) ] (Tensor.slice x [ Range (2, 5) ]);
        assert_tensor (t [| 5 |] [| 3.; 4.; 5.; 4.; 5. |]) y;
        assert_invalid_arg (fun () ->
            Tensor.set_slice_ y [ Range (0, 2) ] (range [| 3 |])) );
      ( "NaN rules" >:: fun _ ->
        let n = Tensor.scalar nan and one = Tensor.scalar 1. in
        assert_bool "minimum (nan, 1)" (is_nan (Tensor.minimum n one));
        assert_bool "minimum (1, nan)" (is_nan (Tensor.minimum one n));
        assert_bool "maximum (1, nan)" (is_nan (Tensor.maximum one n));
        assert_bool "clamp nan" (is_nan (Tensor.clamp ~min:0. ~max:1. n));
        assert_bool "max_axis"
          (is_nan (Tensor.max (t [| 3 |] [| 1.; nan; 2. |])));
        assert_bool "min_axis"
          (is_nan (Tensor.min (t [| 3 |] [| 1.; nan; 2. |])));
        assert_float 0. (Tensor.item (Tensor.eq n n));
        assert_float 1. (Tensor.item (Tensor.ne n n));
        assert_float 0. (Tensor.item (Tensor.lt n one));
        assert_bool "sign" (is_nan (Tensor.sign n));
        assert_float 1. (Tensor.item (Tensor.where n one (Tensor.scalar 0.)));
        assert_tensor
          (t [| 2 |] [| Float.neg_infinity; Float.infinity |])
          (Tensor.logsumexp
             (t [| 2; 2 |]
                [| Float.neg_infinity; Float.neg_infinity; Float.infinity; 0. |])
             1) );
      ( "sigmoid is stable" >:: fun _ ->
        assert_tensor
          (t [| 3 |] [| 0.; 0.5; 1. |])
          (Tensor.sigmoid (t [| 3 |] [| -1000.; 0.; 1000. |])) );
      ( "errors" >:: fun _ ->
        let x = range [| 2; 3 |] in
        assert_invalid_arg (fun () -> Tensor.permute x [| 0; 0 |]);
        assert_invalid_arg (fun () -> Tensor.permute x [| 0 |]);
        assert_invalid_arg (fun () ->
            Tensor.concat ~axis:0 [ x; range [| 2; 2 |] ]);
        assert_invalid_arg (fun () -> Tensor.concat ~axis:0 []);
        assert_invalid_arg (fun () ->
            Tensor.stack ~axis:0 [ x; range [| 3; 2 |] ]);
        assert_invalid_arg (fun () -> Tensor.split x ~axis:1 [ 1; 1 ]);
        assert_invalid_arg (fun () -> Tensor.index_select x ~axis:1 [| 3 |]);
        assert_invalid_arg (fun () -> Tensor.gather x ~axis:1 [| 0; 1; 2 |]);
        assert_invalid_arg (fun () -> Tensor.pad x [| (1, 1) |]);
        assert_invalid_arg (fun () -> Tensor.matmul x x);
        assert_invalid_arg (fun () ->
            Tensor.bmm (range [| 2; 2; 3 |]) (range [| 3; 2; 2 |]));
        assert_invalid_arg (fun () ->
            Tensor.dot (range [| 2 |]) (range [| 3 |]));
        assert_invalid_arg (fun () -> Tensor.masked_fill (range [| 2 |]) x 0.);
        assert_invalid_arg (fun () -> Tensor.arange ~step:0. 1.);
        assert_invalid_arg (fun () -> Tensor.norm ~p:0. x) );
      ( "to_string" >:: fun _ ->
        assert_equal ~printer:Fun.id "tensor [2]\n[3.14, 2.72]"
          (Tensor.to_string ~precision:3 (t [| 2 |] [| 3.14159; 2.71828 |])) );
    ]

  (** The reader on files NumPy wrote in other formats, and round trips. *)
  let npy =
    let tmp name = Filename.concat (Filename.get_temp_dir_name ()) name in
    [
      ( "float32, big-endian and Fortran order" >:: fun _ ->
        assert_tensor ~rtol:1e-6 ~atol:1e-6 !!m2 (fx "f32");
        assert_tensor !!m2 (fx "big_endian");
        assert_tensor !!x3 (fx "fortran");
        assert_tensor (Tensor.scalar 2.5) (fx "scalar") );
      ( "integers and booleans" >:: fun _ ->
        let shape, v = Npy.load_ints (path "int32") in
        assert_equal [| 3; 4 |] shape;
        assert_equal (Array.init 12 (fun i -> i - 3)) v;
        assert_equal
          [| 250; 251; 252; 253; 254; 255 |]
          (snd (Npy.load_ints (path "uint8")));
        assert_equal [| 1; 0; 1 |] (snd (Npy.load_ints (path "bool")));
        assert_tensor (t [| 3 |] [| 1.; 0.; 1. |]) (fx "bool");
        match Npy.load_ints (path "x") with
        | _ -> assert_failure "floats read as integers"
        | exception Failure _ -> () );
      ( "round trips" >:: fun _ ->
        let x = rand_tensor [| 3; 1; 4 |] in
        Npy.save (tmp "flockml_rt.npy") x;
        assert_tensor ~rtol:0. ~atol:0. x (Npy.load (tmp "flockml_rt.npy"));
        Npy.save ~dtype:`Float32 (tmp "flockml_rt32.npy") x;
        assert_tensor ~rtol:1e-6 ~atol:1e-7 x
          (Npy.load (tmp "flockml_rt32.npy"));
        Npy.save (tmp "flockml_rt0.npy") (Tensor.scalar 4.);
        assert_tensor (Tensor.scalar 4.) (Npy.load (tmp "flockml_rt0.npy"));
        Npy.save_ints ~shape:[| 2; 2 |] (tmp "flockml_ri.npy")
          [| 1; -2; 3; max_int |];
        assert_equal
          ([| 2; 2 |], [| 1; -2; 3; max_int |])
          (Npy.load_ints (tmp "flockml_ri.npy"));
        assert_invalid_arg (fun () ->
            Npy.save_ints ~shape:[| 3 |] (tmp "x.npy") [| 1 |]) );
      ( "not a .npy file" >:: fun _ ->
        let p = tmp "flockml_bad.npy" in
        Out_channel.with_open_bin p (fun oc -> output_string oc "hello");
        match Npy.load p with
        | _ -> assert_failure "expected Failure"
        | exception Failure _ -> () );
    ]

  let rng_tests =
    [
      ( "manual_seed makes the default generator reproducible" >:: fun _ ->
        let draw () =
          Flockml.Rng.manual_seed 5;
          ( Tensor.to_array (Tensor.randn [| 4 |]),
            Tensor.to_array (Tensor.rand_uniform ~lo:0. ~hi:1. [| 3 |]),
            Flockml.Rng.permutation 6 )
        in
        assert_equal (draw ()) (draw ()) );
      ( "permutation" >:: fun _ ->
        let p = Flockml.Rng.permutation ~rng 50 in
        assert_equal (Array.init 50 Fun.id)
          (Array.of_list (List.sort compare (Array.to_list p))) );
    ]

  let suite =
    "numpy"
    >::: [
           "elementwise" >::: elementwise;
           "reductions" >::: reductions;
           "shapes and indexing" >::: shapes;
           "linear algebra" >::: linear_algebra;
           "construction" >::: construction;
           "edge cases" >::: edge_cases;
           "npy files" >::: npy;
           "random" >::: rng_tests;
         ]
end

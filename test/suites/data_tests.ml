(* Phase 11: the data pipeline.
   - Sources (in memory, tensor targets, lazy) and the loader: coverage,
     drop_last, and prefetching on the background domain giving exactly the
     batches of the plain loop, transforms included.
   - The background domain itself: order, exceptions, nesting.
   - Transforms, the CIFAR-10 and CSV readers, and the new metrics.
   Run with: dune exec test/test_data.exe (default backend) *)

open OUnit2

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let identical ~msg a b = assert_tensor ~rtol:0. ~atol:0. ~msg a b

  (** A dataset of [n] examples whose input row [i] is all [i], so a batch shows
      which examples it holds. *)
  let numbered n =
    {
      Dataset.inputs = Tensor.init [| n; 2 |] (fun i -> float_of_int i.(0));
      labels = Array.init n Fun.id;
    }

  let all_batches seq = List.of_seq seq

  let loader_tests =
    [
      ( "labelled: get and batch agree" >:: fun _ ->
        let src = Dataset.labelled (numbered 5) in
        let x, l = src.get 3 in
        identical ~msg:"row" (Tensor.full [| 2 |] 3.) x;
        assert_equal 3 l;
        let xb, lb = src.batch [| 4; 1 |] in
        identical ~msg:"batch" (t [| 2; 2 |] [| 4.; 4.; 1.; 1. |]) xb;
        assert_equal [| 4; 1 |] lb );
      ( "of_tensors, of_fn and subset" >:: fun _ ->
        let x = range [| 4; 3 |] and y = range [| 4; 2 |] in
        let src = Dataset.of_tensors x y in
        let xb, yb = src.batch [| 2 |] in
        identical ~msg:"x" (Tensor.take_rows x [| 2 |]) xb;
        identical ~msg:"y" (Tensor.take_rows y [| 2 |]) yb;
        assert_invalid_arg (fun () -> Dataset.of_tensors x (range [| 3; 2 |]));
        let lazy_src =
          Dataset.of_fn ~length:4 ~collate:Dataset.collate_pairs (fun i ->
              ( Tensor.full [| 3 |] (float_of_int i),
                Tensor.scalar (float_of_int (10 * i)) ))
        in
        let xs, ys = lazy_src.batch [| 1; 3 |] in
        identical ~msg:"lazy x" (t [| 2; 3 |] [| 1.; 1.; 1.; 3.; 3.; 3. |]) xs;
        identical ~msg:"lazy y" (t [| 2 |] [| 10.; 30. |]) ys;
        let sub = Dataset.subset lazy_src [| 3; 0 |] in
        assert_equal 2 sub.length;
        identical ~msg:"subset" (t [| 1 |] [| 30. |]) (snd (sub.batch [| 0 |]));
        assert_invalid_arg (fun () -> Dataset.subset lazy_src [| 4 |]) );
      ( "an epoch covers every example once" >:: fun _ ->
        List.iter
          (fun prefetch ->
            let labels =
              Dataset.loader ~shuffle:true ~rng ~prefetch ~batch_size:3
                (Dataset.labelled (numbered 10))
              |> Seq.map snd |> List.of_seq
            in
            assert_equal [ 3; 3; 3; 1 ] (List.map Array.length labels);
            let all = Array.concat labels in
            Array.sort compare all;
            assert_equal (Array.init 10 Fun.id) all)
          [ false; true ] );
      ( "drop_last" >:: fun _ ->
        let n prefetch =
          List.length
            (all_batches
               (Dataset.loader ~drop_last:true ~prefetch ~batch_size:3
                  (Dataset.labelled (numbered 10))))
        in
        assert_equal 3 (n false);
        assert_equal 3 (n true);
        assert_equal 2
          (List.length
             (all_batches
                (Dataset.batches ~drop_last:true ~batch_size:4 (numbered 10))));
        assert_invalid_arg (fun () ->
            Dataset.loader ~batch_size:0 (Dataset.labelled (numbered 3))) );
      ( "prefetching gives the same batches, transforms included" >:: fun _ ->
        let noise r (x, l) =
          ( Tensor.add x
              (Tensor.rand_uniform ~rng:r ~lo:0. ~hi:1. (Tensor.shape x)),
            l )
        in
        let run prefetch =
          Dataset.loader ~shuffle:true ~prefetch
            ~rng:(Random.State.make [| 5 |])
            ~transform:noise ~batch_size:4
            (Dataset.labelled (numbered 13))
          |> all_batches
        in
        List.iter2
          (fun (x, l) (x', l') ->
            identical ~msg:"inputs" x x';
            assert_equal l l')
          (run false) (run true) );
      ( "an exception while loading reaches the consumer" >:: fun _ ->
        let bad =
          Dataset.of_fn ~length:6 ~collate:Array.to_list (fun i ->
              if i = 4 then failwith "broken example" else i)
        in
        assert_raises (Failure "broken example") (fun () ->
            List.iter ignore (all_batches (Dataset.loader ~batch_size:2 bad)))
      );
      ( "abandoning a loader early doesn't block the next one" >:: fun _ ->
        let seq =
          Dataset.loader ~batch_size:1 (Dataset.labelled (numbered 50))
        in
        (match Seq.uncons seq with
        | Some _ -> ()
        | None -> assert_failure "empty");
        assert_equal 5
          (List.length
             (all_batches
                (Dataset.loader ~batch_size:2 (Dataset.labelled (numbered 10)))))
      );
      ( "the background domain: order, nesting, exceptions" >:: fun _ ->
        let log = ref [] in
        let jobs =
          List.init 5 (fun i ->
              Flockml.Background.spawn (fun () ->
                  log := i :: !log;
                  i * i))
        in
        assert_equal [ 0; 1; 4; 9; 16 ] (List.map Flockml.Background.await jobs);
        assert_equal [ 4; 3; 2; 1; 0 ] !log;
        let nested =
          Flockml.Background.spawn (fun () ->
              Flockml.Background.await (Flockml.Background.spawn (fun () -> 7))
              + 1)
        in
        assert_equal 8 (Flockml.Background.await nested);
        assert_raises Exit (fun () ->
            Flockml.Background.await
              (Flockml.Background.spawn (fun () -> raise Exit))) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Transforms                                                               *)
  (* ------------------------------------------------------------------------ *)

  let images = lazy (rand_tensor [| 6; 5; 4; 3 |])

  let transform_tests =
    let ( !! ) = Lazy.force in
    [
      ( "normalize, and in place" >:: fun _ ->
        let x = !!images in
        let mean = [| 0.1; 0.2; 0.3 |] and std = [| 2.; 4.; 0.5 |] in
        let expected =
          Tensor.div (Tensor.sub x (t [| 3 |] mean)) (t [| 3 |] std)
        in
        assert_tensor expected (Transform.normalize ~mean ~std rng x);
        let y = Tensor.copy x in
        Transform.normalize_ ~mean ~std y;
        assert_tensor expected y;
        assert_invalid_arg (fun () ->
            Transform.normalize ~mean:[| 0. |] ~std rng x) );
      ( "random_crop cuts windows of the padded images" >:: fun _ ->
        let x = !!images in
        identical ~msg:"no padding, full size" x
          (Transform.random_crop (5, 4) rng x);
        let y = Transform.random_crop ~padding:2 (5, 4) rng x in
        assert_shape [| 6; 5; 4; 3 |] y;
        let padded = Tensor.pad x [| (0, 0); (2, 2); (2, 2); (0, 0) |] in
        for i = 0 to 5 do
          let img = Tensor.slice y [ Index i; All; All; All ] in
          let found = ref false in
          for dy = 0 to 4 do
            for dx = 0 to 4 do
              if
                Tensor.allclose ~rtol:0. ~atol:0. img
                  (Tensor.slice padded
                     [ Index i; Range (dy, dy + 5); Range (dx, dx + 4); All ])
              then found := true
            done
          done;
          assert_bool (Printf.sprintf "image %d is a window" i) !found
        done;
        assert_invalid_arg (fun () -> Transform.random_crop (6, 4) rng x) );
      ( "flips" >:: fun _ ->
        let x = !!images in
        identical ~msg:"p = 1" (Tensor.flip x [ 2 ])
          (Transform.random_hflip ~p:1. () rng x);
        identical ~msg:"p = 0" x (Transform.random_hflip ~p:0. () rng x);
        identical ~msg:"vertical" (Tensor.flip x [ 1 ])
          (Transform.random_vflip ~p:1. () rng x);
        let y = Transform.random_hflip () (Random.State.make [| 3 |]) x in
        for i = 0 to 5 do
          let img = Tensor.slice y [ Index i; All; All; All ]
          and orig = Tensor.slice x [ Index i; All; All; All ] in
          assert_bool "flipped or not"
            (Tensor.allclose img orig
            || Tensor.allclose img (Tensor.flip orig [ 1 ]))
        done );
      ( "compose and on_inputs, reproducibly" >:: fun _ ->
        let aug =
          Transform.compose
            [
              Transform.random_crop ~padding:1 (5, 4); Transform.random_hflip ();
            ]
        in
        let run seed =
          fst
            (Transform.on_inputs aug
               (Random.State.make [| seed |])
               (!!images, [| 0 |]))
        in
        identical ~msg:"same seed" (run 1) (run 1);
        assert_bool "another seed differs"
          (not (Tensor.allclose (run 1) (run 2))) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Readers                                                                  *)
  (* ------------------------------------------------------------------------ *)

  let with_dir f =
    let dir = Filename.temp_file "flockml_cifar" "" in
    Sys.remove dir;
    Sys.mkdir dir 0o755;
    Fun.protect
      ~finally:(fun () ->
        Array.iter
          (fun f -> Sys.remove (Filename.concat dir f))
          (Sys.readdir dir);
        Sys.rmdir dir)
      (fun () -> f dir)

  let reader_tests =
    [
      ( "CIFAR-10 records become NHWC images" >:: fun _ ->
        with_dir (fun dir ->
            (* two records: label, then red, green and blue planes *)
            let record label base =
              String.make 1 (Char.chr label)
              ^ String.init 3072 (fun k ->
                  Char.chr ((base + (k / 1024 * 50) + (k mod 1024)) mod 256))
            in
            Out_channel.with_open_bin (Filename.concat dir "test_batch.bin")
              (fun oc -> output_string oc (record 7 0 ^ record 2 10));
            let src = Cifar10.source ~dir Cifar10.Test in
            assert_equal 2 src.length;
            let x, labels = src.batch [| 1; 0 |] in
            assert_equal [| 2; 7 |] labels;
            assert_shape [| 2; 32; 32; 3 |] x;
            (* image 1 (row 0 of the batch), pixel (y=1, x=2) = plane offset 34 *)
            List.iter
              (fun c ->
                assert_float
                  (float_of_int ((10 + (c * 50) + 34) mod 256) /. 255.)
                  (Tensor.get x [| 0; 1; 2; c |]))
              [ 0; 1; 2 ];
            let d = Cifar10.load ~limit:1 ~dir Cifar10.Test in
            assert_equal [| 7 |] d.labels;
            Out_channel.with_open_bin (Filename.concat dir "test_batch.bin")
              (fun oc -> output_string oc "short");
            assert_raises
              (Failure
                 (Printf.sprintf "Cifar10: %s is not a whole number of records"
                    (Filename.concat dir "test_batch.bin")))
              (fun () -> Cifar10.source ~dir Cifar10.Test);
            match Cifar10.source ~dir Cifar10.Train with
            | _ -> assert_failure "expected Failure"
            | exception Failure _ -> ()) );
      ( "CSV with a header, quotes and semicolons" >:: fun _ ->
        let names, data =
          Csv.of_string "\"a\";\"b c\";d\n1;2.5;-3\n\n4;5e-1; 6 \n"
        in
        assert_equal [| "a"; "b c"; "d" |] names;
        identical ~msg:"data"
          (t [| 2; 3 |] [| 1.; 2.5; -3.; 4.; 0.5; 6. |])
          data;
        identical ~msg:"columns"
          (t [| 2; 2 |] [| -3.; 1.; 6.; 4. |])
          (Csv.columns (names, data) [ "d"; "a" ]);
        assert_invalid_arg (fun () -> Csv.columns (names, data) [ "zzz" ]) );
      ( "CSV without a header, and errors" >:: fun _ ->
        let names, data = Csv.of_string "1,2\n3,4\n" in
        assert_equal [| "0"; "1" |] names;
        identical ~msg:"data" (t [| 2; 2 |] [| 1.; 2.; 3.; 4. |]) data;
        let _, d = Csv.of_string ~header:true "x\ty\n1\t2\n" in
        identical ~msg:"tabs" (t [| 1; 2 |] [| 1.; 2. |]) d;
        List.iter
          (fun bad ->
            match Csv.of_string bad with
            | _ -> assert_failure ("expected Failure for " ^ bad)
            | exception Failure _ -> ())
          [ "a,b\n1,2\n3\n"; "a,b\n1,x\n" ] );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Metrics                                                                  *)
  (* ------------------------------------------------------------------------ *)

  let metric_tests =
    [
      ( "top_k and top_k_accuracy" >:: fun _ ->
        let scores =
          t [| 3; 4 |] [| 0.1; 0.5; 0.2; 0.2; 3.; 1.; 2.; 0.; 1.; 1.; 1.; 1. |]
        in
        assert_equal
          [| [| 1; 2 |]; [| 0; 2 |]; [| 0; 1 |] |]
          (Metrics.top_k ~k:2 scores);
        assert_float (2. /. 3.)
          (Metrics.top_k_accuracy ~k:2 ~scores ~labels:[| 2; 2; 3 |]);
        assert_float 1.
          (Metrics.top_k_accuracy ~k:4 ~scores ~labels:[| 2; 2; 3 |]);
        assert_invalid_arg (fun () -> Metrics.top_k ~k:5 scores) );
      ( "classification_report, as sklearn's" >:: fun _ ->
        (* sklearn.metrics.classification_report([0,1,2,2,2,1], [0,2,2,2,1,0]) *)
        let r =
          Metrics.classification_report ~num_classes:3
            ~predictions:[| 0; 2; 2; 2; 1; 0 |] ~labels:[| 0; 1; 2; 2; 2; 1 |]
        in
        let close = Array.iter2 (fun e a -> assert_float ~eps:1e-12 e a) in
        close [| 0.5; 0.; 2. /. 3. |] r.precision;
        close [| 1.; 0.; 2. /. 3. |] r.recall;
        close [| 2. /. 3.; 0.; 2. /. 3. |] r.f1;
        assert_equal [| 1; 2; 3 |] r.support;
        assert_float 0.5 r.accuracy;
        assert_float (4. /. 9.) r.macro_f1;
        assert_float
          (((2. /. 3. *. 1.) +. (2. /. 3. *. 3.)) /. 6.)
          r.weighted_f1;
        ignore
          (Format.asprintf "%a"
             (Metrics.pp_report ~names:[| "a"; "b"; "c" |])
             r) );
      ( "running mean" >:: fun _ ->
        let m = Metrics.Running.create () in
        assert_bool "empty is nan" (Float.is_nan (Metrics.Running.mean m));
        Metrics.Running.add m 1.;
        Metrics.Running.add ~weight:3. m 5.;
        assert_float 4. (Metrics.Running.mean m);
        assert_equal 2 (Metrics.Running.count m);
        Metrics.Running.reset m;
        assert_equal 0 (Metrics.Running.count m) );
    ]

  let suite =
    "data"
    >::: [
           "loader" >::: loader_tests;
           "transforms" >::: transform_tests;
           "readers" >::: reader_tests;
           "metrics" >::: metric_tests;
         ]
end

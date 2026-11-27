(* Phase 9: layers, model state, checkpoints and initialisation.
   - Gradient checks of every new layer, with respect to its input and its
     parameters. (Their forward values are checked against PyTorch in
     torch_tests.ml.)
   - Named parameters and buffers, custom and residual layers, summaries.
   - state_dict, safetensors files and checkpoints, including an interrupted
     training run that resumes exactly.
   - Two models written with the public API alone: a ResNet-style CNN and a
     small causal transformer, which train.
   Run with: dune exec test/test_layers.exe (default backend) *)

open OUnit2

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let r ?lo ?hi shape = rand_tensor ?lo ?hi shape

  (** A gradient check of [layer] with respect to its input and to every
      parameter. [build] makes the layer from the parameter nodes. *)
  let layer_check ?eps ?tol ?(train = false) name ~input ~params build =
    gradcheck ?eps ?tol name (input :: params) (function
      | x :: ps -> Layer.forward ~train (build ps) x
      | [] -> assert false)

  let one = function [ a ] -> a | _ -> assert false
  let two = function [ a; b ] -> (a, b) | _ -> assert false

  (* ------------------------------------------------------------------------ *)
  (* Gradient checks                                                          *)
  (* ------------------------------------------------------------------------ *)

  let conv kernel stride padding dilation groups (w, b) =
    Layer.Conv2d { w; b = Some b; kernel; stride; padding; dilation; groups }

  let conv_checks =
    [
      layer_check "conv2d rectangular, two strides"
        ~input:(r [| 2; 5; 6; 2 |])
        ~params:[ r [| 12; 3 |]; r [| 3 |] ]
        (fun ps -> conv (2, 3) (2, 1) (1, 0) (1, 1) 1 (two ps));
      layer_check "conv2d dilated"
        ~input:(r [| 1; 6; 6; 2 |])
        ~params:[ r [| 18; 2 |]; r [| 2 |] ]
        (fun ps -> conv (3, 3) (1, 1) (2, 2) (2, 2) 1 (two ps));
      layer_check "conv2d grouped"
        ~input:(r [| 2; 4; 4; 4 |])
        ~params:[ r [| 18; 6 |]; r [| 6 |] ]
        (fun ps -> conv (3, 3) (1, 1) (1, 1) (1, 1) 2 (two ps));
      layer_check "conv2d depthwise"
        ~input:(r [| 1; 4; 4; 3 |])
        ~params:[ r [| 9; 3 |]; r [| 3 |] ]
        (fun ps -> conv (3, 3) (2, 2) (1, 1) (1, 1) 3 (two ps));
      layer_check "conv1d"
        ~input:(r [| 2; 7; 2 |])
        ~params:[ r [| 6; 3 |]; r [| 3 |] ]
        (fun ps ->
          let w, b = two ps in
          Layer.Conv1d
            {
              w;
              b = Some b;
              kernel = 3;
              stride = 2;
              padding = 1;
              dilation = 1;
              groups = 1;
            });
      layer_check "conv_transpose2d"
        ~input:(r [| 2; 3; 2; 2 |])
        ~params:[ r [| 2; 27 |]; r [| 3 |] ]
        (fun ps ->
          let w, b = two ps in
          Layer.Conv_transpose2d
            {
              w;
              b = Some b;
              kernel = (3, 3);
              stride = 2;
              padding = 1;
              output_padding = 1;
            });
      layer_check "linear without bias, 3-D input"
        ~input:(r [| 2; 3; 4 |])
        ~params:[ r [| 4; 2 |] ]
        (fun ps -> Layer.Linear { w = one ps; b = None });
      layer_check "max_pool2d padded"
        ~input:(r [| 2; 5; 5; 2 |])
        ~params:[]
        (fun _ -> Layer.max_pool2d ~stride:2 ~padding:1 3);
      layer_check "avg_pool2d padded"
        ~input:(r [| 2; 5; 5; 2 |])
        ~params:[]
        (fun _ -> Layer.avg_pool2d ~stride:2 ~padding:1 3);
      layer_check "global_avg_pool"
        ~input:(r [| 2; 3; 4; 2 |])
        ~params:[]
        (fun _ -> Layer.global_avg_pool);
    ]

  let activation_checks =
    List.map
      (fun (name, layer) ->
        layer_check name
          ~input:(r ~lo:(-3.) ~hi:3. [| 3; 5 |])
          ~params:[]
          (fun _ -> layer))
      [
        ("leaky_relu", Layer.leaky_relu ~slope:0.2 ());
        ("elu", Layer.elu ~alpha:0.5 ());
        ("gelu", Layer.gelu ());
        ("gelu tanh", Layer.gelu ~approximate:true ());
        ("silu", Layer.silu);
        ("softplus", Layer.softplus ~beta:2. ~threshold:3. ());
        ("softmax", Layer.softmax);
        ("log_softmax", Layer.log_softmax);
        ("identity", Layer.identity);
      ]
    @ [
        ( "elu doesn't overflow" >:: fun _ ->
          let x = Var.param (t [| 2 |] [| 1000.; -1000. |]) in
          let y = Var.elu x in
          Var.backward (Var.sum y);
          assert_tensor (t [| 2 |] [| 1000.; -1. |]) (Var.value y);
          assert_tensor (t [| 2 |] [| 1.; 0. |]) (Var.grad_exn x) );
        ( "softplus is linear past the threshold" >:: fun _ ->
          let y = Var.softplus (Var.const (t [| 2 |] [| 30.; -30. |])) in
          assert_tensor ~atol:1e-12
            (t [| 2 |] [| 30.; Float.exp (-30.) |])
            (Var.value y) );
      ]

  let norm_checks =
    let bn train name input =
      layer_check ~train name ~input
        ~params:[ r [| 3 |]; r [| 3 |] ]
        (fun ps ->
          let w, b = two ps in
          match Layer.batch_norm 3 with
          | Layer.Batch_norm r -> Layer.Batch_norm { r with w; b }
          | _ -> assert false)
    in
    [
      bn true "batch_norm 1-D, training" (r [| 5; 3 |]);
      bn true "batch_norm 2-D, training" (r [| 2; 2; 3; 3 |]);
      bn false "batch_norm, evaluation" (r [| 2; 2; 3; 3 |]);
      layer_check "layer_norm"
        ~input:(r [| 2; 3; 4 |])
        ~params:[ r [| 3; 4 |]; r [| 3; 4 |] ]
        (fun ps ->
          let w, b = two ps in
          Layer.Layer_norm { w; b; shape = [| 3; 4 |]; eps = 1e-5 });
      layer_check "group_norm"
        ~input:(r [| 2; 2; 2; 4 |])
        ~params:[ r [| 4 |]; r [| 4 |] ]
        (fun ps ->
          let w, b = two ps in
          Layer.Group_norm { w; b; groups = 2; eps = 1e-5 });
      gradcheck "embedding, repeated indices"
        [ r [| 5; 3 |] ]
        (fun ps ->
          Layer.forward
            (Layer.Embedding { w = one ps })
            (Var.const (t [| 2; 3 |] [| 1.; 4.; 1.; 0.; 4.; 4. |])));
    ]

  let sequence_checks =
    let rec_check name kind gates =
      layer_check name
        ~input:(r [| 2; 3; 2 |])
        ~params:
          [
            r [| 2; gates * 3 |];
            r [| 3; gates * 3 |];
            r [| gates * 3 |];
            r [| gates * 3 |];
          ]
        (function
          | [ w_ih; w_hh; b_ih; b_hh ] ->
              Layer.Recurrent { kind; w_ih; w_hh; b_ih; b_hh; hidden = 3 }
          | _ -> assert false)
    in
    let attn causal =
      layer_check
        (if causal then "attention, causal" else "attention")
        ~input:(r [| 2; 3; 4 |])
        ~params:[ r [| 4; 12 |]; r [| 12 |]; r [| 4; 4 |]; r [| 4 |] ]
        (function
          | [ in_w; in_b; out_w; out_b ] ->
              Layer.Attention { in_w; in_b; out_w; out_b; heads = 2; causal }
          | _ -> assert false)
    in
    [
      rec_check "rnn" Layer.Rnn_tanh 1;
      rec_check "rnn relu" Layer.Rnn_relu 1;
      rec_check "lstm" Layer.Lstm 4;
      rec_check "gru" Layer.Gru 3;
      attn false;
      attn true;
      ( "run continues from a state" >:: fun _ ->
        let lstm = Layer.lstm ~rng ~input:2 ~hidden:3 () in
        let x = Var.const (r [| 2; 5; 2 |]) in
        let whole, last = Layer.run lstm x in
        let first, mid =
          Layer.run lstm (Var.narrow x ~axis:1 ~start:0 ~len:2)
        in
        let rest, last' =
          Layer.run ~state:mid lstm (Var.narrow x ~axis:1 ~start:2 ~len:3)
        in
        assert_tensor ~rtol:0. ~atol:1e-14 (Var.value whole)
          (Var.value (Var.concat ~axis:1 [ first; rest ]));
        assert_tensor ~rtol:0. ~atol:1e-14 (Var.value last.h)
          (Var.value last'.h) );
      ( "step = run of one step" >:: fun _ ->
        let gru = Layer.gru ~rng ~input:2 ~hidden:3 () in
        let x = r [| 2; 1; 2 |] in
        let st =
          Layer.step gru
            ~x:(Var.const (Tensor.reshape x [| 2; 2 |]))
            (Layer.init_state gru ~batch:2)
        in
        let out, _ = Layer.run gru (Var.const x) in
        assert_tensor ~atol:1e-14
          (Tensor.reshape (Var.value out) [| 2; 3 |])
          (Var.value st.h) );
      ( "causal attention ignores the future" >:: fun _ ->
        let a =
          Layer.multi_head_attention ~rng ~causal:true ~dim:4 ~heads:2 ()
        in
        let x = r [| 1; 4; 4 |] in
        let x' = Tensor.copy x in
        Tensor.set x' [| 0; 3; 0 |] 5.;
        let y = Var.value (Layer.forward a (Var.const x))
        and y' = Var.value (Layer.forward a (Var.const x')) in
        let first3 v = Tensor.narrow v ~axis:1 ~start:0 ~len:3 in
        assert_tensor ~atol:1e-14 (first3 y) (first3 y');
        assert_bool "the last position changes"
          (not
             (Tensor.allclose
                (Tensor.narrow y ~axis:1 ~start:3 ~len:1)
                (Tensor.narrow y' ~axis:1 ~start:3 ~len:1))) );
      ( "a padding mask hides keys" >:: fun _ ->
        let a = Layer.multi_head_attention ~rng ~dim:4 ~heads:1 () in
        let x = r [| 1; 3; 4 |] in
        let x' = Tensor.copy x in
        Tensor.set x' [| 0; 2; 1 |] 9.;
        let mask = t [| 1; 1; 1; 3 |] [| 0.; 0.; 1. |] in
        let run x =
          let x = Var.const x in
          Var.value (Layer.attention ~mask a ~query:x ~key:x ~value:x)
        in
        let first2 v = Tensor.narrow v ~axis:1 ~start:0 ~len:2 in
        assert_tensor ~atol:1e-14 (first2 (run x)) (first2 (run x')) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Model structure                                                          *)
  (* ------------------------------------------------------------------------ *)

  (** A residual block with a projection shortcut, as a custom layer. *)
  let block ~in_c ~out_c ~stride =
    let conv1 =
      Layer.conv2d ~rng ~bias:false ~stride ~padding:1 ~in_channels:in_c
        ~out_channels:out_c ~kernel:3 ()
    and bn1 = Layer.batch_norm out_c
    and conv2 =
      Layer.conv2d ~rng ~bias:false ~padding:1 ~in_channels:out_c
        ~out_channels:out_c ~kernel:3 ()
    and bn2 = Layer.batch_norm out_c in
    let shortcut =
      if stride = 1 && in_c = out_c then Layer.identity
      else
        Layer.sequential
          [
            Layer.conv2d ~rng ~bias:false ~stride ~in_channels:in_c
              ~out_channels:out_c ~kernel:1 ();
            Layer.batch_norm out_c;
          ]
    in
    Layer.custom ~name:"BasicBlock"
      ~children:
        [
          ("conv1", conv1);
          ("bn1", bn1);
          ("conv2", conv2);
          ("bn2", bn2);
          ("shortcut", shortcut);
        ]
      (fun ~train ?rng x ->
        let f l x = Layer.forward ~train ?rng l x in
        let y = f bn2 (f conv2 (Var.relu (f bn1 (f conv1 x)))) in
        Var.relu (Var.add y (f shortcut x)))

  let resnet () =
    Layer.sequential
      [
        Layer.conv2d ~rng ~bias:false ~padding:1 ~in_channels:3 ~out_channels:8
          ~kernel:3 ();
        Layer.batch_norm 8;
        Layer.relu;
        block ~in_c:8 ~out_c:8 ~stride:1;
        block ~in_c:8 ~out_c:16 ~stride:2;
        Layer.global_avg_pool;
        Layer.linear ~rng ~init:Init.Kaiming_uniform ~in_dim:16 ~out_dim:10 ();
      ]

  (** A one-block causal transformer over tokens, as a custom layer: token and
      position embeddings, pre-norm attention and MLP, each residual. *)
  let gpt ~vocab ~dim ~len =
    let tok = Layer.embedding ~rng ~num:vocab ~dim ()
    and pos = Layer.embedding ~rng ~num:len ~dim ()
    and attn =
      Layer.residual
        (Layer.sequential
           [
             Layer.layer_norm [| dim |];
             Layer.multi_head_attention ~rng ~causal:true ~dim ~heads:2 ();
           ])
    and mlp =
      Layer.residual
        (Layer.sequential
           [
             Layer.layer_norm [| dim |];
             Layer.linear ~rng ~in_dim:dim ~out_dim:(4 * dim) ();
             Layer.gelu ();
             Layer.linear ~rng ~in_dim:(4 * dim) ~out_dim:dim ();
           ])
    and norm = Layer.layer_norm [| dim |]
    and head = Layer.linear ~rng ~bias:false ~in_dim:dim ~out_dim:vocab () in
    Layer.custom ~name:"GPT"
      ~children:
        [
          ("tok", tok);
          ("pos", pos);
          ("attn", attn);
          ("mlp", mlp);
          ("norm", norm);
          ("head", head);
        ]
      (fun ~train ?rng x ->
        let s = Var.shape x in
        let positions = Array.init (s.(0) * s.(1)) (fun i -> i mod s.(1)) in
        let h =
          Var.add (Layer.forward tok x)
            (Layer.embed pos ~shape:[| s.(0); s.(1) |] positions)
        in
        let f l x = Layer.forward ~train ?rng l x in
        f head (f norm (f mlp (f attn h))))

  let structure =
    [
      ( "named_params and named_buffers" >:: fun _ ->
        let model =
          Layer.sequential [ block ~in_c:2 ~out_c:4 ~stride:2; Layer.relu ]
        in
        assert_equal ~printer:(String.concat " ")
          [
            "0.conv1.w";
            "0.bn1.w";
            "0.bn1.b";
            "0.conv2.w";
            "0.bn2.w";
            "0.bn2.b";
            "0.shortcut.0.w";
            "0.shortcut.1.w";
            "0.shortcut.1.b";
          ]
          (List.map fst (Layer.named_params model));
        assert_equal ~printer:(String.concat " ")
          [
            "0.bn1.running_mean";
            "0.bn1.running_var";
            "0.bn1.num_batches_tracked";
            "0.bn2.running_mean";
            "0.bn2.running_var";
            "0.bn2.num_batches_tracked";
            "0.shortcut.1.running_mean";
            "0.shortcut.1.running_var";
            "0.shortcut.1.num_batches_tracked";
          ]
          (List.map fst (Layer.named_buffers model));
        assert_equal
          (List.map snd (Layer.named_params model))
          (Layer.params model) );
      ( "custom params and residual paths" >:: fun _ ->
        let scale = Var.param (Tensor.ones [| 3 |]) in
        let c =
          Layer.custom ~name:"Scale"
            ~params:[ ("s", scale) ]
            ~children:
              [
                ( "inner",
                  Layer.residual (Layer.linear ~rng ~in_dim:3 ~out_dim:3 ()) );
              ]
            (fun ~train:_ ?rng:_ x -> Var.mul x scale)
        in
        assert_equal ~printer:(String.concat " ")
          [ "s"; "inner.w"; "inner.b" ]
          (List.map fst (Layer.named_params c));
        assert_equal ~printer:string_of_int 15 (Layer.num_params c);
        assert_invalid_arg (fun () ->
            Layer.custom ~name:"Dup"
              ~params:[ ("a", scale) ]
              ~buffers:[ ("a", scale) ]
              (fun ~train:_ ?rng:_ x -> x)) );
      ( "residual adds its input" >:: fun _ ->
        let x = r [| 2; 3 |] in
        let y = Layer.forward (Layer.residual Layer.relu) (Var.const x) in
        assert_tensor (Tensor.add x (Tensor.relu x)) (Var.value y) );
      ( "batch norm buffers aren't parameters" >:: fun _ ->
        let bn = Layer.batch_norm 3 in
        assert_equal 2 (List.length (Layer.params bn));
        let x = r [| 4; 3 |] in
        let before =
          Tensor.copy
            (Var.value (List.assoc "running_mean" (Layer.named_buffers bn)))
        in
        ignore (Layer.forward bn (Var.const x));
        assert_tensor before
          (Var.value (List.assoc "running_mean" (Layer.named_buffers bn)));
        ignore (Layer.forward ~train:true bn (Var.const x));
        assert_tensor
          (Tensor.scale 0.1 (Tensor.mean_axis x 0))
          (Var.value (List.assoc "running_mean" (Layer.named_buffers bn))) );
      ( "to_string and summary" >:: fun _ ->
        let model = resnet () in
        let s = Layer.to_string model in
        let contains sub s =
          let n = String.length sub in
          let rec go i =
            i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
          in
          go 0
        in
        assert_bool s
          (contains "BasicBlock(" s && contains "conv1: Conv2d(8 -> 8" s);
        let sum = Layer.summary model ~input_shape:[| 2; 8; 8; 3 |] in
        assert_bool sum (contains "[2; 10]" sum && contains "[2; 4; 4; 16]" sum);
        assert_bool sum
          (contains
             (Printf.sprintf "%d trainable parameters" (Layer.num_params model))
             sum) );
      ( "constructor checks" >:: fun _ ->
        assert_invalid_arg (fun () ->
            Layer.conv2d ~groups:3 ~in_channels:4 ~out_channels:6 ~kernel:3 ());
        assert_invalid_arg (fun () ->
            Layer.multi_head_attention ~dim:6 ~heads:4 ());
        assert_invalid_arg (fun () -> Layer.group_norm ~groups:3 4);
        assert_invalid_arg (fun () ->
            Layer.conv_transpose2d ~stride:2 ~output_padding:2 ~in_channels:1
              ~out_channels:1 ~kernel:3 ()) );
      ( "conv2d output sizes" >:: fun _ ->
        let y =
          Layer.forward
            (Layer.conv2d ~rng ~kernel_size:(3, 1) ~strides:(2, 1)
               ~paddings:(1, 0) ~dilations:(2, 1) ~in_channels:2 ~out_channels:5
               ~kernel:0 ())
            (Var.const (r [| 1; 9; 4; 2 |]))
        in
        (* h: (9 + 2 - 2*2 - 1) / 2 + 1 = 4; w: 4 *)
        assert_shape [| 1; 4; 4; 5 |] (Var.value y) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* State, safetensors and checkpoints                                       *)
  (* ------------------------------------------------------------------------ *)

  let temp () = Filename.temp_file "flockml" ".safetensors"

  let with_temp f =
    let file = temp () in
    Fun.protect ~finally:(fun () -> Sys.remove file) (fun () -> f file)

  let identical ~msg a b = assert_tensor ~rtol:0. ~atol:0. ~msg a b

  let state =
    [
      ( "load_state_dict copies values" >:: fun _ ->
        let a = resnet () and b = resnet () in
        let x = Var.const (r [| 2; 6; 6; 3 |]) in
        ignore (Layer.forward ~train:true a x);
        ignore (Layer.load_state_dict b (Layer.state_dict a));
        identical ~msg:"eval outputs"
          (Var.value (Layer.forward a x))
          (Var.value (Layer.forward b x)) );
      ( "strict loading reports and changes nothing" >:: fun _ ->
        let model =
          Layer.sequential [ Layer.linear ~rng ~in_dim:2 ~out_dim:3 () ]
        in
        let w0 = Tensor.copy (Var.value (List.hd (Layer.params model))) in
        let entries =
          [
            ("0.w", Tensor.ones [| 2; 3 |]);
            ("0.b", Tensor.ones [| 4 |]);
            ("extra", Tensor.ones [| 1 |]);
          ]
        in
        (match Layer.load_state_dict model entries with
        | _ -> assert_failure "expected Failure"
        | exception Failure _ -> ());
        identical ~msg:"unchanged" w0 (Var.value (List.hd (Layer.params model)));
        let report = Layer.load_state_dict ~strict:false model entries in
        assert_equal [] report.missing;
        assert_equal [ "extra" ] report.unexpected;
        assert_equal [ ("0.b", [| 3 |], [| 4 |]) ] report.mismatched;
        identical ~msg:"loaded"
          (Tensor.ones [| 2; 3 |])
          (Var.value (List.hd (Layer.params model))) );
      ( "safetensors round trip" >:: fun _ ->
        with_temp (fun file ->
            let entries =
              [
                ("a", r [| 2; 3 |]);
                ("weird \"name\"\n", r [| 4 |]);
                ("scalar", Tensor.scalar 3.5);
                ("empty", Tensor.zeros [| 0; 2 |]);
              ]
            in
            Safetensors.save ~metadata:[ ("epoch", "3") ] file entries;
            let back = Safetensors.load file in
            assert_equal (List.map fst entries) (List.map fst back);
            List.iter2 (fun (n, a) (_, b) -> identical ~msg:n a b) entries back;
            assert_equal [ ("epoch", "3") ] (Safetensors.load_metadata file);
            Safetensors.save ~dtype:`F32 file entries;
            List.iter2
              (fun (n, a) (_, b) ->
                assert_tensor ~rtol:1e-7 ~atol:1e-7 ~msg:n a b)
              entries (Safetensors.load file)) );
      ( "safetensors errors" >:: fun _ ->
        assert_invalid_arg (fun () ->
            Safetensors.to_string
              [ ("a", Tensor.ones [| 1 |]); ("a", Tensor.ones [| 1 |]) ]);
        let s = Safetensors.to_string [ ("a", Tensor.ones [| 4 |]) ] in
        let truncated = String.sub s 0 (String.length s - 1) in
        List.iter
          (fun bad ->
            match Safetensors.of_string bad with
            | _ -> assert_failure "expected Failure"
            | exception Failure _ -> ())
          [ ""; "\255\255\255\255\255\255\255\127{}"; truncated ] );
      ( "checkpoint round trip is bit-identical" >:: fun _ ->
        with_temp (fun file ->
            let a = resnet () and b = resnet () in
            let x = Var.const (r [| 2; 6; 6; 3 |]) in
            ignore (Layer.forward ~train:true a x);
            Checkpoint.save file a;
            ignore (Checkpoint.load file b);
            identical ~msg:"outputs"
              (Var.value (Layer.forward a x))
              (Var.value (Layer.forward b x));
            assert_raises
              (Failure
                 (Printf.sprintf "Checkpoint.load: %s has no optimizer state"
                    file))
              (fun () ->
                Checkpoint.load
                  ~optim:(Optim.sgd ~lr:0.1 (Layer.params b))
                  file b)) );
      ( "a resumed run matches one that never stopped" >:: fun _ ->
        with_temp (fun file ->
            let x = r [| 4; 6; 6; 3 |] and labels = [| 1; 5; 9; 0 |] in
            let train model opt steps =
              for _ = 1 to steps do
                Optim.zero_grad opt;
                Var.backward
                  (Loss.cross_entropy
                     (Layer.forward ~train:true model (Var.const x))
                     labels);
                Optim.step opt
              done
            in
            List.iter
              (fun make ->
                let a = resnet () in
                let opt_a = make (Layer.params a) in
                train a opt_a 3;
                Checkpoint.save ~optim:opt_a file a;
                train a opt_a 3;
                let b = resnet () in
                let opt_b = make (Layer.params b) in
                ignore (Checkpoint.load ~optim:opt_b file b);
                train b opt_b 3;
                List.iter2
                  (fun (n, ta) (_, tb) -> identical ~msg:n ta tb)
                  (Layer.state_dict a) (Layer.state_dict b))
              [
                (fun ps -> Optim.adamw ~lr:0.01 ps);
                (fun ps -> Optim.sgd ~lr:0.05 ~momentum:0.9 ~nesterov:true ps);
                (fun ps ->
                  Optim.rmsprop ~lr:0.01 ~centered:true ~momentum:0.5 ps);
              ]) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Whole models                                                             *)
  (* ------------------------------------------------------------------------ *)

  let models =
    [
      ( "a ResNet-style CNN trains" >:: fun _ ->
        let model = resnet () in
        let x = Var.const (r [| 4; 8; 8; 3 |]) and labels = [| 3; 1; 4; 1 |] in
        let opt =
          Optim.sgd ~lr:0.05 ~momentum:0.9 ~nesterov:true ~weight_decay:5e-4
            (Layer.params model)
        in
        let loss () =
          Loss.cross_entropy (Layer.forward ~train:true model x) labels
        in
        let first = Tensor.item (Var.value (loss ())) in
        for _ = 1 to 15 do
          Optim.zero_grad opt;
          Var.backward (loss ());
          Optim.step opt
        done;
        let last = Tensor.item (Var.value (loss ())) in
        assert_bool
          (Printf.sprintf "loss %g -> %g" first last)
          (last < first /. 2.) );
      ( "a small transformer trains" >:: fun _ ->
        let vocab = 7 and len = 6 in
        let model = gpt ~vocab ~dim:8 ~len in
        (* learn to predict the next token of two fixed sequences *)
        let seqs = [| 0; 1; 2; 3; 4; 5; 6; 6; 5; 4; 3; 2; 1; 0 |] in
        let tokens =
          Array.init (2 * len) (fun i -> seqs.((i / len * 7) + (i mod len)))
        in
        let targets =
          Array.init (2 * len) (fun i -> seqs.((i / len * 7) + (i mod len) + 1))
        in
        let x =
          Var.const
            (Tensor.of_array [| 2; len |] (Array.map float_of_int tokens))
        in
        let decay, no_decay =
          List.partition
            (fun (n, _) ->
              Filename.extension n = ".w"
              && not (String.length n > 4 && String.sub n 0 4 = "norm"))
            (Layer.named_params model)
        in
        let opt =
          Optim.adamw ~lr:0.02 ~weight_decay:0.01
            ~groups:[ Optim.group ~weight_decay:0. (List.map snd no_decay) ]
            (List.map snd decay)
        in
        let sched =
          Flockml.Schedule.(warmup ~steps:5 (cosine ~lr:0.02 ~total:55 ()))
        in
        let loss () =
          Loss.cross_entropy (Layer.forward ~train:true model x) targets
        in
        let first = Tensor.item (Var.value (loss ())) in
        for step = 0 to 59 do
          Optim.set_lr opt (sched step);
          Optim.zero_grad opt;
          Var.backward (loss ());
          ignore (Var.clip_grad_norm ~max_norm:1.0 (Optim.params opt));
          Optim.step opt
        done;
        let last = Tensor.item (Var.value (loss ())) in
        assert_shape [| 2; len; vocab |] (Var.value (Layer.forward model x));
        assert_bool
          (Printf.sprintf "loss %g -> %g" first last)
          (last < 0.2 *. first) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Initialisation                                                           *)
  (* ------------------------------------------------------------------------ *)

  let init =
    [
      ( "kaiming_uniform bound" >:: fun _ ->
        let x =
          Init.make ~rng Init.Kaiming_uniform ~fan_in:25 ~fan_out:4
            [| 25; 400 |]
        in
        let m = Tensor.item (Tensor.max (Tensor.abs x)) in
        assert_bool "within 1/sqrt fan_in" (m <= 0.2);
        assert_bool "close to it" (m > 0.19) );
      ( "xavier_normal std" >:: fun _ ->
        let x =
          Init.xavier_normal ~rng ~fan_in:100 ~fan_out:300 [| 100; 300 |]
        in
        assert_float ~eps:0.005
          (Float.sqrt (2. /. 400.))
          (Tensor.item (Tensor.std (Tensor.reshape x [| 30000 |]) 0)) );
      ( "orthogonal" >:: fun _ ->
        List.iter
          (fun shape ->
            let q = Init.orthogonal ~rng ~gain:2. shape in
            let rows = shape.(0) and cols = shape.(1) in
            let g =
              if rows <= cols then Tensor.matmul q (Tensor.transpose q)
              else Tensor.matmul (Tensor.transpose q) q
            in
            assert_tensor ~atol:1e-12
              (Tensor.scale 4. (Tensor.eye (min rows cols)))
              g)
          [ [| 4; 6 |]; [| 6; 4 |]; [| 5; 5 |] ] );
      ( "constant and ?init on constructors" >:: fun _ ->
        match
          Layer.linear ~init:(Init.Constant 0.5) ~in_dim:2 ~out_dim:3 ()
        with
        | Layer.Linear { w; _ } ->
            assert_tensor (Tensor.full [| 2; 3 |] 0.5) (Var.value w)
        | _ -> assert_failure "not linear" );
    ]

  let suite =
    "layers"
    >::: [
           "conv and pooling" >::: conv_checks;
           "activations" >::: activation_checks;
           "normalisation and embedding" >::: norm_checks;
           "sequence" >::: sequence_checks;
           "structure" >::: structure;
           "state" >::: state;
           "models" >::: models;
           "init" >::: init;
         ]
end

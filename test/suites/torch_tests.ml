(* Layers, losses, optimizers and schedules (Phases 9 and 10), checked
   against PyTorch itself: the expected outputs are .npy files written by
   test/fixtures/gen_torch.py and committed, so running the tests needs no
   Python. Each case repeats the hyperparameters gen_torch.py used. Weights
   are stored in FlockML's layout already.
   Run with: dune exec test/test_torch.exe (default backend) *)

open OUnit2

let path name =
  Filename.concat
    (Filename.concat (Lazy.force Numpy_tests.dir) "torch")
    (name ^ ".npy")

(** The suite for backend [B]. *)
module Make (B : Flockml.S) = struct
  open B
  open Helpers.Make (B)

  let fx name = Npy.load (path name)
  let fxi name = snd (Npy.load_ints (path name))
  let p name = Var.param (fx name)

  let close ?(tol = 1e-10) ~msg expected actual =
    assert_tensor ~rtol:tol ~atol:tol ~msg expected actual

  (** [layer] on the input [<name>_x], against [<name>_y]. *)
  let forward_case ?tol ?(x = "") ?(y = "") name layer =
    name >:: fun _ ->
    let x = if x = "" then name ^ "_x" else x
    and y = if y = "" then name ^ "_y" else y in
    close ?tol ~msg:name (fx y)
      (Var.value (Layer.forward (layer ()) (Var.const (fx x))))

  (* ------------------------------------------------------------------------ *)
  (* Layers                                                                   *)
  (* ------------------------------------------------------------------------ *)

  let conv_layers =
    List.map
      (fun (name, kernel, stride, padding, dilation, groups) ->
        forward_case name (fun () ->
            Layer.Conv2d
              {
                w = p (name ^ "_w");
                b = Some (p (name ^ "_b"));
                kernel;
                stride;
                padding;
                dilation;
                groups;
              }))
      [
        ("c_basic", (3, 3), (1, 1), (1, 1), (1, 1), 1);
        ("c_rect", (2, 3), (2, 1), (1, 0), (1, 1), 1);
        ("c_dil", (3, 3), (1, 1), (2, 2), (2, 2), 1);
        ("c_groups", (3, 3), (2, 2), (1, 1), (1, 1), 2);
        ("c_depthwise", (3, 3), (1, 1), (1, 1), (1, 1), 3);
      ]
    @ List.map
        (fun (name, kernel, stride, padding, dilation, groups) ->
          forward_case name (fun () ->
              Layer.Conv1d
                {
                  w = p (name ^ "_w");
                  b = Some (p (name ^ "_b"));
                  kernel;
                  stride;
                  padding;
                  dilation;
                  groups;
                }))
        [ ("c1_a", 3, 2, 1, 2, 1); ("c1_b", 3, 1, 1, 1, 2) ]
    @ [
        forward_case "ct" (fun () ->
            Layer.Conv_transpose2d
              {
                w = p "ct_w";
                b = Some (p "ct_b");
                kernel = (3, 3);
                stride = 2;
                padding = 1;
                output_padding = 1;
              });
        forward_case ~x:"pool_x" ~y:"maxpool_y" "max_pool2d padded" (fun () ->
            Layer.max_pool2d ~stride:2 ~padding:1 3);
        forward_case ~x:"pool_x" ~y:"avgpool_y" "avg_pool2d padded" (fun () ->
            Layer.avg_pool2d ~stride:2 ~padding:1 3);
        forward_case ~x:"pool_x" ~y:"avgpool2_y" "avg_pool2d 2x2" (fun () ->
            Layer.avg_pool2d 2);
      ]

  let activations =
    List.map
      (fun (name, layer) ->
        forward_case ~x:"act_x" ~y:name name (fun () -> layer))
      [
        ("leaky_relu", Layer.leaky_relu ~slope:0.1 ());
        ("elu", Layer.elu ~alpha:0.7 ());
        ("gelu", Layer.gelu ());
        ("gelu_tanh", Layer.gelu ~approximate:true ());
        ("silu", Layer.silu);
        ("softplus", Layer.softplus ~beta:2. ~threshold:5. ());
        ("softmax", Layer.softmax);
        ("log_softmax", Layer.log_softmax);
      ]

  let set layer name t =
    Tensor.blit_ ~src:t
      ~dst:(Var.value (List.assoc name (Layer.named_params layer)))

  let normalisation =
    List.map
      (fun name ->
        name >:: fun _ ->
        let bn = Layer.batch_norm ~momentum:0.2 4 in
        set bn "w" (fx (name ^ "_w"));
        set bn "b" (fx (name ^ "_b"));
        let y = Layer.forward ~train:true bn (Var.const (fx (name ^ "_x"))) in
        close ~msg:"train" (fx (name ^ "_y")) (Var.value y);
        let buf n = Var.value (List.assoc n (Layer.named_buffers bn)) in
        close ~msg:"running_mean" (fx (name ^ "_mean")) (buf "running_mean");
        close ~msg:"running_var" (fx (name ^ "_var")) (buf "running_var");
        assert_float 1. (Tensor.item (buf "num_batches_tracked"));
        let y2 = Layer.forward bn (Var.const (fx (name ^ "_x2"))) in
        close ~msg:"eval" (fx (name ^ "_y2")) (Var.value y2))
      [ "bn1"; "bn2" ]
    @ [
        forward_case "ln" (fun () ->
            Layer.Layer_norm
              { w = p "ln_w"; b = p "ln_b"; shape = [| 3; 4 |]; eps = 1e-5 });
        forward_case "gn" (fun () ->
            Layer.Group_norm
              { w = p "gn_w"; b = p "gn_b"; groups = 3; eps = 1e-5 });
      ]

  let recurrent name kind =
    forward_case ~x:"rnn_x" ~y:(name ^ "_y") name (fun () ->
        Layer.Recurrent
          {
            kind;
            w_ih = p (name ^ "_w_ih");
            w_hh = p (name ^ "_w_hh");
            b_ih = p (name ^ "_b_ih");
            b_hh = p (name ^ "_b_hh");
            hidden = 4;
          })

  let mha causal =
    Layer.Attention
      {
        in_w = p "mha_in_w";
        in_b = p "mha_in_b";
        out_w = p "mha_out_w";
        out_b = p "mha_out_b";
        heads = 2;
        causal;
      }

  let sequence =
    [
      recurrent "rnn" Layer.Rnn_tanh;
      recurrent "rnn_relu" Layer.Rnn_relu;
      recurrent "lstm" Layer.Lstm;
      recurrent "gru" Layer.Gru;
      forward_case "mha" (fun () -> mha false);
      forward_case ~x:"mha_x" "mha_causal" (fun () -> mha true);
      ( "mha cross-attention" >:: fun _ ->
        let x = Var.const (fx "mha_x") in
        close ~msg:"cross" (fx "mha_cross_y")
          (Var.value
             (Layer.attention (mha false)
                ~query:(Var.const (fx "mha_q"))
                ~key:x ~value:x)) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Losses: value and gradients                                              *)
  (* ------------------------------------------------------------------------ *)

  (** [f] of params made from the fixtures [inputs], against [<name>] and the
      gradients [<name>_g<i>], for the seed [<name>_seed]. *)
  let loss_case ?(tol = 1e-10) name inputs f =
    name >:: fun _ ->
    let ps = List.map (fun i -> Var.param (fx i)) inputs in
    let out = f ps in
    close ~tol ~msg:(name ^ " value") (fx name) (Var.value out);
    Var.backward (Var.sum (Var.mul out (Var.const (fx (name ^ "_seed")))));
    List.iteri
      (fun i v ->
        close ~tol
          ~msg:(Printf.sprintf "%s gradient %d" name i)
          (fx (Printf.sprintf "%s_g%d" name i))
          (Var.grad_exn v))
      ps

  let reductions = [ ("mean", `Mean); ("sum", `Sum); ("none", `None) ]

  let regression_losses =
    List.concat_map
      (fun (r, reduction) ->
        let pt = [ "loss_pred"; "loss_target" ] in
        [
          loss_case ("mse_" ^ r) pt (binary (Loss.mse ~reduction));
          loss_case ("l1_" ^ r) pt (binary (Loss.l1 ~reduction));
          loss_case ("smooth_l1_" ^ r) pt
            (binary (Loss.smooth_l1 ~reduction ~beta:0.5));
          loss_case ("huber_" ^ r) pt
            (binary (Loss.huber ~reduction ~delta:0.5));
        ])
      reductions

  let classification_losses =
    let labels = lazy (fxi "ce_labels") and weights = lazy (fx "ce_weights") in
    let seq_labels = lazy (fxi "ce_seq_labels") in
    let ( !! ) = Lazy.force in
    let ce ?reduction ?label_smoothing ?(weighted = false) () =
      unary (fun x ->
          Loss.cross_entropy ?reduction ?label_smoothing
            ?weights:(if weighted then Some !!weights else None)
            x !!labels)
    in
    [
      loss_case "ce_mean" [ "ce_logits" ] (ce ());
      loss_case "ce_sum" [ "ce_logits" ] (ce ~reduction:`Sum ());
      loss_case "ce_none" [ "ce_logits" ] (ce ~reduction:`None ());
      loss_case "ce_smooth" [ "ce_logits" ] (ce ~label_smoothing:0.1 ());
      loss_case "ce_weighted" [ "ce_logits" ]
        (ce ~weighted:true ~label_smoothing:0.2 ());
      loss_case "ce_weighted_none" [ "ce_logits" ]
        (ce ~weighted:true ~reduction:`None ());
      ( "nll_weighted" >:: fun _ ->
        let lp = Var.param (Tensor.log_softmax (fx "ce_logits")) in
        let out = Loss.nll ~weights:!!weights lp !!labels in
        close ~msg:"value" (fx "nll_weighted") (Var.value out);
        Var.backward out;
        close ~msg:"gradient" (fx "nll_weighted_g0") (Var.grad_exn lp) );
      loss_case "ce_seq" [ "ce_seq_logits" ]
        (unary (fun x -> Loss.cross_entropy ~label_smoothing:0.1 x !!seq_labels));
      loss_case "ce_seq_none" [ "ce_seq_logits" ]
        (unary (fun x -> Loss.cross_entropy ~reduction:`None x !!seq_labels));
      (* the epsilon inside the logs moves the result by about 1e-6 *)
      loss_case ~tol:1e-5 "bce"
        [ "bce_probs"; "bce_targets" ]
        (binary (Loss.binary_cross_entropy ?reduction:None ?eps:None));
      loss_case "bce_logits_mean"
        [ "bce_logits"; "bce_targets" ]
        (binary (Loss.bce_with_logits ?reduction:None ?pos_weight:None));
      loss_case "bce_logits_pw"
        [ "bce_logits"; "bce_targets" ]
        (binary (fun x t ->
             Loss.bce_with_logits ~reduction:`Sum
               ~pos_weight:(fx "bce_pos_weight") x t));
    ]
    @ List.map
        (fun (r, reduction) ->
          loss_case ("kl_" ^ r)
            [ "kl_input"; "kl_target" ]
            (binary (Loss.kl_div ~reduction)))
        [
          ("mean", `Mean);
          ("sum", `Sum);
          ("batchmean", `Batch_mean);
          ("none", `None);
        ]
    @ [
        loss_case "kl_log_target"
          [ "kl_input"; "kl_target" ]
          (binary (fun i t ->
               Loss.kl_div ~reduction:`Batch_mean ~log_target:true i
                 (Var.log (Var.clamp ~min:1e-3 t))));
      ]

  (* ------------------------------------------------------------------------ *)
  (* Optimizers: the parameters after 10 steps                                *)
  (* ------------------------------------------------------------------------ *)

  let optim_case ?(tol = 1e-10) name make =
    name >:: fun _ ->
    let x = Var.const (fx "opt_x") and y = Var.const (fx "opt_y") in
    let w = Var.param (fx "opt_w0") and b = Var.param (fx "opt_b0") in
    let opt = make w b in
    for _ = 1 to 10 do
      Optim.zero_grad opt;
      Var.backward
        (Var.mean
           (Var.square (Var.sub (Var.tanh (Var.add (Var.matmul x w) b)) y)));
      Optim.step opt
    done;
    close ~tol ~msg:"w" (fx ("opt_" ^ name ^ "_w")) (Var.value w);
    close ~tol ~msg:"b" (fx ("opt_" ^ name ^ "_b")) (Var.value b)

  let optimizers =
    [
      optim_case "sgd" (fun w b -> Optim.sgd ~lr:0.1 [ w; b ]);
      optim_case "sgd_momentum" (fun w b ->
          Optim.sgd ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01 [ w; b ]);
      optim_case "sgd_nesterov" (fun w b ->
          Optim.sgd ~lr:0.1 ~momentum:0.9 ~weight_decay:0.01 ~nesterov:true
            [ w; b ]);
      optim_case "adam" (fun w b -> Optim.adam ~lr:0.05 [ w; b ]);
      optim_case "adam_wd" (fun w b ->
          Optim.adam ~lr:0.05 ~weight_decay:0.1 [ w; b ]);
      optim_case "adamw" (fun w b ->
          Optim.adamw ~lr:0.05 ~weight_decay:0.1 [ w; b ]);
      optim_case "rmsprop" (fun w b -> Optim.rmsprop ~lr:0.01 [ w; b ]);
      optim_case "rmsprop_full" (fun w b ->
          Optim.rmsprop ~lr:0.01 ~momentum:0.9 ~centered:true ~weight_decay:0.01
            [ w; b ]);
      optim_case "adagrad" (fun w b -> Optim.adagrad ~lr:0.1 [ w; b ]);
      optim_case "adagrad_full" (fun w b ->
          Optim.adagrad ~lr:0.1 ~lr_decay:0.01 ~weight_decay:0.01
            ~initial_accumulator_value:0.1 [ w; b ]);
      optim_case "groups" (fun w b ->
          Optim.adamw ~lr:0.05 ~weight_decay:0.1
            ~groups:[ Optim.group ~lr:0.1 ~weight_decay:0. [ b ] ]
            [ w ]);
    ]

  (* ------------------------------------------------------------------------ *)
  (* Schedules                                                                *)
  (* ------------------------------------------------------------------------ *)

  let schedule_case name steps (s : Flockml.Schedule.t) =
    name >:: fun _ ->
    close ~tol:1e-12 ~msg:name (fx name)
      (Tensor.of_array [| steps |] (Array.init steps s))

  let schedules =
    let open Flockml.Schedule in
    [
      schedule_case "sched_step" 25
        (step_decay ~gamma:0.5 ~lr:0.1 ~step_size:5 ());
      schedule_case "sched_multistep" 25
        (multi_step ~gamma:0.2 ~lr:0.1 [ 3; 7; 12 ]);
      schedule_case "sched_exponential" 25 (exponential ~lr:0.1 ~gamma:0.9);
      schedule_case "sched_cosine" 21
        (cosine ~min_lr:0.001 ~lr:0.1 ~total:20 ());
      schedule_case "sched_linear" 25
        (linear ~start_factor:0.25 ~lr:0.1 ~total:10 ());
      schedule_case "sched_one_cycle" 24 (one_cycle ~max_lr:0.1 ~total:25 ());
      ( "sched_plateau" >:: fun _ ->
        let metrics = Tensor.to_array (fx "plateau_metrics") in
        let p = Plateau.create ~factor:0.5 ~patience:2 ~cooldown:1 0.1 in
        let lrs =
          Array.map
            (fun m ->
              let lr = Plateau.lr p in
              ignore (Plateau.step p m : float);
              lr)
            metrics
        in
        close ~tol:1e-12 ~msg:"plateau" (fx "sched_plateau")
          (Tensor.of_array [| Array.length lrs |] lrs) );
      ( "warmup then cosine" >:: fun _ ->
        let s = warmup ~steps:4 (cosine ~lr:1. ~total:10 ()) in
        assert_float 0.25 (s 0);
        assert_float 1. (s 3);
        assert_float 1. (s 4);
        assert_float 0.5 (s 9);
        assert_float 0. (s 14) );
    ]

  (* ------------------------------------------------------------------------ *)
  (* Safetensors written by another implementation                            *)
  (* ------------------------------------------------------------------------ *)

  let safetensors =
    [
      ( "every dtype of a foreign file" >:: fun _ ->
        let file =
          Filename.concat (Filename.dirname (path "x")) "mixed.safetensors"
        in
        let entries = Safetensors.load file in
        assert_equal ~printer:(String.concat ", ")
          [ "a_f16"; "b_bf16"; "c_i64"; "d_f32"; "e_f64"; "f_u8" ]
          (List.map fst entries);
        List.iter
          (fun (name, t) -> close ~tol:0. ~msg:name (fx ("st_" ^ name)) t)
          entries;
        assert_equal (Some "pt")
          (List.assoc_opt "format" (Safetensors.load_metadata file)) );
    ]

  let suite =
    "torch"
    >::: [
           "layers" >::: conv_layers @ activations @ normalisation @ sequence;
           "losses" >::: regression_losses @ classification_losses;
           "optimizers" >::: optimizers;
           "schedules" >::: schedules;
           "safetensors" >::: safetensors;
         ]
end

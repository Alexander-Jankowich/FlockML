(** Train ResNet-20 on CIFAR-10, with augmentation, a cosine schedule and
    Nesterov SGD: the first reference model of the roadmap's Phase 12. Its
    PyTorch twin is bench/torch_cifar.py.

    {v
    ./scripts/download_cifar10.sh     # once
    dune exec --profile release bin/train_cifar.exe -- --epochs 20
    dune exec --profile release bin/train_cifar.exe -- --checkpoint checkpoints/resnet20_cifar10.safetensors
    dune exec --profile release bin/train_cifar.exe -- --checkpoint checkpoints/resnet20_cifar10.safetensors --resume
    v}

    With [--resume], training continues after the epoch the checkpoint was saved
    at; a checkpoint from the last epoch is only evaluated.

    Batches are read, augmented (random crop with padding 4, horizontal flip)
    and normalised on a background domain while the previous step trains. The
    default backend is BLAS float32, the fastest. *)

open Flockml

let data_dir = ref "data/cifar-10"
let epochs = ref 20
let batch_size = ref 128
let lr = ref 0.1
let weight_decay = ref 5e-4
let limit = ref 0
let seed = ref 42
let augment = ref true
let checkpoint = ref ""
let resume = ref false
let log_every = ref 100

let backends : (string * (module Flockml.S)) list =
  [
    ("ocaml", (module Flockml.Default));
    ("blas", (module Flockml.Make (Tensor_blas.F64)));
    ("blas32", (module Flockml.Make (Tensor_blas.F32)));
  ]

let backend = ref "blas32"

let spec =
  [
    ( "--data-dir",
      Arg.Set_string data_dir,
      " CIFAR-10 files (default data/cifar-10)" );
    ("--epochs", Arg.Set_int epochs, " Number of epochs (default 20)");
    ("--batch-size", Arg.Set_int batch_size, " Mini-batch size (default 128)");
    ("--lr", Arg.Set_float lr, " Peak learning rate (default 0.1)");
    ( "--weight-decay",
      Arg.Set_float weight_decay,
      " Weight decay (default 5e-4)" );
    ( "--limit",
      Arg.Set_int limit,
      " Use only the first N training images (0 = all)" );
    ("--seed", Arg.Set_int seed, " Random seed (default 42)");
    ("--no-augment", Arg.Clear augment, " Train without crops and flips");
    ( "--checkpoint",
      Arg.Set_string checkpoint,
      " Save the model and optimizer to this file after every epoch" );
    ( "--resume",
      Arg.Set resume,
      " Start from --checkpoint (it records the epoch it was saved at)" );
    ("--log-every", Arg.Set_int log_every, " Print the loss every N batches");
    ( "--backend",
      Arg.Symbol (List.map fst backends, fun s -> backend := s),
      " Tensor backend: blas32 (Tensor_blas.F32; the default), blas \
       (Tensor_blas.F64) or ocaml (pure OCaml, float64)" );
    ( "--threads",
      Arg.Int Parallel.set_num_threads,
      " Threads for the tensor kernels (default: FLOCKML_THREADS, or up to 8)"
    );
  ]

module Run (B : Flockml.S) = struct
  open B
  module R = Reference_models.Make (B)

  (** Test accuracy (top-1 and top-5) and the per-class report, in batches of
      500 with no graph recorded. *)
  let evaluate model test =
    let preds = ref [] and labs = ref [] and top5 = ref 0. in
    Var.no_grad (fun () ->
        Dataset.loader ~batch_size:500 ~prefetch:true
          ~transform:(Transform.on_inputs R.cifar_normalize)
          test
        |> Seq.iter (fun (x, labels) ->
            let scores = Var.value (Layer.forward model (Var.const x)) in
            preds := Metrics.predict scores :: !preds;
            labs := labels :: !labs;
            top5 :=
              !top5
              +. Metrics.top_k_accuracy ~k:5 ~scores ~labels
                 *. float_of_int (Array.length labels)));
    let predictions = Array.concat (List.rev !preds) in
    let labels = Array.concat (List.rev !labs) in
    ( Metrics.classification_report ~num_classes:10 ~predictions ~labels,
      !top5 /. float_of_int (Array.length labels) )

  let main () =
    let rng = Random.State.make [| !seed |] in
    let limit = if !limit > 0 then Some !limit else None in
    let train = Cifar10.source ?limit ~dir:!data_dir Cifar10.Train in
    let test = Cifar10.source ~dir:!data_dir Cifar10.Test in
    Printf.printf "train: %d images, test: %d images\n%!" train.length
      test.length;
    let model = R.resnet20 ~rng in
    Printf.printf "ResNet-20: %d parameters, backend %s\n%!"
      (Layer.num_params model) !backend;
    let opt =
      Optim.sgd ~lr:!lr ~momentum:0.9 ~nesterov:true ~weight_decay:!weight_decay
        (Layer.params model)
    in
    let steps_per_epoch = (train.length + !batch_size - 1) / !batch_size in
    let schedule =
      Schedule.cosine ~lr:!lr ~total:(!epochs * steps_per_epoch) ()
    in
    let first_epoch =
      if !resume && !checkpoint <> "" then begin
        ignore (Checkpoint.load ~optim:opt !checkpoint model);
        let e =
          int_of_string
            (List.assoc "epoch" (Safetensors.load_metadata !checkpoint))
        in
        Printf.printf "resumed from %s after epoch %d\n%!" !checkpoint e;
        e + 1
      end
      else 1
    in
    let transform = if !augment then R.cifar_augment else R.cifar_normalize in
    let now = Unix.gettimeofday in
    for epoch = first_epoch to !epochs do
      let t0 = now () in
      let epoch_loss = Metrics.Running.create ()
      and recent = Metrics.Running.create () in
      (* A fresh generator per epoch, so a resumed run sees the same data. *)
      let rng = Random.State.make [| !seed; epoch |] in
      Dataset.loader ~shuffle:true ~rng
        ~transform:(Transform.on_inputs transform)
        ~batch_size:!batch_size train
      |> Seq.iteri (fun i (x, labels) ->
          Optim.set_lr opt (schedule (((epoch - 1) * steps_per_epoch) + i));
          Optim.zero_grad opt;
          let loss =
            Loss.cross_entropy
              (Layer.forward ~train:true model (Var.const x))
              labels
          in
          Var.backward loss;
          Optim.step opt;
          let l = Tensor.item (Var.value loss) in
          let n = float_of_int (Array.length labels) in
          Metrics.Running.add ~weight:n epoch_loss l;
          Metrics.Running.add ~weight:n recent l;
          if (i + 1) mod !log_every = 0 then begin
            Printf.printf "  epoch %d  batch %4d  loss %.4f  lr %.4f\n%!" epoch
              (i + 1)
              (Metrics.Running.mean recent)
              (Optim.lr opt);
            Metrics.Running.reset recent
          end);
      let t1 = now () in
      let report, top5 = evaluate model test in
      Printf.printf
        "epoch %d: loss %.4f, train %.1fs, eval %.1fs, test accuracy %.2f%% \
         (top-5 %.2f%%)\n\
         %!"
        epoch
        (Metrics.Running.mean epoch_loss)
        (t1 -. t0)
        (now () -. t1)
        (100. *. report.accuracy) (100. *. top5);
      if !checkpoint <> "" then
        Checkpoint.save ~optim:opt
          ~metadata:[ ("epoch", string_of_int epoch) ]
          !checkpoint model
    done;
    let report, _ = evaluate model test in
    Format.printf "@.%a@." (Metrics.pp_report ~names:Cifar10.classes) report
end

(* Create the checkpoint's directory, so a save after the first epoch can't
   fail on it. *)
let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

let () =
  Arg.parse (Arg.align spec) (fun _ -> ()) "train_cifar [options]";
  if !checkpoint <> "" then mkdir_p (Filename.dirname !checkpoint);
  let module B = (val List.assoc !backend backends) in
  let module R = Run (B) in
  R.main ()

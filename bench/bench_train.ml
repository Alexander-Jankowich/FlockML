(** End-to-end training cost of the reference models, split into data loading,
    forward pass (including the loss), backward pass and optimizer step, with
    heap and GC counts:
    - the MNIST MLP and CNN of [bin/train_mnist.ml] (Adam, batch 64);
    - ResNet-20 on CIFAR-10 with augmentation, as in [bin/train_cifar.ml]
      (Nesterov SGD, batch 128);
    - the character-level GPT of [bin/gpt.ml] on Tiny Shakespeare (AdamW, batch
      32 of 128 characters).

    Compare with bench/torch_baseline.py, bench/torch_cifar.py and
    bench/torch_gpt.py. Always run in release mode:

    {v
    dune exec --profile release bench/bench_train.exe
    dune exec --profile release bench/bench_train.exe -- --model cnn --steps 200
    dune exec --profile release bench/bench_train.exe -- --model resnet20 --steps 50
    dune exec --profile release bench/bench_train.exe -- --backend blas32
    v}

    Batches come from {!Dataset.loader}, which prepares the next one on a
    background domain during the current step ([--no-prefetch] turns that off),
    so the data column is the time the step waited for its batch. By default
    every model trains for one full epoch; [--steps N] stops after N steps, and
    the epoch column is extrapolated from the per-step average. Missing datasets
    are replaced by random data of the same shape. Output is a Markdown table.
*)

open Flockml
open Bench_util

let data_dir = ref ""
let models = ref [ `Mlp; `Cnn ]
let steps = ref 0
let batch_size = ref 0
let warmup = 3
let prefetch = ref true

let spec =
  [
    ( "--model",
      Arg.Symbol
        ( [ "mlp"; "cnn"; "both"; "resnet20"; "gpt"; "all" ],
          function
          | "mlp" -> models := [ `Mlp ]
          | "cnn" -> models := [ `Cnn ]
          | "resnet20" -> models := [ `Resnet20 ]
          | "gpt" -> models := [ `Gpt ]
          | "all" -> models := [ `Mlp; `Cnn; `Resnet20; `Gpt ]
          | _ -> models := [ `Mlp; `Cnn ] ),
      " Which model to time (default both: the MNIST MLP and CNN)" );
    ( "--steps",
      Arg.Set_int steps,
      " Training steps to time (default 0 = one full epoch)" );
    ( "--batch-size",
      Arg.Set_int batch_size,
      " Mini-batch size (default 64 for MNIST, 128 for ResNet-20, 32 for GPT)"
    );
    ( "--threads",
      Arg.Int Parallel.set_num_threads,
      " Threads for the tensor kernels (default: FLOCKML_THREADS, or up to 8)"
    );
    ( "--no-prefetch",
      Arg.Clear prefetch,
      " Prepare each batch in the training loop instead of in the background" );
    backend_arg;
    ( "--data-dir",
      Arg.Set_string data_dir,
      " Location of the dataset (default data/mnist, data/cifar-10 or \
       data/tinyshakespeare); random data is used if it is missing" );
  ]

let model_name = function
  | `Mlp -> "MLP 784-128-10"
  | `Cnn -> "CNN"
  | `Resnet20 -> "ResNet-20"
  | `Gpt -> "GPT (char)"

type totals = {
  mutable data : float;
  mutable forward : float;
  mutable backward : float;
  mutable optim : float;
}
(** Per-phase wall-clock totals, in seconds. *)

let run (module B : Flockml.S) kind =
  let open B in
  let open Models (B) in
  let module R = Reference_models.Make (B) in
  let rng = Random.State.make [| 42 |] in
  let dir default = if !data_dir = "" then default else !data_dir in
  let bs default = if !batch_size > 0 then !batch_size else default in
  let gpt_config =
    { R.vocab = 65; context = 128; dim = 128; heads = 4; layers = 4 }
  in
  (* The model, its optimizer, the batches of one epoch and their count, and
     where the data came from. *)
  let model, opt, batches, steps_per_epoch, source =
    match kind with
    | (`Mlp | `Cnn) as k ->
        let train, source =
          load_train ~data_dir:(dir "data/mnist") ~flatten:(k = `Mlp)
        in
        let model = build_model rng k in
        let b = bs 64 in
        ( model,
          Optim.adam ~lr:1e-3 (Layer.params model),
          Dataset.loader ~shuffle:true ~rng ~prefetch:!prefetch ~batch_size:b
            (Dataset.labelled train),
          (Dataset.length train + b - 1) / b,
          source )
    | `Resnet20 ->
        let d = dir "data/cifar-10" in
        let src, source =
          if Sys.file_exists (Filename.concat d "data_batch_1.bin") then
            (Cifar10.source ~dir:d Cifar10.Train, "CIFAR-10")
          else
            let rng = Random.State.make [| 0 |] in
            ( Dataset.labelled
                {
                  Dataset.inputs =
                    Tensor.rand_uniform ~rng ~lo:0. ~hi:1.
                      [| 50000; 32; 32; 3 |];
                  labels = Array.init 50000 (fun _ -> Random.State.int rng 10);
                },
              "synthetic (no CIFAR-10 files found)" )
        in
        let model = R.resnet20 ~rng in
        let b = bs 128 in
        ( model,
          Optim.sgd ~lr:0.1 ~momentum:0.9 ~nesterov:true ~weight_decay:5e-4
            (Layer.params model),
          Dataset.loader ~shuffle:true ~rng ~prefetch:!prefetch
            ~transform:(Transform.on_inputs R.cifar_augment)
            ~batch_size:b src,
          (src.length + b - 1) / b,
          source )
    | `Gpt ->
        let file = Filename.concat (dir "data/tinyshakespeare") "input.txt" in
        let ids, source =
          if Sys.file_exists file then
            ( (R.text_of_string
                 (In_channel.with_open_bin file In_channel.input_all))
                .ids,
              "Tiny Shakespeare" )
          else
            let rng = Random.State.make [| 0 |] in
            ( Array.init 1_000_000 (fun _ -> Random.State.int rng 65),
              "synthetic (no Tiny Shakespeare file found)" )
        in
        let model = R.gpt ~rng gpt_config in
        let decay, no_decay = R.decay_split model in
        let b = bs 32 in
        let src = R.windows ~context:gpt_config.context ids in
        ( model,
          Optim.adamw ~lr:1e-3 ~weight_decay:0.1
            ~groups:[ Optim.group ~weight_decay:0. (List.map snd no_decay) ]
            (List.map snd decay),
          Dataset.loader ~shuffle:true ~rng ~prefetch:!prefetch ~batch_size:b
            src,
          (* an "epoch" of GPT training: as many characters as the text *)
          Array.length ids / (b * gpt_config.context),
          source )
  in
  let steps =
    if !steps <= 0 then steps_per_epoch else min !steps steps_per_epoch
  in
  if steps <= warmup then
    invalid_arg
      (Printf.sprintf "bench_train: need more than %d steps (the warm-up)"
         warmup);
  let tot = { data = 0.; forward = 0.; backward = 0.; optim = 0. } in
  let batches = ref batches in
  let step ~record =
    let t0 = now () in
    let x, labels =
      match Seq.uncons !batches with
      | Some (b, rest) ->
          batches := rest;
          b
      | None -> failwith "bench_train: ran out of batches"
    in
    let t1 = now () in
    let logits = Layer.forward ~train:true ~rng model (Var.const x) in
    let loss = Loss.cross_entropy logits labels in
    let t2 = now () in
    Var.backward loss;
    let t3 = now () in
    (* zero_grad after the step rather than before the forward pass, so the
       phases split cleanly; nothing reads the gradients in between. *)
    Optim.step opt;
    Optim.zero_grad opt;
    let t4 = now () in
    if record then begin
      tot.data <- tot.data +. (t1 -. t0);
      tot.forward <- tot.forward +. (t2 -. t1);
      tot.backward <- tot.backward +. (t3 -. t2);
      tot.optim <- tot.optim +. (t4 -. t3)
    end
  in
  let timed = steps - warmup in
  for _ = 1 to warmup do
    step ~record:false
  done;
  let a0 = allocated_bytes () and g0 = Gc.quick_stat () in
  let start = now () in
  for _ = 1 to timed do
    step ~record:true
  done;
  let wall = now () -. start in
  let g1 = Gc.quick_stat () and a1 = allocated_bytes () in
  let per_step x = x /. float_of_int timed in
  let ms x = 1000. *. per_step x in
  Printf.printf
    "| %-14s | %5d | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %8.2f | %8.2f | \
     %6.2f | %6.3f |\n\
     %!"
    (model_name kind) timed (ms tot.data) (ms tot.forward) (ms tot.backward)
    (ms tot.optim) (ms wall)
    (per_step wall *. float_of_int steps_per_epoch)
    (per_step (a1 -. a0) /. 1e6)
    (per_step (float_of_int (g1.minor_collections - g0.minor_collections)))
    (per_step (float_of_int (g1.major_collections - g0.major_collections)));
  (source, steps_per_epoch)

let () =
  Arg.parse (Arg.align spec) (fun _ -> ()) "bench_train [options]";
  Printf.printf "FlockML training, %s, %s\n\n"
    (if !prefetch then "batches prefetched" else "batches made in the loop")
    (environment ());
  Printf.printf
    "| %-14s | %5s | %7s | %7s | %7s | %7s | %7s | %8s | %8s | %6s | %6s |\n"
    "model" "steps" "data" "fwd" "bwd" "optim" "step" "epoch s" "MB/step"
    "minor" "major";
  Printf.printf "|%s|\n"
    (String.concat "|"
       (List.map
          (fun w -> String.make (w + 2) '-')
          [ 14; 5; 7; 7; 7; 7; 7; 8; 8; 6; 6 ]));
  let results = List.map (run (backend ())) !models in
  let source, steps_per_epoch = List.hd results in
  Printf.printf
    "\n\
     data / fwd / bwd / optim / step: ms per step (wall clock); epoch s: \
     per-step time x %d steps;\n\
     MB/step: OCaml heap allocation; minor / major: GC collections per step.\n\
     Data: %s. First %d steps are warm-up and not counted.\n"
    steps_per_epoch source warmup

(** End-to-end training cost of the MNIST MLP and CNN from [bin/train_mnist.ml],
    split into data loading, forward pass (including the loss), backward pass
    and optimizer step, with heap and GC counts. Compare with
    bench/torch_baseline.py. Always run in release mode:

    {v
    dune exec --profile release bench/bench_train.exe
    dune exec --profile release bench/bench_train.exe -- --model cnn --steps 200
    dune exec --profile release bench/bench_train.exe -- --backend blas32
    v}

    By default every model trains for one full epoch (938 steps at batch size
    64). With [--steps N] it stops after N steps, and the epoch column is
    extrapolated from the per-step average. Output is a Markdown table. *)

open Flockml
open Bench_util

let data_dir = ref "data/mnist"
let models = ref [ `Mlp; `Cnn ]
let steps = ref 0
let batch_size = ref 64
let warmup = 3
let reuse = ref true

let spec =
  [
    ( "--model",
      Arg.Symbol
        ( [ "mlp"; "cnn"; "both" ],
          function
          | "mlp" -> models := [ `Mlp ]
          | "cnn" -> models := [ `Cnn ]
          | _ -> models := [ `Mlp; `Cnn ] ),
      " Which model to time (default both)" );
    ( "--steps",
      Arg.Set_int steps,
      " Training steps to time (default 0 = one full epoch)" );
    ("--batch-size", Arg.Set_int batch_size, " Mini-batch size (default 64)");
    ( "--threads",
      Arg.Int Parallel.set_num_threads,
      " Threads for the tensor kernels (default: FLOCKML_THREADS, or up to 8)"
    );
    ( "--no-reuse",
      Arg.Clear reuse,
      " Allocate every batch afresh instead of reusing one buffer" );
    backend_arg;
    ( "--data-dir",
      Arg.Set_string data_dir,
      " Location of the IDX files; random data is used if they are missing" );
  ]

let model_name = function `Mlp -> "MLP 784-128-10" | `Cnn -> "CNN"

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
  let rng = Random.State.make [| 42 |] in
  let train, source = load_train ~data_dir:!data_dir ~flatten:(kind = `Mlp) in
  let n = Dataset.length train in
  let steps_per_epoch = (n + !batch_size - 1) / !batch_size in
  let steps =
    if !steps <= 0 then steps_per_epoch else min !steps steps_per_epoch
  in
  if steps <= warmup then
    invalid_arg
      (Printf.sprintf "bench_train: need more than %d steps (the warm-up)"
         warmup);
  let model = build_model rng kind in
  let opt = Optim.adam ~lr:1e-3 (Layer.params model) in
  let tot = { data = 0.; forward = 0.; backward = 0.; optim = 0. } in
  let batches =
    ref
      (Dataset.batches ~shuffle:true ~reuse:!reuse ~rng ~batch_size:!batch_size
         train)
  in
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
  Printf.printf "FlockML training, batch %d, Adam, %s\n\n" !batch_size
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

(** Train an MLP or CNN on MNIST / Fashion-MNIST.

    ./scripts/download_mnist.sh # once dune exec --profile release
    bin/train_mnist.exe -- --epochs 3 dune exec --profile release
    bin/train_mnist.exe -- --model cnn --limit 10000 dune exec --profile release
    bin/train_mnist.exe -- --backend blas32

    Targets: the MLP should reach ~97% test accuracy on MNIST within a few
    epochs; the CNN ~98-99%. Fashion-MNIST is harder (~85-90%). *)

open Flockml

let data_dir = ref "data/mnist"
let epochs = ref 3
let batch_size = ref 64
let lr = ref 1e-3
let optimizer = ref "adam"
let model_kind = ref "mlp"
let limit = ref 0
let seed = ref 42
let log_every = ref 100

(** The tensor backends, chosen with [--backend]. *)
let backends : (string * (module Flockml.S)) list =
  [
    ("ocaml", (module Flockml.Default));
    ("blas", (module Flockml.Make (Tensor_blas.F64)));
    ("blas32", (module Flockml.Make (Tensor_blas.F32)));
  ]

let backend = ref "ocaml"

let spec =
  [
    ( "--data-dir",
      Arg.Set_string data_dir,
      " Directory with the IDX files (default data/mnist)" );
    ("--epochs", Arg.Set_int epochs, " Number of epochs (default 3)");
    ("--batch-size", Arg.Set_int batch_size, " Mini-batch size (default 64)");
    ("--lr", Arg.Set_float lr, " Learning rate (default 1e-3)");
    ( "--optim",
      Arg.Symbol ([ "adam"; "sgd" ], fun s -> optimizer := s),
      " Optimizer" );
    ( "--model",
      Arg.Symbol ([ "mlp"; "cnn" ], fun s -> model_kind := s),
      " Architecture" );
    ( "--limit",
      Arg.Set_int limit,
      " Use only the first N training examples (0 = all)" );
    ("--seed", Arg.Set_int seed, " Random seed (default 42)");
    ("--log-every", Arg.Set_int log_every, " Print the loss every N batches");
    ( "--backend",
      Arg.Symbol (List.map fst backends, fun s -> backend := s),
      " Tensor backend: ocaml (pure OCaml, float64; the default), blas \
       (Tensor_blas.F64) or blas32 (Tensor_blas.F32)" );
    ( "--threads",
      Arg.Int Parallel.set_num_threads,
      " Threads for the tensor kernels (default: FLOCKML_THREADS, or up to 8)"
    );
  ]

(** Training and evaluation, on backend [B]. *)
module Run (B : Flockml.S) = struct
  open B

  (** The architecture named by [--model]: a 784-128-10 MLP, or a two-block
      conv/pool CNN followed by a linear classifier. *)
  let build_model rng = function
    | "mlp" ->
        Layer.sequential
          [
            Layer.linear ~rng ~in_dim:784 ~out_dim:128 ();
            Layer.relu;
            Layer.linear ~rng ~in_dim:128 ~out_dim:10 ();
          ]
    | "cnn" ->
        (* 28x28x1 -> conv 28x28x8 -> pool 14x14x8 -> conv 14x14x16
           -> pool 7x7x16 -> flatten 784 -> linear 10 *)
        Layer.sequential
          [
            Layer.conv2d ~rng ~padding:1 ~in_channels:1 ~out_channels:8
              ~kernel:3 ();
            Layer.relu;
            Layer.max_pool2d 2;
            Layer.conv2d ~rng ~padding:1 ~in_channels:8 ~out_channels:16
              ~kernel:3 ();
            Layer.relu;
            Layer.max_pool2d 2;
            Layer.flatten;
            Layer.linear ~rng ~in_dim:(7 * 7 * 16) ~out_dim:10 ();
          ]
    | other -> failwith ("unknown model " ^ other)

  (** Test accuracy and confusion matrix, predicting in batches of 256 with
      dropout off and no graph recorded. *)
  let evaluate model (test : Dataset.t) =
    let preds =
      Var.no_grad (fun () ->
          Dataset.batches ~batch_size:256 test
          |> Seq.map (fun (x, _) ->
              Metrics.predict (Var.value (Layer.forward model (Var.const x))))
          |> List.of_seq |> Array.concat)
    in
    ( Metrics.accuracy ~predictions:preds ~labels:test.labels,
      Metrics.confusion_matrix ~num_classes:10 ~predictions:preds
        ~labels:test.labels )

  let main () =
    let rng = Random.State.make [| !seed |] in
    let flatten = !model_kind = "mlp" in
    let limit = if !limit > 0 then Some !limit else None in
    let train = Mnist.load ~flatten ?limit ~dir:!data_dir Mnist.Train in
    let test = Mnist.load ~flatten ~dir:!data_dir Mnist.Test in
    Printf.printf "train: %d examples, test: %d examples\n%!"
      (Dataset.length train) (Dataset.length test);
    Printf.printf "backend: %s\n%!"
      (match !backend with
      | "ocaml" -> "pure OCaml (float64)"
      | "blas" -> "Tensor_blas.F64, BLAS " ^ Tensor_blas.blas
      | _ -> "Tensor_blas.F32, BLAS " ^ Tensor_blas.blas);
    let model = build_model rng !model_kind in
    Printf.printf "%s\n%d parameters\n%!" (Layer.to_string model)
      (Layer.num_params model);
    let params = Layer.params model in
    let opt =
      match !optimizer with
      | "sgd" -> Optim.sgd ~momentum:0.9 ~lr:!lr params
      | _ -> Optim.adam ~lr:!lr params
    in
    (* Wall-clock time: Sys.time would be process CPU time, which adds up the
       time of every core once work runs in parallel. *)
    let now = Unix.gettimeofday in
    for epoch = 1 to !epochs do
      let t0 = now () in
      let running = ref 0.0 and count = ref 0 in
      Dataset.batches ~shuffle:true ~reuse:true ~rng ~batch_size:!batch_size
        train
      |> Seq.iteri (fun i (x, labels) ->
          Optim.zero_grad opt;
          let logits = Layer.forward ~train:true ~rng model (Var.const x) in
          let loss = Loss.cross_entropy logits labels in
          Var.backward loss;
          Optim.step opt;
          running := !running +. Tensor.item (Var.value loss);
          incr count;
          if (i + 1) mod !log_every = 0 then begin
            Printf.printf "  epoch %d  batch %5d  loss %.4f\n%!" epoch (i + 1)
              (!running /. float_of_int !count);
            running := 0.0;
            count := 0
          end);
      let t1 = now () in
      let acc, _ = evaluate model test in
      Printf.printf
        "epoch %d done: train %.1fs, eval %.1fs  test accuracy %.2f%%\n%!" epoch
        (t1 -. t0)
        (now () -. t1)
        (100.0 *. acc)
    done;
    let acc, cm = evaluate model test in
    Format.printf "final test accuracy: %.2f%%@.%a@." (100.0 *. acc)
      Metrics.pp_confusion_matrix cm
end

let () =
  Arg.parse (Arg.align spec) (fun _ -> ()) "train_mnist [options]";
  let module B = (val List.assoc !backend backends) in
  let module R = Run (B) in
  R.main ()

(** Regression on a CSV file: the wine-quality score of red wines from 11
    physico-chemical measurements, with an MLP and the Huber (or MSE) loss. The
    third reference model of the roadmap's Phase 12; its PyTorch twin is
    bench/torch_regression.py.

    {v
    ./scripts/download_wine.sh     # once
    dune exec --profile release bin/regression.exe
    dune exec --profile release bin/regression.exe -- --loss mse --epochs 100
    v}

    Every fifth row is held out for testing. The features are standardised with
    the training rows' mean and standard deviation. The test error is compared
    with the baseline of always predicting the training mean. *)

open Flockml

let file = ref "data/wine/winequality-red.csv"
let target = ref "quality"
let epochs = ref 50
let batch_size = ref 32
let lr = ref 1e-3
let hidden = ref 64
let loss_kind = ref "huber"
let seed = ref 42

let spec =
  [
    ( "--data",
      Arg.Set_string file,
      " CSV file (default data/wine/winequality-red.csv)" );
    ("--target", Arg.Set_string target, " Column to predict (default quality)");
    ( "--epochs",
      Arg.Set_int epochs,
      " Number of epochs (default 50; more overfits)" );
    ("--batch-size", Arg.Set_int batch_size, " Mini-batch size (default 32)");
    ("--lr", Arg.Set_float lr, " Adam learning rate (default 1e-3)");
    ("--hidden", Arg.Set_int hidden, " Hidden units per layer (default 64)");
    ( "--loss",
      Arg.Symbol ([ "huber"; "mse" ], fun s -> loss_kind := s),
      " Training loss (default huber)" );
    ("--seed", Arg.Set_int seed, " Random seed (default 42)");
  ]

module R = Reference_models.Make (Flockml.Default)

let () =
  Arg.parse (Arg.align spec) (fun _ -> ()) "regression [options]";
  let rng = Random.State.make [| !seed |] in
  let names, data = Csv.load !file in
  let features = List.filter (( <> ) !target) (Array.to_list names) in
  let x = Csv.columns (names, data) features
  and y = Csv.columns (names, data) [ !target ] in
  let n = (Tensor.shape x).(0) in
  let test_rows = Array.init ((n + 4) / 5) (fun i -> 5 * i) in
  let train_rows =
    Array.of_list (List.filter (fun i -> i mod 5 <> 0) (List.init n Fun.id))
  in
  (* standardise with the training rows' statistics *)
  let xtr = Tensor.take_rows x train_rows in
  let mean = Tensor.mean_axis xtr 0 and std = Tensor.std ~correction:1 xtr 0 in
  let x = Tensor.div (Tensor.sub x mean) std in
  let data = Dataset.of_tensors x y in
  let train = Dataset.subset data train_rows
  and test = Dataset.subset data test_rows in
  Printf.printf "%d rows, %d features: %d to train on, %d to test\n" n
    (List.length features) train.length test.length;
  let model =
    R.regression_mlp ~rng ~features:(List.length features) ~hidden:!hidden
  in
  let opt = Optim.adam ~lr:!lr (Layer.params model) in
  let loss pred t =
    match !loss_kind with "mse" -> Loss.mse pred t | _ -> Loss.huber pred t
  in
  let errors (src : _ Dataset.source) =
    let xs, ys = src.batch (Array.init src.length Fun.id) in
    let pred =
      Var.value (Var.no_grad (fun () -> Layer.forward model (Var.const xs)))
    in
    let d = Tensor.sub pred ys in
    ( Float.sqrt (Tensor.item (Tensor.mean (Tensor.square d))),
      Tensor.item (Tensor.mean (Tensor.abs d)) )
  in
  for epoch = 1 to !epochs do
    let running = Metrics.Running.create () in
    Dataset.loader ~shuffle:true ~rng ~batch_size:!batch_size train
    |> Seq.iter (fun (xb, yb) ->
        Optim.zero_grad opt;
        let l =
          loss (Layer.forward ~train:true model (Var.const xb)) (Var.const yb)
        in
        Var.backward l;
        Optim.step opt;
        Metrics.Running.add
          ~weight:(float_of_int (Tensor.shape xb).(0))
          running
          (Tensor.item (Var.value l)));
    if epoch mod 10 = 0 || epoch = !epochs then begin
      let rmse, mae = errors test in
      Printf.printf "epoch %4d: train loss %.4f, test RMSE %.4f, MAE %.4f\n%!"
        epoch
        (Metrics.Running.mean running)
        rmse mae
    end
  done;
  let _, ytr = train.batch (Array.init train.length Fun.id)
  and _, yte = test.batch (Array.init test.length Fun.id) in
  let m = Tensor.item (Tensor.mean ytr) in
  let base =
    Float.sqrt
      (Tensor.item (Tensor.mean (Tensor.square (Tensor.add_scalar (-.m) yte))))
  in
  let rmse, mae = errors test in
  Printf.printf "test RMSE %.4f, MAE %.4f (predicting the mean: RMSE %.4f)\n"
    rmse mae base

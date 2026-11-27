# FlockML tutorial

This tutorial goes from tensors to a trained, saved model, in five steps.
Each block is a complete program: put it in `bin/` next to `xor.ml`, add its
name to `bin/dune`'s `names`, and run it with
`dune exec --profile release bin/<name>.exe`. `dune test` extracts, compiles
and runs every block (see `doc/dune`), so they stay in step with the library.

`open Flockml` brings in the default backend, plain OCaml float arrays in
float64. To run the same code on the BLAS backend, write
`open Flockml.Make (Tensor_blas.F32)` instead. Only the first line changes.

## 1. Tensors

A `Tensor.t` is an n-dimensional array of floats, row-major, with NumPy's
broadcasting rules. Operations return new tensors; the few that change a
tensor in place end in `_`.

```ocaml
open Flockml

let () =
  let x = Tensor.reshape (Tensor.arange 12.) [| 3; 4 |] in   (* 0..11 as 3x4 *)
  let row = Tensor.of_array [| 4 |] [| 1.; 0.; -1.; 2. |] in
  let y = Tensor.add x row in                                (* broadcasts over rows *)
  let s = Tensor.sum_axis y 1 in                             (* [| 3 |] *)
  let m = Tensor.matmul x (Tensor.transpose x) in            (* [| 3; 3 |] *)
  let top = Tensor.slice x [ Range (0, 2); Step (0, 4, 2) ] in (* x[0:2, ::2] *)
  Format.printf "%a@.%a@.%a@.%a@." Tensor.pp y Tensor.pp s Tensor.pp m Tensor.pp top
```

Indices are `int array`s, masks are tensors of 0s and 1s (`Tensor.gt x
(Tensor.scalar 5.)`), and `Npy.save` / `Npy.load` exchange arrays with NumPy.

## 2. Autograd

A `Var.t` is a node in a computation graph: a tensor, plus how it was
computed. `Var.param` makes a leaf that wants a gradient; `Var.const` one
that doesn't. Every operation on `Var`s records itself, and `Var.backward`
fills in the gradient of a scalar with respect to every parameter.

```ocaml
open Flockml

let () =
  let w = Var.param (Tensor.of_array [| 2 |] [| 1.; -2. |]) in
  let x = Var.const (Tensor.of_array [| 2 |] [| 3.; 4. |]) in
  (* loss = (w . x)² = (3 - 8)² = 25; d loss / d w = 2 (w . x) x *)
  let loss = Var.square (Var.dot w x) in
  Var.backward loss;
  Format.printf "loss %g@.gradient %a@." (Tensor.item (Var.value loss))
    Tensor.pp (Var.grad_exn w);
  (* the same gradient, functionally *)
  let g = Var.gradient (fun w -> Var.square (Var.dot w x)) (Var.value w) in
  Format.printf "again %a@." Tensor.pp g
```

Gradients add up over several `backward` calls until `Var.zero_grad` (or
`Optim.zero_grad`) clears them. `Var.no_grad (fun () -> ...)` runs code
without recording a graph, for evaluation.

## 3. Layers, including your own

A model is a value of `Layer.t`. The built-in layers cover linear,
convolution, normalisation, pooling, attention, recurrent and embedding
layers. For anything else, write a `Custom` layer: a forward function, and
the parameters and sublayers it uses, with names.

```ocaml
open Flockml

(* A gated linear unit: (x · w1) * sigmoid (x · w2) *)
let glu ~in_dim ~out_dim =
  let value = Layer.linear ~in_dim ~out_dim ()
  and gate = Layer.linear ~in_dim ~out_dim () in
  Layer.custom ~name:"GLU"
    ~children:[ ("value", value); ("gate", gate) ]
    (fun ~train ?rng x ->
      let f l = Layer.forward ~train ?rng l x in
      Var.mul (f value) (Var.sigmoid (f gate)))

let model =
  Layer.sequential
    [
      Layer.linear ~in_dim:4 ~out_dim:16 ();
      Layer.layer_norm [| 16 |];
      Layer.residual (glu ~in_dim:16 ~out_dim:16);
      Layer.linear ~in_dim:16 ~out_dim:3 ();
    ]

let () =
  print_endline (Layer.summary model ~input_shape:[| 8; 4 |]);
  List.iter (fun (name, _) -> print_endline name) (Layer.named_params model)
```

The names (`"2.value.w"`, `"2.gate.b"`...) are what checkpoints use. A model
with several inputs or outputs can be a plain OCaml function over a record
of layers, wrapped in `Layer.custom` when it needs saving.

## 4. A training loop

Training is an explicit loop: batches from a `Dataset.loader`, a forward
pass, a loss, `Var.backward`, and an optimizer step. This one learns to
classify points by the quadrant they lie in.

```ocaml
open Flockml

let () =
  Rng.manual_seed 0;
  let n = 2000 in
  let x = Tensor.rand_uniform ~lo:(-1.) ~hi:1. [| n; 2 |] in
  let a = Tensor.to_array x in
  let labels =
    Array.init n (fun i ->
        match (a.(2 * i) > 0., a.((2 * i) + 1) > 0.) with
        | true, true -> 0 | false, true -> 1 | false, false -> 2 | true, false -> 3)
  in
  let data = Dataset.labelled { Dataset.inputs = x; labels } in
  let model =
    Layer.sequential
      [ Layer.linear ~in_dim:2 ~out_dim:32 (); Layer.relu; Layer.linear ~in_dim:32 ~out_dim:4 () ]
  in
  let opt = Optim.adamw ~lr:1e-2 (Layer.params model) in
  let schedule = Schedule.cosine ~lr:1e-2 ~total:(20 * 32) () in
  let step = ref 0 in
  for epoch = 1 to 20 do
    let loss_avg = Metrics.Running.create () in
    Dataset.loader ~shuffle:true ~batch_size:64 data
    |> Seq.iter (fun (xb, yb) ->
           Optim.set_lr opt (schedule !step);
           incr step;
           Optim.zero_grad opt;
           let loss = Loss.cross_entropy (Layer.forward ~train:true model (Var.const xb)) yb in
           Var.backward loss;
           Optim.step opt;
           Metrics.Running.add loss_avg (Tensor.item (Var.value loss)));
    if epoch mod 5 = 0 then
      Printf.printf "epoch %d: loss %.4f\n" epoch (Metrics.Running.mean loss_avg)
  done;
  let predictions = Metrics.predict (Var.value (Layer.forward model (Var.const x))) in
  Printf.printf "accuracy %.3f\n" (Metrics.accuracy ~predictions ~labels)
```

The loader prepares each batch on a background domain while the previous
step runs, and can apply augmentation (`~transform`, see `Transform`).
`Loss`, `Optim` and `Schedule` have PyTorch's choices: label smoothing,
AdamW with parameter groups, warmup and one-cycle schedules, and so on.

## 5. Saving and loading

`Checkpoint.save` writes a model, and optionally its optimizer, to a
`.safetensors` file; `Checkpoint.load` reads it back into a model of the
same shape.

```ocaml
open Flockml

let () =
  let make () = Layer.sequential [ Layer.linear ~in_dim:3 ~out_dim:2 (); Layer.batch_norm 2 ] in
  let model = make () in
  let opt = Optim.sgd ~lr:0.1 ~momentum:0.9 (Layer.params model) in
  let file = Filename.temp_file "tutorial" ".safetensors" in
  Checkpoint.save ~optim:opt ~metadata:[ ("epoch", "1") ] file model;
  let copy = make () in
  let opt' = Optim.sgd ~lr:0.1 ~momentum:0.9 (Layer.params copy) in
  let report = Checkpoint.load ~optim:opt' file copy in
  assert (report.missing = [] && report.unexpected = []);
  let x = Var.const (Tensor.ones [| 4; 3 |]) in
  assert (Tensor.allclose (Var.value (Layer.forward model x)) (Var.value (Layer.forward copy x)));
  print_endline ("saved and loaded " ^ file);
  Sys.remove file
```

The file is plain safetensors, so PyTorch reads it with
`safetensors.torch.load_file`. FlockML's weights are the transposes of
PyTorch's, and images are NHWC: `scripts/check_torch_interop.py` shows the
conversion for every layer type, in both directions.

## Where next

- [FlockML for PyTorch users](pytorch.md): the FlockML name for common
  `torch` calls.
- The reference models: `bin/train_cifar.ml` (ResNet-20 on CIFAR-10),
  `bin/gpt.ml` (a character-level GPT) and `bin/regression.ml` (regression
  on a CSV file).
- The interface of every module, with `dune build @doc`.

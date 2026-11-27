# FlockML

A small PyTorch-like neural network library written from scratch in OCaml.
It provides n-dimensional tensors, reverse-mode automatic differentiation,
layers, losses, SGD and Adam optimizers, and a complete MNIST training
pipeline. The core uses no numerical libraries, only the OCaml standard
library. An optional second backend, `flockml.blas`, runs the same code on
Bigarrays with BLAS, in float64 or float32.

On MNIST, a two-layer MLP reaches **97.3%** test accuracy after 3 epochs, and a
small CNN reaches **98.2%**. On 8 threads of an Apple M2 Pro:
- The pure-OCaml backend trains about as fast as PyTorch on the same CPU: an
  MLP epoch takes 0.57 s and a CNN epoch 4.1 s.
- The BLAS backend is faster than PyTorch: 0.21 s and 3.1 s in float64,
  0.12 s and 2.3 s in float32.

## Quick start

Requirements: OCaml 5.0 or later, dune, and ounit2 for the tests
(`opam install dune ounit2`). The optional BLAS backend needs nothing extra
on macOS, where it uses Accelerate. On Linux, install OpenBLAS (e.g.
`libopenblas-dev`) and `opam install dune-configurator` to use it.
Without them, the BLAS backend still builds, with a slower portable C matrix
multiply.

```sh
dune build                         # compile everything
dune test                          # run all test suites

dune exec bin/xor.exe              # a tiny MLP learns XOR in under a second

./scripts/download_mnist.sh        # fetch MNIST into data/mnist (~55 MB)
dune exec --profile release bin/train_mnist.exe -- --epochs 3
dune exec --profile release bin/train_mnist.exe -- --model cnn
dune exec --profile release bin/train_mnist.exe -- --backend blas32
```

Use `--profile release` for anything performance-sensitive: it turns on
optimisations that the default dev profile leaves off.

## Example

A model is an ordinary value, and training is an explicit loop:

```ocaml
open Flockml

let () =
  let x = Var.const (Tensor.of_array [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]) in
  let y = Var.const (Tensor.of_array [| 4; 1 |] [| 0.; 1.; 1.; 0. |]) in
  let model =
    Layer.sequential
      [ Layer.linear ~in_dim:2 ~out_dim:8 (); Layer.tanh;
        Layer.linear ~in_dim:8 ~out_dim:1 (); Layer.sigmoid ]
  in
  let opt = Optim.adam ~lr:0.05 (Layer.params model) in
  for _ = 1 to 1000 do
    Optim.zero_grad opt;
    let loss = Loss.mse (Layer.forward model x) y in
    Var.backward loss;
    Optim.step opt
  done
```

`Var.backward` fills in the gradient of every parameter, and `Optim.step`
updates the parameters in place. This is the whole of `bin/xor.ml`, apart from
logging.

The same code runs on the BLAS backend: build the stack for it with
`Flockml.Make`, and open that instead of `Flockml`:

```ocaml
module B = Flockml.Make (Tensor_blas.F32)   (* or Tensor_blas.F64 *)
open B
(* ... exactly the code above ... *)
```

This needs `flockml.blas` in the `libraries` of your dune file.

## What's included

| Module | Contents |
|---|---|
| `Tensor` | Row-major float tensors: NumPy-style broadcasting, reductions along any axis, a blocked, multithreaded `matmul`/`gemm_`, reshape, random initialisation, fused in-place optimizer updates, and `im2col` / `col2im` / max-pool kernels for convolution |
| `Var` | Reverse-mode autodiff over 23 primitive ops, including numerically stable `log_softmax` and `sigmoid`, fused `linear` and `softmax_cross_entropy`, and a `no_grad` mode for inference |
| `Layer` | `linear`, `conv2d`, `max_pool2d`, `relu`, `sigmoid`, `tanh`, `flatten`, `dropout`, `sequential` |
| `Loss` | Mean squared error, softmax cross-entropy, binary cross-entropy |
| `Optim` | SGD (with momentum and weight decay) and Adam (with bias correction) |
| `Init` | Xavier-uniform and He-normal weight initialisation |
| `Dataset` | In-memory datasets, Fisher–Yates shuffling, lazy mini-batches, train/validation splits |
| `Mnist` | IDX file loader for MNIST and Fashion-MNIST |
| `Metrics` | Argmax predictions, accuracy, confusion matrices |
| `Graphviz` | Export any computation graph to DOT for visualisation |
| `Parallel` | A small pool of OCaml 5 domains that the tensor kernels use for data parallelism |
| `Make`, `S`, `Default` | `Flockml.Make (T)` builds all of the above for a tensor backend `T`. The top-level modules are `Default = Make (Tensor)` |
| `Layout` | Shape and broadcasting arithmetic shared by the backends |
| `Tensor_blas` (`flockml.blas`) | The optional backend: `F64` and `F32` tensors on Bigarrays, matmul through BLAS (Accelerate or OpenBLAS), every other kernel in vectorised C |

Every public function is documented in its interface: an `.mli`, an
`*_intf.ml`, or the `module type S` at the top of each functor's file.
Run `dune build @doc` to generate HTML documentation in
`_build/default/_doc/_html`.

## Results

Measured on an Apple M2 Pro, release profile, with Adam at learning rate
1e-3 and batch size 64:

| Model | Data | Parameters | Test accuracy | Epoch, 8 threads | Epoch, 1 thread | BLAS f64 | BLAS f32 |
|---|---|---|---|---|---|---|---|
| MLP 784–128–10 | MNIST, 3 epochs | 101,770 | 97.29% | 0.57 s | 2.1 s | 0.21 s | 0.12 s |
| CNN (2 × conv/pool, linear) | MNIST, 3 epochs | 9,098 | 98.20% | 4.1 s | 14.5 s | 3.1 s | 2.3 s |

Times are wall clock, for training only. The BLAS columns use 8 threads with
Accelerate. That backend reaches the same accuracy in float64 (to the
hundredth of a percent at every epoch), and within 0.05% in float32. The
results don't depend on the thread count: every kernel gives bit-identical
output however many threads run it.
[bench/RESULTS.md](bench/RESULTS.md) has the per-phase breakdown and the
PyTorch comparison.

The CNN reaches a higher accuracy with less than a tenth of the MLP's
parameters, because each convolution filter is shared across the whole image.
After training, `train_mnist` prints a confusion matrix showing which digits
are mistaken for which. For the MLP run above, the most common mistakes were
7 read as 2, 3 and 8 confused with each other, and 9 read as 3 or 4.

### Training options

```
--model mlp|cnn     architecture (default mlp)
--epochs N          passes over the training set (default 3)
--batch-size N      mini-batch size (default 64)
--lr X              learning rate (default 1e-3)
--optim adam|sgd    optimizer; sgd uses momentum 0.9 (default adam)
--limit N           train on only the first N examples, for quick runs
--seed N            random seed (default 42)
--log-every N       print the running loss every N batches (default 100)
--data-dir DIR      location of the IDX files (default data/mnist)
--threads N         threads for the tensor kernels (default: FLOCKML_THREADS,
                    or the recommended domain count capped at 8)
--backend B         ocaml (pure OCaml, float64; the default), blas
                    (Tensor_blas.F64) or blas32 (Tensor_blas.F32)
```

For Fashion-MNIST, run `./scripts/download_mnist.sh fashion` and pass
`--data-dir data/fashion-mnist`. Expect roughly 85–90% accuracy, since it is a
much harder dataset.

## How it works

### Autograd: the graph is an algebraic data type

Every `Var.t` node records its value, its gradient, and the `op` that produced
it:

```ocaml
type op =
  | Leaf
  | Add of t * t
  | Matmul of t * t
  | Relu of t
  | Im2col of { input : t; kh : int; kw : int; stride : int; padding : int }
  | Max_pool2d of { input : t; argmax : int array }
  | ...
```

Forward operations compute their result immediately and record a node, so
the graph is rebuilt on every forward pass ("define by run", as in PyTorch).
`backward` sorts the graph topologically, seeds the loss with gradient 1, and
walks it in reverse. At each node it pattern-matches on the `op` to apply that
operation's chain rule. Each op carries exactly what its backward rule needs:
`Max_pool2d` keeps the index of each window's winner, and `Im2col` keeps the
convolution geometry. Subgraphs built only from constants are never visited,
and gradients for inputs that don't need them are never computed.

Because losses and layers are built from `Var` operations, none of them need
backward code of their own.

### Convolutions reduce to matrix multiplication

Images are NHWC (`[batch; height; width; channels]`). `im2col` copies each
output pixel's receptive field into one row of a matrix. A convolution then
becomes a single `matmul` with the flattened filters, followed by a reshape.
Its backward pass, `col2im`, scatter-adds gradients back into the image, where
overlapping windows accumulate. As a result, `im2col` and `max_pool2d` are the
only primitives that convolutional networks add to the engine.

### Backends are swappable through a functor

Everything above the tensors is written against one signature,
`Tensor_intf.S`. `Flockml.Make (T)` builds the whole stack for a backend `T`:
`Var = Autograd.Make (T)`, then `Layer`, `Loss`, `Optim`, `Dataset` and the
rest over it. The modules you get from `open Flockml` are simply
`Make (Tensor)`.

The optional `flockml.blas` library (in `blas/`) is a second backend:
- `Tensor_blas.F64` and `Tensor_blas.F32` store tensors in Bigarrays.
- `gemm_` calls `cblas_dgemm` / `cblas_sgemm`, from Accelerate on macOS or
  OpenBLAS (or another CBLAS) on Linux. On Apple silicon, Accelerate runs
  matrix products on the AMX matrix units, at 660 GFLOP/s in float64 and
  2.2 TFLOP/s in float32.
- Every other kernel is C, written once for both precisions, vectorised by
  the compiler and split across domains with `Parallel`.
- Tensor memory is recycled through a pool, so a training step reuses the
  previous step's buffers.
- A build step finds the BLAS. With none installed, it falls back to a
  portable C matrix multiply, so the library always builds. Set
  `FLOCKML_BLAS=accelerate|openblas|none` while building to force a choice.

The core library doesn't depend on it.

### Design choices

- **Immutable by default, mutable where it pays.** Tensor operations return
  new tensors. The few functions ending in `_` (`axpy_`, `scale_`, ...) mutate
  in place, so optimizer steps don't allocate a copy of every parameter.
- **Layers are data.** A model is a value like
  `Sequential [Linear ...; Relu; ...]`, and `Layer.forward` is a single pattern
  match.
- **Numerical stability is built in.** `log_softmax` subtracts the row maximum
  before exponentiating; `sigmoid` never evaluates `exp` of a large positive
  number; binary cross-entropy guards its logarithms with an epsilon.
- **Laziness for data.** `Dataset.batches` returns a `Seq.t`, so each
  mini-batch is copied out only when the training loop asks for it. With
  `~reuse:true`, every batch is copied into the same buffer.

### How it is fast

The interfaces stay simple; the speed comes from how the kernels in
`tensor.ml` are written.

- **Matmul** follows the GotoBLAS design, scaled down:
  - C is cut into tiles, and each tile is one parallel task.
  - Each task packs its rows of A and columns of B into contiguous buffers,
    which also absorbs any transpose.
  - A 4×4 micro-kernel keeps 16 accumulators in registers. On arm64,
    `ocamlopt` compiles each multiply-add to a single `fmadd` instruction.
  - This reaches about 15 GFLOP/s on one core and over 100 on eight.
- **No closures in hot loops.** A closure call boxes the float it returns,
  which is one allocation per element. Every kernel is a plain loop over
  `float array`s, with shapes checked once up front and unchecked access
  inside.
- **Fewer, bigger operations.** These are each a single kernel, with no
  intermediate tensors:
  - `Var.linear` (matmul plus bias), and the cross-entropy loss;
  - Adam and SGD updates;
  - the matmul backward pass, which writes weight gradients straight into a
    reused gradient buffer.
- **Branch-free where the data is random.** `relu` and max-pooling choose
  with arithmetic instead of `if`, because a comparison of activations is
  mispredicted about half the time.
- **Multicore without changing results.**
  - `Parallel` splits each kernel's output into disjoint ranges, so each
    output element is computed by one task, in a fixed order.
  - Reductions use fixed-size chunks.
  - So the thread count never changes a result. Set it with
    `FLOCKML_THREADS`; it defaults to at most 8.

## Testing

`dune test` runs 669 tests in seven executables, which can also be run one at
a time. The suites themselves are functors over a backend (`test/suites/`),
so the same tests run on both backends:

| Suite | Covers |
|---|---|
| `test_tensor` | construction, broadcasting rules, reductions, linear algebra, in-place ops |
| `test_autograd` | graph mechanics and a gradient check of every elementwise, matrix and reduction op |
| `test_nn` | initialisation statistics, losses, layers, batching, metrics |
| `test_optim` | SGD, momentum, weight decay and Adam against hand-computed updates |
| `test_conv` | `im2col`/`col2im` adjointness, gradient checks of `im2col`, `max_pool2d` and a full conv layer, graph export |
| `test_kernels` | every fast kernel against a plain reference on awkward shapes, gradient checks of the fused ops, gradient-buffer ownership, and bit-identical results at 1, 2, 3 and 8 threads |
| `test_backends` | all six suites again on `Tensor_blas.F64`, then every op and a few MLP and CNN training steps on both backends, checked to agree (to 1e-11 in float64, 1e-5 in float32) |

The central tool is the **gradient check** in `test/suites/helpers.ml`. It compares
every gradient that autograd computes against a central finite difference,
`(f(x + ε) − f(x − ε)) / 2ε`, so any mistake in a backward rule shows up as a
specific input element with the wrong value.

## Visualising a graph

`bin/xor.exe` writes the computation graph of its first training step to
`xor_graph.dot`. To render it, install Graphviz (`brew install graphviz` or
your package manager) and run:

```sh
dot -Tsvg xor_graph.dot -o xor_graph.svg
```

In the rendered graph, parameters are blue, constant inputs grey and the loss
salmon. Each node shows its operation, its shape, and whether it has received
a gradient.

## Performance

Making FlockML competitive in speed with mainstream CPU frameworks is the
current goal: [ROADMAP.md](ROADMAP.md) has the plan, and
[bench/RESULTS.md](bench/RESULTS.md) has the measured numbers for each phase.

| | Before | 1 thread | 8 threads | BLAS f64 | BLAS f32 | Reference |
|---|---|---|---|---|---|---|
| MLP epoch | 16.6 s | 2.1 s | 0.57 s | 0.21 s | 0.12 s | PyTorch 0.51 s (f64), 0.34 s (f32) |
| CNN epoch | 88 s | 14.5 s | 4.1 s | 3.1 s | 2.3 s | PyTorch 4.30 s (f64, 12 threads) |
| 512² matmul | 1.8 GFLOP/s | 15 GFLOP/s | 106 GFLOP/s | 660 GFLOP/s | 2234 GFLOP/s | NumPy 50 (1 thread), 144 (12) |

The first three columns are the pure-OCaml backend. Phases 1–5 of the
roadmap are done; Phase 6 (graph replay, fusion, GPU) is next.

### Benchmarks

```sh
dune exec --profile release bench/bench_train.exe   # MLP and CNN epochs, per phase
dune exec --profile release bench/bench_matmul.exe  # matmul on the shapes training uses
dune exec --profile release bench/bench_ops.exe     # every other op, with allocation
python3 bench/torch_baseline.py                     # the same models in PyTorch
python3 bench/numpy_baseline.py                     # the same matmuls in NumPy
```

Every benchmark takes `--backend ocaml|blas|blas32`. `bench_train` splits
each training step into data loading, forward pass, backward pass and
optimizer update. It also reports heap allocation and GC
collections per step, and accepts `--model mlp|cnn` and `--steps N` for quick
runs. Every benchmark prints a Markdown table and measures wall-clock time.

### Profiling

Build in release mode, start a long run, and sample it:

```sh
dune build --profile release
./_build/default/bench/bench_train.exe --model cnn &
sample $! 10 -file profile.txt          # macOS; look under "Sort by top of stack"
# Linux: perf record -g ./_build/default/bench/bench_train.exe --model cnn
#        perf report
```

OCaml functions appear as `camlFlockml__Module.function_NNN`, for example
`camlFlockml__Tensor.matmul_1304`. A lot of time in `caml_apply2` or
`caml_alloc*` means closure calls or boxing in a hot loop.

## Layout

```
lib/
  flockml.ml               the library's entry: Make (T), module type S, Default
  tensor/
    tensor_intf.ml         module type S: what a numerical backend must provide
    tensor.ml(i)           default backend: row-major float array + shape
    layout.ml(i)           shape and broadcasting arithmetic, shared by backends
  autograd/
    autograd_intf.ml       module type S: the autograd engine's interface
    autograd.ml            Make (T : Tensor_intf.S): graph ADT + backward pass
  nn/
    init.ml                Xavier / He initialisation
    layer.ml               layers as an ADT (Linear, Conv2d, Relu, Sequential...)
    loss.ml                MSE, cross-entropy, binary cross-entropy
  optim/optim.ml           SGD (+momentum, weight decay), Adam
  data/
    dataset.ml             in-memory dataset, shuffling, lazy mini-batches
    mnist.ml               IDX file loader (MNIST and Fashion-MNIST)
  metrics.ml               accuracy, confusion matrix
  viz/graphviz.ml          export a computational graph to DOT
  parallel/parallel.ml(i)  domain pool: parallel_for over disjoint ranges
blas/                      flockml.blas, the optional backend
  tensor_blas.ml(i)        Tensor_blas.F64 / F32: Bigarray storage, shape checks
  tensor_blas_stubs.c      C stubs: CBLAS gemm, the tensor memory pool
  kernels.h                the C kernels, written once for float and double
  config/                  finds the BLAS at build time
bin/
  xor.ml                   smallest end-to-end example
  train_mnist.ml           MLP / CNN training CLI
test/
  suites/                  one OUnit2 suite per subsystem, as a functor over a backend
  test_*.ml                runs each suite on the default backend
  test_backends.ml         every suite on the BLAS backend, and cross-backend agreement
bench/                     training, matmul and per-op benchmarks; PyTorch/NumPy baselines; RESULTS.md
scripts/download_mnist.sh  dataset download
```

Each module in `nn/`, `optim/`, `data/`, `metrics.ml` and `viz/` is a
functor, with its documented signature `S` in the same file.

## License

MIT. See [LICENSE](LICENSE).

# FlockML

A small PyTorch-like neural network library written from scratch in OCaml:
n-dimensional tensors, reverse-mode automatic differentiation, layers, losses,
SGD/Adam, and an MNIST training pipeline. It uses no numerical libraries, only
the OCaml standard library.

> **Status:** scaffolding. The architecture, types, signatures, data loading
> and tests are in place; the algorithms are `Todo.todo` stubs. See
> [ROADMAP.md](ROADMAP.md) for the implementation plan.

## Quick start

```sh
dune build                         # compile everything
dune test                          # run all test suites
dune exec test/test_tensor.exe     # run one phase's tests
grep -rn "Todo.todo" lib           # list what's left to implement

dune exec bin/xor.exe              # end-to-end sanity check (tiny MLP learns XOR)

./scripts/download_mnist.sh        # fetch MNIST into data/mnist
dune exec --profile release bin/train_mnist.exe -- --epochs 3
dune exec --profile release bin/train_mnist.exe -- --model cnn --limit 10000
```

Use `--profile release` for anything performance-sensitive: it turns on
optimisations that the default dev profile leaves off.

## Layout

```
lib/
  todo.ml                  Todo.todo: the placeholder every stub calls
  tensor/
    tensor_intf.ml         module type S: what a numerical backend must provide
    tensor.ml(i)           default backend: row-major float array + shape
  autograd/
    autograd_intf.ml       module type S: the autograd engine's interface
    autograd.ml            Make (T : Tensor_intf.S): graph ADT + backward pass
    var.ml                 Var = Autograd.Make (Tensor), used everywhere else
  nn/
    init.ml(i)             Xavier / He initialisation
    layer.ml(i)            layers as an ADT (Linear, Conv2d, Relu, Sequential...)
    loss.ml(i)             MSE, cross-entropy, binary cross-entropy
  optim/optim.ml(i)        SGD (+momentum, weight decay), Adam
  data/
    dataset.ml(i)          in-memory dataset, shuffling, lazy mini-batches
    mnist.ml(i)            IDX file loader (MNIST and Fashion-MNIST)
  metrics.ml(i)            accuracy, confusion matrix
  viz/graphviz.ml(i)       export a computational graph to DOT
bin/
  xor.ml                   smallest end-to-end test
  train_mnist.ml           MLP / CNN training CLI
test/                      one OUnit2 executable per phase, plus gradcheck helper
bench/                     matmul benchmark vs NumPy
scripts/download_mnist.sh
```

## Design notes

- **The graph is an algebraic data type.** Every `Var.t` records the `op` that
  produced it (`Add of t * t`, `Matmul of t * t`, `Relu of t`, ...).
  `backward` topologically sorts the graph and pattern-matches on each node's
  op to apply the chain rule.
- **Functors for backends.** Autograd is a functor over `Tensor_intf.S`. A
  Bigarray- or BLAS-backed tensor module can reuse the whole autograd engine
  with `Autograd.Make (My_backend)`.
- **Immutable by default, mutable where it pays.** Tensor ops return new
  tensors. The few `*_` functions (`axpy_`, `scale_`, ...) mutate in place, so
  optimizer steps don't allocate.
- **Layers are data.** A model is a value like `Sequential [Linear ...; Relu;
  ...]`, and `Layer.forward` is a single pattern match.
- **CNNs reduce to matmul.** Images are NHWC. `im2col` turns convolution into
  one matrix product, so the only new differentiable primitives are `im2col`
  and `max_pool2d`.

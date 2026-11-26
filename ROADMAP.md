# Implementation roadmap

Every unimplemented function calls `Todo.todo "Module.name"`, so

```sh
grep -rn "Todo.todo" lib
```

is your live checklist. Each stub has a comment above it with a tag like
`TODO [Phase 2, medium]` and hints. This document covers the order to work in,
the approach for each TODO, and how to tell when a phase is done.

Work through the phases in order, because each one builds on the last. Every
phase has a test executable; a phase is done when its tests pass.

| Phase | Area | Done when |
|---|---|---|
| 1 | Tensor | `dune exec test/test_tensor.exe` passes |
| 2 | Autograd | `dune exec test/test_autograd.exe` passes |
| 3 | Init, losses, layers | `dune exec test/test_nn.exe`: init, losses and layers groups pass |
| 4 | Optimizers | `dune exec test/test_optim.exe` passes **and** `dune exec bin/xor.exe` learns XOR |
| 5 | Data, metrics, MNIST | rest of `test_nn.exe` passes; MLP gets about 97% on MNIST |
| 6 | Stretch: CNN, graphviz, benchmarks | `dune exec test/test_conv.exe` passes; CNN gets about 98-99% |

When everything is done, delete the `(flags ...)` line in `lib/dune` so
unused-variable warnings become errors again.

---

## Phase 1: Tensor (`lib/tensor/tensor.ml`)

The representation is fixed: `{ shape : int array; data : float array }`,
row-major and contiguous. Already provided: constructors, `strides`, `pp`,
`allclose`, and every op that is a one-liner on top of `map`/`map2`.

Suggested order:

1. **`flat_index`** (easy). It powers `get`/`set`. Validate the rank and
   bounds, then compute `sum idx.(i) * strides.(i)`.
2. **`map`, `sum`, in-place ops** (easy). Each is a loop over `data`. The
   in-place ops must check that shapes are *exactly* equal, with no
   broadcasting.
3. **`rand_uniform`, `randn`** (easy). Use Box-Muller for `randn`. Make sure
   the log never sees 0: `1.0 -. Random.State.float rng 1.0` lies in (0, 1].
4. **Broadcasting: `broadcast_shapes` → `broadcast_to` → `map2` →
   `sum_to_shape`** (medium). This is the trickiest part of the phase, and
   it's worth getting fully right because autograd depends on it everywhere.
   - First write a private helper `unravel shape flat -> int array`, the
     inverse of `flat_index`.
   - Then write `source_offset src_shape big_index`: right-align the shapes,
     use index 0 on any axis where the source has size 1, and dot with the
     source strides.
   - `broadcast_to` reads `src.(source_offset ...)` for every output index.
   - `sum_to_shape` uses the same mapping but **adds into** the small tensor.
   - `map2`: if the shapes are equal, do a straight loop. Otherwise broadcast
     both operands, then loop.
5. **`reduce_axis`** (medium). Write it once and derive `sum_axis` and
   `max_axis` from it. View the tensor as `(outer, n, inner)`; this one idea
   handles every axis of every rank. Support negative axes.
   **`argmax_axis`** uses the same walk but tracks the index.
6. **`matmul`, `transpose`, `reshape`, `take_rows`, `one_hot`** (easy to
   medium). For `matmul`, get the naive version passing first, then switch to
   the i-p-j loop order (see the hint).

Pitfall: `tensor.ml` defines `exp`, `log`, `sqrt` and `neg`, which shadow the
Stdlib versions. Inside helpers, always write `Float.exp` etc.

## Phase 2: Autograd (`lib/autograd/autograd.ml`)

Provided: the `op` ADT, node construction (`leaf`, `node`), `parents`,
`op_name`, `accumulate`, `Infix`, the worked example `add`, and its backward
case.

1. **Forward ops** (easy). Each one copies the shape of `add`: compute with
   `T`, then wrap with `node value (Op (...)) [inputs]`.
   - `sum_axis`: normalise negative axes *before* storing them in the op.
   - `log_softmax`: compute the whole stable formula at tensor level and
     produce one node. Do **not** chain Var ops.
2. **`topo_sort`** (medium). Use a post-order DFS with a `Hashtbl` of visited
   ids, and don't descend into nodes that don't require a gradient.
3. **`backward`** (medium). Check that the root is a scalar, seed it with
   ones, walk the nodes in reverse topological order, and call `propagate` on
   each node that has a gradient.
4. **Backward rules in `propagate`** (easy to medium). The formula for each
   case is written beside it. Two rules to remember:
   - Every broadcasting binary op must wrap each input's gradient in
     `T.sum_to_shape _ (shape of input)`, as `Add` does.
   - Gradients have the same shape as the *input*. If a rule produces a
     different shape, it's wrong.
   - Reuse `v.value` (the forward output) where you can: `Exp`, `Sigmoid`,
     `Tanh` and `Log_softmax` all have cheap derivatives in terms of their
     output.

**Debug with gradcheck.** `test/helpers.ml` compares every autograd gradient
against a finite-difference estimate, and its failure message names the
input and element that disagree. Get the forward ops plus `Mul` and `Sum`
working first, because the gradcheck harness uses them to reduce outputs to
a scalar.

## Phase 3: Init, losses and layers (`lib/nn/`)

1. **`Init.xavier_uniform`, `Init.he_normal`** (easy). These are one-liners
   on top of `rand_uniform` / `randn`. The formulas are in `init.mli`.
2. **`Loss.mse`** (easy): `mean ((pred - target)^2)` using Var ops.
3. **`Loss.cross_entropy`** (medium): `-sum(one_hot * log_softmax logits) /
   batch`. Two built-in checks: all-zero logits must give `log(classes)`, and
   the gradient must be `(softmax - one_hot) / batch`.
4. **`Loss.binary_cross_entropy`** (easy). Add `eps` inside the logs.
5. **`Layer.forward`** (medium) is one pattern match over the layer ADT. The
   hint lists each case. Dropout uses *inverted* dropout: scale the kept
   units by `1/(1-p)` during training so evaluation needs no change.
6. **`Layer.params`** (easy). Recurse into `Sequential` with
   `List.concat_map`.

Losses and layers need **no backward code**, because they are compositions of
Var ops that autograd already differentiates.

## Phase 4: Optimizers (`lib/optim/optim.ml`)

The state record types and constructors are provided. You write the two
update rules. Both mutate the parameter tensors in place with `Tensor.axpy_`,
`scale_` and `add_`, and skip parameters whose `Var.grad` is `None`.

1. **`sgd_step`** (easy-medium). Use PyTorch-style momentum:
   `v <- mu * v + g; p <- p - lr * v`. Apply weight decay to a *copy* of the
   gradient.
2. **`adam_step`** (medium). Increment `t` once per step, not once per
   parameter. Bias correction divides by `1 - beta^t`.

The expected values in `test/test_optim.ml` are worked out by hand in its
comments. Use them to debug step by step.

**Milestone:** `dune exec bin/xor.exe` should print a falling loss and
predictions close to `0 1 1 0`. If it does, the full stack works.

## Phase 5: Data, metrics and MNIST

1. **`Dataset.shuffled_indices`** (easy). Use Fisher-Yates and swap with
   `j` in `[0, i]`. Swapping with any index gives a biased shuffle.
2. **`Dataset.batches`** (easy-medium). Return a lazy `Seq.t`, and handle the
   last, smaller batch.
3. **`Dataset.split`**, **`Metrics.accuracy`**, **`Metrics.confusion_matrix`**
   (easy).
4. Run it:
   ```sh
   ./scripts/download_mnist.sh
   dune exec --profile release bin/train_mnist.exe -- --epochs 3
   ```
   Expected results: test accuracy around 95% after epoch 1 and around 97%
   after 3 epochs. Fashion-MNIST (`./scripts/download_mnist.sh fashion`,
   then `--data-dir data/fashion-mnist`) should reach about 85-88%.

   If the loss doesn't fall, check in this order: the init scale, the
   learning rate, whether `zero_grad` is called each step, and whether the
   gradients flow to every parameter (look for `None` grads).

The IDX loader (`mnist.ml`) and the training CLI (`bin/train_mnist.ml`) are
already written.

## Phase 6: Stretch goals

1. **CNN** (hard). Implement `Tensor.im2col`, `col2im`, `max_pool2d` and
   `max_pool2d_backward`. Then add the forward ops `Var.im2col` and
   `Var.max_pool2d` with their backward cases, and finally the `Conv2d` and
   `Max_pool2d` cases of `Layer.forward`.
   - Test `im2col` on the tiny examples in `test_conv.ml` before anything
     else. The column order is (kernel row, kernel column, channel).
   - `col2im` uses the same index walk as `im2col` with `+=` in place of `=`.
     The adjoint test (`<im2col x, y> = <x, col2im y>`) catches almost every
     indexing bug.
   - Then run `--model cnn --limit 10000` and aim for about 98%.
2. **Graph visualizer** (easy-medium). Implement `Graphviz.to_dot`, uncomment
   the line in `bin/xor.ml`, and render the graph with
   `dot -Tsvg xor_graph.dot -o xor_graph.svg`.
3. **Benchmarks.** Compare `dune exec --profile release
   bench/bench_matmul.exe` with `python3 bench/numpy_baseline.py`. Try
   different loop orders, then cache blocking. Extend the benchmark to time
   one MNIST epoch against a PyTorch script.
4. **A second backend through the functor.** Write `Tensor_bigarray`
   satisfying `Tensor_intf.S` with `Bigarray.Array1` storage, and instantiate
   `Autograd.Make (Tensor_bigarray)`. Run the same tests against it.
5. **More ideas:** learning-rate schedules (`Optim.set_lr` already exists),
   gradient clipping, batch norm (an op with running statistics), model
   save/load (serialize `Layer.t`'s parameters), and an explicit stack in
   `topo_sort` for very deep graphs.

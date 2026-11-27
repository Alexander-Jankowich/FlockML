# Roadmap

This roadmap has two parts:
- **Part 1, performance (Phases 1–6).** Train the same models as fast as
  PyTorch and NumPy on the same CPU. Phases 1–5 are done.
- **[Part 2, feature parity (Phases 7–13)](#part-2-feature-parity-with-pytorch-and-numpy).**
  Be able to express the models people build with PyTorch and the array code
  people write with NumPy. Phases 7 and 8 are done.

# Part 1: Performance

The library is functionally complete: every op, layer, loss and optimizer is
implemented and tested, and MNIST trains to 97–98%. The next goal is
**speed**. FlockML should train the same models in roughly the same time as
mainstream CPU frameworks (PyTorch, NumPy-backed code) on the same machine,
without giving up the readable, from-scratch core.

## Where we are

**Phases 1–5 are done (2026-11-26).** The pure-OCaml backend went from a
16.56 s MLP epoch and an 88.45 s CNN epoch (Phase 1) to 0.57 s and 4.10 s on
8 threads (Phase 4). The optional BLAS backend (Phase 5) brings them to
0.21 s and 3.14 s in float64, and 0.12 s and 2.27 s in float32. On the
same machine, PyTorch's best is 0.51 s and 4.30 s in float64. Test accuracy
is unchanged. Phase 6 (stretch goals) is next. Full tables:
[bench/RESULTS.md](bench/RESULTS.md).

| Workload | Phase 1 | Pure OCaml, 8T | BLAS f64 | BLAS f32 | Reference |
|---|---|---|---|---|---|
| 512×512 matmul | 1.82 GFLOP/s | 105.6 GFLOP/s | 660 GFLOP/s | 2234 GFLOP/s | NumPy 50 (1T), 144 (12T) |
| MLP epoch | 16.56 s | 0.57 s | 0.21 s | 0.12 s | PyTorch 0.51 s (f64), 0.34 s (f32) |
| CNN epoch | 88.45 s | 4.10 s | 3.14 s | 2.27 s | PyTorch 4.30 s (f64, 12T), 6.40 s (f32) |

### The Phase 1 baseline

Phase 1 baseline (2026-11-26). Measured on an Apple M2 Pro, release profile,
wall clock. FlockML is single-threaded and float64. The references are
PyTorch 2.0.1 and NumPy 1.24.3 (OpenBLAS), with Adam and batch size 64. The
full tables are in [bench/RESULTS.md](bench/RESULTS.md).

| Workload | FlockML | Reference, 1 thread f64 | Gap |
|---|---|---|---|
| 512×512 matmul | 1.82 GFLOP/s | NumPy: 49.3 GFLOP/s (all threads: 142) | 27× (78×) |
| MLP 784–128–10, one epoch | 16.56 s | PyTorch: 0.51 s (f32: 0.33 s) | 32× |
| CNN (2× conv/pool, linear), one epoch | 88.45 s | PyTorch: 6.62 s | 13× |

The profile (macOS `sample` on `train_mnist.exe`) shows where the time goes:

- **MLP:** about 85% of samples land in `Tensor.matmul`. The rest is closure
  calls (`caml_apply2`), `Array.map2`/`iteri` from elementwise ops, fresh
  allocations (`caml_array_make`), and the Adam update.
- **CNN:** `matmul` is still first, but the matrices are tall and skinny
  (`[50176; 9] × [9; 8]`), so the i-p-j inner loop runs only 8 iterations.
  After that come the broadcasting index map `source_offset` (one `mod` and
  one `/` per axis per element, mostly from the conv bias gradient), closure
  calls through `im2col_walk`, `max_pool2d`, and `transpose` copies in the
  matmul backward pass.

## Targets

| Milestone | Matmul (512, f64) | MLP epoch | CNN epoch |
|---|---|---|---|
| Phase 1 baseline | 1.8 GFLOP/s | 16.6 s | 88 s |
| Phase 2 (pure OCaml, 1 core) | ≥ 6 GFLOP/s | ≤ 5 s | ≤ 30 s |
| Phase 4 (pure OCaml, multicore) | ≥ 25 GFLOP/s | ≤ 2 s | ≤ 10 s |
| Phase 5 (BLAS backend) | within 1.5× of NumPy | within 2× of PyTorch | within 2× of PyTorch |
| **Achieved after Phase 4** | **15.0 (1T), 105.6 (8T)** | **2.08 s (1T), 0.57 s (8T)** | **14.5 s (1T), 4.10 s (8T)** |
| **Achieved after Phase 5** (BLAS f64) | **660 (4.6× NumPy)** | **0.21 s (2.4× faster)** | **3.14 s (1.4× faster)** |

The numbers are goals, not predictions. Phase 1 exists to make every one of
them measurable, and each phase's numbers are recorded in `bench/RESULTS.md`.

## Ground rules

1. **Measure first.** Every optimisation lands with a before/after number from
   the benchmark suite. If something doesn't move a benchmark, revert it.
2. **Stay correct.** `dune test` stays green. Every new fast path is
   property-tested against the existing straightforward implementation on
   random shapes, including sizes of 1, odd sizes, and sizes that aren't
   multiples of the block size. Fused ops get a gradient check.
3. **The pure-OCaml core stays dependency-free.** BLAS, C stubs and
   threading libraries go in separate, optional libraries. The plain `Tensor`
   backend remains the reference implementation that the others are tested
   against.
4. **Biggest gap first.** Matmul is about 85% of the time today, so it goes
   first. Allocation and closure overhead only matter once matmul is fast.

---

## Phase 1: Measurement infrastructure ✅

Done 2026-11-26: the baseline is in [bench/RESULTS.md](bench/RESULTS.md).
Without this, nothing after it can be judged.

1. **Wall-clock timing.** `bin/train_mnist.ml` times epochs with `Sys.time`,
   which is *process CPU time*. Once work is spread across cores, it adds up
   the time of every core and reports a slowdown as the program gets faster.
   Switch to `Unix.gettimeofday` (or `Mtime`), and report training time and
   evaluation time separately.
2. **Benchmark suite** (`bench/`):
   - `bench_matmul`: add the shapes training actually uses, not only squares:
     `[64;784]×[784;128]`, `[784;64]×[64;128]` (MLP dW), `[50176;9]×[9;8]`
     and `[12544;72]×[72;16]` (conv), and their transposed backward forms.
   - `bench_ops`: elementwise ops, broadcasting `add` of a bias, `sum_to_shape`
     to a bias, `im2col`/`col2im`, `max_pool2d`, `log_softmax`, and one Adam
     step over the MLP's parameters. Report ns/element or GB/s.
   - `bench_train`: one MLP epoch and one CNN epoch on a fixed seed, with
     forward, backward and optimizer time listed separately.
   - `bench/torch_baseline.py`: the same two models in PyTorch (float64 and
     float32, 1 thread and all threads), so the comparison is like for like.
3. **Allocation and GC numbers.** Print `Gc.quick_stat` deltas (minor and
   major words allocated, major collections) per training step in
   `bench_train`. Several of the later wins are allocation wins and should
   show up here.
4. **A results log.** `bench/RESULTS.md` holds one dated table per phase,
   along with the machine, compiler and flags used.
5. **A profiling recipe.** Document how to profile in the README: `sample` or
   Instruments on macOS, `perf` on Linux.

**Done when** `dune exec --profile release bench/bench_train.exe` and the
PyTorch script produce comparable tables, and the Phase 1 row of
`RESULTS.md` is filled in.

## Phase 2: Single-core pure OCaml ✅

Done 2026-11-26. Targets met: 15 GFLOP/s, a 2.08 s MLP epoch and a 14.5 s
CNN epoch, all on one core. No dependencies were added. The public API only
grew, with new kernels in `Tensor_intf.S` (`gemm_`, `relu`,
`log_softmax`, `softmax_cross_entropy`, `adam_update_`...) and new ops in
`Var` (`linear`, `softmax_cross_entropy`, `no_grad`). The new
`test/test_kernels.ml` checks every fast path against a plain reference
implementation. Status of each item:

### 2a. Matmul (the 85%)

1. **Remove bounds checks in hot loops.** Validate shapes once at the top of
   each kernel, then use `Array.unsafe_get` and `Array.unsafe_set` inside the
   loops, or compile `tensor.ml` alone with `-unsafe`. Keep the checks in
   `get`/`set`.
2. **Register blocking.** Compute a 4×4 (try 4×8 as well) tile of `C` with 16
   local float accumulators. `ocamlopt` keeps local float `let`s and `ref`s
   unboxed, so the tile stays in registers. This cuts loads and stores per
   FMA by about 4×. The inner loop does no allocation, and none of the
   accumulators may escape the function, or they get boxed.
3. **Cache blocking and packing.** Tile over `k` and `n` so a panel of `B`
   stays in L1/L2, and copy ("pack") that panel into a contiguous scratch
   buffer that is reused between calls.
4. **Transposed variants instead of `transpose` copies.** Add
   `gemm ?trans_a ?trans_b ?alpha ?beta a b ~c` to `Tensor_intf.S`. The
   matmul backward (`autograd.ml`, the `Matmul` case of `propagate`) then
   computes `g · bᵀ` and `aᵀ · g` without allocating a transposed copy, and
   can accumulate straight into an existing gradient (`beta = 1`).
5. **A path for tall, skinny shapes.** When `n` or `k` is small (the conv
   layers, with `n = 8` or `16` and `k = 9` or `72`), use a kernel that
   unrolls over the small dimension instead of the general tiled one.
6. **Try flambda.** Build with an `ocaml-option-flambda` switch at `-O3` and
   record the difference. If it helps, document it as the recommended setup
   rather than requiring it.

**2a status.**
- *Done:* items 1–4. The micro-kernel is 4×4: tried against 4×2 and 4×8,
  4×4 was fastest. Packing does four values per step.
- *Item 5:* covered by two shape-only rules instead of a separate kernel.
  Tiles shrink to 16×16 when there are too few of them for the threads, and
  "split-k" cuts a long `k` into fixed chunks when C is tiny (the conv weight
  gradients).
- *Not done:* item 6 (flambda). It needs a separate opam switch, and the
  speedups so far came from code changes that don't depend on it.

### 2b. Elementwise ops, broadcasting and reductions

1. **No closures in hot loops.** `map2 ( +. )` makes an indirect call per
   element (`caml_apply2` in the profile) and boxes every result. `bench_ops`
   shows `add` on 1M floats allocating 55 MB for an 8 MB result. Write direct
   loops for `add`, `sub`, `mul`, `div`, `neg`, `scale`, `relu` and its
   backward. Keep `map`/`map2` for general use.
2. **A cheaper `relu`.** `Float.max` handles NaN and signed zero (it calls
   `caml_signbit`), which makes `relu` about 2× slower than `add`. Use
   `if x > 0. then x else 0.`.
3. **Fast paths for common broadcasts.** `[m;n] op [n]` (bias add) and
   `[m;n] op [m;1]` (row-wise, used by `log_softmax`) become two nested loops
   with no index arithmetic. `sum_to_shape [m;n] → [n]` (every bias
   gradient) becomes a column-sum loop.
4. **Cheaper general broadcasting.** Replace the per-element `mod`/`div`
   unravelling in `source_offset` with an odometer: increment the
   multi-index and offsets as the loop goes, and treat broadcast axes as
   stride 0.
5. **Specialised reductions.** Give `sum_axis` and `max_axis` on the last axis
   a contiguous inner loop. `reduce_axis` currently goes through a closure
   for every element.

**2b status.**
- *All done.*
- `relu` went further than planned. Even `if x > 0.` was mispredicted about
  half the time on real activations, so both relu kernels are branch-free:
  0.74 ns per element, against 7.9 ns before.
- Two of the planned fast paths (the bias broadcast and the bias gradient)
  got their own direct loops, after measurements showed the generic row walk
  costing about 2× on rows of 8.

### 2c. Convolution kernels

1. **`im2col` without a closure.** In NHWC, each `(ky)` row of a receptive
   field is a run of `kw * c` contiguous floats, apart from padding at the
   edges. Copy it with `Array.blit` instead of calling a closure per element,
   and do the same, accumulating, in `col2im`.
2. **`max_pool2d`.** Special-case 2×2 with stride 2, and hoist the index
   arithmetic out of the window loops.
3. **Optional:** a direct 3×3 convolution kernel for small channel counts,
   which skips materialising the column matrix. Benchmark it against
   im2col + gemm before keeping it.

**2c status.**
- *Done:* items 1 and 2.
  - `im2col` hoists all the bounds work out of interior pixels.
  - `max_pool2d` picks the winning index without a branch and reads the
    value back, which keeps it exact.
- *Not done:* item 3 (direct convolution). im2col is now about 10% of a CNN
  step, so it is a candidate for later.

### 2d. Autograd and optimizer overhead

1. **Accumulate gradients in place.** `accumulate` allocates a new tensor
   (`T.add g0 g`) whenever a node receives a second contribution. Track
   whether a node owns its gradient buffer: copy on the first write if the
   buffer is shared, and use `add_` after that. Upstream `g` is sometimes
   passed straight through (`Add`, `Reshape`), so ownership has to be
   explicit.
2. **Persistent parameter gradients.** `zero_grad` sets the gradient to
   `None`, so every step allocates fresh gradient tensors for every
   parameter. Keep one buffer per parameter, have `zero_grad` call `fill_ 0`,
   and let `gemm ~beta:1` write into it directly.
3. **A fused Adam/SGD step.** `adam_step` makes six passes over each
   parameter and allocates two temporaries (`Tensor.mul g g` and the `map2`
   update). Add a single-pass `Tensor.adam_update_` kernel that does no
   allocation. Do the same for SGD with momentum and weight decay.
4. **A fused cross-entropy op.** `Loss.cross_entropy` materialises a one-hot
   matrix, then multiplies, sums and scales it. Add a `Softmax_cross_entropy`
   op that takes integer labels, computes the loss in one pass, and has the
   closed-form backward `(softmax − one_hot) / batch`. Gradient-check it
   against the current composition.
5. **A fused linear op.** Make `x · w + b` a single node, so the bias
   broadcast and its reduction happen in one specialised kernel.
6. **A no-grad mode.** Add `Var.no_grad (fun () -> ...)`, under which forward
   ops return plain values and record no nodes. Use it in
   `train_mnist.evaluate`, and in a future `Layer.predict`.
7. **Cheaper `topo_sort`.** Replace the `Hashtbl` and list append with a
   visit-epoch field, or a dynamic array. This is small, but it runs on every
   step.

**2d status.** All done.
- *Item 2* keeps `zero_grad` returning `None` (tests and the optimizers rely
  on it). The buffer is set aside and reused by the next backward pass, and
  `gemm_` writes weight gradients straight into it.
- *Item 1* found a real aliasing hazard: a gradient shared through `Add`
  could be changed in place by a later backward call. A node now gives up
  in-place rights when it shares its gradient. There is a regression test.

**Done when** the Phase 2 targets are met, `dune test` passes, and the new
property tests (fast path vs. reference) pass.

## Phase 3: Memory and allocation ✅

Done 2026-11-26, as far as measurements justified. Allocation per MLP step
went from 23 MB to 0.3 MB, and minor collections from 9 per step to almost
none. Most of that came from Phase 2 (no boxing, fused ops, reused gradient
buffers).

The GC stats from Phase 1 should guide this phase. Every intermediate tensor
is currently a fresh major-heap float array.

1. **Free graph values after backward.** Once a node's gradient has been
   propagated, its forward value (unless it is a parameter) and its gradient
   can be dropped. Release them during the backward walk instead of keeping
   the whole graph alive until the next step.
2. **A buffer pool.** Training steps have the same shapes every time, so
   recycle buffers by size: a simple free list keyed on `numel`, which ops
   draw from and which backward returns buffers to. Measure the effect on
   major collections and on step time.
3. **GC tuning.** Try `OCAMLRUNPARAM` settings (`s`, `o`) and
   `Gc.set { space_overhead }` in the training binaries. Document whichever
   setting helps.
4. **Batch copies.** `Dataset.batches` allocates a new input tensor per batch
   through `take_rows`. Add a variant that fills a reused buffer.

**Status.**
- *Done:* item 4, as `Dataset.batches ~reuse:true` and `Tensor.take_rows_`.
  It halves the MLP's data time and cuts its allocation per step from 0.7 to
  0.3 MB. `train_mnist` and `bench_train` use it.
- *Item 3 (GC tuning):* measured with no change needed. `o=80`, `120` (the
  default) and `200` performed the same, and `o=400` was slower.
- *Item 1 (freeing graph values):* not done. `Var.value` and `Var.grad` are
  public, so recycling a node's tensors while the caller may still hold the
  node would silently corrupt them.
- *Item 2 (buffer pool):* not done as a general pool, for the same
  ownership reason. Its safe parts are done instead: per-domain packing
  buffers in `gemm_`, and reused gradient buffers. What the CNN still
  allocates (52 MB per step) is its activations and im2col matrices.
  Reusing those safely needs the static graph replay of Phase 6.

## Phase 4: Multicore with OCaml 5 domains ✅

Done 2026-11-26. Targets met: 105.6 GFLOP/s, a 0.57 s MLP epoch and a
4.10 s CNN epoch on 8 threads. Every op, reductions included, gives
bit-identical results at any thread count, which is stricter than planned.
Reductions split their work into fixed chunks that depend only on the
shape. `test_kernels` checks this for each kernel and for whole CNN
training steps, at 1, 2, 3 and 8 threads.

The work is parallelised by splitting outputs into disjoint ranges. This
needs no locks, and float arrays can be shared across domains.

1. **A minimal domain pool.** Either `domainslib` (as an optional
   dependency), or a ~50-line pool of our own built on `Domain`, `Mutex` and
   `Condition`, with a `parallel_for`. The thread count comes from
   `--threads` or `FLOCKML_THREADS`, and defaults to
   `Domain.recommended_domain_count ()`.
2. **Parallel matmul.** Split `C` into row blocks (or `m×n` tiles for skinny
   shapes) across domains, with each domain running the Phase 2 kernel.
3. **Parallel conv kernels.** Split `im2col`, `col2im` and `max_pool2d` over
   the batch dimension.
4. **Parallel elementwise ops and reductions**, only above a size threshold
   (tune it, around 32k elements), so small tensors don't pay the
   synchronisation cost.
5. **Scaling curve.** `bench_train` reports epoch time for 1, 2, 4 and 8
   threads.

**Done when** the Phase 4 targets are met with all cores, and results are
bit-identical to single-threaded runs for every op except reductions.
Reductions are allowed small floating-point differences, which the tests
check with tolerances.

**Status.**
- *Item 1:* our own pool, `lib/parallel/parallel.ml` (no dependency). Chunks
  are claimed from an atomic counter, so slow cores take fewer. Nested calls
  run sequentially.
- *Thread count:* `FLOCKML_THREADS` or `bench_train --threads`. The default
  is capped at 8, because 10 and 12 threads were slower on this machine's
  8 + 4 cores.
- *Items 2–5:* done.
  - Matmul splits C into tiles.
  - The conv kernels split over images.
  - Elementwise ops run in parallel above about 50 µs of work. Anything
    from 25k to 100k measured the same.
  - The scaling curve is in `RESULTS.md`.
- *False sharing:* adjacent per-thread partial sums shared cache lines, which
  made the conv bias gradient no faster on 8 threads than on 1. The partials
  are now padded onto separate 128-byte lines.

## Phase 5: Optional BLAS / C backend through the functor ✅

This is how FlockML closes the gap to "modern ML library" speed: the same
kernels those libraries call. The `Autograd.Make (T : Tensor_intf.S)` design
already exists for this.

1. **Make the whole stack backend-generic.** At present `Layer`, `Loss`,
   `Optim`, `Init`, `Dataset` and `Metrics` all refer to the concrete
   `Tensor` and `Var`. Wrap them in a single `Flockml.Make (T)` functor, and
   keep today's modules as the default instantiation, so existing code and
   the README example don't change. (An alternative is a dune virtual
   library for the tensor implementation. It costs nothing at run time, but
   it rules out testing two backends in one executable, so the functor is
   preferred.)
2. **`Tensor_blas` in a separate library** (`flockml.blas`):
   - Storage is `Bigarray.Array1` (`float64` first). Bigarrays sit outside
     the OCaml heap and never move, so C code can run on them with the
     runtime lock released and its own threads.
   - `gemm` calls `cblas_dgemm`: Accelerate on macOS, OpenBLAS on Linux,
     found by `dune-configurator`.
   - Elementwise ops, reductions, `im2col` and the fused Adam step are
     C stubs compiled with `-O3`, so clang auto-vectorises them for NEON or
     AVX. vDSP on macOS is another option for these.
3. **float32.** OCaml `float array` is float64 only. Bigarray makes `float32`
   storage straightforward, and PyTorch's own numbers show about 2× from
   float32 on this machine. Add the dtype at the backend level as
   `Tensor_blas.F32` and `Tensor_blas.F64`, not as a runtime tag on each
   tensor.
4. **Cross-backend tests.** Instantiate the test suites over both backends
   and check that they agree within tolerance. Gradient checks stay in
   float64.

**Done when** the Phase 5 targets are met, and the dependency-free build
(`dune build` without the optional library's dependencies) still works.

**Status.** Done 2026-11-26. All three targets are met, by a wide margin:
- 512² matmul: 660 GFLOP/s, against NumPy's 144 on 12 threads.
- MLP epoch: 0.21 s, against PyTorch's 0.51 s.
- CNN epoch: 3.14 s, against PyTorch's 4.30 s.
- Float32 brings the epochs to 0.12 s and 2.27 s.
- Test accuracy in float64 is identical to the pure backend's at every
  epoch.

Item by item:
- *Item 1:* `Flockml.Make (T)` builds the whole stack for a backend
  (`Flockml.S`), and the top-level modules are `Flockml.Default =
  Make (Tensor)`, so existing code and the README example are unchanged.
  Each module (`Layer`, `Optim`, `Dataset`...) is now a functor with its
  documented signature `S` in the same file. Autograd's global state (node
  ids, the `no_grad` flag) is shared by every instance. Broadcasting's shape
  arithmetic moved into `Layout`, which both backends use. Measured, the
  pure backend is exactly as fast as before.
- *Item 2:* `flockml.blas` (`blas/`):
  - Storage is a Bigarray.
  - `gemm_` calls CBLAS. `blas/config/discover` finds Accelerate on
    macOS, or OpenBLAS / another CBLAS elsewhere.
  - Every other kernel is C, written once for both element types in
    `blas/kernels.h`, and split across domains with `Parallel`.
  - Tensor memory comes from a recycling pool, and is charged to the minor
    GC. Without that, the first version's CNN was slower than pure OCaml:
    see RESULTS.md.
  - vDSP wasn't needed: clang vectorises the plain loops.
- *Item 3:* `Tensor_blas.F64` and `Tensor_blas.F32`, as two instances of
  one functor over the Bigarray kind. The C stubs dispatch on the kind.
- *Item 4:* the six test suites are functors over a backend (`test/suites/`).
  `test_backends` runs all of them on `Tensor_blas.F64`. It also computes
  every op, and a few MLP and CNN training steps, on both backends, and
  checks they agree: to 1e-11 in float64, 1e-5 in float32. Gradient checks
  run in float64 only. 669 tests in all.
- *Dependency-free build:* the core library is untouched by `blas/`. With
  no BLAS on the system, flockml.blas uses a portable C matrix multiply.
  Without dune-configurator, `discover` falls back to a simpler version.
  All three configurations were built and passed the full test suite.
- *Not done:* thread control for Accelerate. It uses its own threads,
  limited only by the `VECLIB_MAXIMUM_THREADS` environment variable, and
  `FLOCKML_THREADS` doesn't affect it.

## Phase 6: Stretch

- **Static graph replay.** For a fixed architecture and batch shape, record
  the graph once and replay a precomputed schedule with preallocated
  buffers. This is a tiny step toward what `torch.compile` and XLA do.
- **Elementwise fusion.** Combine chains such as `relu (x · w + b)`, or the
  `binary_cross_entropy` composition, into single loops over a recorded
  graph.
- **GPU via Metal** (macOS) through C/Objective-C stubs behind
  `Tensor_intf.S`.
- **Larger benchmarks.** CIFAR-10 with a deeper CNN, where convolution cost
  dominates and the im2col vs. direct-conv tradeoff matters more.

## Order of work

```
Phase 1 (measure) ─▶ 2a (matmul) ─▶ 2b–2d (overheads) ─▶ 3 (memory)
                                     │
                                     └─▶ 4 (domains) ─▶ 5 (BLAS backend) ─▶ 6
```

Phase 5's functor refactor (step 1) didn't depend on the earlier phases, but
it isn't a performance change on its own, so it was done with the phase that
needed it. Phases 1–5 are done. Phase 6 doesn't depend on Part 2, but it is
better done after Phases 8 and 9 (see Part 2's order of work).

---

# Part 2: Feature parity with PyTorch and NumPy

FlockML is now fast: on this machine it trains MLPs and small CNNs faster
than PyTorch. What it can't do yet is express most of the models people
write in PyTorch, or most of the array code people write in NumPy. This is
the gap, area by area, as of 2026-11-26, before Phases 7 and 8 closed the
first two rows:

| Area | FlockML today | PyTorch / NumPy |
|---|---|---|
| Tensor ops | About a dozen elementwise ops. Sum and max along one axis. 2-D `matmul` and `transpose`, `reshape`, `take_rows` | n-d `permute`, slicing, `cat`/`stack`, `gather`/`scatter`, `where`, comparisons, ~100 elementwise ops, batched matmul, `einsum` |
| Autograd | 23 primitives. `Var` has no `sqrt`, `max_axis`, `mean_axis`, `softmax`, slicing or concatenation | Every tensor op is differentiable. Custom `autograd.Function`, hooks, `torch.func.grad` |
| Layers | 9 constructors in a closed variant, so users can't add their own | An open `nn.Module`. Normalisation, embeddings, attention, recurrent layers |
| Model state | Parameters have no names. No save or load | `named_parameters`, `state_dict`, `torch.save`, safetensors |
| Optimizers | SGD and Adam with L2 weight decay, one learning rate for all parameters | AdamW, RMSprop and others, parameter groups, `lr_scheduler`, `clip_grad_norm_` |
| Losses | MSE, cross-entropy, BCE on probabilities (guarded with an epsilon) | Also L1, Huber, NLL, BCE-with-logits and KL. `reduction=`, class weights, label smoothing |
| Data | In-memory `{ inputs; labels : int array }`, MNIST | Any target type, transforms, worker processes, CIFAR and other loaders, `.npy` files |

## Ground rules for Part 2

Part 1's rules still hold: measure, stay correct, keep the core
dependency-free. On top of them:

1. **Both backends, in the same change.** A new tensor op goes into
   `Tensor_intf.S` and is implemented in `Tensor` and in `Tensor_blas`
   together, and it joins the cross-backend agreement test. The pure version
   is the reference. In `Tensor_blas`, an op that is rarely hot can start as
   a plain C loop. It only gets the vectorised, parallel treatment when a
   profile shows it.
2. **Every differentiable op gets a gradient check**, in float64 on both
   backends. Every kernel keeps the rule from Phase 4: results are
   bit-identical at any thread count.
3. **No speed regressions.** The MLP and CNN numbers in `bench_train` must
   not get slower. New features have to cost nothing for code that doesn't
   use them.
4. **Copies, not views.** Slicing, `permute` and `broadcast_to` return new
   tensors. Strided views are a large part of PyTorch's complexity. They
   would also break two existing design choices: "immutable by default", and
   the `Tensor_blas` rule that its pooled storage never gets Bigarray views.
   When a copy shows up in a profile, the fix is a fused op or an in-place
   `_` variant that writes into a buffer, not a view.
5. **Dtype stays a backend choice.** There is no runtime dtype tag. Indices
   stay `int array`, and masks are 0/1 float tensors, as `relu` and
   `max_pool2d` already use internally.
6. **Existing code keeps compiling** where possible. New parameters are
   optional, and the README example stays as it is. Adding a constructor to
   `Layer.t` or `Var`'s op type can break exhaustive matches in user code.
   That is acceptable, but list it in the change log.
7. **Check against the real thing.** Each phase's behaviour is compared
   against PyTorch or NumPy outputs. A script that is run by hand
   (`test/fixtures/gen.py`) writes those outputs as `.npy` files, which are
   committed. `dune test` reads the committed files, so running the tests
   never needs Python.

## Phase 7: A NumPy-level tensor API ✅

1. **Construction.** `arange`, `linspace`, `eye`, `zeros_like`, `ones_like`,
   `full_like`, and `init shape f`, which fills a tensor from a function of
   the multi-index.
2. **Shape manipulation.**
   - `permute t axes` for n dimensions, and `swapaxes`. `transpose` stays the
     2-D swap.
   - `squeeze`, `unsqueeze`, `flatten ?start ?stop`.
   - `concat ~axis`, `stack ~axis`, `split`, `chunk`.
   - `slice` (start, stop and step per axis), and `narrow ~axis ~start ~len`
     for the common case.
   - `pad` (constant), `flip`, `tile`, `repeat`.
3. **Indexing and masks.**
   - `index_select ~axis idx` generalises `take_rows`.
   - `gather` and `scatter_add` along an axis. Embedding and cross-entropy
     backward passes need them.
   - Comparisons (`eq`, `lt`, `le`, `gt`, `ge`) return 0/1 masks.
   - `where mask a b` and `masked_fill`.
4. **Elementwise math.**
   - `abs`, `sign`, `square`, `rsqrt`, `log1p`, `expm1`, `sin`, `cos`.
   - `clamp ?min ?max`, and broadcasting `minimum` and `maximum`.
   - `tanh`, `sigmoid` and `erf` at the tensor level (GELU needs `erf`). Today
     `tanh` and `sigmoid` exist only in `Var`.
5. **Reductions.**
   - `mean_axis`, `min_axis`, `argmin_axis`, `prod`, `cumsum`, `logsumexp`.
   - `var` and `std`, with `?correction`.
   - Global `max` and `min`.
   - Reduction over several axes at once (`?axes`).
   - All of them use the fixed-chunk scheme, so they stay deterministic.
6. **Linear algebra.**
   - Batched `matmul`: `[| b; m; k |] × [| b; k; n |]`, with broadcasting
     over the leading axes. In `Tensor_blas`, this is a loop of CBLAS calls,
     run in parallel over the batch.
   - `dot`, `outer`, `norm ?p`.
   - Stretch: `einsum`, for a fixed list of patterns that map to batched
     matmul.
7. **Random numbers.**
   - `bernoulli`, `randint`, `permutation`.
   - A global default generator with `manual_seed`, used whenever `?rng` is
     left out, so one call makes a whole run reproducible. Today, leaving out
     `?rng` seeds a new generator from the system. That is a change in
     behaviour, so it needs a note in the change log.
8. **NumPy interop.** `Npy.load` and `Npy.save` for `.npy` files, in float64,
   float32 and int formats, written in pure OCaml. This moves arrays between
   FlockML and NumPy, and it is what the fixtures in ground rule 7 use.
9. **Toplevel printing.** Register `pp` with `#install_printer`, so tensors
   print in utop. Add `to_string ?precision`.

**Done when** every op above is in both backends and is checked against
NumPy fixtures, including empty axes, size-1 axes, negative axes and odd
sizes. `bench_ops` covers the new ops that Phases 9–12 will use in hot
paths: `permute`, `concat`, batched matmul, `gather`, `logsumexp`.

**Status.** Done 2026-11-26. Every item except the `einsum` stretch goal is
in both backends. The test count went from 669 to 1045. Training speed is
unchanged, and the numbers are in [bench/RESULTS.md](bench/RESULTS.md).
- *How it is built:*
  - Every structural op (`permute`, `slice`, `set_slice_`, `flip`, `pad`,
    `concat`) is one call to a strided copy, `copy_walk_`. Each backend
    writes that copy once: OCaml in the pure backend, C in the BLAS one.
  - The ops that are compositions (`logsumexp`, `var`, `tile`, `stack`,
    `clamp`, `matmul`'s 1-D cases, construction, sampling...) are written
    once, in `lib/tensor/tensor_derived.ml`, and both backends include them.
  - So the backends agree to the bit in float64 wherever their primitives
    do.
- *Fixtures:* `test/fixtures/gen.py` writes 106 `.npy` files, and
  `test/suites/numpy_tests.ml` checks 88 cases against them on both
  backends. Every new op is also in the cross-backend agreement tests and
  the thread-count tests.
- *Changes from the plan:*
  - `flatten` takes `?first ?last` (inclusive axes) instead of
    `?start ?stop`.
  - `gather` and `scatter_add` take a flat `int array`, from whose length
    the size along the axis follows, since there are no integer tensors.
  - `permutation` lives in the new `Rng` module, with `manual_seed`.
  - `matmul` itself became batched and accepts 1-D operands, as
    `numpy.matmul` does. `bmm` adds the transposed forms.
- *Cold paths in `Tensor_blas`:* `prod_axis`, `cumsum`, `index_add`,
  `gather` and `scatter_add` go through the default backend, like `map`.
  `gather` costs 30× its pure-OCaml time there, so it needs a C kernel
  before a training loop uses it.
- *A bug found by the edge-case tests:* every empty OCaml `float array` is
  the same value, so `gemm_`'s check that `c` doesn't share storage with `a`
  or `b` fired on empty operands. A `[0; k]` matmul raised. It is fixed.

## Phase 8: Complete autograd ✅

1. **A differentiable `Var` version of every Phase 7 op that has a
   gradient.** That includes:
   - `sqrt`, `abs`, `clamp`, `minimum` and `maximum`.
   - `max_axis` and `min_axis`. The gradient goes to the winning element, and
     the first one wins ties, as in `max_pool2d`.
   - `mean_axis`, `var`, `softmax`, `logsumexp`.
   - `permute`, `squeeze`, `unsqueeze`, `broadcast_to`, `pad`.
   - `slice`, `concat`, `stack`, `split`, `index_select`, `gather`, `where`.
   - Batched `matmul`.
2. **Custom differentiable ops**, the equivalent of
   `torch.autograd.Function`:
   `Var.custom ~name ~inputs ~value ~backward`, where `backward` maps the
   upstream gradient to one optional gradient per input. It becomes a
   `Custom` case of the op type, and Graphviz shows its name. It has to
   follow the gradient-ownership rule: a custom backward never gets in-place
   rights to a gradient it shares.
3. **Freezing.** `Var.set_requires_grad p false`, for fine-tuning and frozen
   feature extractors. The optimizers already skip parameters that have no
   gradient.
4. **Gradient utilities.** `clip_grad_norm` and `clip_grad_value` over a
   list of parameters, and the global gradient norm, for logging. Each is one
   fused pass per parameter.
5. **Anomaly detection.** `Var.detect_anomaly (fun () -> ...)` checks every
   forward value and every backward gradient for NaN and infinity. On the
   first bad value, it raises an exception that names the op and the shape.
   It is off by default and costs nothing when off.
6. **A functional API**, in the style of `torch.func` and JAX:
   `Var.grad f x` and `Var.value_and_grad f x`, for
   `f : Var.t -> Var.t` and a tensor `x`. They wrap `param`, `backward` and
   `grad_exn`. First-order only.
7. **Gradient hooks** (optional): `Var.register_hook v (fun g -> g')`, for
   per-parameter clipping or logging. Only add them if Phases 9–12 need them.
   They complicate Phase 6's graph replay.
8. **Stretch: higher-order gradients.** A `~create_graph:true` backward pass,
   which needs every backward rule written in `Var` ops instead of tensor
   kernels. Also forward-mode `jvp`. Only do this if a concrete use appears,
   such as a gradient penalty or Hessian-vector products.

**Done when** every new `Var` op passes a gradient check on both backends.
A user-written `gelu`, made with `Var.custom`, passes a gradient check. The
README documents `custom` and the functional API.

**Status.** Done 2026-11-26. Items 1–6 are done, and 7 and 8 are not
(optional and stretch).
- *Item 1:*
  - 59 new gradient checks, which run on both backends.
  - There is also a causal-attention block (`matmul`, `swapaxes`,
    `masked_fill`, `softmax`) whose gradients are compared across the two
    backends.
  - `matmul` of an n-D tensor by a matrix is flattened to one 2-D product,
    so a linear layer over a sequence still writes its weight gradient in
    place.
  - `prod_axis` has no `Var` version yet.
- *Item 2:* `Var.custom`. Its gradients are stored as shared, and the node
  gives up in-place rights to its own gradient, so a backward rule may
  return `g` itself. A test checks that the upstream gradient isn't changed.
- *Items 3–5:*
  - `set_requires_grad`, `grad_norm`, `clip_grad_norm`, `clip_grad_value`
    and `detect_anomaly` are in `Var`. They need the gradient-ownership
    flag, and clipping a shared gradient copies it first.
  - The norms use existing kernels, not a new fused pass.
  - `detect_anomaly` checks each forward value when its node is made, and
    each gradient after each backward step.
- *Item 6:* the function is called `Var.gradient`, because `Var.grad`
  already reads a node's gradient. `value_and_grad` only lets nodes created
  during the call take part in the backward pass (their ids are at least the
  input's id), so the parameters `f` uses get no gradients.

## Phase 9: Open layers and model state

The closed `Layer.t` variant is the main reason a user can't write their own
model in FlockML.

1. **User-defined layers.**
   - Keep the variant. "Layers are data" is a stated design choice, and
     `forward`, `params` and `to_string` keep working by pattern matching.
   - Add an escape hatch: `Custom of { name; params; buffers; forward }`,
     where `forward : train:bool -> ?rng:Random.State.t -> var -> var`.
   - Add the combinator `Residual of t`, which computes `x + f x`.
   - Document the other pattern too: a model can be a plain OCaml function
     over a record of layers, which covers models with several inputs and
     outputs.
2. **Named parameters and buffers.** `named_params` returns
   `(string * var) list` with paths such as `"0.w"` or `"block1.conv.b"`.
   Sequential layers are named by their index, and custom layers by their
   name. Buffers hold state that isn't trained, such as BatchNorm's running
   statistics. They are saved with the model, but the optimizer never sees
   them.
3. **Checkpoints in the safetensors format:**
   - The file is an 8-byte header length, then a JSON header mapping each
     name to its dtype, shape and offsets, then the raw little-endian data.
   - It needs no dependencies: a small JSON writer, plus a parser for just
     that header.
   - PyTorch reads it through the `safetensors` package, so weights move
     between FlockML and PyTorch in both directions.
   - `load` reports any missing, unexpected or mismatched names and shapes,
     and converts between float64 and float32.
   - Optimizer state goes in the same file, under its own prefix.
4. **New layers.** Each gets a gradient check, a `to_string`, and an entry
   in the README table.
   - *Activations:* `leaky_relu`, `gelu`, `silu`, `softplus`, `elu`,
     `softmax`, `log_softmax`, `identity`.
   - *Normalisation:* `batch_norm` (1-D and 2-D, with running statistics,
     and different behaviour with `~train`), `layer_norm`, `group_norm`.
     Fused forward and backward kernels come later, once a profile shows
     them.
   - *Pooling:* `avg_pool2d`, `global_avg_pool`, and padding for
     `max_pool2d`.
   - *Convolution:* rectangular kernels, `dilation`, `groups` (depthwise),
     `conv1d` and `conv_transpose2d`. The layout stays NHWC.
   - *Embedding:* `embedding ~num ~dim`, which takes an `int array` and
     scatter-adds in its backward pass.
   - *Sequence:* `rnn`, `lstm` and `gru` cells and layers, and
     `multi_head_attention` with an optional causal mask, all built from
     `Var` ops. A fused attention kernel comes later.
5. **Train and eval mode.** Keep the explicit `?train` flag. It is clearer
   than PyTorch's mutable `.train()` and `.eval()`, and BatchNorm only needs
   to read it.
6. **Model summary.** `Layer.summary model ~input_shape` prints each layer's
   output shape and parameter count, like `torchinfo`. It runs a `no_grad`
   forward pass on zeros.
7. **More initialisation.** `kaiming_uniform` (PyTorch's default for
   `Linear`), `xavier_normal`, `orthogonal`, `constant`, and an `?init`
   argument on the constructors. The default stays He-normal, so existing
   results don't change.

**Done when:**
- A ResNet-style CIFAR-10 model and a small transformer can both be written
  with the public API alone.
- On both backends, saving and then loading a checkpoint gives bit-identical
  outputs.
- A FlockML checkpoint loads in PyTorch and gives the same logits. A script
  in `scripts/` checks this; `dune test` doesn't run it.

## Phase 10: Losses, optimizers and schedules

1. **Losses.**
   - A `?reduction` argument on every loss: `` `Mean ``, `` `Sum `` or
     `` `None ``.
   - New losses: `l1`, `smooth_l1` / `huber`, `nll` (on log-probabilities),
     `kl_div`.
   - `bce_with_logits`, a fused op that is stable without an epsilon. It
     becomes the recommended way to do binary classification.
   - `cross_entropy ?label_smoothing ?weights`, and logits with extra axes
     (`[| b; t; c |]` for sequences).
2. **Optimizers.** Each one is a fused `_update_` kernel in both backends.
   - `adamw`, with decoupled weight decay. Today's `adam ~weight_decay` adds
     L2 to the gradient, which is PyTorch's `Adam`, not `AdamW`.
   - Nesterov momentum for SGD, `rmsprop`, `adagrad`.
3. **Parameter groups.** A different learning rate and weight decay for each
   group, for example no decay on biases and normalisation weights.
4. **Learning-rate schedules.** A `Schedule` module of pure functions from
   step to learning rate: step decay, exponential, cosine, linear warmup and
   one-cycle. `reduce_on_plateau` is the one stateful schedule. A schedule is
   applied with `Optim.set_lr opt (sched step)`, which is simpler than
   PyTorch's scheduler objects.
5. **Optimizer state in checkpoints**, through Phase 9's format, so a
   training run can stop and resume exactly.
6. **Gradient accumulation** already works, because gradients add up until
   `zero_grad`. Document it, with an example.

**Done when** every loss matches PyTorch's value and gradient, and every
optimizer matches PyTorch's parameters after 10 steps on a fixed problem.
Both are checked with fixtures, as in ground rule 7. A resumed run gives the
same result as one that never stopped.

## Phase 11: Data pipeline

1. **General datasets.** Targets can be a tensor (for regression,
   multi-label and sequences), not only an `int array`. A dataset signature
   with `length` and `get : int -> input * target` allows lazy, on-disk
   datasets. Today's in-memory dataset becomes one instance of it, and MNIST
   doesn't change.
2. **A data loader.**
   - `batches ?drop_last ?collate`.
   - Prefetching: the next batch is prepared on a separate domain while the
     current step runs. `Parallel` runs nested calls sequentially, so the
     prefetch domain has to stay outside the kernel pool.
3. **Transforms.** Composable functions that take an `rng`: normalise,
   random crop with padding, horizontal flip. They apply to a whole batch, in
   place where possible.
4. **Loaders.** CIFAR-10 (its binary format), CSV files of floats, and `.npy`
   (from Phase 7). A download script for each, like MNIST's.
5. **Metrics.** Top-k accuracy, precision, recall and F1 per class, and a
   running average for the loss.

**Done when** CIFAR-10 trains with augmentation while data loading overlaps
compute. `bench_train`'s data time per step should be close to zero.

## Phase 12: Reference models and documentation

These models are the end-to-end test that the API is enough. Each one gets
an example in `bin/`, a results row in the README, and a PyTorch version in
`bench/`, so accuracy and speed are compared like for like.

1. **ResNet-20 on CIFAR-10.** It uses convolution, batch norm, residual
   blocks, augmentation, a cosine schedule, and SGD with Nesterov momentum.
   This replaces Phase 6's "larger benchmarks" item.
2. **A character-level GPT on Tiny Shakespeare.** It uses embeddings, layer
   norm, causal multi-head attention, GELU, AdamW with warmup, gradient
   clipping, checkpoints and sampling.
3. **A small regression model on a CSV dataset**, using MSE or Huber loss.
   It shows the non-classification path.
4. **Documentation:**
   - Publish the odoc site to GitHub Pages.
   - A tutorial in `doc/`: tensors, then autograd, then a custom layer, a
     training loop, and saving and loading.
   - "FlockML for PyTorch users": a table that maps common `torch` calls to
     FlockML.

**Done when** models 1 and 2 reach accuracy (or loss) within noise of their
PyTorch versions, and their epoch times are recorded in `RESULTS.md`.

## Phase 13: Packaging and tooling

1. **CI on GitHub Actions**, for three setups: macOS with Accelerate, Linux
   with OpenBLAS, and Linux with no BLAS. Each runs `dune build`, `dune test`
   and `dune build @doc`. Formatting is checked only on changed files,
   because ocamlformat reflows untouched code.
2. **An opam release** of `flockml`, with `flockml.blas` in the same package
   as today.
3. **A toplevel setup**: `dune utop` with the printers from Phase 7
   installed, documented in the README.
4. **Stretch: ONNX export** of `Layer` models. This needs a protobuf writer,
   which is a lot of work for the benefit.

## Order of work for Part 2

```
13.1 (CI) ─▶ 7 (tensor API) ─▶ 8 (autograd) ─▶ 9 (layers, state) ─▶ 12 (reference models)
                  │                              │
                  │                              └─▶ 10 (losses, optimizers, schedules)
                  └─▶ 11 (data), alongside 8–10
```

- CI comes first, because it is small and protects everything after it.
- Phases 7–9 are the critical path. Most of Phase 9's layers are
  compositions of Phase 7 and 8 ops.
- Phase 10 needs Phase 9 for named parameters (parameter groups) and
  checkpoints, but nothing else.
- Phase 11 only needs Phase 7's `.npy` support, so it can run alongside the
  others.
- Phase 6's graph replay and fusion should wait until Phase 9 is done. Then
  the set of ops and layers it must replay is known, including `Custom`
  layers and ops.

Phases 7 and 8 are done. CI (13.1) isn't set up yet. Phase 9 is next.

# Performance roadmap

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
needed it. Phases 1–5 are done; Phase 6 is next.

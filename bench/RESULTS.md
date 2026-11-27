# Benchmark results

Each section is one dated run of the benchmark suite, one per roadmap phase
(see [ROADMAP.md](../ROADMAP.md)). All runs use the release profile. To
reproduce:

```sh
dune exec --profile release bench/bench_train.exe     # end-to-end epochs
dune exec --profile release bench/bench_matmul.exe    # matmul shapes
dune exec --profile release bench/bench_ops.exe       # everything else
python3 bench/torch_baseline.py                       # PyTorch reference
python3 bench/numpy_baseline.py                       # NumPy matmul reference
OPENBLAS_NUM_THREADS=1 python3 bench/numpy_baseline.py  # ...on one thread
python3 bench/torch_ops.py                            # PyTorch, Phase 7 ops
```

Every FlockML benchmark takes `--backend ocaml|blas|blas32` (Phase 5): the
pure-OCaml backend (the default), or `Tensor_blas.F64` / `F32`.

---

## Phase 1 baseline (2026-11-26)

**Machine:** Apple M2 Pro (12 cores), macOS. **FlockML:** OCaml 5.3.0, no
flambda, single-threaded, float64. **References:** PyTorch 2.0.1, and
NumPy 1.24.3 with OpenBLAS. MNIST, batch size 64, Adam (lr 1e-3).

### Headline

| Workload | FlockML | Reference (1 thread, f64) | Gap |
|---|---|---|---|
| MLP epoch | 16.56 s | PyTorch 0.51 s | 32× |
| CNN epoch | 88.45 s | PyTorch 6.62 s | 13× |
| 512² matmul | 1.82 GFLOP/s | NumPy 49.3 GFLOP/s (all threads: 142) | 27× (78×) |

### Training, per step (ms, wall clock)

| | data | fwd | bwd | optim | step | epoch s | MB alloc/step | minor GC/step | major GC/step |
|---|---|---|---|---|---|---|---|---|---|
| FlockML MLP | 0.126 | 7.528 | 7.812 | 2.189 | 17.655 | 16.56 | 23.17 | 9.25 | 0.010 |
| PyTorch MLP f64 1T | 0.047 | 0.126 | 0.089 | 0.284 | 0.547 | 0.51 | | | |
| PyTorch MLP f32 1T | 0.033 | 0.075 | 0.054 | 0.186 | 0.348 | 0.33 | | | |
| FlockML CNN | 0.053 | 41.427 | 52.619 | 0.199 | 94.299 | 88.45 | 152.04 | 37.79 | 0.184 |
| PyTorch CNN f64 1T | 0.083 | 3.123 | 3.665 | 0.186 | 7.057 | 6.62 | | | |
| PyTorch CNN f64 12T | 0.096 | 1.783 | 2.965 | 0.331 | 5.175 | 4.85 | | | |
| PyTorch CNN f32 1T | 0.039 | 2.457 | 4.253 | 0.096 | 6.845 | 6.42 | | | |

Epoch times are per-step time × 938 steps. With 12 threads, PyTorch is
*slower* on the MLP (0.74 s in f64) than with one thread: these matrices are
too small to be worth splitting across cores. That is worth remembering for
Phase 4.

### Matmul (GFLOP/s)

The FlockML backward rows (`a^T b`, `a b^T`) include the transposed copy that
autograd makes today. NumPy's transposes are free views.

| case | m×k×n | form | FlockML | NumPy 1T | NumPy 12T |
|---|---|---|---|---|---|
| square 64 | 64×64×64 | a b | 1.85 | 23.97 | 22.47 |
| square 128 | 128×128×128 | a b | 1.83 | 39.48 | 59.88 |
| square 256 | 256×256×256 | a b | 1.82 | 40.29 | 118.46 |
| square 512 | 512×512×512 | a b | 1.82 | 49.30 | 142.41 |
| mlp fc1 fwd | 64×784×128 | a b | 1.80 | 47.28 | 48.49 |
| mlp fc1 dW | 784×64×128 | a^T b | 1.81 | 49.01 | 64.23 |
| mlp fc2 fwd | 64×128×10 | a b | 1.88 | 24.73 | 24.13 |
| mlp fc2 dW | 128×64×10 | a^T b | 1.66 | 24.13 | 24.13 |
| mlp fc2 dX | 64×10×128 | a b^T | 1.78 | 29.56 | 30.01 |
| cnn conv1 fwd | 50176×9×8 | a b | 1.80 | 14.03 | 20.10 |
| cnn conv1 dW | 9×50176×8 | a^T b | 1.61 | 23.15 | 8.03 |
| cnn conv2 fwd | 12544×72×16 | a b | 1.86 | 38.67 | 41.34 |
| cnn conv2 dW | 72×12544×16 | a^T b | 1.71 | 41.47 | 18.06 |
| cnn conv2 dcols | 12544×16×72 | a b^T | 1.79 | 29.55 | 21.37 |
| cnn fc fwd | 64×784×10 | a b | 1.88 | 26.67 | 14.34 |

### Other ops (FlockML)

| op | µs/call | ns/elem | KB alloc/call |
|---|---|---|---|
| add [64;128] + [64;128] | 34.1 | 4.16 | 448.1 |
| add [1M] + [1M] | 4455.1 | 4.46 | 54687.6 |
| mul [1M] * [1M] | 4205.9 | 4.21 | 54687.6 |
| scale [1M] | 3099.0 | 3.10 | 39062.6 |
| exp [1M] | 9562.0 | 9.56 | 39062.6 |
| relu fwd [1M] | 7905.0 | 7.91 | 23437.6 |
| relu bwd [1M] | 7927.9 | 7.93 | 39062.6 |
| bias add [64;128] + [128] | 118.0 | 14.41 | 448.7 |
| bias add [50176;8] + [8] | 5757.1 | 14.34 | 21952.7 |
| row sub [64;10] - [64;1] | 9.8 | 15.27 | 35.7 |
| bias grad [64;128] → [128] | 53.2 | 6.49 | 129.6 |
| bias grad [50176;8] → [8] | 2630.9 | 6.55 | 6272.7 |
| broadcast_to [128] → [64;128] | 61.0 | 7.45 | 192.6 |
| sum_axis [64;10] axis -1 | 3.8 | 5.96 | 30.7 |
| max_axis [64;10] axis -1 | 3.1 | 4.84 | 20.7 |
| sum_axis [64;128] axis 0 | 31.0 | 3.78 | 385.3 |
| log_softmax [64;10] | 30.0 | 46.94 | 150.4 |
| transpose [784;64] | 87.0 | 1.73 | 392.1 |
| im2col [64;28;28;1] 3×3 | 4015.0 | 8.89 | 3528.2 |
| im2col [64;14;14;8] 3×3 | 5268.1 | 5.83 | 7056.2 |
| col2im → [64;14;14;8] 3×3 | 2579.0 | 2.86 | 784.2 |
| max_pool2d [64;28;28;8] 2×2 | 2417.1 | 6.02 | 1568.1 |
| max_pool2d [64;14;14;16] 2×2 | 1253.1 | 6.24 | 784.1 |
| max_pool2d bwd [64;28;28;8] | 606.1 | 1.51 | 3136.2 |
| max_pool2d bwd [64;14;14;16] | 237.0 | 1.18 | 1568.2 |
| adam step (MLP, 101,770 params) | 2157.0 | 21.19 | 19083.8 |
| sgd+momentum step (MLP) | 722.9 | 7.10 | 4771.1 |
| take_rows 64 of [60000;784] | 18.1 | 0.36 | 392.2 |

### What the numbers say

- **Matmul runs at ~1.8 GFLOP/s whatever the shape.** It is limited by its
  inner loop, not by cache behaviour, so register blocking and removing
  bounds checks (Phase 2a) come before cache blocking. On one core NumPy is
  8–27× faster across the shapes that training uses. The smallest gap is the
  skinny conv1 forward pass (`k = 9`, `n = 8`), where NumPy's BLAS also
  struggles.
- **Elementwise ops allocate 3–7× their output size.** `add` on 1M floats
  allocates 55 MB for an 8 MB result. Every value returned from the closure
  passed to `Array.map`/`map2` is boxed, then copied. Direct loops (Phase 2b)
  remove both the boxing and the indirect call.
- **`relu` is ~2× slower than `add`.** `Float.max` handles NaN and signed
  zero (the profile shows `caml_signbit` calls). A plain
  `if x > 0. then x else 0.` is enough.
- **Broadcasting costs ~3.5× a same-shape op** (14 vs 4 ns/elem) because of
  the per-element `mod`/`div` in `source_offset`. The CNN's conv bias add
  alone takes 5.8 ms per call; PyTorch's whole CNN forward pass takes 3.1 ms.
- **Adam costs 7.7× PyTorch's update** (2.19 vs 0.28 ms per step) and
  allocates 19 MB per step for 101,770 parameters, about 190 bytes per
  parameter.
- **The CNN step allocates 152 MB** and runs 38 minor collections per step.
  The allocation work in Phases 2d and 3 should show up clearly here.
- **Data loading doesn't matter yet** (0.13 ms out of 17.7 ms).

---

## Phases 2–4: single-core kernels, memory, multicore (2026-11-26)

Same machine, compiler and references as the Phase 1 baseline. FlockML is
still float64; "8T" means `FLOCKML_THREADS=8`, which is now the default here.
PyTorch and NumPy numbers are the ones from Phase 1.

### Headline

| Workload | Phase 1 | FlockML 1T | FlockML 8T | PyTorch / NumPy 1T | PyTorch / NumPy all threads |
|---|---|---|---|---|---|
| MLP epoch | 16.56 s | 2.08 s | **0.57 s** | 0.51 s | 0.74 s (12T) |
| CNN epoch | 88.45 s | 14.48 s | **4.10 s** | 6.62 s | 4.85 s (12T) |
| 512² matmul | 1.82 GFLOP/s | 14.97 | **105.6** | 49.3 | 142 (12T) |

Every Phase 2 and Phase 4 target in the roadmap is met. Test accuracy is
unchanged: 97.29% for the MLP and 98.20% for the CNN after 3 epochs, the
same as before to the hundredth of a percent.

### Training, per step (ms, wall clock)

| | data | fwd | bwd | optim | step | epoch s | MB alloc/step | minor GC/step | major GC/step |
|---|---|---|---|---|---|---|---|---|---|
| MLP, Phase 1 | 0.126 | 7.528 | 7.812 | 2.189 | 17.655 | 16.56 | 23.17 | 9.25 | 0.010 |
| MLP 1T | 0.032 | 0.900 | 0.959 | 0.327 | 2.218 | 2.08 | 0.30 | 0.01 | 0.000 |
| MLP 8T | 0.045 | 0.239 | 0.248 | 0.079 | 0.611 | 0.57 | 0.30 | 0.01 | 0.000 |
| CNN, Phase 1 | 0.053 | 41.427 | 52.619 | 0.199 | 94.299 | 88.45 | 152.04 | 37.79 | 0.184 |
| CNN 1T | 0.043 | 6.455 | 8.904 | 0.030 | 15.432 | 14.48 | 51.60 | 0.13 | 0.129 |
| CNN 8T | 0.049 | 1.803 | 2.487 | 0.036 | 4.375 | 4.10 | 51.50 | 0.13 | 0.130 |

### Scaling with threads (epoch seconds, 300 timed steps)

| threads | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| MLP | 2.10 | 1.21 | 0.77 | 0.59 | 0.66 |
| CNN | 14.61 | 8.53 | 5.66 | 4.24 | 4.64 |

8 threads is fastest on this 8 performance + 4 efficiency core chip. The
efficiency cores hold back the synchronous kernels, so the default thread
count is capped at 8 (`FLOCKML_THREADS` overrides it).

### Matmul (GFLOP/s)

Backward rows now run `Tensor.gemm_` with `~trans_a`/`~trans_b`, as autograd
does, so no transposed copy is included.

| case | m×k×n | form | Phase 1 | FlockML 1T | FlockML 8T | NumPy 1T | NumPy 12T |
|---|---|---|---|---|---|---|---|
| square 64 | 64×64×64 | a b | 1.85 | 13.41 | 16.79 | 23.97 | 22.47 |
| square 128 | 128×128×128 | a b | 1.83 | 14.66 | 55.15 | 39.48 | 59.88 |
| square 256 | 256×256×256 | a b | 1.82 | 15.03 | 93.95 | 40.29 | 118.46 |
| square 512 | 512×512×512 | a b | 1.82 | 14.97 | 105.64 | 49.30 | 142.41 |
| mlp fc1 fwd | 64×784×128 | a b | 1.80 | 15.00 | 73.80 | 47.28 | 48.49 |
| mlp fc1 dW | 784×64×128 | a^T b | 1.81 | 14.43 | 79.35 | 49.01 | 64.23 |
| mlp fc2 fwd | 64×128×10 | a b | 1.88 | 9.04 | 7.81 | 24.73 | 24.13 |
| mlp fc2 dW | 128×64×10 | a^T b | 1.66 | 9.54 | 7.81 | 24.13 | 24.13 |
| mlp fc2 dX | 64×10×128 | a b^T | 1.78 | 10.91 | 9.04 | 29.56 | 30.01 |
| cnn conv1 fwd | 50176×9×8 | a b | 1.80 | 7.80 | 25.53 | 14.03 | 20.10 |
| cnn conv1 dW | 9×50176×8 | a^T b | 1.61 | 6.21 | 38.65 | 23.15 | 8.03 |
| cnn conv2 fwd | 12544×72×16 | a b | 1.86 | 12.24 | 63.50 | 38.67 | 41.34 |
| cnn conv2 dW | 72×12544×16 | a^T b | 1.71 | 12.16 | 59.45 | 41.47 | 18.06 |
| cnn conv2 dcols | 12544×16×72 | a b^T | 1.79 | 11.74 | 52.94 | 29.55 | 21.37 |
| cnn fc fwd | 64×784×10 | a b | 1.88 | 9.11 | 18.96 | 26.67 | 14.34 |

### Other ops (µs per call)

| op | Phase 1 | 1T | 8T | KB alloc/call (Phase 1 → now) |
|---|---|---|---|---|
| add [1M] + [1M] | 4455.1 | 1205.9 | 186.0 | 54688 → 7813 |
| exp [1M] | 9562.0 | 6274.9 | 860.9 | 39063 → 7813 |
| relu fwd [1M] | 7905.0 | 738.9 | 144.0 | 23438 → 7813 |
| relu bwd [1M] | 7927.9 | 968.0 | 176.9 | 39063 → 7813 |
| bias add [50176;8] + [8] | 5757.1 | 799.9 | 138.0 | 21953 → 3137 |
| bias grad [50176;8] → [8] | 2630.9 | 244.1 | 67.9 | 6273 → 3 |
| broadcast_to [8] → [50176;8] | – | 427.0 | 82.0 | 3137 |
| log_softmax [64;10] | 30.0 | 4.1 | 4.1 | 150 → 5 |
| im2col [64;28;28;1] 3×3 | 4015.0 | 751.0 | 271.1 | 3528 → 3539 |
| im2col [64;14;14;8] 3×3 | 5268.1 | 1194.0 | 269.2 | 7056 → 7067 |
| col2im → [64;14;14;8] 3×3 | 2579.0 | 718.1 | 143.1 | 784 → 795 |
| max_pool2d [64;28;28;8] 2×2 | 2417.1 | 620.1 | 155.0 | 1568 → 1568 |
| max_pool2d bwd [64;28;28;8] | 606.1 | 459.0 | 305.9 | 3136 → 3136 |
| adam step (MLP, 101,770 params) | 2157.0 | 325.9 | 76.1 | 19084 → 1 |
| sgd+momentum step (MLP) | 722.9 | 162.1 | 52.9 | 4771 → 1 |

(The Phase 1 relu rows timed the closure-based `Tensor.map` that autograd
used at the time. The new rows time `Tensor.relu` / `Tensor.relu_backward`,
which replaced it.)

### What the numbers say

- **Single-core matmul is 8× faster** (1.8 → 15 GFLOP/s). The 4×4
  register-blocked kernel keeps 16 accumulators in registers, and on arm64
  `ocamlopt` compiles each `c +. a *. b` to one fused multiply-add. NumPy's
  BLAS is still 3× faster per core, because it uses SIMD (two doubles per
  instruction) and a larger kernel, which OCaml can't express.
- **On 8 threads, matmul beats NumPy on the MLP's own shapes** (74–79
  against 48–64 GFLOP/s), and reaches 106 GFLOP/s at 512².
- **The MLP is now at PyTorch's speed**, and the CNN beats PyTorch's
  12-thread time on 8 threads. Per step, the MLP spends 0.08 ms in Adam
  (PyTorch: 0.28 ms) and 0.49 ms in forward and backward (PyTorch: 0.22 ms).
- **Allocation is down 77×** for the MLP (23 → 0.3 MB per step) and 3× for
  the CNN. Minor collections went from 9–38 per step to almost none. What
  the CNN still allocates is the tensors themselves (im2col matrices,
  activations, gradients). Each op returns a fresh tensor, which is
  inherent to the define-by-run design.
- **Small matmuls don't benefit from threads** (`fc2`, 64×128×10). They
  stay below the parallel threshold, which is right: PyTorch's own 12-thread
  MLP is slower than its 1-thread MLP for the same reason.

---

## Phase 5: the BLAS backend (2026-11-26)

Same machine and compiler. The new backend is `flockml.blas`
(`Tensor_blas.F64` / `F32`), run through `Flockml.Make`: Bigarray storage,
`cblas_dgemm` / `cblas_sgemm` from Apple's **Accelerate**, and every other
kernel in C compiled with `-O3`. Select it in any benchmark with
`--backend blas` (float64) or `--backend blas32` (float32). PyTorch and
NumPy were re-run in this session: PyTorch's CNN on 12 threads measured
4.30 s this time, against 4.85 s in Phase 1.

### Headline

| Workload | Pure OCaml, 8T | BLAS f64, 8T | BLAS f32, 8T | Best PyTorch / NumPy, f64 | Best PyTorch, f32 |
|---|---|---|---|---|---|
| MLP epoch | 0.57 s | **0.21 s** | **0.12 s** | 0.51 s (1T) | 0.34 s (1T) |
| CNN epoch | 4.18 s | **3.14 s** | **2.27 s** | 4.30 s (12T) | 6.40 s (1T) |
| 512² matmul | 107 GFLOP/s | **660** | **2234** | NumPy 144 (12T) | |

Every Phase 5 target is met. Matmul is 4.6× NumPy's 12-thread speed, against
a target of "within 1.5×". The MLP is 2.4× and the CNN 1.4× faster than the
fastest PyTorch configuration in float64, against a target of "within 2×".
Much of the matmul gain is hardware: Accelerate runs matrix products on the
M2's AMX matrix units, and NumPy's OpenBLAS doesn't use them.

Test accuracy after 3 epochs:

| | Pure OCaml | BLAS f64 | BLAS f32 |
|---|---|---|---|
| MLP | 97.29% | 97.29% | 97.25% |
| CNN | 98.20% | 98.20% | 98.17% |

In float64 the BLAS backend matches the pure one at every epoch, to the
hundredth of a percent.

### Training, per step (ms, wall clock)

| | data | fwd | bwd | optim | step | epoch s | MB alloc/step | minor GC/step | major GC/step |
|---|---|---|---|---|---|---|---|---|---|
| MLP, pure 8T | 0.050 | 0.235 | 0.243 | 0.079 | 0.606 | 0.57 | 0.30 | 0.01 | 0.000 |
| MLP, BLAS f64 1T | 0.036 | 0.056 | 0.054 | 0.093 | 0.239 | 0.22 | 0.01 | 0.01 | 0.000 |
| MLP, BLAS f64 8T | 0.040 | 0.068 | 0.059 | 0.058 | 0.224 | 0.21 | 0.01 | 0.01 | 0.000 |
| MLP, BLAS f32 8T | 0.026 | 0.032 | 0.024 | 0.045 | 0.127 | 0.12 | 0.01 | 0.01 | 0.000 |
| PyTorch MLP f64 1T | 0.046 | 0.122 | 0.098 | 0.276 | 0.543 | 0.51 | | | |
| PyTorch MLP f32 1T | 0.033 | 0.068 | 0.067 | 0.192 | 0.361 | 0.34 | | | |
| CNN, pure 8T | 0.046 | 1.839 | 2.537 | 0.035 | 4.457 | 4.18 | 51.50 | 0.13 | 0.130 |
| CNN, BLAS f64 1T | 0.040 | 2.097 | 1.789 | 0.009 | 3.936 | 3.69 | 1.23 | 0.63 | 0.626 |
| CNN, BLAS f64 8T | 0.049 | 1.541 | 1.749 | 0.013 | 3.352 | 3.14 | 1.24 | 0.67 | 0.666 |
| CNN, BLAS f32 8T | 0.033 | 1.363 | 1.014 | 0.007 | 2.416 | 2.27 | 1.24 | 0.33 | 0.334 |
| PyTorch CNN f64 12T | 0.072 | 1.636 | 2.601 | 0.270 | 4.580 | 4.30 | | | |
| PyTorch CNN f32 1T | 0.049 | 2.456 | 4.224 | 0.097 | 6.827 | 6.40 | | | |

"1T" for the BLAS backend means `FLOCKML_THREADS=1 VECLIB_MAXIMUM_THREADS=1`:
one thread for FlockML's kernels and one for Accelerate. "MB alloc/step"
counts the OCaml heap only. The BLAS backend's tensor data lives outside the
heap, in recycled buffers (see below), so the column no longer measures
tensor allocation for it.

### Matmul (GFLOP/s)

| case | m×k×n | form | pure 8T | BLAS f64 | BLAS f64 1T | BLAS f32 | NumPy 1T | NumPy 12T |
|---|---|---|---|---|---|---|---|---|
| square 64 | 64×64×64 | a b | 16.4 | 244 | 244 | 550 | 22.6 | 22.2 |
| square 128 | 128×128×128 | a b | 55.2 | 298 | 279 | 838 | 38.6 | 58.5 |
| square 256 | 256×256×256 | a b | 93.5 | 347 | 349 | 1154 | 39.7 | 118.2 |
| square 512 | 512×512×512 | a b | 107.3 | 660 | 344 | 2234 | 50.4 | 144.3 |
| mlp fc1 fwd | 64×784×128 | a b | 73.9 | 299 | 285 | 980 | 47.2 | 48.5 |
| mlp fc1 dW | 784×64×128 | a^T b | 78.8 | 306 | 306 | 1078 | 49.0 | 63.7 |
| mlp fc2 fwd | 64×128×10 | a b | 8.2 | 76 | 76 | 137 | 24.1 | 23.3 |
| mlp fc2 dW | 128×64×10 | a^T b | 7.8 | 86 | 86 | 172 | 23.6 | 23.4 |
| mlp fc2 dX | 64×10×128 | a b^T | 9.2 | 76 | 76 | 172 | 29.1 | 29.1 |
| cnn conv1 fwd | 50176×9×8 | a b | 25.7 | 41.6 | 41.7 | 26.1 | 13.6 | 19.8 |
| cnn conv1 dW | 9×50176×8 | a^T b | 38.7 | 90.2 | 87.1 | 95.3 | 23.2 | 8.2 |
| cnn conv2 fwd | 12544×72×16 | a b | 63.5 | 176 | 174 | 295 | 38.8 | 39.3 |
| cnn conv2 dW | 72×12544×16 | a^T b | 59.3 | 175 | 177 | 603 | 41.5 | 17.2 |
| cnn conv2 dcols | 12544×16×72 | a b^T | 52.9 | 138 | 138 | 196 | 29.5 | 4.0 |
| cnn fc fwd | 64×784×10 | a b | 19.0 | 91.5 | 91.5 | 168 | 26.7 | 14.2 |

The small `fc2` products take a few microseconds, close to the timer's
resolution, so their rows are only rough.

### Other ops (µs per call, 8 threads)

| op | pure | BLAS f64 | BLAS f32 |
|---|---|---|---|
| add [1M] + [1M] | 186.0 | 124.0 | 60.1 |
| exp [1M] | 860.0 | 817.1 | 227.9 |
| relu fwd [1M] | 143.1 | 93.9 | 60.1 |
| relu bwd [1M] | 173.1 | 124.0 | 60.1 |
| bias add [50176;8] + [8] | 140.9 | 58.2 | 53.9 |
| bias grad [50176;8] → [8] | 68.2 | 57.0 | 57.9 |
| broadcast_to [8] → [50176;8] | 82.0 | 57.0 | 52.0 |
| log_softmax [64;10] | 4.1 | 3.1 | 1.9 |
| transpose [784;64] | 40.1 | 32.9 | 27.9 |
| im2col [64;28;28;1] 3×3 | 186.9 | 174.0 | 167.1 |
| im2col [64;14;14;8] 3×3 | 266.1 | 182.9 | 113.0 |
| col2im → [64;14;14;8] 3×3 | 146.9 | 93.0 | 75.1 |
| max_pool2d [64;28;28;8] 2×2 | 151.2 | 227.9 | 226.0 |
| max_pool2d bwd [64;28;28;8] | 137.1 | 104.0 | 75.1 |
| adam step (MLP, 101,770 params) | 74.1 | 48.2 | 37.9 |
| sgd+momentum step (MLP) | 50.1 | 41.0 | 31.0 |
| take_rows 64 of [60000;784] | 21.0 | 7.2 | 4.1 |

### What the numbers say

- **Matmul is no longer the bottleneck anywhere.** It ran at 15 GFLOP/s
  per core in pure OCaml, and Accelerate runs it at 250–660 GFLOP/s in
  float64 and up to 2.2 TFLOP/s in float32. The MLP step is now dominated
  by the elementwise ops and Adam.
- **The C kernels are memory-bound, and float32 halves the bytes.** With
  8 threads, `add` on 1M elements moves 24 MB in 124 µs (about
  190 GB/s, near the chip's bandwidth). In float32 it takes 60 µs.
- **Off-heap storage needed its own allocator.** The first version
  allocated each tensor with `Bigarray.Array1.create`, and its CNN epoch was
  *slower* than pure OCaml's (5.9 s against 4.6 s), with 7 major GC cycles
  per step. Two things combined:
  - The OCaml runtime charges Bigarray memory to the *major* GC, in
    proportion to the OCaml heap. With the dataset in a Bigarray, that heap
    is tiny, so every few megabytes of activations forced a full cycle.
  - Without those cycles, garbage piled up (6 GB resident set), and fresh
    `malloc` memory page-faulted on first touch.

  Tensors now come from a recycling pool (`fb_create` in the C stubs). A
  freed buffer goes back on a free list for its size, so the next step
  reuses memory that is already mapped. Tensors are charged to the *minor*
  GC (`caml_alloc_custom` with a fixed 128 MB budget), which finalises the
  dead temporaries of a step cheaply. Major cycles went from 7 per CNN step
  to 0.67, and the epoch from 5.9 s to 3.1 s.
- **In-place ops were on the scalar path.** The compiler vectorises
  `o[i] = a[i] + b[i]` only after checking at run time that `o` and `a`
  don't overlap. In `add_` and `scale_` they are the same array, so the
  check failed and the loop ran scalar, 1.5× slower than the allocating op.
  The kernels now have a separate in-place loop.
- **The CNN gains less than the MLP**, because its time is spread over
  im2col, pooling, broadcasts and reshapes, which are memory-bound in both
  backends. `max_pool2d` is the one op that got slower. It still returns its
  winning indices as an OCaml `int array` (6.4 MB per call), and allocating
  that on a small heap costs more than the kernel itself.

---

## Phases 7–8: the NumPy-level API and complete autograd (2026-11-26)

Same machine and compiler, 8 threads. These phases add features, not speed,
so the rule was that nothing may get slower.

### Training is unchanged

Epoch seconds, from the binary built before the change and the one after,
run alternately (two rounds each):

| Workload | Before | After |
|---|---|---|
| MLP, pure OCaml | 0.57, 0.59 | 0.58, 0.59 |
| CNN, pure OCaml | 4.07, 4.11 | 4.12, 4.12 |
| MLP, BLAS f64 | 0.21, 0.21 | 0.21, 0.21 |
| CNN, BLAS f64 | 3.17, 3.11 | 3.14, 3.12 |

Every row of `bench_ops` that existed before is within run-to-run noise on
both backends, in both directions (for example `add [1M]`: 187 µs before,
186 µs after).

### The new hot-path ops (µs per call, median)

The `sequence models` group of `bench_ops`, at the shapes of a small
transformer (batch 32, sequence 64, 4 heads of 32, vocabulary 65), against
PyTorch 2.0.1 in float64 on all 12 threads (`bench/torch_ops.py`):

| Op | Pure OCaml | BLAS f64 | PyTorch f64 |
|---|---|---|---|
| permute [32;64;4;32] -> heads | 79 | 44 | 79 |
| concat 2x [32;64;128] axis -1 | 215 | 141 | 151 |
| slice [32;64;128] -> [:, :, 64:] | 39 | 36 | 20 |
| matmul [32;64;128] x [128;128] | 727 | 123 | 239 |
| bmm q kᵀ [128;64;32] -> 64x64 | 872 | 251 | 589 |
| bmm att v [128;64;64] x [..;32] | 761 | 147 | 378 |
| masked_fill [128;64;64] causal | 514 | 298 | 220 |
| logsumexp [2048;65] axis -1 | 242 | 213 | 168 |
| gather [2048;65] one per row | 9 | 287 | 5 |
| index_select 2048 rows of [65;128] | 63 | 58 | 80 |

### What the numbers say

- **One strided copy covers every structural op.** `permute`, `slice`,
  `set_slice_`, `flip`, `pad` and `concat` are each a single call to a
  coalesced, parallel strided copy (`copy_walk_`), written once per backend.
  They run at memory speed, on par with PyTorch, which also copies when a
  view is made contiguous.
- **Batched matmul is fast where it matters.** A batch times a single
  matrix (a linear layer over a sequence) is one tall `gemm_`. True batches
  are one CBLAS call per matrix in the BLAS backend, 2–2.5× faster than
  PyTorch here. In pure OCaml, they are parallel over the batch, with each
  product sequential.
- **The composed ops cost a few passes.** `logsumexp` and `masked_fill` are
  built from other ops (max, subtract, exp, sum, log; broadcast then select),
  so they make several passes over memory where PyTorch makes one. Fused
  kernels can replace them once a model's profile shows them.
- **`gather` is a cold path in the BLAS backend.** It, `scatter_add`,
  `index_add`, `cumsum` and `prod_axis` convert to the default backend and
  back, like `map`. At 287 µs, `gather` costs 30× the pure-OCaml version. It
  needs a C kernel before cross-entropy over a vocabulary uses it in a
  training loop.


## Phases 9–10: layers, checkpoints, losses, optimizers (2026-11-27)

Phases 9 and 10 add layers, checkpoints, losses, four optimizers and
learning-rate schedules. The ground rule for them is that code that doesn't
use them pays nothing, so this section is an A/B check. The previous commit
and this one were built in release mode and run alternately on the same
machine (Apple M2 Pro, 8 threads, MNIST, batch 64, Adam).

### Training is unchanged

Median of three alternating full-epoch runs (ms per step; epoch seconds in
brackets):

| Model | Backend | Before | After |
|---|---|---|---|
| MLP 784–128–10 | pure OCaml | 0.613 (0.57) | 0.612 (0.57) |
| CNN | pure OCaml | 4.42 (4.15) | 4.37 (4.10) |
| MLP 784–128–10 | BLAS f64 | 0.223 (0.21) | 0.224 (0.21) |
| CNN | BLAS f64 | 3.39 (3.17) | 3.38 (3.17) |

The BLAS MLP row is the median of six further alternating runs, because the
first three were too close to call (0.217–0.228 ms on both sides).

The optimizer kernels changed: Adam's C kernel gained a flag for AdamW's
decoupled decay, SGD's one for Nesterov momentum, and Adam now counts steps
per parameter. The `bench_ops` optimizer rows, alternating four times
(µs per call), show no change:

| Op | Before | After |
|---|---|---|
| Adam step (MLP, 101,770 params), BLAS f64 | 47–50 | 46–52 |
| SGD + momentum step (MLP), BLAS f64 | 38–45 | 36–42 |
| Adam step, pure OCaml | 74–88 | 75–77 |
| SGD + momentum step, pure OCaml | 51–61 | 51–53 |

### What the numbers say

- **The common layers take the same path as before.** `Linear` with a bias
  on a 2-D input is still one fused `Var.linear` node. A square,
  undilated, ungrouped convolution is still `im2col` and one matmul.
- **The new layers are compositions for now,** as the roadmap planned.
  Batch norm, layer norm and attention are built from `Var` ops, which
  makes several passes over memory where a fused kernel would make one. The
  convolutions that `im2col` can't do (dilation, different strides per
  axis) take one strided slice per kernel position. Grouped convolutions
  are one batched matmul. Fused kernels come once a reference model's
  profile (Phase 12) shows where the time goes.
- **Cross-entropy with label smoothing or class weights avoids `gather`.**
  It multiplies the log-probabilities by one constant target matrix, so the
  BLAS backend's slow `gather` (see Phases 7–8) stays off the training path.

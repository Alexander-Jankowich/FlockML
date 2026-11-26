"""Matmul throughput in NumPy, for comparison with bench_matmul.exe.

    python3 bench/numpy_baseline.py

NumPy calls an optimised multithreaded BLAS, so expect it to win by a wide
margin; the interesting questions are by how much, and how far loop
reordering / blocking in OCaml closes the gap.
"""

import time

import numpy as np

print(f"{'n':>6} {'ms/matmul':>12} {'GFLOP/s':>10}")
rng = np.random.default_rng(0)
for n in (64, 128, 256, 512):
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    repeats = max(1, 200_000_000 // (n * n * n))
    a @ b  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        a @ b
    secs = (time.perf_counter() - t0) / repeats
    print(f"{n:>6} {secs * 1e3:>12.3f} {2 * n**3 / secs / 1e9:>10.2f}")

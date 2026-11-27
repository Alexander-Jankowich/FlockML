"""Matmul throughput in NumPy, for comparison with bench_matmul.exe.

    python3 bench/numpy_baseline.py

Uses the same cases as bench/bench_matmul.ml. NumPy calls an optimised,
multithreaded BLAS. To limit it to one thread, set the variable your BLAS
reads, e.g. VECLIB_MAXIMUM_THREADS=1 (Accelerate) or OPENBLAS_NUM_THREADS=1.

In NumPy a transpose is a free view, so the backward-pass rows measure only
the multiplication. In FlockML they also include a transposed copy.
"""

import time

import numpy as np

CASES = [(f"square {n}", n, n, n, "NN") for n in (64, 128, 256, 512)] + [
    ("mlp fc1 fwd", 64, 784, 128, "NN"),
    ("mlp fc1 dW (x^T g)", 784, 64, 128, "TN"),
    ("mlp fc2 fwd", 64, 128, 10, "NN"),
    ("mlp fc2 dW (h^T g)", 128, 64, 10, "TN"),
    ("mlp fc2 dX (g W^T)", 64, 10, 128, "NT"),
    ("cnn conv1 fwd", 50176, 9, 8, "NN"),
    ("cnn conv1 dW", 9, 50176, 8, "TN"),
    ("cnn conv2 fwd", 12544, 72, 16, "NN"),
    ("cnn conv2 dW", 72, 12544, 16, "TN"),
    ("cnn conv2 dcols", 12544, 16, 72, "NT"),
    ("cnn fc fwd", 64, 784, 10, "NN"),
]


def measure(f, min_time=0.5, min_runs=5):
    """Median seconds per call, after one warm-up call (as bench_util.ml)."""
    f()
    times = []
    start = time.perf_counter()
    while len(times) < min_runs or time.perf_counter() - start < min_time:
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    print(f"NumPy {np.__version__} matmul\n")
    print(f"| {'case':<20} | {'mxkxn':>17} | form  | {'ms/call':>10} | {'GFLOP/s':>8} |")
    print(f"|{'-' * 22}|{'-' * 19}|-------|{'-' * 12}|{'-' * 10}|")
    rng = np.random.default_rng(0)
    for name, m, k, n, form in CASES:
        if form == "NN":
            a, b = rng.standard_normal((m, k)), rng.standard_normal((k, n))
            f = lambda: a @ b
        elif form == "TN":
            a, b = rng.standard_normal((k, m)), rng.standard_normal((k, n))
            f = lambda: a.T @ b
        else:
            a, b = rng.standard_normal((m, k)), rng.standard_normal((n, k))
            f = lambda: a @ b.T
        secs = measure(f)
        label = {"NN": "a b  ", "TN": "a^T b", "NT": "a b^T"}[form]
        print(
            f"| {name:<20} | {m:5d}x{k:5d}x{n:5d} | {label} | "
            f"{secs * 1e3:10.3f} | {2 * m * k * n / secs / 1e9:8.2f} |"
        )


if __name__ == "__main__":
    main()

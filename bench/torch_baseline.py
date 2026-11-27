"""PyTorch reference for bench_train.exe: the same MLP and CNN, the same
optimizer (Adam, lr 1e-3) and batch size, with the same per-phase split
(data / forward + loss / backward / optimizer).

    python3 bench/torch_baseline.py                      # all configurations
    python3 bench/torch_baseline.py --steps 200 --model cnn

Runs float64 (FlockML's precision) and float32, each with one thread and with
all of PyTorch's default threads. It reads MNIST from data/mnist if the files
are there, and uses random data of the same shape otherwise. Output is a
Markdown table matching bench_train.exe.
"""

import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn

WARMUP = 3


def load_images(data_dir):
    """MNIST training images as float [n, 784] in [0, 1], with labels."""
    path = os.path.join(data_dir, "train-images-idx3-ubyte")
    if not os.path.exists(path):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(60000, 784, generator=g, dtype=torch.float64)
        y = torch.randint(0, 10, (60000,), generator=g)
        return x, y, "synthetic (no MNIST files found)"
    with open(path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    with open(os.path.join(data_dir, "train-labels-idx1-ubyte"), "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    x = torch.from_numpy(images.reshape(-1, 784).astype(np.float64) / 255.0)
    return x, torch.from_numpy(labels.astype(np.int64)), "MNIST"


def build(kind):
    if kind == "mlp":
        return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(7 * 7 * 16, 10),
    )


def run(kind, dtype, threads, x_all, y_all, steps, batch_size):
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    model = build(kind).to(dtype)
    x_all = x_all.to(dtype)
    if kind == "cnn":
        x_all = x_all.view(-1, 1, 28, 28)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = x_all.shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size
    steps = steps_per_epoch if steps <= 0 else min(steps, steps_per_epoch)
    order = torch.randperm(n)
    tot = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "optim": 0.0}
    gc.collect()
    start = None
    for i in range(steps):
        if i == WARMUP:
            tot = dict.fromkeys(tot, 0.0)
            start = time.perf_counter()
        t0 = time.perf_counter()
        idx = order[i * batch_size:(i + 1) * batch_size]
        x, y = x_all[idx], y_all[idx]
        t1 = time.perf_counter()
        loss = loss_fn(model(x), y)
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        opt.step()
        opt.zero_grad()
        t4 = time.perf_counter()
        for k, dt in zip(tot, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            tot[k] += dt
    wall = time.perf_counter() - start
    timed = steps - WARMUP
    ms = lambda s: 1000 * s / timed
    name = {"mlp": "MLP 784-128-10", "cnn": "CNN"}[kind]
    prec = {torch.float64: "f64", torch.float32: "f32"}[dtype]
    print(
        f"| {name:<14} | {prec:<5} | {threads:7d} | {timed:5d} | {ms(tot['data']):7.3f} | "
        f"{ms(tot['fwd']):7.3f} | {ms(tot['bwd']):7.3f} | {ms(tot['optim']):7.3f} | "
        f"{ms(wall):7.3f} | {wall / timed * steps_per_epoch:8.2f} |",
        flush=True,
    )
    return steps_per_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["mlp", "cnn", "both"], default="both")
    p.add_argument("--steps", type=int, default=0, help="0 = one full epoch")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--dtype", choices=["f64", "f32", "both"], default="both")
    args = p.parse_args()
    if 0 < args.steps <= WARMUP:
        p.error(f"--steps must be more than {WARMUP} (the warm-up)")

    x_all, y_all, source = load_images(args.data_dir)
    kinds = ["mlp", "cnn"] if args.model == "both" else [args.model]
    dtypes = {"f64": [torch.float64], "f32": [torch.float32]}.get(
        args.dtype, [torch.float64, torch.float32])
    all_threads = torch.get_num_threads()

    print(f"PyTorch {torch.__version__} training, batch {args.batch_size}, Adam\n")
    print(f"| {'model':<14} | dtype | threads | steps |    data |     fwd |"
          f"     bwd |   optim |    step |  epoch s |")
    print("|" + "|".join("-" * (w + 2) for w in (14, 5, 7, 5, 7, 7, 7, 7, 7, 8)) + "|")
    steps_per_epoch = 0
    for kind in kinds:
        for dtype in dtypes:
            for threads in sorted({1, all_threads}):
                steps_per_epoch = run(kind, dtype, threads, x_all, y_all,
                                      args.steps, args.batch_size)
    print(f"\ndata / fwd / bwd / optim / step: ms per step (wall clock); "
          f"epoch s: per-step time x {steps_per_epoch} steps.\n"
          f"Data: {source}. First {WARMUP} steps are warm-up and not counted.")


if __name__ == "__main__":
    main()

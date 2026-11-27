"""PyTorch twin of bin/train_cifar.ml: ResNet-20 on CIFAR-10, with the same
architecture (option-A shortcuts, He-normal convolutions, no conv biases),
the same augmentation (pad 4 and random crop, horizontal flip, normalise),
and the same recipe (SGD with Nesterov momentum 0.9, weight decay 5e-4, batch
128, learning rate 0.1 with a per-step cosine schedule).

    python3 bench/torch_cifar.py --epochs 20            # train, as train_cifar
    python3 bench/torch_cifar.py --steps 50             # time training steps
    python3 bench/torch_cifar.py --dtype float64 ...    # FlockML's default precision

Reads data/cifar-10 (run scripts/download_cifar10.sh). Augmentation runs on
whole batches in the training loop, as tensor ops, so it costs about as
little as FlockML's background prefetching.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]


def load(data_dir, split):
    files = [f"data_batch_{i}.bin" for i in range(1, 6)] if split == "train" else ["test_batch.bin"]
    raw = np.concatenate([np.fromfile(os.path.join(data_dir, f), dtype=np.uint8) for f in files])
    raw = raw.reshape(-1, 3073)
    labels = torch.from_numpy(raw[:, 0].astype(np.int64))
    images = torch.from_numpy(raw[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
    return images, labels


class Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.stride, self.extra = stride, out_c - in_c

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        s = x
        if self.stride != 1 or self.extra:
            s = x[:, :, ::self.stride, ::self.stride]
            s = F.pad(s, (0, 0, 0, 0, self.extra // 2, self.extra - self.extra // 2))
        return F.relu(y + s)


def resnet20():
    layers = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()]
    for in_c, out_c, stride in [(16, 16, 1), (16, 32, 2), (32, 64, 2)]:
        layers += [Block(in_c, out_c, stride), Block(out_c, out_c, 1), Block(out_c, out_c, 1)]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)  # He normal, as FlockML's default
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def augment(x, g):
    """Pad by 4 and crop 32x32 at a random place per image, then flip half."""
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (n,), generator=g)
    dx = torch.randint(0, 9, (n,), generator=g)
    rows = (dy[:, None] + torch.arange(32))[:, None, :, None]
    cols = (dx[:, None] + torch.arange(32))[:, None, None, :]
    out = xp[torch.arange(n)[:, None, None, None], torch.arange(3)[None, :, None, None], rows, cols]
    flip = torch.rand(n, generator=g) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0, help="only time this many steps")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--data-dir", default="data/cifar-10")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    x_train, y_train = load(args.data_dir, "train")
    x_test, y_test = load(args.data_dir, "test")
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    x_test = ((x_test - mean) / std).to(dtype)

    model = resnet20().to(dtype)
    print(f"ResNet-20: {sum(p.numel() for p in model.parameters())} parameters, "
          f"{args.dtype}, {torch.get_num_threads()} threads, PyTorch {torch.__version__}")
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    bs = args.batch_size
    steps_per_epoch = (len(x_train) + bs - 1) // bs
    total = steps_per_epoch * args.epochs

    def batches():
        perm = torch.randperm(len(x_train), generator=g)
        for k in range(steps_per_epoch):
            idx = perm[k * bs:(k + 1) * bs]
            yield ((augment(x_train[idx], g) - mean) / std).to(dtype), y_train[idx]

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for k in range(0, len(x_test), 500):
                pred = model(x_test[k:k + 500]).argmax(1)
                correct += (pred == y_test[k:k + 500]).sum().item()
        model.train()
        return correct / len(x_test)

    step = 0
    if args.steps:
        times = []
        for x, y in batches():
            t0 = time.perf_counter()
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
            times.append(time.perf_counter() - t0)
            if len(times) == args.steps + 3:
                break
        ms = 1000 * float(np.median(times[3:]))
        print(f"{ms:.1f} ms per step, {ms * steps_per_epoch / 1000:.1f} s per epoch")
        return

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running, count = 0.0, 0
        for x, y in batches():
            for group in opt.param_groups:
                group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            count += len(y)
            step += 1
        train_time = time.time() - t0
        acc = evaluate()
        print(f"epoch {epoch}: loss {running / count:.4f}, train {train_time:.1f}s, "
              f"test accuracy {100 * acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()

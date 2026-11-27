"""PyTorch twin of bin/regression.ml: an 11-64-64-1 MLP predicting the
quality of red wines, with the same split (every fifth row held out),
standardisation, He-normal initialisation, Adam (lr 1e-3, batch 32) and loss
(Huber by default, or MSE).

    python3 bench/torch_regression.py                 # as bin/regression.exe
    python3 bench/torch_regression.py --loss mse

Reads data/wine/winequality-red.csv (run scripts/download_wine.sh). Runs in
float64, FlockML's default precision.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wine/winequality-red.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    data = np.genfromtxt(args.data, delimiter=";", skip_header=1)
    x, y = torch.tensor(data[:, :-1]), torch.tensor(data[:, -1:])
    test = torch.arange(len(x)) % 5 == 0
    mean, std = x[~test].mean(0), x[~test].std(0)
    x = (x - mean) / std
    xtr, ytr, xte, yte = x[~test], y[~test], x[test], y[test]

    h = args.hidden
    model = nn.Sequential(nn.Linear(x.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                          nn.Linear(h, 1))
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = F.huber_loss if args.loss == "huber" else F.mse_loss

    def errors():
        with torch.no_grad():
            d = model(xte) - yte
        return d.pow(2).mean().sqrt().item(), d.abs().mean().item()

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(xtr))
        total = 0.0
        for k in range(0, len(xtr), args.batch_size):
            idx = perm[k:k + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == args.epochs:
            rmse, mae = errors()
            print(f"epoch {epoch:4d}: train loss {total / len(xtr):.4f}, "
                  f"test RMSE {rmse:.4f}, MAE {mae:.4f}")
    rmse, mae = errors()
    base = (yte - ytr.mean()).pow(2).mean().sqrt().item()
    print(f"test RMSE {rmse:.4f}, MAE {mae:.4f} (predicting the mean: RMSE {base:.4f})")


if __name__ == "__main__":
    main()

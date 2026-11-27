"""PyTorch twin of bin/gpt.ml: the same character-level GPT on Tiny
Shakespeare, with the same data split, initialisation, optimizer (AdamW,
betas 0.9 and 0.99, weight decay 0.1 on weight matrices only), schedule
(linear warmup, then cosine to min_lr), gradient clipping at norm 1, and the
same evaluation windows.

    python3 bench/torch_gpt.py --steps 3000          # train, as bin/gpt.exe
    python3 bench/torch_gpt.py --steps 3000 --sample 500

Reads data/tinyshakespeare/input.txt (run scripts/download_shakespeare.sh).
"""

import argparse
import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Converting mask")


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, context, dim, heads, layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        # N(0, 0.02) for embeddings and linear layers; the attention layers
        # keep MultiheadAttention's own initialisation, as FlockML's do.
        attn_out = {id(b.attn.out_proj) for b in self.blocks}
        for m in self.modules():
            if isinstance(m, nn.Embedding) or (isinstance(m, nn.Linear) and id(m) not in attn_out):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        h = self.tok(x) + self.pos(torch.arange(t))
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.norm(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare/input.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    text = open(args.data, encoding="latin-1").read()
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([index[c] for c in text])
    cut = len(ids) * 9 // 10
    train_ids, val_ids = ids[:cut], ids[cut:]
    ctx, bs = args.context, args.batch_size

    model = GPT(len(chars), ctx, args.dim, args.heads, args.layers)
    print(f"{len(ids)} characters (vocabulary {len(chars)}); GPT: "
          f"{sum(p.numel() for p in model.parameters())} parameters, "
          f"{torch.get_num_threads()} threads, PyTorch {torch.__version__}")
    decay = [p for n, p in model.named_parameters()
             if p.dim() >= 2 and not n.startswith(("tok.", "pos."))]
    no_decay = [p for n, p in model.named_parameters()
                if not (p.dim() >= 2 and not n.startswith(("tok.", "pos.")))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        total = max(1, args.steps - args.warmup)
        t = min(step - args.warmup, total) / total
        return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * t)) / 2

    def batch(source, starts):
        offs = torch.arange(ctx)
        x = source[starts[:, None] + offs]
        y = source[starts[:, None] + offs + 1]
        return x, y

    def estimate(source):
        n = len(source) - ctx
        count = args.eval_batches * bs
        stride = max(1, n // count)
        starts = torch.arange(min(count, n)) * stride
        total, weight = 0.0, 0
        model.eval()
        with torch.no_grad():
            for k in range(0, len(starts), bs):
                x, y = batch(source, starts[k:k + bs])
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                total += loss.item() * y.numel()
                weight += y.numel()
        model.train()
        return total / weight

    n_windows = len(train_ids) - ctx
    perm, pos = torch.randperm(n_windows, generator=g), 0
    losses, t0 = [], time.time()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        if pos + bs > n_windows:
            perm, pos = torch.randperm(n_windows, generator=g), 0
        x, y = batch(train_ids, perm[pos:pos + bs])
        pos += bs
        opt.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            train_time = time.time() - t0
            print(f"step {step + 1:5d}: train loss {np.mean(losses):.4f}, "
                  f"val loss {estimate(val_ids):.4f}, lr {lr_at(step):.2e}, "
                  f"{1000 * train_time / len(losses):.0f} ms/step", flush=True)
            losses, t0 = [], time.time()

    if args.sample:
        model.eval()
        out = [index["\n"]]
        with torch.no_grad():
            for _ in range(args.sample):
                x = torch.tensor([out[-ctx:]])
                p = F.softmax(model(x)[0, -1] / 0.8, dim=-1)
                out.append(torch.multinomial(p, 1, generator=g).item())
        print("\n--- sample ---\n" + "".join(chars[i] for i in out[1:]))


if __name__ == "__main__":
    main()

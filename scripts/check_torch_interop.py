"""Check that checkpoints move between FlockML and PyTorch, both ways.

Run from the repository root (needs PyTorch; `dune test` doesn't run this):

    python3 scripts/check_torch_interop.py

1. FlockML -> PyTorch. scripts/interop.exe saves a ResNet-style CNN and a
   small causal transformer as safetensors, with an input and the logits
   FlockML computes for it. This script builds the same models in PyTorch,
   loads the files (converting FlockML's layouts: weights are transposed, and
   convolution kernels are [kh * kw * in; out] over NHWC images), and checks
   that PyTorch computes the same logits.
2. PyTorch -> FlockML. It gives the PyTorch models new random weights, saves
   them in FlockML's names and layouts as float32 safetensors, has
   interop.exe load them into fresh FlockML models, and checks FlockML's
   logits against PyTorch's.

The safetensors package is used when it is installed; otherwise the format
(an 8-byte header length, a JSON header, raw data) is read and written here.
"""

import json
import warnings
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", message="Converting mask")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9

# safetensors --------------------------------------------------------------

DTYPES = {"F64": np.float64, "F32": np.float32, "I64": np.int64}


def st_load(path):
    try:
        from safetensors.numpy import load_file
        return load_file(path)
    except ImportError:
        pass
    with open(path, "rb") as f:
        data = f.read()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + n])
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        lo, hi = info["data_offsets"]
        dt = np.dtype(DTYPES[info["dtype"]]).newbyteorder("<")
        out[name] = np.frombuffer(data[8 + n + lo:8 + n + hi], dtype=dt).reshape(info["shape"])
    return out


def st_save(path, tensors):
    try:
        from safetensors.numpy import save_file
        return save_file(tensors, path)
    except ImportError:
        pass
    header, blobs, offset = {}, [], 0
    for name, a in tensors.items():
        tag = {np.dtype(np.float64): "F64", np.dtype(np.float32): "F32"}[a.dtype]
        raw = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": tag, "shape": list(a.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


# The models, mirroring scripts/interop.ml ---------------------------------


class Residual(nn.Module):
    """x + body(x); it adds no component to FlockML's parameter names."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return x + self.body(x)


class NHWC(nn.Module):
    """A 2-D layer of PyTorch's (NCHW) applied to FlockML's NHWC images."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2))


def cnn():
    return nn.Sequential(
        NHWC(nn.Conv2d(3, 8, 3, padding=1, bias=False)),
        NHWC(nn.BatchNorm2d(8)),
        nn.ReLU(),
        Residual(nn.Sequential(
            NHWC(nn.Conv2d(8, 8, 3, padding=1, groups=2)),
            NHWC(nn.BatchNorm2d(8)),
            nn.ReLU())),
        NHWC(nn.MaxPool2d(2)),
        NHWC(nn.Conv2d(8, 16, 3, stride=2, padding=2, dilation=2)),
        nn.GELU(),
        GlobalAvgPool(),
        nn.Linear(16, 10),
    )


VOCAB, DIM, LEN = 13, 16, 6


class CausalAttention(nn.MultiheadAttention):
    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        return super().forward(x, x, x, attn_mask=mask, need_weights=False)[0]


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        def block():
            return nn.Sequential(
                Residual(nn.Sequential(nn.LayerNorm(DIM), CausalAttention(DIM, 4, batch_first=True))),
                Residual(nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, 4 * DIM), nn.GELU(),
                                       nn.Linear(4 * DIM, DIM))))

        self.tok = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(LEN, DIM)
        self.blocks = nn.Sequential(block(), block())
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, x):
        x = x.long()
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        return self.head(self.norm(self.blocks(h)))


# Name and layout conversion ----------------------------------------------


def resolve(model, name):
    """The PyTorch module holding FlockML's parameter `name`, and the last
    component of the name ("w", "running_mean"...)."""
    *path, leaf = name.split(".")
    m = model

    def unwrap(m):
        while isinstance(m, (Residual, NHWC)):
            m = m.body if isinstance(m, Residual) else m.layer
        return m

    for c in path:
        m = unwrap(m)
        m = m[int(c)] if c.isdigit() else getattr(m, c)
    return unwrap(m), leaf


def torch_tensor(m, leaf):
    """The tensor of module m that FlockML calls `leaf`, and the functions
    converting between the two layouts."""
    ident = (lambda a: a, lambda a: a)
    if isinstance(m, nn.MultiheadAttention):
        return {"in_w": (m.in_proj_weight, (lambda a: a.T, lambda a: a.T)),
                "in_b": (m.in_proj_bias, ident),
                "out_w": (m.out_proj.weight, (lambda a: a.T, lambda a: a.T)),
                "out_b": (m.out_proj.bias, ident)}[leaf]
    if leaf == "b":
        return m.bias, ident
    if leaf != "w":
        return getattr(m, leaf), ident
    w = m.weight
    if isinstance(m, nn.Linear):
        return w, (lambda a: a.T, lambda a: a.T)
    if isinstance(m, nn.Conv2d):
        co, ci, kh, kw = w.shape
        return w, (lambda a: a.reshape(kh, kw, ci, co).transpose(3, 2, 0, 1),
                   lambda a: a.transpose(2, 3, 1, 0).reshape(kh * kw * ci, co))
    return w, ident


def load_flockml(model, entries):
    names = set()
    for name, a in entries.items():
        m, leaf = resolve(model, name)
        t, (to_torch, _) = torch_tensor(m, leaf)
        with torch.no_grad():
            t.copy_(torch.tensor(np.array(to_torch(a.astype(np.float64)))).to(t.dtype))
        names.add(id(t))
    expected = {id(t) for t in list(model.parameters()) + list(model.buffers())}
    missing = expected - names
    assert not missing, f"{len(missing)} PyTorch tensors weren't in the file"


def save_flockml(model, flockml_names, path):
    out = {}
    for name in flockml_names:
        m, leaf = resolve(model, name)
        t, (_, to_flockml) = torch_tensor(m, leaf)
        out[name] = np.array(to_flockml(t.detach().numpy()), dtype=np.float32, order="C")
    st_save(path, out)


def main():
    subprocess.run(["dune", "build", "scripts/interop.exe"], cwd=ROOT, check=True)
    exe = os.path.join(ROOT, "_build", "default", "scripts", "interop.exe")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([exe, "save", d], check=True)
        worst = 0.0
        for name, make in [("cnn", cnn), ("gpt", GPT)]:
            model = make().eval()
            entries = st_load(os.path.join(d, name + ".safetensors"))
            x = torch.tensor(np.load(os.path.join(d, name + "_x.npy")))
            # 1. FlockML -> PyTorch
            load_flockml(model, entries)
            with torch.no_grad():
                got = model(x).numpy()
            want = np.load(os.path.join(d, name + "_logits.npy"))
            err = np.abs(got - want).max()
            print(f"{name}: FlockML checkpoint in PyTorch, max |logit diff| = {err:.2e}")
            worst = max(worst, err)
            # 2. PyTorch -> FlockML, in float32
            torch.manual_seed(7)
            other = make().eval()
            with torch.no_grad():
                for p in other.parameters():
                    p.copy_(torch.randn_like(p) * 0.3)
                for b in other.buffers():
                    if b.dtype.is_floating_point:
                        b.copy_(torch.rand_like(b) + 0.5)
                    else:
                        b.fill_(5)
                # the weights as they will be read back from float32
                for t in list(other.parameters()) + list(other.buffers()):
                    if t.dtype.is_floating_point:
                        t.copy_(t.float().double())
            save_flockml(other, entries.keys(), os.path.join(d, name + "_torch.safetensors"))
            with torch.no_grad():
                np.save(os.path.join(d, name + "_logits_torch.npy"), other(x).numpy())
        subprocess.run([exe, "load", d], check=True)
        for name in ["cnn", "gpt"]:
            got = np.load(os.path.join(d, name + "_logits_flockml.npy"))
            want = np.load(os.path.join(d, name + "_logits_torch.npy"))
            err = np.abs(got - want).max()
            print(f"{name}: PyTorch checkpoint in FlockML, max |logit diff| = {err:.2e}")
            worst = max(worst, err)
    if worst > TOL:
        sys.exit(f"FAILED: logits differ by {worst:.2e} > {TOL}")
    print("OK: checkpoints agree in both directions")


if __name__ == "__main__":
    main()

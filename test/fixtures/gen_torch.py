"""Write the PyTorch reference outputs that test/suites/torch_tests.ml checks.

Run by hand from the repository root, then commit the files in
test/fixtures/torch/:

    python3 test/fixtures/gen_torch.py

`dune test` only reads the committed files, so running the tests never needs
Python or PyTorch. Everything is float64. Weights are saved in FlockML's
layout (the transposes of PyTorch's, NHWC images), so the OCaml side builds
its layers straight from the files; torch_tests.ml repeats each case's
hyperparameters.
"""

import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch")
os.makedirs(DIR, exist_ok=True)
rng = np.random.default_rng(1)


def save(name, a):
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    np.save(os.path.join(DIR, name + ".npy"), np.asarray(a, dtype=np.float64))


def ints(name, a):
    np.save(os.path.join(DIR, name + ".npy"), np.asarray(a, dtype=np.int64))


def rand(*shape, lo=-1.0, hi=1.0):
    return torch.tensor(rng.uniform(lo, hi, shape))


def nhwc(t):
    return t.permute(0, 2, 3, 1)


def nchw(t):
    return t.permute(0, 3, 1, 2)


# Layers ---------------------------------------------------------------------

# name, in_c, out_c, (kh, kw), (sh, sw), (ph, pw), (dh, dw), groups, h, w
CONV2D = [
    ("c_basic", 3, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1, 5, 6),
    ("c_rect", 2, 3, (2, 3), (2, 1), (1, 0), (1, 1), 1, 6, 7),
    ("c_dil", 2, 2, (3, 3), (1, 1), (2, 2), (2, 2), 1, 6, 6),
    ("c_groups", 4, 6, (3, 3), (2, 2), (1, 1), (1, 1), 2, 7, 5),
    ("c_depthwise", 3, 3, (3, 3), (1, 1), (1, 1), (1, 1), 3, 5, 5),
]
for name, ci, co, k, s, p, d, g, h, w in CONV2D:
    x = rand(2, h, w, ci)
    W = rand(co, ci // g, *k)
    b = rand(co)
    y = F.conv2d(nchw(x), W, b, stride=s, padding=p, dilation=d, groups=g)
    save(name + "_x", x)
    save(name + "_w", W.permute(2, 3, 1, 0).reshape(-1, co))
    save(name + "_b", b)
    save(name + "_y", nhwc(y))

# name, in_c, out_c, k, stride, padding, dilation, groups, length
CONV1D = [
    ("c1_a", 3, 4, 3, 2, 1, 2, 1, 9),
    ("c1_b", 4, 4, 3, 1, 1, 1, 2, 6),
]
for name, ci, co, k, s, p, d, g, length in CONV1D:
    x = rand(2, length, ci)
    W = rand(co, ci // g, k)
    b = rand(co)
    y = F.conv1d(x.permute(0, 2, 1), W, b, stride=s, padding=p, dilation=d, groups=g)
    save(name + "_x", x)
    save(name + "_w", W.permute(2, 1, 0).reshape(-1, co))
    save(name + "_b", b)
    save(name + "_y", y.permute(0, 2, 1))

# conv_transpose2d: in 3, out 2, kernel 3, stride 2, padding 1, output_padding 1
x = rand(2, 4, 3, 3)
W = rand(3, 2, 3, 3)
b = rand(2)
y = F.conv_transpose2d(nchw(x), W, b, stride=2, padding=1, output_padding=1)
save("ct_x", x)
save("ct_w", W.permute(0, 2, 3, 1).reshape(3, -1))
save("ct_b", b)
save("ct_y", nhwc(y))

# pooling
x = rand(2, 5, 6, 3)
save("pool_x", x)
save("maxpool_y", nhwc(F.max_pool2d(nchw(x), 3, stride=2, padding=1)))
save("avgpool_y", nhwc(F.avg_pool2d(nchw(x), 3, stride=2, padding=1)))
save("avgpool2_y", nhwc(F.avg_pool2d(nchw(x), 2)))

# activations
x = rand(4, 5, lo=-4, hi=4)
save("act_x", x)
save("leaky_relu", F.leaky_relu(x, 0.1))
save("elu", F.elu(x, 0.7))
save("gelu", F.gelu(x))
save("gelu_tanh", F.gelu(x, approximate="tanh"))
save("silu", F.silu(x))
save("softplus", F.softplus(x, beta=2.0, threshold=5.0))
save("softmax", F.softmax(x, dim=-1))
save("log_softmax", F.log_softmax(x, dim=-1))

# batch norm: one training step, then evaluation
for name, x, x2 in [
    ("bn1", rand(6, 4), rand(3, 4)),
    ("bn2", rand(2, 3, 3, 4), rand(2, 2, 2, 4)),
]:
    bn = nn.BatchNorm1d(4, momentum=0.2) if x.dim() == 2 else nn.BatchNorm2d(4, momentum=0.2)
    with torch.no_grad():
        bn.weight.copy_(rand(4))
        bn.bias.copy_(rand(4))
    to = (lambda t: t) if x.dim() == 2 else nchw
    back = (lambda t: t) if x.dim() == 2 else nhwc
    bn.train()
    y = back(bn(to(x)))
    bn.eval()
    y2 = back(bn(to(x2)))
    for k, v in [("x", x), ("x2", x2), ("w", bn.weight), ("b", bn.bias), ("y", y),
                 ("y2", y2), ("mean", bn.running_mean), ("var", bn.running_var)]:
        save(f"{name}_{k}", v)

# layer norm over the last two axes
x = rand(2, 3, 4)
ln = nn.LayerNorm([3, 4])
with torch.no_grad():
    ln.weight.copy_(rand(3, 4))
    ln.bias.copy_(rand(3, 4))
save("ln_x", x)
save("ln_w", ln.weight)
save("ln_b", ln.bias)
save("ln_y", ln(x))

# group norm, NHWC with 3 groups of 2 channels
x = rand(2, 3, 3, 6)
gn = nn.GroupNorm(3, 6)
with torch.no_grad():
    gn.weight.copy_(rand(6))
    gn.bias.copy_(rand(6))
save("gn_x", x)
save("gn_w", gn.weight)
save("gn_b", gn.bias)
save("gn_y", nhwc(gn(nchw(x))))

# recurrent layers, batch first: input 3, hidden 4
x = rand(2, 5, 3)
save("rnn_x", x)
for name, layer in [
    ("rnn", nn.RNN(3, 4, batch_first=True)),
    ("rnn_relu", nn.RNN(3, 4, nonlinearity="relu", batch_first=True)),
    ("lstm", nn.LSTM(3, 4, batch_first=True)),
    ("gru", nn.GRU(3, 4, batch_first=True)),
]:
    out = layer(x)[0]
    save(name + "_w_ih", layer.weight_ih_l0.T)
    save(name + "_w_hh", layer.weight_hh_l0.T)
    save(name + "_b_ih", layer.bias_ih_l0)
    save(name + "_b_hh", layer.bias_hh_l0)
    save(name + "_y", out)

# multi-head attention: dim 8, 2 heads
x = rand(2, 5, 8)
mha = nn.MultiheadAttention(8, 2, batch_first=True)
with torch.no_grad():
    mha.in_proj_bias.copy_(rand(24))
    mha.out_proj.bias.copy_(rand(8))
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
save("mha_x", x)
save("mha_in_w", mha.in_proj_weight.T)
save("mha_in_b", mha.in_proj_bias)
save("mha_out_w", mha.out_proj.weight.T)
save("mha_out_b", mha.out_proj.bias)
save("mha_y", mha(x, x, x, need_weights=False)[0])
save("mha_causal_y", mha(x, x, x, attn_mask=causal, need_weights=False)[0])
q = rand(2, 3, 8)
save("mha_q", q)
save("mha_cross_y", mha(q, x, x, need_weights=False)[0])

# Losses: the value and the gradient of every input ----------------------------


def loss_case(name, f, *inputs):
    """The loss, and the gradient of every input for a random seed (the
    gradient of sum(seed * loss) when the loss isn't a scalar)."""
    ts = [t.clone().requires_grad_(True) for t in inputs]
    out = f(*ts)
    seed = torch.ones(()) if out.dim() == 0 else torch.tensor(
        rng.uniform(-1, 1, tuple(out.shape)))
    out.backward(seed)
    save(name, out)
    save(name + "_seed", seed)
    for i, t in enumerate(ts):
        save(f"{name}_g{i}", t.grad)


pred = rand(4, 3, lo=-2, hi=2)
target = rand(4, 3, lo=-2, hi=2)
save("loss_pred", pred)
save("loss_target", target)
for red in ["mean", "sum", "none"]:
    loss_case(f"mse_{red}", lambda p, t: F.mse_loss(p, t, reduction=red), pred, target)
    loss_case(f"l1_{red}", lambda p, t: F.l1_loss(p, t, reduction=red), pred, target)
    loss_case(f"smooth_l1_{red}",
              lambda p, t: F.smooth_l1_loss(p, t, reduction=red, beta=0.5), pred, target)
    loss_case(f"huber_{red}",
              lambda p, t: F.huber_loss(p, t, reduction=red, delta=0.5), pred, target)

logits = rand(5, 4, lo=-3, hi=3)
labels = [0, 3, 1, 1, 2]
weights = torch.tensor([0.5, 2.0, 1.0, 1.5])
save("ce_logits", logits)
ints("ce_labels", labels)
save("ce_weights", weights)
lab = torch.tensor(labels)
loss_case("ce_mean", lambda x: F.cross_entropy(x, lab), logits)
loss_case("ce_sum", lambda x: F.cross_entropy(x, lab, reduction="sum"), logits)
loss_case("ce_none", lambda x: F.cross_entropy(x, lab, reduction="none"), logits)
loss_case("ce_smooth", lambda x: F.cross_entropy(x, lab, label_smoothing=0.1), logits)
loss_case("ce_weighted",
          lambda x: F.cross_entropy(x, lab, weight=weights, label_smoothing=0.2), logits)
loss_case("ce_weighted_none",
          lambda x: F.cross_entropy(x, lab, weight=weights, reduction="none"), logits)
loss_case("nll_weighted",
          lambda x: F.nll_loss(x, lab, weight=weights), F.log_softmax(logits, -1))
seq = rand(2, 3, 4, lo=-3, hi=3)
seq_labels = [[0, 3, 1], [2, 2, 0]]
save("ce_seq_logits", seq)
ints("ce_seq_labels", seq_labels)
st = torch.tensor(seq_labels)
loss_case("ce_seq", lambda x: F.cross_entropy(x.permute(0, 2, 1), st, label_smoothing=0.1),
          seq)
loss_case("ce_seq_none",
          lambda x: F.cross_entropy(x.permute(0, 2, 1), st, reduction="none"), seq)

probs = rand(4, 3, lo=0.05, hi=0.95)
bin_t = torch.tensor(rng.integers(0, 2, (4, 3)).astype(float))
blogits = rand(4, 3, lo=-5, hi=5)
blogits[0, 0] = 60.0
blogits[1, 1] = -60.0
pos_weight = torch.tensor([1.0, 2.5, 0.5])
save("bce_probs", probs)
save("bce_targets", bin_t)
save("bce_logits", blogits)
save("bce_pos_weight", pos_weight)
loss_case("bce", lambda p, t: F.binary_cross_entropy(p, t), probs, bin_t)
loss_case("bce_logits_mean", lambda x, t: F.binary_cross_entropy_with_logits(x, t),
          blogits, bin_t)
loss_case("bce_logits_pw",
          lambda x, t: F.binary_cross_entropy_with_logits(x, t, pos_weight=pos_weight,
                                                          reduction="sum"),
          blogits, bin_t)

kl_in = F.log_softmax(rand(3, 4, lo=-2, hi=2), -1)
kl_t = F.softmax(rand(3, 4, lo=-2, hi=2), -1)
kl_t[0, 1] = 0.0
save("kl_input", kl_in)
save("kl_target", kl_t)
for red in ["mean", "sum", "batchmean", "none"]:
    loss_case(f"kl_{red}", lambda i, t: F.kl_div(i, t, reduction=red), kl_in, kl_t)
loss_case("kl_log_target",
          lambda i, t: F.kl_div(i, t.clamp_min(1e-3).log(), reduction="batchmean",
                                log_target=True),
          kl_in, kl_t)

# Optimizers: parameters after 10 steps ---------------------------------------

ox = rand(6, 3)
oy = rand(6, 2)
w0 = rand(3, 2)
b0 = rand(2)
for k, v in [("opt_x", ox), ("opt_y", oy), ("opt_w0", w0), ("opt_b0", b0)]:
    save(k, v)

OPTIMS = {
    "sgd": lambda w, b: torch.optim.SGD([w, b], lr=0.1),
    "sgd_momentum": lambda w, b: torch.optim.SGD([w, b], lr=0.1, momentum=0.9,
                                                 weight_decay=0.01),
    "sgd_nesterov": lambda w, b: torch.optim.SGD([w, b], lr=0.1, momentum=0.9,
                                                 weight_decay=0.01, nesterov=True),
    "adam": lambda w, b: torch.optim.Adam([w, b], lr=0.05),
    "adam_wd": lambda w, b: torch.optim.Adam([w, b], lr=0.05, weight_decay=0.1),
    "adamw": lambda w, b: torch.optim.AdamW([w, b], lr=0.05, weight_decay=0.1),
    "rmsprop": lambda w, b: torch.optim.RMSprop([w, b], lr=0.01),
    "rmsprop_full": lambda w, b: torch.optim.RMSprop([w, b], lr=0.01, momentum=0.9,
                                                     centered=True, weight_decay=0.01),
    "adagrad": lambda w, b: torch.optim.Adagrad([w, b], lr=0.1),
    "adagrad_full": lambda w, b: torch.optim.Adagrad(
        [w, b], lr=0.1, lr_decay=0.01, weight_decay=0.01, initial_accumulator_value=0.1),
    "groups": lambda w, b: torch.optim.AdamW(
        [{"params": [w]}, {"params": [b], "lr": 0.1, "weight_decay": 0.0}],
        lr=0.05, weight_decay=0.1),
}
for name, make in OPTIMS.items():
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = make(w, b)
    for _ in range(10):
        opt.zero_grad()
        loss = ((torch.tanh(ox @ w + b) - oy) ** 2).mean()
        loss.backward()
        opt.step()
    save(f"opt_{name}_w", w)
    save(f"opt_{name}_b", b)

# Schedules: the learning rate of each of 25 steps ------------------------------


def lrs(make, steps=25, metrics=None):
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.1)
    sched = make(opt)
    out = []
    for i in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        if metrics is None:
            sched.step()
        else:
            sched.step(metrics[i])
    return out


S = torch.optim.lr_scheduler
save("sched_step", lrs(lambda o: S.StepLR(o, step_size=5, gamma=0.5)))
save("sched_multistep", lrs(lambda o: S.MultiStepLR(o, [3, 7, 12], gamma=0.2)))
save("sched_exponential", lrs(lambda o: S.ExponentialLR(o, 0.9)))
save("sched_cosine", lrs(lambda o: S.CosineAnnealingLR(o, T_max=20, eta_min=0.001), 21))
save("sched_linear", lrs(lambda o: S.LinearLR(o, start_factor=0.25, total_iters=10)))
save("sched_one_cycle",
     lrs(lambda o: S.OneCycleLR(o, max_lr=0.1, total_steps=25, cycle_momentum=False), 24))
plateau_metrics = [1.0, 0.9, 0.95, 0.95, 0.95, 0.8, 0.85, 0.85, 0.85, 0.85,
                   0.85, 0.85, 0.7, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
save("plateau_metrics", plateau_metrics)
save("sched_plateau",
     lrs(lambda o: S.ReduceLROnPlateau(o, factor=0.5, patience=2, cooldown=1), 20,
         plateau_metrics))

# A safetensors file with every common dtype -----------------------------------
# Written by hand (the safetensors package isn't needed), as the format
# specifies: an 8-byte header length, a JSON header, then the data.

parts = [
    ("a_f16", "F16", np.array([[0.5, -1.25, 3.0], [1e-5, 65504.0, -0.0]], dtype=np.float16)),
    ("b_bf16", "BF16", np.array([1.5, -2.0, 3.14159], dtype=np.float32)),
    ("c_i64", "I64", np.array([[1, -2], [3, 1 << 40]], dtype=np.int64)),
    ("d_f32", "F32", np.array([0.1, -7.5], dtype=np.float32)),
    ("e_f64", "F64", np.array([np.pi], dtype=np.float64)),
    ("f_u8", "U8", np.array([0, 255], dtype=np.uint8)),
]
header = {"__metadata__": {"format": "pt", "note": "written by gen_torch.py"}}
blobs = []
offset = 0
for name, dtype, a in parts:
    if dtype == "BF16":
        raw = (a.view(np.uint32) >> 16).astype("<u2").tobytes()
        expected = ((a.view(np.uint32) >> 16) << 16).view(np.float32).astype(np.float64)
    else:
        raw = a.astype(a.dtype.newbyteorder("<")).tobytes()
        expected = a.astype(np.float64)
    header[name] = {"dtype": dtype, "shape": list(a.shape),
                    "data_offsets": [offset, offset + len(raw)]}
    offset += len(raw)
    blobs.append(raw)
    save("st_" + name, expected)
h = json.dumps(header).encode()
h += b" " * ((8 - len(h) % 8) % 8)
with open(os.path.join(DIR, "mixed.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))

print("wrote", len(os.listdir(DIR)), "files to", DIR)

"""Write the NumPy reference outputs that test/suites/numpy_tests.ml checks.

Run by hand from the repository root, then commit the .npy files:

    python3 test/fixtures/gen.py

`dune test` only reads the committed files, so running the tests never needs
Python. Every array here is named after the input or the case it belongs to;
numpy_tests.ml repeats each case's parameters (axes, slices, indices).
"""

import math
import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def save(name, a):
    np.save(os.path.join(DIR, name + ".npy"), np.asarray(a))


def ints(name, a):
    np.save(os.path.join(DIR, name + ".npy"), np.asarray(a, dtype=np.int64))


# Inputs ---------------------------------------------------------------------

x = rng.uniform(-3, 3, (4, 5))
pos = rng.uniform(0.1, 3, (4, 5))
small = rng.uniform(-0.9, 3, (4, 5))
x3 = rng.uniform(-2, 2, (3, 4, 5))
near1 = rng.uniform(0.5, 1.5, (3, 4, 5))
row = rng.uniform(-3, 3, (5,))
ia = rng.integers(0, 3, (3, 4)).astype(float)
ib = rng.integers(0, 3, (4,)).astype(float)
mask = np.array([[1.0], [0.0], [1.0]])
m2 = rng.uniform(-2, 2, (3, 4))
a234 = rng.uniform(-1, 1, (2, 3, 4))
b45 = rng.uniform(-1, 1, (4, 5))
a2134 = rng.uniform(-1, 1, (2, 1, 3, 4))
b542 = rng.uniform(-1, 1, (5, 4, 2))
a243 = rng.uniform(-1, 1, (2, 4, 3))
b245 = rng.uniform(-1, 1, (2, 4, 5))
v4 = rng.uniform(-1, 1, (4,))
w4 = rng.uniform(-1, 1, (4,))
v3 = rng.uniform(-1, 1, (3,))
c325 = rng.uniform(-1, 1, (3, 2, 5))
b253 = rng.uniform(-1, 1, (2, 5, 3))

for name, a in [
    ("x", x), ("pos", pos), ("small", small), ("x3", x3), ("near1", near1),
    ("row", row), ("ia", ia), ("ib", ib), ("mask", mask), ("m2", m2),
    ("a234", a234), ("b45", b45), ("a2134", a2134), ("b542", b542),
    ("a243", a243), ("b245", b245), ("v4", v4), ("w4", w4), ("v3", v3),
    ("c325", c325), ("b253", b253),
]:
    save(name, a)

# Elementwise ----------------------------------------------------------------

save("abs", np.abs(x))
save("sign", np.sign(x))
save("square", np.square(x))
save("rsqrt", 1 / np.sqrt(pos))
save("log1p", np.log1p(small))
save("expm1", np.expm1(x))
save("sin", np.sin(x))
save("cos", np.cos(x))
save("tanh", np.tanh(x))
save("sigmoid", 1 / (1 + np.exp(-x)))
save("erf", np.vectorize(math.erf)(x))
save("clamp", np.clip(x, -1.0, 0.5))
save("clamp_min", np.maximum(x, 0.25))
save("minimum", np.minimum(x, row))
save("maximum", np.maximum(x, row))
for op, f in [("eq", np.equal), ("ne", np.not_equal), ("lt", np.less),
              ("le", np.less_equal), ("gt", np.greater),
              ("ge", np.greater_equal)]:
    save(op, f(ia, ib).astype(float))
save("where", np.where(mask != 0, m2, row[:4]))
save("masked_fill", np.where(mask != 0, 9.0, m2))

# Reductions -----------------------------------------------------------------

save("max", np.max(x3))
save("min", np.min(x3))
save("mean_axis1", np.mean(x3, axis=1))
save("min_axis2", np.min(x3, axis=2))
ints("argmin_axis0", np.argmin(x3, axis=0).reshape(-1))
save("prod_axis1", np.prod(near1, axis=1))
save("cumsum_axis1", np.cumsum(x3, axis=1))
m = np.max(x3, axis=2, keepdims=True)
save("logsumexp_axis2", (m + np.log(np.sum(np.exp(x3 - m), axis=2, keepdims=True)))[..., 0])
save("var_axis0", np.var(x3, axis=0))
save("var1_axis2", np.var(x3, axis=2, ddof=1))
save("std_axis1", np.std(x3, axis=1))
save("sum_axes02", np.sum(x3, axis=(0, 2)))
save("mean_axes12_keep", np.mean(x3, axis=(1, 2), keepdims=True))

# Shapes ---------------------------------------------------------------------

save("permute201", np.transpose(x3, (2, 0, 1)))
save("swapaxes02", np.swapaxes(x3, 0, 2))
save("concat1", np.concatenate([c325, x3[:, :4, :]], axis=1))
save("stack1", np.stack([m2, m2 * 2], axis=1))
save("split2_a", x3[:, :, :2])
save("split2_b", x3[:, :, 2:])
save("slice_mixed", x3[1:3, 0:4:2, -1])
save("slice_clip", x3[-10:100, 1:])
save("flip02", np.flip(x3, axis=(0, 2)))
save("pad", np.pad(m2, ((1, 2), (0, 3)), constant_values=7.0))
save("tile", np.tile(m2[:2, :3], (2, 1, 2)))
save("repeat1", np.repeat(m2[:2, :3], 3, axis=1))
idx = [3, 0, 0, 2]
ints("idx", idx)
save("index_select1", np.take(x3, idx, axis=1))
g_idx = np.array([[0, 3], [2, 2], [1, 0]])
ints("gather_idx", g_idx.reshape(-1))
save("gather1", np.take_along_axis(m2, g_idx, axis=1))
src = rng.uniform(-1, 1, (3, 2))
save("scatter_src", src)
out = m2.copy()
for i in range(3):
    for j in range(2):
        out[i, g_idx[i, j]] += src[i, j]
save("scatter_add1", out)
isrc = rng.uniform(-1, 1, (3, 4, 5))
save("index_src", isrc)
out = x3.copy()
for j, k in enumerate(idx):
    out[:, k, :] += isrc[:, j, :]
save("index_add1", out)
t = np.zeros((4, 5))
ssrc = rng.uniform(-1, 1, (2, 2))
save("set_src", ssrc)
t[1:4:2, 1:3] = ssrc
save("set_slice", t)
t = np.zeros((4, 5))
t[:, 2] = v4
save("set_slice_bcast", t)

# Linear algebra -------------------------------------------------------------

save("matmul_3d_2d", a234 @ b45)
save("matmul_bcast", a2134 @ b542)
save("matmul_1d_2d", v4 @ b45)
save("matmul_2d_1d", b45.T @ v4)
save("matmul_1d_1d", v4 @ w4)
save("bmm_trans_a", np.swapaxes(a243, -1, -2) @ b245)
save("bmm_trans_b", a243 @ np.swapaxes(b253, -1, -2))
save("dot", np.dot(v4, w4))
save("outer", np.outer(v4, v3))
save("norm2", np.linalg.norm(x3.reshape(-1)))
save("norm1", np.sum(np.abs(x3)))
save("norminf", np.max(np.abs(x3)))
save("norm3", np.sum(np.abs(x3) ** 3) ** (1 / 3))

# Construction ---------------------------------------------------------------

save("arange_step", np.arange(0.5, 2.0, 0.3))
save("arange5", np.arange(5.0))
save("arange_down", np.arange(5.0, 0.0, -1.5))
save("linspace", np.linspace(-1.0, 2.0, 7))
save("eye", np.eye(4))

# File formats, for the loader -----------------------------------------------

np.save(os.path.join(DIR, "f32.npy"), m2.astype(np.float32))
np.save(os.path.join(DIR, "big_endian.npy"), m2.astype(">f8"))
np.save(os.path.join(DIR, "fortran.npy"), np.asfortranarray(x3))
np.save(os.path.join(DIR, "int32.npy"), np.arange(-3, 9, dtype=np.int32).reshape(3, 4))
np.save(os.path.join(DIR, "uint8.npy"), np.arange(250, 256, dtype=np.uint8))
np.save(os.path.join(DIR, "bool.npy"), np.array([True, False, True]))
np.save(os.path.join(DIR, "scalar.npy"), np.float64(2.5))

print("wrote fixtures to", DIR)

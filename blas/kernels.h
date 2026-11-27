/* The numerical kernels of Tensor_blas, written once for an element type T.

   tensor_blas_stubs.c includes this file twice: with T = double and the
   suffix _d, then with T = float and the suffix _s. Every kernel works on a
   range [lo, hi) of its output, so the OCaml side can split one operation
   across domains with Parallel.parallel_for, and every output element is
   computed by exactly one call, in a fixed order. As in the pure-OCaml
   backend, results therefore don't depend on the thread count.

   Most kernels follow the loop order of the matching OCaml kernel in
   lib/tensor/tensor.ml, so in float64 they produce the same values (often to
   the bit). Scalar reductions (sum, softmax row sums) accumulate in double
   for both element types, which for float64 is the same thing.

   Required macros: T (the element type), K(name) (appends the suffix),
   EXP / LOG / SQRT / POW / FABS / LOG1P / EXPM1 / SIN / COS / TANH / ERF (the
   libm functions for T). */

/* [x] when it is the new maximum, or NaN: Float.max semantics without its
   signed-zero handling, exactly as fmax in tensor.ml. */
static inline T K(fmax_nan)(T acc, T v) { return (v > acc || v != v) ? v : acc; }
static inline T K(fmin_nan)(T acc, T v) { return (v < acc || v != v) ? v : acc; }

/* One element of the binary ops that have no loop of their own: minimum and
   maximum (NaN-propagating, as apply_binop in tensor.ml) and the
   comparisons, which give 0 or 1. */
static inline T K(apply)(int op, T x, T y)
{
  switch (op) {
  case BIN_MIN: return (x < y || x != x) ? x : y;
  case BIN_MAX: return (x > y || x != x) ? x : y;
  case BIN_EQ: return (T)(x == y);
  case BIN_NE: return (T)(x != y);
  case BIN_LT: return (T)(x < y);
  case BIN_LE: return (T)(x <= y);
  case BIN_GT: return (T)(x > y);
  case BIN_GE: return (T)(x >= y);
  default: return x;
  }
}

/* ------------------------------------------------------------------------ */
/* Elementwise                                                               */
/* ------------------------------------------------------------------------ */

/* The loops of [unary] and [binary], over the input pointer A. Both kernels
   expand them twice: with A = a, and with A = o for in-place calls
   (scale_, add_, axpy_...). The compiler vectorises a loop after checking at
   run time that the arrays don't overlap; [o == a] fails that check, which
   left in-place ops on the scalar path, 1.5x slower than the allocating ones.
   Written as o[i] = f(o[i]), the in-place loop needs no such check. */
#define UNARY_LOOPS(A)                                                      \
  switch (op) {                                                             \
  case UN_COPY:                                                             \
    if (o != A) memcpy(o + lo, A + lo, (size_t)(hi - lo) * sizeof(T));      \
    break;                                                                  \
  case UN_NEG:                                                              \
    for (i = lo; i < hi; i++) o[i] = -A[i];                                 \
    break;                                                                  \
  case UN_SCALE:                                                            \
    for (i = lo; i < hi; i++) o[i] = kt * A[i];                             \
    break;                                                                  \
  case UN_ADD_SCALAR:                                                       \
    for (i = lo; i < hi; i++) o[i] = A[i] + kt;                             \
    break;                                                                  \
  case UN_POW:                                                              \
    for (i = lo; i < hi; i++) o[i] = POW(A[i], kt);                         \
    break;                                                                  \
  case UN_EXP:                                                              \
    for (i = lo; i < hi; i++) o[i] = EXP(A[i]);                             \
    break;                                                                  \
  case UN_LOG:                                                              \
    for (i = lo; i < hi; i++) o[i] = LOG(A[i]);                             \
    break;                                                                  \
  case UN_SQRT:                                                             \
    for (i = lo; i < hi; i++) o[i] = SQRT(A[i]);                            \
    break;                                                                  \
  case UN_RELU:                                                             \
    /* Branch-free, as in tensor.ml: x * (x > 0) + 0 turns -0 into +0 and   \
       keeps NaN. */                                                        \
    for (i = lo; i < hi; i++) {                                             \
      T x = A[i];                                                           \
      o[i] = x * (T)(x > 0) + (T)0;                                         \
    }                                                                       \
    break;                                                                  \
  case UN_FILL:                                                             \
    for (i = lo; i < hi; i++) o[i] = kt;                                    \
    break;                                                                  \
  case UN_ABS:                                                              \
    for (i = lo; i < hi; i++) o[i] = FABS(A[i]);                            \
    break;                                                                  \
  case UN_SIGN:                                                             \
    for (i = lo; i < hi; i++) {                                             \
      T x = A[i];                                                           \
      o[i] = x > 0 ? (T)1 : (x < 0 ? (T)-1 : x);                            \
    }                                                                       \
    break;                                                                  \
  case UN_SQUARE:                                                           \
    for (i = lo; i < hi; i++) o[i] = A[i] * A[i];                           \
    break;                                                                  \
  case UN_RSQRT:                                                            \
    for (i = lo; i < hi; i++) o[i] = (T)1 / SQRT(A[i]);                     \
    break;                                                                  \
  case UN_LOG1P:                                                            \
    for (i = lo; i < hi; i++) o[i] = LOG1P(A[i]);                           \
    break;                                                                  \
  case UN_EXPM1:                                                            \
    for (i = lo; i < hi; i++) o[i] = EXPM1(A[i]);                           \
    break;                                                                  \
  case UN_SIN:                                                              \
    for (i = lo; i < hi; i++) o[i] = SIN(A[i]);                             \
    break;                                                                  \
  case UN_COS:                                                              \
    for (i = lo; i < hi; i++) o[i] = COS(A[i]);                             \
    break;                                                                  \
  case UN_TANH:                                                             \
    for (i = lo; i < hi; i++) o[i] = TANH(A[i]);                            \
    break;                                                                  \
  case UN_SIGMOID: /* as tensor.ml: never exp of a large positive number */ \
    for (i = lo; i < hi; i++) {                                             \
      T x = A[i];                                                           \
      if (x >= 0) o[i] = (T)1 / ((T)1 + EXP(-x));                           \
      else {                                                                \
        T e = EXP(x);                                                       \
        o[i] = e / ((T)1 + e);                                              \
      }                                                                     \
    }                                                                       \
    break;                                                                  \
  case UN_ERF:                                                              \
    for (i = lo; i < hi; i++) o[i] = ERF(A[i]);                             \
    break;                                                                  \
  }

#define BINARY_LOOPS(A)                                                     \
  switch (op) {                                                             \
  case BIN_ADD:                                                             \
    for (i = lo; i < hi; i++) o[i] = A[i] + b[i];                           \
    break;                                                                  \
  case BIN_SUB:                                                             \
    for (i = lo; i < hi; i++) o[i] = A[i] - b[i];                           \
    break;                                                                  \
  case BIN_MUL:                                                             \
    for (i = lo; i < hi; i++) o[i] = A[i] * b[i];                           \
    break;                                                                  \
  case BIN_DIV:                                                             \
    for (i = lo; i < hi; i++) o[i] = A[i] / b[i];                           \
    break;                                                                  \
  case BIN_RELU_BACKWARD: /* A = forward input, b = upstream gradient */    \
    for (i = lo; i < hi; i++) o[i] = b[i] * (T)(A[i] > 0);                  \
    break;                                                                  \
  case BIN_AXPY: /* o = A + k * b */                                        \
    for (i = lo; i < hi; i++) o[i] = A[i] + kt * b[i];                      \
    break;                                                                  \
  case BIN_COPY_A:                                                          \
    for (i = lo; i < hi; i++) o[i] = A[i];                                  \
    break;                                                                  \
  default:                                                                  \
    for (i = lo; i < hi; i++) o[i] = K(apply)(op, A[i], b[i]);              \
    break;                                                                  \
  }

/* o[i] = op(a[i]) for i in [lo, hi). [o] may be [a]. Each op is its own
   loop, so the compiler can vectorise it. */
static void K(unary)(int op, const T *a, T *o, intnat lo, intnat hi, double k)
{
  const T kt = (T)k;
  intnat i;
  if (o == a) {
    UNARY_LOOPS(o)
  } else {
    UNARY_LOOPS(a)
  }
}

/* o[i] = a[i] op b[i] for i in [lo, hi), all three contiguous. [o] may be
   [a]. */
static void K(binary)(int op, const T *a, const T *b, T *o, intnat lo,
                      intnat hi, double k)
{
  const T kt = (T)k;
  intnat i;
  if (o == a) {
    BINARY_LOOPS(o)
  } else {
    BINARY_LOOPS(a)
  }
}

/* One row of a broadcasting op: o[j] = a[j * sa] op b[j * sb], j < len. */
static inline void K(binary_row)(int op, const T *a, intnat sa, const T *b,
                                 intnat sb, T *o, intnat len)
{
  intnat j;
  if (sa == 1 && sb == 1) {
    K(binary)(op, a, b, o, 0, len, 0.0);
    return;
  }
  if (op == BIN_COPY_A) {
    if (sa == 1) memcpy(o, a, (size_t)len * sizeof(T));
    else if (sa == 0) for (j = 0; j < len; j++) o[j] = a[0];
    else for (j = 0; j < len; j++) o[j] = a[j * sa];
    return;
  }
  switch (op) {
  case BIN_ADD:
    for (j = 0; j < len; j++) o[j] = a[j * sa] + b[j * sb];
    break;
  case BIN_SUB:
    for (j = 0; j < len; j++) o[j] = a[j * sa] - b[j * sb];
    break;
  case BIN_MUL:
    for (j = 0; j < len; j++) o[j] = a[j * sa] * b[j * sb];
    break;
  case BIN_DIV:
    for (j = 0; j < len; j++) o[j] = a[j * sa] / b[j * sb];
    break;
  default:
    for (j = 0; j < len; j++) o[j] = K(apply)(op, a[j * sa], b[j * sb]);
    break;
  }
}

/* Rows [rlo, rhi) of a coalesced broadcasting walk (see Layout.coalesce):
   [w] holds nd dims, then the strides of a and of b over them. The output
   is contiguous; a row is one pass along the last dim. The multi-index of
   the outer dims is advanced like an odometer, so there is no division per
   row after the first. */
static void K(strided)(int op, const T *a, const T *b, T *o,
                       const struct walk *w, intnat rlo, intnat rhi)
{
  const int nd = w->nd, outer = nd - 1;
  const intnat len = w->dims[outer];
  const intnat sa = w->st[0][outer], sb = w->st[1][outer];
  intnat idx[MAX_DIMS];
  intnat oa = 0, ob = 0, r = rlo;
  int ax;
  for (ax = outer - 1; ax >= 0; ax--) {
    idx[ax] = r % w->dims[ax];
    r /= w->dims[ax];
    oa += idx[ax] * w->st[0][ax];
    ob += idx[ax] * w->st[1][ax];
  }
  for (r = rlo; r < rhi; r++) {
    K(binary_row)(op, a + oa, sa, b + ob, sb, o + r * len, len);
    for (ax = outer - 1; ax >= 0; ax--) {
      if (++idx[ax] < w->dims[ax]) {
        oa += w->st[0][ax];
        ob += w->st[1][ax];
        break;
      }
      idx[ax] = 0;
      oa -= w->st[0][ax] * (w->dims[ax] - 1);
      ob -= w->st[1][ax] * (w->dims[ax] - 1);
    }
  }
}

/* The adjoint of broadcasting (sum_to_shape), rows [rlo, rhi): every
   element of the contiguous [src] is added into the element of [dst] it maps
   to. Operand 0 of the walk is src (inner stride 1), operand 1 is dst. Rows
   are summed in order, as in tensor.ml. */
static void K(sum_rows)(const T *src, T *dst, const struct walk *w,
                        intnat rlo, intnat rhi)
{
  const int nd = w->nd, outer = nd - 1;
  const intnat len = w->dims[outer], so = w->st[1][outer];
  intnat idx[MAX_DIMS];
  intnat oi = 0, od = 0, r = rlo, j;
  int ax;
  for (ax = outer - 1; ax >= 0; ax--) {
    idx[ax] = r % w->dims[ax];
    r /= w->dims[ax];
    oi += idx[ax] * w->st[0][ax];
    od += idx[ax] * w->st[1][ax];
  }
  for (r = rlo; r < rhi; r++) {
    const T *s = src + oi;
    T *d = dst + od;
    if (so == 0) {
      double acc = d[0];
      for (j = 0; j < len; j++) acc += s[j];
      d[0] = (T)acc;
    } else if (so == 1) {
      for (j = 0; j < len; j++) d[j] += s[j];
    } else {
      for (j = 0; j < len; j++) d[j * so] += s[j];
    }
    for (ax = outer - 1; ax >= 0; ax--) {
      if (++idx[ax] < w->dims[ax]) {
        oi += w->st[0][ax];
        od += w->st[1][ax];
        break;
      }
      idx[ax] = 0;
      oi -= w->st[0][ax] * (w->dims[ax] - 1);
      od -= w->st[1][ax] * (w->dims[ax] - 1);
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Reductions                                                                */
/* ------------------------------------------------------------------------ */

/* Sum of a[lo..hi), left to right, in double. */
static double K(sum)(const T *a, intnat lo, intnat hi)
{
  double s = 0.0;
  for (intnat i = lo; i < hi; i++) s += a[i];
  return s;
}

/* Sum (mode 0), max (1) or min (2) along one axis, for outer indices
   [olo, ohi) and inner indices [ilo, ihi). The tensor is viewed as
   (outer, n, inner); [out] already holds the initial value (0, -infinity or
   infinity). The loop order is that of reduce_axis in tensor.ml. */
static void K(reduce_axis)(int mode, const T *x, T *out, intnat n,
                           intnat inner, intnat olo, intnat ohi, intnat ilo,
                           intnat ihi)
{
  const int is_max = mode == 1, is_min = mode == 2;
  for (intnat o = olo; o < ohi; o++) {
    const intnat base = o * n * inner, ob = o * inner;
    if (inner == 1) {
      if (is_max) {
        T acc = out[ob];
        for (intnat k = 0; k < n; k++) acc = K(fmax_nan)(acc, x[base + k]);
        out[ob] = acc;
      } else if (is_min) {
        T acc = out[ob];
        for (intnat k = 0; k < n; k++) acc = K(fmin_nan)(acc, x[base + k]);
        out[ob] = acc;
      } else {
        double acc = out[ob];
        for (intnat k = 0; k < n; k++) acc += x[base + k];
        out[ob] = (T)acc;
      }
    } else if (is_max) {
      for (intnat k = 0; k < n; k++) {
        const T *row = x + base + k * inner;
        T *dst = out + ob;
        for (intnat i = ilo; i < ihi; i++) dst[i] = K(fmax_nan)(dst[i], row[i]);
      }
    } else if (is_min) {
      for (intnat k = 0; k < n; k++) {
        const T *row = x + base + k * inner;
        T *dst = out + ob;
        for (intnat i = ilo; i < ihi; i++) dst[i] = K(fmin_nan)(dst[i], row[i]);
      }
    } else {
      /* Columns are summed a tile at a time into double accumulators, still
         in increasing k: exactly the same as summing in T for float64, and
         much more accurate for float32 over a long axis. */
      double acc[REDUCE_TILE];
      for (intnat i0 = ilo; i0 < ihi; i0 += REDUCE_TILE) {
        const intnat i1 = i0 + REDUCE_TILE < ihi ? i0 + REDUCE_TILE : ihi;
        for (intnat i = i0; i < i1; i++) acc[i - i0] = out[ob + i];
        for (intnat k = 0; k < n; k++) {
          const T *row = x + base + k * inner;
          for (intnat i = i0; i < i1; i++) acc[i - i0] += row[i];
        }
        for (intnat i = i0; i < i1; i++) out[ob + i] = (T)acc[i - i0];
      }
    }
  }
}

/* Index of the first maximum (or, with is_min, minimum) along the axis,
   for every (outer, inner) position j in [lo, hi), written as OCaml ints
   into [res]. */
static void K(argmax_axis)(const T *x, value res, intnat n, intnat inner,
                           intnat lo, intnat hi, int is_min)
{
  for (intnat j = lo; j < hi; j++) {
    const intnat o = j / inner, i = j % inner;
    const T *p = x + o * n * inner + i;
    intnat best = 0;
    for (intnat k = 1; k < n; k++)
      if (is_min ? p[k * inner] < p[best * inner] : p[k * inner] > p[best * inner])
        best = k;
    Field(res, j) = Val_long(best);
  }
}

/* ------------------------------------------------------------------------ */
/* Shape manipulation                                                        */
/* ------------------------------------------------------------------------ */

/* Rows [rlo, rhi) of the transpose of an [m; n] matrix, in 32 x 32 blocks
   so reads and writes both stay within a few cache lines. */
static void K(transpose)(const T *src, T *dst, intnat m, intnat n,
                         intnat rlo, intnat rhi)
{
  const intnat bs = 32;
  for (intnat i0 = rlo; i0 < rhi; i0 += bs) {
    const intnat i1 = i0 + bs < rhi ? i0 + bs : rhi;
    for (intnat j0 = 0; j0 < n; j0 += bs) {
      const intnat j1 = j0 + bs < n ? j0 + bs : n;
      for (intnat i = i0; i < i1; i++)
        for (intnat j = j0; j < j1; j++) dst[j * m + i] = src[i * n + j];
    }
  }
}

/* Rows [rlo, rhi) of a strided copy (copy_walk_ in tensor.ml): operand 0
   of the walk is [src], operand 1 is [dst], and both start at an offset.
   Strides may be negative. memmove, because set_slice_ may copy a tensor
   onto itself. */
static void K(copy_walk)(const T *src, intnat soff, T *dst, intnat doff,
                         const struct walk *w, intnat rlo, intnat rhi)
{
  const int nd = w->nd, outer = nd - 1;
  const intnat len = w->dims[outer];
  const intnat ss = w->st[0][outer], ds = w->st[1][outer];
  intnat idx[MAX_DIMS];
  intnat os = 0, od = 0, r = rlo, j;
  int ax;
  for (ax = outer - 1; ax >= 0; ax--) {
    idx[ax] = r % w->dims[ax];
    r /= w->dims[ax];
    os += idx[ax] * w->st[0][ax];
    od += idx[ax] * w->st[1][ax];
  }
  for (r = rlo; r < rhi; r++) {
    const T *s = src + soff + os;
    T *d = dst + doff + od;
    if (ss == 1 && ds == 1) memmove(d, s, (size_t)len * sizeof(T));
    else for (j = 0; j < len; j++) d[j * ds] = s[j * ss];
    for (ax = outer - 1; ax >= 0; ax--) {
      if (++idx[ax] < w->dims[ax]) {
        os += w->st[0][ax];
        od += w->st[1][ax];
        break;
      }
      idx[ax] = 0;
      os -= w->st[0][ax] * (w->dims[ax] - 1);
      od -= w->st[1][ax] * (w->dims[ax] - 1);
    }
  }
}

/* o[i] = m[i] != 0 ? a[i] : b[i], all four contiguous. */
static void K(select)(const T *m, const T *a, const T *b, T *o, intnat lo,
                      intnat hi)
{
  for (intnat i = lo; i < hi; i++) o[i] = m[i] != 0 ? a[i] : b[i];
}

/* dst row r = src row idx[r], for r in [lo, hi). [idx] is an OCaml int
   array whose entries the OCaml side has bounds-checked. */
static void K(take_rows)(const T *src, T *dst, value idx, intnat row_size,
                         intnat lo, intnat hi)
{
  for (intnat r = lo; r < hi; r++)
    memcpy(dst + r * row_size, src + Long_val(Field(idx, r)) * row_size,
           (size_t)row_size * sizeof(T));
}

/* Conversions to and from an OCaml float array (a flat block of doubles). */
static void K(of_floats)(value fa, T *dst, intnat lo, intnat hi)
{
  for (intnat i = lo; i < hi; i++) dst[i] = (T)Double_flat_field(fa, i);
}

static void K(to_floats)(const T *src, value fa, intnat lo, intnat hi)
{
  for (intnat i = lo; i < hi; i++) Store_double_flat_field(fa, i, (double)src[i]);
}

/* ------------------------------------------------------------------------ */
/* Softmax and cross-entropy (rows of length n)                              */
/* ------------------------------------------------------------------------ */

/* x - max - log (sum (exp (x - max))), one pass per row, as in tensor.ml. */
static void K(log_softmax)(const T *x, T *out, intnat n, intnat rlo,
                           intnat rhi)
{
  for (intnat r = rlo; r < rhi; r++) {
    const T *xr = x + r * n;
    T *orow = out + r * n;
    T m = -INFINITY;
    for (intnat j = 0; j < n; j++) m = K(fmax_nan)(m, xr[j]);
    double s = 0.0;
    for (intnat j = 0; j < n; j++) s += EXP(xr[j] - m);
    const T lse = (T)log(s);
    for (intnat j = 0; j < n; j++) orow[j] = xr[j] - m - lse;
  }
}

/* g - exp y * sum_row g */
static void K(log_softmax_backward)(const T *y, const T *g, T *out, intnat n,
                                    intnat rlo, intnat rhi)
{
  for (intnat r = rlo; r < rhi; r++) {
    const intnat base = r * n;
    double s = 0.0;
    for (intnat j = 0; j < n; j++) s += g[base + j];
    const T st = (T)s;
    for (intnat j = 0; j < n; j++)
      out[base + j] = g[base + j] - EXP(y[base + j]) * st;
  }
}

/* scale * (one_hot labels - exp log_probs), rows [rlo, rhi). */
static void K(softmax_ce_backward)(const T *lp, value labels, T *out,
                                   intnat c, double scale, intnat rlo,
                                   intnat rhi)
{
  const T sc = (T)scale;
  for (intnat r = rlo; r < rhi; r++) {
    const intnat label = Long_val(Field(labels, r));
    for (intnat j = 0; j < c; j++) {
      const intnat i = r * c + j;
      const T y = j == label ? sc : (T)0;
      out[i] = y - EXP(lp[i]) * sc;
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Optimizer steps                                                           */
/* ------------------------------------------------------------------------ */

/* One fused Adam step over [lo, hi): the formulas of adam_update_ in
   tensor.ml. h[8] = lr, beta1, beta2, eps, weight_decay, bc1, bc2, and
   whether the decay is decoupled (AdamW: adamw_update_). */
static void K(adam)(T *p, const T *g, T *m, T *v, const double *h, intnat lo,
                    intnat hi)
{
  const T lr = (T)h[0], b1 = (T)h[1], b2 = (T)h[2], eps = (T)h[3];
  const T wd = (T)h[4], bc1 = (T)h[5], bc2 = (T)h[6];
  const int decoupled = h[7] != 0;
  const T c1 = (T)1 - b1, c2 = (T)1 - b2, neg_lr = -lr;
  const T keep = (T)(1.0 - h[0] * h[4]);
  const int decay = wd > 0 && !decoupled;
  for (intnat i = lo; i < hi; i++) {
    const T pi = decoupled ? keep * p[i] : p[i];
    T gi = g[i];
    if (decay) gi = gi + wd * pi;
    const T mi = b1 * m[i] + c1 * gi;
    const T vi = b2 * v[i] + c2 * (gi * gi);
    m[i] = mi;
    v[i] = vi;
    const T u = mi / bc1 / (SQRT(vi / bc2) + eps);
    p[i] = pi + neg_lr * u;
  }
}

/* One SGD step: h[4] = lr, momentum, weight_decay, and whether the
   momentum is Nesterov's (sgd_nesterov_update_). */
static void K(sgd)(T *p, const T *g, T *vel, const double *h, intnat lo,
                   intnat hi)
{
  const T neg_lr = -(T)h[0], mom = (T)h[1], wd = (T)h[2];
  const int decay = wd > 0, nesterov = h[3] != 0;
  for (intnat i = lo; i < hi; i++) {
    const T pi = p[i];
    T gi = g[i];
    if (decay) gi = gi + wd * pi;
    if (nesterov) {
      const T vi = mom * vel[i] + gi;
      vel[i] = vi;
      p[i] = pi + neg_lr * (gi + mom * vi);
    }
    else if (mom == 0) p[i] = pi + neg_lr * gi;
    else {
      const T vi = mom * vel[i] + gi;
      vel[i] = vi;
      p[i] = pi + neg_lr * vi;
    }
  }
}

/* One RMSprop step (rmsprop_update_): h[6] = lr, alpha, eps, weight_decay,
   momentum, centered. ga is only touched when centered, buf only with
   momentum. */
static void K(rmsprop)(T *p, const T *g, T *sq, T *ga, T *buf, const double *h,
                       intnat lo, intnat hi)
{
  const T neg_lr = -(T)h[0], alpha = (T)h[1], eps = (T)h[2], wd = (T)h[3];
  const T mom = (T)h[4], c = (T)1 - alpha;
  const int centered = h[5] != 0, decay = wd > 0, use_mom = h[4] > 0;
  for (intnat i = lo; i < hi; i++) {
    const T pi = p[i];
    T gi = g[i];
    if (decay) gi = gi + wd * pi;
    const T si = alpha * sq[i] + c * (gi * gi);
    sq[i] = si;
    T denom;
    if (centered) {
      T ai = ga[i];
      ai = ai + c * (gi - ai);
      ga[i] = ai;
      denom = SQRT(si - ai * ai) + eps;
    }
    else denom = SQRT(si) + eps;
    if (use_mom) {
      const T bi = mom * buf[i] + gi / denom;
      buf[i] = bi;
      p[i] = pi + neg_lr * bi;
    }
    else p[i] = pi + neg_lr * (gi / denom);
  }
}

/* One Adagrad step (adagrad_update_): h[3] = lr, eps, weight_decay. */
static void K(adagrad)(T *p, const T *g, T *sum, const double *h, intnat lo,
                       intnat hi)
{
  const T neg_lr = -(T)h[0], eps = (T)h[1], wd = (T)h[2];
  const int decay = wd > 0;
  for (intnat i = lo; i < hi; i++) {
    const T pi = p[i];
    T gi = g[i];
    if (decay) gi = gi + wd * pi;
    const T si = sum[i] + gi * gi;
    sum[i] = si;
    p[i] = pi + neg_lr * (gi / (SQRT(si) + eps));
  }
}

/* ------------------------------------------------------------------------ */
/* Convolution helpers (NHWC)                                                */
/* ------------------------------------------------------------------------ */

/* The im2col window walk of tensor.ml (im2col_image), for images
   [blo, bhi). With scatter = 0 it copies image runs into the column matrix;
   with scatter = 1 it adds the columns back into the image (col2im). Each
   receptive-field row is a run of kw * c contiguous floats, clipped to the
   image at the edges. */
static void K(im2col)(int scatter, T *img, T *cols, const struct conv *g,
                      intnat blo, intnat bhi)
{
  const intnat h = g->h, w = g->w, c = g->c, kh = g->kh, kw = g->kw;
  const intnat stride = g->stride, padding = g->padding;
  const intnat oh = g->oh, ow = g->ow;
  const intnat ncols = kh * kw * c;
  for (intnat b = blo; b < bhi; b++)
    for (intnat oy = 0; oy < oh; oy++)
      for (intnat ky = 0; ky < kh; ky++) {
        const intnat y = oy * stride + ky - padding;
        if (y < 0 || y >= h) continue;
        for (intnat ox = 0; ox < ow; ox++) {
          const intnat x0 = ox * stride - padding;
          const intnat kx_lo = x0 < 0 ? -x0 : 0;
          const intnat kx_hi = w - x0 < kw ? w - x0 : kw;
          if (kx_lo >= kx_hi) continue;
          T *s = img + (((b * h + y) * w) + x0 + kx_lo) * c;
          T *d = cols + ((b * oh + oy) * ow + ox) * ncols + (ky * kw + kx_lo) * c;
          const intnat len = (kx_hi - kx_lo) * c;
          if (scatter)
            for (intnat i = 0; i < len; i++) s[i] += d[i];
          else
            memcpy(d, s, (size_t)len * sizeof(T));
        }
      }
}

/* Max pooling over images [blo, bhi), choosing each window's winner without
   a branch and recording its flat input index in [argmax] (an OCaml int
   array). Ties go to the first maximum, and a NaN in a window's first
   element wins: the rules of max_pool2d in tensor.ml. */
static void K(max_pool)(const T *x, T *out, value argmax, const struct conv *g,
                        intnat blo, intnat bhi)
{
  const intnat h = g->h, w = g->w, c = g->c, size = g->kh, stride = g->stride;
  const intnat oh = g->oh, ow = g->ow, row = w * c;
  for (intnat b = blo; b < bhi; b++)
    for (intnat oy = 0; oy < oh; oy++)
      for (intnat ox = 0; ox < ow; ox++) {
        const intnat o = ((b * oh + oy) * ow + ox) * c;
        const intnat corner = ((b * h + oy * stride) * w + ox * stride) * c;
        for (intnat ch = 0; ch < c; ch++) {
          const intnat i0 = corner + ch;
          intnat a = i0;
          if (size == 2) {
            const intnat i1 = i0 + c, i2 = i0 + row, i3 = i2 + c;
            a += (x[i1] > x[a]) * (i1 - a);
            a += (x[i2] > x[a]) * (i2 - a);
            a += (x[i3] > x[a]) * (i3 - a);
          } else {
            for (intnat ky = 0; ky < size; ky++)
              for (intnat kx = 0; kx < size; kx++) {
                const intnat i = i0 + ky * row + kx * c;
                a += (x[i] > x[a]) * (i - a);
              }
          }
          out[o + ch] = x[a];
          Field(argmax, o + ch) = Val_long(a);
        }
      }
}

/* out[argmax[o]] += g[o] for o in [lo, hi). [argmax] comes from the caller,
   so each index is checked against [size], the length of [out]; the number
   of out-of-range indices (which are skipped) is returned. */
static intnat K(max_pool_backward)(const T *g, value argmax, T *out,
                                   intnat size, intnat lo, intnat hi)
{
  intnat bad = 0;
  for (intnat o = lo; o < hi; o++) {
    const intnat i = Long_val(Field(argmax, o));
    if ((uintnat)i < (uintnat)size) out[i] += g[o];
    else bad++;
  }
  return bad;
}

/* ------------------------------------------------------------------------ */
/* Matrix multiply                                                           */
/* ------------------------------------------------------------------------ */

#ifndef FLOCKML_HAVE_CBLAS
/* The portable fallback when no CBLAS was found at build time: C = alpha *
   op(A) op(B) + beta * C with an i-p-j loop that the compiler vectorises
   along j. op(B) is copied first if it is transposed, so the inner loop
   always reads it contiguously. Correct, but far from BLAS speed. */
static void K(gemm_fallback)(int ta, int tb, intnat m, intnat n, intnat k,
                             double alpha, const T *a, intnat lda, const T *b,
                             intnat ldb, double beta, T *c)
{
  const T al = (T)alpha, be = (T)beta;
  const T *bb = b;
  T *tmp = NULL;
  if (beta == 0.0) memset(c, 0, (size_t)(m * n) * sizeof(T));
  else if (beta != 1.0) for (intnat i = 0; i < m * n; i++) c[i] *= be;
  if (k == 0) return;
  if (tb) {
    tmp = malloc((size_t)(k * n) * sizeof(T));
    if (tmp == NULL) return;
    for (intnat p = 0; p < k; p++)
      for (intnat j = 0; j < n; j++) tmp[p * n + j] = b[j * ldb + p];
    bb = tmp;
    ldb = n;
  }
  for (intnat i = 0; i < m; i++) {
    T *ci = c + i * n;
    for (intnat p = 0; p < k; p++) {
      const T aip = al * (ta ? a[p * lda + i] : a[i * lda + p]);
      const T *bp = bb + p * ldb;
      for (intnat j = 0; j < n; j++) ci[j] += aip * bp[j];
    }
  }
  free(tmp);
}
#endif

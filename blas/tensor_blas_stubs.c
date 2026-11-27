/* C stubs for Tensor_blas: the kernels in kernels.h, instantiated for
   float64 and float32, plus matrix multiply through CBLAS.

   Calling conventions:
   - Tensors arrive as OCaml Bigarrays, whose data lives outside the OCaml
     heap and never moves. Each stub dispatches on the Bigarray's kind.
   - Integer and float parameters arrive as tagged OCaml ints, or packed into
     an OCaml int array / float array, so the stubs need no boxing and every
     native stub has a trivial bytecode wrapper.
   - Every stub except gemm is [@@noalloc]: it allocates nothing, raises
     nothing and holds the runtime lock. Those stubs work on one chunk of a
     Parallel.parallel_for loop, so each call is short.
   - gemm can run for a long time on large matrices, so it releases the
     runtime lock while BLAS computes. Other domains can then collect garbage
     without waiting for it. */

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if defined(FLOCKML_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define FLOCKML_HAVE_CBLAS 1
#elif defined(FLOCKML_CBLAS)
#include <cblas.h>
#define FLOCKML_HAVE_CBLAS 1
#endif

#ifndef FLOCKML_BLAS_NAME
#define FLOCKML_BLAS_NAME "none (portable C fallback)"
#endif

/* Operation codes; they must match the ones in tensor_blas.ml. */
enum { UN_COPY, UN_NEG, UN_SCALE, UN_ADD_SCALAR, UN_POW, UN_EXP, UN_LOG,
       UN_SQRT, UN_RELU, UN_FILL, UN_ABS, UN_SIGN, UN_SQUARE, UN_RSQRT,
       UN_LOG1P, UN_EXPM1, UN_SIN, UN_COS, UN_TANH, UN_SIGMOID, UN_ERF };
enum { BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV, BIN_RELU_BACKWARD, BIN_AXPY,
       BIN_COPY_A, BIN_MIN, BIN_MAX, BIN_EQ, BIN_NE, BIN_LT, BIN_LE, BIN_GT,
       BIN_GE };

/* A coalesced walk over a shape (see Layout.coalesce): nd dims, and the
   stride of each of two operands along each dim. */
#define MAX_DIMS 32

/* Columns per tile when summing along a non-last axis (see reduce_axis). */
#define REDUCE_TILE 256
struct walk {
  int nd;
  intnat dims[MAX_DIMS];
  intnat st[2][MAX_DIMS];
};

/* Read a walk packed by the OCaml side as [| nd; dims; st0; st1 |]. */
static void read_walk(value v, struct walk *w)
{
  w->nd = (int)Long_val(Field(v, 0));
  for (int i = 0; i < w->nd; i++) {
    w->dims[i] = Long_val(Field(v, 1 + i));
    w->st[0][i] = Long_val(Field(v, 1 + w->nd + i));
    w->st[1][i] = Long_val(Field(v, 1 + 2 * w->nd + i));
  }
}

/* Convolution geometry, packed as [| h; w; c; kh; kw; stride; padding; oh;
   ow |]. */
struct conv {
  intnat h, w, c, kh, kw, stride, padding, oh, ow;
};

static void read_conv(value v, struct conv *g)
{
  g->h = Long_val(Field(v, 0));
  g->w = Long_val(Field(v, 1));
  g->c = Long_val(Field(v, 2));
  g->kh = Long_val(Field(v, 3));
  g->kw = Long_val(Field(v, 4));
  g->stride = Long_val(Field(v, 5));
  g->padding = Long_val(Field(v, 6));
  g->oh = Long_val(Field(v, 7));
  g->ow = Long_val(Field(v, 8));
}

/* ------------------------------------------------------------------------ */
/* Tensor storage: Bigarrays whose memory is recycled                        */
/* ------------------------------------------------------------------------ */

/* A training step allocates the same large buffers every time (activations,
   im2col matrices, gradients). Plain malloc hands large blocks back to the
   OS on free, so every step paid a page fault per 16 KB page of fresh
   memory: the BLAS backend's CNN step was 35% slower than the pure-OCaml one
   because of it. So the storage of every tensor comes from here instead:
   - Large blocks are recycled. When the GC finalises a tensor, its block
     goes onto a free list for its size (rounded up to 4 KB), and the next
     tensor of that size reuses it, already mapped. The free lists hold at
     most FLOCKML_POOL_MB megabytes (default 1024); past that, blocks are
     freed.
   - Small blocks (under 64 KB) use malloc and free directly: they come from
     malloc's own reused pages anyway.
   - The GC is told how much memory each tensor holds (see gc_budget
     below), so it collects garbage tensors at a pace that matches the
     allocation rate.
   The tensors are ordinary Bigarrays: only their finaliser differs. The lists
   are protected by a mutex, since finalisers may run on any domain.

   Invariant: these Bigarrays never share their memory with another Bigarray
   (no Bigarray.Array1.sub, reshape or slice of them). A view would outlive
   the block once the original's finaliser recycles it. Tensor_blas.t is
   abstract and the library takes no views, so nothing outside can either. */

/* The runtime's own Bigarray operations. They are exported, but declared
   only for runtime-internal code (under CAML_INTERNALS), so declare them
   here. */
CAMLextern int caml_ba_element_size[];
CAMLextern int caml_ba_compare(value v1, value v2);
CAMLextern intnat caml_ba_hash(value v);
CAMLextern void caml_ba_serialize(value, uintnat *, uintnat *);
CAMLextern uintnat caml_ba_deserialize(void *dst);

#define POOL_MIN_BYTES ((size_t)64 * 1024)
#define POOL_BUCKETS 256

struct pool_block {
  struct pool_block *next;
};

struct pool_bucket {
  size_t size; /* 0 when unused */
  struct pool_block *free;
};

static struct pool_bucket pool_buckets[POOL_BUCKETS];
static size_t pool_bytes = 0, pool_limit = 0;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t pool_round(size_t bytes)
{
  if (bytes < POOL_MIN_BYTES) return bytes < 64 ? 64 : bytes;
  return (bytes + 4095) & ~(size_t)4095;
}

/* The bucket for [size], or a free slot for it; NULL if the table is full.
   Linear probing on a table that holds one entry per distinct tensor size. */
static struct pool_bucket *pool_find(size_t size, int create)
{
  size_t h = (size >> 12) * 2654435761u;
  for (int i = 0; i < POOL_BUCKETS; i++) {
    struct pool_bucket *b = &pool_buckets[(h + (size_t)i) % POOL_BUCKETS];
    if (b->size == size) return b;
    if (b->size == 0) {
      if (!create) return NULL;
      b->size = size;
      return b;
    }
  }
  return NULL;
}

static void *pool_take(size_t size)
{
  void *p = NULL;
  if (size >= POOL_MIN_BYTES) {
    pthread_mutex_lock(&pool_lock);
    struct pool_bucket *b = pool_find(size, 0);
    if (b != NULL && b->free != NULL) {
      p = b->free;
      b->free = b->free->next;
      pool_bytes -= size;
    }
    pthread_mutex_unlock(&pool_lock);
    if (p != NULL) return p;
  }
  /* 64-byte alignment: a cache line, and enough for any SIMD load. */
  if (posix_memalign(&p, 64, size) != 0) return NULL;
  return p;
}

static void pool_give(void *p, size_t size)
{
  if (size >= POOL_MIN_BYTES) {
    pthread_mutex_lock(&pool_lock);
    if (pool_limit == 0) {
      const char *env = getenv("FLOCKML_POOL_MB");
      long mb = env != NULL ? atol(env) : 1024;
      pool_limit = (size_t)(mb > 0 ? mb : 0) * 1024 * 1024 + 1;
    }
    struct pool_bucket *b = pool_find(size, 1);
    if (b != NULL && pool_bytes + size < pool_limit) {
      struct pool_block *blk = p;
      blk->next = b->free;
      b->free = blk;
      pool_bytes += size;
      p = NULL;
    }
    pthread_mutex_unlock(&pool_lock);
  }
  free(p);
}

static void fb_finalize(value v)
{
  struct caml_ba_array *b = Caml_ba_array_val(v);
  if (b->data != NULL) pool_give(b->data, pool_round(caml_ba_byte_size(b)));
  b->data = NULL;
}

/* Bigarray operations, with the recycling finaliser. The identifier is the
   standard one, so marshalling and unmarshalling work as for any Bigarray. */
static struct custom_operations fb_ba_ops = {
  "_bigarr02",
  fb_finalize,
  caml_ba_compare,
  caml_ba_hash,
  caml_ba_serialize,
  caml_ba_deserialize,
  custom_compare_ext_default,
  custom_fixed_length_default
};

/* How the GC is told about tensor memory. caml_alloc_custom (ops, size,
   mem, max) counts each new tensor as mem / max of a minor collection, and
   again of a major cycle if it survives into the major heap. Most tensors
   are temporaries that die within a training step, and a minor collection
   finalises those cheaply, so their blocks return to the pool promptly.
   (caml_alloc_custom_mem would instead charge every tensor to the major GC
   in proportion to the OCaml heap. This backend's heap is tiny, since even
   the dataset lives in a Bigarray, so that forced about 7 major cycles per
   CNN step.) So: a minor collection, and for survivors a major cycle, per
   FLOCKML_GC_BUDGET_MB megabytes of tensors allocated (default 128). */
static mlsize_t gc_budget_bytes = 0;

static mlsize_t gc_budget(void)
{
  if (gc_budget_bytes == 0) {
    const char *env = getenv("FLOCKML_GC_BUDGET_MB");
    long mb = env != NULL ? atol(env) : 128;
    gc_budget_bytes = (mlsize_t)(mb > 0 ? mb : 128) * 1024 * 1024;
  }
  return gc_budget_bytes;
}

/* A 1-D C-layout Bigarray of [n] uninitialised elements of [kind]
   (Bigarray.float32 or float64). */
CAMLprim value fb_create(value kind, value vn)
{
  const int k = Int_val(kind);
  const intnat n = Long_val(vn);
  const size_t bytes = (size_t)n * (size_t)caml_ba_element_size[k];
  const size_t size = pool_round(bytes);
  void *data = pool_take(size);
  if (data == NULL) caml_raise_out_of_memory();
  value res = caml_alloc_custom(&fb_ba_ops,
                                sizeof(struct caml_ba_array) + sizeof(intnat),
                                size, gc_budget());
  struct caml_ba_array *b = Caml_ba_array_val(res);
  b->data = data;
  b->num_dims = 1;
  b->flags = k | CAML_BA_C_LAYOUT | CAML_BA_MANAGED;
  b->proxy = NULL;
  b->dim[0] = n;
  return res;
}

/* ------------------------------------------------------------------------ */
/* The kernels, once per element type                                        */
/* ------------------------------------------------------------------------ */

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

#define T double
#define K(name) CAT(name, _d)
#define EXP exp
#define LOG log
#define SQRT sqrt
#define POW pow
#define FABS fabs
#define LOG1P log1p
#define EXPM1 expm1
#define SIN sin
#define COS cos
#define TANH tanh
#define ERF erf
#include "kernels.h"
#undef T
#undef K
#undef EXP
#undef LOG
#undef SQRT
#undef POW
#undef FABS
#undef LOG1P
#undef EXPM1
#undef SIN
#undef COS
#undef TANH
#undef ERF

#define T float
#define K(name) CAT(name, _s)
#define EXP expf
#define LOG logf
#define SQRT sqrtf
#define POW powf
#define FABS fabsf
#define LOG1P log1pf
#define EXPM1 expm1f
#define SIN sinf
#define COS cosf
#define TANH tanhf
#define ERF erff
#include "kernels.h"
#undef T
#undef K
#undef EXP
#undef LOG
#undef SQRT
#undef POW
#undef FABS
#undef LOG1P
#undef EXPM1
#undef SIN
#undef COS
#undef TANH
#undef ERF

/* ------------------------------------------------------------------------ */
/* Dispatch                                                                  */
/* ------------------------------------------------------------------------ */

#define IS_F32(v) ((Caml_ba_array_val(v)->flags & CAML_BA_KIND_MASK) == CAML_BA_FLOAT32)
#define D(v) ((double *)Caml_ba_data_val(v))
#define S(v) ((float *)Caml_ba_data_val(v))

/* Bytecode wrappers: forward the argv array to the native stub. */
#define BYTE6(f) \
  CAMLprim value f##_byte(value *a, int n) { (void)n; return f(a[0], a[1], a[2], a[3], a[4], a[5]); }
#define BYTE7(f) \
  CAMLprim value f##_byte(value *a, int n) { (void)n; return f(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); }
#define BYTE8(f) \
  CAMLprim value f##_byte(value *a, int n) { (void)n; return f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); }

CAMLprim value fb_blas_name(value unit)
{
  (void)unit;
  return caml_copy_string(FLOCKML_BLAS_NAME);
}

CAMLprim value fb_unary(value op, value src, value dst, value lo, value hi,
                        value k)
{
  if (IS_F32(dst))
    unary_s(Int_val(op), S(src), S(dst), Long_val(lo), Long_val(hi),
            Double_flat_field(k, 0));
  else
    unary_d(Int_val(op), D(src), D(dst), Long_val(lo), Long_val(hi),
            Double_flat_field(k, 0));
  return Val_unit;
}
BYTE6(fb_unary)

CAMLprim value fb_binary(value op, value a, value b, value o, value lo,
                         value hi, value k)
{
  if (IS_F32(o))
    binary_s(Int_val(op), S(a), S(b), S(o), Long_val(lo), Long_val(hi),
             Double_flat_field(k, 0));
  else
    binary_d(Int_val(op), D(a), D(b), D(o), Long_val(lo), Long_val(hi),
             Double_flat_field(k, 0));
  return Val_unit;
}
BYTE7(fb_binary)

CAMLprim value fb_strided(value op, value a, value b, value o, value walk,
                          value rlo, value rhi)
{
  struct walk w;
  read_walk(walk, &w);
  if (IS_F32(o))
    strided_s(Int_val(op), S(a), S(b), S(o), &w, Long_val(rlo), Long_val(rhi));
  else
    strided_d(Int_val(op), D(a), D(b), D(o), &w, Long_val(rlo), Long_val(rhi));
  return Val_unit;
}
BYTE7(fb_strided)

CAMLprim value fb_sum_rows(value src, value dst, value walk, value rlo,
                           value rhi)
{
  struct walk w;
  read_walk(walk, &w);
  if (IS_F32(src))
    sum_rows_s(S(src), S(dst), &w, Long_val(rlo), Long_val(rhi));
  else
    sum_rows_d(D(src), D(dst), &w, Long_val(rlo), Long_val(rhi));
  return Val_unit;
}

double fb_sum(value a, value lo, value hi)
{
  if (IS_F32(a)) return sum_s(S(a), Long_val(lo), Long_val(hi));
  return sum_d(D(a), Long_val(lo), Long_val(hi));
}

CAMLprim value fb_sum_byte(value a, value lo, value hi)
{
  return caml_copy_double(fb_sum(a, lo, hi));
}

/* mode: 0 sum, 1 max, 2 min. geo = [| n; inner; olo; ohi; ilo; ihi |] */
CAMLprim value fb_reduce_axis(value mode, value x, value out, value geo)
{
  const intnat n = Long_val(Field(geo, 0)), inner = Long_val(Field(geo, 1));
  const intnat olo = Long_val(Field(geo, 2)), ohi = Long_val(Field(geo, 3));
  const intnat ilo = Long_val(Field(geo, 4)), ihi = Long_val(Field(geo, 5));
  if (IS_F32(x))
    reduce_axis_s(Int_val(mode), S(x), S(out), n, inner, olo, ohi, ilo, ihi);
  else
    reduce_axis_d(Int_val(mode), D(x), D(out), n, inner, olo, ohi, ilo, ihi);
  return Val_unit;
}

/* geo = [| n; inner; lo; hi; is_min |] */
CAMLprim value fb_argmax_axis(value x, value res, value geo)
{
  const intnat n = Long_val(Field(geo, 0)), inner = Long_val(Field(geo, 1));
  const intnat lo = Long_val(Field(geo, 2)), hi = Long_val(Field(geo, 3));
  const int is_min = (int)Long_val(Field(geo, 4));
  if (IS_F32(x)) argmax_axis_s(S(x), res, n, inner, lo, hi, is_min);
  else argmax_axis_d(D(x), res, n, inner, lo, hi, is_min);
  return Val_unit;
}

/* offs = [| src offset; dst offset |]; rows [rlo, rhi) of the walk. */
CAMLprim value fb_copy_walk(value src, value dst, value walk, value offs,
                            value rlo, value rhi)
{
  struct walk w;
  read_walk(walk, &w);
  const intnat so = Long_val(Field(offs, 0)), dof = Long_val(Field(offs, 1));
  if (IS_F32(dst))
    copy_walk_s(S(src), so, S(dst), dof, &w, Long_val(rlo), Long_val(rhi));
  else
    copy_walk_d(D(src), so, D(dst), dof, &w, Long_val(rlo), Long_val(rhi));
  return Val_unit;
}
BYTE6(fb_copy_walk)

CAMLprim value fb_select(value m, value a, value b, value o, value lo,
                         value hi)
{
  if (IS_F32(o)) select_s(S(m), S(a), S(b), S(o), Long_val(lo), Long_val(hi));
  else select_d(D(m), D(a), D(b), D(o), Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE6(fb_select)

/* geo = [| m; n |]; rows [rlo, rhi) of the source. */
CAMLprim value fb_transpose(value src, value dst, value geo, value rlo,
                            value rhi)
{
  const intnat m = Long_val(Field(geo, 0)), n = Long_val(Field(geo, 1));
  if (IS_F32(src)) transpose_s(S(src), S(dst), m, n, Long_val(rlo), Long_val(rhi));
  else transpose_d(D(src), D(dst), m, n, Long_val(rlo), Long_val(rhi));
  return Val_unit;
}

CAMLprim value fb_take_rows(value src, value dst, value idx, value row_size,
                            value lo, value hi)
{
  if (IS_F32(src))
    take_rows_s(S(src), S(dst), idx, Long_val(row_size), Long_val(lo), Long_val(hi));
  else
    take_rows_d(D(src), D(dst), idx, Long_val(row_size), Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE6(fb_take_rows)

CAMLprim value fb_of_floats(value fa, value dst, value lo, value hi)
{
  if (IS_F32(dst)) of_floats_s(fa, S(dst), Long_val(lo), Long_val(hi));
  else of_floats_d(fa, D(dst), Long_val(lo), Long_val(hi));
  return Val_unit;
}

CAMLprim value fb_to_floats(value src, value fa, value lo, value hi)
{
  if (IS_F32(src)) to_floats_s(S(src), fa, Long_val(lo), Long_val(hi));
  else to_floats_d(D(src), fa, Long_val(lo), Long_val(hi));
  return Val_unit;
}

CAMLprim value fb_log_softmax(value x, value out, value n, value rlo,
                              value rhi)
{
  if (IS_F32(x))
    log_softmax_s(S(x), S(out), Long_val(n), Long_val(rlo), Long_val(rhi));
  else
    log_softmax_d(D(x), D(out), Long_val(n), Long_val(rlo), Long_val(rhi));
  return Val_unit;
}

CAMLprim value fb_log_softmax_backward(value y, value g, value out, value n,
                                       value rlo, value rhi)
{
  if (IS_F32(y))
    log_softmax_backward_s(S(y), S(g), S(out), Long_val(n), Long_val(rlo),
                           Long_val(rhi));
  else
    log_softmax_backward_d(D(y), D(g), D(out), Long_val(n), Long_val(rlo),
                           Long_val(rhi));
  return Val_unit;
}
BYTE6(fb_log_softmax_backward)

CAMLprim value fb_softmax_ce_backward(value lp, value labels, value out,
                                      value c, value scale, value rlo,
                                      value rhi)
{
  if (IS_F32(lp))
    softmax_ce_backward_s(S(lp), labels, S(out), Long_val(c),
                          Double_flat_field(scale, 0), Long_val(rlo),
                          Long_val(rhi));
  else
    softmax_ce_backward_d(D(lp), labels, D(out), Long_val(c),
                          Double_flat_field(scale, 0), Long_val(rlo),
                          Long_val(rhi));
  return Val_unit;
}
BYTE7(fb_softmax_ce_backward)

CAMLprim value fb_adam(value p, value g, value m, value v, value h, value lo,
                       value hi)
{
  const double *hp = (const double *)h;
  if (IS_F32(p))
    adam_s(S(p), S(g), S(m), S(v), hp, Long_val(lo), Long_val(hi));
  else
    adam_d(D(p), D(g), D(m), D(v), hp, Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE7(fb_adam)

CAMLprim value fb_sgd(value p, value g, value vel, value h, value lo,
                      value hi)
{
  const double *hp = (const double *)h;
  if (IS_F32(p)) sgd_s(S(p), S(g), S(vel), hp, Long_val(lo), Long_val(hi));
  else sgd_d(D(p), D(g), D(vel), hp, Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE6(fb_sgd)

CAMLprim value fb_rmsprop(value p, value g, value sq, value ga, value buf,
                          value h, value lo, value hi)
{
  const double *hp = (const double *)h;
  if (IS_F32(p))
    rmsprop_s(S(p), S(g), S(sq), S(ga), S(buf), hp, Long_val(lo), Long_val(hi));
  else
    rmsprop_d(D(p), D(g), D(sq), D(ga), D(buf), hp, Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE8(fb_rmsprop)

CAMLprim value fb_adagrad(value p, value g, value sum, value h, value lo,
                          value hi)
{
  const double *hp = (const double *)h;
  if (IS_F32(p)) adagrad_s(S(p), S(g), S(sum), hp, Long_val(lo), Long_val(hi));
  else adagrad_d(D(p), D(g), D(sum), hp, Long_val(lo), Long_val(hi));
  return Val_unit;
}
BYTE6(fb_adagrad)

CAMLprim value fb_im2col(value scatter, value img, value cols, value geo,
                         value blo, value bhi)
{
  struct conv g;
  read_conv(geo, &g);
  if (IS_F32(img))
    im2col_s(Bool_val(scatter), S(img), S(cols), &g, Long_val(blo), Long_val(bhi));
  else
    im2col_d(Bool_val(scatter), D(img), D(cols), &g, Long_val(blo), Long_val(bhi));
  return Val_unit;
}
BYTE6(fb_im2col)

CAMLprim value fb_max_pool(value x, value out, value argmax, value geo,
                           value blo, value bhi)
{
  struct conv g;
  read_conv(geo, &g);
  if (IS_F32(x))
    max_pool_s(S(x), S(out), argmax, &g, Long_val(blo), Long_val(bhi));
  else
    max_pool_d(D(x), D(out), argmax, &g, Long_val(blo), Long_val(bhi));
  return Val_unit;
}
BYTE6(fb_max_pool)

/* Returns the number of out-of-range indices in argmax[lo..hi). */
CAMLprim value fb_max_pool_backward(value g, value argmax, value out,
                                    value size, value lo, value hi)
{
  intnat bad;
  if (IS_F32(g))
    bad = max_pool_backward_s(S(g), argmax, S(out), Long_val(size),
                              Long_val(lo), Long_val(hi));
  else
    bad = max_pool_backward_d(D(g), argmax, D(out), Long_val(size),
                              Long_val(lo), Long_val(hi));
  return Val_long(bad);
}
BYTE6(fb_max_pool_backward)

/* ------------------------------------------------------------------------ */
/* gemm                                                                      */
/* ------------------------------------------------------------------------ */

/* c <- alpha * op(a) op(b) + beta * c. dims = [| ta; tb; m; n; k; lda; ldb
   |], coef = [| alpha; beta |]; the OCaml side has checked every shape. The
   runtime lock is released for products big enough to take a while; the
   CAMLparam registration keeps the three Bigarrays alive meanwhile. */
CAMLprim value fb_gemm(value a, value b, value c, value dims, value coef)
{
  CAMLparam5(a, b, c, dims, coef);
  const int ta = (int)Long_val(Field(dims, 0)), tb = (int)Long_val(Field(dims, 1));
  const intnat m = Long_val(Field(dims, 2)), n = Long_val(Field(dims, 3));
  const intnat k = Long_val(Field(dims, 4));
  const intnat lda = Long_val(Field(dims, 5)), ldb = Long_val(Field(dims, 6));
  const double alpha = Double_flat_field(coef, 0), beta = Double_flat_field(coef, 1);
  const int f32 = IS_F32(c);
  void *pa = Caml_ba_data_val(a), *pb = Caml_ba_data_val(b), *pc = Caml_ba_data_val(c);
  const int release = (double)m * (double)n * (double)k > 32768.0;
  if (m == 0 || n == 0) CAMLreturn(Val_unit);
  if (release) caml_enter_blocking_section();
#ifdef FLOCKML_HAVE_CBLAS
  if (k == 0) {
    /* BLAS wants lda >= max(1, k); with nothing to sum, c = beta * c. */
    const intnat mn = m * n;
    if (f32) {
      float *cc = pc;
      for (intnat i = 0; i < mn; i++) cc[i] = beta == 0.0 ? 0.0f : (float)beta * cc[i];
    } else {
      double *cc = pc;
      for (intnat i = 0; i < mn; i++) cc[i] = beta == 0.0 ? 0.0 : beta * cc[i];
    }
  } else if (f32) {
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, (int)m, (int)n, (int)k,
                (float)alpha, pa, (int)lda, pb, (int)ldb, (float)beta, pc, (int)n);
  } else {
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, (int)m, (int)n, (int)k, alpha,
                pa, (int)lda, pb, (int)ldb, beta, pc, (int)n);
  }
#else
  if (f32) gemm_fallback_s(ta, tb, m, n, k, alpha, pa, lda, pb, ldb, beta, pc);
  else gemm_fallback_d(ta, tb, m, n, k, alpha, pa, lda, pb, ldb, beta, pc);
#endif
  if (release) caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

/* A batch of products c_i <- op(a_i) op(b_i), one gemm per batch. dims =
   [| ta; tb; m; n; k; lda; ldb; count |], and offs holds, for each batch, the
   element offsets of a_i, b_i and c_i. c is overwritten (beta = 0). */
CAMLprim value fb_gemm_batched(value a, value b, value c, value dims,
                               value coef, value offs)
{
  CAMLparam5(a, b, c, dims, coef);
  CAMLxparam1(offs);
  const int ta = (int)Long_val(Field(dims, 0)), tb = (int)Long_val(Field(dims, 1));
  const intnat m = Long_val(Field(dims, 2)), n = Long_val(Field(dims, 3));
  const intnat k = Long_val(Field(dims, 4));
  const intnat lda = Long_val(Field(dims, 5)), ldb = Long_val(Field(dims, 6));
  const intnat count = Long_val(Field(dims, 7));
  const double alpha = Double_flat_field(coef, 0);
  const int f32 = IS_F32(c);
  const size_t es = f32 ? sizeof(float) : sizeof(double);
  char *pa = Caml_ba_data_val(a), *pb = Caml_ba_data_val(b), *pc = Caml_ba_data_val(c);
  intnat *off = malloc((size_t)(3 * count + 1) * sizeof(intnat));
  if (off == NULL) caml_raise_out_of_memory();
  for (intnat i = 0; i < 3 * count; i++) off[i] = Long_val(Field(offs, i));
  const int release = (double)count * (double)m * (double)n * (double)k > 32768.0;
  if (m > 0 && n > 0) {
    if (release) caml_enter_blocking_section();
    for (intnat i = 0; i < count; i++) {
      void *ai = pa + off[3 * i] * es, *bi = pb + off[3 * i + 1] * es;
      void *ci = pc + off[3 * i + 2] * es;
#ifdef FLOCKML_HAVE_CBLAS
      if (k == 0) memset(ci, 0, (size_t)(m * n) * es);
      else if (f32)
        cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                    tb ? CblasTrans : CblasNoTrans, (int)m, (int)n, (int)k,
                    (float)alpha, ai, (int)lda, bi, (int)ldb, 0.0f, ci, (int)n);
      else
        cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                    tb ? CblasTrans : CblasNoTrans, (int)m, (int)n, (int)k,
                    alpha, ai, (int)lda, bi, (int)ldb, 0.0, ci, (int)n);
#else
      if (f32) gemm_fallback_s(ta, tb, m, n, k, alpha, ai, lda, bi, ldb, 0.0, ci);
      else gemm_fallback_d(ta, tb, m, n, k, alpha, ai, lda, bi, ldb, 0.0, ci);
#endif
    }
    if (release) caml_leave_blocking_section();
  }
  free(off);
  CAMLreturn(Val_unit);
}
BYTE6(fb_gemm_batched)

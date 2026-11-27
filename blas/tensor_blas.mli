(** An optional tensor backend on Bigarrays, with matrix multiply in BLAS.

    It implements the same {!Flockml.Tensor_intf.S} as the default [float array]
    backend, so the whole library runs on it through {!Flockml.Make}:
    {[
    module B = Flockml.Make (Tensor_blas.F32)

    let model =
      B.Layer.sequential
        [
          B.Layer.linear ~in_dim:784 ~out_dim:128 ();
          B.Layer.relu;
          B.Layer.linear ~in_dim:128 ~out_dim:10 ();
        ]
    ]}

    How it computes:
    - Storage is a [Bigarray.Array1], outside the OCaml heap. It never moves, so
      C code can work on it directly.
    - {!S.gemm_} and {!S.matmul} call [cblas_dgemm] / [cblas_sgemm] of the BLAS
      found at build time ({!blas}): Accelerate on macOS, OpenBLAS or another
      CBLAS elsewhere. BLAS uses its own threads; set [VECLIB_MAXIMUM_THREADS]
      (Accelerate) or [OPENBLAS_NUM_THREADS] to limit them.
    - Every other kernel (elementwise ops, broadcasting, reductions, softmax,
      [im2col], pooling, the fused Adam and SGD steps) is C compiled with [-O3],
      so the compiler vectorises it. Each is split across domains with
      {!Flockml.Parallel}, so as in the default backend the result doesn't
      depend on the thread count.
    - Cold paths ([map], [map2], [pp], [allclose], random sampling) convert to
      the default backend and back. Random tensors are therefore the same as
      {!Flockml.Tensor}'s for the same generator.

    In float64 most kernels perform the same operations in the same order as
    {!Flockml.Tensor}, and agree with it to rounding (often to the bit). BLAS
    sums in its own order, so matrix products agree to about 1e-15 relative. *)

val blas : string
(** The BLAS library [gemm_] calls: ["Accelerate"], ["OpenBLAS"], the name of
    another CBLAS library, or ["none (portable C fallback)"] when none was found
    at build time. Set the [FLOCKML_BLAS] environment variable while building to
    force a choice; see [blas/config]. *)

(** The backend for one element type. *)
module type S = sig
  include Flockml.Tensor_intf.S

  val dtype : string
  (** ["float64"] or ["float32"]. Values cross the interface as OCaml [float]s
      and are rounded to the storage type. *)

  val of_tensor : Flockml.Tensor.t -> t
  (** A copy of a default-backend tensor, rounded to {!dtype}. *)

  val to_tensor : t -> Flockml.Tensor.t
  (** A copy as a default-backend tensor. *)
end

module F64 : S
(** Double precision: the same numbers as the default backend, faster. *)

module F32 : S
(** Single precision: half the memory traffic and twice the SIMD width, so about
    twice as fast again on large ops. Use [F64] for gradient checks. *)

(** Shape and stride arithmetic for contiguous row-major tensors.

    These are the backend-independent parts of broadcasting: which shape two
    operands broadcast to, the strides that read a small operand as if it had
    the broadcast shape, and the simplification of such a walk into as few, long
    rows as possible. Both {!Tensor} and the optional BLAS backend
    ([flockml.blas]) are built on them, so they broadcast identically. A new
    backend can use them too. *)

val numel_of_shape : int array -> int
(** The number of elements a tensor of this shape holds (1 for [[||]]). *)

val shape_to_string : int array -> string
(** A shape as ["[2; 3]"], for error messages. *)

val strides : int array -> int array
(** How far apart consecutive indices of each axis are in row-major storage:
    [strides [| 2; 3; 4 |] = [| 12; 4; 1 |]]. *)

val broadcast_shapes : int array -> int array -> int array
(** The NumPy broadcasting rule: right-align the shapes; on each axis the sizes
    must be equal or one of them 1, and the result takes the larger. Raises
    [Invalid_argument] if the shapes are incompatible. *)

val broadcast_strides : src:int array -> out:int array -> int array
(** The strides for reading a tensor of shape [src] as if it had the larger
    shape [out]: its own strides, with 0 on every axis that broadcasting repeats
    (missing leading axes and axes of size 1). *)

val coalesce : int array -> int array array -> int array * int array array
(** [coalesce dims strides] simplifies a walk over [dims] in which operand [o]
    moves by [strides.(o)]: axes of size 1 are dropped, and an axis is merged
    into the one before it whenever every operand moves through the pair
    contiguously. The visiting order stays row-major, but the innermost axis
    becomes as long as possible, so [[50176; 8] + [8]] becomes 50176 rows of 8,
    and [[64; 128] + [64; 128]] one row of 8192. Always returns at least one
    axis. *)

type batched_matmul = {
  rows_a : int;  (** the stored rows of each of [a]'s matrices *)
  cols_a : int;  (** and its stored columns *)
  rows_b : int;
  cols_b : int;
  m : int;  (** rows of each result matrix *)
  n : int;  (** columns of each result matrix *)
  k : int;  (** the length of each dot product *)
  batch_a : int array;  (** [a]'s leading (batch) axes *)
  batch_b : int array;
  batch : int array;  (** the broadcast batch shape of the result *)
}
(** The shapes of a batched matrix product (see [Tensor_intf.S.bmm]). *)

val batched_matmul :
  fn:string ->
  trans_a:bool ->
  trans_b:bool ->
  int array ->
  int array ->
  batched_matmul
(** The geometry of [op(a) op(b)] for operands of the given shapes, each with at
    least 2 axes. Raises [Invalid_argument], naming function [fn], if they can't
    be multiplied. *)

val batch_offset : batch:int array -> strides:int array -> int -> int
(** [batch_offset ~batch ~strides i]: where matrix [i] (in row-major order over
    [batch]) of an operand starts, in matrices, given that operand's
    {!broadcast_strides} over [batch]. *)

(** The signature every numerical backend must satisfy.

    {!Autograd.Make} is a functor over this signature. The default backend is
    {!Tensor} (a plain [float array]), but you can later write a second backend
    (Bigarray, BLAS bindings, Owl, a GPU...) and get autograd, layers and
    optimizers for it by applying the functor again.

    Conventions used throughout:
    - Storage is row-major (C order) and contiguous.
    - A scalar is a tensor of shape [[||]] holding exactly one element.
    - Binary elementwise ops follow NumPy broadcasting rules.
    - Image tensors are NHWC: [[| batch; height; width; channels |]]. *)

module type S = sig
  type t
  (** An n-dimensional array of floats. *)

  (** {1 Construction and inspection} *)

  val shape : t -> int array
  val numel : t -> int
  val ndim : t -> int
  val zeros : int array -> t
  val ones : int array -> t
  val full : int array -> float -> t
  val scalar : float -> t

  val of_array : int array -> float array -> t
  (** [of_array shape data] raises [Invalid_argument] if [Array.length data] is
      not the product of [shape]. *)

  val to_array : t -> float array
  (** A fresh copy of the underlying data, in row-major order. *)

  val copy : t -> t

  val get : t -> int array -> float
  (** [get t idx] is the element at multi-index [idx]. *)

  val set : t -> int array -> float -> unit

  val item : t -> float
  (** The value of a one-element tensor. *)

  val pp : Format.formatter -> t -> unit

  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool
  (** Same shape and [|a - b| <= atol + rtol * |b|] everywhere. *)

  (** {1 Random initialisation} *)

  val rand_uniform :
    ?rng:Random.State.t -> lo:float -> hi:float -> int array -> t

  val randn : ?rng:Random.State.t -> ?mean:float -> ?std:float -> int array -> t

  (** {1 Elementwise (pure, returning new tensors)} *)

  val map : (float -> float) -> t -> t

  val map2 : (float -> float -> float) -> t -> t -> t
  (** Broadcasting elementwise combination. *)

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val neg : t -> t
  val scale : float -> t -> t
  val add_scalar : float -> t -> t
  val pow : t -> float -> t
  val exp : t -> t
  val log : t -> t
  val sqrt : t -> t

  (** {1 Broadcasting} *)

  val broadcast_shapes : int array -> int array -> int array
  (** The shape two operands broadcast to. Raises [Invalid_argument] if they are
      incompatible. *)

  val broadcast_to : t -> int array -> t

  val sum_to_shape : t -> int array -> t
  (** The reverse of {!broadcast_to}: sum over the axes that broadcasting
      expanded so the result has the given (smaller) shape. Autograd needs this
      for the backward pass of every broadcasting op. *)

  (** {1 Reductions} *)

  val sum : t -> t
  (** Sum of all elements, as a scalar tensor. *)

  val mean : t -> t
  val sum_axis : ?keepdims:bool -> t -> int -> t
  val max_axis : ?keepdims:bool -> t -> int -> t

  val argmax_axis : t -> int -> int array
  (** Index of the maximum along [axis], for every position of the other axes,
      flattened in row-major order. For a [[| n; k |]] tensor and [axis = 1]
      this is one class index per row. *)

  (** {1 Linear algebra and shape manipulation} *)

  val matmul : t -> t -> t
  (** 2-D matrix product: [[| m; k |]] x [[| k; n |]] -> [[| m; n |]]. *)

  val transpose : t -> t
  (** 2-D transpose. *)

  val reshape : t -> int array -> t

  val take_rows : t -> int array -> t
  (** [take_rows t idx] selects slices along axis 0 (used for mini-batches). *)

  val one_hot : num_classes:int -> int array -> t
  (** [[| n |]] labels -> [[| n; num_classes |]] matrix of 0s and 1s. *)

  (** {1 In-place (mutating) operations, used by optimizers} *)

  val add_ : t -> t -> unit
  (** [add_ y x] does [y <- y + x]. Shapes must match exactly. *)

  val axpy_ : alpha:float -> x:t -> y:t -> unit
  (** [y <- y + alpha * x]. *)

  val scale_ : float -> t -> unit
  val fill_ : t -> float -> unit

  val blit_ : src:t -> dst:t -> unit
  (** Copy [src]'s data into [dst]. Shapes must match. *)

  (** {1 Convolution helpers (stretch goal: CNNs)} *)

  val im2col : t -> kh:int -> kw:int -> stride:int -> padding:int -> t
  (** [[| n; h; w; c |]] -> [[| n * oh * ow; kh * kw * c |]], one row per output
      position holding the (zero-padded) receptive field, ordered by (kernel
      row, kernel column, channel). *)

  val col2im :
    t ->
    input_shape:int array ->
    kh:int ->
    kw:int ->
    stride:int ->
    padding:int ->
    t
  (** The adjoint of {!im2col}: scatter-add each row back to the input positions
      it was read from. *)

  val max_pool2d : t -> size:int -> stride:int -> t * int array
  (** NHWC max pooling. Also returns, for each output element, the flat index of
      the input element that won (needed for the backward pass). *)

  val max_pool2d_backward :
    grad_out:t -> argmax:int array -> input_shape:int array -> t
end

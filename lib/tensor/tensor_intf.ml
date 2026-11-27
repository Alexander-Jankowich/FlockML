(** The signature every numerical backend must satisfy.

    {!Autograd.Make} is a functor over this signature. The default backend is
    {!Tensor} (a plain [float array]); any other backend (Bigarray, BLAS
    bindings, a GPU...) that implements it gets autograd, layers and optimizers
    for free by applying the functor again.

    Conventions used throughout:
    - Storage is row-major (C order) and contiguous.
    - A scalar is a tensor of shape [[||]] holding exactly one element.
    - Binary elementwise ops follow NumPy broadcasting rules.
    - Image tensors are NHWC: [[| batch; height; width; channels |]].
    - Functions without a trailing underscore are pure and return new tensors;
      functions ending in [_] mutate their argument in place. *)

module type S = sig
  type t
  (** An n-dimensional array of floats. *)

  (** {1 Construction and inspection} *)

  val shape : t -> int array
  (** The size of each axis. Returns a copy, so mutating it is harmless. *)

  val numel : t -> int
  (** Total number of elements (the product of the shape; 1 for a scalar). *)

  val ndim : t -> int
  (** Number of axes (0 for a scalar). *)

  val zeros : int array -> t
  (** A tensor of the given shape filled with [0.]. *)

  val ones : int array -> t
  (** A tensor of the given shape filled with [1.]. *)

  val full : int array -> float -> t
  (** [full shape x] is a tensor of the given shape with every element [x]. *)

  val scalar : float -> t
  (** A zero-dimensional tensor (shape [[||]]) holding one value. *)

  val of_array : int array -> float array -> t
  (** [of_array shape data] wraps a copy of [data], read in row-major order.
      Raises [Invalid_argument] if [Array.length data] is not the product of
      [shape]. *)

  val to_array : t -> float array
  (** A fresh copy of the underlying data, in row-major order. *)

  val copy : t -> t
  (** A deep copy: mutating the result never affects the original. *)

  val get : t -> int array -> float
  (** [get t idx] is the element at multi-index [idx]. Raises
      [Invalid_argument] if [idx] has the wrong length or is out of bounds. *)

  val set : t -> int array -> float -> unit
  (** [set t idx v] overwrites the element at multi-index [idx] with [v]. Same
      bounds checks as {!get}. *)

  val item : t -> float
  (** The value of a one-element tensor. Raises [Invalid_argument] otherwise. *)

  val pp : Format.formatter -> t -> unit
  (** Pretty-print as nested brackets, NumPy style. Tensors with more than 2000
      elements print only their shape. *)

  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool
  (** Same shape and [|a - b| <= atol + rtol * |b|] everywhere. Defaults:
      [rtol = 1e-5], [atol = 1e-8]. *)

  (** {1 Random initialisation}

      Both samplers take an optional generator; without one they seed a fresh
      generator from the system, so pass [~rng] for reproducible results. *)

  val rand_uniform :
    ?rng:Random.State.t -> lo:float -> hi:float -> int array -> t
  (** Independent uniform samples between [lo] (inclusive) and [hi]
      (exclusive). *)

  val randn : ?rng:Random.State.t -> ?mean:float -> ?std:float -> int array -> t
  (** Independent samples from the normal distribution N(mean, std²), generated
      with the Box-Muller transform. Defaults: [mean = 0.], [std = 1.]. *)

  (** {1 Elementwise (pure, returning new tensors)} *)

  val map : (float -> float) -> t -> t
  (** Apply a function to every element. *)

  val map2 : (float -> float -> float) -> t -> t -> t
  (** Broadcasting elementwise combination: the result has shape
      [broadcast_shapes (shape a) (shape b)]. *)

  val add : t -> t -> t
  (** Broadcasting elementwise [a + b]. *)

  val sub : t -> t -> t
  (** Broadcasting elementwise [a - b]. *)

  val mul : t -> t -> t
  (** Broadcasting elementwise (Hadamard) product [a * b]. Not a matrix product;
      see {!matmul}. *)

  val div : t -> t -> t
  (** Broadcasting elementwise [a / b]. *)

  val neg : t -> t
  (** Elementwise negation. *)

  val scale : float -> t -> t
  (** [scale k t] multiplies every element by [k]. *)

  val add_scalar : float -> t -> t
  (** [add_scalar k t] adds [k] to every element. *)

  val pow : t -> float -> t
  (** [pow t p] raises every element to the power [p]. *)

  val exp : t -> t
  (** Elementwise natural exponential. *)

  val log : t -> t
  (** Elementwise natural logarithm. *)

  val sqrt : t -> t
  (** Elementwise square root. *)

  val relu : t -> t
  (** Elementwise [max x 0]. A NaN input stays NaN; [-infinity] gives NaN
      (the kernel is branch-free, so it multiplies by a 0/1 mask). *)

  val relu_backward : input:t -> grad:t -> t
  (** The gradient of {!relu}: [grad] where [input > 0], else 0, computed as
      [grad * mask] (so a NaN or infinite [grad] gives NaN where
      [input <= 0]). Shapes must match exactly. *)

  (** {1 Broadcasting} *)

  val broadcast_shapes : int array -> int array -> int array
  (** The shape two operands broadcast to. Raises [Invalid_argument] if they are
      incompatible. *)

  val broadcast_to : t -> int array -> t
  (** [broadcast_to t shape] repeats [t] along size-1 and missing leading axes
      so it has exactly [shape]. Raises [Invalid_argument] if [t] does not
      broadcast to [shape]. *)

  val sum_to_shape : t -> int array -> t
  (** The reverse of {!broadcast_to}: sum over the axes that broadcasting
      expanded so the result has the given (smaller) shape. Autograd needs this
      for the backward pass of every broadcasting op. *)

  (** {1 Reductions}

      Axis arguments may be negative, counting from the end ([-1] is the last
      axis). *)

  val sum : t -> t
  (** Sum of all elements, as a scalar tensor. *)

  val mean : t -> t
  (** Mean of all elements, as a scalar tensor. *)

  val sum_axis : ?keepdims:bool -> t -> int -> t
  (** [sum_axis t axis] sums along one axis, removing it, or keeping it with
      size 1 when [keepdims] is [true] (default [false]). *)

  val max_axis : ?keepdims:bool -> t -> int -> t
  (** Like {!sum_axis}, but takes the maximum along the axis. *)

  val argmax_axis : t -> int -> int array
  (** Index of the maximum along [axis], for every position of the other axes,
      flattened in row-major order. For a [[| n; k |]] tensor and [axis = 1]
      this is one class index per row. *)

  (** {1 Linear algebra and shape manipulation} *)

  val matmul : t -> t -> t
  (** 2-D matrix product: [[| m; k |]] x [[| k; n |]] -> [[| m; n |]]. *)

  val gemm_ :
    ?trans_a:bool -> ?trans_b:bool -> ?alpha:float -> ?beta:float -> t -> t ->
    c:t -> unit
  (** General matrix multiply, in place:
      [c <- alpha * op(a) op(b) + beta * c], where [op(x)] is [x] transposed
      if the matching [trans_] flag is set. Defaults: no transposes,
      [alpha = 1], [beta = 0]. With [beta = 0], [c]'s previous contents are
      ignored. Transposes are never materialised. [c] must be 2-D with the
      result's shape and must not share storage with [a] or [b]; otherwise
      raises [Invalid_argument].

      Autograd uses it to compute [g · bᵀ] and [aᵀ · g] for the matmul
      backward pass, and to accumulate straight into an existing gradient
      ([beta = 1]). *)

  val transpose : t -> t
  (** 2-D transpose. *)

  val reshape : t -> int array -> t
  (** Same data, new shape with the same number of elements. At most one entry
      may be [-1]; its size is inferred. Raises [Invalid_argument] if the
      element counts disagree. *)

  val take_rows : t -> int array -> t
  (** [take_rows t idx] selects slices along axis 0 (used for mini-batches). *)

  val take_rows_ : t -> int array -> dst:t -> unit
  (** [take_rows_ t idx ~dst] is {!take_rows} written into an existing
      tensor, whose shape must be [t]'s with axis 0 set to
      [Array.length idx]. Lets a training loop reuse one batch buffer. *)

  val one_hot : num_classes:int -> int array -> t
  (** [[| n |]] labels -> [[| n; num_classes |]] matrix of 0s and 1s. *)

  (** {1 Softmax and cross-entropy (along the last axis)} *)

  val log_softmax : t -> t
  (** Numerically stable [x - max - log (sum (exp (x - max)))] along the last
      axis, one fused pass per row. *)

  val log_softmax_backward : output:t -> grad:t -> t
  (** The gradient of {!log_softmax}, given its [output]:
      [grad - exp output * sum_row grad]. *)

  val softmax_cross_entropy : t -> int array -> float * t
  (** [softmax_cross_entropy logits labels], for [[| batch; classes |]]
      logits and one label per row: the mean negative log-likelihood of the
      labels, together with [log_softmax logits], which the backward pass
      needs. Raises [Invalid_argument] on a label out of range or a length
      mismatch. *)

  val softmax_cross_entropy_backward :
    log_probs:t -> labels:int array -> scale:float -> t
  (** [scale * (one_hot labels - exp log_probs)], computed per element as
      [y * scale - softmax * scale]. With [scale = -g / batch], where [g] is
      the upstream gradient of the loss, this is the gradient of
      {!softmax_cross_entropy} with respect to the logits:
      [g * (softmax - one_hot) / batch]. *)

  (** {1 In-place (mutating) operations, used by optimizers} *)

  val add_ : t -> t -> unit
  (** [add_ y x] does [y <- y + x]. Shapes must match exactly. *)

  val axpy_ : alpha:float -> x:t -> y:t -> unit
  (** [y <- y + alpha * x]. Shapes must match exactly. *)

  val scale_ : float -> t -> unit
  (** [scale_ k t] does [t <- k * t]. *)

  val fill_ : t -> float -> unit
  (** [fill_ t x] sets every element of [t] to [x]. *)

  val blit_ : src:t -> dst:t -> unit
  (** Copy [src]'s data into [dst]. Shapes must match. *)

  val adam_update_ :
    lr:float ->
    beta1:float ->
    beta2:float ->
    eps:float ->
    weight_decay:float ->
    bc1:float ->
    bc2:float ->
    param:t ->
    grad:t ->
    m:t ->
    v:t ->
    unit
  (** One Adam step for one parameter, as a single fused pass with no
      temporaries: [g' = g + weight_decay * p],
      [m <- beta1 * m + (1 - beta1) g'], [v <- beta2 * v + (1 - beta2) g'²],
      [p <- p - lr * (m / bc1) / (sqrt (v / bc2) + eps)]. [bc1] and [bc2]
      are the bias corrections [1 - beta^t]. All four tensors must have the
      same shape. *)

  val sgd_update_ :
    lr:float ->
    momentum:float ->
    weight_decay:float ->
    param:t ->
    grad:t ->
    velocity:t ->
    unit
  (** One SGD step for one parameter, in one pass:
      [g' = g + weight_decay * p]; then [p <- p - lr * g'] if [momentum = 0],
      or [v <- momentum * v + g'; p <- p - lr * v]. *)

  (** {1 Convolution helpers} *)

  val im2col : t -> kh:int -> kw:int -> stride:int -> padding:int -> t
  (** [[| n; h; w; c |]] -> [[| n * oh * ow; kh * kw * c |]], one row per output
      position holding the (zero-padded) receptive field, ordered by (kernel
      row, kernel column, channel). This turns a convolution into a single
      {!matmul}. *)

  val col2im :
    t ->
    input_shape:int array ->
    kh:int ->
    kw:int ->
    stride:int ->
    padding:int ->
    t
  (** The adjoint of {!im2col}: scatter-add each row back to the input positions
      it was read from. Overlapping windows accumulate. *)

  val max_pool2d : t -> size:int -> stride:int -> t * int array
  (** NHWC max pooling. Also returns, for each output element, the flat index of
      the input element that won (needed for the backward pass). *)

  val max_pool2d_backward :
    grad_out:t -> argmax:int array -> input_shape:int array -> t
  (** The gradient of {!max_pool2d}: a zero tensor of [input_shape] with each
      element of [grad_out] added at its window's winning position [argmax]. *)
end

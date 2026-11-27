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
      functions ending in [_] mutate their argument in place.
    - Nothing returns a view: slicing, {!S.permute} and friends copy, so a
      result never shares storage with its input.
    - Indices are OCaml [int array]s, and masks are float tensors of 0s and 1s
      (any non-zero value counts as true). *)

(** One axis of a {!S.slice}, NumPy style. Negative positions count from the end
    of the axis ([-1] is the last element). *)
type range =
  | All  (** the whole axis: [x[:]] *)
  | Index of int  (** one position, and the axis is removed: [x[i]] *)
  | Range of int * int
      (** [Range (start, stop)] is [x[start:stop]]: positions [start] up to,
          but not including, [stop]. Out-of-range bounds are clipped to the
          axis, as in NumPy, so the result may be empty. *)
  | Step of int * int * int
      (** [Step (start, stop, step)] is [x[start:stop:step]], for [step > 0].
          Use {!S.flip} to reverse an axis. *)

module type S = sig
  type t
  (** An n-dimensional array of floats. *)

  type nonrec range = range =
    | All
    | Index of int
    | Range of int * int
    | Step of int * int * int
        (** [Tensor_intf.range], re-exported so its constructors can be
            written [Tensor.All], [Tensor.Range (0, 2)]... *)

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
      elements print only their shape. In the toplevel,
      [#install_printer Flockml.Tensor.pp;;] prints tensors this way. *)

  val to_string : ?precision:int -> t -> string
  (** The same text as {!pp}, with [precision] significant digits per element
      (default 4). *)

  val arange : ?start:float -> ?step:float -> float -> t
  (** [arange ?start ?step stop] is the 1-D tensor
      [[start; start + step; ...]] of the values below [stop] (above it when
      [step < 0]), like [numpy.arange]. Defaults: [start = 0.], [step = 1.].
      Raises [Invalid_argument] if [step = 0]. *)

  val linspace : float -> float -> int -> t
  (** [linspace start stop n] is [n] evenly spaced values from [start] to [stop]
      inclusive, like [numpy.linspace]. Raises [Invalid_argument] if [n < 0]. *)

  val eye : int -> t
  (** The [n] x [n] identity matrix. *)

  val zeros_like : t -> t
  (** [zeros (shape t)]. *)

  val ones_like : t -> t
  (** [ones (shape t)]. *)

  val full_like : t -> float -> t
  (** [full (shape t) x]. *)

  val init : int array -> (int array -> float) -> t
  (** [init shape f] has [f idx] at each multi-index [idx], called in row-major
      order. The array passed to [f] is reused between calls. *)

  val allclose : ?rtol:float -> ?atol:float -> t -> t -> bool
  (** Same shape and [|a - b| <= atol + rtol * |b|] everywhere. Defaults:
      [rtol = 1e-5], [atol = 1e-8]. *)

  (** {1 Random initialisation}

      Every sampler takes an optional generator. Without one it draws from
      the global default generator, {!Flockml.Rng.default}, which starts from a
      system seed: call {!Flockml.Rng.manual_seed} once for a reproducible run, or pass
      [~rng]. *)

  val rand_uniform :
    ?rng:Random.State.t -> lo:float -> hi:float -> int array -> t
  (** Independent uniform samples between [lo] (inclusive) and [hi]
      (exclusive). *)

  val randn : ?rng:Random.State.t -> ?mean:float -> ?std:float -> int array -> t
  (** Independent samples from the normal distribution N(mean, std²), generated
      with the Box-Muller transform. Defaults: [mean = 0.], [std = 1.]. *)

  val bernoulli : ?rng:Random.State.t -> p:float -> int array -> t
  (** Independent samples that are [1.] with probability [p] and [0.]
      otherwise. Raises [Invalid_argument] unless [0 <= p <= 1]. *)

  val randint : ?rng:Random.State.t -> lo:int -> hi:int -> int array -> t
  (** Independent uniform integers between [lo] (inclusive) and [hi]
      (exclusive), stored as floats. Raises [Invalid_argument] unless
      [lo < hi]. *)

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

  val abs : t -> t
  (** Elementwise absolute value. *)

  val sign : t -> t
  (** Elementwise [1.] for positive, [-1.] for negative; zeros and NaN are
      returned unchanged. *)

  val square : t -> t
  (** Elementwise [x * x]. *)

  val rsqrt : t -> t
  (** Elementwise [1 / sqrt x]. *)

  val log1p : t -> t
  (** Elementwise [log (1 + x)], accurate for small [x]. *)

  val expm1 : t -> t
  (** Elementwise [exp x - 1], accurate for small [x]. *)

  val sin : t -> t
  (** Elementwise sine. *)

  val cos : t -> t
  (** Elementwise cosine. *)

  val tanh : t -> t
  (** Elementwise hyperbolic tangent. *)

  val sigmoid : t -> t
  (** Elementwise logistic function [1 / (1 + e^-x)], computed as
      [e^x / (1 + e^x)] for negative [x] so it never overflows. *)

  val erf : t -> t
  (** Elementwise error function (used by GELU). *)

  val clamp : ?min:float -> ?max:float -> t -> t
  (** Limit every element to [[min, max]] (either bound may be left out). NaN
      stays NaN. *)

  val minimum : t -> t -> t
  (** Broadcasting elementwise minimum. A NaN in either operand gives NaN, as
      in [numpy.minimum]. *)

  val maximum : t -> t -> t
  (** Broadcasting elementwise maximum, NaN-propagating like {!minimum}. *)

  (** {1 Comparisons and masks}

      Comparisons broadcast and return masks: [1.] where the comparison holds
      and [0.] elsewhere. Any comparison with NaN is false, except {!ne}. *)

  val eq : t -> t -> t
  (** [a = b]. *)

  val ne : t -> t -> t
  (** [a <> b]. *)

  val lt : t -> t -> t
  (** [a < b]. *)

  val le : t -> t -> t
  (** [a <= b]. *)

  val gt : t -> t -> t
  (** [a > b]. *)

  val ge : t -> t -> t
  (** [a >= b]. *)

  val where : t -> t -> t -> t
  (** [where mask a b] is [a] where [mask] is non-zero and [b] elsewhere. All
      three broadcast together. *)

  val masked_fill : t -> t -> float -> t
  (** [masked_fill t mask x] is [t] with [x] wherever [mask] (broadcast to
      [t]'s shape) is non-zero. *)

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
      this is one class index per row. Ties go to the first maximum. *)

  val max : t -> t
  (** The largest element, as a scalar ([-infinity] for an empty tensor). A NaN
      anywhere gives NaN. *)

  val min : t -> t
  (** The smallest element, as a scalar ([infinity] for an empty tensor). *)

  val min_axis : ?keepdims:bool -> t -> int -> t
  (** Like {!max_axis}, but takes the minimum. *)

  val argmin_axis : t -> int -> int array
  (** Like {!argmax_axis}, for the minimum. *)

  val mean_axis : ?keepdims:bool -> t -> int -> t
  (** Mean along one axis. *)

  val sum_axes : ?keepdims:bool -> t -> int list -> t
  (** Sum over several axes at once (in any order; repeats are ignored). With
      [[]] it is a copy. *)

  val mean_axes : ?keepdims:bool -> t -> int list -> t
  (** Mean over several axes at once. *)

  val prod_axis : ?keepdims:bool -> t -> int -> t
  (** Product along one axis. *)

  val cumsum : t -> int -> t
  (** Cumulative sum along an axis: same shape, with element [k] the sum of
      elements [0..k]. *)

  val logsumexp : ?keepdims:bool -> t -> int -> t
  (** [log (sum (exp t))] along an axis, computed stably by subtracting the
      maximum first. Rows of [-infinity] give [-infinity]. *)

  val var : ?correction:int -> ?keepdims:bool -> t -> int -> t
  (** Variance along an axis: the sum of squared deviations from the mean,
      divided by [n - correction]. The default [correction = 0] is NumPy's
      (population) variance; [~correction:1] is the unbiased estimate, PyTorch's
      default. *)

  val std : ?correction:int -> ?keepdims:bool -> t -> int -> t
  (** [sqrt (var t)], with the same arguments. *)

  (** {1 Linear algebra and shape manipulation} *)

  val matmul : t -> t -> t
  (** Matrix product, following [numpy.matmul]:
      - 2-D: [[| m; k |]] x [[| k; n |]] -> [[| m; n |]].
      - More dimensions: a batch of matrix products over the leading axes,
        which broadcast, so [[| b; m; k |]] x [[| k; n |]] multiplies each of
        the [b] matrices by the same one. See {!bmm}.
      - A 1-D operand is treated as a row vector (on the left) or a column
        vector (on the right), and that axis is removed from the result: two
        1-D vectors give their dot product as a scalar. *)

  val bmm : ?trans_a:bool -> ?trans_b:bool -> t -> t -> t
  (** Batched matrix product of two tensors with at least 2 dimensions:
      [[| ...; m; k |]] x [[| ...; k; n |]] -> [[| ...; m; n |]], where the
      leading (batch) axes broadcast. [trans_a] / [trans_b] (default [false])
      transpose the last two axes of that operand first, without a copy. *)

  val dot : t -> t -> t
  (** Dot product of two 1-D tensors of the same length, as a scalar. *)

  val outer : t -> t -> t
  (** Outer product of two 1-D tensors: [[| m |]], [[| n |]] -> [[| m; n |]]. *)

  val norm : ?p:float -> t -> t
  (** The [p]-norm of all elements, [(sum |x|^p)^(1/p)], as a scalar. Default
      [p = 2.]; [p = infinity] is the largest absolute value. *)

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

  val permute : t -> int array -> t
  (** [permute t axes] reorders the axes: axis [i] of the result is axis
      [axes.(i)] of [t]. [axes] must be a permutation of [0 .. ndim - 1]
      (negative entries count from the end). *)

  val swapaxes : t -> int -> int -> t
  (** Exchange two axes. *)

  val squeeze : ?axis:int -> t -> t
  (** Remove [axis], which must have size 1, or without [axis] every axis of
      size 1. *)

  val unsqueeze : t -> int -> t
  (** [unsqueeze t axis] inserts an axis of size 1 at position [axis], which
      may be from [-(ndim + 1)] to [ndim]. *)

  val flatten : ?first:int -> ?last:int -> t -> t
  (** Merge axes [first] to [last] (inclusive; defaults [0] and [-1]) into
      one. *)

  val concat : axis:int -> t list -> t
  (** Join tensors along an existing axis. Their other axes must match. *)

  val stack : axis:int -> t list -> t
  (** Join tensors of the same shape along a new axis, inserted at [axis]. *)

  val split : t -> axis:int -> int list -> t list
  (** [split t ~axis sizes] cuts [t] along [axis] into pieces of the given
      sizes, which must add up to the axis's length. *)

  val chunk : t -> axis:int -> int -> t list
  (** [chunk t ~axis n] cuts [t] into [n] pieces of size [ceil (d / n)] along
      [axis] (the last one smaller), like [torch.chunk]: fewer than [n] when
      [d] is small. *)

  val slice : t -> range list -> t
  (** [slice t ranges] selects along each axis in turn, like NumPy's basic
      indexing: [slice x [ Range (1, 3); All; Index 0 ]] is [x[1:3, :, 0]].
      Axes past the end of [ranges] are kept whole. Raises [Invalid_argument]
      for an out-of-range [Index], a non-positive step, or more ranges than
      axes. *)

  val set_slice_ : t -> range list -> t -> unit
  (** [set_slice_ t ranges src] overwrites the elements {!slice} would select,
      like NumPy's [t[ranges] = src]. [src] must broadcast to the slice's
      shape. *)

  val narrow : t -> axis:int -> start:int -> len:int -> t
  (** [len] positions of [axis], from [start]. Unlike {!slice}, raises
      [Invalid_argument] unless they all lie inside the axis. *)

  val pad : ?value:float -> t -> (int * int) array -> t
  (** [pad t widths] adds [fst widths.(i)] elements before and
      [snd widths.(i)] after axis [i], filled with [value] (default [0.]).
      [widths] has one pair per axis. *)

  val flip : t -> int list -> t
  (** Reverse the order of elements along each of the given axes. *)

  val tile : t -> int array -> t
  (** [tile t reps] repeats the whole tensor [reps.(i)] times along axis [i],
      like [numpy.tile]. If [reps] and the shape have different lengths, the
      shorter is padded with leading 1s. *)

  val repeat : t -> axis:int -> int -> t
  (** [repeat t ~axis n] repeats each element [n] times along [axis], like
      [numpy.repeat]: [[a; b]] becomes [[a; a; b; b]] for [n = 2]. *)

  val index_select : t -> axis:int -> int array -> t
  (** [index_select t ~axis idx] picks positions [idx] (in that order, repeats
      allowed) along [axis], like [numpy.take]. {!take_rows} is the case
      [axis = 0]. *)

  val index_add : t -> axis:int -> int array -> t -> t
  (** [index_add t ~axis idx src] is [t] with slice [i] of [src] (along [axis])
      added into slice [idx.(i)] of [t], accumulating over repeated indices.
      [src] has [t]'s shape, except that [axis] has length [Array.length idx].
      The adjoint of {!index_select}. *)

  val gather : t -> axis:int -> int array -> t
  (** Like [numpy.take_along_axis]. The result has [t]'s shape with [axis]
      resized to [k = Array.length idx / (product of the other axes)], and
      element [(..., j, ...)] is [t]'s element [(..., idx.(flat (..., j, ...)),
      ...)], where [flat] is the row-major position in the result. With [k = 1]
      it picks one element per row, e.g. the log-probability of each label. *)

  val scatter_add : t -> axis:int -> int array -> t -> t
  (** [scatter_add t ~axis idx src] is [t] with each element of [src] added at
      the position {!gather} would have read it from, accumulating over
      repeated indices. The adjoint of {!gather}. *)

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

  val adamw_update_ :
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
  (** AdamW (Loshchilov & Hutter): the weight decay is decoupled from the
      gradient. [p <- p * (1 - lr * weight_decay)], then the step of
      {!adam_update_} with the raw gradient. *)

  val sgd_nesterov_update_ :
    lr:float ->
    momentum:float ->
    weight_decay:float ->
    param:t ->
    grad:t ->
    velocity:t ->
    unit
  (** SGD with Nesterov momentum, in one pass, as PyTorch's
      [SGD(nesterov=True)]: [g' = g + weight_decay * p],
      [v <- momentum * v + g'], [p <- p - lr * (g' + momentum * v)]. *)

  val rmsprop_update_ :
    lr:float ->
    alpha:float ->
    eps:float ->
    weight_decay:float ->
    momentum:float ->
    centered:bool ->
    param:t ->
    grad:t ->
    square_avg:t ->
    grad_avg:t ->
    momentum_buf:t ->
    unit
  (** One RMSprop step, as PyTorch's: [g' = g + weight_decay * p],
      [s <- alpha * s + (1 - alpha) * g'^2]. If [centered], also
      [a <- a + (1 - alpha) * (g' - a)] and the denominator is
      [sqrt (s - a^2) + eps], otherwise [sqrt s + eps]. Then
      [p <- p - lr * g' / denom], or with momentum
      [b <- momentum * b + g' / denom; p <- p - lr * b]. [grad_avg] is only
      read when [centered], and [momentum_buf] only when [momentum > 0]: pass
      any tensor of the right shape otherwise. *)

  val adagrad_update_ :
    lr:float -> eps:float -> weight_decay:float -> param:t -> grad:t -> sum:t -> unit
  (** One Adagrad step: [g' = g + weight_decay * p], [s <- s + g'^2],
      [p <- p - lr * g' / (sqrt s + eps)]. *)

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

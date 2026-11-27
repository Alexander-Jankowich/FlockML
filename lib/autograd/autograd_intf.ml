(** Signature of a reverse-mode automatic differentiation engine.

    A value of type [t] is a node in a computational graph: it holds a tensor
    value, the operation that produced it, and (after {!S.backward}) the gradient
    of the final scalar output with respect to it.

    Every operation builds a {e new} graph each forward pass ("define by run",
    like PyTorch). Parameters are long-lived leaf nodes reused between passes,
    so their gradients accumulate until you call {!S.zero_grad}. *)

module type S = sig
  type tensor
  (** The backend tensor type that node values and gradients are stored in. *)

  type t
  (** A node in the computational graph. *)

  (** {1 Leaves} *)

  val const : tensor -> t
  (** A leaf that does not require a gradient (inputs, targets, masks). *)

  val param : tensor -> t
  (** A leaf that requires a gradient (weights, biases). *)

  val scalar : float -> t
  (** [const] of a scalar tensor. *)

  (** {1 Inspection} *)

  val value : t -> tensor
  (** The tensor this node computed in the forward pass. For a {!param} this is
      the live parameter storage, which optimizers update in place. *)

  val shape : t -> int array
  (** The shape of {!value}. *)

  val grad : t -> tensor option
  (** The accumulated gradient of the last {!backward} output with respect to
      this node, or [None] if no gradient has reached it. *)

  val grad_exn : t -> tensor
  (** Like {!grad}, but raises [Failure] if no gradient has been computed. *)

  val requires_grad : t -> bool
  (** Whether gradients flow to this node: [true] for params and for any node
      computed from at least one node that requires a gradient. *)

  val id : t -> int
  (** A unique, increasing identifier, used for hashing and graph export. *)

  val zero_grad : t -> unit
  (** Reset the gradient of a node (normally a parameter) to [None]. The
      gradient tensor's storage is kept and reused by the next {!backward},
      which overwrites it in place, so copy a gradient (e.g. with
      {!Tensor_intf.S.copy}) if you need it after calling [zero_grad]. *)

  val detach : t -> t
  (** Same value, cut off from the graph. *)

  (** {1 Differentiable operations}

      Each operation computes its value immediately and records how it was
      produced, so {!backward} can apply the chain rule later. Binary
      elementwise ops broadcast like {!Tensor_intf.S.map2}. *)

  val add : t -> t -> t
  (** Broadcasting elementwise [a + b]. *)

  val sub : t -> t -> t
  (** Broadcasting elementwise [a - b]. *)

  val mul : t -> t -> t
  (** Broadcasting elementwise [a * b]. *)

  val div : t -> t -> t
  (** Broadcasting elementwise [a / b]. *)

  val neg : t -> t
  (** Elementwise negation. *)

  val scale : float -> t -> t
  (** [scale k a] multiplies every element by the constant [k]. *)

  val pow : t -> float -> t
  (** [pow a p] raises every element to the constant power [p]. *)

  val exp : t -> t
  (** Elementwise natural exponential. *)

  val log : t -> t
  (** Elementwise natural logarithm. *)

  val matmul : t -> t -> t
  (** Matrix product with the rules of {!Tensor_intf.S.matmul}: 2-D, batched
      with broadcasting, or with a 1-D operand. *)

  val linear : t -> t -> t -> t
  (** [linear x w b] is [x · w + b] for [x : [| m; k |]], [w : [| k; n |]]
      and a bias [b] that broadcasts to [[| m; n |]], as one node. It gives
      the same result as [add (matmul x w) b] (up to rounding), with one pass
      over the output and no intermediate tensor. Its backward pass writes the
      weight gradient directly into the parameter's gradient buffer. *)

  val transpose : t -> t
  (** 2-D transpose. *)

  val reshape : t -> int array -> t
  (** Same elements, new shape; see {!Tensor_intf.S.reshape}. *)

  val sum : t -> t
  (** Sum of all elements, as a scalar node. *)

  val mean : t -> t
  (** Mean of all elements, as a scalar node. *)

  val sum_axis : ?keepdims:bool -> t -> int -> t
  (** Sum along one axis (negative axes count from the end). [keepdims]
      (default [false]) keeps the reduced axis with size 1. *)

  val relu : t -> t
  (** Elementwise [max x 0]. *)

  val sigmoid : t -> t
  (** Elementwise logistic function [1 / (1 + e^-x)], computed without
      overflow for large [|x|]. *)

  val tanh : t -> t
  (** Elementwise hyperbolic tangent. *)

  val log_softmax : t -> t
  (** Numerically stable log-softmax along the last axis. *)

  val softmax_cross_entropy : t -> int array -> t
  (** [softmax_cross_entropy logits labels] is the mean over the batch of
      [-log_softmax(logits)[label]], for [[| batch; classes |]] logits and
      one label per row, as a single scalar node. Its gradient with respect
      to the logits is [(softmax - one_hot labels) / batch]. *)

  val im2col : t -> kh:int -> kw:int -> stride:int -> padding:int -> t
  (** See {!Tensor_intf.S.im2col}. The only primitive conv2d needs. *)

  val max_pool2d : t -> size:int -> stride:int -> t
  (** NHWC max pooling over [size] x [size] windows moved by [stride]. The
      gradient flows only to the element that won each window. *)

  (** {2 Elementwise math}

      The same functions as in {!Tensor_intf.S}, differentiable. *)

  val sqrt : t -> t
  val abs : t -> t
  (** The gradient at 0 is 0. *)

  val sign : t -> t
  (** Its gradient is 0 everywhere. *)

  val square : t -> t
  val rsqrt : t -> t
  val log1p : t -> t
  val expm1 : t -> t
  val sin : t -> t
  val cos : t -> t
  val erf : t -> t

  val clamp : ?min:float -> ?max:float -> t -> t
  (** The gradient passes where [min <= x <= max] and is 0 elsewhere. *)

  val minimum : t -> t -> t
  (** Broadcasting. The gradient goes to the smaller operand, split in half on
      ties (as in PyTorch). *)

  val maximum : t -> t -> t
  (** Like {!minimum}, for the larger operand. *)

  val where : tensor -> t -> t -> t
  (** [where mask a b]: see {!Tensor_intf.S.where}. The mask is a plain tensor;
      gradients flow to [a] where it is non-zero and to [b] elsewhere. *)

  val masked_fill : t -> tensor -> float -> t
  (** See {!Tensor_intf.S.masked_fill}. No gradient flows to the filled
      positions. *)

  (** {2 Reductions} *)

  val max_axis : ?keepdims:bool -> t -> int -> t
  (** Maximum along an axis. The gradient goes to the first maximal element of
      each row. *)

  val min_axis : ?keepdims:bool -> t -> int -> t
  (** Like {!max_axis}, for the minimum. *)

  val mean_axis : ?keepdims:bool -> t -> int -> t
  val sum_axes : ?keepdims:bool -> t -> int list -> t
  val mean_axes : ?keepdims:bool -> t -> int list -> t
  val var : ?correction:int -> ?keepdims:bool -> t -> int -> t
  val std : ?correction:int -> ?keepdims:bool -> t -> int -> t

  val softmax : t -> t
  (** Softmax along the last axis: [exp (log_softmax x)]. *)

  val logsumexp : ?keepdims:bool -> t -> int -> t
  (** Its gradient is the softmax along that axis. *)

  val cumsum : t -> int -> t

  (** {2 Shapes and indexing}

      All of these copy, like their {!Tensor_intf.S} versions, and route
      gradients back to the positions they read. *)

  val permute : t -> int array -> t
  val swapaxes : t -> int -> int -> t
  val squeeze : ?axis:int -> t -> t
  val unsqueeze : t -> int -> t
  val flatten : ?first:int -> ?last:int -> t -> t

  val broadcast_to : t -> int array -> t
  (** Its gradient sums over the broadcast axes. *)

  val slice : t -> Tensor_intf.range list -> t
  val narrow : t -> axis:int -> start:int -> len:int -> t
  val pad : ?value:float -> t -> (int * int) array -> t
  val flip : t -> int list -> t
  val concat : axis:int -> t list -> t
  val stack : axis:int -> t list -> t
  val split : t -> axis:int -> int list -> t list
  val chunk : t -> axis:int -> int -> t list
  val tile : t -> int array -> t
  val repeat : t -> axis:int -> int -> t

  val index_select : t -> axis:int -> int array -> t
  (** Its gradient accumulates over repeated indices, as an embedding lookup
      needs. *)

  val gather : t -> axis:int -> int array -> t

  (** {2 Linear algebra} *)

  val dot : t -> t -> t
  val outer : t -> t -> t

  val norm : ?p:float -> t -> t
  (** The gradient of the 2-norm at 0 is NaN. *)

  (** {1 Custom operations} *)

  val custom :
    name:string ->
    inputs:t list ->
    value:tensor ->
    backward:(tensor -> tensor option list) ->
    t
  (** A differentiable operation defined by its forward value and its backward
      rule, like PyTorch's [torch.autograd.Function]. [value] is the result,
      computed by the caller from the inputs' values. [backward g], given the
      gradient [g] of the loss with respect to the result, returns one entry
      per input: [Some] gradient (of that input's shape) or [None] for no
      gradient. It is only called during {!backward}, and only if some input
      requires a gradient. [name] is what {!op_name} and graph exports show.

      Gradients [backward] returns are never updated in place, so it may
      return [g] itself or share one tensor between inputs. Raises
      [Invalid_argument] during {!backward} if it returns the wrong number of
      entries or a gradient of the wrong shape.
      {[
        (* GELU, with its exact derivative *)
        let gelu x =
          let open Tensor in
          let v = Var.value x in
          let cdf = scale 0.5 (add_scalar 1. (erf (scale (1. /. Float.sqrt 2.) v))) in
          let pdf =
            scale (1. /. Float.sqrt (2. *. Float.pi)) (exp (scale (-0.5) (square v)))
          in
          Var.custom ~name:"gelu" ~inputs:[ x ] ~value:(mul v cdf)
            ~backward:(fun g -> [ Some (mul g (add cdf (mul v pdf))) ])
      ]} *)

  (** {1 Activations}

      Built from the operations above, or as one {!custom} node with a
      closed-form derivative. *)

  val leaky_relu : ?slope:float -> t -> t
  (** [x] where [x > 0], [slope * x] elsewhere (default [slope = 0.01]). *)

  val elu : ?alpha:float -> t -> t
  (** [x] where [x > 0], [alpha * (e^x - 1)] elsewhere (default
      [alpha = 1]). *)

  val gelu : ?approximate:bool -> t -> t
  (** [x * Φ(x)], with [Φ] the standard normal CDF, computed with {!erf}.
      With [~approximate:true], PyTorch's ["tanh"] approximation
      [0.5 x (1 + tanh (sqrt (2/π) (x + 0.044715 x³)))]. *)

  val silu : t -> t
  (** [x * sigmoid x] (also called swish). *)

  val softplus : ?beta:float -> ?threshold:float -> t -> t
  (** [log (1 + e^(beta x)) / beta], computed stably, and [x] where
      [beta x > threshold] (defaults: [beta = 1], [threshold = 20]), as in
      PyTorch. *)

  (** {1 Gradient utilities} *)

  val set_requires_grad : t -> bool -> unit
  (** Freeze ([false]) or unfreeze ([true]) a leaf, such as a parameter for
      fine-tuning: a frozen parameter gets no gradient, and the optimizers
      skip it. Nodes computed from it before the change are unaffected. Raises
      [Invalid_argument] for a node that isn't a leaf. *)

  val grad_norm : t list -> float
  (** The 2-norm of all the nodes' gradients taken together, ignoring nodes
      without one. *)

  val clip_grad_norm : max_norm:float -> t list -> float
  (** If the combined gradient norm exceeds [max_norm], scale every gradient
      by [max_norm / (norm + 1e-6)] in place, as [torch.nn.utils.clip_grad_norm_]
      does. Returns the norm from before clipping. *)

  val clip_grad_value : float -> t list -> unit
  (** [clip_grad_value c nodes] limits every gradient element to [[-c, c]]. *)

  val detect_anomaly : (unit -> 'a) -> 'a
  (** [detect_anomaly f] runs [f] while checking every value computed by a
      forward operation, and every gradient computed by {!backward}, for NaN
      and infinity. The first one found raises [Failure] naming the operation,
      the node and its shape. It costs a pass over each tensor, so use it for
      debugging only; outside it nothing is checked. *)

  (** {1 Functional API} *)

  val value_and_grad : (t -> t) -> tensor -> float * tensor
  (** [value_and_grad f x] is [f]'s (scalar) value at [x] and its gradient
      with respect to [x], in the style of JAX and [torch.func]. Only nodes
      created during the call count: parameters [f] uses don't receive
      gradients, so it has no side effects on a model. Works inside
      {!no_grad}. Raises [Invalid_argument] if [f] doesn't return a scalar. *)

  val gradient : (t -> t) -> tensor -> tensor
  (** [gradient f x] is the gradient part of {!value_and_grad}: JAX's
      [grad]. *)

  (** {1 Backpropagation} *)

  val no_grad : (unit -> 'a) -> 'a
  (** [no_grad f] runs [f] without recording a graph: every operation inside
      returns a constant, with no gradient and no link to its inputs. Use it
      for evaluation and inference, where it saves building (and keeping
      alive) a graph that will never be differentiated. *)

  val backward : t -> unit
  (** [backward loss] computes d loss / d node for every node in the graph that
      requires a gradient, and {e adds} it to that node's [grad]. [loss] must be
      a scalar (one element). *)

  (** {1 Graph introspection (used by {!Graphviz})} *)

  val parents : t -> t list
  (** The direct inputs of the operation that produced this node. *)

  val op_name : t -> string
  (** A short human-readable name for the operation, e.g. ["matmul"],
      ["scale 0.5"], or ["param"] / ["const"] for leaves. *)

  (** Infix operators for the binary operations, for use with a local open:
      [Var.Infix.(a * b + c)]. *)
  module Infix : sig
    val ( + ) : t -> t -> t
    (** {!add}. *)

    val ( - ) : t -> t -> t
    (** {!sub}. *)

    val ( * ) : t -> t -> t
    (** {!mul}. *)

    val ( / ) : t -> t -> t
    (** {!div}. *)
  end
end

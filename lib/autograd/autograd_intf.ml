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
  (** Reset the gradient of a node (normally a parameter) to [None]. *)

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
  (** 2-D matrix product: [[| m; k |]] x [[| k; n |]] -> [[| m; n |]]. *)

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

  val im2col : t -> kh:int -> kw:int -> stride:int -> padding:int -> t
  (** See {!Tensor_intf.S.im2col}. The only primitive conv2d needs. *)

  val max_pool2d : t -> size:int -> stride:int -> t
  (** NHWC max pooling over [size] x [size] windows moved by [stride]. The
      gradient flows only to the element that won each window. *)

  (** {1 Backpropagation} *)

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

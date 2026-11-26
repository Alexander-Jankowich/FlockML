(** Signature of a reverse-mode automatic differentiation engine.

    A value of type [t] is a node in a computational graph: it holds a tensor
    value, the operation that produced it, and (after {!backward}) the gradient
    of the final scalar output with respect to it.

    Every operation builds a {e new} graph each forward pass ("define by run",
    like PyTorch). Parameters are long-lived leaf nodes reused between passes,
    so their gradients accumulate until you call {!zero_grad}. *)

module type S = sig
  type tensor
  type t

  (** {1 Leaves} *)

  val const : tensor -> t
  (** A leaf that does not require a gradient (inputs, targets, masks). *)

  val param : tensor -> t
  (** A leaf that requires a gradient (weights, biases). *)

  val scalar : float -> t
  (** [const] of a scalar tensor. *)

  (** {1 Inspection} *)

  val value : t -> tensor
  val shape : t -> int array
  val grad : t -> tensor option

  val grad_exn : t -> tensor
  (** Raises [Failure] if no gradient has been computed. *)

  val requires_grad : t -> bool
  val id : t -> int

  val zero_grad : t -> unit
  (** Reset the gradient of a node (normally a parameter) to [None]. *)

  val detach : t -> t
  (** Same value, cut off from the graph. *)

  (** {1 Differentiable operations}

      Binary elementwise ops broadcast like {!Tensor_intf.S.map2}. *)

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val neg : t -> t
  val scale : float -> t -> t
  val pow : t -> float -> t
  val exp : t -> t
  val log : t -> t
  val matmul : t -> t -> t
  val transpose : t -> t
  val reshape : t -> int array -> t
  val sum : t -> t
  val mean : t -> t
  val sum_axis : ?keepdims:bool -> t -> int -> t
  val relu : t -> t
  val sigmoid : t -> t
  val tanh : t -> t

  val log_softmax : t -> t
  (** Numerically stable log-softmax along the last axis. *)

  val im2col : t -> kh:int -> kw:int -> stride:int -> padding:int -> t
  (** See {!Tensor_intf.S.im2col}. The only primitive conv2d needs. *)

  val max_pool2d : t -> size:int -> stride:int -> t

  (** {1 Backpropagation} *)

  val backward : t -> unit
  (** [backward loss] computes d loss / d node for every node in the graph that
      requires a gradient, and {e adds} it to that node's [grad]. [loss] must be
      a scalar (one element). *)

  (** {1 Graph introspection (used by {!Graphviz})} *)

  val parents : t -> t list
  (** The direct inputs of the operation that produced this node. *)

  val op_name : t -> string

  module Infix : sig
    val ( + ) : t -> t -> t
    val ( - ) : t -> t -> t
    val ( * ) : t -> t -> t
    val ( / ) : t -> t -> t
  end
end

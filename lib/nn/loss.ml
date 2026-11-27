(** Loss functions. Each returns a scalar {!Var.t} ready for {!Var.backward}.

    All of them are written in terms of {!Var} operations, so they need no
    backward code of their own: autograd differentiates them for free. *)

module type S = sig
  type var
  (** The autograd node type. *)

  val mse : var -> var -> var
  (** Mean squared error: mean((pred - target)^2). *)

  val cross_entropy : logits:var -> labels:int array -> var
  (** Softmax cross-entropy for a [[| batch; classes |]] matrix of raw scores
      and one integer class label per row, averaged over the batch. *)

  val binary_cross_entropy : ?eps:float -> var -> var -> var
  (** [binary_cross_entropy probs targets] = mean(-(t log p + (1 - t) log (1 -
      p))). [probs] are already in (0, 1) (e.g. after a sigmoid); [eps] guards
      against log 0. *)
end

(** The losses, for one autograd engine. *)
module Make (Var : Autograd_intf.S) : S with type var = Var.t = struct
  type var = Var.t

  (** [mean ((pred - target)²)], built from Var ops so autograd differentiates
      it. *)
  let mse pred target = Var.mean (Var.pow (Var.sub pred target) 2.0)

  (** [-sum (one_hot labels * log_softmax logits) / batch]: the mean over the
      batch of [-log p(correct class)]. With all-zero logits the loss is
      [log num_classes].

      It is a single fused {!Var.softmax_cross_entropy} node rather than a
      composition of [log_softmax], [mul], [sum] and [scale]. That gives the
      same value, but skips building the one-hot matrix and three intermediates,
      and its backward pass is the closed form [(softmax - one_hot) / batch]. *)
  let cross_entropy ~logits ~labels = Var.softmax_cross_entropy logits labels

  (** A direct translation of [-mean (t log p + (1 - t) log (1 - p))] into Var
      ops, with [eps] added inside each log so neither is ever [log 0]. *)
  let binary_cross_entropy ?(eps = 1e-7) probs targets =
    let open Var.Infix in
    let one = Var.scalar 1. and eps = Var.scalar eps in
    (* Adding eps keeps both logs finite when p hits exactly 0 or 1. *)
    let log_p = Var.log (probs + eps)
    and log_1mp = Var.log (one - probs + eps) in
    Var.neg (Var.mean ((targets * log_p) + ((one - targets) * log_1mp)))
end

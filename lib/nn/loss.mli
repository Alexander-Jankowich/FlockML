(** Loss functions. Each returns a scalar {!Var.t} ready for {!Var.backward}.

    All of them are written in terms of {!Var} operations, so they need no
    backward code of their own: autograd differentiates them for free. *)

val mse : Var.t -> Var.t -> Var.t
(** Mean squared error: mean((pred - target)^2). *)

val cross_entropy : logits:Var.t -> labels:int array -> Var.t
(** Softmax cross-entropy for a [[| batch; classes |]] matrix of raw scores and
    one integer class label per row, averaged over the batch. *)

val binary_cross_entropy : ?eps:float -> Var.t -> Var.t -> Var.t
(** [binary_cross_entropy probs targets] = mean(-(t log p + (1 - t) log (1 -
    p))). [probs] are already in (0, 1) (e.g. after a sigmoid); [eps] guards
    against log 0. *)

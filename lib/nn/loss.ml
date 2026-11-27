(** [mean ((pred - target)²)], built from Var ops so autograd differentiates
    it. *)
let mse pred target = Var.mean (Var.pow (Var.sub pred target) 2.0)

(** [-sum (one_hot labels * log_softmax logits) / batch]: the mean over the
    batch of [-log p(correct class)]. With all-zero logits the loss is
    [log num_classes].

    It is a single fused {!Var.softmax_cross_entropy} node rather than a
    composition of [log_softmax], [mul], [sum] and [scale]. That gives the
    same value, but skips building the one-hot matrix and three
    intermediates, and its backward pass is the closed form
    [(softmax - one_hot) / batch]. *)
let cross_entropy ~logits ~labels = Var.softmax_cross_entropy logits labels

(** A direct translation of [-mean (t log p + (1 - t) log (1 - p))] into Var
    ops, with [eps] added inside each log so neither is ever [log 0]. *)
let binary_cross_entropy ?(eps = 1e-7) probs targets =
  let open Var.Infix in
  let one = Var.scalar 1. and eps = Var.scalar eps in
  (* Adding eps keeps both logs finite when p hits exactly 0 or 1. *)
  let log_p = Var.log (probs + eps) and log_1mp = Var.log (one - probs + eps) in
  Var.neg (Var.mean ((targets * log_p) + ((one - targets) * log_1mp)))

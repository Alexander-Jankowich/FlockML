(* TODO [Phase 3, easy]: Var.mean (Var.pow (Var.sub pred target) 2.0).
   (Var.Infix lets you write [Var.Infix.(pred - target)].) *)
let mse pred target = Var.mean (Var.pow (Var.sub pred target) 2.0)

(* TODO [Phase 3, medium]:
   1. lp = Var.log_softmax logits                   [batch; classes]
   2. y  = Var.const (Tensor.one_hot ~num_classes labels)
      (num_classes = second dim of [Var.shape logits])
   3. loss = - sum(y * lp) / batch
   This picks out -log p(correct class) for each row and averages.
   Sanity check: with all-zero logits the loss is log(num_classes). *)
let cross_entropy ~logits ~labels =
  let batch = (Var.shape logits).(0) in
  let lp = Var.log_softmax logits in
  let num_classes = (Var.shape logits).(1) in
  let y = Var.const (Tensor.one_hot ~num_classes labels) in
  Var.scale (-1. /. float_of_int batch) (Var.sum (Var.mul y lp))

(* TODO [Phase 3, easy-medium]: straight translation of the formula with Var
   ops. Clamp p into [eps, 1 - eps] first; one way is to compute it on the
   constant side, e.g. log (p + eps) and log (1 - p + eps). *)
let binary_cross_entropy ?(eps = 1e-7) probs targets =
  let open Var.Infix in
  let one = Var.scalar 1. and eps = Var.scalar eps in
  (* Adding eps keeps both logs finite when p hits exactly 0 or 1. *)
  let log_p = Var.log (probs + eps) and log_1mp = Var.log (one - probs + eps) in
  Var.neg (Var.mean ((targets * log_p) + ((one - targets) * log_1mp)))

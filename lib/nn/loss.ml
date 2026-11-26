(* TODO [Phase 3, easy]: Var.mean (Var.pow (Var.sub pred target) 2.0).
   (Var.Infix lets you write [Var.Infix.(pred - target)].) *)
let mse pred target = Todo.todo "Loss.mse"

(* TODO [Phase 3, medium]:
   1. lp = Var.log_softmax logits                   [batch; classes]
   2. y  = Var.const (Tensor.one_hot ~num_classes labels)
      (num_classes = second dim of [Var.shape logits])
   3. loss = - sum(y * lp) / batch
   This picks out -log p(correct class) for each row and averages.
   Sanity check: with all-zero logits the loss is log(num_classes). *)
let cross_entropy ~logits ~labels = Todo.todo "Loss.cross_entropy"

(* TODO [Phase 3, easy-medium]: straight translation of the formula with Var
   ops. Clamp p into [eps, 1 - eps] first; one way is to compute it on the
   constant side, e.g. log (p + eps) and log (1 - p + eps). *)
let binary_cross_entropy ?(eps = 1e-7) probs targets =
  Todo.todo "Loss.binary_cross_entropy"

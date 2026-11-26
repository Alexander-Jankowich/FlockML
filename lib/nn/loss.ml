(** [mean ((pred - target)²)], built from Var ops so autograd differentiates
    it. *)
let mse pred target = Var.mean (Var.pow (Var.sub pred target) 2.0)

(** [-sum (one_hot labels * log_softmax logits) / batch]. The one-hot mask
    picks out [-log p(correct class)] for each row, and the sum is averaged
    over the batch. With all-zero logits the loss is [log num_classes]. *)
let cross_entropy ~logits ~labels =
  let batch = (Var.shape logits).(0) in
  let lp = Var.log_softmax logits in
  let num_classes = (Var.shape logits).(1) in
  let y = Var.const (Tensor.one_hot ~num_classes labels) in
  Var.scale (-1. /. float_of_int batch) (Var.sum (Var.mul y lp))

(** A direct translation of [-mean (t log p + (1 - t) log (1 - p))] into Var
    ops, with [eps] added inside each log so neither is ever [log 0]. *)
let binary_cross_entropy ?(eps = 1e-7) probs targets =
  let open Var.Infix in
  let one = Var.scalar 1. and eps = Var.scalar eps in
  (* Adding eps keeps both logs finite when p hits exactly 0 or 1. *)
  let log_p = Var.log (probs + eps) and log_1mp = Var.log (one - probs + eps) in
  Var.neg (Var.mean ((targets * log_p) + ((one - targets) * log_1mp)))

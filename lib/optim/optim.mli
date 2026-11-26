(** Gradient-based optimizers.

    An optimizer holds a list of parameters plus per-parameter state (velocity
    for momentum, first/second moments for Adam), and on {!step} updates each
    parameter's tensor {e in place} from its gradient. *)

type t

val sgd : ?momentum:float -> ?weight_decay:float -> lr:float -> Var.t list -> t
(** Stochastic gradient descent, optionally with (PyTorch-style) momentum: v <-
    momentum * v + g; p <- p - lr * v. *)

val adam :
  ?beta1:float ->
  ?beta2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  lr:float ->
  Var.t list ->
  t
(** Adam (Kingma & Ba, 2015). Defaults: beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
*)

val step : t -> unit
(** Apply one update using the gradients currently stored in the params.
    Parameters with no gradient are skipped. *)

val zero_grad : t -> unit
(** Clear every parameter's gradient. Call before each backward pass. *)

val lr : t -> float

val set_lr : t -> float -> unit
(** For learning-rate schedules (e.g. step decay between epochs). *)

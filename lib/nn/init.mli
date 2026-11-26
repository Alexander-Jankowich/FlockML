(** Weight initialisation schemes.

    Good initialisation keeps activations and gradients at a sensible scale as
    they pass through many layers. Bad initialisation (e.g. all zeros, or
    [randn] with std 1 for wide layers) is the most common reason a from-scratch
    network refuses to learn. *)

val xavier_uniform :
  ?rng:Random.State.t -> fan_in:int -> fan_out:int -> int array -> Tensor.t
(** Glorot/Xavier: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)). Best for tanh
    / sigmoid layers. *)

val he_normal : ?rng:Random.State.t -> fan_in:int -> int array -> Tensor.t
(** He/Kaiming: N(0, 2 / fan_in). Best for ReLU layers. *)

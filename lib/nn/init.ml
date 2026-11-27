(** Weight initialisation schemes.

    Good initialisation keeps activations and gradients at a sensible scale as
    they pass through many layers. Bad initialisation (e.g. all zeros, or
    [randn] with std 1 for wide layers) is the most common reason a from-scratch
    network refuses to learn. *)

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  val xavier_uniform :
    ?rng:Random.State.t -> fan_in:int -> fan_out:int -> int array -> tensor
  (** Glorot/Xavier: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)). Best for
      tanh / sigmoid layers. *)

  val he_normal : ?rng:Random.State.t -> fan_in:int -> int array -> tensor
  (** He/Kaiming: N(0, 2 / fan_in). Best for ReLU layers. *)
end

(** The initialisers for backend [T]. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  (** The bound [a] makes the variance [a²/3 = 2 / (fan_in + fan_out)], which
      balances the forward and backward signal scales. *)
  let xavier_uniform ?rng ~fan_in ~fan_out shape =
    let a = sqrt (6. /. (float_of_int fan_in +. float_of_int fan_out)) in
    T.rand_uniform ?rng ~lo:(-.a) ~hi:a shape

  (** Standard deviation [sqrt (2 / fan_in)]. The 2 is there because ReLU zeroes
      half its inputs, halving the variance; doubling it restores the scale. *)
  let he_normal ?rng ~fan_in shape =
    let sd = sqrt (2. /. float_of_int fan_in) in
    T.randn ?rng ~std:sd shape
end

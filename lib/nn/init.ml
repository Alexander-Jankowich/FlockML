(** The bound [a] makes the variance [a²/3 = 2 / (fan_in + fan_out)], which
    balances the forward and backward signal scales. *)
let xavier_uniform ?rng ~fan_in ~fan_out shape =
  let a = sqrt (6. /. (float_of_int fan_in +. float_of_int fan_out)) in
  Tensor.rand_uniform ?rng ~lo:(-.a) ~hi:a shape

(** Standard deviation [sqrt (2 / fan_in)]. The 2 is there because ReLU zeroes
    half its inputs, halving the variance; doubling it restores the scale. *)
let he_normal ?rng ~fan_in shape =
  let sd = sqrt (2. /. float_of_int fan_in) in
  Tensor.randn ?rng ~std:sd shape

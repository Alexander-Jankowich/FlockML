(* TODO [Phase 3, easy]: one line on top of Tensor.rand_uniform. *)
let xavier_uniform ?rng ~fan_in ~fan_out shape =
  let a = sqrt (6. /. (float_of_int fan_in +. float_of_int fan_out)) in
  Tensor.rand_uniform ?rng ~lo:(-.a) ~hi:a shape

(* TODO [Phase 3, easy]: one line on top of Tensor.randn (std = sqrt(2/fan_in)).
   Why 2? ReLU zeroes half its inputs, halving the variance; the 2 restores it. *)
let he_normal ?rng ~fan_in shape =
  let sd = sqrt (2. /. float_of_int fan_in) in
  Tensor.randn ?rng ~std:sd shape

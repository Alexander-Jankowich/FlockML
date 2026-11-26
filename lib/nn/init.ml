(* TODO [Phase 3, easy]: one line on top of Tensor.rand_uniform. *)
let xavier_uniform ?rng ~fan_in ~fan_out shape = Todo.todo "Init.xavier_uniform"

(* TODO [Phase 3, easy]: one line on top of Tensor.randn (std = sqrt(2/fan_in)).
   Why 2? ReLU zeroes half its inputs, halving the variance; the 2 restores it. *)
let he_normal ?rng ~fan_in shape = Todo.todo "Init.he_normal"

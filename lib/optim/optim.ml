(* Optimizer state as an ADT: one constructor per algorithm, each carrying
   one state tensor per parameter (same index as [params]). *)

type sgd = {
  mutable lr : float;
  momentum : float;
  weight_decay : float;
  velocity : Tensor.t array;
}

type adam = {
  mutable lr : float;
  beta1 : float;
  beta2 : float;
  eps : float;
  weight_decay : float;
  m : Tensor.t array;  (** running mean of gradients *)
  v : Tensor.t array;  (** running mean of squared gradients *)
  mutable t : int;  (** number of steps taken, for bias correction *)
}

type kind = Sgd of sgd | Adam of adam
type t = { params : Var.t array; kind : kind }

(* ------------------------------------------------------------------------ *)
(* Construction and bookkeeping (provided)                                   *)
(* ------------------------------------------------------------------------ *)

let zeros_like_params params =
  Array.map (fun p -> Tensor.zeros (Var.shape p)) params

let sgd ?(momentum = 0.0) ?(weight_decay = 0.0) ~lr params =
  let params = Array.of_list params in
  {
    params;
    kind =
      Sgd { lr; momentum; weight_decay; velocity = zeros_like_params params };
  }

let adam ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.0) ~lr
    params =
  let params = Array.of_list params in
  {
    params;
    kind =
      Adam
        {
          lr;
          beta1;
          beta2;
          eps;
          weight_decay;
          m = zeros_like_params params;
          v = zeros_like_params params;
          t = 0;
        };
  }

let zero_grad opt = Array.iter Var.zero_grad opt.params

let lr opt =
  match opt.kind with Sgd (s : sgd) -> s.lr | Adam (a : adam) -> a.lr

let set_lr opt lr =
  match opt.kind with
  | Sgd (s : sgd) -> s.lr <- lr
  | Adam (a : adam) -> a.lr <- lr

(* ------------------------------------------------------------------------ *)
(* Update rules                                                              *)
(* ------------------------------------------------------------------------ *)

(* TODO [Phase 4, easy-medium]: for each index i with Var.grad params.(i) =
   Some g (skip None):
     p = Var.value params.(i)        (mutate this tensor in place!)
     if weight_decay > 0:  g <- g + weight_decay * p   (don't mutate the
                                                        stored grad; copy)
     if momentum = 0:      p <- p - lr * g             (Tensor.axpy_)
     else:                 v <- momentum * v + g       (Tensor.scale_, add_)
                           p <- p - lr * v
   Check with the hand-computed numbers in test/test_optim.ml. *)
let sgd_step params (s : sgd) = Todo.todo "Optim.sgd_step"

(* TODO [Phase 4, medium]: Adam.
     t <- t + 1                                  (once per step, not per param)
     for each param with a gradient g:
       (weight decay as in SGD)
       m <- beta1 * m + (1 - beta1) * g
       v <- beta2 * v + (1 - beta2) * g^2
       m_hat = m / (1 - beta1^t)                 (bias correction: m and v
       v_hat = v / (1 - beta2^t)                  start at 0, so early
       p <- p - lr * m_hat / (sqrt v_hat + eps)   estimates are too small)
   Store m, v back with Tensor.blit_ (or update them in place directly).
   Nice property for testing: on step 1, m_hat / sqrt v_hat = sign(g), so
   every parameter moves by almost exactly lr. *)
let adam_step params (a : adam) = Todo.todo "Optim.adam_step"

let step opt =
  match opt.kind with
  | Sgd s -> sgd_step opt.params s
  | Adam a -> adam_step opt.params a

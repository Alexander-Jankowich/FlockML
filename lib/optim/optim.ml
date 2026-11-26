(** Optimizer state as an ADT: one constructor per algorithm, each carrying
    one state tensor per parameter (same index as [params]). *)

(** State for SGD with optional momentum and weight decay. *)
type sgd = {
  mutable lr : float;
  momentum : float;
  weight_decay : float;
  velocity : Tensor.t array;
}

(** State for Adam. *)
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

(** Which algorithm, with its state. *)
type kind = Sgd of sgd | Adam of adam

(** An optimizer: the parameters it updates and its algorithm. *)
type t = { params : Var.t array; kind : kind }

(* ------------------------------------------------------------------------ *)
(* Construction and bookkeeping                                              *)
(* ------------------------------------------------------------------------ *)

(** One zero tensor per parameter, matching its shape: the initial
    per-parameter state (velocity, moments). *)
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

(** One SGD update of every parameter that has a gradient, in place:
    - weight decay: [g <- g + weight_decay * p] (on a copy, never the stored
      gradient);
    - plain: [p <- p - lr * g];
    - with momentum: [v <- momentum * v + g], then [p <- p - lr * v]. *)
let sgd_step params (s : sgd) =
  Array.iteri
    (fun i param ->
      match Var.grad param with
      | None -> () (* not used in this loss: leave it alone *)
      | Some g ->
          (* [p] is the parameter's actual storage, so mutating it updates
             the model; no new Var is created. *)
          let p = Var.value param in
          (* Weight decay = gradient of (wd/2)·||p||². Build a NEW tensor
             rather than mutating [g], which is still the param's grad. *)
          let g =
            if s.weight_decay > 0. then
              Tensor.add g (Tensor.scale s.weight_decay p)
            else g
          in
          if s.momentum = 0. then
            (* p <- p - lr * g *)
            Tensor.axpy_ ~alpha:(-.s.lr) ~x:g ~y:p
          else begin
            (* v <- momentum * v + g;  p <- p - lr * v *)
            let v = s.velocity.(i) in
            Tensor.scale_ s.momentum v;
            Tensor.add_ v g;
            Tensor.axpy_ ~alpha:(-.s.lr) ~x:v ~y:p
          end)
    params

(** One Adam update of every parameter that has a gradient, in place:
    {v
      t <- t + 1                       (once per step, shared by all params)
      m <- beta1 * m + (1 - beta1) * g
      v <- beta2 * v + (1 - beta2) * g^2
      m_hat = m / (1 - beta1^t)        (bias correction: m and v start at 0)
      v_hat = v / (1 - beta2^t)
      p <- p - lr * m_hat / (sqrt v_hat + eps)
    v}
    Weight decay is applied to [g] first, as in SGD. On the first step
    [m_hat / sqrt v_hat = sign g], so every parameter moves by almost exactly
    [lr]. *)
let adam_step params (a : adam) =
  (* One tick per optimizer step, shared by every parameter. *)
  a.t <- a.t + 1;
  (* Bias corrections. m and v start at 0, so early on they underestimate
     the true averages; dividing by (1 - beta^t) undoes that. Both are the
     same for every parameter, so compute them once. *)
  let t = float_of_int a.t in
  let bc1 = 1. -. Float.pow a.beta1 t and bc2 = 1. -. Float.pow a.beta2 t in
  Array.iteri
    (fun i param ->
      match Var.grad param with
      | None -> ()
      | Some g ->
          let p = Var.value param in
          (* Weight decay = gradient of (wd/2)·||p||². Build a NEW tensor
             rather than mutating [g], which is still the param's grad. *)
          let g =
            if a.weight_decay > 0. then
              Tensor.add g (Tensor.scale a.weight_decay p)
            else g
          in
          let m = a.m.(i) and v = a.v.(i) in
          (* m <- beta1 * m + (1 - beta1) * g      (running mean of g) *)
          Tensor.scale_ a.beta1 m;
          Tensor.axpy_ ~alpha:(1. -. a.beta1) ~x:g ~y:m;
          (* v <- beta2 * v + (1 - beta2) * g^2    (running mean of g^2) *)
          Tensor.scale_ a.beta2 v;
          Tensor.axpy_ ~alpha:(1. -. a.beta2) ~x:(Tensor.mul g g) ~y:v;
          (* p <- p - lr * m_hat / (sqrt v_hat + eps), elementwise *)
          let update =
            Tensor.map2
              (fun mi vi -> mi /. bc1 /. (Float.sqrt (vi /. bc2) +. a.eps))
              m v
          in
          Tensor.axpy_ ~alpha:(-.a.lr) ~x:update ~y:p)
    params

let step opt =
  match opt.kind with
  | Sgd s -> sgd_step opt.params s
  | Adam a -> adam_step opt.params a

(** Gradient-based optimizers.

    An optimizer holds a list of parameters plus per-parameter state (velocity
    for momentum, first/second moments for Adam), and on {!step} updates each
    parameter's tensor {e in place} from its gradient. *)

module type S = sig
  type var
  (** The autograd node type of the parameters. *)

  type t
  (** An optimizer bound to a fixed list of parameters, with its per-parameter
      state. *)

  val sgd : ?momentum:float -> ?weight_decay:float -> lr:float -> var list -> t
  (** Stochastic gradient descent, optionally with (PyTorch-style) momentum: v
      <- momentum * v + g; p <- p - lr * v. *)

  val adam :
    ?beta1:float ->
    ?beta2:float ->
    ?eps:float ->
    ?weight_decay:float ->
    lr:float ->
    var list ->
    t
  (** Adam (Kingma & Ba, 2015). Defaults: beta1 = 0.9, beta2 = 0.999, eps =
      1e-8. *)

  val step : t -> unit
  (** Apply one update using the gradients currently stored in the params.
      Parameters with no gradient are skipped. *)

  val zero_grad : t -> unit
  (** Clear every parameter's gradient. Call before each backward pass. *)

  val lr : t -> float
  (** The current learning rate. *)

  val set_lr : t -> float -> unit
  (** For learning-rate schedules (e.g. step decay between epochs). *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Var : Autograd_intf.S with type tensor = T.t) :
  S with type var = Var.t = struct
  type var = Var.t

  (** Optimizer state as an ADT: one constructor per algorithm, each carrying
      one state tensor per parameter (same index as [params]). *)

  type sgd = {
    mutable lr : float;
    momentum : float;
    weight_decay : float;
    velocity : T.t array;
  }
  (** State for SGD with optional momentum and weight decay. *)

  type adam = {
    mutable lr : float;
    beta1 : float;
    beta2 : float;
    eps : float;
    weight_decay : float;
    m : T.t array;  (** running mean of gradients *)
    v : T.t array;  (** running mean of squared gradients *)
    mutable t : int;  (** number of steps taken, for bias correction *)
  }
  (** State for Adam. *)

  (** Which algorithm, with its state. *)
  type kind = Sgd of sgd | Adam of adam

  type t = { params : Var.t array; kind : kind }
  (** An optimizer: the parameters it updates and its algorithm. *)

  (* ------------------------------------------------------------------------ *)
  (* Construction and bookkeeping                                              *)
  (* ------------------------------------------------------------------------ *)

  (** One zero tensor per parameter, matching its shape: the initial
      per-parameter state (velocity, moments). *)
  let zeros_like_params params =
    Array.map (fun p -> T.zeros (Var.shape p)) params

  let sgd ?(momentum = 0.0) ?(weight_decay = 0.0) ~lr params =
    let params = Array.of_list params in
    {
      params;
      kind =
        Sgd { lr; momentum; weight_decay; velocity = zeros_like_params params };
    }

  let adam ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.0)
      ~lr params =
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
      - weight decay: [g' = g + weight_decay * p] (the stored gradient itself is
        never changed);
      - plain: [p <- p - lr * g'];
      - with momentum: [v <- momentum * v + g'], then [p <- p - lr * v].

      Each parameter is updated by one fused pass
      ({!Tensor_intf.S.sgd_update_}), which reads every tensor once and
      allocates nothing. *)
  let sgd_step params (s : sgd) =
    Array.iteri
      (fun i param ->
        match Var.grad param with
        | None -> () (* not used in this loss: leave it alone *)
        | Some grad ->
            (* [Var.value param] is the parameter's actual storage, so
               mutating it updates the model; no new Var is created. *)
            T.sgd_update_ ~lr:s.lr ~momentum:s.momentum
              ~weight_decay:s.weight_decay ~param:(Var.value param) ~grad
              ~velocity:s.velocity.(i))
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
      [lr].

      Each parameter is updated by one fused pass
      ({!Tensor_intf.S.adam_update_}). It does the same arithmetic as the
      step-by-step formulas above, with no temporary tensors (equal up to the
      last bit of rounding). *)
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
        | Some grad ->
            T.adam_update_ ~lr:a.lr ~beta1:a.beta1 ~beta2:a.beta2 ~eps:a.eps
              ~weight_decay:a.weight_decay ~bc1 ~bc2 ~param:(Var.value param)
              ~grad ~m:a.m.(i) ~v:a.v.(i))
      params

  let step opt =
    match opt.kind with
    | Sgd s -> sgd_step opt.params s
    | Adam a -> adam_step opt.params a
end

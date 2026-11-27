(** Gradient-based optimizers.

    An optimizer holds a list of parameters plus per-parameter state (velocity
    for momentum, first/second moments for Adam...), and on {!S.step} updates
    each parameter's tensor {e in place} from its gradient, with one fused
    kernel per parameter. The update rules are PyTorch's, so the same
    hyperparameters give the same trajectory.

    Parameters can be split into groups with their own learning rate and weight
    decay, for example to exempt biases and normalisation weights from decay. *)

module type S = sig
  type var
  (** The autograd node type of the parameters. *)

  type tensor
  (** The backend's tensor type, for {!state_dict}. *)

  type t
  (** An optimizer bound to a fixed list of parameters, with its per-parameter
      state. *)

  type group
  (** Parameters that share a learning rate and weight decay. *)

  val group : ?lr:float -> ?weight_decay:float -> var list -> group
  (** Parameters that use [lr] and [weight_decay] instead of the optimizer's own
      (each defaults to the optimizer's). *)

  (** {1 Algorithms}

      Each takes the parameters of the default group (the optimizer's [lr] and
      [weight_decay]), and [groups] for any others. A parameter may appear only
      once: raises [Invalid_argument] otherwise. *)

  val sgd :
    ?momentum:float ->
    ?nesterov:bool ->
    ?weight_decay:float ->
    ?groups:group list ->
    lr:float ->
    var list ->
    t
  (** Stochastic gradient descent, optionally with (PyTorch-style) momentum:
      [v <- momentum * v + g; p <- p - lr * v], or with [nesterov]
      [p <- p - lr * (g + momentum * v)]. Weight decay adds [weight_decay * p]
      to [g]. *)

  val adam :
    ?beta1:float ->
    ?beta2:float ->
    ?eps:float ->
    ?weight_decay:float ->
    ?groups:group list ->
    lr:float ->
    var list ->
    t
  (** Adam (Kingma & Ba, 2015). Defaults: beta1 = 0.9, beta2 = 0.999, eps =
      1e-8. Weight decay is L2 regularisation added to the gradient, as in
      PyTorch's [Adam]; see {!adamw} for the decoupled form. *)

  val adamw :
    ?beta1:float ->
    ?beta2:float ->
    ?eps:float ->
    ?weight_decay:float ->
    ?groups:group list ->
    lr:float ->
    var list ->
    t
  (** AdamW (Loshchilov & Hutter, 2019): Adam with the weight decay applied
      directly to the parameters, [p <- p (1 - lr weight_decay)], rather than
      through the gradient. [weight_decay] defaults to 0.01, as in PyTorch. *)

  val rmsprop :
    ?alpha:float ->
    ?eps:float ->
    ?weight_decay:float ->
    ?momentum:float ->
    ?centered:bool ->
    ?groups:group list ->
    lr:float ->
    var list ->
    t
  (** RMSprop, as PyTorch's: [alpha] = 0.99, [eps] = 1e-8, no momentum, not
      centered by default. *)

  val adagrad :
    ?lr_decay:float ->
    ?eps:float ->
    ?weight_decay:float ->
    ?initial_accumulator_value:float ->
    ?groups:group list ->
    lr:float ->
    var list ->
    t
  (** Adagrad, as PyTorch's: the step size at step [k] is
      [lr / (1 + (k - 1) lr_decay)]; [eps] = 1e-10. *)

  (** {1 Using an optimizer} *)

  val step : t -> unit
  (** Apply one update using the gradients currently stored in the params.
      Parameters with no gradient (unused, or frozen) are skipped, and their
      step count doesn't advance. *)

  val zero_grad : t -> unit
  (** Clear every parameter's gradient. Call before each backward pass.
      Gradients otherwise add up over several backward passes, which is how to
      accumulate gradients over several small batches. *)

  val params : t -> var list
  (** Every parameter, default group first. *)

  val lr : t -> float
  (** The learning rate of the default group. *)

  val set_lr : t -> float -> unit
  (** Set the default group's learning rate, and scale every other group's by
      the same factor, so their ratios stay as they were built. A schedule is
      applied as [set_lr opt (schedule step)]. *)

  val group_lrs : t -> float list
  (** The learning rate of each group, the default group first. *)

  (** {1 Checkpoints} *)

  val state_dict : t -> (string * tensor) list
  (** The optimizer's state: the default learning rate (["lr"]), and for
      parameter [i] (in {!params} order) its step count (["<i>.step"]) and
      buffers, with PyTorch's names (["<i>.exp_avg"],
      ["<i>.momentum_buffer"]...). These are the live tensors. *)

  val load_state_dict : t -> (string * tensor) list -> unit
  (** Restore a {!state_dict}, so training resumes exactly where it stopped. The
      optimizer must have the same algorithm and parameter shapes. Raises
      [Failure] naming the first entry that is missing or has the wrong shape.
  *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Var : Autograd_intf.S with type tensor = T.t) :
  S with type var = Var.t and type tensor = T.t = struct
  type var = Var.t
  type tensor = T.t

  type group = {
    g_params : Var.t list;
    g_lr : float option;
    g_wd : float option;
  }

  let group ?lr ?weight_decay params =
    { g_params = params; g_lr = lr; g_wd = weight_decay }

  (** The algorithm and its hyperparameters. *)
  type algo =
    | Sgd of { momentum : float; nesterov : bool }
    | Adam of { beta1 : float; beta2 : float; eps : float; decoupled : bool }
    | Rmsprop of {
        alpha : float;
        eps : float;
        momentum : float;
        centered : bool;
      }
    | Adagrad of { lr_decay : float; eps : float }

  type hyper = { mutable lr : float; ratio : float; weight_decay : float }
  (** A group's settings: [ratio] is its learning rate over the default group's,
      as built, which {!set_lr} keeps. *)

  type slot = {
    param : Var.t;
    hyper : hyper;
    mutable steps : int;
    bufs : (string * T.t) list;
        (** the per-parameter state, by PyTorch's names *)
  }

  type t = {
    algo : algo;
    default : hyper;
    groups : hyper list;
    slots : slot array;
  }

  (* ------------------------------------------------------------------------ *)
  (* Construction and bookkeeping                                              *)
  (* ------------------------------------------------------------------------ *)

  (** The state buffers each algorithm keeps for one parameter. *)
  let buffers algo ~init p =
    let z () = T.zeros (Var.shape p) in
    match algo with
    | Sgd { momentum; _ } ->
        if momentum > 0. then [ ("momentum_buffer", z ()) ] else []
    | Adam _ -> [ ("exp_avg", z ()); ("exp_avg_sq", z ()) ]
    | Rmsprop { momentum; centered; _ } ->
        ("square_avg", z ())
        :: (if centered then [ ("grad_avg", z ()) ] else [])
        @ if momentum > 0. then [ ("momentum_buffer", z ()) ] else []
    | Adagrad _ -> [ ("sum", T.full (Var.shape p) init) ]

  let make ?(init = 0.) algo ~lr ~weight_decay ~groups params =
    let default = { lr; ratio = 1.; weight_decay } in
    let hypers =
      List.map
        (fun g ->
          let glr = Option.value g.g_lr ~default:lr in
          ( {
              lr = glr;
              ratio = (if lr = 0. then 1. else glr /. lr);
              weight_decay = Option.value g.g_wd ~default:weight_decay;
            },
            g.g_params ))
        groups
    in
    let all = (default, params) :: hypers in
    let seen = Hashtbl.create 16 in
    let slots =
      List.concat_map
        (fun (hyper, ps) ->
          List.map
            (fun p ->
              if Hashtbl.mem seen (Var.id p) then
                invalid_arg "Optim: a parameter appears more than once";
              Hashtbl.add seen (Var.id p) ();
              { param = p; hyper; steps = 0; bufs = buffers algo ~init p })
            ps)
        all
    in
    { algo; default; groups = List.map fst hypers; slots = Array.of_list slots }

  let sgd ?(momentum = 0.0) ?(nesterov = false) ?(weight_decay = 0.0)
      ?(groups = []) ~lr params =
    if nesterov && momentum <= 0. then
      invalid_arg "Optim.sgd: Nesterov momentum needs momentum > 0";
    make (Sgd { momentum; nesterov }) ~lr ~weight_decay ~groups params

  let adam ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.0)
      ?(groups = []) ~lr params =
    make
      (Adam { beta1; beta2; eps; decoupled = false })
      ~lr ~weight_decay ~groups params

  let adamw ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.01)
      ?(groups = []) ~lr params =
    make
      (Adam { beta1; beta2; eps; decoupled = true })
      ~lr ~weight_decay ~groups params

  let rmsprop ?(alpha = 0.99) ?(eps = 1e-8) ?(weight_decay = 0.0)
      ?(momentum = 0.0) ?(centered = false) ?(groups = []) ~lr params =
    make
      (Rmsprop { alpha; eps; momentum; centered })
      ~lr ~weight_decay ~groups params

  let adagrad ?(lr_decay = 0.0) ?(eps = 1e-10) ?(weight_decay = 0.0)
      ?(initial_accumulator_value = 0.0) ?(groups = []) ~lr params =
    make ~init:initial_accumulator_value
      (Adagrad { lr_decay; eps })
      ~lr ~weight_decay ~groups params

  let zero_grad opt = Array.iter (fun s -> Var.zero_grad s.param) opt.slots
  let params opt = Array.to_list (Array.map (fun s -> s.param) opt.slots)
  let lr opt = opt.default.lr

  let set_lr opt lr =
    opt.default.lr <- lr;
    List.iter (fun h -> h.lr <- lr *. h.ratio) opt.groups

  let group_lrs opt = opt.default.lr :: List.map (fun h -> h.lr) opt.groups

  (* ------------------------------------------------------------------------ *)
  (* Update rules                                                              *)
  (* ------------------------------------------------------------------------ *)

  let buf s name = List.assoc name s.bufs

  (** A buffer that the kernel only reads in some configurations: the real one
      if there is one, otherwise any tensor of the right shape. *)
  let buf_or s name =
    match List.assoc_opt name s.bufs with
    | Some b -> b
    | None -> buf s (fst (List.hd s.bufs))

  (** One update of one parameter, by one fused kernel that reads every tensor
      once and allocates nothing. [Var.value param] is the parameter's actual
      storage, so changing it updates the model.

      Adam's bias corrections [1 - beta^t] use the parameter's own step count:
      [m] and [v] start at 0, so early on they underestimate the true averages,
      and dividing by the corrections undoes that. On the first step
      [m_hat / sqrt v_hat = sign g], so every parameter moves by almost exactly
      [lr]. *)
  let update algo s grad =
    s.steps <- s.steps + 1;
    let lr = s.hyper.lr and weight_decay = s.hyper.weight_decay in
    let param = Var.value s.param in
    match algo with
    | Sgd { momentum; nesterov } ->
        let velocity =
          if momentum > 0. then buf s "momentum_buffer" else param
        in
        if nesterov then
          T.sgd_nesterov_update_ ~lr ~momentum ~weight_decay ~param ~grad
            ~velocity
        else T.sgd_update_ ~lr ~momentum ~weight_decay ~param ~grad ~velocity
    | Adam { beta1; beta2; eps; decoupled } ->
        let t = float_of_int s.steps in
        let bc1 = 1. -. Float.pow beta1 t and bc2 = 1. -. Float.pow beta2 t in
        let m = buf s "exp_avg" and v = buf s "exp_avg_sq" in
        if decoupled then
          T.adamw_update_ ~lr ~beta1 ~beta2 ~eps ~weight_decay ~bc1 ~bc2 ~param
            ~grad ~m ~v
        else
          T.adam_update_ ~lr ~beta1 ~beta2 ~eps ~weight_decay ~bc1 ~bc2 ~param
            ~grad ~m ~v
    | Rmsprop { alpha; eps; momentum; centered } ->
        T.rmsprop_update_ ~lr ~alpha ~eps ~weight_decay ~momentum ~centered
          ~param ~grad ~square_avg:(buf s "square_avg")
          ~grad_avg:(buf_or s "grad_avg")
          ~momentum_buf:(buf_or s "momentum_buffer")
    | Adagrad { lr_decay; eps } ->
        let clr = lr /. (1. +. (float_of_int (s.steps - 1) *. lr_decay)) in
        T.adagrad_update_ ~lr:clr ~eps ~weight_decay ~param ~grad
          ~sum:(buf s "sum")

  let step opt =
    Array.iter
      (fun s ->
        match Var.grad s.param with
        | None -> () (* not used in this loss, or frozen: leave it alone *)
        | Some grad -> update opt.algo s grad)
      opt.slots

  (* ------------------------------------------------------------------------ *)
  (* Checkpoints                                                               *)
  (* ------------------------------------------------------------------------ *)

  let state_dict opt =
    ("lr", T.scalar opt.default.lr)
    :: List.concat
         (List.mapi
            (fun i s ->
              let k name = Printf.sprintf "%d.%s" i name in
              (k "step", T.scalar (float_of_int s.steps))
              :: List.map (fun (n, b) -> (k n, b)) s.bufs)
            (Array.to_list opt.slots))

  let load_state_dict opt entries =
    let find name expected =
      match List.assoc_opt name entries with
      | None ->
          failwith (Printf.sprintf "Optim.load_state_dict: no entry %s" name)
      | Some t when T.shape t <> expected ->
          failwith
            (Printf.sprintf
               "Optim.load_state_dict: %s has shape %s, expected %s" name
               (Layout.shape_to_string (T.shape t))
               (Layout.shape_to_string expected))
      | Some t -> t
    in
    (* Check everything before changing anything. *)
    let lr = T.item (find "lr" [||]) in
    let loads =
      List.concat
        (List.mapi
           (fun i s ->
             let k name = Printf.sprintf "%d.%s" i name in
             let steps = int_of_float (T.item (find (k "step") [||])) in
             (fun () -> s.steps <- steps)
             :: List.map
                  (fun (n, b) ->
                    let src = find (k n) (T.shape b) in
                    fun () -> T.blit_ ~src ~dst:b)
                  s.bufs)
           (Array.to_list opt.slots))
    in
    List.iter (fun f -> f ()) loads;
    set_lr opt lr
end

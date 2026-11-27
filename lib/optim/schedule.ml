(** Learning-rate schedules.

    A schedule is a pure function from the step number (counting from 0) to a
    learning rate, applied with [Optim.set_lr opt (schedule step)] before each
    step, or each epoch. Being plain functions, schedules compose: {!warmup}
    takes the schedule to run after the warmup.

    Each one gives the same learning rates as the PyTorch scheduler of the same
    name, called once per step. {!Plateau} is the one schedule with state, since
    it reacts to a measured metric. *)

type t = int -> float
(** The learning rate at each step. *)

let constant lr : t = fun _ -> lr

(** [lr * gamma^(step / step_size)]: PyTorch's [StepLR]. *)
let step_decay ?(gamma = 0.1) ~lr ~step_size () : t =
  if step_size <= 0 then
    invalid_arg "Schedule.step_decay: step_size must be positive";
  fun step -> lr *. Float.pow gamma (float_of_int (step / step_size))

(** [lr * gamma^k], where [k] is how many [milestones] are [<= step]: PyTorch's
    [MultiStepLR]. *)
let multi_step ?(gamma = 0.1) ~lr milestones : t =
 fun step ->
  let k = List.length (List.filter (fun m -> m <= step) milestones) in
  lr *. Float.pow gamma (float_of_int k)

(** [lr * gamma^step]: PyTorch's [ExponentialLR]. *)
let exponential ~lr ~gamma : t =
 fun step -> lr *. Float.pow gamma (float_of_int step)

(** Half a cosine from [lr] down to [min_lr] over [total] steps, then [min_lr]:
    PyTorch's [CosineAnnealingLR] with [T_max = total], for the first [total]
    steps. *)
let cosine ?(min_lr = 0.) ~lr ~total () : t =
  if total <= 0 then invalid_arg "Schedule.cosine: total must be positive";
  fun step ->
    let t = float_of_int (min step total) /. float_of_int total in
    min_lr +. ((lr -. min_lr) *. (1. +. Float.cos (Float.pi *. t)) /. 2.)

(** From [start_factor * lr] to [end_factor * lr] in a straight line over
    [total] steps, then [end_factor * lr]: PyTorch's [LinearLR]. *)
let linear ?(start_factor = 1. /. 3.) ?(end_factor = 1.) ~lr ~total () : t =
 fun step ->
  let t =
    if total <= 0 then 1.
    else float_of_int (min step total) /. float_of_int total
  in
  lr *. (start_factor +. ((end_factor -. start_factor) *. t))

(** [warmup ~steps after] rises linearly to [after 0] over the first [steps]
    steps ([after 0 * (step + 1) / steps], so step 0 is not 0), then follows
    [after] from its own step 0. *)
let warmup ~steps (after : t) : t =
 fun step ->
  if step < steps then after 0 *. float_of_int (step + 1) /. float_of_int steps
  else after (step - steps)

(** PyTorch's [OneCycleLR] with its defaults (cosine annealing, two phases):
    from [max_lr / div_factor] up to [max_lr] over the first [pct_start] of
    [total] steps, then down to [max_lr / div_factor / final_div_factor] at step
    [total - 1]. *)
let one_cycle ?(pct_start = 0.3) ?(div_factor = 25.) ?(final_div_factor = 1e4)
    ~max_lr ~total () : t =
  if total <= 1 then invalid_arg "Schedule.one_cycle: total must be at least 2";
  let initial = max_lr /. div_factor in
  let final = initial /. final_div_factor in
  let up_end = (pct_start *. float_of_int total) -. 1. in
  let down_end = float_of_int total -. 1. in
  let anneal a b pct =
    b +. ((a -. b) /. 2. *. (Float.cos (Float.pi *. pct) +. 1.))
  in
  fun step ->
    let s = float_of_int (min step (total - 1)) in
    if s <= up_end then anneal initial max_lr (s /. up_end)
    else anneal max_lr final ((s -. up_end) /. (down_end -. up_end))

(** PyTorch's [ReduceLROnPlateau]: multiply the learning rate by [factor] once
    the metric has not improved for more than [patience] calls. *)
module Plateau = struct
  type t = {
    mutable lr : float;
    factor : float;
    patience : int;
    threshold : float;
    min_lr : float;
    cooldown : int;
    mode : [ `Min | `Max ];
    mutable best : float;
    mutable bad : int;
    mutable cooling : int;
  }

  (** [threshold] is relative, as PyTorch's default [threshold_mode="rel"]: in
      [`Min] mode (the default) a value improves on the best if it is below
      [best * (1 - threshold)]. *)
  let create ?(factor = 0.1) ?(patience = 10) ?(threshold = 1e-4) ?(min_lr = 0.)
      ?(cooldown = 0) ?(mode = `Min) lr =
    if not (factor < 1.) then invalid_arg "Schedule.Plateau: factor must be < 1";
    {
      lr;
      factor;
      patience;
      threshold;
      min_lr;
      cooldown;
      mode;
      best =
        (match mode with `Min -> Float.infinity | `Max -> Float.neg_infinity);
      bad = 0;
      cooling = 0;
    }

  let lr p = p.lr

  (** Record one measurement (say, the validation loss after an epoch) and
      return the learning rate to use next. *)
  let step p metric =
    let better =
      match p.mode with
      | `Min -> metric < p.best *. (1. -. p.threshold)
      | `Max -> metric > p.best *. (1. +. p.threshold)
    in
    if better then begin
      p.best <- metric;
      p.bad <- 0
    end
    else p.bad <- p.bad + 1;
    if p.cooling > 0 then begin
      p.cooling <- p.cooling - 1;
      p.bad <- 0
    end;
    if p.bad > p.patience then begin
      p.lr <- Float.max p.min_lr (p.lr *. p.factor);
      p.cooling <- p.cooldown;
      p.bad <- 0
    end;
    p.lr
end

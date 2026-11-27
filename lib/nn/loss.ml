(** Loss functions. Each returns a {!Var.t} ready for {!Var.backward}: a scalar,
    unless [~reduction:`None] asks for the loss of every element.

    They are written in terms of {!Var} operations, so autograd differentiates
    them. The ones that need a stable closed form ([cross_entropy],
    [bce_with_logits]) are single fused nodes. Each matches the PyTorch loss of
    the same name, value and gradient, with one difference: classes are the
    {e last} axis of the logits, as everywhere in FlockML. *)

type reduction = [ `Mean | `Sum | `None ]
(** How the per-element losses become the result: their mean (the default),
    their sum, or [`None] to keep them all. *)

module type S = sig
  type var
  (** The autograd node type. *)

  type tensor
  (** The backend's tensor type, for class weights. *)

  val mse : ?reduction:reduction -> var -> var -> var
  (** Mean squared error: [(pred - target)²]. *)

  val l1 : ?reduction:reduction -> var -> var -> var
  (** Mean absolute error: [|pred - target|]. *)

  val smooth_l1 : ?reduction:reduction -> ?beta:float -> var -> var -> var
  (** [0.5 d² / beta] where [|d| < beta], [|d| - 0.5 beta] elsewhere, for
      [d = pred - target]. [beta] defaults to 1; [beta = 0] is {!l1}. *)

  val huber : ?reduction:reduction -> ?delta:float -> var -> var -> var
  (** [0.5 d²] where [|d| <= delta], [delta (|d| - 0.5 delta)] elsewhere.
      [delta] defaults to 1, and must be positive. *)

  val cross_entropy :
    ?reduction:reduction ->
    ?label_smoothing:float ->
    ?weights:tensor ->
    var ->
    int array ->
    var
  (** [cross_entropy logits labels] is the softmax cross-entropy for logits
      [[| ...; classes |]] (for example [[| batch; classes |]], or
      [[| batch; time; classes |]] for sequences) and one label per row, in
      row-major order. [`None] keeps the leading shape.
      - [weights] ([[| classes |]]) weights each row's loss by its label's
        weight, and [`Mean] then divides by the sum of those weights, as PyTorch
        does.
      - [label_smoothing] [e] (default 0) trains towards
        [(1 - e) one_hot + e / classes].

      Without either, and with [`Mean], it is the single fused
      {!Var.softmax_cross_entropy} node. With all-zero logits the loss is
      [log classes]. Raises [Invalid_argument] for a label out of range or the
      wrong number of labels. *)

  val nll : ?reduction:reduction -> ?weights:tensor -> var -> int array -> var
  (** [nll log_probs labels] is the negative log-likelihood of [log_probs] (the
      output of a log-softmax), [-log_probs[label]], with [weights] as in
      {!cross_entropy}. *)

  val binary_cross_entropy :
    ?reduction:reduction -> ?eps:float -> var -> var -> var
  (** [binary_cross_entropy probs targets] = [-(t log p + (1 - t) log (1 - p))].
      [probs] are already in (0, 1) (e.g. after a sigmoid); [eps] (default 1e-7)
      guards against log 0. Prefer {!bce_with_logits}. *)

  val bce_with_logits :
    ?reduction:reduction -> ?pos_weight:tensor -> var -> var -> var
  (** [bce_with_logits logits targets] is {!binary_cross_entropy} of
      [sigmoid logits], computed in one stable node without an epsilon:
      [-(pw t log σ(x) + (1 - t) log (1 - σ(x)))], where [log σ(x)] is
      [-softplus (-x)]. [pos_weight] (broadcast against the logits, default 1)
      weights the positive term. The gradient is
      [(1 - t) σ(x) - pw t (1 - σ(x))], which is [σ(x) - t] without
      [pos_weight]. *)

  val kl_div :
    ?reduction:[ `Mean | `Sum | `None | `Batch_mean ] ->
    ?log_target:bool ->
    var ->
    var ->
    var
  (** [kl_div input target], as PyTorch: [input] holds log-probabilities and
      [target] probabilities (or log-probabilities with [log_target]), and the
      loss is [target (log target - input)], 0 where [target = 0]. [`Batch_mean]
      divides the sum by the size of the first axis, which is the true KL
      divergence per sample; [`Mean] (the default, as PyTorch) divides by the
      number of elements. *)
end

(** The losses, for one backend. *)
module Make (T : Tensor_intf.S) (Var : Autograd_intf.S with type tensor = T.t) :
  S with type var = Var.t and type tensor = T.t = struct
  type var = Var.t
  type tensor = T.t

  let reduce (reduction : reduction) x =
    match reduction with `Mean -> Var.mean x | `Sum -> Var.sum x | `None -> x

  (** [mean ((pred - target)²)], built from Var ops so autograd differentiates
      it. *)
  let mse ?(reduction = `Mean) pred target =
    reduce reduction (Var.pow (Var.sub pred target) 2.0)

  let l1 ?(reduction = `Mean) pred target =
    reduce reduction (Var.abs (Var.sub pred target))

  (** The quadratic piece [0.5 k d²] where [|d| < cut], and [linear] (a function
      of [|d|]) elsewhere. *)
  let piecewise ~cut ~k ~linear pred target =
    let d = Var.sub pred target in
    let ad = Var.abs d in
    let near = T.lt (Var.value ad) (T.scalar cut) in
    Var.where near (Var.scale (0.5 *. k) (Var.square d)) (linear ad)

  let smooth_l1 ?(reduction = `Mean) ?(beta = 1.0) pred target =
    if beta < 0. then invalid_arg "Loss.smooth_l1: beta must be >= 0";
    if beta = 0. then l1 ~reduction pred target
    else
      reduce reduction
        (piecewise ~cut:beta ~k:(1. /. beta)
           ~linear:(fun ad -> Var.add ad (Var.scalar (-0.5 *. beta)))
           pred target)

  (** As PyTorch, the quadratic piece includes [|d| = delta]; both pieces agree
      there. *)
  let huber ?(reduction = `Mean) ?(delta = 1.0) pred target =
    if not (delta > 0.) then invalid_arg "Loss.huber: delta must be positive";
    reduce reduction
      (piecewise ~cut:(Float.succ delta) ~k:1.
         ~linear:(fun ad ->
           Var.scale delta (Var.add ad (Var.scalar (-0.5 *. delta))))
         pred target)

  (** [(rows, classes, leading shape)] for logits [[| ...; classes |]], after
      checking the labels. *)
  let rows_of ~fn logits labels =
    let s = Var.shape logits in
    let nd = Array.length s in
    if nd < 1 then invalid_arg (Printf.sprintf "Loss.%s: scalar logits" fn);
    let c = s.(nd - 1) in
    let lead = Array.sub s 0 (nd - 1) in
    let rows = Layout.numel_of_shape lead in
    if Array.length labels <> rows then
      invalid_arg
        (Printf.sprintf "Loss.%s: %d labels for %d rows of logits %s" fn
           (Array.length labels) rows (Layout.shape_to_string s));
    Array.iter
      (fun l ->
        if l < 0 || l >= c then
          invalid_arg
            (Printf.sprintf "Loss.%s: label %d out of range for %d classes" fn l
               c))
      labels;
    (rows, c, lead)

  (** The loss of each row, [-sum_c target_c weight_c lp_c], for log
      probabilities [lp] ([[| rows; classes |]]), and the [`Mean] divisor (the
      sum of the label weights, or the number of rows). The target weights form
      one constant matrix, so the whole loss is two Var ops on top of [lp]. *)
  let weighted_nll ~fn ?weights ?(smoothing = 0.) lp labels rows c =
    let w =
      match weights with
      | None -> Array.make c 1.
      | Some w ->
          if T.shape w <> [| c |] then
            invalid_arg
              (Printf.sprintf "Loss.%s: weights of shape %s for %d classes" fn
                 (Layout.shape_to_string (T.shape w))
                 c);
          T.to_array w
    in
    let off = smoothing /. float_of_int c in
    let target = Array.make (rows * c) 0. in
    let denom = ref 0. in
    Array.iteri
      (fun i l ->
        denom := !denom +. w.(l);
        for j = 0 to c - 1 do
          let q = (if j = l then 1. -. smoothing else 0.) +. off in
          target.((i * c) + j) <- q *. w.(j)
        done)
      labels;
    let per_row =
      Var.neg
        (Var.sum_axis
           (Var.mul lp (Var.const (T.of_array [| rows; c |] target)))
           1)
    in
    (per_row, if weights = None then float_of_int rows else !denom)

  let finish (reduction : reduction) (per_row, denom) lead =
    match reduction with
    | `None -> Var.reshape per_row lead
    | `Sum -> Var.sum per_row
    | `Mean -> Var.scale (1. /. denom) (Var.sum per_row)

  (** The plain case is a single fused {!Var.softmax_cross_entropy} node: it
      skips the target matrix and intermediates, and its backward pass is the
      closed form [(softmax - one_hot) / batch]. *)
  let cross_entropy ?(reduction = `Mean) ?(label_smoothing = 0.) ?weights logits
      labels =
    let rows, c, lead = rows_of ~fn:"cross_entropy" logits labels in
    if label_smoothing < 0. || label_smoothing > 1. then
      invalid_arg "Loss.cross_entropy: label_smoothing must be in [0, 1]";
    let flat =
      if Array.length lead = 1 then logits else Var.reshape logits [| rows; c |]
    in
    if reduction = `Mean && label_smoothing = 0. && weights = None then
      Var.softmax_cross_entropy flat labels
    else
      let lp = Var.log_softmax flat in
      finish reduction
        (weighted_nll ~fn:"cross_entropy" ?weights ~smoothing:label_smoothing lp
           labels rows c)
        lead

  let nll ?(reduction = `Mean) ?weights log_probs labels =
    let rows, c, lead = rows_of ~fn:"nll" log_probs labels in
    let lp =
      if Array.length lead = 1 then log_probs
      else Var.reshape log_probs [| rows; c |]
    in
    finish reduction (weighted_nll ~fn:"nll" ?weights lp labels rows c) lead

  (** A direct translation of [-(t log p + (1 - t) log (1 - p))] into Var ops,
      with [eps] added inside each log so neither is ever [log 0]. *)
  let binary_cross_entropy ?(reduction = `Mean) ?(eps = 1e-7) probs targets =
    let open Var.Infix in
    let one = Var.scalar 1. and eps = Var.scalar eps in
    (* Adding eps keeps both logs finite when p hits exactly 0 or 1. *)
    let log_p = Var.log (probs + eps)
    and log_1mp = Var.log (one - probs + eps) in
    reduce reduction (Var.neg ((targets * log_p) + ((one - targets) * log_1mp)))

  (** [softplus z = max z 0 + log (1 + e^-|z|)], which never overflows. *)
  let softplus z = T.add (T.relu z) (T.log1p (T.exp (T.neg (T.abs z))))

  let bce_with_logits ?(reduction = `Mean) ?pos_weight logits targets =
    let x = Var.value logits and t = Var.value targets in
    let s = T.broadcast_shapes (T.shape x) (T.shape t) in
    if s <> T.shape x then
      invalid_arg
        (Printf.sprintf "Loss.bce_with_logits: targets %s don't match logits %s"
           (Layout.shape_to_string (T.shape t))
           (Layout.shape_to_string (T.shape x)));
    let t = T.broadcast_to t s in
    let pt = match pos_weight with Some pw -> T.mul pw t | None -> t in
    let sp_neg = softplus (T.neg x) and sp_pos = softplus x in
    (* pw t softplus(-x) + (1 - t) softplus(x) *)
    let value = T.add (T.mul pt sp_neg) (T.sub sp_pos (T.mul t sp_pos)) in
    let loss =
      Var.custom ~name:"bce_with_logits" ~inputs:[ logits; targets ] ~value
        ~backward:(fun g ->
          let sg = T.sigmoid x in
          (* d/dx = (1 - t) σ - pw t (1 - σ);  d/dt = pw softplus(-x) - softplus(x) *)
          let dx = T.sub (T.sub sg (T.mul t sg)) (T.sub pt (T.mul pt sg)) in
          let dt =
            match pos_weight with
            | Some pw -> T.sub (T.mul (T.broadcast_to pw s) sp_neg) sp_pos
            | None -> T.sub sp_neg sp_pos
          in
          [
            Some (T.mul g dx);
            Some (T.sum_to_shape (T.mul g dt) (T.shape (Var.value targets)));
          ])
    in
    reduce reduction loss

  let kl_div ?(reduction = `Mean) ?(log_target = false) input target =
    let pointwise =
      if log_target then Var.mul (Var.exp target) (Var.sub target input)
      else
        (* xlogy(t, t) - t input, with 0 log 0 = 0 *)
        let pos = T.gt (Var.value target) (T.scalar 0.) in
        let t_log_t =
          Var.where pos
            (Var.mul target (Var.log (Var.where pos target (Var.scalar 1.))))
            (Var.scalar 0.)
        in
        Var.sub t_log_t (Var.mul target input)
    in
    match reduction with
    | `Mean -> Var.mean pointwise
    | `Sum -> Var.sum pointwise
    | `None -> pointwise
    | `Batch_mean ->
        let n = match Var.shape pointwise with [||] -> 1 | s -> s.(0) in
        Var.scale (1. /. float_of_int n) (Var.sum pointwise)
end

(** Classification metrics, and a running average for logging. *)

type report = {
  precision : float array;
  recall : float array;
  f1 : float array;
  support : int array;  (** how many examples of each class there are *)
  accuracy : float;
  macro_f1 : float;  (** the unweighted mean of [f1] *)
  weighted_f1 : float;  (** the mean of [f1], weighted by [support] *)
}
(** Precision, recall and F1 of each class, as [sklearn]'s
    [classification_report] computes them. A class that is never predicted has
    precision 0, and one that never occurs has recall 0. *)

(** A running (weighted) mean, for losses over an epoch. *)
module Running = struct
  type t = { mutable sum : float; mutable weight : float; mutable count : int }

  let create () = { sum = 0.; weight = 0.; count = 0 }

  (** [add ?weight r x] adds [x] with [weight] (default 1): weight a batch's
      mean loss by its size to get the mean per example. *)
  let add ?(weight = 1.) r x =
    r.sum <- r.sum +. (weight *. x);
    r.weight <- r.weight +. weight;
    r.count <- r.count + 1

  (** The mean so far, or [nan] before anything was added. *)
  let mean r = if r.weight = 0. then Float.nan else r.sum /. r.weight

  (** How many values were added. *)
  let count r = r.count

  let reset r =
    r.sum <- 0.;
    r.weight <- 0.;
    r.count <- 0
end

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  val predict : tensor -> int array
  (** Predicted class per row of a [[| n; classes |]] score matrix (argmax). *)

  val accuracy : predictions:int array -> labels:int array -> float
  (** Fraction in [0, 1] of positions where prediction = label. *)

  val confusion_matrix :
    num_classes:int ->
    predictions:int array ->
    labels:int array ->
    int array array
  (** [m.(true_label).(predicted)] counts. *)

  val pp_confusion_matrix : Format.formatter -> int array array -> unit
  (** Print a confusion matrix as an aligned table, true classes down the side
      and predicted classes across the top. *)

  val top_k : k:int -> tensor -> int array array
  (** The [k] highest-scoring classes of each row of a [[| n; classes |]] score
      matrix, best first (ties go to the lower class). Raises [Invalid_argument]
      unless [1 <= k <= classes]. *)

  val top_k_accuracy : k:int -> scores:tensor -> labels:int array -> float
  (** The fraction of rows whose label is among their [k] best classes. *)

  type nonrec report = report = {
    precision : float array;
    recall : float array;
    f1 : float array;
    support : int array;
    accuracy : float;
    macro_f1 : float;
    weighted_f1 : float;
  }

  val classification_report :
    num_classes:int -> predictions:int array -> labels:int array -> report
  (** Per-class precision, recall and F1. *)

  val pp_report : ?names:string array -> Format.formatter -> report -> unit
  (** Print a report as a table, one row per class. *)

  module Running = Running
  (** A running mean: [create], [add ?weight], [mean], [count], [reset]. *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  (** The highest-scoring class of each row. *)
  let predict scores = T.argmax_axis scores 1

  (** Raise [Invalid_argument] (naming function [fn]) unless [predictions] and
      [labels] have the same length. *)
  let check_lengths ~fn ~predictions ~labels =
    let np = Array.length predictions and nl = Array.length labels in
    if np <> nl then
      invalid_arg
        (Printf.sprintf "Metrics.%s: %d predictions but %d labels" fn np nl)

  (** Empty input returns [nan]: accuracy of zero examples is 0/0, undefined.
      Returning 0.0 would make an empty eval set look like a terrible model, and
      1.0 like a perfect one; [nan] is impossible to mistake for a real score.
  *)
  let accuracy ~predictions ~labels =
    check_lengths ~fn:"accuracy" ~predictions ~labels;
    let n = Array.length labels in
    if n = 0 then Float.nan
    else begin
      let correct = ref 0 in
      Array.iteri
        (fun i label -> if predictions.(i) = label then incr correct)
        labels;
      float_of_int !correct /. float_of_int n
    end

  (** Count each (true, predicted) pair. The diagonal holds the correct
      predictions; off-diagonal cells show which classes get confused. Raises
      [Invalid_argument] on a length mismatch or a class that is negative or at
      least [num_classes]. *)
  let confusion_matrix ~num_classes ~predictions ~labels =
    check_lengths ~fn:"confusion_matrix" ~predictions ~labels;
    let in_range c = c >= 0 && c < num_classes in
    let m = Array.make_matrix num_classes num_classes 0 in
    Array.iteri
      (fun i label ->
        let pred = predictions.(i) in
        if not (in_range label && in_range pred) then
          invalid_arg
            (Printf.sprintf
               "Metrics.confusion_matrix: position %d has label %d, prediction \
                %d; both must be in [0, %d)"
               i label pred num_classes);
        (* rows = true class, columns = predicted class *)
        m.(label).(pred) <- m.(label).(pred) + 1)
      labels;
    m

  (** One row per true class, one column per predicted class. *)
  let pp_confusion_matrix fmt m =
    let n = Array.length m in
    Format.fprintf fmt "@[<v>true\\pred";
    for j = 0 to n - 1 do
      Format.fprintf fmt "%6d" j
    done;
    Array.iteri
      (fun i row ->
        Format.fprintf fmt "@,%9d" i;
        Array.iter (fun c -> Format.fprintf fmt "%6d" c) row)
      m;
    Format.fprintf fmt "@]"

  module Running = Running

  let top_k ~k scores =
    let s = T.shape scores in
    if Array.length s <> 2 || k < 1 || k > s.(1) then
      invalid_arg
        (Printf.sprintf "Metrics.top_k: k = %d for scores of shape %s" k
           (Layout.shape_to_string s));
    let n = s.(0) and c = s.(1) and a = T.to_array scores in
    Array.init n (fun i ->
        let idx = Array.init c Fun.id in
        (* best first; stable, so equal scores keep the lower class first *)
        Array.stable_sort
          (fun x y -> Float.compare a.((i * c) + y) a.((i * c) + x))
          idx;
        Array.sub idx 0 k)

  let top_k_accuracy ~k ~scores ~labels =
    let best = top_k ~k scores in
    check_lengths ~fn:"top_k_accuracy"
      ~predictions:(Array.map (fun r -> r.(0)) best)
      ~labels;
    let n = Array.length labels in
    if n = 0 then Float.nan
    else
      let hits = ref 0 in
      Array.iteri (fun i l -> if Array.mem l best.(i) then incr hits) labels;
      float_of_int !hits /. float_of_int n

  type nonrec report = report = {
    precision : float array;
    recall : float array;
    f1 : float array;
    support : int array;
    accuracy : float;
    macro_f1 : float;
    weighted_f1 : float;
  }

  let classification_report ~num_classes ~predictions ~labels =
    let m = confusion_matrix ~num_classes ~predictions ~labels in
    let ratio a b = if b = 0 then 0. else float_of_int a /. float_of_int b in
    let support = Array.map (Array.fold_left ( + ) 0) m in
    let predicted =
      Array.init num_classes (fun j ->
          Array.fold_left (fun acc row -> acc + row.(j)) 0 m)
    in
    let precision =
      Array.init num_classes (fun j -> ratio m.(j).(j) predicted.(j))
    in
    let recall =
      Array.init num_classes (fun j -> ratio m.(j).(j) support.(j))
    in
    let f1 =
      Array.init num_classes (fun j ->
          let p = precision.(j) and r = recall.(j) in
          if p +. r = 0. then 0. else 2. *. p *. r /. (p +. r))
    in
    let total = Array.fold_left ( + ) 0 support in
    let sumf = Array.fold_left ( +. ) 0. in
    {
      precision;
      recall;
      f1;
      support;
      accuracy = accuracy ~predictions ~labels;
      macro_f1 =
        (if num_classes = 0 then Float.nan
         else sumf f1 /. float_of_int num_classes);
      weighted_f1 =
        (if total = 0 then Float.nan
         else
           sumf (Array.mapi (fun j f -> f *. float_of_int support.(j)) f1)
           /. float_of_int total);
    }

  let pp_report ?names fmt r =
    let name j =
      match names with
      | Some ns when j < Array.length ns -> ns.(j)
      | _ -> string_of_int j
    in
    let w =
      Array.fold_left max 8
        (Array.init (Array.length r.f1) (fun j -> String.length (name j)))
    in
    Format.fprintf fmt "@[<v>%*s  precision  recall      f1  support" w "";
    Array.iteri
      (fun j f ->
        Format.fprintf fmt "@,%*s  %9.4f  %6.4f  %6.4f  %7d" w (name j)
          r.precision.(j) r.recall.(j) f r.support.(j))
      r.f1;
    Format.fprintf fmt
      "@,@,%*s  accuracy %.4f, macro F1 %.4f, weighted F1 %.4f@]" w ""
      r.accuracy r.macro_f1 r.weighted_f1
end

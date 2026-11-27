(** Classification metrics. *)

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
end

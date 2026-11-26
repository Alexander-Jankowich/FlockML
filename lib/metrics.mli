(** Classification metrics. *)

val predict : Tensor.t -> int array
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

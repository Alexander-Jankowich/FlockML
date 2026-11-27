(** An in-memory labelled dataset and mini-batch iteration. *)

type t = {
  inputs : Tensor.t;  (** [[| n; ... |]]: one example per row along axis 0 *)
  labels : int array;  (** [n] class labels *)
}

val length : t -> int
(** The number of examples. *)

val shuffled_indices : Random.State.t -> int -> int array
(** A uniformly random permutation of [0 .. n-1]. *)

val batches :
  ?shuffle:bool ->
  ?reuse:bool ->
  ?rng:Random.State.t ->
  batch_size:int ->
  t ->
  (Tensor.t * int array) Seq.t
(** Mini-batches covering the dataset once (one epoch). The last batch may be
    smaller. [shuffle] defaults to [false].

    With [~reuse:true] (default [false]), every full-size batch is written
    into one tensor that the sequence allocates once, so a batch's inputs are
    only valid until the next batch is requested. This suits a training loop
    that finishes each step before taking the next batch, and saves an
    allocation per step. *)

val split : ?rng:Random.State.t -> ratio:float -> t -> t * t
(** [split ~ratio d] randomly partitions [d] into (first, second) with [ratio]
    of the examples in [first], e.g. a train/validation split. Without [rng] the
    split differs on every run. *)

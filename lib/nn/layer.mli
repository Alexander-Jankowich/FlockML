(** Neural network layers, represented as a plain algebraic data type.

    A model is just a value of type {!t} (usually a {!Sequential}), and
    {!forward} is one big pattern match. Parameters live inside the constructors
    as {!Var.param} leaves, so the optimizer can update them in place. *)

type t =
  | Linear of { w : Var.t; b : Var.t }
      (** y = x @ w + b, with w : [[| in; out |]], b : [[| out |]] *)
  | Conv2d of {
      w : Var.t;
      b : Var.t;
      kernel : int;
      stride : int;
      padding : int;
    }
      (** NHWC conv. w : [[| kernel * kernel * in_c; out_c |]], b :
          [[| out_c |]] *)
  | Max_pool2d of { size : int; stride : int }
  | Relu
  | Sigmoid
  | Tanh
  | Flatten  (** [[| n; ... |]] -> [[| n; product of the rest |]] *)
  | Dropout of float  (** drop probability *)
  | Sequential of t list

(** {1 Constructors} *)

val linear : ?rng:Random.State.t -> in_dim:int -> out_dim:int -> unit -> t

val conv2d :
  ?rng:Random.State.t ->
  ?stride:int ->
  ?padding:int ->
  in_channels:int ->
  out_channels:int ->
  kernel:int ->
  unit ->
  t

val max_pool2d : ?stride:int -> int -> t
(** [max_pool2d size]; [stride] defaults to [size]. *)

val relu : t
val sigmoid : t
val tanh : t
val flatten : t
val dropout : float -> t
val sequential : t list -> t

(** {1 Running a model} *)

val forward : ?train:bool -> ?rng:Random.State.t -> t -> Var.t -> Var.t
(** [train] (default [false]) only changes the behaviour of {!Dropout}. *)

val params : t -> Var.t list
(** Every trainable parameter, in a deterministic order. *)

val num_params : t -> int
val to_string : t -> string

(** Neural network layers, represented as a plain algebraic data type.

    A model is just a value of type {!t} (usually a {!Sequential}), and
    {!forward} is one big pattern match. Parameters live inside the constructors
    as {!Var.param} leaves, so the optimizer can update them in place. *)

type t =
  | Linear of { w : Var.t; b : Var.t }
      (** [y = x @ w + b], with [w : [| in; out |]], [b : [| out |]] *)
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
(** [linear ~in_dim ~out_dim ()] is a fully connected layer with He-initialised
    weights [[| in_dim; out_dim |]] and zero biases. *)

val conv2d :
  ?rng:Random.State.t ->
  ?stride:int ->
  ?padding:int ->
  in_channels:int ->
  out_channels:int ->
  kernel:int ->
  unit ->
  t
(** A 2-D convolution over NHWC input with square [kernel] x [kernel] filters.
    [stride] defaults to 1 and [padding] (zeros on every side) to 0, so the
    output is [(h + 2 padding - kernel) / stride + 1] pixels high (likewise
    wide) with [out_channels] channels. *)

val max_pool2d : ?stride:int -> int -> t
(** [max_pool2d size]; [stride] defaults to [size]. *)

val relu : t
(** Elementwise [max x 0]. *)

val sigmoid : t
(** Elementwise logistic function, squashing values into (0, 1). *)

val tanh : t
(** Elementwise hyperbolic tangent, squashing values into (-1, 1). *)

val flatten : t
(** Collapse every axis after the batch axis into one. *)

val dropout : float -> t
(** [dropout p] zeroes each element with probability [p] during training and
    scales the survivors by [1 / (1 - p)]; in evaluation it is the identity.
    Raises [Invalid_argument] unless [0 <= p < 1]. *)

val sequential : t list -> t
(** Apply layers in order, each one's output feeding the next. *)

(** {1 Running a model} *)

val forward : ?train:bool -> ?rng:Random.State.t -> t -> Var.t -> Var.t
(** [train] (default [false]) only changes the behaviour of {!Dropout}. *)

val params : t -> Var.t list
(** Every trainable parameter, in a deterministic order. *)

val num_params : t -> int
(** Total number of trainable scalars across all parameters. *)

val to_string : t -> string
(** A multi-line, PyTorch-style summary such as
    ["Sequential(\n  Linear(784 -> 128)\n  ReLU\n...)"]. *)

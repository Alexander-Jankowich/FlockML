(** Neural network layers, represented as a plain algebraic data type.

    A model is just a value of type {!t} (usually a {!Sequential}), and
    {!forward} is one big pattern match. Parameters live inside the constructors
    as {!Var.param} leaves, so the optimizer can update them in place. *)

module type S = sig
  type var
  (** The autograd node type that parameters and activations are. *)

  type t =
    | Linear of { w : var; b : var }
        (** [y = x @ w + b], with [w : [| in; out |]], [b : [| out |]] *)
    | Conv2d of { w : var; b : var; kernel : int; stride : int; padding : int }
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
  (** [linear ~in_dim ~out_dim ()] is a fully connected layer with
      He-initialised weights [[| in_dim; out_dim |]] and zero biases. *)

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

  val forward : ?train:bool -> ?rng:Random.State.t -> t -> var -> var
  (** [train] (default [false]) only changes the behaviour of {!Dropout}. *)

  val params : t -> var list
  (** Every trainable parameter, in a deterministic order. *)

  val num_params : t -> int
  (** Total number of trainable scalars across all parameters. *)

  val to_string : t -> string
  (** A multi-line, PyTorch-style summary such as
      ["Sequential(\n  Linear(784 -> 128)\n  ReLU\n...)"]. *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Var : Autograd_intf.S with type tensor = T.t) :
  S with type var = Var.t = struct
  type var = Var.t

  module Init = Init.Make (T)

  type t =
    | Linear of { w : Var.t; b : Var.t }
    | Conv2d of {
        w : Var.t;
        b : Var.t;
        kernel : int;
        stride : int;
        padding : int;
      }
    | Max_pool2d of { size : int; stride : int }
    | Relu
    | Sigmoid
    | Tanh
    | Flatten
    | Dropout of float
    | Sequential of t list

  (* ------------------------------------------------------------------------ *)
  (* Constructors                                                             *)
  (* ------------------------------------------------------------------------ *)

  (** Weights use He initialisation (suited to ReLU); biases start at zero. *)
  let linear ?rng ~in_dim ~out_dim () =
    Linear
      {
        w = Var.param (Init.he_normal ?rng ~fan_in:in_dim [| in_dim; out_dim |]);
        b = Var.param (T.zeros [| out_dim |]);
      }

  (** Weights are stored pre-flattened as [[| kernel * kernel * in_c; out_c |]]
      (the layout {!Tensor_intf.S.im2col} produces), He-initialised; biases
      start at zero. *)
  let conv2d ?rng ?(stride = 1) ?(padding = 0) ~in_channels ~out_channels
      ~kernel () =
    let fan_in = kernel * kernel * in_channels in
    Conv2d
      {
        w = Var.param (Init.he_normal ?rng ~fan_in [| fan_in; out_channels |]);
        b = Var.param (T.zeros [| out_channels |]);
        kernel;
        stride;
        padding;
      }

  let max_pool2d ?stride size =
    Max_pool2d { size; stride = Option.value stride ~default:size }

  let relu = Relu
  let sigmoid = Sigmoid
  let tanh = Tanh
  let flatten = Flatten

  let dropout p =
    if p < 0.0 || p >= 1.0 then invalid_arg "Layer.dropout: p must be in [0, 1)";
    Dropout p

  let sequential layers = Sequential layers

  (* ------------------------------------------------------------------------ *)
  (* Forward pass                                                              *)
  (* ------------------------------------------------------------------------ *)

  (** One pattern match over the layer type. Every case is built from Var ops,
      so no layer needs backward code of its own.
      - [Linear]: [x @ w + b], with [b] broadcast over the batch, as one fused
        {!Var.linear} node.
      - [Conv2d]: im2col turns each output pixel's receptive field into a row,
        one matmul applies every filter to every patch, and a reshape restores
        NHWC.
      - [Dropout]: inverted dropout, so evaluation needs no rescaling.
      - [Sequential]: a left fold, passing [train] and [rng] down. *)
  let rec forward ?(train = false) ?rng layer x =
    match layer with
    | Linear r -> Var.linear x r.w r.b
    | Conv2d r ->
        (* x : [n; h; w; in_c]   r.w : [kernel*kernel*in_c; out_c]   r.b : [out_c] *)
        let s = Var.shape x in
        let n = s.(0) and h = s.(1) and wd = s.(2) in
        let oh = ((h + (2 * r.padding) - r.kernel) / r.stride) + 1
        and ow = ((wd + (2 * r.padding) - r.kernel) / r.stride) + 1
        and out_c = (Var.shape r.w).(1) in
        (* One row per output pixel, holding its receptive field:
         [n*oh*ow; kernel*kernel*in_c] *)
        let cols =
          Var.im2col x ~kh:r.kernel ~kw:r.kernel ~stride:r.stride
            ~padding:r.padding
        in
        (* Every filter applied to every patch at once: [n*oh*ow; out_c] *)
        let out = Var.linear cols r.w r.b in
        Var.reshape out [| n; oh; ow; out_c |]
    | Max_pool2d r -> Var.max_pool2d x ~size:r.size ~stride:r.stride
    | Relu -> Var.relu x
    | Sigmoid -> Var.sigmoid x
    | Tanh -> Var.tanh x
    | Flatten ->
        (* [n; d1; d2; ...] -> [n; d1*d2*...]: keep the batch axis, merge the rest *)
        let s = Var.shape x in
        let n = s.(0) in
        let numel = Array.fold_left ( * ) 1 s in
        Var.reshape x [| n; numel / n |]
    | Dropout p ->
        if not train then x
        else
          (* Inverted dropout: keep each unit with probability 1 - p and scale
             the survivors by 1/(1-p), so the expected value is unchanged and
             eval mode needs no rescaling. *)
          let keep = 1. /. (1. -. p) in
          let mask =
            T.rand_uniform ?rng ~lo:0. ~hi:1. (Var.shape x)
            |> T.map (fun u -> if u > p then keep else 0.)
          in
          Var.mul x (Var.const mask)
    | Sequential layers ->
        (* Feed each layer's output into the next, threading train/rng down. *)
        List.fold_left (fun acc layer -> forward ~train ?rng layer acc) x layers

  (** [[w; b]] for Linear and Conv2d, concatenated in order for Sequential, and
      nothing for the parameter-free layers. *)
  let rec params = function
    | Linear r -> [ r.w; r.b ]
    | Conv2d r -> [ r.w; r.b ]
    | Sequential layers -> List.concat_map params layers
    | _ -> []

  (* ------------------------------------------------------------------------ *)
  (* Utilities                                                                *)
  (* ------------------------------------------------------------------------ *)

  (** Sum of the element counts of every parameter. *)
  let num_params layer =
    List.fold_left
      (fun acc p -> acc + Array.fold_left ( * ) 1 (Var.shape p))
      0 (params layer)

  (** Indent nested Sequential layers by two spaces per level. *)
  let rec to_string = function
    | Linear { w; _ } ->
        let s = Var.shape w in
        Printf.sprintf "Linear(%d -> %d)" s.(0) s.(1)
    | Conv2d { w; kernel; stride; padding; _ } ->
        let s = Var.shape w in
        Printf.sprintf "Conv2d(%d -> %d, kernel=%d, stride=%d, padding=%d)"
          (s.(0) / (kernel * kernel))
          s.(1) kernel stride padding
    | Max_pool2d { size; stride } ->
        Printf.sprintf "MaxPool2d(size=%d, stride=%d)" size stride
    | Relu -> "ReLU"
    | Sigmoid -> "Sigmoid"
    | Tanh -> "Tanh"
    | Flatten -> "Flatten"
    | Dropout p -> Printf.sprintf "Dropout(p=%g)" p
    | Sequential layers ->
        let inner =
          layers
          |> List.map (fun l ->
              to_string l |> String.split_on_char '\n'
              |> List.map (fun line -> "  " ^ line)
              |> String.concat "\n")
        in
        "Sequential(\n" ^ String.concat "\n" inner ^ "\n)"
end

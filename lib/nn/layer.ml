(** Neural network layers, represented as a plain algebraic data type.

    A model is just a value of type {!S.t} (usually a {!S.Sequential}), and
    {!S.forward} is one big pattern match. Parameters live inside the
    constructors as {!Var.param} leaves, so the optimizer can update them in
    place.

    The variant covers the common layers. Anything else is a {!S.Custom} layer,
    which carries its own forward function, parameters and sublayers, so it
    takes part in {!S.named_params}, checkpoints and {!S.summary} like the
    built-in layers. A model with several inputs or outputs can also be a plain
    OCaml function over a record of layers, wrapped in a [Custom] layer when it
    needs to be saved. *)

type load_report = {
  missing : string list;
  unexpected : string list;
  mismatched : (string * int array * int array) list;
}
(** What {!S.load_state_dict} found: the model's entries that were missing, the
    entries that matched nothing in the model, and those whose shape differed
    (name, the model's shape, the loaded shape). *)

(** The recurrent cells: an Elman RNN with [tanh] or [relu], an LSTM and a GRU,
    with PyTorch's equations and gate order. *)
type recurrent_kind = Rnn_tanh | Rnn_relu | Lstm | Gru

module type S = sig
  type var
  (** The autograd node type that parameters and activations are. *)

  type tensor
  (** The backend's tensor type. *)

  type nonrec load_report = load_report = {
    missing : string list;
    unexpected : string list;
    mismatched : (string * int array * int array) list;
  }

  type nonrec recurrent_kind = recurrent_kind =
    | Rnn_tanh
    | Rnn_relu
    | Lstm
    | Gru

  (** Weight layouts follow FlockML's [x · w] convention, so a weight is the
      transpose of PyTorch's: a linear layer's [w] is [[| in; out |]], and
      images are NHWC. Parameter names (in {!named_params}) are given with each
      constructor. *)
  type t =
    | Linear of { w : var; b : var option }
        (** [y = x · w + b], with [w : [| in; out |]], [b : [| out |]] ("w",
            "b"). [x] may have any number of leading axes. *)
    | Conv2d of {
        w : var;
        b : var option;
        kernel : int * int;
        stride : int * int;
        padding : int * int;
        dilation : int * int;
        groups : int;
      }
        (** NHWC convolution. [w : [| kh * kw * in_c / groups; out_c |]], rows
            ordered by (kernel row, kernel column, channel); output channels
            [j * out_c / groups ...] belong to group [j]. [b : [| out_c |]]. *)
    | Conv1d of {
        w : var;
        b : var option;
        kernel : int;
        stride : int;
        padding : int;
        dilation : int;
        groups : int;
      }
        (** Convolution over [[| n; length; channels |]].
            [w : [| kernel * in_c / groups; out_c |]]. *)
    | Conv_transpose2d of {
        w : var;
        b : var option;
        kernel : int * int;
        stride : int;
        padding : int;
        output_padding : int;
      }
        (** The adjoint of a convolution, for upsampling.
            [w : [| in_c; kh * kw * out_c |]]. *)
    | Max_pool2d of { size : int; stride : int; padding : int }
        (** NHWC max pooling. Padding counts as [-infinity]. *)
    | Avg_pool2d of { size : int; stride : int; padding : int }
        (** NHWC average pooling. Padding counts as zeros, and every window is
            divided by [size * size], as PyTorch's default. *)
    | Global_avg_pool
        (** [[| n; ...; c |]] -> [[| n; c |]]: the mean over every axis but the
            first and last. *)
    | Relu
    | Sigmoid
    | Tanh
    | Leaky_relu of float  (** the negative slope *)
    | Elu of float  (** alpha *)
    | Gelu of { approximate : bool }
    | Silu
    | Softplus of { beta : float; threshold : float }
    | Softmax  (** along the last axis *)
    | Log_softmax  (** along the last axis *)
    | Identity
    | Flatten  (** [[| n; ... |]] -> [[| n; product of the rest |]] *)
    | Dropout of float  (** drop probability *)
    | Batch_norm of {
        w : var;
        b : var;
        running_mean : var;
        running_var : var;
        num_batches_tracked : var;
        momentum : float;
        eps : float;
      }
        (** Normalises each channel (the last axis) over every other axis, with
            batch statistics in training and running statistics in evaluation.
            The running statistics are buffers ("running_mean", "running_var",
            "num_batches_tracked"); [w] and [b] are the per-channel scale and
            shift ("w", "b"). *)
    | Layer_norm of { w : var; b : var; shape : int array; eps : float }
        (** Normalises over the trailing axes [shape] ("w", "b" of that shape).
        *)
    | Group_norm of { w : var; b : var; groups : int; eps : float }
        (** Normalises each group of channels over the group and every axis but
            the first; then a per-channel scale and shift. *)
    | Embedding of { w : var }
        (** A lookup table [w : [| num; dim |]] ("w"). The input holds indices
            (as floats), and the output appends an axis of size [dim]. *)
    | Recurrent of {
        kind : recurrent_kind;
        w_ih : var;
        w_hh : var;
        b_ih : var;
        b_hh : var;
        hidden : int;
      }
        (** A recurrent layer over [[| batch; time; in |]], returning every
            step's hidden state, [[| batch; time; hidden |]].
            [w_ih : [| in; gates * hidden |]],
            [w_hh : [| hidden; gates * hidden |]], biases
            [[| gates * hidden |]], with PyTorch's gate order (LSTM: i, f, g, o;
            GRU: r, z, n). *)
    | Attention of {
        in_w : var;
        in_b : var;
        out_w : var;
        out_b : var;
        heads : int;
        causal : bool;
      }
        (** Multi-head self-attention over [[| batch; time; dim |]], as
            PyTorch's [MultiheadAttention]: [in_w : [| dim; 3 dim |]] packs the
            query, key and value projections, [out_w : [| dim; dim |]]. With
            [causal], position [i] only attends to positions [<= i]. *)
    | Residual of t  (** [x + f x] *)
    | Sequential of t list
    | Custom of {
        name : string;
        params : (string * var) list;
        buffers : (string * var) list;
        children : (string * t) list;
        forward : train:bool -> ?rng:Random.State.t -> var -> var;
      }
        (** A user-defined layer. [params] and [buffers] are its own tensors,
            and [children] the layers it calls in [forward]; all three are named
            for {!named_params} and checkpoints. *)

  (** {1 Constructors}

      Every constructor that makes weights takes [?rng] and [?init] (the default
      is given with each), and makes biases zero unless said otherwise. *)

  val linear :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    ?bias:bool ->
    in_dim:int ->
    out_dim:int ->
    unit ->
    t
  (** A fully connected layer, [He_normal] by default. [bias] defaults to
      [true]. *)

  val conv2d :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    ?bias:bool ->
    ?stride:int ->
    ?padding:int ->
    ?dilation:int ->
    ?groups:int ->
    ?kernel_size:int * int ->
    ?strides:int * int ->
    ?paddings:int * int ->
    ?dilations:int * int ->
    in_channels:int ->
    out_channels:int ->
    kernel:int ->
    unit ->
    t
  (** A 2-D convolution over NHWC input, [He_normal] by default. [kernel],
      [stride] (default 1), [padding] (zeros, default 0) and [dilation] (default
      1) apply to both axes; [kernel_size], [strides], [paddings] and
      [dilations] give (height, width) pairs instead. The output is
      [(h + 2 padding - dilation (kernel - 1) - 1) / stride + 1] pixels high
      (likewise wide). [groups] (default 1) splits the channels into independent
      groups; [groups = in_channels] is a depthwise convolution. Raises
      [Invalid_argument] if [groups] doesn't divide both channel counts. *)

  val conv1d :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    ?bias:bool ->
    ?stride:int ->
    ?padding:int ->
    ?dilation:int ->
    ?groups:int ->
    in_channels:int ->
    out_channels:int ->
    kernel:int ->
    unit ->
    t
  (** A 1-D convolution over [[| n; length; channels |]], as {!conv2d}. *)

  val conv_transpose2d :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    ?bias:bool ->
    ?stride:int ->
    ?padding:int ->
    ?output_padding:int ->
    in_channels:int ->
    out_channels:int ->
    kernel:int ->
    unit ->
    t
  (** A transposed convolution over NHWC input: the output is
      [(h - 1) stride - 2 padding + kernel + output_padding] pixels high
      (likewise wide). [output_padding] must be less than [stride]. *)

  val max_pool2d : ?stride:int -> ?padding:int -> int -> t
  (** [max_pool2d size]; [stride] defaults to [size], [padding] to 0. *)

  val avg_pool2d : ?stride:int -> ?padding:int -> int -> t
  (** [avg_pool2d size]; as {!max_pool2d}. *)

  val global_avg_pool : t

  val relu : t
  (** Elementwise [max x 0]. *)

  val sigmoid : t
  (** Elementwise logistic function, squashing values into (0, 1). *)

  val tanh : t
  (** Elementwise hyperbolic tangent, squashing values into (-1, 1). *)

  val leaky_relu : ?slope:float -> unit -> t
  (** [slope] defaults to 0.01. *)

  val elu : ?alpha:float -> unit -> t
  (** [alpha] defaults to 1. *)

  val gelu : ?approximate:bool -> unit -> t
  (** The exact GELU, or PyTorch's tanh approximation. *)

  val silu : t
  val softplus : ?beta:float -> ?threshold:float -> unit -> t
  val softmax : t
  val log_softmax : t
  val identity : t

  val flatten : t
  (** Collapse every axis after the batch axis into one. *)

  val dropout : float -> t
  (** [dropout p] zeroes each element with probability [p] during training and
      scales the survivors by [1 / (1 - p)]; in evaluation it is the identity.
      Raises [Invalid_argument] unless [0 <= p < 1]. *)

  val batch_norm : ?momentum:float -> ?eps:float -> int -> t
  (** [batch_norm channels] works on [[| n; c |]] (BatchNorm1d) and
      [[| n; h; w; c |]] (BatchNorm2d) alike. [momentum] (default 0.1) is the
      weight of the new batch in the running statistics, which use the unbiased
      variance, as in PyTorch; [eps] defaults to [1e-5]. *)

  val layer_norm : ?eps:float -> int array -> t
  (** [layer_norm shape] normalises over the trailing axes [shape]. *)

  val group_norm : ?eps:float -> groups:int -> int -> t
  (** [group_norm ~groups channels]. *)

  val embedding :
    ?rng:Random.State.t -> ?init:Init.scheme -> num:int -> dim:int -> unit -> t
  (** A table of [num] vectors of size [dim], [Normal 1.] by default, as in
      PyTorch. *)

  val rnn :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    ?nonlinearity:[ `Tanh | `Relu ] ->
    input:int ->
    hidden:int ->
    unit ->
    t

  val lstm :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    input:int ->
    hidden:int ->
    unit ->
    t

  val gru :
    ?rng:Random.State.t ->
    ?init:Init.scheme ->
    input:int ->
    hidden:int ->
    unit ->
    t
  (** The recurrent layers. Every weight and bias is drawn from U(-1/sqrt
      hidden, 1/sqrt hidden) by default, as in PyTorch; [?init] replaces that
      for the weights, and the biases are then zero. *)

  val multi_head_attention :
    ?rng:Random.State.t -> ?causal:bool -> dim:int -> heads:int -> unit -> t
  (** Self-attention with [heads] heads of size [dim / heads]. The input
      projection is Xavier-uniform and the output projection [Kaiming_uniform],
      with zero biases, as in PyTorch. [causal] defaults to [false]. *)

  val residual : t -> t
  val sequential : t list -> t

  val custom :
    ?params:(string * var) list ->
    ?buffers:(string * var) list ->
    ?children:(string * t) list ->
    name:string ->
    (train:bool -> ?rng:Random.State.t -> var -> var) ->
    t
  (** [custom ~name forward]: see {!Custom}. Raises [Invalid_argument] if two of
      its entries share a name. *)

  (** {1 Running a model} *)

  val forward : ?train:bool -> ?rng:Random.State.t -> t -> var -> var
  (** [train] (default [false]) changes the behaviour of {!Dropout} and
      {!Batch_norm}, and is passed to {!Custom} layers. *)

  val embed : t -> ?shape:int array -> int array -> var
  (** [embed layer idx] looks up an {!Embedding} with integer indices:
      [[| len; dim |]], or [shape] with [dim] appended. *)

  type state = { h : var; c : var option }
  (** A recurrent layer's state: the hidden state [[| batch; hidden |]], and for
      an LSTM the cell state. *)

  val init_state : t -> batch:int -> state
  (** Zeros, for a {!Recurrent} layer. *)

  val step : t -> x:var -> state -> state
  (** One time step of a {!Recurrent} layer: [x : [| batch; in |]]. *)

  val run : ?state:state -> t -> var -> var * state
  (** A {!Recurrent} layer over a whole sequence, from [state] (zeros by
      default): every step's hidden state, and the final state. *)

  val attention : ?mask:tensor -> t -> query:var -> key:var -> value:var -> var
  (** An {!Attention} layer with separate query, key and value inputs
      ([[| batch; tq; dim |]], [[| batch; tk; dim |]] twice), for
      cross-attention. [mask] (non-zero = not attended) broadcasts to
      [[| batch; heads; tq; tk |]], and combines with the layer's causal mask.
  *)

  (** {1 Parameters and state} *)

  val params : t -> var list
  (** Every trainable parameter, in a deterministic order. *)

  val named_params : t -> (string * var) list
  (** {!params} with their paths: a layer's own names, joined with ["."] to the
      index of each {!Sequential} element and the name of each {!Custom} child,
      as ["0.w"] or ["block1.conv.b"]. {!Residual} adds no component. *)

  val named_buffers : t -> (string * var) list
  (** The state that isn't trained (batch norm's running statistics), named the
      same way. Optimizers never see it; checkpoints save it. *)

  val state_dict : t -> (string * tensor) list
  (** The values of {!named_params}, then {!named_buffers}. These are the live
      tensors, not copies. *)

  val load_state_dict :
    ?strict:bool -> t -> (string * tensor) list -> load_report
  (** Copy each entry into the parameter or buffer of the same name. With
      [strict] (the default), raise [Failure] listing every problem, and change
      nothing, unless the names and shapes match exactly; otherwise load what
      matches and return the report. *)

  val num_params : t -> int
  (** Total number of trainable scalars across all parameters. *)

  val to_string : t -> string
  (** A multi-line, PyTorch-style summary such as
      ["Sequential(\n  Linear(784 -> 128)\n  ReLU\n...)"]. *)

  val summary : t -> input_shape:int array -> string
  (** A table of each layer's output shape and parameter count, like
      [torchinfo], from a [no_grad] evaluation-mode forward pass on zeros of
      [input_shape]. Sequential, residual and custom layers list their children,
      indented. *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Var : Autograd_intf.S with type tensor = T.t) :
  S with type var = Var.t and type tensor = T.t = struct
  type var = Var.t
  type tensor = T.t

  module Init = Init.Make (T)

  type nonrec load_report = load_report = {
    missing : string list;
    unexpected : string list;
    mismatched : (string * int array * int array) list;
  }

  type nonrec recurrent_kind = recurrent_kind =
    | Rnn_tanh
    | Rnn_relu
    | Lstm
    | Gru

  type t =
    | Linear of { w : var; b : var option }
    | Conv2d of {
        w : var;
        b : var option;
        kernel : int * int;
        stride : int * int;
        padding : int * int;
        dilation : int * int;
        groups : int;
      }
    | Conv1d of {
        w : var;
        b : var option;
        kernel : int;
        stride : int;
        padding : int;
        dilation : int;
        groups : int;
      }
    | Conv_transpose2d of {
        w : var;
        b : var option;
        kernel : int * int;
        stride : int;
        padding : int;
        output_padding : int;
      }
    | Max_pool2d of { size : int; stride : int; padding : int }
    | Avg_pool2d of { size : int; stride : int; padding : int }
    | Global_avg_pool
    | Relu
    | Sigmoid
    | Tanh
    | Leaky_relu of float
    | Elu of float
    | Gelu of { approximate : bool }
    | Silu
    | Softplus of { beta : float; threshold : float }
    | Softmax
    | Log_softmax
    | Identity
    | Flatten
    | Dropout of float
    | Batch_norm of {
        w : var;
        b : var;
        running_mean : var;
        running_var : var;
        num_batches_tracked : var;
        momentum : float;
        eps : float;
      }
    | Layer_norm of { w : var; b : var; shape : int array; eps : float }
    | Group_norm of { w : var; b : var; groups : int; eps : float }
    | Embedding of { w : var }
    | Recurrent of {
        kind : recurrent_kind;
        w_ih : var;
        w_hh : var;
        b_ih : var;
        b_hh : var;
        hidden : int;
      }
    | Attention of {
        in_w : var;
        in_b : var;
        out_w : var;
        out_b : var;
        heads : int;
        causal : bool;
      }
    | Residual of t
    | Sequential of t list
    | Custom of {
        name : string;
        params : (string * var) list;
        buffers : (string * var) list;
        children : (string * t) list;
        forward : train:bool -> ?rng:Random.State.t -> var -> var;
      }

  type state = { h : var; c : var option }

  let shape_str = Layout.shape_to_string

  (* ------------------------------------------------------------------------ *)
  (* Constructors                                                             *)
  (* ------------------------------------------------------------------------ *)

  let weight ?rng ~init ~fan_in ~fan_out shape =
    Var.param (Init.make ?rng init ~fan_in ~fan_out shape)

  let zeros_param n = Var.param (T.zeros [| n |])
  let bias_param bias n = if bias then Some (zeros_param n) else None

  (** Weights use He initialisation (suited to ReLU) by default; biases start at
      zero. *)
  let linear ?rng ?(init = Init.He_normal) ?(bias = true) ~in_dim ~out_dim () =
    Linear
      {
        w =
          weight ?rng ~init ~fan_in:in_dim ~fan_out:out_dim
            [| in_dim; out_dim |];
        b = bias_param bias out_dim;
      }

  let check_groups ~fn ~groups in_c out_c =
    if groups < 1 || in_c mod groups <> 0 || out_c mod groups <> 0 then
      invalid_arg
        (Printf.sprintf
           "Layer.%s: groups = %d must divide in_channels = %d and \
            out_channels = %d"
           fn groups in_c out_c)

  (** Weights are stored pre-flattened as [[| kh * kw * in_c / groups; out_c |]]
      (the layout {!Tensor_intf.S.im2col} produces). *)
  let conv2d ?rng ?(init = Init.He_normal) ?(bias = true) ?(stride = 1)
      ?(padding = 0) ?(dilation = 1) ?(groups = 1) ?kernel_size ?strides
      ?paddings ?dilations ~in_channels ~out_channels ~kernel () =
    check_groups ~fn:"conv2d" ~groups in_channels out_channels;
    let pair d = function Some p -> p | None -> (d, d) in
    let ((kh, kw) as kernel) = pair kernel kernel_size in
    let fan_in = kh * kw * (in_channels / groups) in
    Conv2d
      {
        w =
          weight ?rng ~init ~fan_in
            ~fan_out:(kh * kw * out_channels / groups)
            [| fan_in; out_channels |];
        b = bias_param bias out_channels;
        kernel;
        stride = pair stride strides;
        padding = pair padding paddings;
        dilation = pair dilation dilations;
        groups;
      }

  let conv1d ?rng ?(init = Init.He_normal) ?(bias = true) ?(stride = 1)
      ?(padding = 0) ?(dilation = 1) ?(groups = 1) ~in_channels ~out_channels
      ~kernel () =
    check_groups ~fn:"conv1d" ~groups in_channels out_channels;
    let fan_in = kernel * (in_channels / groups) in
    Conv1d
      {
        w =
          weight ?rng ~init ~fan_in
            ~fan_out:(kernel * out_channels / groups)
            [| fan_in; out_channels |];
        b = bias_param bias out_channels;
        kernel;
        stride;
        padding;
        dilation;
        groups;
      }

  let conv_transpose2d ?rng ?(init = Init.He_normal) ?(bias = true)
      ?(stride = 1) ?(padding = 0) ?(output_padding = 0) ~in_channels
      ~out_channels ~kernel () =
    if output_padding < 0 || output_padding >= max stride 1 then
      invalid_arg "Layer.conv_transpose2d: output_padding must be < stride";
    (* PyTorch computes the fans of a transposed convolution's weight as if
       it were a convolution from out_channels to in_channels. *)
    let fan_in = out_channels * kernel * kernel in
    Conv_transpose2d
      {
        w =
          weight ?rng ~init ~fan_in
            ~fan_out:(in_channels * kernel * kernel)
            [| in_channels; kernel * kernel * out_channels |];
        b = bias_param bias out_channels;
        kernel = (kernel, kernel);
        stride;
        padding;
        output_padding;
      }

  let max_pool2d ?stride ?(padding = 0) size =
    Max_pool2d { size; stride = Option.value stride ~default:size; padding }

  let avg_pool2d ?stride ?(padding = 0) size =
    Avg_pool2d { size; stride = Option.value stride ~default:size; padding }

  let global_avg_pool = Global_avg_pool
  let relu = Relu
  let sigmoid = Sigmoid
  let tanh = Tanh
  let leaky_relu ?(slope = 0.01) () = Leaky_relu slope
  let elu ?(alpha = 1.0) () = Elu alpha
  let gelu ?(approximate = false) () = Gelu { approximate }
  let silu = Silu

  let softplus ?(beta = 1.0) ?(threshold = 20.0) () =
    Softplus { beta; threshold }

  let softmax = Softmax
  let log_softmax = Log_softmax
  let identity = Identity
  let flatten = Flatten

  let dropout p =
    if p < 0.0 || p >= 1.0 then invalid_arg "Layer.dropout: p must be in [0, 1)";
    Dropout p

  let batch_norm ?(momentum = 0.1) ?(eps = 1e-5) c =
    Batch_norm
      {
        w = Var.param (T.ones [| c |]);
        b = zeros_param c;
        running_mean = Var.const (T.zeros [| c |]);
        running_var = Var.const (T.ones [| c |]);
        num_batches_tracked = Var.const (T.zeros [||]);
        momentum;
        eps;
      }

  let layer_norm ?(eps = 1e-5) shape =
    Layer_norm
      {
        w = Var.param (T.ones shape);
        b = Var.param (T.zeros shape);
        shape = Array.copy shape;
        eps;
      }

  let group_norm ?(eps = 1e-5) ~groups c =
    if groups < 1 || c mod groups <> 0 then
      invalid_arg
        (Printf.sprintf "Layer.group_norm: %d groups don't divide %d channels"
           groups c);
    Group_norm
      { w = Var.param (T.ones [| c |]); b = zeros_param c; groups; eps }

  let embedding ?rng ?(init = Init.Normal 1.0) ~num ~dim () =
    Embedding { w = weight ?rng ~init ~fan_in:num ~fan_out:dim [| num; dim |] }

  let recurrent kind ?rng ?init ~input ~hidden () =
    let gates =
      match kind with Rnn_tanh | Rnn_relu -> 1 | Lstm -> 4 | Gru -> 3
    in
    let g = gates * hidden in
    let bound = 1. /. Float.sqrt (float_of_int hidden) in
    let w shape fan_in =
      match init with
      | Some init -> weight ?rng ~init ~fan_in ~fan_out:g shape
      | None -> Var.param (Init.uniform ?rng ~bound shape)
    in
    let b () =
      match init with
      | Some _ -> zeros_param g
      | None -> Var.param (Init.uniform ?rng ~bound [| g |])
    in
    let w_ih = w [| input; g |] input in
    let w_hh = w [| hidden; g |] hidden in
    let b_ih = b () in
    let b_hh = b () in
    Recurrent { kind; w_ih; w_hh; b_ih; b_hh; hidden }

  let rnn ?rng ?init ?(nonlinearity = `Tanh) ~input ~hidden () =
    recurrent
      (match nonlinearity with `Tanh -> Rnn_tanh | `Relu -> Rnn_relu)
      ?rng ?init ~input ~hidden ()

  let lstm ?rng ?init ~input ~hidden () =
    recurrent Lstm ?rng ?init ~input ~hidden ()

  let gru ?rng ?init ~input ~hidden () =
    recurrent Gru ?rng ?init ~input ~hidden ()

  let multi_head_attention ?rng ?(causal = false) ~dim ~heads () =
    if heads < 1 || dim mod heads <> 0 then
      invalid_arg
        (Printf.sprintf
           "Layer.multi_head_attention: %d heads don't divide dim = %d" heads
           dim);
    Attention
      {
        in_w =
          weight ?rng ~init:Init.Xavier_uniform ~fan_in:dim ~fan_out:(3 * dim)
            [| dim; 3 * dim |];
        in_b = zeros_param (3 * dim);
        out_w =
          weight ?rng ~init:Init.Kaiming_uniform ~fan_in:dim ~fan_out:dim
            [| dim; dim |];
        out_b = zeros_param dim;
        heads;
        causal;
      }

  let residual f = Residual f
  let sequential layers = Sequential layers

  let check_unique ~fn names =
    let sorted = List.sort compare names in
    let rec go = function
      | a :: (b :: _ as rest) ->
          if a = b then
            invalid_arg
              (Printf.sprintf "Layer.%s: the name %S is used twice" fn a)
          else go rest
      | _ -> ()
    in
    go sorted

  let custom ?(params = []) ?(buffers = []) ?(children = []) ~name forward =
    check_unique ~fn:"custom"
      (List.map fst params @ List.map fst buffers @ List.map fst children);
    Custom { name; params; buffers; children; forward }

  (* ------------------------------------------------------------------------ *)
  (* Forward pass                                                              *)
  (* ------------------------------------------------------------------------ *)

  let add_bias y = function Some b -> Var.add y b | None -> y

  (** [x · w + b] for any number of leading axes. A 2-D input with a bias is one
      fused {!Var.linear} node. *)
  let affine x w b =
    match b with
    | Some b when Array.length (Var.shape x) = 2 -> Var.linear x w b
    | _ -> add_bias (Var.matmul x w) b

  (** The columns of a convolution, [[| n * oh * ow; kh * kw * c |]], and
      [(n, oh, ow)]. {!Var.im2col} does the common cases (no dilation, the same
      stride on both axes, or a single row). The others take one strided slice
      of the padded input per kernel position, concatenated in the same (kernel
      row, kernel column, channel) order. *)
  let conv_columns x ~kernel:(kh, kw) ~stride:(sh, sw) ~padding:(ph, pw)
      ~dilation:(dh, dw) =
    let s = Var.shape x in
    if Array.length s <> 4 then
      invalid_arg
        (Printf.sprintf "Layer.conv2d: expected NHWC input, got %s"
           (shape_str s));
    let n = s.(0) and h = s.(1) and w = s.(2) in
    let hp = h + (2 * ph) and wp = w + (2 * pw) in
    let eh = (dh * (kh - 1)) + 1 and ew = (dw * (kw - 1)) + 1 in
    if hp < eh || wp < ew then
      invalid_arg
        (Printf.sprintf "Layer.conv2d: a %dx%d kernel doesn't fit input %s" eh
           ew (shape_str s));
    let oh = ((hp - eh) / sh) + 1 and ow = ((wp - ew) / sw) + 1 in
    let cols =
      if dh = 1 && dw = 1 && sh = sw && ph = pw then
        Var.im2col x ~kh ~kw ~stride:sh ~padding:ph
      else begin
        let xp =
          if ph = 0 && pw = 0 then x
          else Var.pad x [| (0, 0); (ph, ph); (pw, pw); (0, 0) |]
        in
        if dh = 1 && dw = 1 && kh = 1 && hp = 1 then
          Var.im2col xp ~kh ~kw ~stride:sw ~padding:0
        else
          let taps =
            List.concat
              (List.init kh (fun ky ->
                   List.init kw (fun kx ->
                       let y0 = ky * dh and x0 = kx * dw in
                       Var.slice xp
                         [
                           Tensor_intf.All;
                           Step (y0, y0 + ((oh - 1) * sh) + 1, sh);
                           Step (x0, x0 + ((ow - 1) * sw) + 1, sw);
                           All;
                         ])))
          in
          let c = s.(3) in
          Var.reshape (Var.concat ~axis:3 taps) [| n * oh * ow; kh * kw * c |]
      end
    in
    (cols, n, oh, ow)

  (** A convolution as columns times weights. With groups, the columns and the
      weights are split into [groups] blocks and multiplied as one batched
      matmul. *)
  let conv ~fn x ~w ~b ~kernel ~stride ~padding ~dilation ~groups =
    let in_c = (Var.shape x).(Array.length (Var.shape x) - 1) in
    let ws = Var.shape w in
    let kh, kw = kernel in
    if ws.(0) * groups <> kh * kw * in_c then
      invalid_arg
        (Printf.sprintf "Layer.%s: weight %s doesn't fit %d input channels" fn
           (shape_str ws) in_c);
    let cols, n, oh, ow = conv_columns x ~kernel ~stride ~padding ~dilation in
    let out_c = ws.(1) in
    let out =
      if groups = 1 then affine cols w b
      else begin
        let rows = n * oh * ow and cg = in_c / groups and og = out_c / groups in
        let k = kh * kw * cg in
        let c =
          Var.reshape
            (Var.permute
               (Var.reshape cols [| rows; kh * kw; groups; cg |])
               [| 2; 0; 1; 3 |])
            [| groups; rows; k |]
        in
        let wg =
          Var.permute (Var.reshape w [| k; groups; og |]) [| 1; 0; 2 |]
        in
        let y = Var.permute (Var.matmul c wg) [| 1; 0; 2 |] in
        add_bias (Var.reshape y [| rows; out_c |]) b
      end
    in
    Var.reshape out [| n; oh; ow; out_c |]

  (** The adjoint of [im2col] as one node: its backward pass is [im2col]. *)
  let col2im cols ~input_shape ~kh ~kw ~stride ~padding =
    Var.custom ~name:"col2im" ~inputs:[ cols ]
      ~value:(T.col2im (Var.value cols) ~input_shape ~kh ~kw ~stride ~padding)
      ~backward:(fun g -> [ Some (T.im2col g ~kh ~kw ~stride ~padding) ])

  (** The upper triangle (excluding the diagonal) of a [tq] x [tk] matrix: the
      positions a causal model may not attend to. *)
  let causal_mask tq tk =
    T.init [| tq; tk |] (fun i -> if i.(1) > i.(0) + (tk - tq) then 1. else 0.)

  let split_heads x heads =
    let s = Var.shape x in
    let b = s.(0) and t = s.(1) and d = s.(2) in
    Var.permute (Var.reshape x [| b; t; heads; d / heads |]) [| 0; 2; 1; 3 |]

  let attend ?mask ~heads ~causal ~out_w ~out_b q k v =
    let s = Var.shape q in
    let b = s.(0) and tq = s.(1) and d = s.(2) in
    let tk = (Var.shape k).(1) in
    let q = split_heads q heads
    and k = split_heads k heads
    and v = split_heads v heads in
    let scores =
      Var.scale
        (1. /. Float.sqrt (float_of_int (d / heads)))
        (Var.matmul q (Var.swapaxes k (-1) (-2)))
    in
    let mask =
      match (mask, causal) with
      | None, false -> None
      | Some m, false -> Some m
      | None, true -> Some (causal_mask tq tk)
      | Some m, true ->
          Some (T.maximum (T.ne m (T.scalar 0.)) (causal_mask tq tk))
    in
    let scores =
      match mask with
      | Some m -> Var.masked_fill scores m Float.neg_infinity
      | None -> scores
    in
    let out = Var.matmul (Var.softmax scores) v in
    let out = Var.reshape (Var.permute out [| 0; 2; 1; 3 |]) [| b; tq; d |] in
    Var.add (Var.matmul out out_w) out_b

  let attention ?mask layer ~query ~key ~value =
    match layer with
    | Attention r ->
        let d = (Var.shape r.in_w).(0) in
        let proj x i =
          Var.add
            (Var.matmul x (Var.narrow r.in_w ~axis:1 ~start:(i * d) ~len:d))
            (Var.narrow r.in_b ~axis:0 ~start:(i * d) ~len:d)
        in
        attend ?mask ~heads:r.heads ~causal:r.causal ~out_w:r.out_w
          ~out_b:r.out_b (proj query 0) (proj key 1) (proj value 2)
    | _ -> invalid_arg "Layer.attention: not an Attention layer"

  (** Normalise [x] over [axes] with its own mean and (biased) variance. *)
  let normalise x axes eps =
    let mean = Var.mean_axes ~keepdims:true x axes in
    let centered = Var.sub x mean in
    let var = Var.mean_axes ~keepdims:true (Var.square centered) axes in
    Var.mul centered (Var.rsqrt (Var.add var (Var.scalar eps)))

  (* Recurrent layers ------------------------------------------------------ *)

  let gate_count = function Rnn_tanh | Rnn_relu -> 1 | Lstm -> 4 | Gru -> 3

  type cell = {
    kind : recurrent_kind;
    w_ih : var;
    w_hh : var;
    b_ih : var;
    b_hh : var;
    hidden : int;
  }

  let recurrent_parts = function
    | Recurrent { kind; w_ih; w_hh; b_ih; b_hh; hidden } ->
        { kind; w_ih; w_hh; b_ih; b_hh; hidden }
    | _ -> invalid_arg "Layer: not a Recurrent layer"

  let init_state layer ~batch =
    let r = recurrent_parts layer in
    let z () = Var.const (T.zeros [| batch; r.hidden |]) in
    { h = z (); c = (match r.kind with Lstm -> Some (z ()) | _ -> None) }

  (** One step, given [xw = x · w_ih + b_ih] for this step. *)
  let step_projected layer xw st =
    let r = recurrent_parts layer in
    let hw = Var.linear st.h r.w_hh r.b_hh in
    match r.kind with
    | Rnn_tanh -> { st with h = Var.tanh (Var.add xw hw) }
    | Rnn_relu -> { st with h = Var.relu (Var.add xw hw) }
    | Lstm -> (
        match Var.chunk (Var.add xw hw) ~axis:1 4 with
        | [ i; f; g; o ] ->
            let c =
              match st.c with
              | Some c -> c
              | None -> invalid_arg "Layer.step: an LSTM needs a cell state"
            in
            let c' =
              Var.add
                (Var.mul (Var.sigmoid f) c)
                (Var.mul (Var.sigmoid i) (Var.tanh g))
            in
            { h = Var.mul (Var.sigmoid o) (Var.tanh c'); c = Some c' }
        | _ -> assert false)
    | Gru -> (
        match (Var.chunk xw ~axis:1 3, Var.chunk hw ~axis:1 3) with
        | [ xr; xz; xn ], [ hr; hz; hn ] ->
            let r = Var.sigmoid (Var.add xr hr) in
            let z = Var.sigmoid (Var.add xz hz) in
            let n = Var.tanh (Var.add xn (Var.mul r hn)) in
            (* (1 - z) n + z h = n + z (h - n) *)
            { st with h = Var.add n (Var.mul z (Var.sub st.h n)) }
        | _ -> assert false)

  let step layer ~x st =
    let r = recurrent_parts layer in
    step_projected layer (Var.linear x r.w_ih r.b_ih) st

  (** The input projection of every step is one matmul; then the steps run in
      order. *)
  let run ?state layer x =
    let r = recurrent_parts layer in
    let s = Var.shape x in
    if Array.length s <> 3 then
      invalid_arg
        (Printf.sprintf "Layer.run: expected [batch; time; features], got %s"
           (shape_str s));
    let b = s.(0) and t = s.(1) in
    let g = gate_count r.kind * r.hidden in
    let st =
      match state with Some st -> st | None -> init_state layer ~batch:b
    in
    let xw = Var.add (Var.matmul x r.w_ih) r.b_ih in
    let st = ref st and outs = ref [] in
    for i = 0 to t - 1 do
      let xi = Var.reshape (Var.narrow xw ~axis:1 ~start:i ~len:1) [| b; g |] in
      st := step_projected layer xi !st;
      outs := !st.h :: !outs
    done;
    let out =
      if t = 0 then Var.const (T.zeros [| b; 0; r.hidden |])
      else Var.stack ~axis:1 (List.rev !outs)
    in
    (out, !st)

  let embed layer ?shape idx =
    match layer with
    | Embedding { w } ->
        let out = Var.index_select w ~axis:0 idx in
        let dim = (Var.shape w).(1) in
        let shape =
          match shape with Some s -> s | None -> [| Array.length idx |]
        in
        Var.reshape out (Array.append shape [| dim |])
    | _ -> invalid_arg "Layer.embed: not an Embedding layer"

  (** One pattern match over the layer type. Every case is built from Var ops,
      so no layer needs backward code of its own. *)
  let rec forward ?(train = false) ?rng layer x =
    match layer with
    | Linear r -> affine x r.w r.b
    | Conv2d r ->
        conv ~fn:"conv2d" x ~w:r.w ~b:r.b ~kernel:r.kernel ~stride:r.stride
          ~padding:r.padding ~dilation:r.dilation ~groups:r.groups
    | Conv1d r ->
        (* [n; l; c] as an image one pixel high *)
        let s = Var.shape x in
        if Array.length s <> 3 then
          invalid_arg
            (Printf.sprintf
               "Layer.conv1d: expected [n; length; channels], got %s"
               (shape_str s));
        let y =
          conv ~fn:"conv1d"
            (Var.reshape x [| s.(0); 1; s.(1); s.(2) |])
            ~w:r.w ~b:r.b ~kernel:(1, r.kernel) ~stride:(1, r.stride)
            ~padding:(0, r.padding) ~dilation:(1, r.dilation) ~groups:r.groups
        in
        let ys = Var.shape y in
        Var.reshape y [| ys.(0); ys.(2); ys.(3) |]
    | Conv_transpose2d r ->
        let s = Var.shape x in
        if Array.length s <> 4 then
          invalid_arg
            (Printf.sprintf
               "Layer.conv_transpose2d: expected NHWC input, got %s"
               (shape_str s));
        let n = s.(0) and h = s.(1) and w = s.(2) and c = s.(3) in
        let kh, kw = r.kernel in
        let out_c = (Var.shape r.w).(1) / (kh * kw) in
        let size d k =
          ((d - 1) * r.stride) - (2 * r.padding) + k + r.output_padding
        in
        let oh = size h kh and ow = size w kw in
        let cols = Var.matmul (Var.reshape x [| n * h * w; c |]) r.w in
        let y =
          col2im cols ~input_shape:[| n; oh; ow; out_c |] ~kh ~kw
            ~stride:r.stride ~padding:r.padding
        in
        add_bias y r.b
    | Max_pool2d r ->
        let x =
          if r.padding = 0 then x
          else
            let p = r.padding in
            Var.pad ~value:Float.neg_infinity x
              [| (0, 0); (p, p); (p, p); (0, 0) |]
        in
        Var.max_pool2d x ~size:r.size ~stride:r.stride
    | Avg_pool2d r ->
        let s = Var.shape x in
        let cols, n, oh, ow =
          conv_columns x ~kernel:(r.size, r.size) ~stride:(r.stride, r.stride)
            ~padding:(r.padding, r.padding) ~dilation:(1, 1)
        in
        let c = s.(3) in
        let m =
          Var.mean_axis
            (Var.reshape cols [| n * oh * ow; r.size * r.size; c |])
            1
        in
        Var.reshape m [| n; oh; ow; c |]
    | Global_avg_pool ->
        let nd = Array.length (Var.shape x) in
        if nd <= 2 then x
        else Var.mean_axes x (List.init (nd - 2) (fun i -> i + 1))
    | Relu -> Var.relu x
    | Sigmoid -> Var.sigmoid x
    | Tanh -> Var.tanh x
    | Leaky_relu slope -> Var.leaky_relu ~slope x
    | Elu alpha -> Var.elu ~alpha x
    | Gelu { approximate } -> Var.gelu ~approximate x
    | Silu -> Var.silu x
    | Softplus { beta; threshold } -> Var.softplus ~beta ~threshold x
    | Softmax -> Var.softmax x
    | Log_softmax -> Var.log_softmax x
    | Identity -> x
    | Flatten ->
        (* [n; d1; d2; ...] -> [n; d1*d2*...]: keep the batch axis, merge the rest *)
        let s = Var.shape x in
        let n = s.(0) in
        let numel = Array.fold_left ( * ) 1 s in
        Var.reshape x [| n; (if n = 0 then 0 else numel / n) |]
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
    | Batch_norm r ->
        let s = Var.shape x in
        let nd = Array.length s in
        let axes = List.init (nd - 1) Fun.id in
        if train then begin
          let mean = Var.mean_axes x axes in
          let centered = Var.sub x mean in
          let var = Var.mean_axes (Var.square centered) axes in
          (* The running statistics: the unbiased variance, as PyTorch. *)
          let count = T.numel (Var.value x) / max 1 s.(nd - 1) in
          let unbias = float_of_int count /. float_of_int (max 1 (count - 1)) in
          let update buf v =
            let buf = Var.value buf in
            T.blit_
              ~src:
                (T.add (T.scale (1. -. r.momentum) buf) (T.scale r.momentum v))
              ~dst:buf
          in
          update r.running_mean (Var.value mean);
          update r.running_var (T.scale unbias (Var.value var));
          let nb = Var.value r.num_batches_tracked in
          T.fill_ nb (T.item nb +. 1.);
          let xhat =
            Var.mul centered (Var.rsqrt (Var.add var (Var.scalar r.eps)))
          in
          Var.add (Var.mul xhat r.w) r.b
        end
        else begin
          (* y = x * scale + shift, with scale = w / sqrt (var + eps) *)
          let inv =
            Var.const (T.rsqrt (T.add_scalar r.eps (Var.value r.running_var)))
          in
          let scale = Var.mul r.w inv in
          let shift = Var.sub r.b (Var.mul r.running_mean scale) in
          Var.add (Var.mul x scale) shift
        end
    | Layer_norm r ->
        let nd = Array.length (Var.shape x) and k = Array.length r.shape in
        let axes = List.init k (fun i -> nd - k + i) in
        Var.add (Var.mul (normalise x axes r.eps) r.w) r.b
    | Group_norm r ->
        let s = Var.shape x in
        let nd = Array.length s in
        let n = s.(0) and c = s.(nd - 1) in
        let spatial = T.numel (Var.value x) / max 1 (n * c) in
        let g = Var.reshape x [| n; spatial; r.groups; c / r.groups |] in
        let y = Var.reshape (normalise g [ 1; 3 ] r.eps) s in
        Var.add (Var.mul y r.w) r.b
    | Embedding _ ->
        let v = Var.value x in
        let idx = Array.map int_of_float (T.to_array v) in
        embed layer ~shape:(T.shape v) idx
    | Recurrent _ -> fst (run layer x)
    | Attention r -> (
        let d = (Var.shape r.in_w).(0) in
        let qkv = Var.add (Var.matmul x r.in_w) r.in_b in
        match Var.split qkv ~axis:(-1) [ d; d; d ] with
        | [ q; k; v ] ->
            attend ~heads:r.heads ~causal:r.causal ~out_w:r.out_w ~out_b:r.out_b
              q k v
        | _ -> assert false)
    | Residual f -> Var.add x (forward ~train ?rng f x)
    | Sequential layers ->
        (* Feed each layer's output into the next, threading train/rng down. *)
        List.fold_left (fun acc layer -> forward ~train ?rng layer acc) x layers
    | Custom r -> r.forward ~train ?rng x

  (* ------------------------------------------------------------------------ *)
  (* Parameters and state                                                     *)
  (* ------------------------------------------------------------------------ *)

  let join prefix name =
    if prefix = "" then name
    else if name = "" then prefix
    else prefix ^ "." ^ name

  let opt name = function Some v -> [ (name, v) ] | None -> []

  (** A layer's own parameters and buffers, and its named children. *)
  let parts = function
    | Linear { w; b }
    | Conv2d { w; b; _ }
    | Conv1d { w; b; _ }
    | Conv_transpose2d { w; b; _ } ->
        (("w", w) :: opt "b" b, [], [])
    | Batch_norm r ->
        ( [ ("w", r.w); ("b", r.b) ],
          [
            ("running_mean", r.running_mean);
            ("running_var", r.running_var);
            ("num_batches_tracked", r.num_batches_tracked);
          ],
          [] )
    | Layer_norm { w; b; _ } | Group_norm { w; b; _ } ->
        ([ ("w", w); ("b", b) ], [], [])
    | Embedding { w } -> ([ ("w", w) ], [], [])
    | Recurrent r ->
        ( [
            ("w_ih", r.w_ih);
            ("w_hh", r.w_hh);
            ("b_ih", r.b_ih);
            ("b_hh", r.b_hh);
          ],
          [],
          [] )
    | Attention r ->
        ( [
            ("in_w", r.in_w);
            ("in_b", r.in_b);
            ("out_w", r.out_w);
            ("out_b", r.out_b);
          ],
          [],
          [] )
    | Residual f -> ([], [], [ ("", f) ])
    | Sequential layers ->
        ([], [], List.mapi (fun i l -> (string_of_int i, l)) layers)
    | Custom r -> (r.params, r.buffers, r.children)
    | Max_pool2d _ | Avg_pool2d _ | Global_avg_pool | Relu | Sigmoid | Tanh
    | Leaky_relu _ | Elu _ | Gelu _ | Silu | Softplus _ | Softmax | Log_softmax
    | Identity | Flatten | Dropout _ ->
        ([], [], [])

  let rec named ~buffers prefix layer =
    let own, bufs, children = parts layer in
    List.map (fun (n, v) -> (join prefix n, v)) (if buffers then bufs else own)
    @ List.concat_map (fun (n, l) -> named ~buffers (join prefix n) l) children

  let named_params layer = named ~buffers:false "" layer
  let named_buffers layer = named ~buffers:true "" layer
  let params layer = List.map snd (named_params layer)

  let state_dict layer =
    List.map
      (fun (n, v) -> (n, Var.value v))
      (named_params layer @ named_buffers layer)

  let load_state_dict ?(strict = true) layer entries =
    let own = state_dict layer in
    let given = Hashtbl.create 16 in
    List.iter (fun (n, t) -> Hashtbl.replace given n t) entries;
    let mine = Hashtbl.create 16 in
    List.iter (fun (n, _) -> Hashtbl.replace mine n ()) own;
    let missing =
      List.filter (fun (n, _) -> not (Hashtbl.mem given n)) own |> List.map fst
    in
    let unexpected =
      List.filter (fun (n, _) -> not (Hashtbl.mem mine n)) entries
      |> List.map fst
    in
    let mismatched =
      List.filter_map
        (fun (n, t) ->
          match Hashtbl.find_opt given n with
          | Some src when T.shape src <> T.shape t ->
              Some (n, T.shape t, T.shape src)
          | _ -> None)
        own
    in
    let report = { missing; unexpected; mismatched } in
    if strict && (missing <> [] || unexpected <> [] || mismatched <> []) then begin
      let lines =
        List.map (fun n -> "missing " ^ n) missing
        @ List.map (fun n -> "unexpected " ^ n) unexpected
        @ List.map
            (fun (n, a, b) ->
              Printf.sprintf "%s has shape %s in the model and %s in the state"
                n (shape_str a) (shape_str b))
            mismatched
      in
      failwith ("Layer.load_state_dict: " ^ String.concat "; " lines)
    end;
    List.iter
      (fun (n, t) ->
        match Hashtbl.find_opt given n with
        | Some src when T.shape src = T.shape t -> T.blit_ ~src ~dst:t
        | _ -> ())
      own;
    report

  (* ------------------------------------------------------------------------ *)
  (* Utilities                                                                *)
  (* ------------------------------------------------------------------------ *)

  (** Sum of the element counts of every parameter. *)
  let num_params layer =
    List.fold_left
      (fun acc p -> acc + Array.fold_left ( * ) 1 (Var.shape p))
      0 (params layer)

  let pair_str (a, b) =
    if a = b then string_of_int a else Printf.sprintf "(%d, %d)" a b

  (** The one-line description of a layer that isn't a container. *)
  let describe = function
    | Linear { w; b } ->
        let s = Var.shape w in
        Printf.sprintf "Linear(%d -> %d%s)" s.(0) s.(1)
          (if b = None then ", bias=false" else "")
    | Conv2d { w; b; kernel; stride; padding; dilation; groups } ->
        let s = Var.shape w and kh, kw = kernel in
        Printf.sprintf
          "Conv2d(%d -> %d, kernel=%s, stride=%s, padding=%s%s%s%s)"
          (s.(0) * groups / (kh * kw))
          s.(1) (pair_str kernel) (pair_str stride) (pair_str padding)
          (if dilation = (1, 1) then "" else ", dilation=" ^ pair_str dilation)
          (if groups = 1 then "" else Printf.sprintf ", groups=%d" groups)
          (if b = None then ", bias=false" else "")
    | Conv1d { w; b; kernel; stride; padding; dilation; groups } ->
        let s = Var.shape w in
        Printf.sprintf
          "Conv1d(%d -> %d, kernel=%d, stride=%d, padding=%d%s%s%s)"
          (s.(0) * groups / kernel)
          s.(1) kernel stride padding
          (if dilation = 1 then "" else Printf.sprintf ", dilation=%d" dilation)
          (if groups = 1 then "" else Printf.sprintf ", groups=%d" groups)
          (if b = None then ", bias=false" else "")
    | Conv_transpose2d { w; kernel; stride; padding; output_padding; _ } ->
        let s = Var.shape w and kh, kw = kernel in
        Printf.sprintf
          "ConvTranspose2d(%d -> %d, kernel=%s, stride=%d, padding=%d%s)" s.(0)
          (s.(1) / (kh * kw))
          (pair_str kernel) stride padding
          (if output_padding = 0 then ""
           else Printf.sprintf ", output_padding=%d" output_padding)
    | Max_pool2d { size; stride; padding } ->
        Printf.sprintf "MaxPool2d(size=%d, stride=%d%s)" size stride
          (if padding = 0 then "" else Printf.sprintf ", padding=%d" padding)
    | Avg_pool2d { size; stride; padding } ->
        Printf.sprintf "AvgPool2d(size=%d, stride=%d%s)" size stride
          (if padding = 0 then "" else Printf.sprintf ", padding=%d" padding)
    | Global_avg_pool -> "GlobalAvgPool"
    | Relu -> "ReLU"
    | Sigmoid -> "Sigmoid"
    | Tanh -> "Tanh"
    | Leaky_relu s -> Printf.sprintf "LeakyReLU(slope=%g)" s
    | Elu a -> Printf.sprintf "ELU(alpha=%g)" a
    | Gelu { approximate } -> if approximate then "GELU(tanh)" else "GELU"
    | Silu -> "SiLU"
    | Softplus { beta; threshold } ->
        Printf.sprintf "Softplus(beta=%g, threshold=%g)" beta threshold
    | Softmax -> "Softmax"
    | Log_softmax -> "LogSoftmax"
    | Identity -> "Identity"
    | Flatten -> "Flatten"
    | Dropout p -> Printf.sprintf "Dropout(p=%g)" p
    | Batch_norm { w; momentum; eps; _ } ->
        Printf.sprintf "BatchNorm(%d, momentum=%g, eps=%g)"
          (Var.shape w).(0)
          momentum eps
    | Layer_norm { shape; eps; _ } ->
        Printf.sprintf "LayerNorm(%s, eps=%g)" (shape_str shape) eps
    | Group_norm { w; groups; eps; _ } ->
        Printf.sprintf "GroupNorm(%d groups, %d channels, eps=%g)" groups
          (Var.shape w).(0)
          eps
    | Embedding { w } ->
        let s = Var.shape w in
        Printf.sprintf "Embedding(%d, %d)" s.(0) s.(1)
    | Recurrent { kind; w_ih; hidden; _ } ->
        Printf.sprintf "%s(%d -> %d)"
          (match kind with
          | Rnn_tanh -> "RNN"
          | Rnn_relu -> "RNN(relu)"
          | Lstm -> "LSTM"
          | Gru -> "GRU")
          (Var.shape w_ih).(0)
          hidden
    | Attention { in_w; heads; causal; _ } ->
        Printf.sprintf "MultiHeadAttention(dim=%d, heads=%d%s)"
          (Var.shape in_w).(0)
          heads
          (if causal then ", causal" else "")
    | Residual _ -> "Residual"
    | Sequential _ -> "Sequential"
    | Custom { name; _ } -> name

  let indent s =
    String.split_on_char '\n' s
    |> List.map (fun l -> "  " ^ l)
    |> String.concat "\n"

  (** Containers list their children, indented by two spaces per level. *)
  let rec to_string layer =
    let _, _, children = parts layer in
    match layer with
    | (Sequential _ | Residual _ | Custom _) when children <> [] ->
        let inner =
          List.map
            (fun (n, l) ->
              let s = to_string l in
              indent (match layer with Custom _ -> n ^ ": " ^ s | _ -> s))
            children
        in
        describe layer ^ "(\n" ^ String.concat "\n" inner ^ "\n)"
    | _ -> describe layer

  let summary layer ~input_shape =
    let rows = ref [] in
    let count l = num_params l in
    (* Containers run their children one by one, so each child gets a row;
       a custom layer's forward function is opaque, so its children are
       listed without shapes. *)
    let rec go depth layer x =
      let row = (depth, describe layer, ref "", count layer) in
      rows := row :: !rows;
      let set_shape y =
        let _, _, s, _ = row in
        s := shape_str (Var.shape y)
      in
      let y =
        match layer with
        | Sequential layers ->
            List.fold_left (fun acc l -> go (depth + 1) l acc) x layers
        | Residual f -> Var.add x (go (depth + 1) f x)
        | Custom r ->
            let y = forward layer x in
            List.iter (fun (_, c) -> list (depth + 1) c) r.children;
            y
        | _ -> forward layer x
      in
      set_shape y;
      y
    and list depth layer =
      rows := (depth, describe layer, ref "", count layer) :: !rows;
      let _, _, children = parts layer in
      List.iter (fun (_, c) -> list (depth + 1) c) children
    in
    Var.no_grad (fun () ->
        ignore (go 0 layer (Var.const (T.zeros input_shape))));
    let rows = List.rev !rows in
    let name (d, s, _, _) = String.make (2 * d) ' ' ^ s in
    let w1 = List.fold_left (fun m r -> max m (String.length (name r))) 5 rows
    and w2 =
      List.fold_left (fun m (_, _, s, _) -> max m (String.length !s)) 12 rows
    in
    let line a b c = Printf.sprintf "%-*s  %-*s  %10s" w1 a w2 b c in
    let rule = String.make (w1 + w2 + 14) '-' in
    String.concat "\n"
      ([ line "Layer" "Output shape" "Params"; rule ]
      @ List.map
          (fun ((_, _, s, p) as r) -> line (name r) !s (string_of_int p))
          rows
      @ [
          rule;
          Printf.sprintf "Input %s, %d trainable parameters"
            (shape_str input_shape) (num_params layer);
        ])
end

(** Weight initialisation schemes.

    Good initialisation keeps activations and gradients at a sensible scale as
    they pass through many layers. Bad initialisation (e.g. all zeros, or
    [randn] with std 1 for wide layers) is the most common reason a from-scratch
    network refuses to learn. *)

(** A named scheme, as the layer constructors take it ([?init]). The fans are
    filled in by the constructor. *)
type scheme =
  | He_normal  (** N(0, 2 / fan_in): FlockML's default. *)
  | Kaiming_uniform
      (** PyTorch's default for [Linear] and [Conv]: [kaiming_uniform] with
          [a = sqrt 5], which is U(-1 / sqrt fan_in, 1 / sqrt fan_in). *)
  | Xavier_uniform  (** U(-a, a), a = sqrt (6 / (fan_in + fan_out)). *)
  | Xavier_normal  (** N(0, 2 / (fan_in + fan_out)). *)
  | Orthogonal  (** A random (semi-)orthogonal matrix. *)
  | Normal of float  (** N(0, std²). *)
  | Uniform of float  (** U(-bound, bound). *)
  | Constant of float  (** Every element the same. *)

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  type nonrec scheme = scheme =
    | He_normal
    | Kaiming_uniform
    | Xavier_uniform
    | Xavier_normal
    | Orthogonal
    | Normal of float
    | Uniform of float
    | Constant of float

  val xavier_uniform :
    ?rng:Random.State.t -> fan_in:int -> fan_out:int -> int array -> tensor
  (** Glorot/Xavier: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)). Best for
      tanh / sigmoid layers. *)

  val xavier_normal :
    ?rng:Random.State.t -> fan_in:int -> fan_out:int -> int array -> tensor
  (** Glorot/Xavier: N(0, 2 / (fan_in + fan_out)). *)

  val he_normal : ?rng:Random.State.t -> fan_in:int -> int array -> tensor
  (** He/Kaiming: N(0, 2 / fan_in). Best for ReLU layers. *)

  val kaiming_uniform :
    ?rng:Random.State.t -> ?a:float -> fan_in:int -> int array -> tensor
  (** As [torch.nn.init.kaiming_uniform_]: U(-b, b) with
      [b = sqrt (2 / (1 + a²)) * sqrt (3 / fan_in)]. [a] (the negative slope of
      the following leaky ReLU) defaults to 0, which is He-uniform. PyTorch's
      [Linear] uses [a = sqrt 5]. *)

  val orthogonal : ?rng:Random.State.t -> ?gain:float -> int array -> tensor
  (** A (semi-)orthogonal matrix times [gain] (default 1), as
      [torch.nn.init.orthogonal_]: the shape is viewed as a matrix whose columns
      are the last axis, and that matrix has orthonormal rows or columns,
      whichever there are fewer of. Raises [Invalid_argument] for fewer than 2
      dimensions. *)

  val normal : ?rng:Random.State.t -> std:float -> int array -> tensor
  (** N(0, std²). *)

  val uniform : ?rng:Random.State.t -> bound:float -> int array -> tensor
  (** U(-bound, bound). *)

  val constant : float -> int array -> tensor
  (** Every element [x]. *)

  val make :
    ?rng:Random.State.t ->
    scheme ->
    fan_in:int ->
    fan_out:int ->
    int array ->
    tensor
  (** The tensor that [scheme] gives for a weight of this shape and fans. *)
end

(** The initialisers for backend [T]. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  type nonrec scheme = scheme =
    | He_normal
    | Kaiming_uniform
    | Xavier_uniform
    | Xavier_normal
    | Orthogonal
    | Normal of float
    | Uniform of float
    | Constant of float

  (** The bound [a] makes the variance [a²/3 = 2 / (fan_in + fan_out)], which
      balances the forward and backward signal scales. *)
  let xavier_uniform ?rng ~fan_in ~fan_out shape =
    let a = sqrt (6. /. (float_of_int fan_in +. float_of_int fan_out)) in
    T.rand_uniform ?rng ~lo:(-.a) ~hi:a shape

  let xavier_normal ?rng ~fan_in ~fan_out shape =
    let sd = sqrt (2. /. (float_of_int fan_in +. float_of_int fan_out)) in
    T.randn ?rng ~std:sd shape

  (** Standard deviation [sqrt (2 / fan_in)]. The 2 is there because ReLU zeroes
      half its inputs, halving the variance; doubling it restores the scale. *)
  let he_normal ?rng ~fan_in shape =
    let sd = sqrt (2. /. float_of_int fan_in) in
    T.randn ?rng ~std:sd shape

  let kaiming_uniform ?rng ?(a = 0.) ~fan_in shape =
    let gain = sqrt (2. /. (1. +. (a *. a))) in
    let b = gain *. sqrt (3. /. float_of_int fan_in) in
    T.rand_uniform ?rng ~lo:(-.b) ~hi:b shape

  let normal ?rng ~std shape = T.randn ?rng ~std shape

  let uniform ?rng ~bound shape =
    T.rand_uniform ?rng ~lo:(-.bound) ~hi:bound shape

  let constant x shape = T.full shape x

  (** Modified Gram–Schmidt, run twice for accuracy, on the columns of an [m] x
      [n] row-major matrix with [m >= n]. Each column is made positive along its
      own direction, as the sign correction of a QR decomposition with a
      positive diagonal does. *)
  let orthonormalise_columns a m n =
    for _ = 1 to 2 do
      for j = 0 to n - 1 do
        for k = 0 to j - 1 do
          let d = ref 0. in
          for i = 0 to m - 1 do
            d := !d +. (a.((i * n) + j) *. a.((i * n) + k))
          done;
          for i = 0 to m - 1 do
            a.((i * n) + j) <- a.((i * n) + j) -. (!d *. a.((i * n) + k))
          done
        done;
        let s = ref 0. in
        for i = 0 to m - 1 do
          s := !s +. (a.((i * n) + j) *. a.((i * n) + j))
        done;
        let s = Float.sqrt !s in
        for i = 0 to m - 1 do
          a.((i * n) + j) <- a.((i * n) + j) /. s
        done
      done
    done

  let orthogonal ?rng ?(gain = 1.) shape =
    let nd = Array.length shape in
    if nd < 2 then invalid_arg "Init.orthogonal: needs at least 2 dimensions";
    let cols = shape.(nd - 1) in
    let rows = Layout.numel_of_shape shape / max cols 1 in
    (* Orthonormalise the columns of the taller of the matrix and its
       transpose. *)
    let m = max rows cols and n = min rows cols in
    let a = T.to_array (T.randn ?rng [| m; n |]) in
    if n > 0 then orthonormalise_columns a m n;
    let out =
      if rows >= cols then a
      else
        (* a is [cols; rows]: transpose it *)
        Array.init (rows * cols) (fun k ->
            let i = k / cols and j = k mod cols in
            a.((j * rows) + i))
    in
    T.of_array shape (Array.map (fun x -> gain *. x) out)

  let make ?rng scheme ~fan_in ~fan_out shape =
    match scheme with
    | He_normal -> he_normal ?rng ~fan_in shape
    | Kaiming_uniform -> kaiming_uniform ?rng ~a:(sqrt 5.) ~fan_in shape
    | Xavier_uniform -> xavier_uniform ?rng ~fan_in ~fan_out shape
    | Xavier_normal -> xavier_normal ?rng ~fan_in ~fan_out shape
    | Orthogonal -> orthogonal ?rng shape
    | Normal std -> normal ?rng ~std shape
    | Uniform bound -> uniform ?rng ~bound shape
    | Constant x -> constant x shape
end

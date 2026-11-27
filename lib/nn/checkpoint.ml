(** Saving and loading a model, and optionally its optimizer, as one safetensors
    file.

    The model's entries are {!Layer.S.state_dict}'s names ("0.w",
    "3.running_mean"...). The optimizer's are {!Optim.S.state_dict}'s, under the
    prefix ["optim."]. Since the file is plain safetensors, PyTorch can read it;
    FlockML's weights are the transposes of PyTorch's (a linear layer's [w] is
    [[| in; out |]]), and images are NHWC. *)

module type S = sig
  type layer
  type optim

  val save :
    ?dtype:Safetensors.dtype ->
    ?metadata:(string * string) list ->
    ?optim:optim ->
    string ->
    layer ->
    unit
  (** [save path model] writes every parameter and buffer of [model], and with
      [optim] the optimizer's state, as float64 unless [~dtype:`F32]. Float64
      round-trips every backend's values exactly. *)

  val load :
    ?strict:bool -> ?optim:optim -> string -> layer -> Layer.load_report
  (** [load path model] copies the file's values into [model] (and into [optim],
      if given), converting from the file's dtype. The model's part is
      [Layer.load_state_dict]: with [strict] (the default) every name and shape
      must match, or it raises [Failure] and changes nothing. Raises [Failure]
      if [optim] is given and the file has no matching optimizer state. *)
end

let optim_prefix = "optim."

(** The module for one backend's layers and optimizers. *)
module Make
    (T : Tensor_intf.S)
    (Layer : Layer.S with type tensor = T.t)
    (Optim : Optim.S with type tensor = T.t)
    (St : Safetensors.S with type tensor = T.t) :
  S with type layer = Layer.t and type optim = Optim.t = struct
  type layer = Layer.t
  type optim = Optim.t

  let save ?dtype ?metadata ?optim path model =
    let opt =
      match optim with
      | Some o ->
          List.map (fun (n, t) -> (optim_prefix ^ n, t)) (Optim.state_dict o)
      | None -> []
    in
    St.save ?dtype ?metadata path (Layer.state_dict model @ opt)

  let load ?(strict = true) ?optim path model =
    let entries = St.load path in
    let lp = String.length optim_prefix in
    let is_opt (n, _) =
      String.length n > lp && String.sub n 0 lp = optim_prefix
    in
    let opt, own = List.partition is_opt entries in
    (match optim with
    | Some o ->
        if opt = [] then
          failwith
            (Printf.sprintf "Checkpoint.load: %s has no optimizer state" path);
        (* Nothing changes unless both parts load: a strict model check that
           would fail raises here, and the optimizer checks its own entries
           before changing anything. *)
        let sd = Layer.state_dict model in
        let fits =
          List.length sd = List.length own
          && List.for_all
               (fun (n, t) ->
                 match List.assoc_opt n own with
                 | Some s -> T.shape s = T.shape t
                 | None -> false)
               sd
        in
        if strict && not fits then
          ignore (Layer.load_state_dict ~strict model own : Layer.load_report);
        Optim.load_state_dict o
          (List.map
             (fun (n, t) -> (String.sub n lp (String.length n - lp), t))
             opt)
    | None -> ());
    Layer.load_state_dict ~strict model own
end

(** Data augmentation and preprocessing, applied to whole batches.

    A transform maps a batch of inputs to a new batch, drawing any randomness
    from the generator it is given, so a run is reproducible. Transforms compose
    with {!S.compose}, and {!S.on_inputs} applies one to the input part of an
    [(inputs, targets)] batch, as [Dataset.loader]'s [?transform] takes it.

    Image transforms work on NHWC batches [[| n; h; w; c |]]. They make new
    tensors (only {!S.normalize_} works in place): with the loader's prefetching
    they run on the background domain, off the training step. *)

module type S = sig
  type tensor

  type t = Random.State.t -> tensor -> tensor
  (** A transform of a batch of inputs. *)

  val compose : t list -> t
  (** Apply the transforms in order. *)

  val on_inputs : t -> Random.State.t -> tensor * 'y -> tensor * 'y
  (** Apply a transform to the inputs of an [(inputs, targets)] batch. *)

  val normalize : mean:float array -> std:float array -> t
  (** [(x - mean) / std] per channel (the last axis). Raises [Invalid_argument]
      if the arrays don't have one entry per channel. *)

  val normalize_ : mean:float array -> std:float array -> tensor -> unit
  (** {!normalize}, in place. *)

  val random_crop : ?padding:int -> int * int -> t
  (** [random_crop ~padding (h, w)] pads each image with [padding] zeros on
      every side (default 0), then cuts out an [h] x [w] window at a uniformly
      random position, independently for each image: the standard CIFAR
      augmentation is [random_crop ~padding:4 (32, 32)]. Raises
      [Invalid_argument] if the window doesn't fit. *)

  val random_hflip : ?p:float -> unit -> t
  (** Mirror each image left to right with probability [p] (default 0.5). *)

  val random_vflip : ?p:float -> unit -> t
  (** Mirror each image top to bottom with probability [p] (default 0.5). *)
end

(** The transforms for backend [T]. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t
  type t = Random.State.t -> tensor -> tensor

  let compose ts rng x = List.fold_left (fun x f -> f rng x) x ts
  let on_inputs f rng (x, y) = (f rng x, y)

  (** The mean and [1 / std] as tensors that broadcast over the last axis. *)
  let channel_stats ~fn ~mean ~std x =
    let c = (T.shape x).(T.ndim x - 1) in
    if Array.length mean <> c || Array.length std <> c then
      invalid_arg
        (Printf.sprintf "Transform.%s: %d means and %d stds for %d channels" fn
           (Array.length mean) (Array.length std) c);
    ( T.of_array [| c |] mean,
      T.of_array [| c |] (Array.map (fun s -> 1. /. s) std) )

  let normalize ~mean ~std _rng x =
    let m, inv = channel_stats ~fn:"normalize" ~mean ~std x in
    T.mul (T.sub x m) inv

  let normalize_ ~mean ~std x =
    let m, inv = channel_stats ~fn:"normalize_" ~mean ~std x in
    T.blit_ ~src:(T.mul (T.sub x m) inv) ~dst:x

  let image_shape ~fn x =
    match T.shape x with
    | [| n; h; w; c |] -> (n, h, w, c)
    | s ->
        invalid_arg
          (Printf.sprintf "Transform.%s: expected NHWC images, got %s" fn
             (Layout.shape_to_string s))

  let random_crop ?(padding = 0) (oh, ow) rng x =
    let n, h, w, _ = image_shape ~fn:"random_crop" x in
    let hp = h + (2 * padding) and wp = w + (2 * padding) in
    if oh > hp || ow > wp || oh < 1 || ow < 1 then
      invalid_arg
        (Printf.sprintf "Transform.random_crop: %dx%d doesn't fit %dx%d" oh ow
           hp wp);
    let xp =
      if padding = 0 then x
      else T.pad x [| (0, 0); (padding, padding); (padding, padding); (0, 0) |]
    in
    if n = 0 then T.slice xp [ All; Range (0, oh); Range (0, ow); All ]
    else
      T.stack ~axis:0
        (List.init n (fun i ->
             let dy = Random.State.int rng (hp - oh + 1) in
             let dx = Random.State.int rng (wp - ow + 1) in
             T.slice xp
               [ Index i; Range (dy, dy + oh); Range (dx, dx + ow); All ]))

  (** One coin per image; then one flip of the whole batch and a select. *)
  let random_flip ~fn ~axis ?(p = 0.5) () rng x =
    let n, _, _, _ = image_shape ~fn x in
    let coins =
      Array.init n (fun _ -> if Random.State.float rng 1. < p then 1. else 0.)
    in
    if Array.for_all (fun c -> c = 0.) coins then x
    else T.where (T.of_array [| n; 1; 1; 1 |] coins) (T.flip x [ axis ]) x

  let random_hflip ?p () = random_flip ~fn:"random_hflip" ~axis:2 ?p ()
  let random_vflip ?p () = random_flip ~fn:"random_vflip" ~axis:1 ?p ()
end

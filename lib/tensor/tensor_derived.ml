(** The part of {!Tensor_intf.S} that is written once for every backend.

    These functions are compositions of a few primitives: every structural op
    (slicing, permuting, flipping, padding, concatenating) is one call to the
    backend's strided copy, {!BASE.copy_walk_}, and the rest are built from
    ordinary tensor ops. A backend implements {!BASE} and includes
    [Tensor_derived.Make (...)] at the end of its implementation, so all the
    backends compute these the same way, and agree to the bit wherever their
    primitives do.

    Documentation of each function is in {!Tensor_intf.S}. *)

open Tensor_intf

(** What a backend provides to build the derived functions on. *)
module type BASE = sig
  type t

  val shape : t -> int array
  val numel : t -> int
  val ndim : t -> int
  val full : int array -> float -> t
  val scalar : float -> t
  val of_array : int array -> float array -> t
  val copy : t -> t
  val reshape : t -> int array -> t
  val broadcast_to : t -> int array -> t
  val sum_to_shape : t -> int array -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val scale : float -> t -> t
  val pow : t -> float -> t
  val exp : t -> t
  val log : t -> t
  val sqrt : t -> t
  val abs : t -> t
  val sum : t -> t
  val sum_axis : ?keepdims:bool -> t -> int -> t
  val max_axis : ?keepdims:bool -> t -> int -> t
  val max : t -> t
  val minimum : t -> t -> t
  val maximum : t -> t -> t
  val where : t -> t -> t -> t

  val mm : t -> t -> t
  (** The 2-D matrix product. *)

  val bmm : ?trans_a:bool -> ?trans_b:bool -> t -> t -> t

  val uninit : int array -> t
  (** A tensor whose contents are about to be overwritten. *)

  val copy_walk_ :
    src:t ->
    src_offset:int ->
    dst:t ->
    dst_offset:int ->
    int array ->
    int array ->
    int array ->
    unit
  (** [copy_walk_ ~src ~src_offset ~dst ~dst_offset dims ss ds] visits every
      multi-index [i] of [dims] and copies element
      [src_offset + sum i.(a) * ss.(a)] of [src]'s storage to element
      [dst_offset + sum i.(a) * ds.(a)] of [dst]'s. Strides may be negative or
      zero (for the source), but each destination element must be written at
      most once, so the copy can be split across threads. The caller has checked
      every index. *)
end

module Make (B : BASE) = struct
  open B

  let shape_to_string = Layout.shape_to_string

  let fail fn fmt =
    Printf.ksprintf (fun s -> invalid_arg ("Tensor." ^ fn ^ ": " ^ s)) fmt

  (** A possibly negative axis of a tensor with [nd] axes, checked. *)
  let axis_of ~fn nd axis =
    let a = if axis < 0 then axis + nd else axis in
    if a < 0 || a >= nd then
      fail fn "axis %d out of range for %d dimensions" axis nd;
    a

  (** The shape without the axes for which [drop] is true. *)
  let remove_axes s drop =
    Array.of_list (List.filteri (fun i _ -> not (drop i)) (Array.to_list s))

  (* -------------------------------------------------------------------- *)
  (* Construction                                                          *)
  (* -------------------------------------------------------------------- *)

  let zeros_like t = full (shape t) 0.0
  let ones_like t = full (shape t) 1.0
  let full_like t x = full (shape t) x

  (** The count is [ceil ((stop - start) / step)], as in NumPy, and element [i]
      is [start + i * step]. *)
  let arange ?(start = 0.0) ?(step = 1.0) stop =
    if step = 0.0 then fail "arange" "step must not be 0";
    let n = Float.to_int (Float.ceil ((stop -. start) /. step)) in
    let n = Int.max 0 n in
    of_array [| n |] (Array.init n (fun i -> start +. (float_of_int i *. step)))

  (** As NumPy: [start + i * step], with the last element set to [stop] exactly.
  *)
  let linspace start stop n =
    if n < 0 then fail "linspace" "negative count %d" n;
    let step = if n > 1 then (stop -. start) /. float_of_int (n - 1) else 0.0 in
    let a = Array.init n (fun i -> start +. (float_of_int i *. step)) in
    if n > 1 then a.(n - 1) <- stop;
    of_array [| n |] a

  let eye n =
    if n < 0 then fail "eye" "negative size %d" n;
    of_array [| n; n |]
      (Array.init (n * n) (fun i -> if i / n = i mod n then 1.0 else 0.0))

  (** Walks the multi-index like an odometer, so [f] sees row-major order. *)
  let init s f =
    let n = Layout.numel_of_shape s and nd = Array.length s in
    let idx = Array.make nd 0 in
    let a = Array.create_float n in
    for i = 0 to n - 1 do
      a.(i) <- f idx;
      let ax = ref (nd - 1) in
      while !ax >= 0 do
        idx.(!ax) <- idx.(!ax) + 1;
        if idx.(!ax) < s.(!ax) then ax := -1
        else begin
          idx.(!ax) <- 0;
          decr ax
        end
      done
    done;
    of_array s a

  let bernoulli ?rng ~p s =
    if not (p >= 0.0 && p <= 1.0) then fail "bernoulli" "p = %g not in [0, 1]" p;
    let rng = Rng.get rng in
    of_array s
      (Array.init (Layout.numel_of_shape s) (fun _ ->
           if Random.State.float rng 1.0 < p then 1.0 else 0.0))

  let randint ?rng ~lo ~hi s =
    if lo >= hi then fail "randint" "empty range [%d, %d)" lo hi;
    let rng = Rng.get rng in
    of_array s
      (Array.init (Layout.numel_of_shape s) (fun _ ->
           float_of_int (lo + Random.State.full_int rng (hi - lo))))

  (* -------------------------------------------------------------------- *)
  (* Structural copies                                                     *)
  (* -------------------------------------------------------------------- *)

  (** [copy_walk_] when there is anything to copy. *)
  let walk ~src ~src_offset ~dst ~dst_offset dims ss ds =
    if Layout.numel_of_shape dims > 0 then
      copy_walk_ ~src ~src_offset ~dst ~dst_offset dims ss ds

  let permute t axes =
    let s = shape t in
    let nd = Array.length s in
    if Array.length axes <> nd then
      fail "permute" "%d axes for a tensor of shape %s" (Array.length axes)
        (shape_to_string s);
    let axes = Array.map (axis_of ~fn:"permute" nd) axes in
    let seen = Array.make nd false in
    Array.iter
      (fun a ->
        if seen.(a) then fail "permute" "axis %d appears twice" a;
        seen.(a) <- true)
      axes;
    let st = Layout.strides s in
    let out_shape = Array.map (fun a -> s.(a)) axes in
    let out = uninit out_shape in
    walk ~src:t ~src_offset:0 ~dst:out ~dst_offset:0 out_shape
      (Array.map (fun a -> st.(a)) axes)
      (Layout.strides out_shape);
    out

  let swapaxes t i j =
    let nd = ndim t in
    let i = axis_of ~fn:"swapaxes" nd i and j = axis_of ~fn:"swapaxes" nd j in
    permute t
      (Array.init nd (fun a -> if a = i then j else if a = j then i else a))

  (** For each axis: its first position, number of positions, step, and whether
      it stays in the result. *)
  let resolve ~fn s ranges =
    let nd = Array.length s in
    let ranges = Array.of_list ranges in
    if Array.length ranges > nd then
      fail fn "%d ranges for a tensor of shape %s" (Array.length ranges)
        (shape_to_string s);
    Array.init nd (fun ax ->
        let d = s.(ax) in
        let clip x = Int.max 0 (Int.min d (if x < 0 then x + d else x)) in
        let range start stop step =
          if step <= 0 then fail fn "step %d is not positive" step;
          let a = clip start and b = clip stop in
          (a, (if b > a then (b - a + step - 1) / step else 0), step, true)
        in
        match if ax < Array.length ranges then ranges.(ax) else All with
        | All -> (0, d, 1, true)
        | Index i ->
            let j = if i < 0 then i + d else i in
            if j < 0 || j >= d then
              fail fn "index %d out of range for axis %d (size %d)" i ax d;
            (j, 1, 1, false)
        | Range (a, b) -> range a b 1
        | Step (a, b, step) -> range a b step)

  (** The element offset, shape and strides of a slice of a tensor of shape [s].
  *)
  let slice_walk ~fn s ranges =
    let r = resolve ~fn s ranges and st = Layout.strides s in
    let offset = ref 0 in
    Array.iteri
      (fun ax (start, _, _, _) -> offset := !offset + (start * st.(ax)))
      r;
    let kept =
      List.filter
        (fun ax ->
          let _, _, _, k = r.(ax) in
          k)
        (List.init (Array.length s) Fun.id)
    in
    let dims =
      Array.of_list
        (List.map
           (fun ax ->
             let _, n, _, _ = r.(ax) in
             n)
           kept)
    in
    let strides =
      Array.of_list
        (List.map
           (fun ax ->
             let _, _, step, _ = r.(ax) in
             st.(ax) * step)
           kept)
    in
    (!offset, dims, strides)

  let slice t ranges =
    let offset, dims, strides = slice_walk ~fn:"slice" (shape t) ranges in
    let out = uninit dims in
    walk ~src:t ~src_offset:offset ~dst:out ~dst_offset:0 dims strides
      (Layout.strides dims);
    out

  let set_slice_ t ranges src =
    let offset, dims, strides = slice_walk ~fn:"set_slice_" (shape t) ranges in
    let src = if shape src = dims then src else broadcast_to src dims in
    walk ~src ~src_offset:0 ~dst:t ~dst_offset:offset dims (Layout.strides dims)
      strides

  let narrow t ~axis ~start ~len =
    let s = shape t in
    let a = axis_of ~fn:"narrow" (Array.length s) axis in
    if start < 0 || len < 0 || start + len > s.(a) then
      fail "narrow" "positions %d to %d outside axis %d (size %d)" start
        (start + len) a s.(a);
    slice t
      (List.init (a + 1) (fun i ->
           if i = a then Range (start, start + len) else All))

  let flip t axes =
    let s = shape t in
    let nd = Array.length s in
    let flipped = Array.make nd false in
    List.iter (fun a -> flipped.(axis_of ~fn:"flip" nd a) <- true) axes;
    let st = Layout.strides s in
    let offset = ref 0 in
    let strides =
      Array.mapi
        (fun a d ->
          if flipped.(a) then begin
            offset := !offset + (Int.max 0 (d - 1) * st.(a));
            -st.(a)
          end
          else st.(a))
        s
    in
    let out = uninit s in
    walk ~src:t ~src_offset:!offset ~dst:out ~dst_offset:0 s strides st;
    out

  let pad ?(value = 0.0) t widths =
    let s = shape t in
    if Array.length widths <> Array.length s then
      fail "pad" "%d widths for a tensor of shape %s" (Array.length widths)
        (shape_to_string s);
    Array.iter
      (fun (l, r) -> if l < 0 || r < 0 then fail "pad" "negative width")
      widths;
    let out_shape =
      Array.mapi (fun a d -> d + fst widths.(a) + snd widths.(a)) s
    in
    let out = full out_shape value in
    let ost = Layout.strides out_shape in
    let offset = ref 0 in
    Array.iteri (fun a (l, _) -> offset := !offset + (l * ost.(a))) widths;
    walk ~src:t ~src_offset:0 ~dst:out ~dst_offset:!offset s (Layout.strides s)
      ost;
    out

  let concat ~axis ts =
    match ts with
    | [] -> fail "concat" "no tensors"
    | first :: _ ->
        let s0 = shape first in
        let nd = Array.length s0 in
        let a = axis_of ~fn:"concat" nd axis in
        List.iter
          (fun t ->
            let s = shape t in
            if
              Array.length s <> nd
              || Array.exists Fun.id
                   (Array.mapi (fun i d -> i <> a && d <> s0.(i)) s)
            then
              fail "concat" "cannot join %s and %s along axis %d"
                (shape_to_string s0) (shape_to_string s) a)
          ts;
        let out_shape = Array.copy s0 in
        out_shape.(a) <- List.fold_left (fun acc t -> acc + (shape t).(a)) 0 ts;
        let out = uninit out_shape in
        let ost = Layout.strides out_shape in
        ignore
          (List.fold_left
             (fun off t ->
               let s = shape t in
               walk ~src:t ~src_offset:0 ~dst:out
                 ~dst_offset:(off * ost.(a))
                 s (Layout.strides s) ost;
               off + s.(a))
             0 ts);
        out

  let unsqueeze t axis =
    let s = shape t in
    let nd = Array.length s in
    let a = if axis < 0 then axis + nd + 1 else axis in
    if a < 0 || a > nd then
      fail "unsqueeze" "axis %d out of range for %d dimensions" axis nd;
    reshape t
      (Array.init (nd + 1) (fun i ->
           if i < a then s.(i) else if i = a then 1 else s.(i - 1)))

  let squeeze ?axis t =
    let s = shape t in
    match axis with
    | None -> reshape t (remove_axes s (fun i -> s.(i) = 1))
    | Some axis ->
        let a = axis_of ~fn:"squeeze" (Array.length s) axis in
        if s.(a) <> 1 then fail "squeeze" "axis %d has size %d, not 1" a s.(a);
        reshape t (remove_axes s (fun i -> i = a))

  let flatten ?(first = 0) ?(last = -1) t =
    let s = shape t in
    let nd = Array.length s in
    if nd = 0 then reshape t [| 1 |]
    else begin
      let f = axis_of ~fn:"flatten" nd first
      and l = axis_of ~fn:"flatten" nd last in
      if f > l then fail "flatten" "first axis %d after last axis %d" f l;
      let merged = Layout.numel_of_shape (Array.sub s f (l - f + 1)) in
      reshape t
        (Array.concat
           [ Array.sub s 0 f; [| merged |]; Array.sub s (l + 1) (nd - l - 1) ])
    end

  let stack ~axis ts =
    match ts with
    | [] -> fail "stack" "no tensors"
    | first :: _ ->
        let s0 = shape first in
        List.iter
          (fun t ->
            if shape t <> s0 then
              fail "stack" "shapes %s and %s differ" (shape_to_string s0)
                (shape_to_string (shape t)))
          ts;
        let nd = Array.length s0 in
        let a = if axis < 0 then axis + nd + 1 else axis in
        if a < 0 || a > nd then
          fail "stack" "axis %d out of range for %d dimensions" axis nd;
        concat ~axis:a (List.map (fun t -> unsqueeze t a) ts)

  let split t ~axis sizes =
    let s = shape t in
    let a = axis_of ~fn:"split" (Array.length s) axis in
    if
      List.exists (fun n -> n < 0) sizes
      || List.fold_left ( + ) 0 sizes <> s.(a)
    then fail "split" "sizes don't add up to %d, the length of axis %d" s.(a) a;
    List.rev
      (snd
         (List.fold_left
            (fun (start, acc) len ->
              (start + len, narrow t ~axis:a ~start ~len :: acc))
            (0, []) sizes))

  let chunk t ~axis n =
    let s = shape t in
    let a = axis_of ~fn:"chunk" (Array.length s) axis in
    if n <= 0 then fail "chunk" "%d chunks" n;
    let d = s.(a) in
    let size = Int.max 1 ((d + n - 1) / n) in
    if d = 0 then [ copy t ]
    else
      List.init
        ((d + size - 1) / size)
        (fun i ->
          narrow t ~axis:a ~start:(i * size)
            ~len:(Int.min size (d - (i * size))))

  (** [[| 1; d0; 1; d1; ... |]] broadcast to [[| r0; d0; r1; d1; ... |]]. *)
  let tile t reps =
    let s = shape t in
    if Array.exists (fun r -> r < 0) reps then fail "tile" "negative repetition";
    let k = Int.max (Array.length s) (Array.length reps) in
    let lead a = Array.append (Array.make (k - Array.length a) 1) a in
    let s = lead s and reps = lead reps in
    let inter =
      Array.init (2 * k) (fun i -> if i mod 2 = 0 then 1 else s.(i / 2))
    in
    let big =
      Array.init (2 * k) (fun i ->
          if i mod 2 = 0 then reps.(i / 2) else s.(i / 2))
    in
    reshape
      (broadcast_to (reshape t inter) big)
      (Array.init k (fun i -> reps.(i) * s.(i)))

  let repeat t ~axis n =
    let s = shape t in
    let nd = Array.length s in
    let a = axis_of ~fn:"repeat" nd axis in
    if n < 0 then fail "repeat" "negative count %d" n;
    let ins x =
      Array.concat
        [ Array.sub s 0 (a + 1); [| x |]; Array.sub s (a + 1) (nd - a - 1) ]
    in
    let out = Array.copy s in
    out.(a) <- s.(a) * n;
    reshape (broadcast_to (reshape t (ins 1)) (ins n)) out

  (* -------------------------------------------------------------------- *)
  (* Elementwise and masks                                                 *)
  (* -------------------------------------------------------------------- *)

  let masked_fill t mask x =
    let s = shape t in
    if Layout.broadcast_shapes (shape mask) s <> s then
      fail "masked_fill" "mask %s does not broadcast to %s"
        (shape_to_string (shape mask))
        (shape_to_string s);
    where mask (scalar x) t

  (** [maximum] then [minimum], so NaN stays NaN and, if [min > max], every
      element becomes [max], as in PyTorch. *)
  let clamp ?min ?max t =
    let t = match min with Some lo -> maximum t (scalar lo) | None -> t in
    match max with
    | Some hi -> minimum t (scalar hi)
    | None -> if min = None then copy t else t

  (* -------------------------------------------------------------------- *)
  (* Reductions                                                            *)
  (* -------------------------------------------------------------------- *)

  let mean_axis ?(keepdims = false) t axis =
    let s = shape t in
    let a = axis_of ~fn:"mean_axis" (Array.length s) axis in
    scale (1.0 /. float_of_int s.(a)) (sum_axis ~keepdims t a)

  (** The shape with every reduced axis set to 1, and the number of elements
      each output sums. *)
  let reduced ~fn t axes =
    let s = shape t in
    let nd = Array.length s in
    let drop = Array.make nd false in
    List.iter (fun a -> drop.(axis_of ~fn nd a) <- true) axes;
    let kept = Array.mapi (fun i d -> if drop.(i) then 1 else d) s in
    let count = ref 1 in
    Array.iteri (fun i d -> if drop.(i) then count := !count * d) s;
    (kept, drop, !count)

  (** One [sum_to_shape], which already reduces several axes at once in a fixed
      order. *)
  let sum_axes ?(keepdims = false) t axes =
    let kept, drop, _ = reduced ~fn:"sum_axes" t axes in
    let r = sum_to_shape t kept in
    if keepdims then r else reshape r (remove_axes kept (fun i -> drop.(i)))

  let mean_axes ?keepdims t axes =
    let _, _, count = reduced ~fn:"mean_axes" t axes in
    scale (1.0 /. float_of_int count) (sum_axes ?keepdims t axes)

  (** The maximum is clamped to a finite value before it is subtracted, so a row
      that is all [-infinity] (or contains [+infinity]) gives [-infinity] (or
      [+infinity]) instead of NaN. *)
  let logsumexp ?(keepdims = false) t axis =
    let s = shape t in
    let a = axis_of ~fn:"logsumexp" (Array.length s) axis in
    let m = max_axis ~keepdims:true t a in
    let m =
      maximum (minimum m (scalar Float.max_float)) (scalar (-.Float.max_float))
    in
    let r = add m (log (sum_axis ~keepdims:true (exp (sub t m)) a)) in
    if keepdims then r else reshape r (remove_axes (shape r) (fun i -> i = a))

  (** Two passes: the mean, then the squared deviations from it. *)
  let var ?(correction = 0) ?(keepdims = false) t axis =
    let s = shape t in
    let a = axis_of ~fn:"var" (Array.length s) axis in
    let d =
      sub t (scale (1.0 /. float_of_int s.(a)) (sum_axis ~keepdims:true t a))
    in
    scale
      (1.0 /. float_of_int (s.(a) - correction))
      (sum_axis ~keepdims (mul d d) a)

  let std ?correction ?keepdims t axis = sqrt (var ?correction ?keepdims t axis)

  (* -------------------------------------------------------------------- *)
  (* Linear algebra                                                        *)
  (* -------------------------------------------------------------------- *)

  let matmul a b =
    let sa = shape a and sb = shape b in
    let na = Array.length sa and nb = Array.length sb in
    if na = 0 || nb = 0 then
      fail "matmul" "cannot multiply %s by %s" (shape_to_string sa)
        (shape_to_string sb);
    let a' = if na = 1 then reshape a [| 1; sa.(0) |] else a in
    let b' = if nb = 1 then reshape b [| sb.(0); 1 |] else b in
    let r = if ndim a' = 2 && ndim b' = 2 then mm a' b' else bmm a' b' in
    if na > 1 && nb > 1 then r
    else
      let rs = shape r in
      let nr = Array.length rs in
      reshape r
        (remove_axes rs (fun i ->
             (na = 1 && i = nr - 2) || (nb = 1 && i = nr - 1)))

  let dot a b =
    let sa = shape a and sb = shape b in
    if Array.length sa <> 1 || sa <> sb then
      fail "dot" "expected two 1-D tensors of the same length, got %s and %s"
        (shape_to_string sa) (shape_to_string sb);
    sum (mul a b)

  let outer a b =
    let sa = shape a and sb = shape b in
    if Array.length sa <> 1 || Array.length sb <> 1 then
      fail "outer" "expected two 1-D tensors, got %s and %s"
        (shape_to_string sa) (shape_to_string sb);
    mul (reshape a [| sa.(0); 1 |]) (reshape b [| 1; sb.(0) |])

  let norm ?(p = 2.0) t =
    if p = Float.infinity then max (abs t)
    else if not (p > 0.0) then fail "norm" "p = %g is not positive" p
    else if p = 2.0 then sqrt (sum (mul t t))
    else if p = 1.0 then sum (abs t)
    else pow (sum (pow (abs t) p)) (1.0 /. p)
end

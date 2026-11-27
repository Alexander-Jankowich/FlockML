(* See layout.mli. Moved here from tensor.ml so every backend shares them. *)

(** Number of elements a tensor of this shape holds. *)
let numel_of_shape shape = Array.fold_left ( * ) 1 shape

(** Render a shape as ["[2; 3]"], for error messages. *)
let shape_to_string shape =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int shape)) ^ "]"

(** How far apart consecutive indices of each axis are in [data]:
    [strides [|2; 3; 4|] = [|12; 4; 1|]]. *)
let strides shape =
  let n = Array.length shape in
  let s = Array.make n 1 in
  for i = n - 2 downto 0 do
    s.(i) <- s.(i + 1) * shape.(i + 1)
  done;
  s

(** NumPy rule: right-align the shapes; on each axis the sizes must be equal or
    one of them must be 1, and the result takes the larger. Walking both shapes
    from the last axis and prepending to [acc] builds the result in order. *)
let broadcast_shapes a b =
  let rec aux acc l1 l2 =
    match (l1, l2) with
    | [], [] -> acc
    | x :: xs, [] -> aux (x :: acc) xs []
    | [], y :: ys -> aux (y :: acc) [] ys
    | x :: xs, y :: ys ->
        if x = y || y = 1 then aux (x :: acc) xs ys
        else if x = 1 then aux (y :: acc) xs ys
        else
          invalid_arg
            (Printf.sprintf
               "Tensor.broadcast_shapes: cannot broadcast %s and %s"
               (shape_to_string a) (shape_to_string b))
  in
  let rev_list s = List.rev (Array.to_list s) in
  Array.of_list (aux [] (rev_list a) (rev_list b))

(** The strides for reading a tensor of shape [src] as if it had the larger
    shape [out]: its own strides, with 0 on every axis broadcasting repeats
    (missing leading axes and axes of size 1). Walking [out]'s multi-index with
    these strides visits the source element each position comes from. *)
let broadcast_strides ~src ~out =
  let k = Array.length out - Array.length src in
  let st = strides src in
  Array.init (Array.length out) (fun ax ->
      let s = ax - k in
      if s < 0 || src.(s) = 1 then 0 else st.(s))

(** Simplify a walk over [dims] in which operand [o] moves by [sts.(o)]: drop
    axes of size 1, and merge an axis into the one before it whenever every
    operand moves through the pair contiguously. The visiting order stays
    row-major, but the innermost axis becomes as long as possible, so
    [[50176; 8] + [8]] is 50176 rows of 8, and [[64; 128] + [64; 128]] is one
    row of 8192. Always returns at least one axis. *)
let coalesce dims (sts : int array array) =
  let nd = Array.length dims and nops = Array.length sts in
  let out_dims = Array.make (max nd 1) 1 in
  let out_sts = Array.init nops (fun _ -> Array.make (max nd 1) 0) in
  let len = ref 0 in
  for ax = 0 to nd - 1 do
    let d = dims.(ax) in
    if d <> 1 then begin
      let prev = !len - 1 in
      let mergeable =
        prev >= 0
        &&
        let ok = ref true in
        for o = 0 to nops - 1 do
          if out_sts.(o).(prev) <> sts.(o).(ax) * d then ok := false
        done;
        !ok
      in
      if mergeable then begin
        out_dims.(prev) <- out_dims.(prev) * d;
        for o = 0 to nops - 1 do
          out_sts.(o).(prev) <- sts.(o).(ax)
        done
      end
      else begin
        out_dims.(!len) <- d;
        for o = 0 to nops - 1 do
          out_sts.(o).(!len) <- sts.(o).(ax)
        done;
        incr len
      end
    end
  done;
  let len = max !len 1 in
  (Array.sub out_dims 0 len, Array.map (fun s -> Array.sub s 0 len) out_sts)

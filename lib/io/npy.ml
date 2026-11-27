(** Reading and writing NumPy's [.npy] files, in pure OCaml.

    A [.npy] file is a short text header (a Python dict literal giving the
    element type, the memory order and the shape) followed by the raw elements.
    This module reads every common numeric type (float64, float32, signed and
    unsigned integers of 1 to 8 bytes, booleans), in either byte order and in C
    or Fortran order, and writes float64, float32 or int64 in C order. So arrays
    move between FlockML and NumPy with [numpy.save] / [numpy.load]. *)

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  val save : ?dtype:[ `Float64 | `Float32 ] -> string -> tensor -> unit
  (** [save path t] writes [t] to [path], as float64 unless [~dtype:`Float32] is
      given. *)

  val load : string -> tensor
  (** Read an array of any numeric type, converting its elements to floats.
      Raises [Failure] (naming the file) for a file that isn't a [.npy] array of
      a supported type. *)

  val save_ints : ?shape:int array -> string -> int array -> unit
  (** Write integers (for example labels or indices) as int64. [shape] defaults
      to 1-D. Raises [Invalid_argument] if it doesn't match the number of
      values. *)

  val load_ints : string -> int array * int array
  (** Read an array of integers or booleans, as its shape and its values in
      row-major order. Raises [Failure] for floating-point data. *)
end

(** The parts that don't depend on the tensor type. *)

let magic = "\147NUMPY"

let fail path fmt =
  Printf.ksprintf (fun s -> failwith (Printf.sprintf "Npy: %s: %s" path s)) fmt

(** The header dict for [descr] and [shape], padded with spaces and ended with a
    newline so the data starts on a 64-byte boundary. *)
let header descr shape =
  let dims =
    match Array.to_list shape with
    | [] -> "()"
    | [ d ] -> Printf.sprintf "(%d,)" d
    | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"
  in
  let dict =
    Printf.sprintf "{'descr': '%s', 'fortran_order': False, 'shape': %s, }"
      descr dims
  in
  let unpadded = String.length magic + 4 + String.length dict + 1 in
  let pad = (64 - (unpadded mod 64)) mod 64 in
  let dict = dict ^ String.make pad ' ' ^ "\n" in
  let len = String.length dict in
  if len >= 65536 then begin
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 (Int32.of_int len);
    magic ^ "\002\000" ^ Bytes.to_string b ^ dict
  end
  else begin
    let b = Bytes.create 2 in
    Bytes.set_uint16_le b 0 len;
    magic ^ "\001\000" ^ Bytes.to_string b ^ dict
  end

let write path contents =
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc contents)

type dtype = { kind : char; size : int; big : bool }
(** The element type: its kind (['f'], ['i'], ['u'] or ['b']), its size in
    bytes, and whether it is big-endian. *)

type parsed = {
  dtype : dtype;
  fortran : bool;
  shape : int array;
  data : string;
  offset : int;
}

(** The text after [key] in the header dict, starting at its value. *)
let value_of ~path dict key =
  let pat = "'" ^ key ^ "'" in
  let pat' = "\"" ^ key ^ "\"" in
  let find p =
    let lp = String.length p and ld = String.length dict in
    let rec go i =
      if i + lp > ld then None
      else if String.sub dict i lp = p then Some (i + lp)
      else go (i + 1)
    in
    go 0
  in
  match (find pat, find pat') with
  | Some i, _ | None, Some i ->
      let j = String.index_from dict i ':' + 1 in
      let rec skip j =
        if j < String.length dict && dict.[j] = ' ' then skip (j + 1) else j
      in
      String.sub dict (skip j) (String.length dict - skip j)
  | None, None -> fail path "header has no '%s'" key

let parse path s =
  let n = String.length s in
  if n < 10 || String.sub s 0 6 <> magic then fail path "not a .npy file";
  let major = Char.code s.[6] in
  let hlen, start =
    if major = 1 then (Char.code s.[8] lor (Char.code s.[9] lsl 8), 10)
    else if n >= 12 then (Int32.to_int (String.get_int32_le s 8), 12)
    else fail path "truncated header"
  in
  if start + hlen > n then fail path "truncated header";
  let dict = String.sub s start hlen in
  let descr =
    let v = value_of ~path dict "descr" in
    let q = v.[0] in
    String.sub v 1 (String.index_from v 1 q - 1)
  in
  let dtype =
    if String.length descr < 3 then fail path "unsupported dtype %s" descr
    else
      let big =
        match descr.[0] with
        | '>' -> true
        | '<' | '|' -> false
        | '=' -> Sys.big_endian
        | _ -> fail path "unsupported dtype %s" descr
      in
      let kind = descr.[1] in
      let size =
        match
          int_of_string_opt (String.sub descr 2 (String.length descr - 2))
        with
        | Some k -> k
        | None -> fail path "unsupported dtype %s" descr
      in
      (match (kind, size) with
      | 'f', (4 | 8) | ('i' | 'u'), (1 | 2 | 4 | 8) | 'b', 1 -> ()
      | _ -> fail path "unsupported dtype %s" descr);
      { kind; size; big }
  in
  let fortran =
    let v = value_of ~path dict "fortran_order" in
    String.length v >= 4 && String.sub v 0 4 = "True"
  in
  let shape =
    let v = value_of ~path dict "shape" in
    let close = String.index v ')' in
    String.sub v 1 (close - 1)
    |> String.split_on_char ',' |> List.map String.trim
    |> List.filter (fun x -> x <> "")
    |> List.map (fun x ->
        match int_of_string_opt x with
        | Some d -> d
        | None -> fail path "bad shape entry %s" x)
    |> Array.of_list
  in
  let count = Layout.numel_of_shape shape in
  if n - (start + hlen) < count * dtype.size then fail path "truncated data";
  { dtype; fortran; shape; data = s; offset = start + hlen }

let read path = parse path (In_channel.with_open_bin path In_channel.input_all)

(** Element [i] as a float, and as an int. *)
let float_at p i =
  let o = p.offset + (i * p.dtype.size) and s = p.data and big = p.dtype.big in
  match (p.dtype.kind, p.dtype.size) with
  | 'f', 8 ->
      Int64.float_of_bits
        (if big then String.get_int64_be s o else String.get_int64_le s o)
  | 'f', _ ->
      Int32.float_of_bits
        (if big then String.get_int32_be s o else String.get_int32_le s o)
  | 'i', 1 -> float_of_int (String.get_int8 s o)
  | 'i', 2 ->
      float_of_int
        (if big then String.get_int16_be s o else String.get_int16_le s o)
  | 'i', 4 ->
      Int32.to_float
        (if big then String.get_int32_be s o else String.get_int32_le s o)
  | 'i', _ ->
      Int64.to_float
        (if big then String.get_int64_be s o else String.get_int64_le s o)
  | ('u' | 'b'), 1 -> float_of_int (String.get_uint8 s o)
  | 'u', 2 ->
      float_of_int
        (if big then String.get_uint16_be s o else String.get_uint16_le s o)
  | 'u', 4 ->
      let x =
        if big then String.get_int32_be s o else String.get_int32_le s o
      in
      Int64.to_float (Int64.logand (Int64.of_int32 x) 0xFFFF_FFFFL)
  | _ ->
      let x =
        if big then String.get_int64_be s o else String.get_int64_le s o
      in
      if Int64.compare x 0L >= 0 then Int64.to_float x
      else Int64.to_float x +. 18446744073709551616.0

(** Integers are read as integers, so 64-bit values keep every bit (an unsigned
    value above [max_int] wraps). *)
let int_at p i =
  let o = p.offset + (i * p.dtype.size) and s = p.data and big = p.dtype.big in
  match (p.dtype.kind, p.dtype.size) with
  | 'i', 1 -> String.get_int8 s o
  | 'i', 2 -> if big then String.get_int16_be s o else String.get_int16_le s o
  | 'i', 4 ->
      Int32.to_int
        (if big then String.get_int32_be s o else String.get_int32_le s o)
  | ('u' | 'b'), 1 -> String.get_uint8 s o
  | 'u', 2 -> if big then String.get_uint16_be s o else String.get_uint16_le s o
  | 'u', 4 ->
      Int32.to_int
        (if big then String.get_int32_be s o else String.get_int32_le s o)
      land 0xFFFF_FFFF
  | _ ->
      Int64.to_int
        (if big then String.get_int64_be s o else String.get_int64_le s o)

(** The position in Fortran-ordered storage of each row-major position. *)
let fortran_index shape =
  let nd = Array.length shape in
  let fst = Array.make nd 1 in
  for a = 1 to nd - 1 do
    fst.(a) <- fst.(a - 1) * shape.(a - 1)
  done;
  let idx = Array.make nd 0 in
  Array.init (Layout.numel_of_shape shape) (fun _ ->
      let f = ref 0 in
      for a = 0 to nd - 1 do
        f := !f + (idx.(a) * fst.(a))
      done;
      let a = ref (nd - 1) in
      while !a >= 0 do
        idx.(!a) <- idx.(!a) + 1;
        if idx.(!a) < shape.(!a) then a := -1
        else begin
          idx.(!a) <- 0;
          decr a
        end
      done;
      !f)

(** The elements in row-major order. *)
let elements p get =
  let n = Layout.numel_of_shape p.shape in
  if p.fortran then Array.map (get p) (fortran_index p.shape)
  else Array.init n (get p)

(** The module for one backend. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  let save ?(dtype = `Float64) path t =
    let a = T.to_array t in
    let size = match dtype with `Float64 -> 8 | `Float32 -> 4 in
    let b = Bytes.create (Array.length a * size) in
    Array.iteri
      (fun i x ->
        match dtype with
        | `Float64 -> Bytes.set_int64_le b (i * 8) (Int64.bits_of_float x)
        | `Float32 -> Bytes.set_int32_le b (i * 4) (Int32.bits_of_float x))
      a;
    write path
      (header
         (match dtype with `Float64 -> "<f8" | `Float32 -> "<f4")
         (T.shape t)
      ^ Bytes.to_string b)

  let load path =
    let p = read path in
    T.of_array p.shape (elements p float_at)

  let save_ints ?shape path values =
    let shape =
      match shape with Some s -> s | None -> [| Array.length values |]
    in
    if Layout.numel_of_shape shape <> Array.length values then
      invalid_arg
        (Printf.sprintf "Npy.save_ints: shape %s for %d values"
           (Layout.shape_to_string shape)
           (Array.length values));
    let b = Bytes.create (Array.length values * 8) in
    Array.iteri
      (fun i x -> Bytes.set_int64_le b (i * 8) (Int64.of_int x))
      values;
    write path (header "<i8" shape ^ Bytes.to_string b)

  let load_ints path =
    let p = read path in
    if p.dtype.kind = 'f' then
      fail path "floating-point data, expected integers";
    (p.shape, elements p int_at)
end

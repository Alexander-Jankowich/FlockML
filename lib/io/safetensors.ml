(** Reading and writing the safetensors format, in pure OCaml.

    A [.safetensors] file is:
    - an 8-byte little-endian header length [n];
    - [n] bytes of JSON, mapping each tensor's name to its dtype, shape and byte
      range ["data_offsets"] in the data that follows, plus an optional
      ["__metadata__"] object of strings;
    - the raw little-endian, row-major data of every tensor.

    It is the format of Hugging Face's [safetensors] package, so PyTorch reads
    these files with [safetensors.torch.load_file] and writes them with
    [save_file]. FlockML writes float64 or float32, and reads every float and
    integer dtype the format defines except the 8-bit floats. Integers and
    booleans become floats. *)

type dtype = [ `F64 | `F32 ]
(** The element type to write. *)

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  val save :
    ?dtype:dtype ->
    ?metadata:(string * string) list ->
    string ->
    (string * tensor) list ->
    unit
  (** [save path entries] writes the named tensors to [path], as float64 unless
      [~dtype:`F32] is given. Raises [Invalid_argument] if two entries share a
      name, or a name is ["__metadata__"]. *)

  val load : string -> (string * tensor) list
  (** Every tensor in the file, converted to the backend's floats, in the order
      of their data. Raises [Failure] (naming the file) for a file that isn't
      valid safetensors, or holds an unsupported dtype. *)

  val load_metadata : string -> (string * string) list
  (** The file's ["__metadata__"], or [[]]. *)

  val to_string :
    ?dtype:dtype ->
    ?metadata:(string * string) list ->
    (string * tensor) list ->
    string
  (** The bytes {!save} writes. *)

  val of_string : ?path:string -> string -> (string * tensor) list
  (** {!load} from bytes in memory; [path] names them in errors. *)
end

(* -------------------------------------------------------------------------- *)
(* A small JSON reader and writer: enough for the header                     *)
(* -------------------------------------------------------------------------- *)

type json =
  | Null
  | Bool of bool
  | Number of float
  | String of string
  | List of json list
  | Object of (string * json) list

let fail path fmt =
  Printf.ksprintf
    (fun s -> failwith (Printf.sprintf "Safetensors: %s: %s" path s))
    fmt

(** A JSON string literal, with the escapes JSON requires. *)
let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c when Char.code c < 0x20 ->
          Buffer.add_string b (Printf.sprintf "\\u%04x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

(** Parse a JSON document. Numbers are read as floats, which holds every offset
    and dimension exactly up to 2^53 bytes. *)
let parse_json ~path s =
  let n = String.length s in
  let pos = ref 0 in
  let peek () = if !pos < n then s.[!pos] else '\000' in
  let rec ws () =
    if
      !pos < n
      && match s.[!pos] with ' ' | '\n' | '\r' | '\t' -> true | _ -> false
    then begin
      incr pos;
      ws ()
    end
  in
  let expect c =
    ws ();
    if peek () <> c then fail path "bad header: expected '%c' at byte %d" c !pos;
    incr pos
  in
  let utf8 b cp =
    if cp < 0x80 then Buffer.add_char b (Char.chr cp)
    else if cp < 0x800 then begin
      Buffer.add_char b (Char.chr (0xC0 lor (cp lsr 6)));
      Buffer.add_char b (Char.chr (0x80 lor (cp land 0x3F)))
    end
    else begin
      Buffer.add_char b (Char.chr (0xE0 lor (cp lsr 12)));
      Buffer.add_char b (Char.chr (0x80 lor ((cp lsr 6) land 0x3F)));
      Buffer.add_char b (Char.chr (0x80 lor (cp land 0x3F)))
    end
  in
  let string () =
    expect '"';
    let b = Buffer.create 16 in
    let rec go () =
      if !pos >= n then fail path "bad header: unterminated string";
      let c = s.[!pos] in
      incr pos;
      match c with
      | '"' -> ()
      | '\\' ->
          if !pos >= n then fail path "bad header: unterminated string";
          let e = s.[!pos] in
          incr pos;
          (match e with
          | '"' | '\\' | '/' -> Buffer.add_char b e
          | 'n' -> Buffer.add_char b '\n'
          | 't' -> Buffer.add_char b '\t'
          | 'r' -> Buffer.add_char b '\r'
          | 'b' -> Buffer.add_char b '\b'
          | 'f' -> Buffer.add_char b '\012'
          | 'u' when !pos + 4 <= n ->
              let cp = int_of_string ("0x" ^ String.sub s !pos 4) in
              pos := !pos + 4;
              utf8 b cp
          | _ -> fail path "bad header: bad escape");
          go ()
      | c ->
          Buffer.add_char b c;
          go ()
    in
    go ();
    Buffer.contents b
  in
  let rec value () =
    ws ();
    match peek () with
    | '{' ->
        incr pos;
        ws ();
        if peek () = '}' then begin
          incr pos;
          Object []
        end
        else
          let rec members acc =
            let k = string () in
            expect ':';
            let v = value () in
            ws ();
            match peek () with
            | ',' ->
                incr pos;
                members ((k, v) :: acc)
            | '}' ->
                incr pos;
                Object (List.rev ((k, v) :: acc))
            | _ -> fail path "bad header: expected ',' or '}' at byte %d" !pos
          in
          members []
    | '[' ->
        incr pos;
        ws ();
        if peek () = ']' then begin
          incr pos;
          List []
        end
        else
          let rec items acc =
            let v = value () in
            ws ();
            match peek () with
            | ',' ->
                incr pos;
                items (v :: acc)
            | ']' ->
                incr pos;
                List (List.rev (v :: acc))
            | _ -> fail path "bad header: expected ',' or ']' at byte %d" !pos
          in
          items []
    | '"' -> String (string ())
    | 't' when !pos + 4 <= n && String.sub s !pos 4 = "true" ->
        pos := !pos + 4;
        Bool true
    | 'f' when !pos + 5 <= n && String.sub s !pos 5 = "false" ->
        pos := !pos + 5;
        Bool false
    | 'n' when !pos + 4 <= n && String.sub s !pos 4 = "null" ->
        pos := !pos + 4;
        Null
    | _ -> (
        let start = !pos in
        while
          !pos < n
          &&
          match s.[!pos] with
          | '0' .. '9' | '-' | '+' | '.' | 'e' | 'E' -> true
          | _ -> false
        do
          incr pos
        done;
        if !pos = start then fail path "bad header: unexpected byte %d" start;
        match float_of_string_opt (String.sub s start (!pos - start)) with
        | Some f -> Number f
        | None -> fail path "bad header: bad number at byte %d" start)
  in
  let v = value () in
  ws ();
  if !pos <> n then fail path "bad header: trailing bytes";
  v

(* -------------------------------------------------------------------------- *)
(* Element encodings                                                           *)
(* -------------------------------------------------------------------------- *)

(** Size in bytes and a reader of element [i] of a buffer, for each dtype. *)
let reader ~path dtype : int * (string -> int -> float) =
  match dtype with
  | "F64" -> (8, fun s o -> Int64.float_of_bits (String.get_int64_le s o))
  | "F32" -> (4, fun s o -> Int32.float_of_bits (String.get_int32_le s o))
  | "BF16" ->
      ( 2,
        fun s o ->
          Int32.float_of_bits
            (Int32.shift_left (Int32.of_int (String.get_uint16_le s o)) 16) )
  | "F16" ->
      ( 2,
        fun s o ->
          let h = String.get_uint16_le s o in
          let sign = if h land 0x8000 <> 0 then -1. else 1. in
          let e = (h lsr 10) land 0x1F and m = h land 0x3FF in
          if e = 0 then sign *. Float.ldexp (float_of_int m) (-24)
          else if e = 31 then
            if m = 0 then sign *. Float.infinity else Float.nan
          else sign *. Float.ldexp (float_of_int (m lor 0x400)) (e - 25) )
  | "I64" -> (8, fun s o -> Int64.to_float (String.get_int64_le s o))
  | "U64" ->
      ( 8,
        fun s o ->
          let x = String.get_int64_le s o in
          if Int64.compare x 0L >= 0 then Int64.to_float x
          else Int64.to_float x +. 18446744073709551616.0 )
  | "I32" -> (4, fun s o -> Int32.to_float (String.get_int32_le s o))
  | "U32" ->
      ( 4,
        fun s o ->
          Int64.to_float
            (Int64.logand
               (Int64.of_int32 (String.get_int32_le s o))
               0xFFFF_FFFFL) )
  | "I16" -> (2, fun s o -> float_of_int (String.get_int16_le s o))
  | "U16" -> (2, fun s o -> float_of_int (String.get_uint16_le s o))
  | "I8" -> (1, fun s o -> float_of_int (String.get_int8 s o))
  | "U8" | "BOOL" -> (1, fun s o -> float_of_int (String.get_uint8 s o))
  | d -> fail path "unsupported dtype %s" d

type entry = { name : string; shape : int array; data : float array }

(** The header and data of a file, decoded to floats. *)
let decode ~path s =
  let n = String.length s in
  if n < 8 then fail path "too short for a header length";
  let hlen = String.get_int64_le s 0 in
  if Int64.compare hlen 0L < 0 || Int64.compare hlen (Int64.of_int (n - 8)) > 0
  then fail path "header length %Ld exceeds the file" hlen;
  let hlen = Int64.to_int hlen in
  let base = 8 + hlen in
  let fields =
    match parse_json ~path (String.sub s 8 hlen) with
    | Object fields -> fields
    | _ -> fail path "the header is not a JSON object"
  in
  let metadata =
    match List.assoc_opt "__metadata__" fields with
    | Some (Object kvs) ->
        List.filter_map (function k, String v -> Some (k, v) | _ -> None) kvs
    | _ -> []
  in
  let int_of ~what = function
    | Number f when Float.is_integer f && f >= 0. -> int_of_float f
    | _ -> fail path "bad %s" what
  in
  let entries =
    List.filter_map
      (fun (name, v) ->
        if name = "__metadata__" then None
        else
          match v with
          | Object info ->
              let get k =
                match List.assoc_opt k info with
                | Some x -> x
                | None -> fail path "%s has no %s" name k
              in
              let dtype =
                match get "dtype" with
                | String d -> d
                | _ -> fail path "bad dtype of %s" name
              in
              let shape =
                match get "shape" with
                | List ds ->
                    Array.of_list
                      (List.map (int_of ~what:("shape of " ^ name)) ds)
                | _ -> fail path "bad shape of %s" name
              in
              let lo, hi =
                match get "data_offsets" with
                | List [ a; b ] ->
                    ( int_of ~what:"data_offsets" a,
                      int_of ~what:"data_offsets" b )
                | _ -> fail path "bad data_offsets of %s" name
              in
              let size, read = reader ~path dtype in
              let count = Layout.numel_of_shape shape in
              if hi < lo || hi - lo <> count * size || base + hi > n then
                fail path
                  "the data_offsets of %s don't fit its shape or the file" name;
              let data =
                Array.init count (fun i -> read s (base + lo + (i * size)))
              in
              Some (lo, { name; shape; data })
          | _ -> fail path "bad entry %s" name)
      fields
  in
  ( List.map snd (List.stable_sort (fun (a, _) (b, _) -> compare a b) entries),
    metadata )

let encode ?(dtype = `F64) ?(metadata = [])
    (entries : (string * int array * float array) list) =
  let seen = Hashtbl.create 16 in
  List.iter
    (fun (name, _, _) ->
      if name = "__metadata__" then
        invalid_arg "Safetensors.save: __metadata__ is not a tensor name";
      if Hashtbl.mem seen name then
        invalid_arg
          (Printf.sprintf "Safetensors.save: the name %S is used twice" name);
      Hashtbl.add seen name ())
    entries;
  let size, tag = match dtype with `F64 -> (8, "F64") | `F32 -> (4, "F32") in
  let offset = ref 0 in
  let header =
    List.map
      (fun (name, shape, data) ->
        let lo = !offset in
        offset := lo + (Array.length data * size);
        Printf.sprintf
          "%s:{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[%d,%d]}"
          (quote name) tag
          (String.concat "," (Array.to_list (Array.map string_of_int shape)))
          lo !offset)
      entries
  in
  let meta =
    if metadata = [] then []
    else
      [
        "\"__metadata__\":{"
        ^ String.concat ","
            (List.map (fun (k, v) -> quote k ^ ":" ^ quote v) metadata)
        ^ "}";
      ]
  in
  let json = "{" ^ String.concat "," (meta @ header) ^ "}" in
  (* Pad with spaces so the data starts on an 8-byte boundary, as the
     reference implementation does. *)
  let json = json ^ String.make ((8 - (String.length json mod 8)) mod 8) ' ' in
  let b = Bytes.create (8 + String.length json + !offset) in
  Bytes.set_int64_le b 0 (Int64.of_int (String.length json));
  Bytes.blit_string json 0 b 8 (String.length json);
  let pos = ref (8 + String.length json) in
  List.iter
    (fun (_, _, data) ->
      Array.iter
        (fun x ->
          (match dtype with
          | `F64 -> Bytes.set_int64_le b !pos (Int64.bits_of_float x)
          | `F32 -> Bytes.set_int32_le b !pos (Int32.bits_of_float x));
          pos := !pos + size)
        data)
    entries;
  Bytes.to_string b

(** The module for one backend. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  let to_string ?dtype ?metadata entries =
    encode ?dtype ?metadata
      (List.map (fun (name, t) -> (name, T.shape t, T.to_array t)) entries)

  let save ?dtype ?metadata path entries =
    let s = to_string ?dtype ?metadata entries in
    Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc s)

  let of_string ?(path = "<string>") s =
    List.map
      (fun e -> (e.name, T.of_array e.shape e.data))
      (fst (decode ~path s))

  let read path = In_channel.with_open_bin path In_channel.input_all
  let load path = of_string ~path (read path)
  let load_metadata path = snd (decode ~path (read path))
end

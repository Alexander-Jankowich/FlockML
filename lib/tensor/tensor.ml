(** Default backend: a contiguous row-major float array plus a shape.

    The element at multi-index [[| i0; i1; ...; ik |]] lives at
    [data.(i0 * strides.(0) + i1 * strides.(1) + ... + ik * strides.(k))].

    This file defines [exp], [log], [sqrt] and [neg], which shadow the Stdlib
    versions, so helper code calls [Float.exp], [Float.log], [Float.sqrt] etc.
    explicitly.

    {2 How the kernels are written}

    - Every kernel checks its shapes once, up front, and then reads and writes
      with [uget]/[uset] (unchecked access). Their [float array] annotation
      makes the compiler emit plain unboxed float loads and stores.
    - Hot loops never call a closure per element. A closure call boxes the
      float it returns (an allocation per element), and the call itself costs
      more than an addition.
    - Large loops are split across domains with {!Parallel.parallel_for}. Each
      output element is always computed by exactly one task in a fixed order,
      so results don't depend on the number of threads. *)

(** A shape plus its elements in row-major order. *)
type t = { shape : int array; data : float array }

let[@inline] uget (a : float array) i = Array.unsafe_get a i
let[@inline] uset (a : float array) i (x : float) = Array.unsafe_set a i x

(* Integer min and max. [Stdlib.min] is polymorphic, so each call goes
   through the generic comparison in the runtime, which is far too slow for
   index arithmetic inside loops. *)
let[@inline] min (a : int) b = if a < b then a else b
let[@inline] max (a : int) b = if a > b then a else b

(* ------------------------------------------------------------------------ *)
(* Construction and inspection                                               *)
(* ------------------------------------------------------------------------ *)

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

let shape t = Array.copy t.shape
let numel t = Array.length t.data
let ndim t = Array.length t.shape

let of_array shape data =
  let expected = numel_of_shape shape in
  if expected <> Array.length data then
    invalid_arg
      (Printf.sprintf "Tensor.of_array: shape %s needs %d elements, got %d"
         (shape_to_string shape) expected (Array.length data));
  { shape = Array.copy shape; data = Array.copy data }

let full shape x =
  { shape = Array.copy shape; data = Array.make (numel_of_shape shape) x }

let zeros shape = full shape 0.0
let ones shape = full shape 1.0
let scalar x = { shape = [||]; data = [| x |] }
let to_array t = Array.copy t.data
let copy t = { shape = Array.copy t.shape; data = Array.copy t.data }

let item t =
  if numel t <> 1 then
    invalid_arg
      (Printf.sprintf "Tensor.item: tensor of shape %s has %d elements"
         (shape_to_string t.shape) (numel t));
  t.data.(0)

let allclose ?(rtol = 1e-5) ?(atol = 1e-8) a b =
  a.shape = b.shape
  && begin
    let ok = ref true in
    Array.iteri
      (fun i x ->
        let y = b.data.(i) in
        if Float.abs (x -. y) > atol +. (rtol *. Float.abs y) then ok := false)
      a.data;
    !ok
  end

let pp fmt t =
  let nd = Array.length t.shape in
  if numel t > 2000 then
    Format.fprintf fmt "tensor %s (%d elements)" (shape_to_string t.shape)
      (numel t)
  else if nd = 0 then Format.fprintf fmt "tensor %g" t.data.(0)
  else begin
    let st = strides t.shape in
    let rec go axis offset =
      Format.fprintf fmt "[";
      if axis = nd - 1 then
        for i = 0 to t.shape.(axis) - 1 do
          if i > 0 then Format.fprintf fmt ", ";
          Format.fprintf fmt "%.4g" t.data.(offset + i)
        done
      else begin
        Format.fprintf fmt "@[<v>";
        for i = 0 to t.shape.(axis) - 1 do
          if i > 0 then Format.fprintf fmt ",@,";
          go (axis + 1) (offset + (i * st.(axis)))
        done;
        Format.fprintf fmt "@]"
      end;
      Format.fprintf fmt "]"
    in
    Format.fprintf fmt "@[<v>tensor %s@," (shape_to_string t.shape);
    go 0 0;
    Format.fprintf fmt "@]"
  end

(* ------------------------------------------------------------------------ *)
(* Indexing                                                                  *)
(* ------------------------------------------------------------------------ *)

(** The offset into [data] of a multi-index, with rank and bounds checks. *)
let flat_index t idx =
  if Array.length idx <> ndim t then
    invalid_arg
      (Printf.sprintf "Tensor.flat_index: expected %d indices, got %d" (ndim t)
         (Array.length idx));
  let st = strides t.shape in
  let offset = ref 0 in
  for i = 0 to Array.length idx - 1 do
    if 0 <= idx.(i) && idx.(i) < t.shape.(i) then
      offset := !offset + (idx.(i) * st.(i))
    else
      invalid_arg
        (Printf.sprintf
           "Tensor.flat_index: index %d out of bounds for axis %d (size %d)"
           idx.(i) i t.shape.(i))
  done;
  !offset

let get t idx = t.data.(flat_index t idx)
let set t idx v = t.data.(flat_index t idx) <- v

(* ------------------------------------------------------------------------ *)
(* Random initialisation                                                      *)
(* ------------------------------------------------------------------------ *)

(** Unwrap the optional generator, or make a fresh self-seeded one. *)
let get_rng = function
  | Some rng -> rng
  | None -> Random.State.make_self_init ()

(** Samples from U(lo, hi): one draw per element. *)
let rand_uniform ?rng ~lo ~hi shape =
  let rng = get_rng rng in
  let data =
    Array.init (numel_of_shape shape) (fun _ ->
        lo +. Random.State.float rng (hi -. lo))
  in
  { shape = Array.copy shape; data }

(** Samples from N(mean, std²) via Box-Muller: every pair of uniforms gives
    two independent standard normals. *)
let randn ?rng ?(mean = 0.0) ?(std = 1.0) shape =
  let rng = get_rng rng in
  let n = numel_of_shape shape in
  let data = Array.make n 0.0 in
  let i = ref 0 in
  while !i < n do
    (* Random.State.float rng 1.0 is in [0, 1), so 1 - that is in (0, 1]:
       never 0, so the log is safe *)
    let u1 = 1.0 -. Random.State.float rng 1.0 in
    let u2 = Random.State.float rng 1.0 in
    let r = Float.sqrt (-2.0 *. Float.log u1) in
    let theta = 2.0 *. Float.pi *. u2 in
    data.(!i) <- mean +. (std *. r *. Float.cos theta);
    (* the second sample of the pair; skip it if n is odd and we're at the end *)
    if !i + 1 < n then data.(!i + 1) <- mean +. (std *. r *. Float.sin theta);
    i := !i + 2
  done;
  { shape = Array.copy shape; data }

(* ------------------------------------------------------------------------ *)
(* Broadcasting                                                               *)
(* ------------------------------------------------------------------------ *)

(** NumPy rule: right-align the shapes; on each axis the sizes must be equal
    or one of them must be 1, and the result takes the larger. Walking both
    shapes from the last axis and prepending to [acc] builds the result in
    order. *)
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

(** Raise [Invalid_argument] (naming function [fn]) unless [small] broadcasts
    to exactly [big]. *)
let check_broadcastable ~fn ~small ~big =
  if broadcast_shapes small big <> big then
    invalid_arg
      (Printf.sprintf "Tensor.%s: %s does not broadcast to %s" fn
         (shape_to_string small) (shape_to_string big))

(** The strides for reading a tensor of shape [src] as if it had the larger
    shape [out]: its own strides, with 0 on every axis broadcasting repeats
    (missing leading axes and axes of size 1). Walking [out]'s multi-index
    with these strides visits the source element each position comes
    from. *)
let broadcast_strides ~src ~out =
  let k = Array.length out - Array.length src in
  let st = strides src in
  Array.init (Array.length out) (fun ax ->
      let s = ax - k in
      if s < 0 || src.(s) = 1 then 0 else st.(s))

(** Simplify a walk over [dims] in which operand [o] moves by [sts.(o)]:
    drop axes of size 1, and merge an axis into the one before it whenever
    every operand moves through the pair contiguously. The visiting order
    stays row-major, but the innermost axis becomes as long as possible, so
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

(** Visit rows [[lo, hi)] of a coalesced walk, where a row is one pass along
    the last axis. Before each call [f row], [offs.(o)] holds operand [o]'s
    offset at the start of that row. The multi-index over the outer axes is
    advanced like an odometer, so there is no division per element (or even
    per row). The row's length and inner strides are left to [f]. *)
let for_rows dims (sts : int array array) ~lo ~hi (offs : int array) f =
  let outer = Array.length dims - 1 and nops = Array.length sts in
  let idx = Array.make (max outer 1) 0 in
  let r = ref lo in
  for ax = outer - 1 downto 0 do
    idx.(ax) <- !r mod dims.(ax);
    r := !r / dims.(ax)
  done;
  for o = 0 to nops - 1 do
    let s = ref 0 in
    for ax = 0 to outer - 1 do
      s := !s + (idx.(ax) * sts.(o).(ax))
    done;
    offs.(o) <- !s
  done;
  for row = lo to hi - 1 do
    f row;
    let ax = ref (outer - 1) in
    while !ax >= 0 do
      let a = !ax in
      idx.(a) <- idx.(a) + 1;
      if idx.(a) < dims.(a) then begin
        for o = 0 to nops - 1 do
          offs.(o) <- offs.(o) + sts.(o).(a)
        done;
        ax := -1
      end
      else begin
        idx.(a) <- 0;
        for o = 0 to nops - 1 do
          offs.(o) <- offs.(o) - (sts.(o).(a) * (dims.(a) - 1))
        done;
        decr ax
      end
    done
  done

(** [out.(o + j) <- src.(i + j * s)] for [j < len]. *)
let copy_run (src : float array) i s (out : float array) o len =
  if s = 1 then
    for j = 0 to len - 1 do
      uset out (o + j) (uget src (i + j))
    done
  else
    for j = 0 to len - 1 do
      uset out (o + j) (uget src (i + (j * s)))
    done

(** Every output element reads the source element it maps to. *)
let broadcast_to t target =
  check_broadcastable ~fn:"broadcast_to" ~small:t.shape ~big:target;
  let n = numel_of_shape target in
  let out = Array.create_float n in
  if n > 0 then begin
    let dims, sts =
      coalesce target [| broadcast_strides ~src:t.shape ~out:target |]
    in
    let last = Array.length dims - 1 in
    let len = dims.(last) and s = sts.(0).(last) in
    let src = t.data in
    if Array.length dims = 2 && sts.(0).(0) = 0 && s = 1 then
      (* The bias case, [n] -> [m; n]: every row is a copy of [src]. It gets
         its own loop: measured, copying short rows from inside the generic
         [for_rows] callback below ran at about 8 ns per element against
         about 1 ns here. *)
      Parallel.parallel_for ~work:len (n / len) (fun lo hi ->
          for row = lo to hi - 1 do
            if len >= 64 then Array.blit src 0 out (row * len) len
            else copy_run src 0 1 out (row * len) len
          done)
    else
      Parallel.parallel_for ~work:len (n / len) (fun lo hi ->
          let offs = [| 0 |] in
          for_rows dims sts ~lo ~hi offs (fun row ->
              let o = row * len and i = offs.(0) in
              if s = 1 then Array.blit src i out o len
              else if s = 0 then Array.fill out o len (uget src i)
              else copy_run src i s out o len))
  end;
  { shape = Array.copy target; data = out }

(** A zeroed array of [n] floats followed by 16 unused ones (128 bytes, a
    cache line on Apple silicon), for per-task partial results. Partials
    allocated one after another would otherwise share cache lines, and
    tasks adding into neighbouring partials would keep stealing those lines
    from each other ("false sharing"). Measured: with unpadded partials the
    conv bias gradient got no faster on 8 threads than on 1. *)
let padded_partial n = Array.make (n + 16) 0.0

(** [dst.(j) += src.(r * len + j)] for rows [r] in [[lo, hi)]: the bias
    gradient, [[m; n] -> [n]], summing rows in order. *)
let add_rows_into_vector (src : float array) (dst : float array) len lo hi =
  for r = lo to hi - 1 do
    let base = r * len in
    for j = 0 to len - 1 do
      uset dst j (uget dst j +. uget src (base + j))
    done
  done

(** Rows of at least this many input elements are summed in one chunk by
    {!sum_to_shape}. Fixed, rather than derived from the thread count, so the
    order of the additions (and therefore the rounding) never depends on how
    many threads there are. *)
let reduction_chunk = 65536

(** The adjoint of [broadcast_to]: every element of [t] is added into the
    element of the (smaller) result it maps to.

    Small inputs are summed in row-major order. Large inputs that reduce to a
    small result (the common case: a bias gradient, [[50176; 8] -> [8]]) are
    cut into fixed-size chunks of rows. Each chunk is summed into its own
    partial result in parallel, and the partials are then added in chunk
    order. *)
let sum_to_shape t target =
  check_broadcastable ~fn:"sum_to_shape" ~small:target ~big:t.shape;
  let out = Array.make (numel_of_shape target) 0.0 in
  let n = Array.length t.data in
  if n > 0 then begin
    let dims, sts =
      coalesce t.shape
        [| strides t.shape; broadcast_strides ~src:target ~out:t.shape |]
    in
    let last = Array.length dims - 1 in
    let len = dims.(last) and so = sts.(1).(last) in
    let rows = n / len and src = t.data in
    (* t is contiguous, so its innermost coalesced stride is 1 *)
    (* [[m; n] -> [n]] (a bias gradient) gets a direct loop; like
       broadcast_to, it ran about twice as fast as the generic walk. *)
    let vector_case = Array.length dims = 2 && sts.(1).(0) = 0 && so = 1 in
    let add_rows (dst : float array) lo hi =
      if vector_case then add_rows_into_vector src dst len lo hi
      else begin
        let offs = [| 0; 0 |] in
        for_rows dims sts ~lo ~hi offs (fun _ ->
            let i = offs.(0) and o = offs.(1) in
            if so = 0 then begin
              let acc = ref (uget dst o) in
              for j = 0 to len - 1 do
                acc := !acc +. uget src (i + j)
              done;
              uset dst o !acc
            end
            else
              for j = 0 to len - 1 do
                let d = o + (j * so) in
                uset dst d (uget dst d +. uget src (i + j))
              done)
      end
    in
    let chunk_rows = max 1 (reduction_chunk / len) in
    let chunks = (rows + chunk_rows - 1) / chunk_rows in
    let m = Array.length out in
    if chunks <= 1 || m * chunks > n then add_rows out 0 rows
    else begin
      let partials = Array.init chunks (fun _ -> padded_partial m) in
      Parallel.parallel_for ~work:(chunk_rows * len) chunks (fun lo hi ->
          for c = lo to hi - 1 do
            add_rows partials.(c) (c * chunk_rows)
              (min rows ((c + 1) * chunk_rows))
          done);
      Array.iter
        (fun (p : float array) ->
          for i = 0 to m - 1 do
            uset out i (uget out i +. uget p i)
          done)
        partials
    end
  end;
  { shape = Array.copy target; data = out }

(* ------------------------------------------------------------------------ *)
(* Elementwise                                                                *)
(* ------------------------------------------------------------------------ *)

(** A new tensor of [t]'s shape, filled by [kernel src dst lo hi] over
    disjoint ranges, in parallel for large tensors. [work] is the rough cost
    per element in nanoseconds. Each operation passes its own [kernel], a
    closure called once per range, whose loop body is specialised code. *)
let unary ?(work = 1) t
    (kernel : float array -> float array -> int -> int -> unit) =
  let n = Array.length t.data in
  let out = Array.create_float n in
  Parallel.parallel_for ~work n (fun lo hi -> kernel t.data out lo hi);
  { shape = Array.copy t.shape; data = out }

(** General-purpose map. Each call to [f] boxes its result, so the
    arithmetic below uses dedicated loops instead. *)
let map f t = { shape = Array.copy t.shape; data = Array.map f t.data }

(** The arithmetic operations that {!binop} implements. *)
type binop = Add | Sub | Mul | Div

(** One row of a binary op: [o.(oo + i) <- a.(ao + i*sa) op b.(bo + i*sb)].
    Rows where both operands are contiguous get their own loops. *)
let binop_row op (o : float array) oo (a : float array) ao sa (b : float array)
    bo sb len =
  if sa = 1 && sb = 1 then
    match op with
    | Add ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + i) +. uget b (bo + i))
        done
    | Sub ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + i) -. uget b (bo + i))
        done
    | Mul ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + i) *. uget b (bo + i))
        done
    | Div ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + i) /. uget b (bo + i))
        done
  else
    match op with
    | Add ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + (i * sa)) +. uget b (bo + (i * sb)))
        done
    | Sub ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + (i * sa)) -. uget b (bo + (i * sb)))
        done
    | Mul ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + (i * sa)) *. uget b (bo + (i * sb)))
        done
    | Div ->
        for i = 0 to len - 1 do
          uset o (oo + i) (uget a (ao + (i * sa)) /. uget b (bo + (i * sb)))
        done

(** A broadcasting binary op. Equal shapes are one flat loop. Otherwise the
    walk over the output is coalesced, and each row reads both operands with
    their broadcast strides. *)
let binop op a b =
  if a.shape = b.shape then begin
    let n = Array.length a.data in
    let out = Array.create_float n in
    Parallel.parallel_for n (fun lo hi ->
        binop_row op out lo a.data lo 1 b.data lo 1 (hi - lo));
    { shape = Array.copy a.shape; data = out }
  end
  else begin
    let shape = broadcast_shapes a.shape b.shape in
    let n = numel_of_shape shape in
    let out = Array.create_float n in
    if n > 0 then begin
      let dims, sts =
        coalesce shape
          [|
            broadcast_strides ~src:a.shape ~out:shape;
            broadcast_strides ~src:b.shape ~out:shape;
          |]
      in
      let last = Array.length dims - 1 in
      let len = dims.(last) and sa = sts.(0).(last) and sb = sts.(1).(last) in
      Parallel.parallel_for ~work:len (n / len) (fun lo hi ->
          let offs = [| 0; 0 |] in
          for_rows dims sts ~lo ~hi offs (fun row ->
              binop_row op out (row * len) a.data offs.(0) sa b.data offs.(1)
                sb len))
    end;
    { shape; data = out }
  end

(** Equal shapes take a direct loop; otherwise both operands are read through
    their broadcast strides. [f] is called (and its result boxed) per element,
    so this is for general use, not hot paths. *)
let map2 f a b =
  if a.shape = b.shape then
    { shape = Array.copy a.shape; data = Array.map2 f a.data b.data }
  else begin
    let shape = broadcast_shapes a.shape b.shape in
    let n = numel_of_shape shape in
    let out = Array.create_float n in
    if n > 0 then begin
      let dims, sts =
        coalesce shape
          [|
            broadcast_strides ~src:a.shape ~out:shape;
            broadcast_strides ~src:b.shape ~out:shape;
          |]
      in
      let last = Array.length dims - 1 in
      let len = dims.(last) and sa = sts.(0).(last) and sb = sts.(1).(last) in
      let offs = [| 0; 0 |] in
      for_rows dims sts ~lo:0 ~hi:(n / len) offs (fun row ->
          for j = 0 to len - 1 do
            uset out
              ((row * len) + j)
              (f (uget a.data (offs.(0) + (j * sa)))
                 (uget b.data (offs.(1) + (j * sb))))
          done)
    end;
    { shape; data = out }
  end

let add a b = binop Add a b
let sub a b = binop Sub a b
let mul a b = binop Mul a b
let div a b = binop Div a b

let neg t =
  unary t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (-.uget a i)
      done)

let scale k t =
  unary t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (k *. uget a i)
      done)

let add_scalar k t =
  unary t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (uget a i +. k)
      done)

let pow t p =
  unary ~work:20 t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (Float.pow (uget a i) p)
      done)

let exp t =
  unary ~work:5 t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (Float.exp (uget a i))
      done)

let log t =
  unary ~work:5 t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (Float.log (uget a i))
      done)

let sqrt t =
  unary ~work:2 t (fun a o lo hi ->
      for i = lo to hi - 1 do
        uset o i (Float.sqrt (uget a i))
      done)

(** [1.0] if [x > 0], else [0.0], without a branch: the comparison becomes a
    0/1 integer and then a float. Activations are positive about half the
    time, essentially at random, so a branch here is mispredicted about half
    the time, which made relu 3x slower than an addition. *)
let[@inline] positive_mask (x : float) = Float.of_int (Bool.to_int (x > 0.0))

(** [x * mask + 0.0]: [x] when positive, otherwise [+0.0] (the [+ 0.0] turns
    the [-0.0] of a negative [x] times zero into [+0.0]). NaN stays NaN.
    The one difference from [max x 0] is that [-infinity] gives NaN. *)
let relu t =
  unary t (fun a o lo hi ->
      for i = lo to hi - 1 do
        let x = uget a i in
        uset o i ((x *. positive_mask x) +. 0.0)
      done)

(** Raise [Invalid_argument] (naming function [fn]) unless [a] and [b] have
    exactly the same shape. *)
let check_same_shape ~fn a b =
  if a.shape <> b.shape then
    invalid_arg
      (Printf.sprintf "Tensor.%s: shape mismatch %s vs %s" fn
         (shape_to_string a.shape) (shape_to_string b.shape))

let relu_backward ~input ~grad =
  check_same_shape ~fn:"relu_backward" input grad;
  let x = input.data and g = grad.data in
  let n = Array.length x in
  let out = Array.create_float n in
  Parallel.parallel_for n (fun lo hi ->
      for i = lo to hi - 1 do
        uset out i (uget g i *. positive_mask (uget x i))
      done);
  { shape = Array.copy input.shape; data = out }

(* ------------------------------------------------------------------------ *)
(* Reductions                                                                 *)
(* ------------------------------------------------------------------------ *)

let sum t =
  let s = ref 0.0 in
  for i = 0 to Array.length t.data - 1 do
    s := !s +. uget t.data i
  done;
  scalar !s

let mean t = scale (1.0 /. float_of_int (numel t)) (sum t)

(** Turn a possibly negative axis ([-1] = last) into a valid non-negative one,
    raising [Invalid_argument] (naming function [fn]) if it is out of range. *)
let normalize_axis ~fn t axis =
  let nd = ndim t in
  let a = if axis < 0 then axis + nd else axis in
  if a < 0 || a >= nd then
    invalid_arg
      (Printf.sprintf "Tensor.%s: axis %d out of range for shape %s" fn axis
         (shape_to_string t.shape));
  a

(** View the shape as [(outer, n, inner)] around [axis]. Element [(o, k, i)]
    is at [data.(o * n * inner + k * inner + i)]. *)
let split_around t axis =
  let outer = numel_of_shape (Array.sub t.shape 0 axis) in
  let n = t.shape.(axis) in
  let inner =
    numel_of_shape (Array.sub t.shape (axis + 1) (ndim t - axis - 1))
  in
  (outer, n, inner)

(** Shape after reducing [axis]: removed, or kept with size 1 if [keepdims]. *)
let reduced_shape ~keepdims t axis =
  if keepdims then Array.mapi (fun i d -> if i = axis then 1 else d) t.shape
  else
    Array.of_list (List.filteri (fun i _ -> i <> axis) (Array.to_list t.shape))

(** [Float.max] semantics (a NaN wins) without its signed-zero handling. *)
let[@inline] fmax (acc : float) (v : float) =
  if v > acc || v <> v then v else acc

(** Sum or max along [axis]. Every output element is folded over [k] in
    increasing order, by a single task: the same order as a plain sequential
    loop, so the result doesn't depend on the thread count. When the axis is
    the last one ([inner = 1]) each output is a contiguous run. Otherwise
    whole rows are combined into the output row. *)
let reduce_axis ~is_max ~keepdims t axis =
  let axis =
    normalize_axis ~fn:(if is_max then "max_axis" else "sum_axis") t axis
  in
  let outer, n, inner = split_around t axis in
  let init = if is_max then Float.neg_infinity else 0.0 in
  let out = Array.make (outer * inner) init in
  let x = t.data in
  let block o ilo ihi =
    let base = o * n * inner and ob = o * inner in
    if inner = 1 then begin
      let acc = ref init in
      if is_max then
        for k = 0 to n - 1 do
          acc := fmax !acc (uget x (base + k))
        done
      else
        for k = 0 to n - 1 do
          acc := !acc +. uget x (base + k)
        done;
      uset out ob !acc
    end
    else
      for k = 0 to n - 1 do
        let row = base + (k * inner) in
        if is_max then
          for i = ilo to ihi - 1 do
            uset out (ob + i) (fmax (uget out (ob + i)) (uget x (row + i)))
          done
        else
          for i = ilo to ihi - 1 do
            uset out (ob + i) (uget out (ob + i) +. uget x (row + i))
          done
      done
  in
  if outer > 1 then
    Parallel.parallel_for ~work:(n * inner) outer (fun lo hi ->
        for o = lo to hi - 1 do
          block o 0 inner
        done)
  else if outer = 1 && inner >= 1024 then
    (* Reducing the leading axis: tasks take disjoint column ranges. Each
       task reads every row, so this only pays off when there are enough
       columns per task; otherwise it stays sequential. *)
    Parallel.parallel_for ~work:n inner (fun lo hi -> block 0 lo hi)
  else if outer = 1 then block 0 0 inner;
  { shape = reduced_shape ~keepdims t axis; data = out }

let sum_axis ?(keepdims = false) t axis =
  reduce_axis ~is_max:false ~keepdims t axis

let max_axis ?(keepdims = false) t axis =
  reduce_axis ~is_max:true ~keepdims t axis

let argmax_axis t axis =
  let axis = normalize_axis ~fn:"argmax_axis" t axis in
  let outer, n, inner = split_around t axis in
  Array.init (outer * inner) (fun j ->
      let o = j / inner and i = j mod inner in
      let at k = t.data.((o * n * inner) + (k * inner) + i) in
      let best = ref 0 in
      for k = 1 to n - 1 do
        if at k > at !best then best := k
      done;
      !best)

(* ------------------------------------------------------------------------ *)
(* Matrix multiplication                                                      *)
(* ------------------------------------------------------------------------ *)

(* The structure follows the classic GotoBLAS/BLIS design, scaled down:

   - C is cut into tiles of up to [tile_m] x [tile_n]. Each tile is one
     parallel task that owns its part of C, so tasks never write to the same
     element.
   - A task walks k in blocks of [kc_max]. For each block it copies ("packs")
     its rows of op(A) and its columns of op(B) into contiguous scratch
     buffers, in exactly the order the micro-kernel reads them. Packing also
     absorbs the transposes, so all four combinations of [trans_a]/[trans_b]
     run the same inner loop and never materialise a transposed copy.
   - The 4x4 micro-kernel keeps a 4x4 block of C in 16 local float
     accumulators, which ocamlopt keeps unboxed in registers. Each step along
     k loads 4 values of A and 4 of B and does 16 multiply-adds, a quarter of
     the memory traffic of the old i-p-j loop.

   The tile size only decides which task computes which part of C; every
   element of C is still summed over k in the same order. So tiles may
   shrink (down to 16 x 16) when there are too few of them to keep every
   thread busy, without changing any result.

   When C is tiny and k is long (a convolution's weight gradient, such as
   [9 x 50176] x [50176 x 8]), there aren't enough tiles to share out, so k
   is cut into fixed chunks instead ("split-k"). Each chunk's partial product
   is computed in parallel, and the partials are added up in order. Both
   decisions depend only on the shapes, never on the thread count. *)

let kc_max = 256
let tile_m = 64
let tile_n = 64
let split_k_chunk = 2048

(** Per-domain packing buffers, grown on demand and reused across calls. *)
let pack_buffers : (float array ref * float array ref) Domain.DLS.key =
  Domain.DLS.new_key (fun () -> (ref [||], ref [||]))

let scratch (r : float array ref) size =
  if Array.length !r < size then r := Array.create_float size;
  !r

(** Pack rows [[i0, i0 + rows)] and k-range [[p0, p0 + kc)] of op(A) into
    panels of 4 rows: element [(i, p)] goes to
    [ap.((i / 4) * kc * 4 + (p - p0) * 4 + i mod 4)]. Rows past the end are
    zero, so the kernel can always compute a full 4x4 block.

    op(A)(i, p) is [a.(i * rs + p * ps)]: [rs = lda, ps = 1] for A as
    stored, and [rs = 1, ps = lda] for its transpose. A full panel is copied
    four values per step along k. With the short k of the convolution
    shapes ([k = 9]), per-element loop overhead would otherwise cost as much
    as the multiply itself. *)
let pack_a ~trans (a : float array) ~lda i0 rows p0 kc (ap : float array) =
  let rs, ps = if trans then (1, lda) else (lda, 1) in
  let panels = (rows + 3) / 4 in
  for q = 0 to panels - 1 do
    let base = q * kc * 4 and i = q * 4 in
    let start = ((i0 + i) * rs) + (p0 * ps) in
    if i + 4 <= rows then
      for p = 0 to kc - 1 do
        let src = start + (p * ps) and dst = base + (p * 4) in
        uset ap dst (uget a src);
        uset ap (dst + 1) (uget a (src + rs));
        uset ap (dst + 2) (uget a (src + (2 * rs)));
        uset ap (dst + 3) (uget a (src + (3 * rs)))
      done
    else
      for p = 0 to kc - 1 do
        let src = start + (p * ps) and dst = base + (p * 4) in
        for r = 0 to 3 do
          uset ap (dst + r)
            (if i + r < rows then uget a (src + (r * rs)) else 0.0)
        done
      done
  done

(** Pack columns [[j0, j0 + cols)] and k-range [[p0, p0 + kc)] of op(B) into
    panels of 4 columns: element [(p, j)] goes to
    [bp.((j / 4) * kc * 4 + (p - p0) * 4 + j mod 4)], zero-padded.
    op(B)(p, j) is [b.(p * ps + j * cs)]: [ps = ldb, cs = 1] as stored,
    [ps = 1, cs = ldb] transposed. *)
let pack_b ~trans (b : float array) ~ldb j0 cols p0 kc (bp : float array) =
  let ps, cs = if trans then (1, ldb) else (ldb, 1) in
  let panels = (cols + 3) / 4 in
  for q = 0 to panels - 1 do
    let base = q * kc * 4 and j = q * 4 in
    let start = (p0 * ps) + ((j0 + j) * cs) in
    if j + 4 <= cols then
      for p = 0 to kc - 1 do
        let src = start + (p * ps) and dst = base + (p * 4) in
        uset bp dst (uget b src);
        uset bp (dst + 1) (uget b (src + cs));
        uset bp (dst + 2) (uget b (src + (2 * cs)));
        uset bp (dst + 3) (uget b (src + (3 * cs)))
      done
    else
      for p = 0 to kc - 1 do
        let src = start + (p * ps) and dst = base + (p * 4) in
        for c = 0 to 3 do
          uset bp (dst + c)
            (if j + c < cols then uget b (src + (c * cs)) else 0.0)
        done
      done
  done

(** [C[ci.., cj..] += alpha * (4 x kc panel of A) (kc x 4 panel of B)],
    writing only the top-left [mr] x [nr] corner (smaller at the edges of
    C). *)
let kernel_4x4 kc (ap : float array) ao (bp : float array) bo (c : float array)
    ldc ci cj mr nr alpha =
  let c00 = ref 0. and c01 = ref 0. and c02 = ref 0. and c03 = ref 0. in
  let c10 = ref 0. and c11 = ref 0. and c12 = ref 0. and c13 = ref 0. in
  let c20 = ref 0. and c21 = ref 0. and c22 = ref 0. and c23 = ref 0. in
  let c30 = ref 0. and c31 = ref 0. and c32 = ref 0. and c33 = ref 0. in
  for p = 0 to kc - 1 do
    let a = ao + (4 * p) and b = bo + (4 * p) in
    let a0 = uget ap a and a1 = uget ap (a + 1) in
    let a2 = uget ap (a + 2) and a3 = uget ap (a + 3) in
    let b0 = uget bp b and b1 = uget bp (b + 1) in
    let b2 = uget bp (b + 2) and b3 = uget bp (b + 3) in
    c00 := !c00 +. (a0 *. b0);
    c01 := !c01 +. (a0 *. b1);
    c02 := !c02 +. (a0 *. b2);
    c03 := !c03 +. (a0 *. b3);
    c10 := !c10 +. (a1 *. b0);
    c11 := !c11 +. (a1 *. b1);
    c12 := !c12 +. (a1 *. b2);
    c13 := !c13 +. (a1 *. b3);
    c20 := !c20 +. (a2 *. b0);
    c21 := !c21 +. (a2 *. b1);
    c22 := !c22 +. (a2 *. b2);
    c23 := !c23 +. (a2 *. b3);
    c30 := !c30 +. (a3 *. b0);
    c31 := !c31 +. (a3 *. b1);
    c32 := !c32 +. (a3 *. b2);
    c33 := !c33 +. (a3 *. b3)
  done;
  let r0 = (ci * ldc) + cj in
  if mr = 4 && nr = 4 then begin
    let r1 = r0 + ldc in
    let r2 = r1 + ldc in
    let r3 = r2 + ldc in
    uset c r0 (uget c r0 +. (alpha *. !c00));
    uset c (r0 + 1) (uget c (r0 + 1) +. (alpha *. !c01));
    uset c (r0 + 2) (uget c (r0 + 2) +. (alpha *. !c02));
    uset c (r0 + 3) (uget c (r0 + 3) +. (alpha *. !c03));
    uset c r1 (uget c r1 +. (alpha *. !c10));
    uset c (r1 + 1) (uget c (r1 + 1) +. (alpha *. !c11));
    uset c (r1 + 2) (uget c (r1 + 2) +. (alpha *. !c12));
    uset c (r1 + 3) (uget c (r1 + 3) +. (alpha *. !c13));
    uset c r2 (uget c r2 +. (alpha *. !c20));
    uset c (r2 + 1) (uget c (r2 + 1) +. (alpha *. !c21));
    uset c (r2 + 2) (uget c (r2 + 2) +. (alpha *. !c22));
    uset c (r2 + 3) (uget c (r2 + 3) +. (alpha *. !c23));
    uset c r3 (uget c r3 +. (alpha *. !c30));
    uset c (r3 + 1) (uget c (r3 + 1) +. (alpha *. !c31));
    uset c (r3 + 2) (uget c (r3 + 2) +. (alpha *. !c32));
    uset c (r3 + 3) (uget c (r3 + 3) +. (alpha *. !c33))
  end
  else begin
    (* An edge block: copy the accumulators out and write the valid part. *)
    let acc =
      [|
        !c00; !c01; !c02; !c03; !c10; !c11; !c12; !c13;
        !c20; !c21; !c22; !c23; !c30; !c31; !c32; !c33;
      |]
    in
    for r = 0 to mr - 1 do
      for col = 0 to nr - 1 do
        let i = r0 + (r * ldc) + col in
        uset c i (uget c i +. (alpha *. acc.((r * 4) + col)))
      done
    done
  end

(** [C[r0..r1, c0..c1] += alpha * op(A)[r0..r1, k0..k1] op(B)[k0..k1, c0..c1]]
    for one tile, in k-blocks of [kc_max]. *)
let gemm_tile ~ta ~tb (a : float array) lda (b : float array) ldb
    (c : float array) ldc alpha ~k0 ~k1 ~r0 ~r1 ~c0 ~c1 =
  let ra, rb = Domain.DLS.get pack_buffers in
  let rows = r1 - r0 and cols = c1 - c0 in
  let row_panels = (rows + 3) / 4 and col_panels = (cols + 3) / 4 in
  let ap = scratch ra (row_panels * 4 * kc_max) in
  let bp = scratch rb (col_panels * 4 * kc_max) in
  let p0 = ref k0 in
  while !p0 < k1 do
    let kc = min kc_max (k1 - !p0) in
    pack_a ~trans:ta a ~lda r0 rows !p0 kc ap;
    pack_b ~trans:tb b ~ldb c0 cols !p0 kc bp;
    for qj = 0 to col_panels - 1 do
      let nr = min 4 (cols - (qj * 4)) in
      for qi = 0 to row_panels - 1 do
        kernel_4x4 kc ap (qi * kc * 4) bp (qj * kc * 4) c ldc
          (r0 + (qi * 4))
          (c0 + (qj * 4))
          (min 4 (rows - (qi * 4)))
          nr alpha
      done
    done;
    p0 := !p0 + kc
  done

let scale_ k t =
  let d = t.data in
  Parallel.parallel_for (Array.length d) (fun lo hi ->
      for i = lo to hi - 1 do
        uset d i (k *. uget d i)
      done)

let gemm_ ?(trans_a = false) ?(trans_b = false) ?(alpha = 1.0) ?(beta = 0.0) a
    b ~c =
  let fail () =
    invalid_arg
      (Printf.sprintf "Tensor.gemm_: cannot compute %s%s x %s%s into %s"
         (shape_to_string a.shape)
         (if trans_a then "^T" else "")
         (shape_to_string b.shape)
         (if trans_b then "^T" else "")
         (shape_to_string c.shape))
  in
  if ndim a <> 2 || ndim b <> 2 || ndim c <> 2 then fail ();
  if c.data == a.data || c.data == b.data then
    invalid_arg "Tensor.gemm_: c must not share storage with a or b";
  let m, k =
    if trans_a then (a.shape.(1), a.shape.(0)) else (a.shape.(0), a.shape.(1))
  in
  let kb, n =
    if trans_b then (b.shape.(1), b.shape.(0)) else (b.shape.(0), b.shape.(1))
  in
  if k <> kb || c.shape.(0) <> m || c.shape.(1) <> n then fail ();
  if beta = 0.0 then Array.fill c.data 0 (m * n) 0.0
  else if beta <> 1.0 then scale_ beta c;
  if m > 0 && n > 0 && k > 0 then begin
    let lda = a.shape.(1) and ldb = b.shape.(1) in
    let tile_m = ref tile_m and tile_n = ref tile_n in
    let count () =
      ((m + !tile_m - 1) / !tile_m) * ((n + !tile_n - 1) / !tile_n)
    in
    let wanted = 2 * Parallel.num_threads () in
    (* An axis can usefully shrink while its tile is above 16 and the matrix
       is longer than half a tile along it. *)
    let can_m () = !tile_m > 16 && m > !tile_m / 2
    and can_n () = !tile_n > 16 && n > !tile_n / 2 in
    while count () < wanted && (can_m () || can_n ()) do
      if can_m () && (!tile_m >= !tile_n || not (can_n ())) then
        tile_m := !tile_m / 2
      else tile_n := !tile_n / 2
    done;
    let tile_m = !tile_m and tile_n = !tile_n in
    let tiles_m = (m + tile_m - 1) / tile_m
    and tiles_n = (n + tile_n - 1) / tile_n in
    let tiles = tiles_m * tiles_n in
    let tile ~k0 ~k1 (dst : float array) alpha t =
      let ti = t / tiles_n and tj = t mod tiles_n in
      gemm_tile ~ta:trans_a ~tb:trans_b a.data lda b.data ldb dst n alpha ~k0
        ~k1 ~r0:(ti * tile_m)
        ~r1:(min m ((ti + 1) * tile_m))
        ~c0:(tj * tile_n)
        ~c1:(min n ((tj + 1) * tile_n))
    in
    let default_tiles =
      ((m + 63) / 64) * ((n + 63) / 64)
    in
    if default_tiles < 4 && k >= 2 * split_k_chunk then begin
      let chunks = (k + split_k_chunk - 1) / split_k_chunk in
      let partials = Array.init chunks (fun _ -> padded_partial (m * n)) in
      Parallel.parallel_for
        ~work:(m * n * split_k_chunk / 4)
        chunks
        (fun lo hi ->
          for ch = lo to hi - 1 do
            let k0 = ch * split_k_chunk in
            let k1 = min k (k0 + split_k_chunk) in
            for t = 0 to tiles - 1 do
              tile ~k0 ~k1 partials.(ch) 1.0 t
            done
          done);
      let d = c.data in
      for i = 0 to (m * n) - 1 do
        let s = ref 0.0 in
        for ch = 0 to chunks - 1 do
          s := !s +. uget partials.(ch) i
        done;
        uset d i (uget d i +. (alpha *. !s))
      done
    end
    else
      let work = min m tile_m * min n tile_n * k / 4 in
      Parallel.parallel_for ~work tiles (fun lo hi ->
          for t = lo to hi - 1 do
            tile ~k0:0 ~k1:k c.data alpha t
          done)
  end

let matmul a b =
  if ndim a <> 2 || ndim b <> 2 || a.shape.(1) <> b.shape.(0) then
    invalid_arg
      (Printf.sprintf "Tensor.matmul: cannot multiply %s by %s"
         (shape_to_string a.shape) (shape_to_string b.shape));
  let m = a.shape.(0) and n = b.shape.(1) in
  let c = { shape = [| m; n |]; data = Array.make (m * n) 0.0 } in
  gemm_ ~beta:1.0 a b ~c;
  c

(* ------------------------------------------------------------------------ *)
(* Shape manipulation                                                         *)
(* ------------------------------------------------------------------------ *)

(** Copied in 32 x 32 blocks, so both the reads and the writes stay within a
    few cache lines at a time. *)
let transpose t =
  if ndim t <> 2 then
    invalid_arg
      (Printf.sprintf "Tensor.transpose: expected a 2-D tensor, got %s"
         (shape_to_string t.shape));
  let m = t.shape.(0) and n = t.shape.(1) in
  let out = Array.create_float (m * n) and src = t.data in
  let bs = 32 in
  let i0 = ref 0 in
  while !i0 < m do
    let j0 = ref 0 in
    while !j0 < n do
      for i = !i0 to min m (!i0 + bs) - 1 do
        for j = !j0 to min n (!j0 + bs) - 1 do
          uset out ((j * m) + i) (uget src ((i * n) + j))
        done
      done;
      j0 := !j0 + bs
    done;
    i0 := !i0 + bs
  done;
  { shape = [| n; m |]; data = out }

(** Supports one [-1] entry, whose size is inferred from the element count. *)
let reshape t new_shape =
  let fail () =
    invalid_arg
      (Printf.sprintf "Tensor.reshape: cannot reshape %s into %s"
         (shape_to_string t.shape)
         (shape_to_string new_shape))
  in
  let new_shape = Array.copy new_shape in
  let unknown =
    Array.to_list new_shape |> List.filter (fun d -> d = -1) |> List.length
  in
  if unknown > 1 then fail ();
  if unknown = 1 then begin
    let known =
      Array.fold_left (fun acc d -> if d = -1 then acc else acc * d) 1 new_shape
    in
    if known = 0 || numel t mod known <> 0 then fail ();
    Array.iteri
      (fun i d -> if d = -1 then new_shape.(i) <- numel t / known)
      new_shape
  end;
  if numel_of_shape new_shape <> numel t then fail ();
  { shape = new_shape; data = Array.copy t.data }

let take_rows t idx =
  if ndim t = 0 then invalid_arg "Tensor.take_rows: scalar tensor";
  let rows = t.shape.(0) in
  let row_size = if rows = 0 then 0 else numel t / rows in
  let out = Array.make (Array.length idx * row_size) 0.0 in
  Array.iteri
    (fun r i ->
      if i < 0 || i >= rows then
        invalid_arg
          (Printf.sprintf "Tensor.take_rows: row %d out of range (%d rows)" i
             rows);
      Array.blit t.data (i * row_size) out (r * row_size) row_size)
    idx;
  let shape = Array.copy t.shape in
  shape.(0) <- Array.length idx;
  { shape; data = out }

let take_rows_ t idx ~dst =
  if ndim t = 0 then invalid_arg "Tensor.take_rows_: scalar tensor";
  let expected = Array.copy t.shape in
  expected.(0) <- Array.length idx;
  if dst.shape <> expected then
    invalid_arg
      (Printf.sprintf "Tensor.take_rows_: dst has shape %s, expected %s"
         (shape_to_string dst.shape) (shape_to_string expected));
  let rows = t.shape.(0) in
  let row_size = if rows = 0 then 0 else numel t / rows in
  Array.iteri
    (fun r i ->
      if i < 0 || i >= rows then
        invalid_arg
          (Printf.sprintf "Tensor.take_rows_: row %d out of range (%d rows)" i
             rows);
      Array.blit t.data (i * row_size) dst.data (r * row_size) row_size)
    idx

let one_hot ~num_classes labels =
  let n = Array.length labels in
  let data = Array.make (n * num_classes) 0.0 in
  Array.iteri
    (fun i label ->
      if label < 0 || label >= num_classes then
        invalid_arg
          (Printf.sprintf "Tensor.one_hot: label %d not in [0, %d)" label
             num_classes);
      data.((i * num_classes) + label) <- 1.0)
    labels;
  { shape = [| n; num_classes |]; data }

(* ------------------------------------------------------------------------ *)
(* Softmax                                                                    *)
(* ------------------------------------------------------------------------ *)

(** [(rows, n)] for a tensor viewed as rows along its last axis. *)
let last_axis_rows ~fn t =
  if ndim t = 0 then invalid_arg (Printf.sprintf "Tensor.%s: scalar tensor" fn);
  let n = t.shape.(ndim t - 1) in
  ((if n = 0 then 0 else numel t / n), n)

(** One pass per row: the row maximum [m], then
    [lse = log (sum (exp (x - m)))], then [x - m - lse]. These are the same
    operations, in the same order, as composing [max_axis], [sub], [exp],
    [sum_axis] and [log], but without the five intermediate tensors. *)
let log_softmax t =
  let rows, n = last_axis_rows ~fn:"log_softmax" t in
  let x = t.data in
  let out = Array.create_float (Array.length x) in
  Parallel.parallel_for ~work:(n * 12) rows (fun lo hi ->
      for r = lo to hi - 1 do
        let base = r * n in
        let m = ref Float.neg_infinity in
        for j = 0 to n - 1 do
          m := fmax !m (uget x (base + j))
        done;
        let m = !m in
        let s = ref 0.0 in
        for j = 0 to n - 1 do
          s := !s +. Float.exp (uget x (base + j) -. m)
        done;
        let lse = Float.log !s in
        for j = 0 to n - 1 do
          uset out (base + j) (uget x (base + j) -. m -. lse)
        done
      done);
  { shape = Array.copy t.shape; data = out }

(** [g - exp out * sum_row g], row by row. *)
let log_softmax_backward ~output ~grad =
  check_same_shape ~fn:"log_softmax_backward" output grad;
  let rows, n = last_axis_rows ~fn:"log_softmax_backward" output in
  let y = output.data and g = grad.data in
  let out = Array.create_float (Array.length y) in
  Parallel.parallel_for ~work:(n * 8) rows (fun lo hi ->
      for r = lo to hi - 1 do
        let base = r * n in
        let s = ref 0.0 in
        for j = 0 to n - 1 do
          s := !s +. uget g (base + j)
        done;
        let s = !s in
        for j = 0 to n - 1 do
          uset out (base + j)
            (uget g (base + j) -. (Float.exp (uget y (base + j)) *. s))
        done
      done);
  { shape = Array.copy output.shape; data = out }

(** Check a [[| batch; classes |]] logits matrix against one label per row. *)
let check_labels ~fn t labels =
  if ndim t <> 2 then
    invalid_arg
      (Printf.sprintf "Tensor.%s: expected [batch; classes] logits, got %s" fn
         (shape_to_string t.shape));
  let b = t.shape.(0) and c = t.shape.(1) in
  if Array.length labels <> b then
    invalid_arg
      (Printf.sprintf "Tensor.%s: %d labels for a batch of %d" fn
         (Array.length labels) b);
  Array.iter
    (fun l ->
      if l < 0 || l >= c then
        invalid_arg
          (Printf.sprintf "Tensor.%s: label %d not in [0, %d)" fn l c))
    labels;
  (b, c)

let softmax_cross_entropy logits labels =
  let b, c = check_labels ~fn:"softmax_cross_entropy" logits labels in
  let lp = log_softmax logits in
  let s = ref 0.0 in
  for r = 0 to b - 1 do
    s := !s +. uget lp.data ((r * c) + labels.(r))
  done;
  (-1.0 /. float_of_int b *. !s, lp)

let softmax_cross_entropy_backward ~log_probs ~labels ~scale =
  let b, c =
    check_labels ~fn:"softmax_cross_entropy_backward" log_probs labels
  in
  let lp = log_probs.data in
  let out = Array.create_float (b * c) in
  Parallel.parallel_for ~work:(c * 8) b (fun lo hi ->
      for r = lo to hi - 1 do
        let label = Array.unsafe_get labels r in
        for j = 0 to c - 1 do
          let i = (r * c) + j in
          let y = if j = label then scale else 0.0 in
          uset out i (y -. (Float.exp (uget lp i) *. scale))
        done
      done);
  { shape = [| b; c |]; data = out }

(* ------------------------------------------------------------------------ *)
(* In-place operations                                                        *)
(* ------------------------------------------------------------------------ *)

(** [y <- y + x] *)
let add_ y x =
  check_same_shape ~fn:"add_" y x;
  let yd = y.data and xd = x.data in
  Parallel.parallel_for (Array.length yd) (fun lo hi ->
      for i = lo to hi - 1 do
        uset yd i (uget yd i +. uget xd i)
      done)

(** [y <- y + alpha * x] *)
let axpy_ ~alpha ~x ~y =
  check_same_shape ~fn:"axpy_" y x;
  let yd = y.data and xd = x.data in
  Parallel.parallel_for (Array.length yd) (fun lo hi ->
      for i = lo to hi - 1 do
        uset yd i (uget yd i +. (alpha *. uget xd i))
      done)

let fill_ t x = Array.fill t.data 0 (numel t) x

let blit_ ~src ~dst =
  check_same_shape ~fn:"blit_" dst src;
  Array.blit src.data 0 dst.data 0 (numel src)

let check_all_same ~fn first rest =
  List.iter (fun t -> check_same_shape ~fn first t) rest

(** One fused pass per element, with no temporaries, doing the same
    arithmetic as the step-by-step version. Results can differ from it in
    the last bit: on arm64, ocamlopt turns [a *. b +. c] into one
    fused multiply-add (rounded once), and which product it fuses depends on
    how the expression is written. The formulas:
    [g' = g + wd * p], [m <- b1 * m + (1 - b1) * g'],
    [v <- b2 * v + (1 - b2) * g' * g'],
    [p <- p + (-lr) * (m / bc1 / (sqrt (v / bc2) + eps))]. *)
let adam_update_ ~lr ~beta1 ~beta2 ~eps ~weight_decay ~bc1 ~bc2 ~param ~grad
    ~m ~v =
  check_all_same ~fn:"adam_update_" param [ grad; m; v ];
  let p = param.data and g = grad.data and md = m.data and vd = v.data in
  let c1 = 1. -. beta1 and c2 = 1. -. beta2 and neg_lr = -.lr in
  let decay = weight_decay > 0. in
  Parallel.parallel_for ~work:6 (Array.length p) (fun lo hi ->
      for i = lo to hi - 1 do
        let pi = uget p i in
        let gi = uget g i in
        let gi = if decay then gi +. (weight_decay *. pi) else gi in
        let mi = (beta1 *. uget md i) +. (c1 *. gi) in
        let vi = (beta2 *. uget vd i) +. (c2 *. (gi *. gi)) in
        uset md i mi;
        uset vd i vi;
        let u = mi /. bc1 /. (Float.sqrt (vi /. bc2) +. eps) in
        uset p i (pi +. (neg_lr *. u))
      done)

(** As {!adam_update_}: [g' = g + wd * p], then either [p <- p + (-lr) * g']
    or [v <- momentum * v + g'] and [p <- p + (-lr) * v]. *)
let sgd_update_ ~lr ~momentum ~weight_decay ~param ~grad ~velocity =
  check_all_same ~fn:"sgd_update_" param [ grad; velocity ];
  let p = param.data and g = grad.data and vd = velocity.data in
  let neg_lr = -.lr and decay = weight_decay > 0. in
  Parallel.parallel_for ~work:2 (Array.length p) (fun lo hi ->
      for i = lo to hi - 1 do
        let pi = uget p i in
        let gi = uget g i in
        let gi = if decay then gi +. (weight_decay *. pi) else gi in
        if momentum = 0. then uset p i (pi +. (neg_lr *. gi))
        else begin
          let vi = (momentum *. uget vd i) +. gi in
          uset vd i vi;
          uset p i (pi +. (neg_lr *. vi))
        end
      done)

(* ------------------------------------------------------------------------ *)
(* Convolution helpers                                                        *)
(* ------------------------------------------------------------------------ *)

(** How many positions a window of size [kernel] takes along an axis of
    length [size], padded by [padding] on each side and moved by [stride]. *)
let output_size ~size ~kernel ~stride ~padding =
  ((size + (2 * padding) - kernel) / stride) + 1

(** [(n, h, w, c, oh, ow)] for an NHWC input and a [kh] x [kw] window, after
    the checks every convolution kernel needs before it may use unchecked
    access: the window must fit inside the padded input. *)
let conv_geometry ~fn input_shape ~kh ~kw ~stride ~padding =
  if Array.length input_shape <> 4 then
    invalid_arg
      (Printf.sprintf "Tensor.%s: expected an NHWC tensor [n; h; w; c]" fn);
  if kh <= 0 || kw <= 0 || stride <= 0 || padding < 0 then
    invalid_arg (Printf.sprintf "Tensor.%s: bad kernel, stride or padding" fn);
  let n = input_shape.(0)
  and h = input_shape.(1)
  and w = input_shape.(2)
  and c = input_shape.(3) in
  if h + (2 * padding) < kh || w + (2 * padding) < kw then
    invalid_arg
      (Printf.sprintf "Tensor.%s: %dx%d window larger than the padded input"
         fn kh kw);
  let oh = output_size ~size:h ~kernel:kh ~stride ~padding
  and ow = output_size ~size:w ~kernel:kw ~stride ~padding in
  (n, h, w, c, oh, ow)

(** The window walk shared by [im2col] (copy in) and [col2im] (add back),
    for one image [b]. For each output row [oy] and kernel row [ky] (skipped
    when it falls on padding), the output pixels whose whole window row lies
    inside the image are the interior: each copies [kw * c] contiguous floats
    between the image and the column matrix, in a tight loop with all the
    bounds work hoisted out. The few edge pixels clip their run to the
    image. *)
let im2col_image ~scatter (img : float array) (cols : float array) ~b ~h ~w ~c
    ~kh ~kw ~stride ~padding ~oh ~ow =
  let ncols = kh * kw * c and run = kw * c in
  (* interior output columns: 0 <= ox*stride - padding and
     ox*stride - padding + kw <= w *)
  let ox_lo = min ow ((padding + stride - 1) / stride) in
  let ox_hi = max ox_lo (min ow (((w - kw + padding) / stride) + 1)) in
  let copy s d len =
    if scatter then
      for i = 0 to len - 1 do
        uset img (s + i) (uget img (s + i) +. uget cols (d + i))
      done
    else
      for i = 0 to len - 1 do
        uset cols (d + i) (uget img (s + i))
      done
  in
  let edge ~oy ~ky ~y ox =
    let x0 = (ox * stride) - padding in
    let kx_lo = max 0 (-x0) and kx_hi = min kw (w - x0) in
    if kx_lo < kx_hi then
      copy
        (((((b * h) + y) * w) + x0 + kx_lo) * c)
        ((((((b * oh) + oy) * ow) + ox) * ncols) + (((ky * kw) + kx_lo) * c))
        ((kx_hi - kx_lo) * c)
  in
  for oy = 0 to oh - 1 do
    for ky = 0 to kh - 1 do
      let y = (oy * stride) + ky - padding in
      if y >= 0 && y < h then begin
        for ox = 0 to ox_lo - 1 do
          edge ~oy ~ky ~y ox
        done;
        let row_base = (((b * h) + y) * w) - padding in
        let col_base = ((((b * oh) + oy) * ow) * ncols) + (ky * run) in
        for ox = ox_lo to ox_hi - 1 do
          let s = (row_base + (ox * stride)) * c
          and d = col_base + (ox * ncols) in
          if scatter then
            for i = 0 to run - 1 do
              uset img (s + i) (uget img (s + i) +. uget cols (d + i))
            done
          else
            for i = 0 to run - 1 do
              uset cols (d + i) (uget img (s + i))
            done
        done;
        for ox = ox_hi to ow - 1 do
          edge ~oy ~ky ~y ox
        done
      end
    done
  done

(** Padding positions are never visited, so they stay 0. *)
let im2col t ~kh ~kw ~stride ~padding =
  let n, h, w, c, oh, ow =
    conv_geometry ~fn:"im2col" t.shape ~kh ~kw ~stride ~padding
  in
  let ncols = kh * kw * c in
  let out = Array.make (n * oh * ow * ncols) 0.0 in
  Parallel.parallel_for ~work:(oh * ow * ncols) n (fun lo hi ->
      for b = lo to hi - 1 do
        im2col_image ~scatter:false t.data out ~b ~h ~w ~c ~kh ~kw ~stride
          ~padding ~oh ~ow
      done);
  { shape = [| n * oh * ow; ncols |]; data = out }

(** Same walk as [im2col], with the data flowing the other way and
    accumulating. Images are independent, so each task owns whole images. *)
let col2im t ~input_shape ~kh ~kw ~stride ~padding =
  let n, h, w, c, oh, ow =
    conv_geometry ~fn:"col2im" input_shape ~kh ~kw ~stride ~padding
  in
  let shape = [| n * oh * ow; kh * kw * c |] in
  if shape <> t.shape then
    invalid_arg
      (Printf.sprintf "Tensor.col2im: expected columns of shape %s, got %s"
         (shape_to_string shape) (shape_to_string t.shape));
  let out = Array.make (numel_of_shape input_shape) 0.0 in
  Parallel.parallel_for ~work:(oh * ow * kh * kw * c) n (fun lo hi ->
      for b = lo to hi - 1 do
        im2col_image ~scatter:true out t.data ~b ~h ~w ~c ~kh ~kw ~stride
          ~padding ~oh ~ow
      done);
  { shape = Array.copy input_shape; data = out }

(** Each output starts at the window's first element and moves to any
    strictly larger value, visiting the window row by row, so ties go to the
    first maximum. The winner's index is chosen without a branch
    ([arg + (v > best) * (i - arg)]) and its value is read back from the
    input, which keeps it exact. Comparisons of activations are close to
    random, so a branch would be mispredicted often. 2x2 windows, the usual
    case, are unrolled. If a window starts with NaN, the NaN wins. *)
let max_pool2d t ~size ~stride =
  let n, h, w, c, oh, ow =
    conv_geometry ~fn:"max_pool2d" t.shape ~kh:size ~kw:size ~stride ~padding:0
  in
  let total = n * oh * ow * c in
  let out = Array.create_float total
  and argmax = Array.make total 0
  and x = t.data in
  let row = w * c in
  Parallel.parallel_for ~work:(oh * ow * c * size * size) n (fun lo hi ->
      for b = lo to hi - 1 do
        for oy = 0 to oh - 1 do
          for ox = 0 to ow - 1 do
            let o = ((((b * oh) + oy) * ow) + ox) * c in
            let corner =
              ((((b * h) + (oy * stride)) * w) + (ox * stride)) * c
            in
            for ch = 0 to c - 1 do
              let i0 = corner + ch in
              let arg = ref i0 in
              if size = 2 then begin
                let i1 = i0 + c and i2 = i0 + row in
                let i3 = i2 + c in
                let a = !arg in
                let a = a + (Bool.to_int (uget x i1 > uget x a) * (i1 - a)) in
                let a = a + (Bool.to_int (uget x i2 > uget x a) * (i2 - a)) in
                let a = a + (Bool.to_int (uget x i3 > uget x a) * (i3 - a)) in
                arg := a
              end
              else
                for ky = 0 to size - 1 do
                  for kx = 0 to size - 1 do
                    let i = i0 + (ky * row) + (kx * c) and a = !arg in
                    arg := a + (Bool.to_int (uget x i > uget x a) * (i - a))
                  done
                done;
              uset out (o + ch) (uget x !arg);
              Array.unsafe_set argmax (o + ch) !arg
            done
          done
        done
      done);
  ({ shape = [| n; oh; ow; c |]; data = out }, argmax)

(** Only the winning input of each window gets gradient. Indices are
    bounds-checked, since [argmax] comes from the caller. When [grad_out] is
    NHWC with the input's batch size, the winners of image [b] all lie in
    image [b], so images are processed in parallel. *)
let max_pool2d_backward ~grad_out ~argmax ~input_shape =
  if Array.length argmax <> numel grad_out then
    invalid_arg "Tensor.max_pool2d_backward: argmax and grad_out sizes differ";
  let out = Array.make (numel_of_shape input_shape) 0.0 and g = grad_out.data in
  let scatter lo hi =
    for o = lo to hi - 1 do
      let i = argmax.(o) in
      out.(i) <- out.(i) +. uget g o
    done
  in
  let n = if Array.length input_shape > 0 then input_shape.(0) else 0 in
  if ndim grad_out = 4 && grad_out.shape.(0) = n && n > 0 then begin
    let per_image = numel grad_out / n in
    Parallel.parallel_for ~work:per_image n (fun lo hi ->
        scatter (lo * per_image) (hi * per_image))
  end
  else scatter 0 (Array.length argmax);
  { shape = Array.copy input_shape; data = out }

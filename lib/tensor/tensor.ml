(* Default backend: a contiguous row-major float array plus a shape.

   The element at multi-index [| i0; i1; ...; ik |] lives at
   data.(i0 * strides.(0) + i1 * strides.(1) + ... + ik * strides.(k)).

   Note: this file later defines [exp], [log], [sqrt] and [neg], which shadow
   the Stdlib versions. Use [Float.exp], [Float.log], [Float.sqrt] etc. inside
   helper code to avoid confusing type errors. *)

type t = { shape : int array; data : float array }

(* ------------------------------------------------------------------------ *)
(* Construction and inspection (provided)                                    *)
(* ------------------------------------------------------------------------ *)

let numel_of_shape shape = Array.fold_left ( * ) 1 shape

let shape_to_string shape =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int shape)) ^ "]"

(* strides [|2; 3; 4|] = [|12; 4; 1|] *)
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

(* Multi-index -> offset into [data]. *)
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

(* Unwrap the optional generator, or make a fresh self-seeded one. *)
let get_rng = function
  | Some rng -> rng
  | None -> Random.State.make_self_init ()

(* Samples from U(lo, hi): one draw per element. *)
let rand_uniform ?rng ~lo ~hi shape =
  let rng = get_rng rng in
  let data =
    Array.init (numel_of_shape shape) (fun _ ->
        lo +. Random.State.float rng (hi -. lo))
  in
  { shape = Array.copy shape; data }

(* Samples from N(mean, std^2) via Box-Muller: every pair of uniforms gives
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

(* NumPy rule: right-align the shapes; on each axis the sizes must be equal or
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

(* The core broadcasting trick, shared by broadcast_to, sum_to_shape and map2.

   [source_offset ~src ~big] returns a function that maps a flat index into a
   tensor of the big (broadcast) shape to the offset of the element it comes
   from in a tensor of the small shape [src].

   It peels the multi-index off [flat] one axis at a time, from the last axis
   to the first (this is "unravel", done without allocating an array). For each
   axis, the matching source axis is [ax - k], where k = number of leading axes
   that [src] lacks. Source axes of size 1 always use index 0, so they
   contribute nothing to the offset. *)
let source_offset ~src ~big =
  let st = strides src in
  let k = Array.length big - Array.length src in
  fun flat ->
    let rest = ref flat and offset = ref 0 in
    for ax = Array.length big - 1 downto 0 do
      let i = !rest mod big.(ax) in
      rest := !rest / big.(ax);
      let sax = ax - k in
      if sax >= 0 && src.(sax) <> 1 then offset := !offset + (i * st.(sax))
    done;
    !offset

let check_broadcastable ~fn ~small ~big =
  if broadcast_shapes small big <> big then
    invalid_arg
      (Printf.sprintf "Tensor.%s: %s does not broadcast to %s" fn
         (shape_to_string small) (shape_to_string big))

(* Every output element reads the source element it maps to. *)
let broadcast_to t target =
  check_broadcastable ~fn:"broadcast_to" ~small:t.shape ~big:target;
  let src = source_offset ~src:t.shape ~big:target in
  {
    shape = Array.copy target;
    data = Array.init (numel_of_shape target) (fun i -> t.data.(src i));
  }

(* The adjoint of broadcast_to: same mapping, but every big element is ADDED
   into the small element it maps to. *)
let sum_to_shape t target =
  check_broadcastable ~fn:"sum_to_shape" ~small:target ~big:t.shape;
  let dst = source_offset ~src:target ~big:t.shape in
  let out = Array.make (numel_of_shape target) 0.0 in
  Array.iteri
    (fun i v ->
      let o = dst i in
      out.(o) <- out.(o) +. v)
    t.data;
  { shape = Array.copy target; data = out }

(* ------------------------------------------------------------------------ *)
(* Elementwise                                                                *)
(* ------------------------------------------------------------------------ *)

let map f t = { shape = Array.copy t.shape; data = Array.map f t.data }

let map2 f a b =
  if a.shape = b.shape then
    (* fast path: no broadcasting needed *)
    { shape = Array.copy a.shape; data = Array.map2 f a.data b.data }
  else begin
    let out_shape = broadcast_shapes a.shape b.shape in
    let ia = source_offset ~src:a.shape ~big:out_shape
    and ib = source_offset ~src:b.shape ~big:out_shape in
    {
      shape = out_shape;
      data =
        Array.init (numel_of_shape out_shape) (fun i ->
            f a.data.(ia i) b.data.(ib i));
    }
  end

(* Derived ops: once [map] and [map2] work, these work too. *)
let add a b = map2 ( +. ) a b
let sub a b = map2 ( -. ) a b
let mul a b = map2 ( *. ) a b
let div a b = map2 ( /. ) a b
let neg t = map Float.neg t
let scale k t = map (fun x -> k *. x) t
let add_scalar k t = map (fun x -> x +. k) t
let pow t p = map (fun x -> Float.pow x p) t
let exp t = map Float.exp t
let log t = map Float.log t
let sqrt t = map Float.sqrt t

(* ------------------------------------------------------------------------ *)
(* Reductions                                                                 *)
(* ------------------------------------------------------------------------ *)

let sum t = scalar (Array.fold_left ( +. ) 0.0 t.data)
let mean t = scale (1.0 /. float_of_int (numel t)) (sum t)

(* Turn a possibly negative axis (-1 = last) into a valid non-negative one. *)
let normalize_axis ~fn t axis =
  let nd = ndim t in
  let a = if axis < 0 then axis + nd else axis in
  if a < 0 || a >= nd then
    invalid_arg
      (Printf.sprintf "Tensor.%s: axis %d out of range for shape %s" fn axis
         (shape_to_string t.shape));
  a

(* View the shape as (outer, n, inner) around [axis]. Element (o, k, i) is at
   data.(o * n * inner + k * inner + i). *)
let split_around t axis =
  let outer = numel_of_shape (Array.sub t.shape 0 axis) in
  let n = t.shape.(axis) in
  let inner =
    numel_of_shape (Array.sub t.shape (axis + 1) (ndim t - axis - 1))
  in
  (outer, n, inner)

(* Shape after reducing [axis]: removed, or kept with size 1 if keepdims. *)
let reduced_shape ~keepdims t axis =
  if keepdims then Array.mapi (fun i d -> if i = axis then 1 else d) t.shape
  else
    Array.of_list (List.filteri (fun i _ -> i <> axis) (Array.to_list t.shape))

let reduce_axis ~keepdims ~init ~f t axis =
  let axis = normalize_axis ~fn:"reduce_axis" t axis in
  let outer, n, inner = split_around t axis in
  let out = Array.make (outer * inner) init in
  for o = 0 to outer - 1 do
    for k = 0 to n - 1 do
      for i = 0 to inner - 1 do
        let j = (o * inner) + i in
        out.(j) <- f out.(j) t.data.((o * n * inner) + (k * inner) + i) k
      done
    done
  done;
  { shape = reduced_shape ~keepdims t axis; data = out }

let sum_axis ?(keepdims = false) t axis =
  reduce_axis ~keepdims ~init:0.0 ~f:(fun acc v _ -> acc +. v) t axis

let max_axis ?(keepdims = false) t axis =
  reduce_axis ~keepdims ~init:Float.neg_infinity
    ~f:(fun acc v _ -> Float.max acc v)
    t axis

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
(* Linear algebra and shape manipulation                                      *)
(* ------------------------------------------------------------------------ *)

(* i-p-j loop order: the innermost loop walks a row of [b] and a row of [c]
   contiguously, which is far more cache friendly than the textbook i-j-p. *)
let matmul a b =
  if ndim a <> 2 || ndim b <> 2 || a.shape.(1) <> b.shape.(0) then
    invalid_arg
      (Printf.sprintf "Tensor.matmul: cannot multiply %s by %s"
         (shape_to_string a.shape) (shape_to_string b.shape));
  let m = a.shape.(0) and k = a.shape.(1) and n = b.shape.(1) in
  let c = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for p = 0 to k - 1 do
      let a_ip = a.data.((i * k) + p) in
      let b_row = p * n and c_row = i * n in
      for j = 0 to n - 1 do
        c.(c_row + j) <- c.(c_row + j) +. (a_ip *. b.data.(b_row + j))
      done
    done
  done;
  { shape = [| m; n |]; data = c }

let transpose t =
  if ndim t <> 2 then
    invalid_arg
      (Printf.sprintf "Tensor.transpose: expected a 2-D tensor, got %s"
         (shape_to_string t.shape));
  let m = t.shape.(0) and n = t.shape.(1) in
  let out = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      out.((j * m) + i) <- t.data.((i * n) + j)
    done
  done;
  { shape = [| n; m |]; data = out }

(* Supports one -1 entry, whose size is inferred from the element count. *)
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
(* In-place operations                                                        *)
(* ------------------------------------------------------------------------ *)

let check_same_shape ~fn a b =
  if a.shape <> b.shape then
    invalid_arg
      (Printf.sprintf "Tensor.%s: shape mismatch %s vs %s" fn
         (shape_to_string a.shape) (shape_to_string b.shape))

(* y <- y + x *)
let add_ y x =
  check_same_shape ~fn:"add_" y x;
  Array.iteri (fun i v -> y.data.(i) <- y.data.(i) +. v) x.data

(* y <- y + alpha * x *)
let axpy_ ~alpha ~x ~y =
  check_same_shape ~fn:"axpy_" y x;
  Array.iteri (fun i v -> y.data.(i) <- y.data.(i) +. (alpha *. v)) x.data

let scale_ k t = Array.iteri (fun i v -> t.data.(i) <- k *. v) t.data
let fill_ t x = Array.fill t.data 0 (numel t) x

let blit_ ~src ~dst =
  check_same_shape ~fn:"blit_" dst src;
  Array.blit src.data 0 dst.data 0 (numel src)

(* ------------------------------------------------------------------------ *)
(* Convolution helpers (stretch goal, Phase 6)                                *)
(* ------------------------------------------------------------------------ *)

let output_size ~size ~kernel ~stride ~padding =
  ((size + (2 * padding) - kernel) / stride) + 1

(* The index walk shared by im2col and col2im. For every (output row, column)
   pair that reads a real (non-padding) input element, call
   [f col_offset input_offset]. Returns the column-matrix shape. *)
let im2col_walk input_shape ~kh ~kw ~stride ~padding f =
  if Array.length input_shape <> 4 then
    invalid_arg "Tensor.im2col: expected an NHWC tensor [n; h; w; c]";
  let n = input_shape.(0)
  and h = input_shape.(1)
  and w = input_shape.(2)
  and c = input_shape.(3) in
  let oh = output_size ~size:h ~kernel:kh ~stride ~padding
  and ow = output_size ~size:w ~kernel:kw ~stride ~padding in
  let cols = kh * kw * c in
  for b = 0 to n - 1 do
    for oy = 0 to oh - 1 do
      for ox = 0 to ow - 1 do
        let row = (((b * oh) + oy) * ow) + ox in
        for ky = 0 to kh - 1 do
          for kx = 0 to kw - 1 do
            let y = (oy * stride) + ky - padding
            and x = (ox * stride) + kx - padding in
            if y >= 0 && y < h && x >= 0 && x < w then
              for ch = 0 to c - 1 do
                f
                  ((row * cols) + (((ky * kw) + kx) * c) + ch)
                  ((((((b * h) + y) * w) + x) * c) + ch)
              done
          done
        done
      done
    done
  done;
  [| n * oh * ow; cols |]

(* Padding positions are never visited, so they stay 0. *)
let im2col t ~kh ~kw ~stride ~padding =
  let shape = im2col_walk t.shape ~kh ~kw ~stride ~padding (fun _ _ -> ()) in
  let out = Array.make (numel_of_shape shape) 0.0 in
  ignore
    (im2col_walk t.shape ~kh ~kw ~stride ~padding (fun col pos ->
         out.(col) <- t.data.(pos)));
  { shape; data = out }

(* Same walk, with the data flowing the other way and accumulating. *)
let col2im t ~input_shape ~kh ~kw ~stride ~padding =
  let out = Array.make (numel_of_shape input_shape) 0.0 in
  let shape =
    im2col_walk input_shape ~kh ~kw ~stride ~padding (fun col pos ->
        out.(pos) <- out.(pos) +. t.data.(col))
  in
  if shape <> t.shape then
    invalid_arg
      (Printf.sprintf "Tensor.col2im: expected columns of shape %s, got %s"
         (shape_to_string shape) (shape_to_string t.shape));
  { shape = Array.copy input_shape; data = out }

let max_pool2d t ~size ~stride =
  if ndim t <> 4 then
    invalid_arg "Tensor.max_pool2d: expected an NHWC tensor [n; h; w; c]";
  let n = t.shape.(0)
  and h = t.shape.(1)
  and w = t.shape.(2)
  and c = t.shape.(3) in
  let oh = output_size ~size:h ~kernel:size ~stride ~padding:0
  and ow = output_size ~size:w ~kernel:size ~stride ~padding:0 in
  let total = n * oh * ow * c in
  let out = Array.make total Float.neg_infinity
  and argmax = Array.make total 0 in
  for b = 0 to n - 1 do
    for oy = 0 to oh - 1 do
      for ox = 0 to ow - 1 do
        for ch = 0 to c - 1 do
          let o = (((((b * oh) + oy) * ow) + ox) * c) + ch in
          for ky = 0 to size - 1 do
            for kx = 0 to size - 1 do
              let y = (oy * stride) + ky and x = (ox * stride) + kx in
              let i = (((((b * h) + y) * w) + x) * c) + ch in
              if t.data.(i) > out.(o) then begin
                out.(o) <- t.data.(i);
                argmax.(o) <- i
              end
            done
          done
        done
      done
    done
  done;
  ({ shape = [| n; oh; ow; c |]; data = out }, argmax)

(* Only the winning input of each window gets gradient. *)
let max_pool2d_backward ~grad_out ~argmax ~input_shape =
  let out = Array.make (numel_of_shape input_shape) 0.0 in
  Array.iteri (fun o i -> out.(i) <- out.(i) +. grad_out.data.(o)) argmax;
  { shape = Array.copy input_shape; data = out }

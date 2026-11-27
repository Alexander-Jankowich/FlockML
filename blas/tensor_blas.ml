(* The Bigarray + BLAS backend. The numerical work happens in C
   (tensor_blas_stubs.c and kernels.h). This file checks shapes, works out
   broadcasting walks with Flockml.Layout, and splits each kernel across
   domains with Flockml.Parallel.

   It mirrors lib/tensor/tensor.ml function by function, including its error
   checks and its fixed reduction chunks. Where a function here says "as in
   tensor.ml", that file has the reasoning. *)

open Bigarray
open Flockml

type 'k buf = (float, 'k, c_layout) Array1.t

(* ------------------------------------------------------------------------ *)
(* C stubs                                                                   *)
(*                                                                           *)
(* All but [gemm] are noalloc: they run one chunk of a parallel loop, hold   *)
(* the runtime lock, and neither allocate nor raise. Float parameters are    *)
(* passed in float arrays, so nothing is boxed.                              *)
(* ------------------------------------------------------------------------ *)

external blas_name : unit -> string = "fb_blas_name"

(* Storage whose memory is recycled when the GC frees it; see fb_create in
   tensor_blas_stubs.c. *)
external create_buf : (float, 'k) kind -> int -> 'k buf = "fb_create"

external unary : int -> 'k buf -> 'k buf -> int -> int -> float array -> unit
  = "fb_unary_byte" "fb_unary"
[@@noalloc]

external binary :
  int -> 'k buf -> 'k buf -> 'k buf -> int -> int -> float array -> unit
  = "fb_binary_byte" "fb_binary"
[@@noalloc]

external strided :
  int -> 'k buf -> 'k buf -> 'k buf -> int array -> int -> int -> unit
  = "fb_strided_byte" "fb_strided"
[@@noalloc]

external sum_rows : 'k buf -> 'k buf -> int array -> int -> int -> unit
  = "fb_sum_rows"
[@@noalloc]

external sum_range : 'k buf -> int -> int -> (float[@unboxed])
  = "fb_sum_byte" "fb_sum"
[@@noalloc]

external reduce_axis_c : bool -> 'k buf -> 'k buf -> int array -> unit
  = "fb_reduce_axis"
[@@noalloc]

external argmax_c : 'k buf -> int array -> int array -> unit = "fb_argmax_axis"
[@@noalloc]

external transpose_c : 'k buf -> 'k buf -> int array -> int -> int -> unit
  = "fb_transpose"
[@@noalloc]

external take_rows_c :
  'k buf -> 'k buf -> int array -> int -> int -> int -> unit
  = "fb_take_rows_byte" "fb_take_rows"
[@@noalloc]

external of_floats : float array -> 'k buf -> int -> int -> unit
  = "fb_of_floats"
[@@noalloc]

external to_floats : 'k buf -> float array -> int -> int -> unit
  = "fb_to_floats"
[@@noalloc]

external log_softmax_c : 'k buf -> 'k buf -> int -> int -> int -> unit
  = "fb_log_softmax"
[@@noalloc]

external log_softmax_backward_c :
  'k buf -> 'k buf -> 'k buf -> int -> int -> int -> unit
  = "fb_log_softmax_backward_byte" "fb_log_softmax_backward"
[@@noalloc]

external softmax_ce_backward_c :
  'k buf -> int array -> 'k buf -> int -> float array -> int -> int -> unit
  = "fb_softmax_ce_backward_byte" "fb_softmax_ce_backward"
[@@noalloc]

external adam_c :
  'k buf -> 'k buf -> 'k buf -> 'k buf -> float array -> int -> int -> unit
  = "fb_adam_byte" "fb_adam"
[@@noalloc]

external sgd_c : 'k buf -> 'k buf -> 'k buf -> float array -> int -> int -> unit
  = "fb_sgd_byte" "fb_sgd"
[@@noalloc]

external im2col_c : bool -> 'k buf -> 'k buf -> int array -> int -> int -> unit
  = "fb_im2col_byte" "fb_im2col"
[@@noalloc]

external max_pool_c :
  'k buf -> 'k buf -> int array -> int array -> int -> int -> unit
  = "fb_max_pool_byte" "fb_max_pool"
[@@noalloc]

external max_pool_backward_c :
  'k buf -> int array -> 'k buf -> int -> int -> int -> int
  = "fb_max_pool_backward_byte" "fb_max_pool_backward"
[@@noalloc]

external gemm_c : 'k buf -> 'k buf -> 'k buf -> int array -> float array -> unit
  = "fb_gemm"

(* Operation codes: the enums at the top of tensor_blas_stubs.c. *)
let un_copy = 0
and un_neg = 1
and un_scale = 2
and un_add_scalar = 3
and un_pow = 4
and un_exp = 5
and un_log = 6
and un_sqrt = 7
and un_relu = 8
and un_fill = 9

let bin_add = 0
and bin_sub = 1
and bin_mul = 2
and bin_div = 3
and bin_relu_backward = 4
and bin_axpy = 5
and bin_copy_a = 6

let blas = blas_name ()

module type S = sig
  include Tensor_intf.S

  val dtype : string
  val of_tensor : Tensor.t -> t
  val to_tensor : t -> Tensor.t
end

module type KIND = sig
  type elt

  val kind : (float, elt) Bigarray.kind
  val dtype : string
end

module Make (K : KIND) : S = struct
  type t = { shape : int array; data : K.elt buf }

  let dtype = K.dtype

  (* Integer min and max, as in tensor.ml: the polymorphic Stdlib ones go
     through the generic comparison. *)
  let[@inline] min (a : int) b = if a < b then a else b
  let[@inline] max (a : int) b = if a > b then a else b
  let numel_of_shape = Layout.numel_of_shape
  let shape_to_string = Layout.shape_to_string
  let no_param = [| 0.0 |]

  (** Uninitialised storage for [n] elements, from the recycling pool. *)
  let create n = create_buf K.kind n

  let wrap shape data = { shape = Array.copy shape; data }

  (* -------------------------------------------------------------------- *)
  (* Construction and inspection                                           *)
  (* -------------------------------------------------------------------- *)

  let shape t = Array.copy t.shape
  let numel t = Array1.dim t.data
  let ndim t = Array.length t.shape

  (** [dst <- op src] over every element, in parallel when large. *)
  let unary_into ?(work = 1) op ?(k = 0.0) src dst =
    let n = Array1.dim dst and kk = [| k |] in
    Parallel.parallel_for ~work n (fun lo hi -> unary op src dst lo hi kk)

  let unary_op ?work op ?k t =
    let out = create (numel t) in
    unary_into ?work op ?k t.data out;
    wrap t.shape out

  let filled n x =
    let d = create n in
    unary_into un_fill ~k:x d d;
    d

  let full shape x = wrap shape (filled (numel_of_shape shape) x)
  let zeros shape = full shape 0.0
  let ones shape = full shape 1.0
  let scalar x = full [||] x

  let of_array shape data =
    let expected = numel_of_shape shape in
    if expected <> Array.length data then
      invalid_arg
        (Printf.sprintf
           "Tensor_blas.of_array: shape %s needs %d elements, got %d"
           (shape_to_string shape) expected (Array.length data));
    let d = create expected in
    Parallel.parallel_for expected (fun lo hi -> of_floats data d lo hi);
    wrap shape d

  let to_array t =
    let n = numel t in
    let a = Array.create_float n in
    Parallel.parallel_for n (fun lo hi -> to_floats t.data a lo hi);
    a

  let copy t = unary_op un_copy t
  let to_tensor t = Tensor.of_array t.shape (to_array t)
  let of_tensor x = of_array (Tensor.shape x) (Tensor.to_array x)

  let item t =
    if numel t <> 1 then
      invalid_arg
        (Printf.sprintf "Tensor_blas.item: tensor of shape %s has %d elements"
           (shape_to_string t.shape) (numel t));
    Array1.get t.data 0

  (* The cold paths go through the default backend. *)
  let allclose ?rtol ?atol a b =
    a.shape = b.shape && Tensor.allclose ?rtol ?atol (to_tensor a) (to_tensor b)

  let pp fmt t = Tensor.pp fmt (to_tensor t)

  let flat_index t idx =
    if Array.length idx <> ndim t then
      invalid_arg
        (Printf.sprintf "Tensor_blas.get: expected %d indices, got %d" (ndim t)
           (Array.length idx));
    let st = Layout.strides t.shape in
    let offset = ref 0 in
    Array.iteri
      (fun i x ->
        if x < 0 || x >= t.shape.(i) then
          invalid_arg
            (Printf.sprintf
               "Tensor_blas.get: index %d out of bounds for axis %d (size %d)" x
               i t.shape.(i));
        offset := !offset + (x * st.(i)))
      idx;
    !offset

  let get t idx = Array1.unsafe_get t.data (flat_index t idx)
  let set t idx v = Array1.unsafe_set t.data (flat_index t idx) v

  let rand_uniform ?rng ~lo ~hi shape =
    of_tensor (Tensor.rand_uniform ?rng ~lo ~hi shape)

  let randn ?rng ?mean ?std shape =
    of_tensor (Tensor.randn ?rng ?mean ?std shape)

  (* -------------------------------------------------------------------- *)
  (* Broadcasting                                                          *)
  (* -------------------------------------------------------------------- *)

  let broadcast_shapes = Layout.broadcast_shapes

  let check_broadcastable ~fn ~small ~big =
    if broadcast_shapes small big <> big then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: %s does not broadcast to %s" fn
           (shape_to_string small) (shape_to_string big))

  (** A coalesced walk packed for C as [[| nd; dims; strides0; strides1 |]]. A
      one-operand walk repeats its strides as the second operand's. *)
  let pack_walk dims (sts : int array array) =
    let nd = Array.length dims in
    if nd > 32 then invalid_arg "Tensor_blas: more than 32 dimensions";
    let s1 = if Array.length sts > 1 then sts.(1) else sts.(0) in
    Array.concat [ [| nd |]; dims; sts.(0); s1 ]

  let map f t =
    let d = create (numel t) in
    let a = Array.map f (to_array t) in
    of_floats a d 0 (Array.length a);
    wrap t.shape d

  let map2 f a b = of_tensor (Tensor.map2 f (to_tensor a) (to_tensor b))

  (** A broadcasting binary op: one flat loop for equal shapes, otherwise a
      coalesced walk over the output rows. *)
  let binop op a b =
    if a.shape = b.shape then begin
      let n = numel a in
      let out = create n in
      Parallel.parallel_for n (fun lo hi ->
          binary op a.data b.data out lo hi no_param);
      wrap a.shape out
    end
    else begin
      let shape = broadcast_shapes a.shape b.shape in
      let n = numel_of_shape shape in
      let out = create n in
      if n > 0 then begin
        let dims, sts =
          Layout.coalesce shape
            [|
              Layout.broadcast_strides ~src:a.shape ~out:shape;
              Layout.broadcast_strides ~src:b.shape ~out:shape;
            |]
        in
        let w = pack_walk dims sts and len = dims.(Array.length dims - 1) in
        Parallel.parallel_for ~work:len (n / len) (fun lo hi ->
            strided op a.data b.data out w lo hi)
      end;
      { shape; data = out }
    end

  let add a b = binop bin_add a b
  let sub a b = binop bin_sub a b
  let mul a b = binop bin_mul a b
  let div a b = binop bin_div a b
  let neg t = unary_op un_neg t
  let scale k t = unary_op un_scale ~k t
  let add_scalar k t = unary_op un_add_scalar ~k t
  let pow t p = unary_op ~work:20 un_pow ~k:p t
  let exp t = unary_op ~work:5 un_exp t
  let log t = unary_op ~work:5 un_log t
  let sqrt t = unary_op ~work:2 un_sqrt t
  let relu t = unary_op un_relu t

  let check_same_shape ~fn a b =
    if a.shape <> b.shape then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: shape mismatch %s vs %s" fn
           (shape_to_string a.shape) (shape_to_string b.shape))

  let relu_backward ~input ~grad =
    check_same_shape ~fn:"relu_backward" input grad;
    let n = numel input in
    let out = create n in
    Parallel.parallel_for n (fun lo hi ->
        binary bin_relu_backward input.data grad.data out lo hi no_param);
    wrap input.shape out

  let broadcast_to t target =
    check_broadcastable ~fn:"broadcast_to" ~small:t.shape ~big:target;
    let n = numel_of_shape target in
    let out = create n in
    if n > 0 then begin
      let dims, sts =
        Layout.coalesce target
          [| Layout.broadcast_strides ~src:t.shape ~out:target |]
      in
      let w = pack_walk dims sts and len = dims.(Array.length dims - 1) in
      Parallel.parallel_for ~work:len (n / len) (fun lo hi ->
          strided bin_copy_a t.data t.data out w lo hi)
    end;
    wrap target out

  (** Rows of at least this many elements are summed in one chunk: the same
      fixed constant as tensor.ml, so the chunking never depends on the thread
      count. *)
  let reduction_chunk = 65536

  (** A zeroed buffer of [n] elements followed by 32 unused ones, for per-task
      partial sums, so neighbouring partials never share a cache line. *)
  let padded_partial n = filled (n + 32) 0.0

  (** As in tensor.ml: small inputs are summed row by row in order; large inputs
      that reduce to a small result are cut into fixed chunks of rows, summed
      into partials in parallel, and the partials added in order. *)
  let sum_to_shape t target =
    check_broadcastable ~fn:"sum_to_shape" ~small:target ~big:t.shape;
    let m = numel_of_shape target in
    let out = filled m 0.0 in
    let n = numel t in
    if n > 0 then begin
      let dims, sts =
        Layout.coalesce t.shape
          [|
            Layout.strides t.shape;
            Layout.broadcast_strides ~src:target ~out:t.shape;
          |]
      in
      let w = pack_walk dims sts and len = dims.(Array.length dims - 1) in
      let rows = n / len in
      let chunk_rows = max 1 (reduction_chunk / len) in
      let chunks = (rows + chunk_rows - 1) / chunk_rows in
      if chunks <= 1 || m * chunks > n then sum_rows t.data out w 0 rows
      else begin
        let partials = Array.init chunks (fun _ -> padded_partial m) in
        Parallel.parallel_for ~work:(chunk_rows * len) chunks (fun lo hi ->
            for c = lo to hi - 1 do
              sum_rows t.data partials.(c) w (c * chunk_rows)
                (min rows ((c + 1) * chunk_rows))
            done);
        Array.iter (fun p -> binary bin_add out p out 0 m no_param) partials
      end
    end;
    wrap target out

  (* -------------------------------------------------------------------- *)
  (* Reductions                                                            *)
  (* -------------------------------------------------------------------- *)

  (** In order, like tensor.ml's [sum]. *)
  let sum t = scalar (sum_range t.data 0 (numel t))

  let mean t = scale (1.0 /. float_of_int (numel t)) (sum t)

  let normalize_axis ~fn t axis =
    let nd = ndim t in
    let a = if axis < 0 then axis + nd else axis in
    if a < 0 || a >= nd then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: axis %d out of range for shape %s" fn
           axis (shape_to_string t.shape));
    a

  let split_around t axis =
    let outer = numel_of_shape (Array.sub t.shape 0 axis) in
    let inner =
      numel_of_shape (Array.sub t.shape (axis + 1) (ndim t - axis - 1))
    in
    (outer, t.shape.(axis), inner)

  let reduced_shape ~keepdims t axis =
    if keepdims then Array.mapi (fun i d -> if i = axis then 1 else d) t.shape
    else
      Array.of_list
        (List.filteri (fun i _ -> i <> axis) (Array.to_list t.shape))

  (** The parallel structure of tensor.ml's [reduce_axis]: one task per outer
      index, or per column range when reducing the leading axis. *)
  let reduce_axis ~is_max ~keepdims t axis =
    let axis =
      normalize_axis ~fn:(if is_max then "max_axis" else "sum_axis") t axis
    in
    let outer, n, inner = split_around t axis in
    let out =
      filled (outer * inner) (if is_max then Float.neg_infinity else 0.0)
    in
    let run olo ohi ilo ihi =
      reduce_axis_c is_max t.data out [| n; inner; olo; ohi; ilo; ihi |]
    in
    if outer > 1 then
      Parallel.parallel_for ~work:(n * inner) outer (fun lo hi ->
          run lo hi 0 inner)
    else if outer = 1 && inner >= 1024 then
      Parallel.parallel_for ~work:n inner (fun lo hi -> run 0 1 lo hi)
    else if outer = 1 then run 0 1 0 inner;
    { shape = reduced_shape ~keepdims t axis; data = out }

  let sum_axis ?(keepdims = false) t axis =
    reduce_axis ~is_max:false ~keepdims t axis

  let max_axis ?(keepdims = false) t axis =
    reduce_axis ~is_max:true ~keepdims t axis

  let argmax_axis t axis =
    let axis = normalize_axis ~fn:"argmax_axis" t axis in
    let outer, n, inner = split_around t axis in
    let res = Array.make (outer * inner) 0 in
    argmax_c t.data res [| n; inner; 0; outer * inner |];
    res

  (* -------------------------------------------------------------------- *)
  (* Matrix multiplication                                                 *)
  (* -------------------------------------------------------------------- *)

  let gemm_ ?(trans_a = false) ?(trans_b = false) ?(alpha = 1.0) ?(beta = 0.0) a
      b ~c =
    let fail () =
      invalid_arg
        (Printf.sprintf "Tensor_blas.gemm_: cannot compute %s%s x %s%s into %s"
           (shape_to_string a.shape)
           (if trans_a then "^T" else "")
           (shape_to_string b.shape)
           (if trans_b then "^T" else "")
           (shape_to_string c.shape))
    in
    if ndim a <> 2 || ndim b <> 2 || ndim c <> 2 then fail ();
    if c.data == a.data || c.data == b.data then
      invalid_arg "Tensor_blas.gemm_: c must not share storage with a or b";
    let m, k =
      if trans_a then (a.shape.(1), a.shape.(0)) else (a.shape.(0), a.shape.(1))
    in
    let kb, n =
      if trans_b then (b.shape.(1), b.shape.(0)) else (b.shape.(0), b.shape.(1))
    in
    if k <> kb || c.shape.(0) <> m || c.shape.(1) <> n then fail ();
    gemm_c a.data b.data c.data
      [|
        Bool.to_int trans_a;
        Bool.to_int trans_b;
        m;
        n;
        k;
        max 1 a.shape.(1);
        max 1 b.shape.(1);
      |]
      [| alpha; beta |]

  let matmul a b =
    if ndim a <> 2 || ndim b <> 2 || a.shape.(1) <> b.shape.(0) then
      invalid_arg
        (Printf.sprintf "Tensor_blas.matmul: cannot multiply %s by %s"
           (shape_to_string a.shape) (shape_to_string b.shape));
    let c =
      {
        shape = [| a.shape.(0); b.shape.(1) |];
        data = create (a.shape.(0) * b.shape.(1));
      }
    in
    gemm_ a b ~c;
    c

  (* -------------------------------------------------------------------- *)
  (* Shape manipulation                                                    *)
  (* -------------------------------------------------------------------- *)

  let transpose t =
    if ndim t <> 2 then
      invalid_arg
        (Printf.sprintf "Tensor_blas.transpose: expected a 2-D tensor, got %s"
           (shape_to_string t.shape));
    let m = t.shape.(0) and n = t.shape.(1) in
    let out = create (m * n) and geo = [| m; n |] in
    Parallel.parallel_for ~work:n m (fun lo hi ->
        transpose_c t.data out geo lo hi);
    { shape = [| n; m |]; data = out }

  (** Copies, like tensor.ml's [reshape], so the result never aliases [t]. *)
  let reshape t new_shape =
    let fail () =
      invalid_arg
        (Printf.sprintf "Tensor_blas.reshape: cannot reshape %s into %s"
           (shape_to_string t.shape)
           (shape_to_string new_shape))
    in
    let new_shape = Array.copy new_shape in
    let unknown =
      Array.fold_left (fun acc d -> if d = -1 then acc + 1 else acc) 0 new_shape
    in
    if unknown > 1 then fail ();
    if unknown = 1 then begin
      let known =
        Array.fold_left
          (fun acc d -> if d = -1 then acc else acc * d)
          1 new_shape
      in
      if known = 0 || numel t mod known <> 0 then fail ();
      Array.iteri
        (fun i d -> if d = -1 then new_shape.(i) <- numel t / known)
        new_shape
    end;
    if numel_of_shape new_shape <> numel t then fail ();
    { (copy t) with shape = new_shape }

  (** The rows of [t] as [(rows, row_size)], after checking [idx]. *)
  let check_rows ~fn t idx =
    if ndim t = 0 then
      invalid_arg (Printf.sprintf "Tensor_blas.%s: scalar tensor" fn);
    let rows = t.shape.(0) in
    Array.iter
      (fun i ->
        if i < 0 || i >= rows then
          invalid_arg
            (Printf.sprintf "Tensor_blas.%s: row %d out of range (%d rows)" fn i
               rows))
      idx;
    (rows, if rows = 0 then 0 else numel t / rows)

  let take_rows t idx =
    let _, row_size = check_rows ~fn:"take_rows" t idx in
    let out = create (Array.length idx * row_size) in
    take_rows_c t.data out idx row_size 0 (Array.length idx);
    let shape = Array.copy t.shape in
    shape.(0) <- Array.length idx;
    { shape; data = out }

  let take_rows_ t idx ~dst =
    if ndim t = 0 then invalid_arg "Tensor_blas.take_rows_: scalar tensor";
    let expected = Array.copy t.shape in
    expected.(0) <- Array.length idx;
    if dst.shape <> expected then
      invalid_arg
        (Printf.sprintf "Tensor_blas.take_rows_: dst has shape %s, expected %s"
           (shape_to_string dst.shape)
           (shape_to_string expected));
    let _, row_size = check_rows ~fn:"take_rows_" t idx in
    take_rows_c t.data dst.data idx row_size 0 (Array.length idx)

  let one_hot ~num_classes labels =
    let n = Array.length labels in
    let t = zeros [| n; num_classes |] in
    Array.iteri
      (fun i label ->
        if label < 0 || label >= num_classes then
          invalid_arg
            (Printf.sprintf "Tensor_blas.one_hot: label %d not in [0, %d)" label
               num_classes);
        Array1.unsafe_set t.data ((i * num_classes) + label) 1.0)
      labels;
    t

  (* -------------------------------------------------------------------- *)
  (* Softmax                                                               *)
  (* -------------------------------------------------------------------- *)

  let last_axis_rows ~fn t =
    if ndim t = 0 then
      invalid_arg (Printf.sprintf "Tensor_blas.%s: scalar tensor" fn);
    let n = t.shape.(ndim t - 1) in
    ((if n = 0 then 0 else numel t / n), n)

  let log_softmax t =
    let rows, n = last_axis_rows ~fn:"log_softmax" t in
    let out = create (numel t) in
    Parallel.parallel_for ~work:(n * 12) rows (fun lo hi ->
        log_softmax_c t.data out n lo hi);
    wrap t.shape out

  let log_softmax_backward ~output ~grad =
    check_same_shape ~fn:"log_softmax_backward" output grad;
    let rows, n = last_axis_rows ~fn:"log_softmax_backward" output in
    let out = create (numel output) in
    Parallel.parallel_for ~work:(n * 8) rows (fun lo hi ->
        log_softmax_backward_c output.data grad.data out n lo hi);
    wrap output.shape out

  let check_labels ~fn t labels =
    if ndim t <> 2 then
      invalid_arg
        (Printf.sprintf
           "Tensor_blas.%s: expected [batch; classes] logits, got %s" fn
           (shape_to_string t.shape));
    let b = t.shape.(0) and c = t.shape.(1) in
    if Array.length labels <> b then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: %d labels for a batch of %d" fn
           (Array.length labels) b);
    Array.iter
      (fun l ->
        if l < 0 || l >= c then
          invalid_arg
            (Printf.sprintf "Tensor_blas.%s: label %d not in [0, %d)" fn l c))
      labels;
    (b, c)

  let softmax_cross_entropy logits labels =
    let b, c = check_labels ~fn:"softmax_cross_entropy" logits labels in
    let lp = log_softmax logits in
    let s = ref 0.0 in
    for r = 0 to b - 1 do
      s := !s +. Array1.unsafe_get lp.data ((r * c) + labels.(r))
    done;
    (-1.0 /. float_of_int b *. !s, lp)

  let softmax_cross_entropy_backward ~log_probs ~labels ~scale =
    let b, c =
      check_labels ~fn:"softmax_cross_entropy_backward" log_probs labels
    in
    let out = create (b * c) and sc = [| scale |] in
    Parallel.parallel_for ~work:(c * 8) b (fun lo hi ->
        softmax_ce_backward_c log_probs.data labels out c sc lo hi);
    { shape = [| b; c |]; data = out }

  (* -------------------------------------------------------------------- *)
  (* In-place operations                                                   *)
  (* -------------------------------------------------------------------- *)

  let add_ y x =
    check_same_shape ~fn:"add_" y x;
    Parallel.parallel_for (numel y) (fun lo hi ->
        binary bin_add y.data x.data y.data lo hi no_param)

  let axpy_ ~alpha ~x ~y =
    check_same_shape ~fn:"axpy_" y x;
    let k = [| alpha |] in
    Parallel.parallel_for (numel y) (fun lo hi ->
        binary bin_axpy y.data x.data y.data lo hi k)

  let scale_ k t = unary_into un_scale ~k t.data t.data
  let fill_ t x = unary_into un_fill ~k:x t.data t.data

  let blit_ ~src ~dst =
    check_same_shape ~fn:"blit_" dst src;
    unary_into un_copy src.data dst.data

  let check_all_same ~fn first rest =
    List.iter (fun t -> check_same_shape ~fn first t) rest

  let adam_update_ ~lr ~beta1 ~beta2 ~eps ~weight_decay ~bc1 ~bc2 ~param ~grad
      ~m ~v =
    check_all_same ~fn:"adam_update_" param [ grad; m; v ];
    let h = [| lr; beta1; beta2; eps; weight_decay; bc1; bc2 |] in
    Parallel.parallel_for ~work:6 (numel param) (fun lo hi ->
        adam_c param.data grad.data m.data v.data h lo hi)

  let sgd_update_ ~lr ~momentum ~weight_decay ~param ~grad ~velocity =
    check_all_same ~fn:"sgd_update_" param [ grad; velocity ];
    let h = [| lr; momentum; weight_decay |] in
    Parallel.parallel_for ~work:2 (numel param) (fun lo hi ->
        sgd_c param.data grad.data velocity.data h lo hi)

  (* -------------------------------------------------------------------- *)
  (* Convolution helpers                                                   *)
  (* -------------------------------------------------------------------- *)

  let output_size ~size ~kernel ~stride ~padding =
    ((size + (2 * padding) - kernel) / stride) + 1

  (** The checks of tensor.ml's [conv_geometry], returning the geometry packed
      for C: [[| h; w; c; kh; kw; stride; padding; oh; ow |]], and the batch
      size. *)
  let conv_geometry ~fn input_shape ~kh ~kw ~stride ~padding =
    if Array.length input_shape <> 4 then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: expected an NHWC tensor [n; h; w; c]"
           fn);
    if kh <= 0 || kw <= 0 || stride <= 0 || padding < 0 then
      invalid_arg
        (Printf.sprintf "Tensor_blas.%s: bad kernel, stride or padding" fn);
    let n = input_shape.(0)
    and h = input_shape.(1)
    and w = input_shape.(2)
    and c = input_shape.(3) in
    if h + (2 * padding) < kh || w + (2 * padding) < kw then
      invalid_arg
        (Printf.sprintf
           "Tensor_blas.%s: %dx%d window larger than the padded input" fn kh kw);
    let oh = output_size ~size:h ~kernel:kh ~stride ~padding
    and ow = output_size ~size:w ~kernel:kw ~stride ~padding in
    (n, [| h; w; c; kh; kw; stride; padding; oh; ow |])

  (** Padding positions are never written, so they stay 0. *)
  let im2col t ~kh ~kw ~stride ~padding =
    let n, geo = conv_geometry ~fn:"im2col" t.shape ~kh ~kw ~stride ~padding in
    let c = geo.(2) and oh = geo.(7) and ow = geo.(8) in
    let ncols = kh * kw * c in
    let out = filled (n * oh * ow * ncols) 0.0 in
    Parallel.parallel_for
      ~work:(oh * ow * ncols)
      n
      (fun lo hi -> im2col_c false t.data out geo lo hi);
    { shape = [| n * oh * ow; ncols |]; data = out }

  let col2im t ~input_shape ~kh ~kw ~stride ~padding =
    let n, geo =
      conv_geometry ~fn:"col2im" input_shape ~kh ~kw ~stride ~padding
    in
    let c = geo.(2) and oh = geo.(7) and ow = geo.(8) in
    let shape = [| n * oh * ow; kh * kw * c |] in
    if shape <> t.shape then
      invalid_arg
        (Printf.sprintf
           "Tensor_blas.col2im: expected columns of shape %s, got %s"
           (shape_to_string shape) (shape_to_string t.shape));
    let out = filled (numel_of_shape input_shape) 0.0 in
    Parallel.parallel_for
      ~work:(oh * ow * kh * kw * c)
      n
      (fun lo hi -> im2col_c true out t.data geo lo hi);
    wrap input_shape out

  let max_pool2d t ~size ~stride =
    let n, geo =
      conv_geometry ~fn:"max_pool2d" t.shape ~kh:size ~kw:size ~stride
        ~padding:0
    in
    let c = geo.(2) and oh = geo.(7) and ow = geo.(8) in
    let total = n * oh * ow * c in
    let out = create total and argmax = Array.make total 0 in
    Parallel.parallel_for
      ~work:(oh * ow * c * size * size)
      n
      (fun lo hi -> max_pool_c t.data out argmax geo lo hi);
    ({ shape = [| n; oh; ow; c |]; data = out }, argmax)

  (** [argmax] comes from the caller, so the C kernel checks every index (in
      parallel) and counts the bad ones. Images are processed in parallel when
      the winners of image [b] all lie in image [b], as in tensor.ml. *)
  let max_pool2d_backward ~grad_out ~argmax ~input_shape =
    if Array.length argmax <> numel grad_out then
      invalid_arg
        "Tensor_blas.max_pool2d_backward: argmax and grad_out sizes differ";
    let size = numel_of_shape input_shape in
    let out = filled size 0.0 and bad = Atomic.make 0 in
    let scatter lo hi =
      let b = max_pool_backward_c grad_out.data argmax out size lo hi in
      if b > 0 then ignore (Atomic.fetch_and_add bad b)
    in
    let n = if Array.length input_shape > 0 then input_shape.(0) else 0 in
    if ndim grad_out = 4 && grad_out.shape.(0) = n && n > 0 then begin
      let per_image = numel grad_out / n in
      Parallel.parallel_for ~work:per_image n (fun lo hi ->
          scatter (lo * per_image) (hi * per_image))
    end
    else scatter 0 (Array.length argmax);
    if Atomic.get bad > 0 then
      invalid_arg "Tensor_blas.max_pool2d_backward: index out of bounds";
    wrap input_shape out
end

module F64 = Make (struct
  type elt = float64_elt

  let kind = Float64
  let dtype = "float64"
end)

module F32 = Make (struct
  type elt = float32_elt

  let kind = Float32
  let dtype = "float32"
end)

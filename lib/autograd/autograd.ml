(** Reverse-mode automatic differentiation, as a functor over the tensor
    backend.

    How it fits together:
    - Every differentiable function (add, matmul, relu...) computes its output
      value eagerly with the backend [T] {e and} records a node whose [op] says
      how it was produced. These records are the computational graph.
    - [backward loss] walks that graph from the output back to the leaves,
      pattern matching on each node's [op] to apply the chain rule.

    Inside the functor [exp], [log], [tanh] etc. are redefined to work on graph
    nodes, so the implementation writes [Float.exp] / [T.exp] explicitly for
    floats / tensors. *)

(* State shared by every instance of the functor. With several backends in
   one program (say the default one and a BLAS one), node ids stay unique
   across all of them, and {!Autograd_intf.S.no_grad} switches recording off
   for all of them at once. *)

(** The last node id handed out by {!fresh_id}. *)
let next_id = Atomic.make 0

(** A new id, larger than every id handed out before it. *)
let fresh_id () = Atomic.fetch_and_add next_id 1 + 1

(** Per-domain recording state. *)
type mode = {
  mutable recording : bool;  (** false inside [no_grad]: no inputs recorded *)
  mutable anomaly : bool;  (** inside [detect_anomaly]: check every tensor *)
  mutable since : int;
      (** only nodes with at least this id take part in gradients: 0
          normally, and inside [value_and_grad] the id of its input, so the
          nodes that existed before the call (a model's parameters, say) are
          treated as constants *)
}

let mode = Domain.DLS.new_key (fun () -> { recording = true; anomaly = false; since = 0 })

(** Distinguishes one [topo_sort] from the next, so marking a node visited
    is one integer store rather than a hash-table insertion. *)
let pass = Atomic.make 0

(** [Make (T)] builds an autograd engine whose node values and gradients are
    tensors of backend [T]. *)
module Make (T : Tensor_intf.S) : Autograd_intf.S with type tensor = T.t =
struct
  type tensor = T.t

  (** The elementwise functions recorded by the [Unary] op. *)
  type unary = Sqrt | Abs | Sign | Square | Rsqrt | Log1p | Expm1 | Sin | Cos | Erf

  (** The graph as an algebraic data type: one constructor per primitive op,
      carrying the input nodes plus whatever the backward pass needs that is
      not already stored in the nodes. *)
  type op =
    | Leaf
    | Add of t * t
    | Sub of t * t
    | Mul of t * t
    | Div of t * t
    | Neg of t
    | Scale of float * t
    | Pow of t * float
    | Exp of t
    | Log of t
    | Matmul of t * t
    | Linear of t * t * t  (** [x], [w], [b] of [x · w + b] *)
    | Transpose of t
    | Reshape of t
    | Sum of t
    | Mean of t
    | Sum_axis of t * int  (** the non-negative axis that was reduced *)
    | Relu of t
    | Sigmoid of t
    | Tanh of t
    | Log_softmax of t
    | Softmax_cross_entropy of {
        logits : t;
        labels : int array;
        log_probs : tensor;  (** [log_softmax logits], kept for backward *)
      }
    | Im2col of { input : t; kh : int; kw : int; stride : int; padding : int }
    | Max_pool2d of { input : t; argmax : int array }
        (** [argmax] holds the flat input index that won each output window *)
    | Unary of unary * t
    | Clamp of { input : t; lo : float option; hi : float option }
    | Extreme of { is_max : bool; a : t; b : t }  (** [maximum] / [minimum] *)
    | Extreme_axis of { input : t; axis : int; is_max : bool; idx : int array }
        (** [max_axis] / [min_axis]; [idx] holds the winner of each row *)
    | Sum_axes of t * int list  (** the non-negative axes that were summed *)
    | Cumsum of t * int
    | Bmm of t * t  (** a batched matmul *)
    | Broadcast of t
    | Permute of t * int array  (** non-negative axes *)
    | Slice of t * Tensor_intf.range list
    | Pad of t * (int * int) array
    | Flip of t * int list
    | Concat of t list * int  (** the inputs, and the non-negative axis *)
    | Index_select of { input : t; axis : int; idx : int array }
    | Gather of { input : t; axis : int; idx : int array }
    | Where of { mask : tensor; a : t; b : t }
    | Custom of {
        name : string;
        inputs : t list;
        backward : tensor -> tensor option list;
      }

  (** A graph node. *)
  and t = {
    id : int;  (** unique, increasing: handy for hashing and for graphviz *)
    value : tensor;  (** the result of the forward computation *)
    mutable grad : tensor option;  (** filled in by [backward] *)
    mutable grad_owned : bool;
        (** whether [grad] is a buffer this node may update in place. A
            gradient handed over from another node (e.g. passed straight
            through an [Add]) is shared, and must be copied before it is
            changed. *)
    mutable spare : tensor option;
        (** the gradient buffer from before the last [zero_grad], reused by
            the next backward pass instead of allocating a new one *)
    mutable mark : int;  (** the [topo_sort] pass that last visited it *)
    mutable requires_grad : bool;
        (** whether gradients should flow to this node; mutable only for
            leaves, through [set_requires_grad] *)
    op : op;  (** how this node was produced *)
  }

  (* ---------------------------------------------------------------------- *)
  (* Node construction and inspection                                        *)
  (* ---------------------------------------------------------------------- *)

  let make value ~requires_grad op =
    {
      id = fresh_id ();
      value;
      grad = None;
      grad_owned = false;
      spare = None;
      mark = 0;
      requires_grad;
      op;
    }

  (** A node with no inputs: a constant or a parameter. *)
  let leaf value ~requires_grad = make value ~requires_grad Leaf

  (** Whether every element is finite: [x * 0] is 0 for finite [x] and NaN
      otherwise, so the sum is NaN exactly when something isn't finite. *)
  let finite t = not (Float.is_nan (T.item (T.sum (T.mul t (T.scalar 0.0)))))

  (** The node takes part in gradients: it requires one, and (inside
      [value_and_grad]) was created during the call. *)
  let[@inline] counts ~since p = p.requires_grad && p.id >= since

  let const value = leaf value ~requires_grad:false
  let param value = leaf value ~requires_grad:true
  let scalar x = const (T.scalar x)
  let value v = v.value
  let shape v = T.shape v.value
  let grad v = v.grad

  let grad_exn v =
    match v.grad with
    | Some g -> g
    | None ->
        failwith (Printf.sprintf "Var.grad_exn: node %d has no gradient" v.id)

  let requires_grad v = v.requires_grad
  let id v = v.id

  let zero_grad v =
    (match v.grad with Some g when v.grad_owned -> v.spare <- Some g | _ -> ());
    v.grad <- None;
    v.grad_owned <- false

  let detach v = const v.value

  let parents v =
    match v.op with
    | Leaf -> []
    | Add (a, b) | Sub (a, b) | Mul (a, b) | Div (a, b) | Matmul (a, b) ->
        [ a; b ]
    | Linear (x, w, b) -> [ x; w; b ]
    | Neg a
    | Scale (_, a)
    | Pow (a, _)
    | Exp a
    | Log a
    | Transpose a
    | Reshape a
    | Sum a
    | Mean a
    | Sum_axis (a, _)
    | Relu a
    | Sigmoid a
    | Tanh a
    | Log_softmax a
    | Softmax_cross_entropy { logits = a; _ }
    | Im2col { input = a; _ }
    | Max_pool2d { input = a; _ }
    | Unary (_, a)
    | Clamp { input = a; _ }
    | Extreme_axis { input = a; _ }
    | Sum_axes (a, _)
    | Cumsum (a, _)
    | Broadcast a
    | Permute (a, _)
    | Slice (a, _)
    | Pad (a, _)
    | Flip (a, _)
    | Index_select { input = a; _ }
    | Gather { input = a; _ } ->
        [ a ]
    | Extreme { a; b; _ } | Bmm (a, b) | Where { a; b; _ } -> [ a; b ]
    | Concat (inputs, _) | Custom { inputs; _ } -> inputs

  let unary_name = function
    | Sqrt -> "sqrt"
    | Abs -> "abs"
    | Sign -> "sign"
    | Square -> "square"
    | Rsqrt -> "rsqrt"
    | Log1p -> "log1p"
    | Expm1 -> "expm1"
    | Sin -> "sin"
    | Cos -> "cos"
    | Erf -> "erf"

  let op_name v =
    match v.op with
    | Leaf -> if v.requires_grad then "param" else "const"
    | Add _ -> "add"
    | Sub _ -> "sub"
    | Mul _ -> "mul"
    | Div _ -> "div"
    | Neg _ -> "neg"
    | Scale (k, _) -> Printf.sprintf "scale %g" k
    | Pow (_, p) -> Printf.sprintf "pow %g" p
    | Exp _ -> "exp"
    | Log _ -> "log"
    | Matmul _ -> "matmul"
    | Linear _ -> "linear"
    | Transpose _ -> "transpose"
    | Reshape _ -> "reshape"
    | Sum _ -> "sum"
    | Mean _ -> "mean"
    | Sum_axis (_, ax) -> Printf.sprintf "sum_axis %d" ax
    | Relu _ -> "relu"
    | Sigmoid _ -> "sigmoid"
    | Tanh _ -> "tanh"
    | Log_softmax _ -> "log_softmax"
    | Softmax_cross_entropy _ -> "softmax_cross_entropy"
    | Im2col _ -> "im2col"
    | Max_pool2d _ -> "max_pool2d"
    | Unary (u, _) -> unary_name u
    | Clamp _ -> "clamp"
    | Extreme { is_max; _ } -> if is_max then "maximum" else "minimum"
    | Extreme_axis { axis; is_max; _ } ->
        Printf.sprintf "%s %d" (if is_max then "max_axis" else "min_axis") axis
    | Sum_axes _ -> "sum_axes"
    | Cumsum (_, ax) -> Printf.sprintf "cumsum %d" ax
    | Bmm _ -> "bmm"
    | Broadcast _ -> "broadcast_to"
    | Permute _ -> "permute"
    | Slice _ -> "slice"
    | Pad _ -> "pad"
    | Flip _ -> "flip"
    | Concat (_, ax) -> Printf.sprintf "concat %d" ax
    | Index_select { axis; _ } -> Printf.sprintf "index_select %d" axis
    | Gather { axis; _ } -> Printf.sprintf "gather %d" axis
    | Where _ -> "where"
    | Custom { name; _ } -> name

  (** [detect_anomaly]'s check: raise unless [t], computed by node [v], is
      finite. *)
  let check ~what v t =
    if not (finite t) then
      failwith
        (Printf.sprintf
           "Var.detect_anomaly: the %s computed by %s (node %d, shape %s) is not \
            finite"
           what (op_name v) v.id
           (Layout.shape_to_string (T.shape t)))

  (** [node value op inputs] wraps the result of an operation. An interior node
      requires a gradient iff any of its inputs does; that is what lets
      [backward] skip whole subgraphs built only from constants. Under
      {!no_grad} the result is a plain constant, so the inputs (and the whole
      graph behind them) are not kept alive. *)
  let node value op inputs =
    let m = Domain.DLS.get mode in
    let v =
      if not m.recording then leaf value ~requires_grad:false
      else
        make value
          ~requires_grad:(List.exists (counts ~since:m.since) inputs)
          op
    in
    if m.anomaly then check ~what:"value" v value;
    v

  let no_grad f =
    let m = Domain.DLS.get mode in
    let saved = m.recording in
    m.recording <- false;
    Fun.protect ~finally:(fun () -> m.recording <- saved) f

  (* ---------------------------------------------------------------------- *)
  (* Forward operations                                                      *)
  (*                                                                         *)
  (* Each one computes its value with T, then records it with [node].        *)
  (* ---------------------------------------------------------------------- *)

  let add a b = node (T.add a.value b.value) (Add (a, b)) [ a; b ]
  let sub a b = node (T.sub a.value b.value) (Sub (a, b)) [ a; b ]
  let mul a b = node (T.mul a.value b.value) (Mul (a, b)) [ a; b ]
  let div a b = node (T.div a.value b.value) (Div (a, b)) [ a; b ]
  let neg a = node (T.neg a.value) (Neg a) [ a ]
  let scale k a = node (T.scale k a.value) (Scale (k, a)) [ a ]
  let pow a p = node (T.pow a.value p) (Pow (a, p)) [ a ]
  let exp a = node (T.exp a.value) (Exp a) [ a ]
  let log a = node (T.log a.value) (Log a) [ a ]

  (** The output starts as [b] broadcast to [[| m; n |]], and the product is
      accumulated straight into it ([beta = 1]), so the bias add costs no
      separate pass and no extra tensor. *)
  let linear x w b =
    let xs = T.shape x.value and ws = T.shape w.value in
    if Array.length xs <> 2 || Array.length ws <> 2 || xs.(1) <> ws.(0) then
      invalid_arg
        (Printf.sprintf "Var.linear: cannot multiply [%d dims] by [%d dims]"
           (Array.length xs) (Array.length ws));
    let out = T.broadcast_to b.value [| xs.(0); ws.(1) |] in
    T.gemm_ ~beta:1.0 x.value w.value ~c:out;
    node out (Linear (x, w, b)) [ x; w; b ]

  let transpose a = node (T.transpose a.value) (Transpose a) [ a ]
  let reshape a new_shape = node (T.reshape a.value new_shape) (Reshape a) [ a ]
  let sum a = node (T.sum a.value) (Sum a) [ a ]
  let mean a = node (T.mean a.value) (Mean a) [ a ]

  (** A negative axis is normalised (axis + ndim) before it is stored in
      [Sum_axis], so the backward pass only ever sees non-negative axes. *)
  let sum_axis ?(keepdims = false) a axis =
    let axis = if axis < 0 then axis + T.ndim a.value else axis in
    node (T.sum_axis ~keepdims a.value axis) (Sum_axis (a, axis)) [ a ]

  let relu a = node (T.relu a.value) (Relu a) [ a ]

  (** [1 / (1 + e^-x)] for [x >= 0] and the equivalent [e^x / (1 + e^x)] for
      [x < 0], so the exponential never overflows. *)
  let sigmoid a = node (T.sigmoid a.value) (Sigmoid a) [ a ]

  let tanh a = node (T.tanh a.value) (Tanh a) [ a ]

  (** Computed stably as [x - m - log (sum (exp (x - m)))], where [m] is the
      row maximum, so the largest exponent is [e^0]. The backend does it in a
      single fused pass, producing a single node: one closed-form backward
      rule is faster and more stable than differentiating a chain of Var
      ops. *)
  let log_softmax a = node (T.log_softmax a.value) (Log_softmax a) [ a ]

  let softmax_cross_entropy logits labels =
    let loss, log_probs = T.softmax_cross_entropy logits.value labels in
    node (T.scalar loss)
      (Softmax_cross_entropy { logits; labels = Array.copy labels; log_probs })
      [ logits ]

  (** The convolution parameters are recorded in the op so the backward pass
      can scatter gradients back with [col2im]. *)
  let im2col a ~kh ~kw ~stride ~padding =
    node
      (T.im2col a.value ~kh ~kw ~stride ~padding)
      (Im2col { input = a; kh; kw; stride; padding })
      [ a ]

  (** The winning index of each window ([argmax]) is kept in the op, so the
      backward pass can route gradients without searching the windows again. *)
  let max_pool2d a ~size ~stride =
    let value, argmax = T.max_pool2d a.value ~size ~stride in
    node value (Max_pool2d { input = a; argmax }) [ a ]

  (** A possibly negative axis of [a], normalised; the backend's checks report
      a bad one when the value is computed. *)
  let norm_axis a axis = if axis < 0 then axis + T.ndim a.value else axis

  let unary u f a = node (f a.value) (Unary (u, a)) [ a ]
  let sqrt a = unary Sqrt T.sqrt a
  let abs a = unary Abs T.abs a
  let sign a = unary Sign T.sign a
  let square a = unary Square T.square a
  let rsqrt a = unary Rsqrt T.rsqrt a
  let log1p a = unary Log1p T.log1p a
  let expm1 a = unary Expm1 T.expm1 a
  let sin a = unary Sin T.sin a
  let cos a = unary Cos T.cos a
  let erf a = unary Erf T.erf a

  let clamp ?min ?max a =
    node (T.clamp ?min ?max a.value) (Clamp { input = a; lo = min; hi = max }) [ a ]

  let minimum a b =
    node (T.minimum a.value b.value) (Extreme { is_max = false; a; b }) [ a; b ]

  let maximum a b =
    node (T.maximum a.value b.value) (Extreme { is_max = true; a; b }) [ a; b ]

  let where mask a b =
    node (T.where mask a.value b.value) (Where { mask; a; b }) [ a; b ]

  let masked_fill a mask x =
    let s = T.shape a.value in
    if T.broadcast_shapes (T.shape mask) s <> s then
      invalid_arg
        (Printf.sprintf "Var.masked_fill: mask %s does not broadcast to %s"
           (Layout.shape_to_string (T.shape mask)) (Layout.shape_to_string s));
    where mask (scalar x) a

  (** The winner of each row is recorded, so the backward pass scatters the
      gradient without searching again. *)
  let extreme_axis ~is_max ?(keepdims = false) a axis =
    let value =
      if is_max then T.max_axis ~keepdims a.value axis
      else T.min_axis ~keepdims a.value axis
    in
    let axis = norm_axis a axis in
    let idx =
      if is_max then T.argmax_axis a.value axis else T.argmin_axis a.value axis
    in
    node value (Extreme_axis { input = a; axis; is_max; idx }) [ a ]

  let max_axis ?keepdims a axis = extreme_axis ~is_max:true ?keepdims a axis
  let min_axis ?keepdims a axis = extreme_axis ~is_max:false ?keepdims a axis

  let mean_axis ?(keepdims = false) a axis =
    let s = T.shape a.value in
    let n = s.(norm_axis a axis) in
    scale (1. /. float_of_int n) (sum_axis ~keepdims a axis)

  let sum_axes ?(keepdims = false) a axes =
    let value = T.sum_axes ~keepdims a.value axes in
    let axes = List.sort_uniq compare (List.map (norm_axis a) axes) in
    node value (Sum_axes (a, axes)) [ a ]

  let mean_axes ?keepdims a axes =
    let s = T.shape a.value in
    let count =
      List.fold_left ( * ) 1
        (List.map (fun ax -> s.(ax)) (List.sort_uniq compare (List.map (norm_axis a) axes)))
    in
    scale (1. /. float_of_int count) (sum_axes ?keepdims a axes)

  (** Composed from Var ops, as {!Tensor_intf.S.var} is from tensor ops, so
      autograd differentiates it. *)
  let var ?(correction = 0) ?(keepdims = false) a axis =
    let n = (T.shape a.value).(norm_axis a axis) in
    let d = sub a (mean_axis ~keepdims:true a axis) in
    scale
      (1. /. float_of_int (n - correction))
      (sum_axis ~keepdims (mul d d) axis)

  let std ?correction ?keepdims a axis = sqrt (var ?correction ?keepdims a axis)
  let softmax a = exp (log_softmax a)

  let reshape_value a value = node value (Reshape a) [ a ]

  (** The (clamped) maximum is a constant, so the gradient is exactly the
      softmax along the axis. *)
  let logsumexp ?(keepdims = false) a axis =
    let ax = norm_axis a axis in
    let m =
      T.clamp ~min:(-.Float.max_float) ~max:Float.max_float
        (T.max_axis ~keepdims:true a.value ax)
    in
    let m = const m in
    let r = add m (log (sum_axis ~keepdims:true (exp (sub a m)) ax)) in
    if keepdims then r else reshape_value r (T.squeeze ~axis:ax r.value)

  let cumsum a axis =
    node (T.cumsum a.value axis) (Cumsum (a, norm_axis a axis)) [ a ]

  let permute a axes =
    let value = T.permute a.value axes in
    node value (Permute (a, Array.map (norm_axis a) axes)) [ a ]

  let swapaxes a i j =
    let nd = T.ndim a.value in
    let i = norm_axis a i and j = norm_axis a j in
    permute a (Array.init nd (fun x -> if x = i then j else if x = j then i else x))

  let squeeze ?axis a = reshape_value a (T.squeeze ?axis a.value)
  let unsqueeze a axis = reshape_value a (T.unsqueeze a.value axis)
  let flatten ?first ?last a = reshape_value a (T.flatten ?first ?last a.value)

  let broadcast_to a s = node (T.broadcast_to a.value s) (Broadcast a) [ a ]
  let slice a ranges = node (T.slice a.value ranges) (Slice (a, ranges)) [ a ]

  let narrow a ~axis ~start ~len =
    let value = T.narrow a.value ~axis ~start ~len in
    let ax = norm_axis a axis in
    let ranges =
      List.init (ax + 1) (fun i ->
          if i = ax then Tensor_intf.Range (start, start + len) else Tensor_intf.All)
    in
    node value (Slice (a, ranges)) [ a ]

  let pad ?value a widths =
    node (T.pad ?value a.value widths) (Pad (a, Array.copy widths)) [ a ]

  let flip a axes =
    node (T.flip a.value axes) (Flip (a, List.map (norm_axis a) axes)) [ a ]

  let concat ~axis inputs =
    let value = T.concat ~axis (List.map (fun a -> a.value) inputs) in
    let ax = norm_axis (List.hd inputs) axis in
    node value (Concat (inputs, ax)) inputs

  let stack ~axis inputs =
    let value = T.stack ~axis (List.map (fun a -> a.value) inputs) in
    let nd = T.ndim (List.hd inputs).value in
    let ax = if axis < 0 then axis + nd + 1 else axis in
    reshape_value
      (concat ~axis:ax (List.map (fun a -> unsqueeze a ax) inputs))
      value

  let split a ~axis sizes =
    let ax = norm_axis a axis in
    let s = T.shape a.value in
    if ax < 0 || ax >= Array.length s
       || List.exists (fun n -> n < 0) sizes
       || List.fold_left ( + ) 0 sizes <> s.(ax)
    then invalid_arg "Var.split: sizes don't add up to the length of the axis";
    List.rev
      (snd
         (List.fold_left
            (fun (start, acc) len -> (start + len, narrow a ~axis:ax ~start ~len :: acc))
            (0, []) sizes))

  let chunk a ~axis n =
    let sizes = List.map (fun c -> (T.shape c).(norm_axis a axis)) (T.chunk a.value ~axis n) in
    if (T.shape a.value).(norm_axis a axis) = 0 then [ reshape_value a (T.copy a.value) ]
    else split a ~axis sizes

  (** Built from reshape and broadcast_to, as in Tensor_derived, so the gradient
      sums the copies. *)
  let tile a reps =
    let s = T.shape a.value in
    if Array.exists (fun r -> r < 0) reps then invalid_arg "Var.tile: negative repetition";
    let k = Int.max (Array.length s) (Array.length reps) in
    let lead x = Array.append (Array.make (k - Array.length x) 1) x in
    let s = lead s and reps = lead reps in
    let inter = Array.init (2 * k) (fun i -> if i mod 2 = 0 then 1 else s.(i / 2)) in
    let big = Array.init (2 * k) (fun i -> if i mod 2 = 0 then reps.(i / 2) else s.(i / 2)) in
    reshape
      (broadcast_to (reshape a inter) big)
      (Array.init k (fun i -> reps.(i) * s.(i)))

  let repeat a ~axis n =
    let s = T.shape a.value in
    let nd = Array.length s in
    let ax = norm_axis a axis in
    if ax < 0 || ax >= nd then invalid_arg "Var.repeat: axis out of range";
    if n < 0 then invalid_arg "Var.repeat: negative count";
    let ins x = Array.concat [ Array.sub s 0 (ax + 1); [| x |]; Array.sub s (ax + 1) (nd - ax - 1) ] in
    let out = Array.copy s in
    out.(ax) <- s.(ax) * n;
    reshape (broadcast_to (reshape a (ins 1)) (ins n)) out

  let index_select a ~axis idx =
    let value = T.index_select a.value ~axis idx in
    node value (Index_select { input = a; axis = norm_axis a axis; idx = Array.copy idx }) [ a ]

  let gather a ~axis idx =
    let value = T.gather a.value ~axis idx in
    node value (Gather { input = a; axis = norm_axis a axis; idx = Array.copy idx }) [ a ]

  (** 2-D operands are one [Matmul] node, whose backward pass writes weight
      gradients in place. A 1-D operand is reshaped to 2-D around it, and an
      n-D left operand with a 2-D right one (a linear layer on a batch of
      sequences) is flattened to 2-D, so it still uses the fast path. Only
      true batches of matrices are a [Bmm] node. *)
  let rec matmul a b =
    let sa = T.shape a.value and sb = T.shape b.value in
    let na = Array.length sa and nb = Array.length sb in
    if na = 2 && nb = 2 then
      node (T.matmul a.value b.value) (Matmul (a, b)) [ a; b ]
    else if na = 1 || nb = 1 then begin
      if na = 0 || nb = 0 then ignore (T.matmul a.value b.value);
      let a' = if na = 1 then reshape a [| 1; sa.(0) |] else a in
      let b' = if nb = 1 then reshape b [| sb.(0); 1 |] else b in
      let r = matmul a' b' in
      let rs = T.shape r.value in
      let nr = Array.length rs in
      reshape r
        (Array.of_list
           (List.filteri
              (fun i _ -> not ((na = 1 && i = nr - 2) || (nb = 1 && i = nr - 1)))
              (Array.to_list rs)))
    end
    else if nb = 2 && na > 2 && sa.(na - 1) = sb.(0) then begin
      let rows = Layout.numel_of_shape (Array.sub sa 0 (na - 1)) in
      let r = matmul (reshape a [| rows; sa.(na - 1) |]) b in
      reshape r (Array.append (Array.sub sa 0 (na - 1)) [| sb.(1) |])
    end
    else node (T.matmul a.value b.value) (Bmm (a, b)) [ a; b ]

  let dot a b =
    let sa = T.shape a.value and sb = T.shape b.value in
    if Array.length sa <> 1 || sa <> sb then
      invalid_arg
        (Printf.sprintf "Var.dot: expected two 1-D tensors of the same length, got %s and %s"
           (Layout.shape_to_string sa) (Layout.shape_to_string sb));
    sum (mul a b)

  let outer a b =
    let sa = T.shape a.value and sb = T.shape b.value in
    if Array.length sa <> 1 || Array.length sb <> 1 then
      invalid_arg
        (Printf.sprintf "Var.outer: expected two 1-D tensors, got %s and %s"
           (Layout.shape_to_string sa) (Layout.shape_to_string sb));
    mul (reshape a [| sa.(0); 1 |]) (reshape b [| 1; sb.(0) |])

  let norm ?(p = 2.0) a =
    if p = Float.infinity then
      max_axis (reshape (abs a) [| T.numel a.value |]) 0
    else if not (p > 0.0) then
      invalid_arg (Printf.sprintf "Var.norm: p = %g is not positive" p)
    else if p = 2.0 then sqrt (sum (mul a a))
    else if p = 1.0 then sum (abs a)
    else pow (sum (pow (abs a) p)) (1. /. p)

  let custom ~name ~inputs ~value ~backward =
    node value (Custom { name; inputs; backward }) inputs

  (* Activations. The piecewise ones are compositions; the smooth ones are
     one custom node each, with their closed-form derivative. *)

  let positive a = T.gt a.value (T.scalar 0.)
  let leaky_relu ?(slope = 0.01) a = where (positive a) a (scale slope a)

  (** The exponential only sees [min x 0], so a large [x] can't overflow
      into the branch that [where] discards (whose gradient would then be
      [0 * inf]). *)
  let elu ?(alpha = 1.0) a =
    where (positive a) a (scale alpha (expm1 (clamp ~max:0. a)))

  let gelu ?(approximate = false) a =
    let x = a.value in
    if not approximate then begin
      let cdf = T.scale 0.5 (T.add_scalar 1. (T.erf (T.scale (1. /. Float.sqrt 2.) x))) in
      let pdf =
        T.scale (1. /. Float.sqrt (2. *. Float.pi)) (T.exp (T.scale (-0.5) (T.square x)))
      in
      custom ~name:"gelu" ~inputs:[ a ] ~value:(T.mul x cdf)
        ~backward:(fun g -> [ Some (T.mul g (T.add cdf (T.mul x pdf))) ])
    end
    else begin
      let k = Float.sqrt (2. /. Float.pi) and c = 0.044715 in
      let x2 = T.square x in
      let t = T.tanh (T.scale k (T.add x (T.scale c (T.mul x2 x)))) in
      let half_1pt = T.scale 0.5 (T.add_scalar 1. t) in
      custom ~name:"gelu_tanh" ~inputs:[ a ] ~value:(T.mul x half_1pt)
        ~backward:(fun g ->
          (* 0.5 (1 + t) + 0.5 x (1 - t²) k (1 + 3 c x²) *)
          let dt =
            T.mul
              (T.scale (0.5 *. k) (T.mul x (T.add_scalar 1. (T.neg (T.square t)))))
              (T.add_scalar 1. (T.scale (3. *. c) x2))
          in
          [ Some (T.mul g (T.add half_1pt dt)) ])
    end

  let silu a =
    let x = a.value in
    let s = T.sigmoid x in
    custom ~name:"silu" ~inputs:[ a ] ~value:(T.mul x s)
      ~backward:(fun g ->
        (* s (1 + x (1 - s)) *)
        [ Some (T.mul g (T.mul s (T.add_scalar 1. (T.mul x (T.add_scalar 1. (T.neg s)))))) ])

  (** [log (1 + e^(beta x)) / beta], and [x] itself where [beta x] exceeds
      [threshold], as PyTorch computes it. *)
  let softplus ?(beta = 1.0) ?(threshold = 20.0) a =
    let x = a.value in
    let z = T.scale beta x in
    let linear = T.gt z (T.scalar threshold) in
    let value = T.where linear x (T.scale (1. /. beta) (T.log1p (T.exp z))) in
    custom ~name:"softplus" ~inputs:[ a ] ~value
      ~backward:(fun g -> [ Some (T.mul g (T.where linear (T.scalar 1.) (T.sigmoid z))) ])

  (* ---------------------------------------------------------------------- *)
  (* Backward pass                                                           *)
  (* ---------------------------------------------------------------------- *)

  (** [accumulate ~owned v g] adds [g] into [v.grad] (if [v] requires a
      gradient). Gradients accumulate because a node used in several places
      receives a contribution from each use (multivariate chain rule): for
      [y = x * x], [dy/dx = x + x].

      [owned] says whether [g] was freshly computed for [v] alone. Then [v]
      may keep it and later add into it in place. A shared [g] (one that is
      also another node's gradient) is stored as is, but copied before
      anything is added to it. *)
  let accumulate ?(owned = true) v g =
    if v.requires_grad then
      match v.grad with
      | None ->
          v.grad <- Some g;
          v.grad_owned <- owned
      | Some g0 when v.grad_owned -> T.add_ g0 g
      | Some g0 ->
          v.grad <- Some (T.add g0 g);
          v.grad_owned <- true

  (** A buffer that the gradient of [v] can be written into directly, plus
      the [beta] to write it with: [0] for a buffer whose contents don't
      matter yet (the [spare] from before [zero_grad], or a fresh one), [1]
      to add onto an existing gradient. Lets [gemm_] produce weight
      gradients with no temporary matrix. *)
  let grad_buffer v =
    match v.grad with
    | Some g when v.grad_owned -> (g, 1.0)
    | Some g ->
        let g = T.copy g in
        v.grad <- Some g;
        v.grad_owned <- true;
        (g, 1.0)
    | None ->
        let shape = T.shape v.value in
        let buf =
          match v.spare with
          | Some s when T.shape s = shape -> s
          | _ -> T.zeros shape
        in
        v.spare <- None;
        v.grad <- Some buf;
        v.grad_owned <- true;
        (buf, 0.0)

  (** [propagate v g] applies one step of the chain rule: given the gradient
      [g] of the loss with respect to [v] (same shape as [v.value]), it adds
      the right gradient into each of [v]'s inputs.

      Notation in the cases below: [g] = upstream gradient, [out] = [v.value],
      [a]/[b] = inputs. Broadcasting binary ops reduce each input's gradient
      back to that input's shape with [T.sum_to_shape]. Inputs that do not
      require a gradient are skipped without computing anything. *)
  let propagate ~since v g =
    let wants x = counts ~since x in
    (* [acc x f] only computes the gradient [f ()] if [x] actually needs one,
       e.g. skipping d/dX of a matmul whose X is a constant input batch. *)
    let acc x f = if wants x then accumulate x (f ()) in
    (* For broadcasting binary ops: reduce back to the input's shape. *)
    let acc_bcast x f =
      acc x (fun () -> T.sum_to_shape (f ()) (T.shape x.value))
    in
    (* Gradient [g] passed through unchanged: shared unless it needs
       reducing to a broadcast input's shape. Once shared, [v] gives up the
       right to update [g] in place (a later backward call would otherwise
       change [x]'s gradient behind its back), so its next accumulation
       copies first. *)
    let pass_through x =
      if wants x then
        let xs = T.shape x.value in
        if xs = T.shape g then begin
          accumulate ~owned:false x g;
          v.grad_owned <- false
        end
        else accumulate x (T.sum_to_shape g xs)
    in
    (* da += alpha * op(p) op(q), written straight into a's gradient. *)
    let gemm_into x ?trans_a ?trans_b p q =
      if wants x then begin
        let dst, beta = grad_buffer x in
        T.gemm_ ?trans_a ?trans_b ~beta p q ~c:dst
      end
    in
    let out = v.value in
    match v.op with
    | Leaf -> ()
    (* Add: da = g, db = g *)
    | Add (a, b) ->
        pass_through a;
        pass_through b
    (* Sub: da = g, db = -g *)
    | Sub (a, b) ->
        pass_through a;
        acc_bcast b (fun () -> T.neg g)
    (* Mul: da = g * b, db = g * a  (each input gets the OTHER one) *)
    | Mul (a, b) ->
        acc_bcast a (fun () -> T.mul g b.value);
        acc_bcast b (fun () -> T.mul g a.value)
    (* Div: da = g / b, db = -g * a / b^2 *)
    | Div (a, b) ->
        acc_bcast a (fun () -> T.div g b.value);
        acc_bcast b (fun () ->
            T.neg (T.div (T.mul g a.value) (T.mul b.value b.value)))
    (* Neg: da = -g;  Scale: da = k * g  (unary, so no shape reduction) *)
    | Neg a -> acc a (fun () -> T.neg g)
    | Scale (k, a) -> acc a (fun () -> T.scale k g)
    (* Pow: da = g * p * a^(p-1) *)
    | Pow (a, p) ->
        acc a (fun () -> T.mul (T.scale p g) (T.pow a.value (p -. 1.)))
    (* Exp: da = g * out  (reuse the forward value) *)
    | Exp a -> acc a (fun () -> T.mul g out)
    (* Log: da = g / a *)
    | Log a -> acc a (fun () -> T.div g a.value)
    (* Matmul (out = a @ b), [m;k] x [k;n] -> [m;n]:
         da = g @ transpose b   : [m;n] x [n;k] -> [m;k]
         db = transpose a @ g   : [k;m] x [m;n] -> [k;n]
       gemm_ reads the transposes in place and adds into the gradients. *)
    | Matmul (a, b) ->
        gemm_into a ~trans_b:true g b.value;
        gemm_into b ~trans_a:true a.value g
    (* Linear (out = x @ w + b): as Matmul for x and w; the bias gradient
       sums g over the batch rows. *)
    | Linear (x, w, b) ->
        gemm_into x ~trans_b:true g w.value;
        gemm_into w ~trans_a:true x.value g;
        acc b (fun () -> T.sum_to_shape g (T.shape b.value))
    (* Transpose: da = transpose g *)
    | Transpose a -> acc a (fun () -> T.transpose g)
    (* Reshape: da = reshape g (shape of a) *)
    | Reshape a -> acc a (fun () -> T.reshape g (T.shape a.value))
    (* Sum: g is a scalar; da = broadcast_to g (shape a)
       Mean: same, divided by numel a *)
    | Sum a -> acc a (fun () -> T.broadcast_to g (T.shape a.value))
    | Mean a ->
        acc a (fun () ->
            let n = float_of_int (T.numel a.value) in
            T.scale (1. /. n) (T.broadcast_to g (T.shape a.value)))
    (* Sum_axis (a, axis): g lacks [axis] (unless keepdims was used). Reshape
       g to a's shape with that axis set to 1, then broadcast back. The
       reshape works in both cases since the element count is the same. *)
    | Sum_axis (a, axis) ->
        acc a (fun () ->
            let a_shape = T.shape a.value in
            let kept = Array.copy a_shape in
            kept.(axis) <- 1;
            T.broadcast_to (T.reshape g kept) a_shape)
    (* Relu: da = g where a > 0, else 0 *)
    | Relu a -> acc a (fun () -> T.relu_backward ~input:a.value ~grad:g)
    (* Sigmoid: da = g * out * (1 - out) *)
    | Sigmoid a ->
        acc a (fun () -> T.mul g (T.map (fun s -> s *. (1. -. s)) out))
    (* Tanh: da = g * (1 - out^2) *)
    | Tanh a -> acc a (fun () -> T.mul g (T.map (fun t -> 1. -. (t *. t)) out))
    (* Log_softmax along the last axis:
         softmax = exp out
         da = g - softmax * sum_axis ~keepdims:true g (-1) *)
    | Log_softmax a ->
        acc a (fun () -> T.log_softmax_backward ~output:out ~grad:g)
    (* Softmax cross-entropy (loss = -mean of the labels' log-probabilities):
         d logits = g * (softmax - one_hot) / batch *)
    | Softmax_cross_entropy { logits; labels; log_probs } ->
        acc logits (fun () ->
            let batch = float_of_int (Array.length labels) in
            T.softmax_cross_entropy_backward ~log_probs ~labels
              ~scale:(-1. /. batch *. T.item g))
    (* Im2col: scatter-add the column gradients back to the input. *)
    | Im2col { input; kh; kw; stride; padding } ->
        acc input (fun () ->
            T.col2im g ~input_shape:(T.shape input.value) ~kh ~kw ~stride
              ~padding)
    (* Max_pool2d: route each output's gradient to its window's winner. *)
    | Max_pool2d { input; argmax } ->
        acc input (fun () ->
            T.max_pool2d_backward ~grad_out:g ~argmax
              ~input_shape:(T.shape input.value))
    (* Elementwise: da = g * f'(a), using the output where it is cheaper. *)
    | Unary (u, a) ->
        acc a (fun () ->
            let x = a.value in
            match u with
            | Sqrt -> T.div (T.scale 0.5 g) out
            | Abs -> T.mul g (T.sign x)
            | Sign -> T.zeros (T.shape x)
            | Square -> T.mul g (T.scale 2. x)
            | Rsqrt -> T.mul g (T.scale (-0.5) (T.mul out (T.square out)))
            | Log1p -> T.div g (T.add_scalar 1. x)
            | Expm1 -> T.mul g (T.add_scalar 1. out)
            | Sin -> T.mul g (T.cos x)
            | Cos -> T.neg (T.mul g (T.sin x))
            | Erf ->
                T.mul g
                  (T.scale (2. /. Float.sqrt Float.pi) (T.exp (T.neg (T.square x)))))
    (* Clamp: g where lo <= a <= hi, else 0 *)
    | Clamp { input; lo; hi } ->
        acc input (fun () ->
            let x = input.value in
            let bound f = function
              | Some c -> Some (f x (T.scalar c))
              | None -> None
            in
            match (bound T.ge lo, bound T.le hi) with
            | Some m, Some n -> T.mul g (T.mul m n)
            | Some m, None | None, Some m -> T.mul g m
            | None, None -> T.copy g)
    (* Maximum / minimum: g to the winner, half to each on a tie *)
    | Extreme { is_max; a; b } ->
        let wins x y =
          let beats = if is_max then T.gt x y else T.lt x y in
          T.add beats (T.scale 0.5 (T.eq x y))
        in
        acc_bcast a (fun () -> T.mul g (wins a.value b.value));
        acc_bcast b (fun () -> T.mul g (wins b.value a.value))
    (* Max_axis / min_axis: g to the winner of each row *)
    | Extreme_axis { input; axis; idx; _ } ->
        acc input (fun () ->
            let s = T.shape input.value in
            let kept = Array.copy s in
            kept.(axis) <- 1;
            T.scatter_add (T.zeros s) ~axis idx (T.reshape g kept))
    (* Sum_axes: as Sum_axis, for several axes at once *)
    | Sum_axes (a, axes) ->
        acc a (fun () ->
            let s = T.shape a.value in
            let kept = Array.mapi (fun i d -> if List.mem i axes then 1 else d) s in
            T.broadcast_to (T.reshape g kept) s)
    (* Cumsum: da_k = sum of g_j for j >= k, a reversed cumulative sum *)
    | Cumsum (a, axis) ->
        acc a (fun () -> T.flip (T.cumsum (T.flip g [ axis ]) axis) [ axis ])
    (* Batched matmul: as Matmul for each matrix, then summed over any batch
       axes that were broadcast. *)
    | Bmm (a, b) ->
        acc_bcast a (fun () -> T.bmm ~trans_b:true g b.value);
        acc_bcast b (fun () -> T.bmm ~trans_a:true a.value g)
    | Broadcast a -> acc a (fun () -> T.sum_to_shape g (T.shape a.value))
    (* Permute: apply the inverse permutation *)
    | Permute (a, axes) ->
        acc a (fun () ->
            let inv = Array.make (Array.length axes) 0 in
            Array.iteri (fun i ax -> inv.(ax) <- i) axes;
            T.permute g inv)
    (* Slice: g written into zeros at the positions that were read *)
    | Slice (a, ranges) ->
        acc a (fun () ->
            let z = T.zeros (T.shape a.value) in
            T.set_slice_ z ranges g;
            z)
    (* Pad: the inner part of g *)
    | Pad (a, widths) ->
        acc a (fun () ->
            let s = T.shape a.value in
            T.slice g
              (Array.to_list
                 (Array.mapi (fun ax (l, _) -> Tensor_intf.Range (l, l + s.(ax))) widths)))
    | Flip (a, axes) -> acc a (fun () -> T.flip g axes)
    (* Concat: each input gets its own part of g *)
    | Concat (inputs, axis) ->
        ignore
          (List.fold_left
             (fun start x ->
               let len = (T.shape x.value).(axis) in
               acc x (fun () -> T.narrow g ~axis ~start ~len);
               start + len)
             0 inputs)
    | Index_select { input; axis; idx } ->
        acc input (fun () -> T.index_add (T.zeros (T.shape input.value)) ~axis idx g)
    | Gather { input; axis; idx } ->
        acc input (fun () ->
            T.scatter_add (T.zeros (T.shape input.value)) ~axis idx g)
    (* Where: g goes to a where the mask is set, to b elsewhere *)
    | Where { mask; a; b } ->
        let zero = T.scalar 0. in
        acc_bcast a (fun () -> T.where mask g zero);
        acc_bcast b (fun () -> T.where mask zero g)
    (* Custom: the user's rule. Its results may alias g or each other, so
       they are stored as shared, and v gives up in-place rights to g. *)
    | Custom { name; inputs; backward } ->
        let gs = backward g in
        if List.length gs <> List.length inputs then
          invalid_arg
            (Printf.sprintf
               "Var.custom: the backward rule of %s returned %d gradients for %d \
                inputs"
               name (List.length gs) (List.length inputs));
        List.iter2
          (fun x gx ->
            match gx with
            | Some gx when wants x ->
                if T.shape gx <> T.shape x.value then
                  invalid_arg
                    (Printf.sprintf
                       "Var.custom: the backward rule of %s returned a gradient \
                        of shape %s for an input of shape %s"
                       name
                       (Layout.shape_to_string (T.shape gx))
                       (Layout.shape_to_string (T.shape x.value)));
                accumulate ~owned:false x gx
            | _ -> ())
          inputs gs;
        v.grad_owned <- false

  (** Every node reachable from [root] that requires a gradient, in topological
      order: each node appears after all of its parents. A post-order DFS that
      marks visited nodes with the current pass number; it does not descend
      into nodes that need no gradient, since nothing below them does either.
      The recursion is as deep as the graph, which is fine for MLPs and
      CNNs. *)
  let topo_sort ~since root =
    let p = Atomic.fetch_and_add pass 1 + 1 in
    let order = ref [] in
    let rec visit v =
      if counts ~since v && v.mark <> p then begin
        v.mark <- p;
        List.iter visit (parents v);
        order := v :: !order
      end
    in
    visit root;
    (* [order] was built by prepending, so it is reversed: put parents first. *)
    List.rev !order

  (** Seeds the root with [d loss / d loss = 1], then propagates in reverse
      topological order (outputs first). That order guarantees a node's
      gradient is complete, with every consumer's contribution added, before
      it is pushed further down. Only scalar roots are supported: a non-scalar
      output would need a caller-supplied seed vector. *)
  let backward root =
    if T.numel root.value <> 1 then
      invalid_arg
        (Printf.sprintf
           "Var.backward: expected a scalar output, got %d elements"
           (T.numel root.value));
    let m = Domain.DLS.get mode in
    let since = m.since in
    if counts ~since root then accumulate root (T.ones (shape root));
    (* Outputs first: reverse of topological order. *)
    List.iter
      (fun v ->
        match v.grad with
        | Some g ->
            propagate ~since v g;
            if m.anomaly then
              List.iter
                (fun p ->
                  match p.grad with
                  | Some gp when counts ~since p ->
                      if not (finite gp) then
                        failwith
                          (Printf.sprintf
                             "Var.detect_anomaly: the gradient computed by the \
                              backward pass of %s (node %d), for its input node \
                              %d (shape %s), is not finite"
                             (op_name v) v.id p.id
                             (Layout.shape_to_string (T.shape gp)))
                  | _ -> ())
                (parents v)
        | None -> ())
      (List.rev (topo_sort ~since root))

  (* ---------------------------------------------------------------------- *)
  (* Gradient utilities                                                      *)
  (* ---------------------------------------------------------------------- *)

  let set_requires_grad v b =
    match v.op with
    | Leaf -> v.requires_grad <- b
    | _ ->
        invalid_arg
          (Printf.sprintf "Var.set_requires_grad: node %d (%s) is not a leaf" v.id
             (op_name v))

  let grad_norm vs =
    let sq =
      List.fold_left
        (fun acc v ->
          match v.grad with
          | Some g -> acc +. T.item (T.sum (T.mul g g))
          | None -> acc)
        0. vs
    in
    Float.sqrt sq

  (** Replace [v]'s gradient by [f g], in place when [v] owns it. *)
  let update_grad v f ~in_place =
    match v.grad with
    | Some g when v.grad_owned -> in_place g
    | Some g ->
        v.grad <- Some (f g);
        v.grad_owned <- true
    | None -> ()

  let clip_grad_norm ~max_norm vs =
    let total = grad_norm vs in
    let coef = max_norm /. (total +. 1e-6) in
    if coef < 1. then
      List.iter
        (fun v -> update_grad v (T.scale coef) ~in_place:(T.scale_ coef))
        vs;
    total

  let clip_grad_value c vs =
    let clip g = T.clamp ~min:(-.c) ~max:c g in
    List.iter
      (fun v -> update_grad v clip ~in_place:(fun g -> T.blit_ ~src:(clip g) ~dst:g))
      vs

  let detect_anomaly f =
    let m = Domain.DLS.get mode in
    let saved = m.anomaly in
    m.anomaly <- true;
    Fun.protect ~finally:(fun () -> m.anomaly <- saved) f

  (** [x] becomes a fresh parameter, and [since] its id, so only the graph
      built from it inside [f] takes part in [backward]. *)
  let value_and_grad f x =
    let m = Domain.DLS.get mode in
    let recording = m.recording and since = m.since in
    m.recording <- true;
    let p = param x in
    m.since <- p.id;
    Fun.protect
      ~finally:(fun () ->
        m.recording <- recording;
        m.since <- since)
      (fun () ->
        let out = f p in
        if T.numel out.value <> 1 then
          invalid_arg
            (Printf.sprintf "Var.grad: f returned %d elements, not a scalar"
               (T.numel out.value));
        backward out;
        let g =
          match p.grad with
          | Some g -> if p.grad_owned then g else T.copy g
          | None -> T.zeros (T.shape x)
        in
        (T.item out.value, g))

  let gradient f x = snd (value_and_grad f x)

  module Infix = struct
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
  end
end

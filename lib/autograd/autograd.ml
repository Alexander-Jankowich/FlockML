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

(** [Make (T)] builds an autograd engine whose node values and gradients are
    tensors of backend [T]. *)
module Make (T : Tensor_intf.S) : Autograd_intf.S with type tensor = T.t =
struct
  type tensor = T.t

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
    requires_grad : bool;  (** whether gradients should flow to this node *)
    op : op;  (** how this node was produced *)
  }

  (* ---------------------------------------------------------------------- *)
  (* Node construction and inspection                                        *)
  (* ---------------------------------------------------------------------- *)

  (** The last id handed out by {!fresh_id}. *)
  let next_id = Atomic.make 0

  (** A new id, larger than every id handed out before it. *)
  let fresh_id () = Atomic.fetch_and_add next_id 1 + 1

  (** False inside {!no_grad}: new nodes then record no inputs. *)
  let recording = Domain.DLS.new_key (fun () -> ref true)

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

  (** [node value op inputs] wraps the result of an operation. An interior node
      requires a gradient iff any of its inputs does; that is what lets
      [backward] skip whole subgraphs built only from constants. Under
      {!no_grad} the result is a plain constant, so the inputs (and the whole
      graph behind them) are not kept alive. *)
  let node value op inputs =
    if not !(Domain.DLS.get recording) then leaf value ~requires_grad:false
    else
      make value
        ~requires_grad:(List.exists (fun p -> p.requires_grad) inputs)
        op

  let no_grad f =
    let r = Domain.DLS.get recording in
    let saved = !r in
    r := false;
    Fun.protect ~finally:(fun () -> r := saved) f

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
    | Max_pool2d { input = a; _ } ->
        [ a ]

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
  let matmul a b = node (T.matmul a.value b.value) (Matmul (a, b)) [ a; b ]

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
  let sigmoid a =
    let stable x =
      if x >= 0. then 1. /. (1. +. Float.exp (-.x))
      else
        let e = Float.exp x in
        e /. (1. +. e)
    in
    node (T.map stable a.value) (Sigmoid a) [ a ]

  let tanh a = node (T.map Float.tanh a.value) (Tanh a) [ a ]

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
  let propagate v g =
    (* [acc x f] only computes the gradient [f ()] if [x] actually needs one,
       e.g. skipping d/dX of a matmul whose X is a constant input batch. *)
    let acc x f = if x.requires_grad then accumulate x (f ()) in
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
      if x.requires_grad then
        let xs = T.shape x.value in
        if xs = T.shape g then begin
          accumulate ~owned:false x g;
          v.grad_owned <- false
        end
        else accumulate x (T.sum_to_shape g xs)
    in
    (* da += alpha * op(p) op(q), written straight into a's gradient. *)
    let gemm_into x ?trans_a ?trans_b p q =
      if x.requires_grad then begin
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

  (** Distinguishes one [topo_sort] from the next, so marking a node visited
      is one integer store rather than a hash-table insertion. *)
  let pass = Atomic.make 0

  (** Every node reachable from [root] that requires a gradient, in topological
      order: each node appears after all of its parents. A post-order DFS that
      marks visited nodes with the current pass number; it does not descend
      into nodes that need no gradient, since nothing below them does either.
      The recursion is as deep as the graph, which is fine for MLPs and
      CNNs. *)
  let topo_sort root =
    let p = Atomic.fetch_and_add pass 1 + 1 in
    let order = ref [] in
    let rec visit v =
      if v.requires_grad && v.mark <> p then begin
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
    accumulate root (T.ones (shape root));
    (* Outputs first: reverse of topological order. *)
    List.iter
      (fun v -> match v.grad with Some g -> propagate v g | None -> ())
      (List.rev (topo_sort root))

  module Infix = struct
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
  end
end

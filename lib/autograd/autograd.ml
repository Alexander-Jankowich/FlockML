(* Reverse-mode automatic differentiation, as a functor over the tensor
   backend.

   How it fits together:
   - Every differentiable function (add, matmul, relu...) computes its output
     value eagerly with the backend [T] AND records a node whose [op] says how
     it was produced. This record IS the computational graph.
   - [backward loss] walks that graph from the output back to the leaves,
     pattern matching on each node's [op] to apply the chain rule.

   Inside this functor [exp], [log], [tanh] etc. get redefined to work on
   graph nodes. Use [Float.exp] / [T.exp] for floats / tensors to be explicit. *)

module Make (T : Tensor_intf.S) : Autograd_intf.S with type tensor = T.t =
struct
  type tensor = T.t

  (* The graph as an algebraic data type: one constructor per primitive op,
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
    | Transpose of t
    | Reshape of t
    | Sum of t
    | Mean of t
    | Sum_axis of t * int  (** the non-negative axis that was reduced *)
    | Relu of t
    | Sigmoid of t
    | Tanh of t
    | Log_softmax of t
    | Im2col of { input : t; kh : int; kw : int; stride : int; padding : int }
    | Max_pool2d of { input : t; argmax : int array }

  and t = {
    id : int;  (** unique, increasing: handy for hashing and for graphviz *)
    value : tensor;
    mutable grad : tensor option;
    requires_grad : bool;
    op : op;
  }

  (* ---------------------------------------------------------------------- *)
  (* Node construction and inspection (provided)                             *)
  (* ---------------------------------------------------------------------- *)

  let next_id = ref 0

  let fresh_id () =
    incr next_id;
    !next_id

  let leaf value ~requires_grad =
    { id = fresh_id (); value; grad = None; requires_grad; op = Leaf }

  (* An interior node requires a gradient iff any of its inputs does; that is
     what lets [backward] skip whole subgraphs built only from constants. *)
  let node value op inputs =
    {
      id = fresh_id ();
      value;
      grad = None;
      requires_grad = List.exists (fun p -> p.requires_grad) inputs;
      op;
    }

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
  let zero_grad v = v.grad <- None
  let detach v = const v.value

  let parents v =
    match v.op with
    | Leaf -> []
    | Add (a, b) | Sub (a, b) | Mul (a, b) | Div (a, b) | Matmul (a, b) ->
        [ a; b ]
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
    | Transpose _ -> "transpose"
    | Reshape _ -> "reshape"
    | Sum _ -> "sum"
    | Mean _ -> "mean"
    | Sum_axis (_, ax) -> Printf.sprintf "sum_axis %d" ax
    | Relu _ -> "relu"
    | Sigmoid _ -> "sigmoid"
    | Tanh _ -> "tanh"
    | Log_softmax _ -> "log_softmax"
    | Im2col _ -> "im2col"
    | Max_pool2d _ -> "max_pool2d"

  (* ---------------------------------------------------------------------- *)
  (* Forward operations                                                      *)
  (*                                                                         *)
  (* Pattern: compute the value with T, then wrap it with [node].             *)
  (* [add] is done as a worked example; the rest follow the same shape.      *)
  (* ---------------------------------------------------------------------- *)

  let add a b = node (T.add a.value b.value) (Add (a, b)) [ a; b ]

  (* TODO [Phase 2, easy]: follow [add]. *)
  let sub a b = Todo.todo "Var.sub"
  let mul a b = Todo.todo "Var.mul"
  let div a b = Todo.todo "Var.div"
  let neg a = Todo.todo "Var.neg"
  let scale k a = Todo.todo "Var.scale"
  let pow a p = Todo.todo "Var.pow"
  let exp a = Todo.todo "Var.exp"
  let log a = Todo.todo "Var.log"
  let matmul a b = Todo.todo "Var.matmul"
  let transpose a = Todo.todo "Var.transpose"
  let reshape a new_shape = Todo.todo "Var.reshape"
  let sum a = Todo.todo "Var.sum"
  let mean a = Todo.todo "Var.mean"

  (* TODO [Phase 2, easy]: normalise a negative axis (axis + ndim) before
     storing it in [Sum_axis], so backward never has to think about it. *)
  let sum_axis ?(keepdims = false) a axis = Todo.todo "Var.sum_axis"

  (* TODO [Phase 2, easy]: value = T.map (fun x -> Float.max x 0.0). *)
  let relu a = Todo.todo "Var.relu"

  (* TODO [Phase 2, easy]: 1 / (1 + e^-x). For a numerically stable version
     use e^x / (1 + e^x) when x < 0, so you never compute e^(large). *)
  let sigmoid a = Todo.todo "Var.sigmoid"

  (* TODO [Phase 2, easy]: T.map Float.tanh. *)
  let tanh a = Todo.todo "Var.tanh"

  (* TODO [Phase 2, medium]: log_softmax along the last axis, stably:
       m   = max_axis ~keepdims:true x (-1)
       out = x - m - log (sum_axis ~keepdims:true (exp (x - m)) (-1))
     All of this is tensor-level work (T functions) producing ONE node; do not build
     it out of Var ops, or backward gets slower and less stable. *)
  let log_softmax a = Todo.todo "Var.log_softmax"

  (* TODO [Phase 6]: value = T.im2col; record the conv parameters in the op. *)
  let im2col a ~kh ~kw ~stride ~padding = Todo.todo "Var.im2col"

  (* TODO [Phase 6]: T.max_pool2d returns (value, argmax); keep argmax in the
     op for backward. *)
  let max_pool2d a ~size ~stride = Todo.todo "Var.max_pool2d"

  (* ---------------------------------------------------------------------- *)
  (* Backward pass                                                           *)
  (* ---------------------------------------------------------------------- *)

  (* Add [g] into [v.grad] (provided). Gradients ACCUMULATE because a node
     used in several places receives a contribution from each use
     (multivariate chain rule): for y = x * x, dy/dx = x + x. *)
  let accumulate v g =
    if v.requires_grad then
      v.grad <- Some (match v.grad with None -> g | Some g0 -> T.add g0 g)

  (* Given a node [v] and the gradient [g] of the loss w.r.t. [v] (same shape
     as v.value), push the right gradient into each of v's inputs.

     Notation below: g = upstream gradient, out = v.value, a/b = inputs.
     For any broadcasting binary op, wrap each input's gradient in
     [T.sum_to_shape _ (T.shape a.value)] before accumulating (see [Add]).

     Optional optimisation: skip the work for an input whose
     [requires_grad] is false. *)
  let propagate v g =
    match v.op with
    | Leaf -> ()
    | Add (a, b) ->
        (* worked example: d(a+b)/da = d(a+b)/db = 1 *)
        accumulate a (T.sum_to_shape g (T.shape a.value));
        accumulate b (T.sum_to_shape g (T.shape b.value))
    (* TODO [Phase 2, easy]  Sub: da = g, db = -g *)
    | Sub _ -> Todo.todo "backward Sub"
    (* TODO [Phase 2, easy]  Mul: da = g * b, db = g * a *)
    | Mul _ -> Todo.todo "backward Mul"
    (* TODO [Phase 2, easy]  Div: da = g / b, db = -g * a / b^2 *)
    | Div _ -> Todo.todo "backward Div"
    (* TODO [Phase 2, easy]  Neg: da = -g;  Scale: da = k * g *)
    | Neg _ -> Todo.todo "backward Neg"
    | Scale _ -> Todo.todo "backward Scale"
    (* TODO [Phase 2, easy]  Pow: da = g * p * a^(p-1) *)
    | Pow _ -> Todo.todo "backward Pow"
    (* TODO [Phase 2, easy]  Exp: da = g * out  (reuse the forward value!) *)
    | Exp _ -> Todo.todo "backward Exp"
    (* TODO [Phase 2, easy]  Log: da = g / a *)
    | Log _ -> Todo.todo "backward Log"
    (* TODO [Phase 2, medium]  Matmul (out = a @ b):
         da = g @ transpose b      db = transpose a @ g
       Sanity-check the shapes on paper: [m;k] x [k;n]. *)
    | Matmul _ -> Todo.todo "backward Matmul"
    (* TODO [Phase 2, easy]  Transpose: da = transpose g *)
    | Transpose _ -> Todo.todo "backward Transpose"
    (* TODO [Phase 2, easy]  Reshape: da = reshape g (shape of a) *)
    | Reshape _ -> Todo.todo "backward Reshape"
    (* TODO [Phase 2, easy]  Sum: g is a scalar; da = broadcast_to g (shape a)
                             Mean: same, divided by numel a *)
    | Sum _ -> Todo.todo "backward Sum"
    | Mean _ -> Todo.todo "backward Mean"
    (* TODO [Phase 2, medium]  Sum_axis (a, axis): g lacks [axis] (unless
       keepdims was used). Reshape g to a's shape with a.shape.(axis) replaced
       by 1, then broadcast_to a's shape. *)
    | Sum_axis _ -> Todo.todo "backward Sum_axis"
    (* TODO [Phase 2, easy]  Relu: da = g where a > 0, else 0 (T.map2) *)
    | Relu _ -> Todo.todo "backward Relu"
    (* TODO [Phase 2, easy]  Sigmoid: da = g * out * (1 - out) *)
    | Sigmoid _ -> Todo.todo "backward Sigmoid"
    (* TODO [Phase 2, easy]  Tanh: da = g * (1 - out^2) *)
    | Tanh _ -> Todo.todo "backward Tanh"
    (* TODO [Phase 2, medium]  Log_softmax along the last axis:
         softmax = exp out
         da = g - softmax * sum_axis ~keepdims:true g (-1)
       Derive it once on paper; it's a classic interview question. *)
    | Log_softmax _ -> Todo.todo "backward Log_softmax"
    (* TODO [Phase 6]  Im2col: da = T.col2im g ~input_shape:(shape of input) ... *)
    | Im2col _ -> Todo.todo "backward Im2col"
    (* TODO [Phase 6]  Max_pool2d: da = T.max_pool2d_backward ~grad_out:g ... *)
    | Max_pool2d _ -> Todo.todo "backward Max_pool2d"

  (* TODO [Phase 2, medium]: return every node reachable from [root] in
     topological order: each node appears AFTER all of its [parents].
     - DFS post-order: visit(v) = if not seen then (mark seen; visit each
       parent; push v). Use a [Hashtbl] keyed on [v.id] for "seen".
     - Don't descend into nodes with [requires_grad = false]; nothing below
       them needs a gradient.
     - Deep graphs (e.g. RNNs) can overflow the stack with naive recursion;
       an explicit stack is a good follow-up but not needed for MLPs/CNNs. *)
  let topo_sort root = Todo.todo "Var.topo_sort"

  (* TODO [Phase 2, medium]:
     1. Raise [Invalid_argument] unless [T.numel root.value = 1]; gradients
        of non-scalar outputs need a seed vector, which we don't support.
     2. Seed: accumulate root (T.ones (shape root)).  (d loss / d loss = 1)
     3. Walk [topo_sort root] in REVERSE (outputs first). For each node whose
        grad is [Some g], call [propagate node g].
     Because of the ordering, a node's grad is complete (all its consumers
     have contributed) before it is propagated further. *)
  let backward root = Todo.todo "Var.backward"

  module Infix = struct
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
  end
end

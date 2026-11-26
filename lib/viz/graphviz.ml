(* TODO [Phase 6, easy-medium]:
   - Traverse the graph from [root] with Var.parents (DFS + Hashtbl of seen
     ids, like topo_sort but descending into every node).
   - Emit with a Buffer:
       digraph G {
         rankdir=LR;
         n12 [shape=box, label="matmul\n[64; 10]"];
         n7 -> n12;
         ...
       }
     Label: Var.op_name v, the shape, and maybe "grad" if Var.grad is Some.
   - Nice touches: color params differently (op_name = "param"), and show
     the scalar value on the root (the loss). *)
let to_dot root = Todo.todo "Graphviz.to_dot"

let save_dot path v =
  Out_channel.with_open_text path (fun oc -> output_string oc (to_dot v))

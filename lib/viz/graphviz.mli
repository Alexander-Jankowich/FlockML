(** Export a computational graph to Graphviz DOT format (stretch goal).

    Render with: [dot -Tsvg graph.dot -o graph.svg] *)

val to_dot : Var.t -> string
(** All nodes reachable from the given output, one box per node labelled with
    its op name and shape, with an edge from each input to its consumer. *)

val save_dot : string -> Var.t -> unit
(** [save_dot path v] writes [to_dot v] to [path]. *)

(** A node's shape as ["[64; 10]"], for labels. *)
let shape_string v =
  "["
  ^ String.concat "; " (Array.to_list (Array.map string_of_int (Var.shape v)))
  ^ "]"

(** Colour by role so the structure reads at a glance: trainable weights,
    constant inputs, the output (loss), and everything in between. *)
let fill_color ~is_root v =
  if is_root then "salmon"
  else
    match Var.op_name v with
    | "param" -> "lightblue"
    | "const" -> "lightgray"
    | _ -> "white"

(** A depth-first walk from [root] that emits each node once, followed by an
    edge from each of its inputs. Each label holds the op name and shape, the
    value if the root is a scalar (the loss), and [(grad)] once a gradient has
    been computed. *)
let to_dot root =
  let buf = Buffer.create 4096 in
  let seen = Hashtbl.create 64 in
  Buffer.add_string buf "digraph G {\n";
  Buffer.add_string buf "  rankdir=LR;\n";
  Buffer.add_string buf
    "  node [shape=box, style=filled, fontname=\"Helvetica\"];\n";
  (* Unlike topo_sort, descend into every node (constants too): the picture
     should show the whole graph, not just the differentiable part. *)
  let rec visit v =
    let id = Var.id v in
    if not (Hashtbl.mem seen id) then begin
      Hashtbl.add seen id ();
      let is_root = id = Var.id root in
      (* "\n" inside a DOT string is a line break in the rendered label. *)
      let label =
        Var.op_name v ^ "\\n" ^ shape_string v
        ^ (if is_root && Array.fold_left ( * ) 1 (Var.shape v) = 1 then
             Printf.sprintf "\\n= %g" (Tensor.item (Var.value v))
           else "")
        ^ match Var.grad v with Some _ -> "\\n(grad)" | None -> ""
      in
      Printf.bprintf buf "  n%d [label=\"%s\", fillcolor=%s];\n" id label
        (fill_color ~is_root v);
      List.iter
        (fun p ->
          visit p;
          Printf.bprintf buf "  n%d -> n%d;\n" (Var.id p) id)
        (Var.parents v)
    end
  in
  visit root;
  Buffer.add_string buf "}\n";
  Buffer.contents buf

(** Writes the DOT text as a plain text file, replacing any existing one. *)
let save_dot path v =
  Out_channel.with_open_text path (fun oc -> output_string oc (to_dot v))

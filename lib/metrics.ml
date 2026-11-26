let predict scores = Tensor.argmax_axis scores 1

(* TODO [Phase 5, easy]: count matches / length. Raise Invalid_argument on a
   length mismatch, and decide what an empty input should return. *)
let accuracy ~predictions ~labels = Todo.todo "Metrics.accuracy"

(* TODO [Phase 5, easy]: Array.make_matrix num_classes num_classes 0, then
   increment m.(labels.(i)).(predictions.(i)) for every i. The diagonal is the
   correct predictions; off-diagonal cells show which digits get confused. *)
let confusion_matrix ~num_classes ~predictions ~labels =
  Todo.todo "Metrics.confusion_matrix"

let pp_confusion_matrix fmt m =
  let n = Array.length m in
  Format.fprintf fmt "@[<v>true\\pred";
  for j = 0 to n - 1 do
    Format.fprintf fmt "%6d" j
  done;
  Array.iteri
    (fun i row ->
      Format.fprintf fmt "@,%9d" i;
      Array.iter (fun c -> Format.fprintf fmt "%6d" c) row)
    m;
  Format.fprintf fmt "@]"

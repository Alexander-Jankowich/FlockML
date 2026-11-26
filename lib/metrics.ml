let predict scores = Tensor.argmax_axis scores 1

(* TODO [Phase 5, easy]: count matches / length. Raise Invalid_argument on a
   length mismatch, and decide what an empty input should return. *)
let check_lengths ~fn ~predictions ~labels =
  let np = Array.length predictions and nl = Array.length labels in
  if np <> nl then
    invalid_arg
      (Printf.sprintf "Metrics.%s: %d predictions but %d labels" fn np nl)

(* Empty input returns [nan]: accuracy of zero examples is 0/0, undefined.
   Returning 0.0 would make an empty eval set look like a terrible model, and
   1.0 like a perfect one; nan is impossible to mistake for a real score. *)
let accuracy ~predictions ~labels =
  check_lengths ~fn:"accuracy" ~predictions ~labels;
  let n = Array.length labels in
  if n = 0 then Float.nan
  else begin
    let correct = ref 0 in
    Array.iteri
      (fun i label -> if predictions.(i) = label then incr correct)
      labels;
    float_of_int !correct /. float_of_int n
  end

(* TODO [Phase 5, easy]: Array.make_matrix num_classes num_classes 0, then
   increment m.(labels.(i)).(predictions.(i)) for every i. The diagonal is the
   correct predictions; off-diagonal cells show which digits get confused. *)
let confusion_matrix ~num_classes ~predictions ~labels =
  check_lengths ~fn:"confusion_matrix" ~predictions ~labels;
  let in_range c = c >= 0 && c < num_classes in
  let m = Array.make_matrix num_classes num_classes 0 in
  Array.iteri
    (fun i label ->
      let pred = predictions.(i) in
      if not (in_range label && in_range pred) then
        invalid_arg
          (Printf.sprintf
             "Metrics.confusion_matrix: position %d has label %d, prediction \
              %d; both must be in [0, %d)"
             i label pred num_classes);
      (* rows = true class, columns = predicted class *)
      m.(label).(pred) <- m.(label).(pred) + 1)
    labels;
  m

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

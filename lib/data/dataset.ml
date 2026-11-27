type t = { inputs : Tensor.t; labels : int array }
(** Inputs stacked along axis 0, with one label per row. *)

(** The number of examples, i.e. the number of labels. *)
let length d = Array.length d.labels

(** Fisher-Yates shuffle: for [i] from [n - 1] down to 1, swap slot [i] with a
    random slot in [[0, i]]. Every permutation is equally likely, unlike the
    tempting "swap with any index" loop, which is biased. *)
let shuffled_indices rng n =
  let idx = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    (* Pick from the not-yet-fixed prefix [0, i] only, then lock slot i. *)
    let j = Random.State.int rng (i + 1) in
    let tmp = idx.(i) in
    idx.(i) <- idx.(j);
    idx.(j) <- tmp
  done;
  idx

(** Use the given generator, or seed a fresh one from the system so results
    differ per run. Pass [~rng] explicitly for reproducible experiments. *)
let get_rng = function Some r -> r | None -> Random.State.make_self_init ()

(** Build a sub-dataset from the given row indices (copies the rows). *)
let take d idx =
  {
    inputs = Tensor.take_rows d.inputs idx;
    labels = Array.map (fun i -> d.labels.(i)) idx;
  }

(** A lazy sequence: each batch is copied out only when it is consumed, so an
    epoch never holds the whole shuffled dataset twice. The order (shuffled or
    not) is fixed when the sequence is created. Raises [Invalid_argument] if
    [batch_size <= 0].

    With [~reuse:true], every full-size batch is copied into the same
    tensor, created once per sequence. That saves allocating a fresh batch
    (400 KB for MNIST) every step, which matters once a step takes well
    under a millisecond. *)
let batches ?(shuffle = false) ?(reuse = false) ?rng ~batch_size d =
  if batch_size <= 0 then invalid_arg "Dataset.batches: batch_size must be > 0";
  let n = length d in
  (* Decided once, when the Seq is created, so one epoch = one permutation. *)
  let order =
    if shuffle then shuffled_indices (get_rng rng) n else Array.init n Fun.id
  in
  let buffer =
    lazy
      (let s = Tensor.shape d.inputs in
       s.(0) <- batch_size;
       Tensor.zeros s)
  in
  let rec from start () =
    if start >= n then Seq.Nil
    else
      let stop = min n (start + batch_size) in
      let idx = Array.sub order start (stop - start) in
      let labels = Array.map (fun i -> d.labels.(i)) idx in
      let inputs =
        if reuse && stop - start = batch_size then begin
          let dst = Lazy.force buffer in
          Tensor.take_rows_ d.inputs idx ~dst;
          dst
        end
        else Tensor.take_rows d.inputs idx
      in
      Seq.Cons ((inputs, labels), from stop)
  in
  from 0

(** Shuffle the indices, cut at [floor (ratio * n)], and copy out each half.
    Raises [Invalid_argument] unless [0 <= ratio <= 1]. *)
let split ?rng ~ratio d =
  if ratio < 0.0 || ratio > 1.0 then
    invalid_arg "Dataset.split: ratio must be in [0, 1]";
  let n = length d in
  let order = shuffled_indices (get_rng rng) n in
  let cut = int_of_float (ratio *. float_of_int n) in
  (take d (Array.sub order 0 cut), take d (Array.sub order cut (n - cut)))

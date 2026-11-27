(** Datasets and mini-batch iteration.

    Two levels:
    - {!S.t}, an in-memory labelled dataset (all inputs in one tensor, one
      integer class per row), with {!S.batches}. MNIST and CIFAR-10 load into
      it.
    - {!S.source}, any indexable collection of examples: in memory or read on
      demand, with integer labels or tensor targets (regression, multi-label,
      sequences). {!S.loader} turns one into an epoch of batches, shuffled,
      augmented with {!Transform}s, and prepared on a background domain while
      the training step runs. *)

module type S = sig
  type tensor
  (** The backend's tensor type. *)

  type t = {
    inputs : tensor;  (** [[| n; ... |]]: one example per row along axis 0 *)
    labels : int array;  (** [n] class labels *)
  }

  val length : t -> int
  (** The number of examples. *)

  val shuffled_indices : Random.State.t -> int -> int array
  (** A uniformly random permutation of [0 .. n-1]. *)

  val batches :
    ?shuffle:bool ->
    ?drop_last:bool ->
    ?reuse:bool ->
    ?rng:Random.State.t ->
    batch_size:int ->
    t ->
    (tensor * int array) Seq.t
  (** Mini-batches covering the dataset once (one epoch). The last batch may be
      smaller, unless [drop_last] (default [false]) leaves it out. [shuffle]
      defaults to [false].

      With [~reuse:true] (default [false]), every full-size batch is written
      into one tensor that the sequence allocates once, so a batch's inputs are
      only valid until the next batch is requested. This suits a training loop
      that finishes each step before taking the next batch, and saves an
      allocation per step. *)

  val split : ?rng:Random.State.t -> ratio:float -> t -> t * t
  (** [split ~ratio d] randomly partitions [d] into (first, second) with [ratio]
      of the examples in [first], e.g. a train/validation split. Without [rng]
      it draws from the global generator ({!Rng}), so it differs on every run
      unless {!Rng.manual_seed} was called. *)

  (** {1 General datasets} *)

  type ('example, 'batch) source = {
    length : int;  (** the number of examples *)
    get : int -> 'example;  (** one example *)
    batch : int array -> 'batch;
        (** the examples at these indices, collated into one batch *)
  }
  (** Any dataset: [get] reads one example, and [batch] several at once, which
      an in-memory dataset does with one copy rather than one per example. Both
      may read from disk or compute on demand. *)

  val labelled : t -> (tensor * int, tensor * int array) source
  (** An in-memory dataset as a source. An example is one input row (without the
      leading axis) and its label. *)

  val of_tensors : tensor -> tensor -> (tensor * tensor, tensor * tensor) source
  (** [of_tensors inputs targets]: example [i] is row [i] of each, so targets
      can be anything with one row per example: regression values, multi-label
      masks, sequences. Raises [Invalid_argument] if the first axes differ. *)

  val of_fn :
    length:int -> collate:('e array -> 'b) -> (int -> 'e) -> ('e, 'b) source
  (** [of_fn ~length ~collate get]: a lazy dataset, whose batches are its
      examples, read one by one and combined by [collate]. *)

  val subset : ('e, 'b) source -> int array -> ('e, 'b) source
  (** The examples at the given indices, in that order. *)

  val collate_labelled : (tensor * int) array -> tensor * int array
  (** Stack inputs along a new first axis, and gather the labels. *)

  val collate_pairs : (tensor * tensor) array -> tensor * tensor
  (** Stack both parts along a new first axis. *)

  val loader :
    ?shuffle:bool ->
    ?drop_last:bool ->
    ?rng:Random.State.t ->
    ?transform:(Random.State.t -> 'b -> 'b) ->
    ?prefetch:bool ->
    batch_size:int ->
    ('e, 'b) source ->
    'b Seq.t
  (** One epoch of batches of [source], in order or shuffled ([shuffle], default
      [false]), with the last short batch kept unless [drop_last]. [transform]
      is applied to every batch (for example [Transform.on_inputs] of an
      augmentation).

      With [prefetch] (default [true]), batch [k + 1] is read and transformed on
      a background domain ({!Background}) while the caller works on batch [k],
      so data loading overlaps the training step. The batches are the same
      either way: the order is fixed when the sequence is made, and the
      transforms draw from their own generator, split from [rng] (or the global
      one) at that point, in batch order. Traverse the sequence once. Raises
      [Invalid_argument] if [batch_size <= 0]. *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  type t = { inputs : T.t; labels : int array }
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

  (** Use the given generator, or the global default one (see {!Rng}). *)
  let get_rng = Rng.get

  (** Build a sub-dataset from the given row indices (copies the rows). *)
  let take d idx =
    {
      inputs = T.take_rows d.inputs idx;
      labels = Array.map (fun i -> d.labels.(i)) idx;
    }

  (** A lazy sequence: each batch is copied out only when it is consumed, so an
      epoch never holds the whole shuffled dataset twice. The order (shuffled or
      not) is fixed when the sequence is created. Raises [Invalid_argument] if
      [batch_size <= 0].

      With [~reuse:true], every full-size batch is copied into the same tensor,
      created once per sequence. That saves allocating a fresh batch (400 KB for
      MNIST) every step, which matters once a step takes well under a
      millisecond. *)
  let batches ?(shuffle = false) ?(drop_last = false) ?(reuse = false) ?rng
      ~batch_size d =
    if batch_size <= 0 then
      invalid_arg "Dataset.batches: batch_size must be > 0";
    let n = length d in
    let n = if drop_last then n / batch_size * batch_size else n in
    (* Decided once, when the Seq is created, so one epoch = one permutation. *)
    let order =
      if shuffle then shuffled_indices (get_rng rng) n else Array.init n Fun.id
    in
    let buffer =
      lazy
        (let s = T.shape d.inputs in
         s.(0) <- batch_size;
         T.zeros s)
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
            T.take_rows_ d.inputs idx ~dst;
            dst
          end
          else T.take_rows d.inputs idx
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

  (* ------------------------------------------------------------------------ *)
  (* General datasets                                                         *)
  (* ------------------------------------------------------------------------ *)

  type ('example, 'batch) source = {
    length : int;
    get : int -> 'example;
    batch : int array -> 'batch;
  }

  (** Row [i] of [x], without the leading axis. *)
  let row x i =
    let s = T.shape x in
    T.reshape (T.take_rows x [| i |]) (Array.sub s 1 (Array.length s - 1))

  let labelled d =
    {
      length = length d;
      get = (fun i -> (row d.inputs i, d.labels.(i)));
      batch =
        (fun idx ->
          (T.take_rows d.inputs idx, Array.map (fun i -> d.labels.(i)) idx));
    }

  let of_tensors x y =
    let n = (T.shape x).(0) in
    if T.ndim y = 0 || (T.shape y).(0) <> n then
      invalid_arg
        (Printf.sprintf "Dataset.of_tensors: %d inputs but targets of shape %s"
           n
           (Layout.shape_to_string (T.shape y)));
    {
      length = n;
      get = (fun i -> (row x i, row y i));
      batch = (fun idx -> (T.take_rows x idx, T.take_rows y idx));
    }

  let of_fn ~length ~collate get =
    { length; get; batch = (fun idx -> collate (Array.map get idx)) }

  let subset src idx =
    Array.iter
      (fun i ->
        if i < 0 || i >= src.length then
          invalid_arg
            (Printf.sprintf "Dataset.subset: index %d out of range for %d" i
               src.length))
      idx;
    {
      length = Array.length idx;
      get = (fun i -> src.get idx.(i));
      batch = (fun js -> src.batch (Array.map (fun j -> idx.(j)) js));
    }

  let stack xs = T.stack ~axis:0 (Array.to_list xs)
  let collate_labelled exs = (stack (Array.map fst exs), Array.map snd exs)
  let collate_pairs exs = (stack (Array.map fst exs), stack (Array.map snd exs))

  (** Batch [k] is made by one job on the background domain, which starts when
      batch [k - 1] is handed out, so at most one batch is prepared ahead. Each
      node of the sequence is computed once (a [Lazy]), so forcing it again
      doesn't start another job. *)
  let loader ?(shuffle = false) ?(drop_last = false) ?rng ?transform
      ?(prefetch = true) ~batch_size src =
    if batch_size <= 0 then invalid_arg "Dataset.loader: batch_size must be > 0";
    let n = src.length in
    let rng = get_rng rng in
    let order =
      if shuffle then shuffled_indices rng n else Array.init n Fun.id
    in
    let count =
      if drop_last then n / batch_size else (n + batch_size - 1) / batch_size
    in
    let make =
      match transform with
      | None ->
          fun k ->
            let lo = k * batch_size in
            src.batch (Array.sub order lo (min batch_size (n - lo)))
      | Some f ->
          let trng = Random.State.split rng in
          fun k ->
            let lo = k * batch_size in
            f trng (src.batch (Array.sub order lo (min batch_size (n - lo))))
    in
    if not prefetch then Seq.init count make
    else
      let start k =
        if k < count then Some (Background.spawn (fun () -> make k)) else None
      in
      let rec from k job =
        let node =
          lazy
            (match job with
            | None -> Seq.Nil
            | Some j ->
                let b = Background.await j in
                Seq.Cons (b, from (k + 1) (start (k + 1))))
        in
        fun () -> Lazy.force node
      in
      (* The first job starts when the sequence is first forced. *)
      let first = lazy (from 0 (start 0)) in
      fun () -> (Lazy.force first) ()
end

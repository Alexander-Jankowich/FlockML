(** One background domain that runs jobs while the caller does other work: the
    data loader prepares the next batch on it while the current training step
    runs.

    It is separate from {!Parallel}'s pool on purpose. [Parallel] runs a nested
    call sequentially, so a batch prepared inside the pool could never overlap a
    step that uses the pool. Kernels called on the background domain run
    sequentially ({!Parallel.sequentially}): if they used the pool, the training
    step's kernels would run on one thread while they held it.

    Jobs run one at a time, in the order they were spawned. The domain is
    started on first use and stopped at exit. *)

type 'a outcome =
  | Pending
  | Done of 'a
  | Failed of exn * Printexc.raw_backtrace

type 'a future = {
  mutable outcome : 'a outcome;
  lock : Mutex.t;
  ready : Condition.t;
}
(** The result of a job that may still be running. *)

let jobs : (unit -> unit) Queue.t = Queue.create ()
let lock = Mutex.create ()
let wake = Condition.create ()
let stopping = ref false
let worker : (Domain.id * unit Domain.t) option ref = ref None

let rec loop () =
  Mutex.lock lock;
  while Queue.is_empty jobs && not !stopping do
    Condition.wait wake lock
  done;
  if Queue.is_empty jobs then Mutex.unlock lock
  else begin
    let job = Queue.pop jobs in
    Mutex.unlock lock;
    job ();
    loop ()
  end

let stop () =
  match !worker with
  | None -> ()
  | Some (_, d) ->
      Mutex.lock lock;
      stopping := true;
      Condition.broadcast wake;
      Mutex.unlock lock;
      Domain.join d;
      worker := None;
      stopping := false

(** Start the domain, once. Called with [lock] held. *)
let ensure_started () =
  if !worker = None then begin
    let d = Domain.spawn (fun () -> Parallel.sequentially loop) in
    worker := Some (Domain.get_id d, d);
    at_exit stop
  end

let on_worker () =
  match !worker with Some (id, _) -> Domain.self () = id | None -> false

let finish fut outcome =
  Mutex.lock fut.lock;
  fut.outcome <- outcome;
  Condition.broadcast fut.ready;
  Mutex.unlock fut.lock

let run f =
  match f () with
  | v -> Done v
  | exception e -> Failed (e, Printexc.get_raw_backtrace ())

(** [spawn f] starts [f ()] on the background domain and returns at once. Called
    from the background domain itself, it runs [f] straight away, since a job
    waiting for a later job would never finish. *)
let spawn f =
  let fut =
    { outcome = Pending; lock = Mutex.create (); ready = Condition.create () }
  in
  if on_worker () then fut.outcome <- run f
  else begin
    Mutex.lock lock;
    ensure_started ();
    Queue.push (fun () -> finish fut (run f)) jobs;
    Condition.signal wake;
    Mutex.unlock lock
  end;
  fut

(** Wait for a job's result. An exception the job raised is re-raised here, with
    its backtrace. *)
let await fut =
  Mutex.lock fut.lock;
  while match fut.outcome with Pending -> true | _ -> false do
    Condition.wait fut.ready fut.lock
  done;
  let o = fut.outcome in
  Mutex.unlock fut.lock;
  match o with
  | Done v -> v
  | Failed (e, bt) -> Printexc.raise_with_backtrace e bt
  | Pending -> assert false

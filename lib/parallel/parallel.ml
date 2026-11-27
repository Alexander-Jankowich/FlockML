(* A small pool of OCaml 5 domains running [parallel_for] loops.

   One job runs at a time. The caller publishes it under [lock], bumps
   [generation] and wakes the workers. Then every participant, the caller
   included, claims chunks by incrementing the job's [next] counter, so a
   slow core simply ends up taking fewer chunks. Whoever finishes the last
   chunk wakes the caller. Workers sleep on a condition variable between
   jobs, so an idle pool costs nothing.

   Nested calls (a kernel invoked from inside another kernel's chunk) and
   calls that arrive while another job is running take the sequential path,
   so a parallel_for can never deadlock waiting for itself. *)

type job = {
  body : int -> int -> unit;
  n : int;
  chunk : int;
  chunks : int;
  next : int Atomic.t;  (** next chunk to claim *)
  remaining : int Atomic.t;  (** chunks not yet finished *)
  mutable error : exn option;  (** first exception raised by a chunk *)
}

let lock = Mutex.create ()
let work_ready = Condition.create ()
let job_done = Condition.create ()
let current : job option ref = ref None
let generation = ref 0
let shutting_down = ref false
let workers : unit Domain.t list ref = ref []

(** Set while a job is running; a second job runs sequentially instead. *)
let busy = Atomic.make false

let default_threads () =
  match Sys.getenv_opt "FLOCKML_THREADS" with
  | Some s -> (
      match int_of_string_opt (String.trim s) with
      | Some n when n >= 1 -> n
      | _ -> invalid_arg "FLOCKML_THREADS must be a positive integer")
  | None ->
      (* Capped at 8: on an 8+4-core Apple M2 Pro, training ran fastest with
         8 threads and got slower with 10 or 12, because the slower
         efficiency cores hold back every synchronous kernel. Set
         FLOCKML_THREADS to override. *)
      min 8 (Domain.recommended_domain_count ())

let threads = ref None

let num_threads () =
  match !threads with
  | Some n -> n
  | None ->
      let n = default_threads () in
      threads := Some n;
      n

(** Work, in rough nanoseconds of single-threaded time, below which a loop runs
    sequentially. Waking the pool costs a few microseconds, so smaller jobs lose
    more to synchronisation than they gain. *)
let min_parallel_work =
  ref
    (match Sys.getenv_opt "FLOCKML_MIN_PARALLEL_WORK" with
    | Some s -> ( match int_of_string_opt s with Some n -> n | None -> 50_000)
    | None -> 50_000)

let run_chunks job =
  let rec loop () =
    let c = Atomic.fetch_and_add job.next 1 in
    if c < job.chunks then begin
      let lo = c * job.chunk in
      let hi = min job.n (lo + job.chunk) in
      (try job.body lo hi
       with e ->
         Mutex.lock lock;
         if job.error = None then job.error <- Some e;
         Mutex.unlock lock);
      if Atomic.fetch_and_add job.remaining (-1) = 1 then begin
        Mutex.lock lock;
        Condition.broadcast job_done;
        Mutex.unlock lock
      end;
      loop ()
    end
  in
  loop ()

let rec worker seen =
  Mutex.lock lock;
  while (not !shutting_down) && !generation = seen do
    Condition.wait work_ready lock
  done;
  if !shutting_down then Mutex.unlock lock
  else begin
    let gen = !generation and job = !current in
    Mutex.unlock lock;
    Option.iter run_chunks job;
    worker gen
  end

let stop_workers () =
  Mutex.lock lock;
  shutting_down := true;
  Condition.broadcast work_ready;
  Mutex.unlock lock;
  List.iter Domain.join !workers;
  workers := [];
  shutting_down := false

let () = at_exit stop_workers

(** Spawn workers until there are [num_threads () - 1] of them. *)
let ensure_workers () =
  let wanted = num_threads () - 1 in
  let have = List.length !workers in
  if have < wanted then begin
    Mutex.lock lock;
    let gen = !generation in
    Mutex.unlock lock;
    for _ = have + 1 to wanted do
      workers := Domain.spawn (fun () -> worker gen) :: !workers
    done
  end

let set_num_threads n =
  if n < 1 then invalid_arg "Parallel.set_num_threads: need at least 1 thread";
  if n < List.length !workers + 1 then stop_workers ();
  threads := Some n

let sequential body n = if n > 0 then body 0 n

let parallel_for ?(work = 1) n body =
  let threads = num_threads () in
  let total = n * max work 1 in
  if threads <= 1 || n <= 1 || total < !min_parallel_work then sequential body n
  else if not (Atomic.compare_and_set busy false true) then sequential body n
  else begin
    Fun.protect
      ~finally:(fun () -> Atomic.set busy false)
      (fun () ->
        ensure_workers ();
        (* Enough chunks to balance fast and slow cores, but none smaller
           than the sequential threshold's share. *)
        let max_chunks = threads * 4 in
        let min_items = max 1 (!min_parallel_work / 4 / max work 1) in
        let chunks = max 1 (min max_chunks (n / min_items)) in
        let chunk = (n + chunks - 1) / chunks in
        let chunks = (n + chunk - 1) / chunk in
        let job =
          {
            body;
            n;
            chunk;
            chunks;
            next = Atomic.make 0;
            remaining = Atomic.make chunks;
            error = None;
          }
        in
        Mutex.lock lock;
        current := Some job;
        incr generation;
        Condition.broadcast work_ready;
        Mutex.unlock lock;
        run_chunks job;
        Mutex.lock lock;
        while Atomic.get job.remaining > 0 do
          Condition.wait job_done lock
        done;
        current := None;
        Mutex.unlock lock;
        Option.iter raise job.error)
  end

(** Data parallelism over OCaml 5 domains, used by the tensor kernels.

    Every kernel that uses this splits its output into disjoint pieces, and each
    output element is computed by exactly one task in a fixed order. As a
    result, every tensor operation gives bit-identical results whatever the
    thread count.

    The thread count defaults to the [FLOCKML_THREADS] environment variable, or
    failing that [min 8 (Domain.recommended_domain_count ())]. Beyond 8,
    machines that mix fast and slow cores got slower in our measurements. Worker
    domains are started on first use and stopped at exit. *)

val num_threads : unit -> int
(** The number of threads parallel loops use, including the caller's. *)

val set_num_threads : int -> unit
(** Change the thread count. [1] makes every kernel sequential. Raises
    [Invalid_argument] if the count is less than 1. *)

val parallel_for : ?work:int -> int -> (int -> int -> unit) -> unit
(** [parallel_for ~work n body] covers the indices [0] to [n - 1] with calls
    [body lo hi] (for [lo <= i < hi]) on disjoint ranges, possibly from several
    domains at once, and returns when all of them have finished. [work] is a
    rough cost per item, in nanoseconds of single-threaded time (default 1).
    Loops whose total work is too small to be worth splitting run sequentially
    in the caller. If a [body] call raises, the first exception is re-raised
    after every chunk has finished.

    [body] must only write to state owned by its own range, and should avoid
    allocating: OCaml's minor collections pause every domain. Nested and
    concurrent calls are safe and run sequentially. *)

val sequentially : (unit -> 'a) -> 'a
(** [sequentially f] runs [f] with every {!parallel_for} in the current domain
    sequential, without touching the pool. {!Background} runs its jobs this way,
    so preparing a batch never takes the pool from the training step running in
    another domain. Results are the same either way, since no kernel's output
    depends on the thread count. *)

val min_parallel_work : int ref
(** Total work (in nanoseconds) below which {!parallel_for} stays sequential.
    Defaults to 50,000, or the [FLOCKML_MIN_PARALLEL_WORK] environment variable.
    Exposed for benchmarking and tuning: MNIST training was equally fast
    anywhere from 25,000 to 100,000. *)

(** The global default random generator.

    Every function in FlockML that samples (the tensor samplers, weight
    initialisation, dropout, shuffling, train/validation splits) takes an
    optional [?rng]. When it is left out, they all draw from this one generator,
    so a single {!manual_seed} at the start of a program makes the whole run
    reproducible, as [torch.manual_seed] does. Without a call to {!manual_seed},
    the generator starts from a system seed and runs differ.

    The generator is shared by every backend and every domain, but it is not
    synchronised: draw from it on one domain at a time (normally the main one),
    or pass an explicit [~rng] to code running elsewhere. *)

val manual_seed : int -> unit
(** Reset the default generator to a fixed state determined by the seed. *)

val default : unit -> Random.State.t
(** The default generator itself. *)

val get : Random.State.t option -> Random.State.t
(** [get (Some rng)] is [rng]; [get None] is {!default}[ ()]. What every sampler
    does with its [?rng] argument. *)

val permutation : ?rng:Random.State.t -> int -> int array
(** A uniformly random permutation of [0 .. n - 1] (a Fisher–Yates shuffle). *)

(** The global default generator; see rng.mli. *)

let state = ref (Random.State.make_self_init ())
let manual_seed seed = state := Random.State.make [| seed |]
let default () = !state
let get = function Some rng -> rng | None -> !state

(** For [i] from [n - 1] down to 1, swap slot [i] with a uniformly chosen slot
    in [[0, i]]. *)
let permutation ?rng n =
  let rng = get rng in
  let a = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    let j = Random.State.int rng (i + 1) in
    let tmp = a.(i) in
    a.(i) <- a.(j);
    a.(j) <- tmp
  done;
  a

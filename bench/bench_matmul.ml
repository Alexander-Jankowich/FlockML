(** Matmul throughput, to compare loop orders and against NumPy
   (bench/numpy_baseline.py). Always run in release mode:

     dune exec --profile release bench/bench_matmul.exe *)

open Flockml

(** Average seconds per call of [f] over [repeats] calls, after one warm-up
    call. *)
let time_it ~repeats f =
  f ();
  (* warm-up *)
  let t0 = Unix.gettimeofday () in
  for _ = 1 to repeats do
    f ()
  done;
  (Unix.gettimeofday () -. t0) /. float_of_int repeats

let () =
  let rng = Random.State.make [| 0 |] in
  Printf.printf "%6s %12s %10s\n" "n" "ms/matmul" "GFLOP/s";
  List.iter
    (fun n ->
      let a = Tensor.randn ~rng [| n; n |]
      and b = Tensor.randn ~rng [| n; n |] in
      let repeats = max 1 (200_000_000 / (n * n * n)) in
      let secs = time_it ~repeats (fun () -> ignore (Tensor.matmul a b)) in
      let gflops = 2.0 *. float_of_int (n * n * n) /. secs /. 1e9 in
      Printf.printf "%6d %12.3f %10.2f\n%!" n (secs *. 1000.0) gflops)
    [ 64; 128; 256; 512 ]

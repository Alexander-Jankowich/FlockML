(* The performance kernels against plain references, the fused ops' gradients,
   gradient-buffer ownership and thread-count independence, on the default
   backend. The tests are in suites/kernel_tests.ml.
   Run with: dune exec test/test_kernels.exe

   Parallel.min_parallel_work is set to 0, so even tiny shapes are split into
   chunks across threads: the parallel code paths are exercised and checked,
   not only the sequential fallback. *)

let () = Flockml.Parallel.min_parallel_work := 0

module Suite = Flockml_tests.Kernel_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

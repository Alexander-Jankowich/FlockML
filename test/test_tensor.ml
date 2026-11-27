(* Tensor construction, broadcasting, reductions, linear algebra and in-place ops, on the default backend. The tests are in
   suites/tensor_tests.ml. Run with: dune exec test/test_tensor.exe *)

module Suite = Flockml_tests.Tensor_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

(* The NumPy-level tensor API against NumPy's own outputs, on the default
   backend. The tests are in suites/numpy_tests.ml.
   Run with: dune exec test/test_numpy.exe *)

module Suite = Flockml_tests.Numpy_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

(* Autograd graph mechanics and gradient checks, on the default backend. The tests are in
   suites/autograd_tests.ml. Run with: dune exec test/test_autograd.exe *)

module Suite = Flockml_tests.Autograd_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

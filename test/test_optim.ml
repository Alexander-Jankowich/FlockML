(* Optimizers against hand-computed updates, on the default backend. The tests are in
   suites/optim_tests.ml. Run with: dune exec test/test_optim.exe *)

module Suite = Flockml_tests.Optim_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

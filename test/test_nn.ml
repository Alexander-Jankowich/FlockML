(* Initialisation, losses, layers, data and metrics, on the default backend. The tests are in
   suites/nn_tests.ml. Run with: dune exec test/test_nn.exe *)

module Suite = Flockml_tests.Nn_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

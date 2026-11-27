(* Layers, model state, checkpoints and initialisation, on the default
   backend. The tests are in suites/layers_tests.ml.
   Run with: dune exec test/test_layers.exe *)

module Suite = Flockml_tests.Layers_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

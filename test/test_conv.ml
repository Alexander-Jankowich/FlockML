(* Convolution, pooling and graph export, on the default backend. The tests are in
   suites/conv_tests.ml. Run with: dune exec test/test_conv.exe *)

module Suite = Flockml_tests.Conv_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

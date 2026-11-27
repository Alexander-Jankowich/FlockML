(* The data pipeline: sources, the prefetching loader, transforms, the
   CIFAR-10 and CSV readers, and metrics, on the default backend. The tests
   are in suites/data_tests.ml. Run with: dune exec test/test_data.exe *)

module Suite = Flockml_tests.Data_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

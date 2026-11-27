(* Layers, losses, optimizers and schedules against PyTorch's own outputs, on
   the default backend. The tests are in suites/torch_tests.ml.
   Run with: dune exec test/test_torch.exe *)

module Suite = Flockml_tests.Torch_tests.Make (Flockml.Default)

let () = OUnit2.run_test_tt_main Suite.suite

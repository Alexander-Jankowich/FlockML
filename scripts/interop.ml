(* The FlockML half of scripts/check_torch_interop.py, which checks that
   checkpoints move between FlockML and PyTorch in both directions. `dune
   test` doesn't run it.

     dune exec scripts/interop.exe -- save DIR
       builds two models (a ResNet-style CNN and a small causal transformer),
       trains the CNN's batch-norm statistics a little, and writes for each
       model DIR/<model>.safetensors, an input DIR/<model>_x.npy and its
       evaluation-mode logits DIR/<model>_logits.npy.

     dune exec scripts/interop.exe -- load DIR
       builds the same models with other weights, loads
       DIR/<model>_torch.safetensors (written by PyTorch) into them, and writes
       their logits for DIR/<model>_x.npy to DIR/<model>_logits_flockml.npy.

   The Python script mirrors these models in PyTorch, so keep the two in
   step. *)

open Flockml

let cnn ~rng =
  Layer.sequential
    [
      Layer.conv2d ~rng ~bias:false ~padding:1 ~in_channels:3 ~out_channels:8
        ~kernel:3 ();
      Layer.batch_norm 8;
      Layer.relu;
      Layer.residual
        (Layer.sequential
           [
             Layer.conv2d ~rng ~groups:2 ~padding:1 ~in_channels:8
               ~out_channels:8 ~kernel:3 ();
             Layer.batch_norm 8;
             Layer.relu;
           ]);
      Layer.max_pool2d 2;
      Layer.conv2d ~rng ~stride:2 ~padding:2 ~dilation:2 ~in_channels:8
        ~out_channels:16 ~kernel:3 ();
      Layer.gelu ();
      Layer.global_avg_pool;
      Layer.linear ~rng ~in_dim:16 ~out_dim:10 ();
    ]

let vocab = 13
let dim = 16
let len = 6

let gpt ~rng =
  let block () =
    Layer.sequential
      [
        Layer.residual
          (Layer.sequential
             [
               Layer.layer_norm [| dim |];
               Layer.multi_head_attention ~rng ~causal:true ~dim ~heads:4 ();
             ]);
        Layer.residual
          (Layer.sequential
             [
               Layer.layer_norm [| dim |];
               Layer.linear ~rng ~in_dim:dim ~out_dim:(4 * dim) ();
               Layer.gelu ();
               Layer.linear ~rng ~in_dim:(4 * dim) ~out_dim:dim ();
             ]);
      ]
  in
  let tok = Layer.embedding ~rng ~num:vocab ~dim ()
  and pos = Layer.embedding ~rng ~num:len ~dim ()
  and blocks = Layer.sequential [ block (); block () ]
  and norm = Layer.layer_norm [| dim |]
  and head = Layer.linear ~rng ~bias:false ~in_dim:dim ~out_dim:vocab () in
  Layer.custom ~name:"GPT"
    ~children:
      [
        ("tok", tok);
        ("pos", pos);
        ("blocks", blocks);
        ("norm", norm);
        ("head", head);
      ]
    (fun ~train ?rng x ->
      let s = Var.shape x in
      let positions = Array.init (s.(0) * s.(1)) (fun i -> i mod s.(1)) in
      let h =
        Var.add (Layer.forward tok x)
          (Layer.embed pos ~shape:[| s.(0); s.(1) |] positions)
      in
      let f l x = Layer.forward ~train ?rng l x in
      f head (f norm (f blocks h)))

let models ~seed =
  let rng = Random.State.make [| seed |] in
  [ ("cnn", cnn ~rng); ("gpt", gpt ~rng) ]

let input name =
  let rng = Random.State.make [| 42 |] in
  match name with
  | "cnn" -> Tensor.randn ~rng [| 2; 8; 8; 3 |]
  | _ -> Tensor.randint ~rng ~lo:0 ~hi:vocab [| 2; len |]

let logits model x =
  Var.value (Var.no_grad (fun () -> Layer.forward model (Var.const x)))

let () =
  match Sys.argv with
  | [| _; "save"; dir |] ->
      List.iter
        (fun (name, model) ->
          let x = input name in
          (* a few training-mode passes, so the CNN's running statistics
             aren't their initial values *)
          if name = "cnn" then
            for i = 1 to 3 do
              ignore
                (Var.no_grad (fun () ->
                     Layer.forward ~train:true model
                       (Var.const (Tensor.add_scalar (float_of_int i) x))))
            done;
          let file ext = Filename.concat dir (name ^ ext) in
          Checkpoint.save (file ".safetensors") model;
          Npy.save (file "_x.npy") x;
          Npy.save (file "_logits.npy") (logits model x);
          Printf.printf "wrote %s: %d tensors\n" (file ".safetensors")
            (List.length (Layer.state_dict model)))
        (models ~seed:1)
  | [| _; "load"; dir |] ->
      List.iter
        (fun (name, model) ->
          let file ext = Filename.concat dir (name ^ ext) in
          ignore (Checkpoint.load (file "_torch.safetensors") model);
          Npy.save
            (file "_logits_flockml.npy")
            (logits model (Npy.load (file "_x.npy")));
          Printf.printf "loaded %s\n" (file "_torch.safetensors"))
        (models ~seed:2)
  | _ ->
      prerr_endline "usage: interop.exe (save | load) DIR";
      exit 2

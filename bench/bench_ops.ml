(** Per-op cost of everything that isn't matmul: elementwise ops, broadcasting,
    reductions, convolution helpers and the optimizer step, all on the shapes
    MNIST training uses. Always run in release mode:

    {v
    dune exec --profile release bench/bench_ops.exe
    dune exec --profile release bench/bench_ops.exe -- --backend blas
    v}

    Columns: microseconds per call, nanoseconds per element (of the larger of
    input and output), and OCaml heap bytes allocated per call. Output is a
    Markdown table. *)

open Flockml
open Bench_util

type case = { name : string; elements : int; run : unit -> unit }
(** One benchmark: a name, the number of elements it touches, and the call. *)

let case name elements f =
  { name; elements; run = (fun () -> ignore (Sys.opaque_identity (f ()))) }

(** The cases, on backend [B]. *)
module Cases (B : Flockml.S) = struct
  open B

  let rng = Random.State.make [| 0 |]
  let randn shape = Tensor.randn ~rng shape
  let numel = Array.fold_left ( * ) 1

  let elementwise =
    let mlp_act = randn [| 64; 128 |] and big = randn [| 1_000_000 |] in
    let big2 = randn [| 1_000_000 |] in
    [
      case "add [64;128] + [64;128]" (64 * 128) (fun () ->
          Tensor.add mlp_act mlp_act);
      case "add [1M] + [1M]" 1_000_000 (fun () -> Tensor.add big big2);
      case "mul [1M] * [1M]" 1_000_000 (fun () -> Tensor.mul big big2);
      case "scale [1M]" 1_000_000 (fun () -> Tensor.scale 0.5 big);
      case "exp [1M]" 1_000_000 (fun () -> Tensor.exp big);
      case "relu fwd [1M]" 1_000_000 (fun () -> Tensor.relu big);
      case "relu bwd [1M]" 1_000_000 (fun () ->
          Tensor.relu_backward ~input:big ~grad:big2);
    ]

  let broadcasting =
    let mlp_act = randn [| 64; 128 |] and mlp_bias = randn [| 128 |] in
    let conv_act = randn [| 50176; 8 |] and conv_bias = randn [| 8 |] in
    let logits = randn [| 64; 10 |] and row_max = randn [| 64; 1 |] in
    [
      case "bias add [64;128] + [128]" (64 * 128) (fun () ->
          Tensor.add mlp_act mlp_bias);
      case "bias add [50176;8] + [8]" (50176 * 8) (fun () ->
          Tensor.add conv_act conv_bias);
      case "row sub [64;10] - [64;1]" (64 * 10) (fun () ->
          Tensor.sub logits row_max);
      case "bias grad [64;128] -> [128]" (64 * 128) (fun () ->
          Tensor.sum_to_shape mlp_act [| 128 |]);
      case "bias grad [50176;8] -> [8]" (50176 * 8) (fun () ->
          Tensor.sum_to_shape conv_act [| 8 |]);
      case "broadcast_to [128] -> [64;128]" (64 * 128) (fun () ->
          Tensor.broadcast_to mlp_bias [| 64; 128 |]);
      case "broadcast_to [8] -> [50176;8]" (50176 * 8) (fun () ->
          Tensor.broadcast_to conv_bias [| 50176; 8 |]);
    ]

  let reductions =
    let logits = randn [| 64; 10 |] and act = randn [| 64; 128 |] in
    let input_t = randn [| 784; 64 |] in
    [
      case "sum_axis [64;10] axis -1" (64 * 10) (fun () ->
          Tensor.sum_axis ~keepdims:true logits (-1));
      case "max_axis [64;10] axis -1" (64 * 10) (fun () ->
          Tensor.max_axis ~keepdims:true logits (-1));
      case "sum_axis [64;128] axis 0" (64 * 128) (fun () ->
          Tensor.sum_axis act 0);
      case "log_softmax [64;10]" (64 * 10) (fun () -> Tensor.log_softmax logits);
      case "transpose [784;64]" (784 * 64) (fun () -> Tensor.transpose input_t);
    ]

  let convolution =
    let img1 = randn [| 64; 28; 28; 1 |] and img2 = randn [| 64; 14; 14; 8 |] in
    let cols1 = Tensor.im2col img1 ~kh:3 ~kw:3 ~stride:1 ~padding:1 in
    let cols2 = Tensor.im2col img2 ~kh:3 ~kw:3 ~stride:1 ~padding:1 in
    let act1 = randn [| 64; 28; 28; 8 |]
    and act2 = randn [| 64; 14; 14; 16 |] in
    let pooled1, argmax1 = Tensor.max_pool2d act1 ~size:2 ~stride:2 in
    let pooled2, argmax2 = Tensor.max_pool2d act2 ~size:2 ~stride:2 in
    [
      case "im2col [64;28;28;1] 3x3"
        (numel (Tensor.shape cols1))
        (fun () -> Tensor.im2col img1 ~kh:3 ~kw:3 ~stride:1 ~padding:1);
      case "im2col [64;14;14;8] 3x3"
        (numel (Tensor.shape cols2))
        (fun () -> Tensor.im2col img2 ~kh:3 ~kw:3 ~stride:1 ~padding:1);
      case "col2im -> [64;14;14;8] 3x3"
        (numel (Tensor.shape cols2))
        (fun () ->
          Tensor.col2im cols2 ~input_shape:[| 64; 14; 14; 8 |] ~kh:3 ~kw:3
            ~stride:1 ~padding:1);
      case "max_pool2d [64;28;28;8] 2x2"
        (numel (Tensor.shape act1))
        (fun () -> Tensor.max_pool2d act1 ~size:2 ~stride:2);
      case "max_pool2d [64;14;14;16] 2x2"
        (numel (Tensor.shape act2))
        (fun () -> Tensor.max_pool2d act2 ~size:2 ~stride:2);
      case "max_pool2d bwd [64;28;28;8]"
        (numel (Tensor.shape act1))
        (fun () ->
          Tensor.max_pool2d_backward ~grad_out:pooled1 ~argmax:argmax1
            ~input_shape:[| 64; 28; 28; 8 |]);
      case "max_pool2d bwd [64;14;14;16]"
        (numel (Tensor.shape act2))
        (fun () ->
          Tensor.max_pool2d_backward ~grad_out:pooled2 ~argmax:argmax2
            ~input_shape:[| 64; 14; 14; 16 |]);
    ]

  (** Optimizer steps over the MLP's parameters, with gradients filled in by one
      real backward pass. Each step updates the parameters, which doesn't change
      its cost. *)
  let optimizer =
    let model =
      let open Models (B) in
      build_model rng `Mlp
    in
    let params = Layer.params model in
    let x = Var.const (randn [| 64; 784 |]) in
    let labels = Array.init 64 (fun i -> i mod 10) in
    Var.backward (Loss.cross_entropy ~logits:(Layer.forward model x) ~labels);
    let n = Layer.num_params model in
    let adam = Optim.adam ~lr:1e-3 params
    and sgd = Optim.sgd ~momentum:0.9 ~lr:1e-3 params in
    [
      case "adam step (MLP, 101,770 params)" n (fun () -> Optim.step adam);
      case "sgd+momentum step (MLP)" n (fun () -> Optim.step sgd);
    ]

  let data =
    let train = randn [| 60000; 784 |] in
    let idx = Array.init 64 (fun i -> i * 937) in
    [
      case "take_rows 64 of [60000;784]" (64 * 784) (fun () ->
          Tensor.take_rows train idx);
    ]

  (** The NumPy-level ops a small transformer leans on, at its shapes: batch 32,
      sequence 64, 4 heads of 32, a vocabulary of 65. *)
  let sequence =
    let x = randn [| 32; 64; 128 |] and w = randn [| 128; 128 |] in
    let heads = randn [| 32; 64; 4; 32 |] in
    let q = randn [| 128; 64; 32 |] and k = randn [| 128; 64; 32 |] in
    let att = randn [| 128; 64; 64 |] in
    let causal =
      Tensor.init [| 64; 64 |] (fun i -> if i.(1) > i.(0) then 1. else 0.)
    in
    let logits = randn [| 2048; 65 |] and emb = randn [| 65; 128 |] in
    let tokens = Array.init 2048 (fun i -> i * 31 mod 65) in
    [
      case "permute [32;64;4;32] -> heads"
        (numel (Tensor.shape heads))
        (fun () -> Tensor.permute heads [| 0; 2; 1; 3 |]);
      case "concat 2x [32;64;128] axis -1"
        (2 * numel (Tensor.shape x))
        (fun () -> Tensor.concat ~axis:(-1) [ x; x ]);
      case "slice [32;64;128] -> [:, :, 64:]"
        (numel (Tensor.shape x) / 2)
        (fun () -> Tensor.slice x [ All; All; Range (64, 128) ]);
      case "matmul [32;64;128] x [128;128]"
        (numel (Tensor.shape x))
        (fun () -> Tensor.matmul x w);
      case "bmm q k^T [128;64;32] -> 64x64"
        (128 * 64 * 64)
        (fun () -> Tensor.bmm ~trans_b:true q k);
      case "bmm att v [128;64;64] x [..;32]"
        (128 * 64 * 32)
        (fun () -> Tensor.bmm att q);
      case "masked_fill [128;64;64] causal"
        (128 * 64 * 64)
        (fun () -> Tensor.masked_fill att causal (-1e9));
      case "logsumexp [2048;65] axis -1" (2048 * 65) (fun () ->
          Tensor.logsumexp logits (-1));
      case "gather [2048;65] one per row" (2048 * 65) (fun () ->
          Tensor.gather logits ~axis:1 tokens);
      case "index_select 2048 rows of [65;128]" (2048 * 128) (fun () ->
          Tensor.index_select emb ~axis:0 tokens);
    ]

  let groups =
    [
      ("elementwise", elementwise);
      ("broadcasting", broadcasting);
      ("reductions", reductions);
      ("convolution", convolution);
      ("optimizer", optimizer);
      ("data", data);
      ("sequence models", sequence);
    ]
end

let () =
  Arg.parse
    (Arg.align
       [
         backend_arg;
         ( "--threads",
           Arg.Int Parallel.set_num_threads,
           " Threads for the tensor kernels (default: FLOCKML_THREADS, or up \
            to 8)" );
       ])
    (fun _ -> ())
    "bench_ops [options]";
  let module B = (val backend ()) in
  let module C = Cases (B) in
  Printf.printf "FlockML ops, %s\n\n" (environment ());
  Printf.printf "| %-32s | %12s | %8s | %12s |\n" "op" "us/call" "ns/elem"
    "KB alloc";
  Printf.printf "|%s|%s|%s|%s|\n" (String.make 34 '-') (String.make 14 '-')
    (String.make 10 '-') (String.make 14 '-');
  List.iter
    (fun (group, cases) ->
      Printf.printf "| **%s** | | | |\n%!" group;
      List.iter
        (fun c ->
          let r = measure ~min_time:0.3 c.run in
          Printf.printf "| %-32s | %12.1f | %8.2f | %12.1f |\n%!" c.name
            (r.secs *. 1e6)
            (r.secs *. 1e9 /. float_of_int c.elements)
            (r.alloc_bytes /. 1024.))
        cases)
    C.groups

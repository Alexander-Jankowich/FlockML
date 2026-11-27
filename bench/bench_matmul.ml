(** Matmul throughput on square matrices and on the shapes that MNIST training
    actually multiplies, to compare against NumPy (bench/numpy_baseline.py,
    which uses the same shape list). Always run in release mode:

    {v dune exec --profile release bench/bench_matmul.exe v}

    The backward-pass rows time the computation as autograd performs it:
    [Tensor.gemm_] with [~trans_a] / [~trans_b], which reads the transposed
    operand in place. Output is a Markdown table. *)

open Flockml
open Bench_util

type case = {
  name : string;
  m : int;  (** rows of the result *)
  k : int;  (** shared dimension *)
  n : int;  (** columns of the result *)
  form : [ `NN | `TN | `NT ];
      (** [`NN]: a[m;k] b[k;n]; [`TN]: transpose a[k;m] b[k;n]; [`NT]: a[m;k]
          transpose b[n;k] *)
}

let square n =
  { name = Printf.sprintf "square %d" n; m = n; k = n; n; form = `NN }

let cases =
  List.map square [ 64; 128; 256; 512 ]
  @ [
      (* MLP 784-128-10, batch 64 *)
      { name = "mlp fc1 fwd"; m = 64; k = 784; n = 128; form = `NN };
      { name = "mlp fc1 dW (x^T g)"; m = 784; k = 64; n = 128; form = `TN };
      { name = "mlp fc2 fwd"; m = 64; k = 128; n = 10; form = `NN };
      { name = "mlp fc2 dW (h^T g)"; m = 128; k = 64; n = 10; form = `TN };
      { name = "mlp fc2 dX (g W^T)"; m = 64; k = 10; n = 128; form = `NT };
      (* CNN, batch 64: im2col rows are n*oh*ow *)
      { name = "cnn conv1 fwd"; m = 50176; k = 9; n = 8; form = `NN };
      { name = "cnn conv1 dW"; m = 9; k = 50176; n = 8; form = `TN };
      { name = "cnn conv2 fwd"; m = 12544; k = 72; n = 16; form = `NN };
      { name = "cnn conv2 dW"; m = 72; k = 12544; n = 16; form = `TN };
      { name = "cnn conv2 dcols"; m = 12544; k = 16; n = 72; form = `NT };
      { name = "cnn fc fwd"; m = 64; k = 784; n = 10; form = `NN };
    ]

let run rng c =
  let a_shape, b_shape =
    match c.form with
    | `NN -> ([| c.m; c.k |], [| c.k; c.n |])
    | `TN -> ([| c.k; c.m |], [| c.k; c.n |])
    | `NT -> ([| c.m; c.k |], [| c.n; c.k |])
  in
  let a = Tensor.randn ~rng a_shape and b = Tensor.randn ~rng b_shape in
  let out = Tensor.zeros [| c.m; c.n |] in
  let f =
    match c.form with
    | `NN -> fun () -> ignore (Tensor.matmul a b)
    | `TN -> fun () -> Tensor.gemm_ ~trans_a:true a b ~c:out
    | `NT -> fun () -> Tensor.gemm_ ~trans_b:true a b ~c:out
  in
  let r = measure f in
  let flops = 2.0 *. float_of_int (c.m * c.k * c.n) in
  Printf.printf "| %-20s | %5dx%5dx%5d | %s | %10.3f | %8.2f |\n%!" c.name c.m
    c.k c.n
    (match c.form with `NN -> "a b  " | `TN -> "a^T b" | `NT -> "a b^T")
    (r.secs *. 1000.0)
    (flops /. r.secs /. 1e9)

let () =
  Printf.printf "FlockML matmul, %s\n\n" (environment ());
  Printf.printf "| %-20s | %17s | %s | %10s | %8s |\n" "case" "mxkxn" "form "
    "ms/call" "GFLOP/s";
  Printf.printf "|%s|%s|%s|%s|%s|\n" (String.make 22 '-') (String.make 19 '-')
    "-------" (String.make 12 '-') (String.make 10 '-');
  let rng = Random.State.make [| 0 |] in
  List.iter (run rng) cases

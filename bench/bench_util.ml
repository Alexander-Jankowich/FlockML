(** Helpers shared by the benchmark executables: wall-clock timing, allocation
    counting, the [--backend] option, and the MNIST models used by
    [bin/train_mnist.ml], for any backend. *)

open Flockml

(** Wall-clock seconds. [Sys.time] would give process CPU time, which adds up
    the time of every core once work runs in parallel. *)
let now = Unix.gettimeofday

(** Bytes allocated on the OCaml heap so far (minor and major). *)
let allocated_bytes = Gc.allocated_bytes

type measurement = {
  secs : float;  (** median wall-clock seconds per call *)
  alloc_bytes : float;  (** OCaml heap bytes allocated per call *)
}

(** Calls [f] once to warm up, then keeps calling it until at least [min_time]
    seconds and [min_runs] calls have passed. Returns the median time per call,
    which is less sensitive to a stray GC or scheduler pause than the mean, and
    the mean allocation per call. *)
let measure ?(min_time = 0.5) ?(min_runs = 5) f =
  ignore (Sys.opaque_identity (f ()));
  let times = ref [] and runs = ref 0 in
  let a0 = allocated_bytes () and start = now () in
  while !runs < min_runs || now () -. start < min_time do
    let t0 = now () in
    ignore (Sys.opaque_identity (f ()));
    times := (now () -. t0) :: !times;
    incr runs
  done;
  let alloc = (allocated_bytes () -. a0) /. float_of_int !runs in
  let sorted = Array.of_list !times in
  Array.sort compare sorted;
  { secs = sorted.(Array.length sorted / 2); alloc_bytes = alloc }

(** The backends every benchmark can run on, chosen with [--backend]. *)
let backends : (string * (module Flockml.S)) list =
  [
    ("ocaml", (module Flockml.Default));
    ("blas", (module Flockml.Make (Tensor_blas.F64)));
    ("blas32", (module Flockml.Make (Tensor_blas.F32)));
  ]

let backend_name = ref "ocaml"

(** The backend selected with [--backend]. *)
let backend () = List.assoc !backend_name backends

(** The [--backend] command-line option. *)
let backend_arg =
  ( "--backend",
    Arg.Symbol (List.map fst backends, fun s -> backend_name := s),
    " Tensor backend: ocaml (pure OCaml, float64; the default), blas \
     (Tensor_blas.F64) or blas32 (Tensor_blas.F32)" )

let describe_backend () =
  match !backend_name with
  | "ocaml" -> "pure-OCaml backend (float64)"
  | "blas" -> Printf.sprintf "Tensor_blas.F64 (BLAS: %s)" Tensor_blas.blas
  | _ -> Printf.sprintf "Tensor_blas.F32 (BLAS: %s)" Tensor_blas.blas

(** A one-line description of the backend, build and machine, for result tables.
*)
let environment () =
  Printf.sprintf
    "%s; OCaml %s, %s, %d-bit, %d threads (of %d recommended domains)"
    (describe_backend ()) Sys.ocaml_version Sys.os_type Sys.word_size
    (Parallel.num_threads ())
    (Domain.recommended_domain_count ())

let shape_to_string s =
  "[" ^ String.concat ";" (Array.to_list (Array.map string_of_int s)) ^ "]"

(** Directory of the MNIST IDX files, if they are there. *)
let mnist_available dir =
  Sys.file_exists (Filename.concat dir "train-images-idx3-ubyte")

(** The benchmark data and models, for one backend. *)
module Models (B : Flockml.S) = struct
  open B

  (** MNIST training data, or random data of the same shape when the files are
      missing. Timings don't depend on the pixel values. *)
  let load_train ~data_dir ~flatten =
    if mnist_available data_dir then
      (Mnist.load ~flatten ~dir:data_dir Mnist.Train, "MNIST")
    else begin
      let rng = Random.State.make [| 0 |] in
      let shape =
        if flatten then [| 60000; 784 |] else [| 60000; 28; 28; 1 |]
      in
      let inputs = Tensor.rand_uniform ~rng ~lo:0. ~hi:1. shape in
      let labels = Array.init 60000 (fun _ -> Random.State.int rng 10) in
      ({ Dataset.inputs; labels }, "synthetic (no MNIST files found)")
    end

  (** The architectures from [bin/train_mnist.ml]. Keep the two in sync, so
      benchmark numbers describe the models people actually train. *)
  let build_model rng = function
    | `Mlp ->
        Layer.sequential
          [
            Layer.linear ~rng ~in_dim:784 ~out_dim:128 ();
            Layer.relu;
            Layer.linear ~rng ~in_dim:128 ~out_dim:10 ();
          ]
    | `Cnn ->
        Layer.sequential
          [
            Layer.conv2d ~rng ~padding:1 ~in_channels:1 ~out_channels:8
              ~kernel:3 ();
            Layer.relu;
            Layer.max_pool2d 2;
            Layer.conv2d ~rng ~padding:1 ~in_channels:8 ~out_channels:16
              ~kernel:3 ();
            Layer.relu;
            Layer.max_pool2d 2;
            Layer.flatten;
            Layer.linear ~rng ~in_dim:(7 * 7 * 16) ~out_dim:10 ();
          ]
end

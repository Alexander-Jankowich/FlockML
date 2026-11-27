(** Loader for CIFAR-10's binary format: run [scripts/download_cifar10.sh]
    first.

    Each file is a run of 3,073-byte records: a label byte (0 to 9), then the 32
    x 32 image as 1,024 red, 1,024 green and 1,024 blue bytes, each plane
    row-major. Images come out NHWC, [[| n; 32; 32; 3 |]], with pixels scaled to
    [0, 1]. *)

module type S = sig
  type tensor
  type dataset

  type ('e, 'b) source
  (** [Dataset.source]. *)

  type split =
    | Train
    | Test
        (** 50,000 training images ([data_batch_1.bin] to [data_batch_5.bin]) or
            10,000 test images ([test_batch.bin]). *)

  val classes : string array
  (** The class names, by label: airplane, automobile, bird... *)

  val mean : float array
  (** The per-channel mean of the training images, for [Transform.normalize]. *)

  val std : float array
  (** The per-channel standard deviation of the training images. *)

  val source :
    ?limit:int ->
    dir:string ->
    split ->
    (tensor * int, tensor * int array) source
  (** The images, kept as bytes (30 MB per 10,000) and converted to floats a
      batch at a time, so the training set takes 150 MB instead of 1.2 GB as
      float64. [limit] keeps the first [limit] images. Raises [Failure] if a
      file is missing or malformed. *)

  val load : ?limit:int -> dir:string -> split -> dataset
  (** Every image converted at once, as an in-memory dataset. *)
end

let record = 3073
let pixels = 3072

(** The file contents of a split, concatenated. *)
let read_split ~dir files =
  String.concat ""
    (List.map
       (fun f ->
         let path = Filename.concat dir f in
         if not (Sys.file_exists path) then
           failwith
             (Printf.sprintf
                "Cifar10: %s not found. Run scripts/download_cifar10.sh first."
                path);
         let s = In_channel.with_open_bin path In_channel.input_all in
         if String.length s mod record <> 0 then
           failwith
             (Printf.sprintf "Cifar10: %s is not a whole number of records" path);
         s)
       files)

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Dataset : Dataset.S with type tensor = T.t) :
  S
    with type tensor = T.t
     and type dataset = Dataset.t
     and type ('e, 'b) source = ('e, 'b) Dataset.source = struct
  type tensor = T.t
  type dataset = Dataset.t
  type ('e, 'b) source = ('e, 'b) Dataset.source
  type split = Train | Test

  let classes =
    [|
      "airplane";
      "automobile";
      "bird";
      "cat";
      "deer";
      "dog";
      "frog";
      "horse";
      "ship";
      "truck";
    |]

  let mean = [| 0.4914; 0.4822; 0.4465 |]
  let std = [| 0.2470; 0.2435; 0.2616 |]

  let files = function
    | Train -> List.init 5 (fun i -> Printf.sprintf "data_batch_%d.bin" (i + 1))
    | Test -> [ "test_batch.bin" ]

  (** Images [idx] of [bytes] as an NHWC tensor: the planes are interleaved into
      channels. *)
  let decode bytes idx =
    let n = Array.length idx in
    let out = Array.make (n * pixels) 0. in
    Array.iteri
      (fun b i ->
        let base = (i * record) + 1 and dst = b * pixels in
        for p = 0 to 1023 do
          for c = 0 to 2 do
            out.(dst + (p * 3) + c) <-
              float_of_int
                (Char.code (String.unsafe_get bytes (base + (c * 1024) + p)))
              /. 255.
          done
        done)
      idx;
    T.of_array [| n; 32; 32; 3 |] out

  let label bytes i = Char.code bytes.[i * record]

  let source ?limit ~dir split =
    let bytes = read_split ~dir (files split) in
    let n = String.length bytes / record in
    let n = match limit with Some k -> max 0 (min k n) | None -> n in
    {
      Dataset.length = n;
      get =
        (fun i ->
          (T.reshape (decode bytes [| i |]) [| 32; 32; 3 |], label bytes i));
      batch = (fun idx -> (decode bytes idx, Array.map (label bytes) idx));
    }

  let load ?limit ~dir split =
    let src = source ?limit ~dir split in
    let inputs, labels = src.batch (Array.init src.length Fun.id) in
    { Dataset.inputs; labels }
end

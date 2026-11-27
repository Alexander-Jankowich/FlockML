(** Loader for MNIST-format (IDX) files. Works for both MNIST and Fashion-MNIST:
    run [scripts/download_mnist.sh] first.

    Pixels are scaled to [0, 1]. *)

module type S = sig
  type dataset
  (** The dataset type that {!load} returns. *)

  (** Which half of the dataset to load: 60,000 training images or 10,000 test
      images. *)
  type split = Train | Test

  val load : ?flatten:bool -> ?limit:int -> dir:string -> split -> dataset
  (** [flatten] (default [true]) gives inputs of shape [[| n; 784 |]] for MLPs;
      [false] gives NHWC [[| n; 28; 28; 1 |]] for CNNs. [limit] keeps only the
      first [limit] examples, which is handy for quick experiments. *)
end

(** The module for one backend. *)
module Make (T : Tensor_intf.S) (Dataset : Dataset.S with type tensor = T.t) :
  S with type dataset = Dataset.t = struct
  type dataset = Dataset.t

  (** IDX format (big-endian):
      {v
      images: int32 magic = 2051, int32 n, int32 rows, int32 cols, then
              n * rows * cols unsigned bytes
      labels: int32 magic = 2049, int32 n, then n unsigned bytes
      v} *)

  (** Which half of the dataset to load. *)
  type split = Train | Test

  (** The whole file as a string, with a helpful error if it is missing. *)
  let read_file path =
    if not (Sys.file_exists path) then
      failwith
        (Printf.sprintf
           "Mnist: %s not found. Run scripts/download_mnist.sh first." path);
    In_channel.with_open_bin path In_channel.input_all

  (** The big-endian 32-bit integer at byte offset [off]. *)
  let int32_at s off = Int32.to_int (String.get_int32_be s off)

  (** Fail unless the file starts with the expected magic number. A mismatch
      usually means the file is still gzipped. *)
  let check_magic path s expected =
    let magic = int32_at s 0 in
    if magic <> expected then
      failwith
        (Printf.sprintf
           "Mnist: %s has magic number %d, expected %d (is it still gzipped?)"
           path magic expected)

  (** Read an IDX image file as [(n, rows, cols, pixels)], with pixels scaled
      from bytes to [[0, 1]]. *)
  let load_images path =
    let s = read_file path in
    check_magic path s 2051;
    let n = int32_at s 4 and rows = int32_at s 8 and cols = int32_at s 12 in
    let pixels =
      Array.init
        (n * rows * cols)
        (fun i -> float_of_int (Char.code s.[16 + i]) /. 255.0)
    in
    (n, rows, cols, pixels)

  (** Read an IDX label file as one integer class per example. *)
  let load_labels path =
    let s = read_file path in
    check_magic path s 2049;
    let n = int32_at s 4 in
    Array.init n (fun i -> Char.code s.[8 + i])

  (** The (images, labels) file names for each split. *)
  let file_names = function
    | Train -> ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    | Test -> ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

  (** Load both files, check they agree, keep the first [limit] examples and
      shape the pixels for an MLP (flat) or a CNN (NHWC). *)
  let load ?(flatten = true) ?limit ~dir split =
    let image_file, label_file = file_names split in
    let n, rows, cols, pixels = load_images (Filename.concat dir image_file) in
    let labels = load_labels (Filename.concat dir label_file) in
    if Array.length labels <> n then
      failwith "Mnist: image and label files have different lengths";
    let n = match limit with None -> n | Some k -> max 0 (min k n) in
    let pixels = Array.sub pixels 0 (n * rows * cols) in
    let shape =
      if flatten then [| n; rows * cols |] else [| n; rows; cols; 1 |]
    in
    { Dataset.inputs = T.of_array shape pixels; labels = Array.sub labels 0 n }
end

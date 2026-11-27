(** Loader for MNIST-format (IDX) files. Works for both MNIST and Fashion-MNIST:
    run [scripts/download_mnist.sh] first.

    Pixels are scaled to [0, 1]. *)

(** Which half of the dataset to load: 60,000 training images or 10,000 test
    images. *)
type split = Train | Test

val load : ?flatten:bool -> ?limit:int -> dir:string -> split -> Dataset.t
(** [flatten] (default [true]) gives inputs of shape [[| n; 784 |]] for MLPs;
    [false] gives NHWC [[| n; 28; 28; 1 |]] for CNNs. [limit] keeps only the
    first [limit] examples, which is handy for quick experiments. *)

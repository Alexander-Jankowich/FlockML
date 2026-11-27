(** FlockML: tensors, reverse-mode autodiff, layers, losses and optimizers.

    Everything above the tensors is written once, against the backend signature
    {!Tensor_intf.S}, and {!Make} builds the whole stack for a given backend.
    The modules at the top level ({!Tensor}, {!Var}, {!Layer}, {!Optim}...) are
    {!Default}, the stack for the dependency-free [float array] backend, so
    ordinary code just writes [open Flockml].

    To run the same model on another backend, apply {!Make} to it, e.g. with the
    optional BLAS library ([flockml.blas]):
    {[
    module B = Flockml.Make (Tensor_blas.F32)

    let model = B.Layer.sequential [ B.Layer.linear ~in_dim:784 ~out_dim:10 () ]
    ]} *)

(** {1 Building a stack} *)

(** Every module of FlockML that depends on the tensor type, for one backend.
    Each module's type equations tie it to the same {!Tensor} and {!Var}, so the
    parts fit together exactly as the default ones do. *)
module type S = sig
  module Tensor : Tensor_intf.S
  (** The numerical backend. *)

  module Var : Autograd_intf.S with type tensor = Tensor.t
  (** Autograd over {!Tensor}: [Autograd.Make (Tensor)]. *)

  module Init : Init.S with type tensor = Tensor.t
  (** Weight initialisation. *)

  module Layer : Layer.S with type var = Var.t
  (** Layers and models. *)

  module Loss : Loss.S with type var = Var.t
  (** Loss functions. *)

  module Optim : Optim.S with type var = Var.t
  (** SGD and Adam. *)

  module Dataset : Dataset.S with type tensor = Tensor.t
  (** In-memory datasets and mini-batches. *)

  module Mnist : Mnist.S with type dataset = Dataset.t
  (** The MNIST / Fashion-MNIST loader. *)

  module Metrics : Metrics.S with type tensor = Tensor.t
  (** Predictions, accuracy, confusion matrices. *)

  module Npy : Npy.S with type tensor = Tensor.t
  (** Reading and writing NumPy [.npy] files. *)

  module Graphviz : Graphviz.S with type var = Var.t
  (** Graph export to DOT. *)
end

(** [Make (T)] is FlockML on backend [T]: autograd, layers, losses, optimizers,
    data loading and metrics, all computing with [T]. *)
module Make (T : Tensor_intf.S) : S with module Tensor = T = struct
  module Tensor = T
  module Var = Autograd.Make (T)
  module Init = Init.Make (T)
  module Layer = Layer.Make (T) (Var)
  module Loss = Loss.Make (Var)
  module Optim = Optim.Make (T) (Var)
  module Dataset = Dataset.Make (T)
  module Mnist = Mnist.Make (T) (Dataset)
  module Metrics = Metrics.Make (T)
  module Npy = Npy.Make (T)
  module Graphviz = Graphviz.Make (T) (Var)
end

(** {1 Backend-independent modules} *)

module Tensor_intf = Tensor_intf
module Autograd_intf = Autograd_intf
module Autograd = Autograd
module Layout = Layout
module Tensor_derived = Tensor_derived
module Parallel = Parallel
module Rng = Rng

let manual_seed = Rng.manual_seed

(** {1 The default stack}

    [Make (Tensor)]: the pure-OCaml backend, with no dependencies. *)

module Tensor = Tensor
module Default = Make (Tensor)
module Var = Default.Var
module Init = Default.Init
module Layer = Default.Layer
module Loss = Default.Loss
module Optim = Default.Optim
module Dataset = Default.Dataset
module Mnist = Default.Mnist
module Metrics = Default.Metrics
module Npy = Default.Npy
module Graphviz = Default.Graphviz

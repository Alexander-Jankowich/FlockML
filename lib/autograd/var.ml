(** The concrete autograd engine: {!Autograd.Make} applied to the default
    [float array] backend {!Tensor}. Layers, losses and optimizers all use this
    module. To try another backend, apply the functor to it instead. *)

include Autograd.Make (Tensor)

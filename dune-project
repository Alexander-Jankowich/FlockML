(lang dune 3.0)

(name flockml)

(generate_opam_files true)

(source
 (github Alexander-Jankowich/FlockML))

(authors "Alexander Jankowich")

(maintainers "Alexander Jankowich")

(license MIT)

(package
 (name flockml)
 (synopsis "A small PyTorch-like neural network library written from scratch in OCaml")
 (description
  "Tensors, reverse-mode automatic differentiation, layers, losses, optimizers (SGD, Adam) and an MNIST training pipeline, built without external numerical libraries.")
 (depends
  (ocaml
   (>= 5.0))
  dune
  (ounit2 :with-test)))

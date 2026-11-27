# Changes

Changes that can break existing code, and changes in behaviour. New
functions and modules are listed in [ROADMAP.md](ROADMAP.md) with their
phase.

## Phases 11 and 12: the data pipeline and reference models

No breaking changes for code that uses the library. `Dataset.S` and
`Metrics.S` have new members (sources, the loader, top-k accuracy,
classification reports), and `Flockml.S` has new modules (`Cifar10`, `Csv`,
`Transform`), so only code that implements those signatures itself needs
updating. `Dataset.batches` gained an optional `?drop_last`.

## Phases 9 and 10: open layers, checkpoints, losses, optimizers, schedules

Breaking:
- **`Layer.t` has new constructors** (`Conv1d`, `Batch_norm`, `Attention`,
  `Custom`, `Residual`...), so an exhaustive `match` on a layer needs new
  cases.
- **`Layer.Linear` and `Layer.Conv2d` take an optional bias:** the record
  field `b` is now a `var option`. `Conv2d`'s `kernel`, `stride` and
  `padding` are now `(height, width)` pairs, and it has new fields
  `dilation` and `groups`. `Max_pool2d` has a new field, `padding`. The
  constructor functions (`Layer.linear`, `Layer.conv2d`, `Layer.max_pool2d`)
  are unchanged apart from new optional arguments, so only code that builds
  or matches the records directly needs changing:
  `Layer.Linear { w; b = Some b }`.
- **`Loss.cross_entropy` takes the logits and labels positionally:**
  `Loss.cross_entropy logits labels`, as PyTorch's
  `F.cross_entropy(input, target)`, instead of `~logits ~labels`. OCaml
  can't add optional arguments (`?reduction`, `?label_smoothing`,
  `?weights`) in front of labelled ones only, so this change was needed to
  add them. `Loss.nll` follows the same form.
- **Functor signatures:** `Loss.Make` takes the tensor backend as well
  (`Loss.Make (T) (Var)`), and `Layer.S`, `Loss.S` and `Optim.S` expose
  `type tensor`. `Flockml.Make` is unchanged.
- **`Tensor_intf.S` has four new kernels** (`adamw_update_`,
  `sgd_nesterov_update_`, `rmsprop_update_`, `adagrad_update_`), which a
  third-party backend has to implement.

Changes in behaviour:
- **Adam counts steps per parameter**, as PyTorch does: a parameter that
  had no gradient on some step doesn't advance its bias correction. When
  every parameter gets a gradient on every step, which is the usual case,
  the results are identical to before.

## Phases 7 and 8: the NumPy-level tensor API and complete autograd

Changes in behaviour:
- **Leaving out `?rng` uses the global default generator** (`Rng`), which
  `Flockml.manual_seed` seeds. Before, each call seeded a new generator from
  the system. A run is now reproducible with one `manual_seed` call.

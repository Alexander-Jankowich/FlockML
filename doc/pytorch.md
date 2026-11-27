# FlockML for PyTorch users

FlockML follows PyTorch's semantics wherever it can: the same losses,
optimizer updates, schedules and layer equations, checked against PyTorch's
own outputs by `test_torch`. What differs is mostly layout and style:

- **Layouts.** Images are NHWC (`[n; h; w; c]`), not NCHW. Weights are the
  transposes of PyTorch's, because a layer computes `x · w`: a linear
  layer's `w` is `[in; out]`. Convolution kernels are stored flattened as
  `[kh * kw * in_c; out_c]`. Classes are the *last* axis of the logits.
- **Values, not objects.** A model is a value of the variant `Layer.t`.
  Training and evaluation mode is an explicit `~train` argument, not
  `.train()` / `.eval()`. A learning-rate schedule is a function from the
  step to a rate.
- **Copies, not views.** Slicing and `permute` return new tensors. There
  is no `.contiguous()`, and no aliasing to worry about.
- **No dtype tag.** The precision is the backend's: `Flockml.Default`
  (float64, pure OCaml), or `Flockml.Make (Tensor_blas.F32)` / `F64`.
  Indices are `int array`s, and masks are 0/1 float tensors.

## Tensors

| PyTorch | FlockML |
|---|---|
| `torch.tensor(data).reshape(2, 3)` | `Tensor.of_array [| 2; 3 |] data` |
| `torch.zeros(2, 3)`, `ones`, `full` | `Tensor.zeros [| 2; 3 |]`, `ones`, `full` |
| `torch.arange(0, 10, 2)`, `linspace`, `eye` | `Tensor.arange ~start:0. ~step:2. 10.`, `linspace`, `eye` |
| `torch.randn(s)`, `rand`, `randint`, `bernoulli` | `Tensor.randn s`, `rand_uniform ~lo ~hi s`, `randint`, `bernoulli` |
| `torch.manual_seed(0)` | `Flockml.manual_seed 0` |
| `x.shape`, `x.numel()`, `x.item()` | `Tensor.shape x`, `numel x`, `item x` |
| `x + y`, `x * y` (broadcasting) | `Tensor.add x y`, `Tensor.mul x y` |
| `x @ y`, `torch.bmm` | `Tensor.matmul x y`, `Tensor.bmm ?trans_a ?trans_b` |
| `x.sum(dim=1, keepdim=True)` | `Tensor.sum_axis ~keepdims:true x 1` |
| `x.mean(dim=(1, 2))` | `Tensor.mean_axes x [ 1; 2 ]` |
| `x.max(dim=1).values`, `.argmax(1)` | `Tensor.max_axis x 1`, `argmax_axis x 1` |
| `x.var(dim=0, unbiased=True)` | `Tensor.var ~correction:1 x 0` |
| `torch.logsumexp(x, 1)`, `x.cumsum(1)` | `Tensor.logsumexp x 1`, `cumsum x 1` |
| `x.view(4, -1)` / `reshape` | `Tensor.reshape x [| 4; -1 |]` |
| `x.permute(0, 2, 1)`, `x.transpose(1, 2)` | `Tensor.permute x [| 0; 2; 1 |]`, `swapaxes x 1 2` |
| `x.squeeze(1)`, `x.unsqueeze(0)`, `x.flatten(1)` | `Tensor.squeeze ~axis:1 x`, `unsqueeze x 0`, `flatten ~first:1 x` |
| `x[:, 1:3, ::2]` | `Tensor.slice x [ All; Range (1, 3); Step (0, max_int, 2) ]` |
| `x[0]` | `Tensor.slice x [ Index 0 ]` |
| `x[:, 1:3] = y` | `Tensor.set_slice_ x [ All; Range (1, 3) ] y` |
| `torch.cat([a, b], 0)`, `stack`, `split`, `chunk` | `Tensor.concat ~axis:0 [ a; b ]`, `stack`, `split`, `chunk` |
| `F.pad(x, ...)`, `x.flip(1)`, `x.repeat(...)` | `Tensor.pad x [| (l, r); ... |]`, `flip x [ 1 ]`, `tile x reps` |
| `x.index_select(0, idx)`, `gather`, `scatter_add` | `Tensor.index_select x ~axis:0 idx`, `gather`, `scatter_add` |
| `x > y`, `torch.where(m, a, b)`, `masked_fill` | `Tensor.gt x y`, `where m a b`, `masked_fill x m v` |
| `x.clamp(min=0)`, `torch.maximum` | `Tensor.clamp ~min:0. x`, `maximum` |
| `np.save` / `np.load` | `Npy.save` / `Npy.load` |

## Autograd

| PyTorch | FlockML |
|---|---|
| `torch.tensor(..., requires_grad=True)` | `Var.param t` |
| a tensor without gradient | `Var.const t` |
| `loss.backward()` | `Var.backward loss` |
| `p.grad` | `Var.grad p` (an option) or `Var.grad_exn p` |
| `p.grad = None` | `Var.zero_grad p` |
| `x.detach()` | `Var.detach x` |
| `with torch.no_grad():` | `Var.no_grad (fun () -> ...)` |
| `p.requires_grad_(False)` | `Var.set_requires_grad p false` |
| `torch.autograd.Function` | `Var.custom ~name ~inputs ~value ~backward` |
| `torch.func.grad(f)(x)` | `Var.gradient f x`, or `Var.value_and_grad f x` |
| `torch.nn.utils.clip_grad_norm_(ps, 1.0)` | `Var.clip_grad_norm ~max_norm:1.0 ps` |
| `torch.autograd.detect_anomaly()` | `Var.detect_anomaly (fun () -> ...)` |
| `F.relu`, `F.gelu`, `F.silu`, `F.softplus`... | `Var.relu`, `Var.gelu`, `Var.silu`, `Var.softplus`... |
| `F.softmax(x, -1)`, `F.log_softmax(x, -1)` | `Var.softmax x`, `Var.log_softmax x` |

## Layers

| PyTorch | FlockML |
|---|---|
| `nn.Linear(i, o, bias=False)` | `Layer.linear ~in_dim:i ~out_dim:o ~bias:false ()` |
| `nn.Conv2d(i, o, 3, stride=2, padding=1, groups=g)` | `Layer.conv2d ~in_channels:i ~out_channels:o ~kernel:3 ~stride:2 ~padding:1 ~groups:g ()` |
| `nn.Conv1d`, `nn.ConvTranspose2d` | `Layer.conv1d`, `Layer.conv_transpose2d` |
| `nn.MaxPool2d(2)`, `nn.AvgPool2d(3, 2, 1)` | `Layer.max_pool2d 2`, `Layer.avg_pool2d ~stride:2 ~padding:1 3` |
| `nn.AdaptiveAvgPool2d(1)` + `Flatten` | `Layer.global_avg_pool` |
| `nn.BatchNorm1d(c)` / `BatchNorm2d(c)` | `Layer.batch_norm c` (either rank) |
| `nn.LayerNorm(d)`, `nn.GroupNorm(g, c)` | `Layer.layer_norm [| d |]`, `Layer.group_norm ~groups:g c` |
| `nn.Embedding(n, d)` | `Layer.embedding ~num:n ~dim:d ()` |
| `nn.LSTM(i, h, batch_first=True)`, `GRU`, `RNN` | `Layer.lstm ~input:i ~hidden:h ()`, `gru`, `rnn` (always batch first) |
| `nn.MultiheadAttention(d, h, batch_first=True)` | `Layer.multi_head_attention ~dim:d ~heads:h ()`; `~causal:true` for a causal mask |
| `nn.Dropout(p)`, `nn.Identity()` | `Layer.dropout p`, `Layer.identity` |
| `nn.Sequential(...)` | `Layer.sequential [ ... ]` |
| a subclass of `nn.Module` | `Layer.custom ~name ~params ~children forward` |
| `model(x)`, after `model.train()` | `Layer.forward ~train:true model x` |
| `model.parameters()`, `named_parameters()` | `Layer.params model`, `Layer.named_params model` |
| `model.buffers()` | `Layer.named_buffers model` |
| `model.state_dict()`, `load_state_dict(sd)` | `Layer.state_dict model`, `Layer.load_state_dict model sd` |
| `torchinfo.summary(model, input_size)` | `Layer.summary model ~input_shape` |
| `nn.init.kaiming_uniform_`, `xavier_normal_`, `orthogonal_` | `Init.kaiming_uniform`, `xavier_normal`, `orthogonal`, or `~init:Init.Orthogonal` on a constructor |

## Losses

| PyTorch | FlockML |
|---|---|
| `F.cross_entropy(logits, y)` | `Loss.cross_entropy logits y` |
| `F.cross_entropy(..., weight=w, label_smoothing=0.1)` | `Loss.cross_entropy ~weights:w ~label_smoothing:0.1 logits y` |
| `F.cross_entropy(logits.permute(0, 2, 1), y)` (sequences) | `Loss.cross_entropy logits y`, with `logits : [b; t; c]` and `y` flattened |
| `F.nll_loss`, `F.kl_div(..., reduction="batchmean")` | `Loss.nll`, `Loss.kl_div ~reduction:`Batch_mean` |
| `F.mse_loss`, `F.l1_loss`, `F.smooth_l1_loss`, `F.huber_loss` | `Loss.mse`, `l1`, `smooth_l1 ~beta`, `huber ~delta` |
| `F.binary_cross_entropy_with_logits(x, t, pos_weight=pw)` | `Loss.bce_with_logits ~pos_weight:pw x t` |
| `reduction="sum"` / `"none"` | ``~reduction:`Sum`` / ``~reduction:`None`` |

## Optimizers and schedules

| PyTorch | FlockML |
|---|---|
| `optim.SGD(ps, lr, momentum=0.9, nesterov=True)` | `Optim.sgd ~lr ~momentum:0.9 ~nesterov:true ps` |
| `optim.Adam`, `AdamW`, `RMSprop`, `Adagrad` | `Optim.adam`, `adamw`, `rmsprop`, `adagrad` |
| parameter groups `[{"params": b, "weight_decay": 0}]` | `~groups:[ Optim.group ~weight_decay:0. b ]` |
| `opt.zero_grad()`, `opt.step()` | `Optim.zero_grad opt`, `Optim.step opt` |
| `opt.state_dict()` | `Optim.state_dict opt` |
| `CosineAnnealingLR(opt, T)`, then `sched.step()` | `let s = Schedule.cosine ~lr ~total:t ()` and `Optim.set_lr opt (s step)` |
| `StepLR`, `MultiStepLR`, `ExponentialLR`, `LinearLR`, `OneCycleLR` | `Schedule.step_decay`, `multi_step`, `exponential`, `linear`, `one_cycle` |
| warmup then cosine | `Schedule.(warmup ~steps:100 (cosine ~lr ~total ()))` |
| `ReduceLROnPlateau` | `Schedule.Plateau.create lr`, then `Plateau.step p metric` |

## Data, metrics and checkpoints

| PyTorch | FlockML |
|---|---|
| `TensorDataset(x, y)` | `Dataset.of_tensors x y`, or `Dataset.labelled { inputs; labels }` |
| a `Dataset` with `__len__` / `__getitem__` | `Dataset.of_fn ~length ~collate get` |
| `Subset(ds, idx)` | `Dataset.subset ds idx` |
| `DataLoader(ds, batch_size, shuffle=True, drop_last=True, num_workers=1)` | `Dataset.loader ~batch_size ~shuffle:true ~drop_last:true ds` (prefetches by default) |
| `transforms.RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `Normalize` | `Transform.random_crop ~padding:4 (32, 32)`, `random_hflip ()`, `normalize ~mean ~std` |
| `torchvision.datasets.CIFAR10` | `Cifar10.source ~dir Cifar10.Train` |
| `np.loadtxt` / `pandas.read_csv` of numbers | `Csv.load path` |
| `sklearn.metrics.classification_report` | `Metrics.classification_report`, `Metrics.pp_report` |
| top-5 accuracy | `Metrics.top_k_accuracy ~k:5 ~scores ~labels` |
| `torch.save(model.state_dict(), f)` | `Checkpoint.save f model` (safetensors; add `~optim` for the optimizer) |
| `model.load_state_dict(torch.load(f))` | `Checkpoint.load f model` |
| `safetensors.torch.save_file` / `load_file` | `Safetensors.save` / `Safetensors.load` |

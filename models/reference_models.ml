(** The reference models of Phase 12, written with FlockML's public API only.
    Each has a PyTorch twin in [bench/] with the same architecture and
    initialisation, so accuracy and speed compare like for like. *)

module Make (B : Flockml.S) = struct
  open B

  (** {1 ResNet-20 for CIFAR-10}

      He et al. (2016), section 4.2: a 3x3 convolution to 16 channels, then
      three stages of three basic blocks with 16, 32 and 64 channels (the second
      and third start with stride 2), global average pooling and a linear
      classifier: 20 weighted layers, 0.27 M parameters. Shortcuts that change
      shape are the paper's option A: subsample by 2 and pad the new channels
      with zeros, so they add no parameters. Convolutions have no bias (batch
      norm follows each) and He-normal weights. *)

  let conv3 ?(stride = 1) ~rng in_c out_c =
    Layer.conv2d ~rng ~bias:false ~stride ~padding:1 ~in_channels:in_c
      ~out_channels:out_c ~kernel:3 ()

  (** [x + f x], with [x] subsampled and zero-padded when the block changes
      shape, then a ReLU. *)
  let basic_block ~rng ~stride in_c out_c =
    let conv1 = conv3 ~rng ~stride in_c out_c
    and bn1 = Layer.batch_norm out_c
    and conv2 = conv3 ~rng out_c out_c
    and bn2 = Layer.batch_norm out_c in
    let shortcut x =
      if stride = 1 && in_c = out_c then x
      else
        let x =
          Var.slice x
            [
              Tensor.All;
              Step (0, max_int, stride);
              Step (0, max_int, stride);
              All;
            ]
        in
        let extra = out_c - in_c in
        Var.pad x [| (0, 0); (0, 0); (0, 0); (extra / 2, extra - (extra / 2)) |]
    in
    Layer.custom ~name:"BasicBlock"
      ~children:
        [ ("conv1", conv1); ("bn1", bn1); ("conv2", conv2); ("bn2", bn2) ]
      (fun ~train ?rng x ->
        let f l x = Layer.forward ~train ?rng l x in
        let y = f bn2 (f conv2 (Var.relu (f bn1 (f conv1 x)))) in
        Var.relu (Var.add y (shortcut x)))

  let resnet ~rng ~blocks =
    let stage in_c out_c stride =
      List.init blocks (fun i ->
          if i = 0 then basic_block ~rng ~stride in_c out_c
          else basic_block ~rng ~stride:1 out_c out_c)
    in
    Layer.sequential
      ([ conv3 ~rng 3 16; Layer.batch_norm 16; Layer.relu ]
      @ stage 16 16 1 @ stage 16 32 2 @ stage 32 64 2
      @ [ Layer.global_avg_pool; Layer.linear ~rng ~in_dim:64 ~out_dim:10 () ])

  (** ResNet-20: three blocks per stage. *)
  let resnet20 ~rng = resnet ~rng ~blocks:3

  (** {1 A character-level GPT}

      A decoder-only transformer in the style of GPT-2 / nanoGPT: token and
      position embeddings, [layers] pre-norm blocks (causal multi-head
      attention, then a GELU MLP four times as wide, each with a residual
      connection), a final layer norm and a linear head to the vocabulary.
      Embeddings and the MLP and head weights are N(0, 0.02); the attention
      layers keep PyTorch's [MultiheadAttention] initialisation. *)

  type gpt_config = {
    vocab : int;
    context : int;  (** the longest sequence *)
    dim : int;
    heads : int;
    layers : int;
  }

  let gpt ~rng c =
    let init = Init.Normal 0.02 in
    let block () =
      Layer.sequential
        [
          Layer.residual
            (Layer.sequential
               [
                 Layer.layer_norm [| c.dim |];
                 Layer.multi_head_attention ~rng ~causal:true ~dim:c.dim
                   ~heads:c.heads ();
               ]);
          Layer.residual
            (Layer.sequential
               [
                 Layer.layer_norm [| c.dim |];
                 Layer.linear ~rng ~init ~in_dim:c.dim ~out_dim:(4 * c.dim) ();
                 Layer.gelu ();
                 Layer.linear ~rng ~init ~in_dim:(4 * c.dim) ~out_dim:c.dim ();
               ]);
        ]
    in
    let tok = Layer.embedding ~rng ~init ~num:c.vocab ~dim:c.dim ()
    and pos = Layer.embedding ~rng ~init ~num:c.context ~dim:c.dim ()
    and blocks = Layer.sequential (List.init c.layers (fun _ -> block ()))
    and norm = Layer.layer_norm [| c.dim |]
    and head =
      Layer.linear ~rng ~init ~bias:false ~in_dim:c.dim ~out_dim:c.vocab ()
    in
    Layer.custom ~name:"GPT"
      ~children:
        [
          ("tok", tok);
          ("pos", pos);
          ("blocks", blocks);
          ("norm", norm);
          ("head", head);
        ]
      (fun ~train ?rng x ->
        (* x: [batch; time] token ids, as floats *)
        let s = Var.shape x in
        let b = s.(0) and t = s.(1) in
        if t > c.context then
          invalid_arg
            (Printf.sprintf "GPT: %d tokens is longer than the context, %d" t
               c.context);
        let h =
          Var.add (Layer.forward tok x)
            (Layer.embed pos ~shape:[| 1; t |] (Array.init t Fun.id))
        in
        ignore b;
        let f l x = Layer.forward ~train ?rng l x in
        f head (f norm (f blocks h)))

  (** The parameters AdamW should decay: the weight matrices, not the biases,
      norm scales or embeddings, as nanoGPT does. *)
  let decay_split model =
    List.partition
      (fun (name, p) ->
        Array.length (Var.shape p) >= 2
        && not
             (String.starts_with ~prefix:"tok." name
             || String.starts_with ~prefix:"pos." name))
      (Layer.named_params model)

  (** {1 A regression MLP}

      [features -> hidden -> hidden -> 1], with ReLUs. *)

  let regression_mlp ~rng ~features ~hidden =
    Layer.sequential
      [
        Layer.linear ~rng ~in_dim:features ~out_dim:hidden ();
        Layer.relu;
        Layer.linear ~rng ~in_dim:hidden ~out_dim:hidden ();
        Layer.relu;
        Layer.linear ~rng ~in_dim:hidden ~out_dim:1 ();
      ]

  (** {1 Data} *)

  (** The standard CIFAR-10 augmentation: pad by 4 and crop back to 32 x 32 at a
      random position, mirror half the images, then normalise with the training
      set's channel statistics. *)
  let cifar_augment =
    Transform.compose
      [
        Transform.random_crop ~padding:4 (32, 32);
        Transform.random_hflip ();
        Transform.normalize ~mean:Cifar10.mean ~std:Cifar10.std;
      ]

  (** Normalisation only, for evaluation. *)
  let cifar_normalize = Transform.normalize ~mean:Cifar10.mean ~std:Cifar10.std

  type text = {
    chars : char array;  (** the vocabulary, sorted *)
    ids : int array;
  }
  (** A text as character ids. *)

  (** The distinct characters of [s], sorted, and [s] in their ids. *)
  let text_of_string s =
    let seen = Array.make 256 false in
    String.iter (fun ch -> seen.(Char.code ch) <- true) s;
    let chars =
      Array.of_list
        (List.filter_map
           (fun i -> if seen.(i) then Some (Char.chr i) else None)
           (List.init 256 Fun.id))
    in
    let id = Array.make 256 0 in
    Array.iteri (fun i ch -> id.(Char.code ch) <- i) chars;
    {
      chars;
      ids = Array.init (String.length s) (fun i -> id.(Char.code s.[i]));
    }

  let decode text ids =
    String.init (Array.length ids) (fun i -> text.chars.(ids.(i)))

  (** Every window of [context] characters, with the next character at each
      position as its target: a batch is the token ids [[| b; context |]] (as
      floats, for {!Layer.forward}) and the [b * context] targets. *)
  let windows ~context ids =
    let length = max 0 (Array.length ids - context) in
    Dataset.of_fn ~length
      ~collate:(fun starts ->
        let b = Array.length starts in
        let x =
          Tensor.of_array [| b; context |]
            (Array.init (b * context) (fun k ->
                 float_of_int ids.(starts.(k / context) + (k mod context))))
        in
        ( x,
          Array.init (b * context) (fun k ->
              ids.(starts.(k / context) + (k mod context) + 1)) ))
      Fun.id
end

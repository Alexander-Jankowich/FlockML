(** A character-level GPT on Tiny Shakespeare: the second reference model of the
    roadmap's Phase 12. It uses embeddings, layer norm, causal multi-head
    attention, GELU, AdamW with warmup and a cosine schedule, gradient clipping,
    checkpoints and sampling. Its PyTorch twin is bench/torch_gpt.py.

    {v
    ./scripts/download_shakespeare.sh     # once
    dune exec --profile release bin/gpt.exe -- --checkpoint checkpoints/gpt_shakespeare.safetensors
    dune exec --profile release bin/gpt.exe -- --checkpoint checkpoints/gpt_shakespeare.safetensors --steps 0 --sample 500
    v}

    The text is split 90/10 into training and validation characters. Every step
    trains on a batch of random 128-character windows, predicting each next
    character. *)

open Flockml

let data = ref "data/tinyshakespeare/input.txt"
let steps = ref 3000
let batch_size = ref 32
let context = ref 128
let dim = ref 128
let heads = ref 4
let layers = ref 4
let lr = ref 1e-3
let min_lr = ref 1e-4
let warmup = ref 100
let eval_every = ref 250
let eval_batches = ref 20
let checkpoint = ref ""
let resume = ref false
let sample = ref 0
let temperature = ref 0.8
let top_k = ref 0
let seed = ref 42

let backends : (string * (module Flockml.S)) list =
  [
    ("ocaml", (module Flockml.Default));
    ("blas", (module Flockml.Make (Tensor_blas.F64)));
    ("blas32", (module Flockml.Make (Tensor_blas.F32)));
  ]

let backend = ref "blas32"

let spec =
  [
    ( "--data",
      Arg.Set_string data,
      " The text (default data/tinyshakespeare/input.txt)" );
    ( "--steps",
      Arg.Set_int steps,
      " Training steps (default 3000; 0 = only sample)" );
    ("--batch-size", Arg.Set_int batch_size, " Windows per batch (default 32)");
    ("--context", Arg.Set_int context, " Characters per window (default 128)");
    ("--dim", Arg.Set_int dim, " Embedding size (default 128)");
    ("--heads", Arg.Set_int heads, " Attention heads (default 4)");
    ("--layers", Arg.Set_int layers, " Transformer blocks (default 4)");
    ("--lr", Arg.Set_float lr, " Peak learning rate (default 1e-3)");
    ("--min-lr", Arg.Set_float min_lr, " Final learning rate (default 1e-4)");
    ("--warmup", Arg.Set_int warmup, " Warmup steps (default 100)");
    ( "--eval-every",
      Arg.Set_int eval_every,
      " Steps between evaluations (default 250)" );
    ( "--eval-batches",
      Arg.Set_int eval_batches,
      " Batches per evaluation (default 20)" );
    ( "--checkpoint",
      Arg.Set_string checkpoint,
      " Save the model and optimizer here at every evaluation, and load from \
       it to sample or --resume" );
    ("--resume", Arg.Set resume, " Continue training from --checkpoint");
    ( "--sample",
      Arg.Set_int sample,
      " Characters to generate at the end (default 0)" );
    ( "--temperature",
      Arg.Set_float temperature,
      " Sampling temperature (default 0.8)" );
    ( "--top-k",
      Arg.Set_int top_k,
      " Sample from the k likeliest characters (0 = all)" );
    ("--seed", Arg.Set_int seed, " Random seed (default 42)");
    ( "--backend",
      Arg.Symbol (List.map fst backends, fun s -> backend := s),
      " Tensor backend: blas32 (the default), blas or ocaml" );
    ( "--threads",
      Arg.Int Parallel.set_num_threads,
      " Threads for the tensor kernels (default: FLOCKML_THREADS, or up to 8)"
    );
  ]

module Run (B : Flockml.S) = struct
  open B
  module R = Reference_models.Make (B)

  (** The mean loss over [n] batches of windows spread evenly over [ids]: the
      same windows every time, so evaluations are comparable. *)
  let estimate_loss model ids =
    let src = R.windows ~context:!context ids in
    let count = !eval_batches * !batch_size in
    let stride = max 1 (src.length / count) in
    let starts = Array.init (min count src.length) (fun i -> i * stride) in
    let total = Metrics.Running.create () in
    Var.no_grad (fun () ->
        Dataset.loader ~batch_size:!batch_size (Dataset.subset src starts)
        |> Seq.iter (fun (x, targets) ->
            let logits = Layer.forward model (Var.const x) in
            Metrics.Running.add
              ~weight:(float_of_int (Array.length targets))
              total
              (Tensor.item (Var.value (Loss.cross_entropy logits targets)))));
    Metrics.Running.mean total

  (** Generate [n] characters after [prompt], one at a time: each step runs the
      model on the last [context] characters and samples the next one from its
      softmax at [temperature], among the [top_k] likeliest. *)
  let generate model text ~rng ~prompt n =
    (* the prompt in the model's vocabulary, most recent character first *)
    let index ch =
      match Array.find_index (( = ) ch) text.R.chars with
      | Some i -> i
      | None -> 0
    in
    let ids =
      ref
        (List.rev_map index
           (List.init (String.length prompt) (String.get prompt)))
    in
    let out = Buffer.create n in
    Var.no_grad (fun () ->
        for _ = 1 to n do
          let window =
            Array.of_list
              (List.rev (List.filteri (fun i _ -> i < !context) !ids))
          in
          let t = Array.length window in
          let x = Tensor.of_array [| 1; t |] (Array.map float_of_int window) in
          let logits = Var.value (Layer.forward model (Var.const x)) in
          let v = Array.length text.R.chars in
          let last =
            Tensor.to_array
              (Tensor.slice logits [ Index 0; Index (t - 1); All ])
          in
          let scaled = Array.map (fun l -> l /. !temperature) last in
          (* keep the top_k largest *)
          if !top_k > 0 && !top_k < v then begin
            let sorted = Array.copy scaled in
            Array.sort (fun a b -> Float.compare b a) sorted;
            let cut = sorted.(!top_k - 1) in
            Array.iteri
              (fun i l -> if l < cut then scaled.(i) <- Float.neg_infinity)
              scaled
          end;
          let m = Array.fold_left Float.max Float.neg_infinity scaled in
          let p = Array.map (fun l -> Float.exp (l -. m)) scaled in
          let total = Array.fold_left ( +. ) 0. p in
          let u = Random.State.float rng total in
          let rec pick i acc =
            if i = v - 1 || acc +. p.(i) > u then i
            else pick (i + 1) (acc +. p.(i))
          in
          let next = pick 0 0. in
          Buffer.add_char out text.R.chars.(next);
          ids := next :: !ids
        done);
    Buffer.contents out

  let main () =
    let rng = Random.State.make [| !seed |] in
    let text =
      R.text_of_string (In_channel.with_open_bin !data In_channel.input_all)
    in
    let n = Array.length text.ids in
    let cut = n * 9 / 10 in
    let train_ids = Array.sub text.ids 0 cut
    and val_ids = Array.sub text.ids cut (n - cut) in
    let config =
      {
        R.vocab = Array.length text.chars;
        context = !context;
        dim = !dim;
        heads = !heads;
        layers = !layers;
      }
    in
    let model = R.gpt ~rng config in
    Printf.printf
      "%d characters (vocabulary %d): %d to train on, %d to validate\n" n
      config.vocab cut (n - cut);
    Printf.printf
      "GPT: %d layers, %d heads, dim %d, context %d: %d parameters, backend %s\n\
       %!"
      !layers !heads !dim !context (Layer.num_params model) !backend;
    let decay, no_decay = R.decay_split model in
    let opt =
      Optim.adamw ~lr:!lr ~beta2:0.99 ~weight_decay:0.1
        ~groups:[ Optim.group ~weight_decay:0. (List.map snd no_decay) ]
        (List.map snd decay)
    in
    let warmup_steps = !warmup and lr = !lr and min_lr = !min_lr in
    let schedule =
      Schedule.(
        warmup ~steps:warmup_steps
          (cosine ~min_lr ~lr ~total:(max 1 (!steps - warmup_steps)) ()))
    in
    let first =
      if !checkpoint <> "" && (!resume || !steps = 0) then begin
        ignore (Checkpoint.load ~optim:opt !checkpoint model);
        let s =
          int_of_string
            (List.assoc "step" (Safetensors.load_metadata !checkpoint))
        in
        Printf.printf "loaded %s (step %d)\n%!" !checkpoint s;
        s
      end
      else 0
    in
    let train = R.windows ~context:!context train_ids in
    let batches = ref Seq.empty and epoch = ref 0 in
    let next_batch () =
      match Seq.uncons !batches with
      | Some (b, rest) ->
          batches := rest;
          b
      | None -> (
          incr epoch;
          let rng = Random.State.make [| !seed; !epoch; first |] in
          batches :=
            Dataset.loader ~shuffle:true ~rng ~batch_size:!batch_size train;
          match Seq.uncons !batches with
          | Some (b, rest) ->
              batches := rest;
              b
          | None -> failwith "gpt: the text is shorter than one window")
    in
    let now = Unix.gettimeofday in
    let t0 = ref (now ()) and losses = Metrics.Running.create () in
    for step = first to !steps - 1 do
      Optim.set_lr opt (schedule step);
      Optim.zero_grad opt;
      let x, targets = next_batch () in
      let loss =
        Loss.cross_entropy
          (Layer.forward ~train:true model (Var.const x))
          targets
      in
      Var.backward loss;
      ignore (Var.clip_grad_norm ~max_norm:1.0 (Optim.params opt) : float);
      Optim.step opt;
      Metrics.Running.add losses (Tensor.item (Var.value loss));
      if (step + 1) mod !eval_every = 0 || step = !steps - 1 then begin
        let train_time = now () -. !t0 in
        let val_loss = estimate_loss model val_ids in
        Printf.printf
          "step %5d: train loss %.4f, val loss %.4f, lr %.2e, %.0f ms/step\n%!"
          (step + 1)
          (Metrics.Running.mean losses)
          val_loss (Optim.lr opt)
          (1000. *. train_time /. float_of_int (Metrics.Running.count losses));
        Metrics.Running.reset losses;
        if !checkpoint <> "" then
          Checkpoint.save ~optim:opt
            ~metadata:[ ("step", string_of_int (step + 1)) ]
            !checkpoint model;
        t0 := now ()
      end
    done;
    if !sample > 0 then
      print_endline
        ("\n--- sample ---\n" ^ generate model text ~rng ~prompt:"\n" !sample)
end

(* Create the checkpoint's directory, so a save after the first epoch can't
   fail on it. *)
let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

let () =
  Arg.parse (Arg.align spec) (fun _ -> ()) "gpt [options]";
  if !checkpoint <> "" then mkdir_p (Filename.dirname !checkpoint);
  let module B = (val List.assoc !backend backends) in
  let module R = Run (B) in
  R.main ()

(** Shared test utilities, for any backend. They rely only on the simplest
    tensor functions (of_array, to_array, allclose, pp...), so a failing test
    points at the code under test rather than at the harness. *)

open OUnit2

module Make (B : Flockml.S) = struct
  open B

  (** A fixed-seed generator, so every test run sees the same random data. *)
  let rng = Random.State.make [| 1234 |]

  (** Shorthand for {!Tensor.of_array}. *)
  let t shape data = Tensor.of_array shape data

  (** A tensor rendered as a string, for failure messages. *)
  let pp_tensor x = Format.asprintf "%a" Tensor.pp x

  (** Fail unless two tensors have the same shape and are elementwise close. *)
  let assert_tensor ?(rtol = 1e-5) ?(atol = 1e-6) ?msg expected actual =
    assert_equal ?msg
      ~cmp:(Tensor.allclose ~rtol ~atol)
      ~printer:pp_tensor expected actual

  (** Fail unless two floats differ by at most [eps]. *)
  let assert_float ?(eps = 1e-6) ?msg expected actual =
    assert_equal ?msg
      ~cmp:(fun a b -> Float.abs (a -. b) <= eps)
      ~printer:string_of_float expected actual

  (** Fail unless the tensor has exactly the expected shape. *)
  let assert_shape ?msg expected tensor =
    let show s =
      "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"
    in
    assert_equal ?msg ~printer:show expected (Tensor.shape tensor)

  (** Fail unless calling [f ()] raises [Invalid_argument]. *)
  let assert_invalid_arg ?(msg = "expected Invalid_argument") f =
    match f () with
    | _ -> assert_failure msg
    | exception Invalid_argument _ -> ()

  (** A tensor of uniform samples from [[lo, hi)], drawn from {!rng}. *)
  let rand_tensor ?(lo = -1.0) ?(hi = 1.0) shape =
    let n = Array.fold_left ( * ) 1 shape in
    t shape (Array.init n (fun _ -> lo +. Random.State.float rng (hi -. lo)))

  (** A tensor holding [1, 2, 3, ...] in row-major order: easy to reason about
      in index-shuffling tests. *)
  let range shape =
    let n = Array.fold_left ( * ) 1 shape in
    t shape (Array.init n (fun i -> float_of_int (i + 1)))

  (** Adapt a one-argument function to the list interface of {!gradcheck}. *)
  let unary f = function [ a ] -> f a | _ -> invalid_arg "unary"

  (** Adapt a two-argument function to the list interface of {!gradcheck}. *)
  let binary f = function [ a; b ] -> f a b | _ -> invalid_arg "binary"

  (** Adapt a three-argument function to the list interface of {!gradcheck}. *)
  let ternary f = function [ a; b; c ] -> f a b c | _ -> invalid_arg "ternary"

  (** Numerical gradient check: the single most useful test for autograd.

      [f] maps input nodes to an output node of any shape. The output is reduced
      to a scalar with fixed random weights [w] ([loss = sum (w * out)]) so
      every output element matters. For every input element the autograd
      gradient is compared with the central difference
      [(loss (x + eps) - loss (x - eps)) / (2 eps)]. This relies on [Var.mul]
      and [Var.sum], which are gradchecked too. *)
  let gradcheck ?(eps = 1e-6) ?(tol = 1e-4) name inputs f =
    name >:: fun _ ->
    let params = List.map Var.param inputs in
    let out = f params in
    let w = rand_tensor (Var.shape out) in
    Var.backward (Var.sum (Var.mul out (Var.const w)));
    let wa = Tensor.to_array w in
    let objective xs =
      let v = Tensor.to_array (Var.value (f (List.map Var.const xs))) in
      let acc = ref 0.0 in
      Array.iteri (fun i x -> acc := !acc +. (x *. wa.(i))) v;
      !acc
    in
    List.iteri
      (fun k p ->
        let x0 = List.nth inputs k in
        let shape = Tensor.shape x0 and base = Tensor.to_array x0 in
        let g =
          match Var.grad p with
          | Some g -> g
          | None -> assert_failure (Printf.sprintf "input %d got no gradient" k)
        in
        assert_shape ~msg:(Printf.sprintf "shape of grad of input %d" k) shape g;
        let analytic = Tensor.to_array g in
        Array.iteri
          (fun i _ ->
            let with_delta delta =
              List.mapi
                (fun j x ->
                  if j <> k then x
                  else begin
                    let a = Array.copy base in
                    a.(i) <- a.(i) +. delta;
                    t shape a
                  end)
                inputs
            in
            let numeric =
              (objective (with_delta eps) -. objective (with_delta (-.eps)))
              /. (2.0 *. eps)
            in
            if
              Float.abs (numeric -. analytic.(i))
              > tol *. (1.0 +. Float.abs numeric)
            then
              assert_failure
                (Printf.sprintf
                   "d loss / d input%d[%d]: autograd says %g, finite \
                    difference says %g"
                   k i analytic.(i) numeric))
          base)
      params
end

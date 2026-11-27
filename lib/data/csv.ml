(** Reading CSV files of numbers.

    Fields are separated by one character (a comma by default, or detected from
    the first line), surrounding spaces and double quotes are stripped, and
    blank lines are skipped. A header row of column names is detected when the
    first line holds anything that isn't a number. *)

module type S = sig
  type tensor

  val load : ?sep:char -> ?header:bool -> string -> string array * tensor
  (** [load path] is the column names and the data, [[| rows; columns |]].
      Without a header the columns are named ["0"], ["1"]... [sep] defaults to
      whichever of [','], [';'] and tab occurs most in the first line. Raises
      [Failure] naming the line for a field that isn't a number or a row of the
      wrong length. *)

  val of_string : ?sep:char -> ?header:bool -> string -> string array * tensor
  (** {!load} from text in memory. *)

  val columns : string array * tensor -> string list -> tensor
  (** [columns (names, data) wanted]: the named columns, in the order given.
      Raises [Invalid_argument] for a name that isn't there. *)
end

let strip s =
  let s = String.trim s in
  let n = String.length s in
  if n >= 2 && s.[0] = '"' && s.[n - 1] = '"' then String.sub s 1 (n - 2) else s

let count c s = String.fold_left (fun k x -> if x = c then k + 1 else k) 0 s

(** The module for one backend. *)
module Make (T : Tensor_intf.S) : S with type tensor = T.t = struct
  type tensor = T.t

  let parse ~path ?sep ?header text =
    let lines =
      String.split_on_char '\n' text
      |> List.mapi (fun i l -> (i + 1, String.trim l))
      |> List.filter (fun (_, l) -> l <> "")
    in
    match lines with
    | [] -> ([||], T.zeros [| 0; 0 |])
    | (_, first) :: _ ->
        let sep =
          match sep with
          | Some c -> c
          | None ->
              List.fold_left
                (fun best c ->
                  if count c first > count best first then c else best)
                ',' [ ';'; '\t' ]
        in
        let fields l =
          Array.of_list (List.map strip (String.split_on_char sep l))
        in
        let numeric f = Float.of_string_opt f <> None in
        let header =
          match header with
          | Some h -> h
          | None -> not (Array.for_all numeric (fields first))
        in
        let names, rows =
          match lines with
          | (_, h) :: rest when header -> (fields h, rest)
          | _ -> (Array.init (Array.length (fields first)) string_of_int, lines)
        in
        let cols = Array.length names in
        let data = Array.make (List.length rows * cols) 0. in
        List.iteri
          (fun r (lineno, l) ->
            let fs = fields l in
            if Array.length fs <> cols then
              failwith
                (Printf.sprintf "Csv: %s, line %d: %d fields, expected %d" path
                   lineno (Array.length fs) cols);
            Array.iteri
              (fun c f ->
                match Float.of_string_opt f with
                | Some x -> data.((r * cols) + c) <- x
                | None ->
                    failwith
                      (Printf.sprintf "Csv: %s, line %d: %S is not a number"
                         path lineno f))
              fs)
          rows;
        (names, T.of_array [| List.length rows; cols |] data)

  let of_string ?sep ?header text = parse ~path:"<string>" ?sep ?header text

  let load ?sep ?header path =
    parse ~path ?sep ?header
      (In_channel.with_open_bin path In_channel.input_all)

  let columns (names, data) wanted =
    let idx =
      List.map
        (fun w ->
          match Array.find_index (( = ) w) names with
          | Some i -> i
          | None -> invalid_arg (Printf.sprintf "Csv.columns: no column %S" w))
        wanted
    in
    T.index_select data ~axis:1 (Array.of_list idx)
end

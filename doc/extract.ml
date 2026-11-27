(* Write the ```ocaml blocks of a Markdown file to <prefix>_1.ml,
   <prefix>_2.ml..., so doc/dune can compile and run every block of the
   tutorial as part of `dune test`. *)

let () =
  match Sys.argv with
  | [| _; file; prefix; expected |] ->
      let lines = In_channel.with_open_text file In_channel.input_lines in
      let blocks = ref [] and current = ref None in
      List.iter
        (fun l ->
          match !current with
          | None ->
              if String.trim l = "```ocaml" then
                current := Some (Buffer.create 256)
          | Some b ->
              if String.trim l = "```" then begin
                blocks := Buffer.contents b :: !blocks;
                current := None
              end
              else begin
                Buffer.add_string b l;
                Buffer.add_char b '\n'
              end)
        lines;
      let blocks = List.rev !blocks in
      if List.length blocks <> int_of_string expected then begin
        Printf.eprintf "%s has %d OCaml blocks; doc/dune expects %s\n" file
          (List.length blocks) expected;
        exit 1
      end;
      List.iteri
        (fun i code ->
          Out_channel.with_open_text
            (Printf.sprintf "%s_%d.ml" prefix (i + 1))
            (fun oc -> Printf.fprintf oc "# 1 %S\n%s" file code))
        blocks
  | _ ->
      prerr_endline "usage: extract FILE.md PREFIX COUNT";
      exit 2

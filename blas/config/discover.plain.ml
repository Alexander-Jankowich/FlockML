(* The discover program for builds without dune-configurator: Accelerate when
   the framework is present (macOS), otherwise the portable C fallback. Install
   dune-configurator to also find OpenBLAS and other CBLAS libraries. *)

let write file flags =
  Out_channel.with_open_text file (fun oc ->
      output_string oc "(";
      List.iter (fun f -> Printf.fprintf oc "%S " f) flags;
      output_string oc ")\n")

let () =
  let forced = Sys.getenv_opt "FLOCKML_BLAS" in
  let accelerate =
    forced <> Some "none"
    && Sys.file_exists "/System/Library/Frameworks/Accelerate.framework"
  in
  if accelerate then begin
    write "c_flags.sexp"
      [
        "-DFLOCKML_BLAS_NAME=\"Accelerate\"";
        "-DFLOCKML_ACCELERATE";
        "-DACCELERATE_NEW_LAPACK";
      ];
    write "c_library_flags.sexp" [ "-framework"; "Accelerate" ]
  end
  else begin
    write "c_flags.sexp"
      [ "-DFLOCKML_BLAS_NAME=\"none (portable C fallback)\"" ];
    write "c_library_flags.sexp" []
  end

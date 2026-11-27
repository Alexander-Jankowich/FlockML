(* Find a CBLAS for flockml.blas, in this order:
   - Accelerate, on macOS;
   - OpenBLAS through pkg-config;
   - a library named openblas, cblas or blas that provides cblas.h;
   - otherwise none: the stubs then use a portable C matrix multiply.

   Set FLOCKML_BLAS to accelerate, openblas, cblas, blas or none to force a
   choice (none is useful to test the fallback). *)

module C = Configurator.V1

let cblas_test =
  {|#include <cblas.h>
int main(void) {
  double a = 1, b = 2, c = 0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0, &a, 1,
              &b, 1, 0.0, &c, 1);
  return c == 2.0 ? 0 : 1;
}|}

let accelerate_test =
  {|#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
int main(void) {
  double a = 1, b = 2, c = 0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0, &a, 1,
              &b, 1, 0.0, &c, 1);
  return c == 2.0 ? 0 : 1;
}|}

type found = { name : string; cflags : string list; libs : string list }

let name_flag name = Printf.sprintf "-DFLOCKML_BLAS_NAME=\"%s\"" name

let accelerate c =
  let cflags = [ "-DFLOCKML_ACCELERATE"; "-DACCELERATE_NEW_LAPACK" ]
  and libs = [ "-framework"; "Accelerate" ] in
  if
    C.ocaml_config_var c "system" = Some "macosx"
    && C.c_test c accelerate_test ~c_flags:cflags ~link_flags:libs
  then Some { name = "Accelerate"; cflags; libs }
  else None

let with_cblas c ~name ~cflags ~libs =
  let cflags = "-DFLOCKML_CBLAS" :: cflags in
  if C.c_test c cblas_test ~c_flags:cflags ~link_flags:libs then
    Some { name; cflags; libs }
  else None

let pkg_config_openblas c =
  match C.Pkg_config.get c with
  | None -> None
  | Some pc -> (
      match C.Pkg_config.query pc ~package:"openblas" with
      | None -> None
      | Some conf ->
          with_cblas c ~name:"OpenBLAS" ~cflags:conf.cflags ~libs:conf.libs)

let library c lib = with_cblas c ~name:lib ~cflags:[] ~libs:[ "-l" ^ lib ]
let fallback = { name = "none (portable C fallback)"; cflags = []; libs = [] }

let rec first c = function
  | [] -> fallback
  | f :: rest -> ( match f c with Some found -> found | None -> first c rest)

let () =
  C.main ~name:"flockml_blas" (fun c ->
      let candidates =
        match Sys.getenv_opt "FLOCKML_BLAS" with
        | Some "none" -> []
        | Some "accelerate" -> [ accelerate ]
        | Some "openblas" ->
            [ pkg_config_openblas; (fun c -> library c "openblas") ]
        | Some ("cblas" | "blas") as lib ->
            [ (fun c -> library c (Option.get lib)) ]
        | _ ->
            [
              accelerate;
              pkg_config_openblas;
              (fun c -> library c "openblas");
              (fun c -> library c "cblas");
              (fun c -> library c "blas");
            ]
      in
      let found = first c candidates in
      C.Flags.write_sexp "c_flags.sexp" (name_flag found.name :: found.cflags);
      C.Flags.write_sexp "c_library_flags.sexp" found.libs)

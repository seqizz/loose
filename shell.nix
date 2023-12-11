{ pkgs ? import <nixpkgs> {} }:
let python =
  let packageOverrides = self: super: {
    jc = super.jc.overridePythonAttrs (old: rec {
      pname = "jc";
      src = pkgs.fetchFromGitHub {
        owner = "kellyjonbrazil";
        repo = pname;
        rev = "dev";
        sha256 = "sha256-+DWhbFUQ80pPLquAMNL8EH8b4y0oe5qlnd0HEuhGPwE=";
      };
    });
  };
  in pkgs.python3.override {
    inherit packageOverrides;
    self = python;
  };
in
pkgs.mkShell {
    propagatedBuildInputs = with python.pkgs; [
      jc
      pyyaml
      pykwalify
      xdg-base-dirs
    ];
}

{
  description = "A simple tool to apply XrandR configurations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
        pyedid = python.pkgs.callPackage ./nix/pyedid.nix {};
      in {
        packages.default = python.pkgs.buildPythonApplication {
          pname = "loose";
          version = "0.2.7";
          pyproject = true;

          src = ./.;

          nativeBuildInputs = with python.pkgs; [
            hatchling
            pkgs.installShellFiles
          ];

          buildInputs = with python.pkgs; [
            shtab # For command line completion
          ];

          propagatedBuildInputs = with python.pkgs; [
            filelock
            jc
            pkgs.xorg.xrandr
            pyedid
            pyyaml
            typing-extensions
            xdg-base-dirs
            yamale
          ];

          # Sadly shtab doesn't have fish completion yet
          postInstall = ''
            export HOME=$TMPDIR
            $out/bin/loose -s bash > $HOME/loose.bash
            $out/bin/loose -s zsh > $HOME/loose.zsh
              installShellCompletion --cmd loose \
                --bash $HOME/loose.bash \
                --zsh $HOME/loose.zsh
          '';

          meta = with pkgs.lib; {
            description = "Another xrandr wrapper for multi-monitor setups";
            homepage = "https://git.gurkan.in/gurkan/loose";
            license = licenses.gpl3;
            maintainers = ["seqizz"];
            platforms = platforms.linux;
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.xorg.xrandr
          ];
        };
      }
    );
}

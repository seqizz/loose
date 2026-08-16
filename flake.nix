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
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        pyedid = python.pkgs.callPackage ./nix/pyedid.nix {};
      in rec {
        packages.default = python.pkgs.buildPythonApplication {
          pname = "loose";
          version = pyproject.project.version;
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
            pkgs.xrandr
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

        # Variant with wlr-randr on PATH, for wlroots (sway/river/labwc) sessions.
        # It is kept out of the default package so X11-only machines don't pull
        # the Wayland closure in; the wlroots backend checks PATH at runtime.
        packages.loose-wayland = pkgs.symlinkJoin {
          name = "loose-wayland-${packages.default.version}";
          paths = [ packages.default ];
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/loose \
              --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.wlr-randr ]}
          '';
        };

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.xrandr
            pkgs.wlr-randr
          ];
        };
      }
    );
}

{
  description = "A Nix-flake-based python development environment with venv and pip for use with default python projects";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self , nixpkgs ,... }: let
    system = "x86_64-linux";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
      };
      venvDir = "./.venv";
    in pkgs.mkShell {
      inherit venvDir;
      packages = with pkgs; [
        python313
        python313Packages.pip
        python313Packages.venvShellHook
        python313Packages.pandas
        python313Packages.numpy
        pyright
      ];

      postShellHook = ''
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/"
        echo "$(python --version)"
      '';
      postVenvCreation = ''
        unset SOURCE_DATE_EPOCH
        pip install -r requirements.txt
      '';
    };
  };
}



{
  description = "Python shell flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
    lib = nixpkgs.lib;
    forAllSystems = function: lib.genAttrs [
      "aarch64-linux"
      "x86_64-linux"
    ] function;
  in {
    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
      };

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        jupyterlab
        seaborn
        nltk
        wordcloud
        mlflow
        boto3
      ]);
    in {
      default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          pythonEnv
          awscli
        ];
      };
    });
  };
}

{
  description = "Python shell flake";

  inputs = {
    # nixpkgs.url = "github:nixos/nixpkgs/a4e9f9a895197d62b3546021ee5d58a4596bda9c";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
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
        config.allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
          "postman"
        ];
      };

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        jupyterlab
        seaborn
        nltk
        wordcloud
        mlflow
        boto3
        optuna
        lightgbm
        xgboost
        dvc
        python-dotenv
        dvc-s3
        flask-cors
        imbalanced-learn
      ]);
    in {
      default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          pythonEnv
          awscli
          postman
        ];
      };
    });
  };
}

final: prev: {
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (python-final: python-prev: {
      lightgbm = python-final.lightgbm.override {
        gpuSupport = false; # Fixes https://github.com/NixOS/nixpkgs/issues/377755
      };
    })
  ];
}

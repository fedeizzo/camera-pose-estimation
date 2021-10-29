{ pkgs ? import <nixpkgs> {} }:

let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix/";
    ref = "refs/heads/master";
    rev = "dc94135e31d5c90c40a00a6cbdf9330526e8153b";
  }) {
    python = "python39";
    pypiDataRev = "523d0a516d4ba8d999f507838e9b7a5db476437c";
    pypiDataSha256 = "0w1l53wccspwix97v42yzp3nhf73l0vydvwyz5x12vfsaikwvpy6";
  };
  pythonEnv = mach-nix.mkPython rec {
    requirements = builtins.readFile ./requirements.txt;
    providers = {
      _defualt = "wheel";
    };
    packagesExtra = [
      ./config_parser
    ];
  };
in
mach-nix.nixpkgs.mkShell {
  buildInputs = [
    pkgs.ffmpeg
    pkgs.colmapWithCuda
    pkgs.python39Packages.pytorch-bin
    pkgs.meshlab
    pythonEnv
  ];
  shellHook = ''
    export PATH=$PATH:$(pwd)/camera-pose-estimation/tools
  '';
}

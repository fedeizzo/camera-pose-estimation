{ pkgs ? import <nixpkgs> {} }:
let
  shell = builtins.getEnv "SHELL";
in
(pkgs.buildFHSUserEnv {
  name = "camera-pose-estimation";
  targetPkgs = pkgs: (with pkgs; [
    pkgs.ffmpeg
    pkgs.colmapWithCuda
    pkgs.meshlab
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.setuptools
    pkgs.black
    pkgs.nodePackages.pyright
    pkgs.python39Packages.mypy
    pkgs.python39Packages.debugpy
  ]);
  runScript = shell;
}).env

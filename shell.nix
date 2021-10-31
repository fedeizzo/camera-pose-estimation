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
  ]);
  runScript = shell;
}).env

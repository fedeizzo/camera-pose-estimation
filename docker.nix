let
  requirements = import ./requirements.nix;
  mach-nix = requirements.mach-nix;
  deps = requirements.deps;
in
mach-nix.nixpkgs.dockerTools.buildLayeredImage {
  name = "camera-pose-estimation";
  contents = [
    deps.python
    ./camera-pose-estimation
    ./notebooks
  ] ++ deps.otherDeps;
  config = {
    Cmd = [ "python" ];
    WorkingDir = "/camera-pose-estimation";
  };
}

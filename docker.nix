let
  requirements = import ./requirements.nix;
  mach-nix = requirements.mach-nix;
  deps = requirements.deps;
  pkgs = requirements.pkgs;
in
mach-nix.nixpkgs.dockerTools.buildLayeredImage {
  name = "camera-pose-estimation";
  tag = "latest";
  contents = [
    deps.python
    ./camera-pose-estimation
    ./notebooks
  ] ++ deps.runtime;
  config = {
    Cmd = [ "python" ];
    WorkingDir = "/camera-pose-estimation";
  };
}

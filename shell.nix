let
  requirements = import ./requirements.nix;
  mach-nix = requirements.mach-nix;
  deps = requirements.deps;
in
mach-nix.nixpkgs.mkShell {
  buildInputs = [
    deps.python
  ] ++ deps.docs ++ deps.devel ++ deps.runtime;
  shellHook = ''
    export PATH=$PATH:$(pwd)/camera-pose-estimation/tools
    export LD_PRELOAD="/run/opengl-driver/lib/libcuda.so"
  '';
}

let
  inherit (import <nixpkgs> { }) fetchFromGitHub;
  nixpkgs = fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "nixos-21.05";
    sha256 = "sha256-13B6tgKXygEBWxwj9+vIjuWyzwNF1XPLjJiFAvE7A88=";
  };
  pkgs = import nixpkgs { };
  mach-nix = import
    (builtins.fetchGit {
      url = "https://github.com/DavHau/mach-nix/";
      rev = "31b21203a1350bff7c541e9dfdd4e07f76d874be";
    })
    {
      python = "python39";
      pypiDataRev = "43f5b07a0b1614ee80723b2ad2f8e29a7b246353";
      pypiDataSha256 = "sha256:0psv5w679bgc90ga13m3pz518sw4f35by61gv7c64yl409p70rf9";
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
    # python code 
    pythonEnv
    pkgs.python39Packages.pytorch-bin

    # dataset generation and exploration
    pkgs.ffmpeg
    pkgs.colmapWithCuda
    pkgs.meshlab
    pkgs.cloudcompare

    # documentation generation
    pkgs.texlive.combined.scheme-full
  ];
  shellHook = ''
    export PATH=$PATH:$(pwd)/camera-pose-estimation/tools
    export LD_PRELOAD="/run/opengl-driver/lib/libcuda.so"
  '';
}

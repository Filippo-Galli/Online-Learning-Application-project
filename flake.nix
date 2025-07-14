{
  description = "Python dev environment with venv and requirements.txt support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        
        # Define system libraries needed for common Python packages
        systemLibs = with pkgs; [
          stdenv.cc.cc.lib
          gcc-unwrapped.lib
          zeromq
          openssl
          libffi
          zlib
          pkg-config
          cmake
          gcc
        ];
        
        # Create library path
        libPath = pkgs.lib.makeLibraryPath systemLibs;
        
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            python.pkgs.virtualenv
            python.pkgs.pip
            python.pkgs.setuptools
            python.pkgs.wheel
          ] ++ systemLibs;

          # Environment variables for building packages with C extensions
          shellHook = ''
            echo "🐍 Python environment (Nix-based) loaded!"
            echo "Using Python: ${python.version}"
            
            # Set library and include paths for pip installations
            export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
            export LIBRARY_PATH="${libPath}:$LIBRARY_PATH"
            export C_INCLUDE_PATH="${pkgs.lib.makeSearchPathOutput "dev" "include" systemLibs}:$C_INCLUDE_PATH"
            export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" systemLibs}:$PKG_CONFIG_PATH"
            
            # Ensure pip uses system libraries when building wheels
            export LDFLAGS="-L${pkgs.lib.makeLibraryPath systemLibs}"
            export CPPFLAGS="-I${pkgs.lib.makeSearchPathOutput "dev" "include" systemLibs}"
            
            # Python build flags
            export PYTHONPATH="$PYTHONPATH"
            export PIP_NO_BUILD_ISOLATION=false
          '';
        };
      }
    );
}
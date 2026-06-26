{
  description = "creative-tui — shader canvas + TUI shell + Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
        targets = [ "x86_64-unknown-linux-gnu" ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rustToolchain
          pkg-config
          udev
          alsa-lib
          vulkan-loader
          libxkbcommon
          wayland
        ];

        shellHook = ''
          echo "=== creative-tui ==="
          echo "cargo build && cargo run"
          echo "TUI commands: speed N | color R G B | time"
          echo "Ctrl+C to quit"
        '';
      };
    };
}

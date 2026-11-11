#!/bin/sh
# Bootstrap the Linux/glibc Docker builders without executing a mutable script.
# Digest source: https://static.rust-lang.org/rustup/archive/1.29.3/<target>/rustup-init.sha256
# Update the version and all target digests together after provenance review.
set -eu

version=1.29.3
if command -v dpkg >/dev/null 2>&1 && architecture=$(dpkg --print-architecture); then
    : # Debian's userland architecture can differ from the host kernel's.
else
    architecture=$(uname -m)
fi

case "$architecture" in
    amd64|x86_64) target=x86_64-unknown-linux-gnu ;;
    arm64|aarch64) target=aarch64-unknown-linux-gnu ;;
    i386|i486|i586|i686) target=i686-unknown-linux-gnu ;;
    armel|armv5tel|armv5tejl|arm|xscale) target=arm-unknown-linux-gnueabi ;;
    armv6l) target=arm-unknown-linux-gnueabihf ;;
    armhf|armv7l|armv8l)
        target=armv7-unknown-linux-gnueabihf
        # Match upstream rustup's fallback for ARM CPUs without NEON/SIMD.
        if ! (grep '^Features' /proc/cpuinfo | grep -E -q 'neon|simd'); then
            target=arm-unknown-linux-gnueabihf
        fi
        ;;
    ppc64el|ppc64le) target=powerpc64le-unknown-linux-gnu ;;
    riscv64) target=riscv64gc-unknown-linux-gnu ;;
    s390x) target=s390x-unknown-linux-gnu ;;
    *) printf 'Unsupported Docker builder architecture: %s\n' "$architecture" >&2; exit 1 ;;
esac

case "$target" in
    x86_64-unknown-linux-gnu) checksum=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55 ;;
    aarch64-unknown-linux-gnu) checksum=9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282 ;;
    i686-unknown-linux-gnu) checksum=ef58377096e940ed8ba0d7f8841e6d125922d9d218e04897a05b7f667cf6f5c5 ;;
    arm-unknown-linux-gnueabi) checksum=4053cbe46140f8c6dd9c657e0b8fead09ec35f77bcd740c8a32f8ce54a80eb57 ;;
    arm-unknown-linux-gnueabihf) checksum=56319bf0c965276fac4c5be99241782d03b647cc79af8934ba9afda0351288cc ;;
    armv7-unknown-linux-gnueabihf) checksum=3d5771c716551ee55368be40e02859705f06d725b31df07c15e3593be2d2a5de ;;
    powerpc64le-unknown-linux-gnu) checksum=d5b321db031525fbee38b736846f07a16df22b354a1964bbb79ff83495dbc0c9 ;;
    riscv64gc-unknown-linux-gnu) checksum=e4a960952692ac3dfb5fa4f74e593517241b83e43a93fbdcf2213c811748f1ae ;;
    s390x-unknown-linux-gnu) checksum=df1762614062280a326ac988019ed1f8ceebeafa5fac95164a7201ff10e31d4b ;;
esac

temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
# rustup dispatches installer/proxy mode by argv[0], so retain this basename.
installer="$temporary/rustup-init"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
    "https://static.rust-lang.org/rustup/archive/$version/$target/rustup-init" \
    --output "$installer"
printf '%s  %s\n' "$checksum" "$installer" | sha256sum --check --strict -
chmod 700 "$installer"
"$installer" "$@"

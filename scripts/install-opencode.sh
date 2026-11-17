#!/bin/sh
# Install the pinned OpenCode release into the glibc Wrap E2E runtime.
# Archive digests are from the GitHub v1.17.8 release asset metadata.
set -eu

version=${1:?OpenCode version is required}
if [ "$version" != 1.17.8 ]; then
    printf 'Unsupported unreviewed OpenCode version: %s\n' "$version" >&2
    exit 1
fi
case "$(uname -m)" in
    x86_64)
        target=linux-x64
        # Preserve the vendor installer's fallback on CPUs without AVX2.
        if ! grep -qwi avx2 /proc/cpuinfo 2>/dev/null; then
            target=linux-x64-baseline
        fi
        ;;
    aarch64|arm64) target=linux-arm64 ;;
    *) printf 'Unsupported OpenCode Docker architecture\n' >&2; exit 1 ;;
esac
case "$target" in
    linux-x64) checksum=13fffc29227093462b7d14777ea6b804cfd94c2612819c101bbf933d862b1e5f ;;
    linux-x64-baseline) checksum=9b34bf34bdc66ea34ddd5858a131febf28b6247693acbfb5fb5c9ad94d90388b ;;
    linux-arm64) checksum=cfb34a6a9a40f9a2adb0d0f3ab387b97307823d075572bfb54a553228adcebbd ;;
esac

temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
archive="$temporary/opencode.tar.gz"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
    "https://github.com/anomalyco/opencode/releases/download/v$version/opencode-$target.tar.gz" \
    --output "$archive"
printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --strict -
tar --extract --gzip --file "$archive" --directory "$temporary" --no-same-owner opencode
if [ ! -f "$temporary/opencode" ] || [ -L "$temporary/opencode" ]; then
    printf 'Expected a regular OpenCode binary in the verified archive\n' >&2
    exit 1
fi
mkdir -p "$HOME/.opencode/bin"
install -m 0755 -- "$temporary/opencode" "$HOME/.opencode/bin/opencode"

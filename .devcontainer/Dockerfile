ARG VARIANT=3.12-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}

# Single-wheel architecture (post-#355): `uv sync` builds `headroom-ai`
# from the local pyproject.toml using maturin (declared in build-system).
# Maturin needs rust + cargo. The rustls-everywhere refactor (PR #371)
# eliminated `openssl-sys` from our build tree, so this image no longer
# needs `pkg-config` or `libssl-dev`.
#
# Rust toolchain is provisioned via the official devcontainer feature
# (`ghcr.io/devcontainers/features/rust:1`) declared in devcontainer.json
# so /usr/local/cargo gets the right ownership/permissions for the
# `vscode` runtime user. A manual rustup install in this Dockerfile would
# leave /usr/local/cargo/registry root-owned and the registry cache
# write would fail with "Permission denied" the first time uv sync
# triggers maturin → cargo as the vscode user.
#
# Drop /etc/apt/sources.list.d/yarn.list before apt-get update — the
# base image's yarnpkg.com source has an expired GPG key that aborts
# the whole RUN with "NO_PUBKEY 62D54FD4003F6525". The maturin build
# doesn't need yarn.
RUN rm -f /etc/apt/sources.list.d/yarn.list

RUN python -m pip install --no-cache-dir uv==0.6.17 'maturin>=1.5,<2.0'

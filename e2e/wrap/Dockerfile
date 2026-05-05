# ─── Stage 1: build the headroom-ai wheel in manylinux ─────────────────────
# Building from source inside `node:22-bookworm` produced a `_core.so` that
# referenced `__isoc23_strtoll` (a glibc 2.38+ symbol). The runtime in the
# same image couldn't resolve it at import time because something in the
# build chain — gcc version, libc6-dev backport, or cc-rs compiling
# transitive C deps against newer headers — emitted C23-era symbols that
# the runtime libc.so.6 doesn't have. Building inside the manylinux_2_28
# container (AlmaLinux 8, glibc 2.28 baseline) guarantees the wheel works
# on any glibc 2.28+ runtime, including bookworm.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

# No OpenSSL system deps required. As of the rustls-everywhere refactor,
# all HTTP+TLS in our Rust crates goes through `rustls`. fastembed's
# `hf-hub-rustls-tls` + `ort-download-binaries-rustls-tls` features
# replace the default native-tls path; `cargo tree -p headroom-py -i
# openssl-sys` returns "not found", confirming this Dockerfile no
# longer needs `openssl-devel`, `pkgconfig`, or any perl modules
# (Time::Piece, IPC::Cmd) for the openssl-src vendored Configure
# script. The historical `yum install openssl-devel pkgconfig
# perl-IPC-Cmd` line is intentionally absent.

# Install rust toolchain into the manylinux container. Match the
# rust-toolchain.toml at repo root (1.95.0 + rustfmt + clippy) so cargo
# doesn't try to mutate the toolchain on first invocation.
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/opt/python/cp311-cp311/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal -c rustfmt -c clippy --default-toolchain 1.95.0

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY headroom/ headroom/

# Build the wheel with maturin. PYO3_USE_ABI3_FORWARD_COMPATIBILITY allows
# building against Python 3.14+ until we bump pyo3 past 0.22.
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
# Build for Python 3.11 — aider-chat==0.86.2 requires Python <3.12, and
# trixie's default python3.13 is too new for aider. The manylinux image
# has python interpreters for every version at /opt/python/cpXY-cpXY/.
# Stage 2 uses `python:3.11-slim` (now trixie-based, glibc 2.41).
RUN /opt/python/cp311-cp311/bin/pip install 'maturin>=1.5,<2.0' && \
    /opt/python/cp311-cp311/bin/maturin build --release --out /dist --interpreter python3.11

# ─── Stage 2: python+node runtime ───────────────────────────────────────────
# Base on `python:3.11-slim` (now trixie, glibc 2.41) instead of
# `node:22-bookworm` (glibc 2.36): the wheel's `_core.so` references
# three glibc 2.38+ C23 wrappers (`__isoc23_strtol{,l,ul}`) that one of
# our transitive C/C++ deps emits during cc-rs compilation even when
# built inside manylinux_2_28 (probably libstdc++'s `<cstdlib>` resolving
# `std::strtoll` to the C23 variant when the toolchain has newer
# headers). Bookworm's libc.so.6 doesn't export these wrappers, so
# `import headroom._core` fails at runtime. Trixie's glibc 2.41 does.
#
# We need Python 3.11 here (aider-chat==0.86.2 requires Python <3.12).
# Trixie's default `python3` is 3.13 — too new for aider. python:3.11-slim
# gives us Python 3.11 + trixie glibc + apt access for installing Node.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    AIDER_CHAT_VERSION=0.86.2 \
    CODEX_VERSION=0.118.0 \
    OPENCLAW_VERSION=2026.4.7 \
    PATH="/opt/headroom-venv/bin:/opt/aider-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Node.js 22 from NodeSource (matches what node:22-trixie gives,
# but layered onto python:3.11-slim so we get py3.11 + node22 + glibc 2.41
# in one image). Also install git for any package that clones at install.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Bring in the prebuilt manylinux wheel from stage 1.
COPY --from=builder /dist/*.whl /tmp/wheels/

# `sdk/typescript` and `plugins/openclaw` are wired into the wrap-e2e
# harness (e2e/wrap/run.py invokes `npx openclaw`). DO NOT copy
# `headroom/` or `pyproject.toml` from the workspace — they would shadow
# the installed wheel's `headroom/` package via cwd and Python would
# import the source-only `headroom/` (which has no `_core.so`),
# triggering the same `ModuleNotFoundError` the wheel install fixes.
COPY sdk/typescript ./sdk/typescript
COPY plugins/openclaw ./plugins/openclaw

# Install the prebuilt manylinux wheel with [proxy] extras. Pip resolves
# extras from the wheel's metadata and pulls deps from public PyPI. The
# wheel's `_core.so` was compiled against glibc 2.28, so it loads cleanly
# against bookworm's glibc 2.36 at runtime.
RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade pip && \
    /opt/headroom-venv/bin/python -m pip install "$(ls /tmp/wheels/headroom_ai-*.whl)[proxy]" && \
    /opt/headroom-venv/bin/python -c "from headroom._core import DiffCompressor; print('headroom._core OK')" && \
    python -m venv /opt/aider-venv && \
    /opt/aider-venv/bin/python -m pip install --upgrade pip && \
    /opt/aider-venv/bin/python -m pip install "aider-chat==${AIDER_CHAT_VERSION}" && \
    npm install -g --no-fund --no-audit "@openai/codex@${CODEX_VERSION}" "openclaw@${OPENCLAW_VERSION}"

COPY e2e/wrap ./e2e/wrap

CMD ["python", "e2e/wrap/run.py"]

# ─── Stage 1: build the headroom-ai wheel in manylinux ─────────────────────
# See e2e/wrap/Dockerfile for the full rationale. tl;dr: building from
# source inside `node:22-bookworm` produced a `_core.so` referencing
# `__isoc23_strtoll` (glibc 2.38+) that the same image's runtime libc
# couldn't resolve. Building inside manylinux_2_28 (AlmaLinux 8, glibc
# 2.28 baseline) yields a wheel portable to any glibc 2.28+ runtime.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

# No OpenSSL system deps required (rustls-everywhere refactor). See
# e2e/wrap/Dockerfile for the full rationale.

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

ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
# Build for Python 3.11 to match Stage 2's `python:3.11-slim` runtime.
RUN /opt/python/cp311-cp311/bin/pip install 'maturin>=1.5,<2.0' && \
    /opt/python/cp311-cp311/bin/maturin build --release --out /dist --interpreter python3.11

# ─── Stage 2: python runtime ───────────────────────────────────────────────
# `python:3.11-slim` (now trixie, glibc 2.41) — see e2e/wrap/Dockerfile
# for why we need glibc ≥ 2.38 (the wheel's `_core.so` references C23
# symbols that bookworm's 2.36 doesn't export). The init e2e harness
# only needs python (no node) — `headroom init -g <target>` is Python.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/headroom-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Single-wheel refactor side effect: `headroom` is now installed
    # from a wheel into site-packages, so `__file__.parents[2]` no
    # longer points at the repo root (it points at site-packages).
    # `_marketplace_source()` falls back to `chopratejas/headroom`
    # (the GitHub remote) instead of the local `/workspace` path the
    # `seq_claude_local` e2e assertion expects. Pin the source via
    # the existing override env var so the local `/workspace`
    # marketplace.json (COPY'd below) is used.
    HEADROOM_MARKETPLACE_SOURCE=/workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /dist/*.whl /tmp/wheels/

# Init e2e harness imports from e2e._lib + .claude-plugin assets.
# DO NOT copy `headroom/` or `pyproject.toml` from the workspace —
# the installed wheel must be authoritative. The harness packages
# (e2e/_lib/, .claude-plugin/, plugins/) don't shadow `headroom`.
COPY .claude-plugin ./.claude-plugin
COPY .github/plugin ./.github/plugin
COPY plugins/headroom-agent-hooks ./plugins/headroom-agent-hooks
COPY e2e/__init__.py ./e2e/__init__.py
COPY e2e/_lib ./e2e/_lib
COPY e2e/init ./e2e/init

RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade pip && \
    /opt/headroom-venv/bin/python -m pip install "$(ls /tmp/wheels/headroom_ai-*.whl)[proxy]" && \
    /opt/headroom-venv/bin/python -c "from headroom._core import DiffCompressor; print('headroom._core OK')"

CMD ["python", "e2e/init/run.py"]

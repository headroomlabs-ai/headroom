ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.6.17
# Pinned 2026-04-15. Update via Dependabot or: docker pull python:3.11-slim
ARG PYTHON_DIGEST=sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf
# Pinned 2026-04-15. Update via Dependabot or: docker pull gcr.io/distroless/python3-debian13
ARG DISTROLESS_DIGEST=sha256:ed3a4beb46f8f8baac068743ba1b1f95ea3f793422129cf6dd23967f779b6018
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages

# ---- Build stage: compile native extensions, build wheel ----
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST} AS builder

ARG UV_VERSION

# build-essential / g++ for any C extension wheels uv may need to build
# from source. curl + ca-certificates are required by the rustup
# bootstrap below. patchelf for maturin's wheel-link repair on linux.
# No OpenSSL system deps required: the rustls-everywhere refactor
# eliminated `openssl-sys` from our build tree by switching fastembed
# to `hf-hub-rustls-tls` + `ort-download-binaries-rustls-tls`.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    ca-certificates \
    patchelf \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==${UV_VERSION}

# Rust toolchain for the headroom._core extension. With single-wheel
# architecture (post-#355), `pip install -e .` invokes maturin via
# pyproject.toml's [build-system], which calls cargo. No more separate
# headroom-core-py package.
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal -c rustfmt -c clippy --default-toolchain 1.95.0

WORKDIR /build

# Copy the full set of files maturin needs to build the wheel: the root
# pyproject.toml + Cargo workspace + Rust crates + Python source. The
# uv install builds + installs the wheel in one shot.
COPY pyproject.toml uv.lock README.md ./
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY headroom/ headroom/

ARG HEADROOM_EXTRAS=proxy,code
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/build/target \
    uv pip install --system ".[${HEADROOM_EXTRAS}]"

# Build-stage smoke check: verify the extension loads end-to-end inside
# the build image before we copy site-packages into the runtime image.
# If this fails, the runtime image would fail Phase A0's fail-loud
# startup check on every restart. Run from /tmp so cwd doesn't shadow
# site-packages with /build/headroom/ (which has no _core.so since
# maturin installed the .so into site-packages).
RUN cd /tmp && python -c "from headroom._core import DiffCompressor, SmartCrusher; \
    print(f'build-stage rust core verify OK: {DiffCompressor.__name__}, {SmartCrusher.__name__}')"

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST} AS runtime-slim-base

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom

RUN mkdir -p /home/nonroot /data && \
    if [ "$RUNTIME_USER" = "nonroot" ]; then \
      groupadd --gid 1000 nonroot && \
      useradd --uid 1000 --gid nonroot --create-home nonroot && \
      mkdir -p /home/nonroot/.headroom && \
      chown -R nonroot:nonroot /data /home/nonroot; \
    else \
      mkdir -p /root/.headroom; \
    fi

USER ${RUNTIME_USER}
WORKDIR /home/nonroot

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8787/readyz"]

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

FROM ${DISTROLESS_IMAGE}@${DISTROLESS_DIGEST} AS runtime-slim

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}

USER ${RUNTIME_USER}
WORKDIR /app

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHON_SITE_PACKAGES}

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/readyz', timeout=5)"]

ENTRYPOINT ["python3", "-m", "headroom.cli", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

# Default published image remains python-slim runtime
FROM runtime-slim-base AS runtime

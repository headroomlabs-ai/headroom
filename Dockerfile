ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.18
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages

# ---- Build stage: compile native extensions, build wheel ----
FROM python:${PYTHON_VERSION}-slim AS builder

ARG UV_VERSION
ARG PYTHON_SITE_PACKAGES
ARG HEADROOM_BUILD_VERSION=""

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
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    uv pip install --system ".[${HEADROOM_EXTRAS}]"

RUN --mount=type=bind,source=.,target=/context,readonly \
    HEADROOM_BUILD_VERSION="${HEADROOM_BUILD_VERSION}" PYTHON_SITE_PACKAGES="${PYTHON_SITE_PACKAGES}" python - <<'PY'
import hashlib
import os
from pathlib import Path


def git_revision(context: Path) -> str | None:
    git_dir = context / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_name = head.removeprefix("ref: ").strip()
        ref_path = git_dir / ref_name
        if ref_path.exists():
            head = ref_path.read_text(encoding="utf-8").strip()
        else:
            packed_refs = git_dir / "packed-refs"
            if not packed_refs.exists():
                return None
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref_name:
                    head = sha
                    break
            else:
                return None
    return head[:12] if len(head) >= 7 and all(c in "0123456789abcdef" for c in head.lower()) else None


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    inputs = (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "crates",
        "headroom",
    )
    for name in inputs:
        path = root / name
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            digest.update(file.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:12]


build_version = os.environ["HEADROOM_BUILD_VERSION"].strip()
if not build_version:
    print("no Headroom build version override provided; using installed package metadata")
    raise SystemExit(0)
if build_version == "source-build":
    revision = git_revision(Path("/context"))
    build_version = (
        f"source-build+g{revision}"
        if revision
        else f"source-build+sha256.{source_digest(Path('/build'))}"
    )

package_dir = Path(os.environ["PYTHON_SITE_PACKAGES"]) / "headroom"
(package_dir / "_build_info.py").write_text(
    "BUILD_VERSION = " + repr(build_version) + "\n",
    encoding="utf-8",
)
print("baked Headroom build version: " + build_version)
PY

# Build-stage smoke check: verify the extension loads end-to-end inside
# the build image before we copy site-packages into the runtime image.
# If this fails, the runtime image would fail Phase A0's fail-loud
# startup check on every restart. Run from /tmp so cwd doesn't shadow
# site-packages with /build/headroom/ (which has no _core.so since
# maturin installed the .so into site-packages).
RUN cd /tmp && python -c "from headroom._core import DiffCompressor, SmartCrusher; \
    print(f'build-stage rust core verify OK: {DiffCompressor.__name__}, {SmartCrusher.__name__}')"

# Build the native Rust reverse proxy binary and stage it for the runtime
# images (issue #976). These images already run "the proxy"; bundling the
# native `headroom-proxy` binary lets operators front the Python proxy with
# the Rust SigV4 / live-zone compression path from the same image. The
# binary is copied out of the cache-mounted target dir into a persistent
# path so the COPY in the runtime stages can pick it up.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked --bin headroom-proxy && \
    cp target/release/headroom-proxy /usr/local/bin/headroom-proxy

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim AS runtime-slim-base

ARG RUNTIME_USER=nonroot
ARG RUNTIME_HOME=/home/nonroot
ARG PYTHON_SITE_PACKAGES

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom
# Native Rust reverse proxy binary (issue #976).
COPY --from=builder /usr/local/bin/headroom-proxy /usr/local/bin/headroom-proxy

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
WORKDIR ${RUNTIME_HOME}

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Declare ~/.headroom as a volume so Docker (and ACA) can attach persistent
# storage here.  Bare `docker run` gets an anonymous volume as a fallback so
# state is never silently written to the ephemeral container layer.
# RUNTIME_HOME defaults to /home/nonroot (the published image default); pass
# --build-arg RUNTIME_HOME=/root when building with RUNTIME_USER=root.
VOLUME ${RUNTIME_HOME}/.headroom

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8787/readyz"]

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

FROM ${DISTROLESS_IMAGE} AS runtime-slim

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

COPY --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
# Native Rust reverse proxy binary (issue #976).
COPY --from=builder /usr/local/bin/headroom-proxy /usr/local/bin/headroom-proxy

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

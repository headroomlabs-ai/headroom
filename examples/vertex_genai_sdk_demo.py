#!/usr/bin/env python3
"""End-to-end demo: google-genai SDK -> Headroom proxy -> Vertex AI (Gemini Enterprise).

Proves that standard Vertex native configurations route properly through the proxy.

What this script does
---------------------
1. Spawns the Headroom proxy as a subprocess (backend=vertex).
2. Waits for /readyz.
3. Configures standard google-genai SDK with vertexai=True hitting the proxy.
4. Sends an inference probe to validate native proxy connectivity.
5. Sends an inference probe with thinking configs to validate extensions.
6. Tears the proxy back down.

Requirements
------------
- GCP credentials with Vertex AI access (run `gcloud auth application-default login`)
- ``pip install "headroom-ai[proxy]" google-genai``

Run
---
    export GCP_PROJECT_ID=your-project
    python examples/vertex_genai_sdk_demo.py

Note the proxy needs the ``[proxy]`` extra (fastapi, uvicorn, httpx[http2]); with
only bare ``httpx`` installed it fails at startup on the missing ``h2`` package.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

# Defer imports until runtime to guarantee proxy has a chance to start
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai is not installed.")
    print("Run `pip install google-genai` and try again.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_PORT = 8787
# The google-genai SDK always builds `publishers/google/...` paths, so this demo
# can only exercise Gemini. Claude on Vertex lives under `publishers/anthropic`
# and uses :rawPredict -- see tests/test_proxy_vertex_native_integration.py.
#
# `global` is the default because the evergreen "-latest" aliases are global-only;
# Gemini 3.x currently has no US regional endpoint.
DEFAULT_REGION = "global"
DEFAULT_MODEL = "gemini-flash-latest"
# Other options: gemini-3.5-flash (also servable in europe-west2, asia-northeast1)


# ----------------------------------------------------------------------------
# Proxy lifecycle
# ----------------------------------------------------------------------------


def start_proxy(port: int, region: str) -> subprocess.Popen[bytes]:
    """Spawn `headroom proxy --backend vertex` as a subprocess."""
    env = os.environ.copy()
    env.setdefault("HEADROOM_LOG", "INFO")

    cmd = [
        sys.executable,
        "-m",
        "headroom.cli",
        "proxy",
        "--backend",
        "vertex",
        "--region",
        region,
        "--port",
        str(port),
    ]
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    log_path = Path("/tmp") / f"vertex_genai_sdk_demo_{port}.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"  proxy logs -> {log_path}", file=sys.stderr)
    return proc


def wait_for_proxy_ready(port: int, timeout_s: float = 30.0) -> None:
    """Poll /readyz until the proxy answers or timeout."""
    url = f"http://127.0.0.1:{port}/readyz"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(
        f"Proxy on port {port} did not become ready within {timeout_s}s; last error: {last_err!r}"
    )


def stop_proxy(proc: subprocess.Popen[bytes]) -> None:
    """Politely shut the proxy down."""
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ----------------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------------


def explain_failure(exc: Exception, region: str, model_id: str) -> str:
    """Turn a raw Vertex exception into something the reader can act on."""
    msg = str(exc)
    hints: list[str] = []
    if "404" in msg or "NOT_FOUND" in msg:
        hints.append(
            f"'{model_id}' is not available to this project at location '{region}'. "
            "Evergreen '-latest' aliases and Gemini 3.x are global-only today; "
            "Gemini 3.x has no US regional endpoint (try --region global, or "
            "europe-west2 / asia-northeast1 for gemini-3.5-flash)."
        )
    if "401" in msg or "UNAUTHENTICATED" in msg:
        hints.append(
            "Credentials are missing or stale (tokens expire hourly). Run: "
            "gcloud auth application-default login"
        )
    if "403" in msg or "PERMISSION_DENIED" in msg:
        hints.append(
            "The project cannot serve this model: enable aiplatform.googleapis.com, "
            "grant roles/aiplatform.user, and (for partner models) enable the model "
            "in Model Garden for this project."
        )
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        hints.append("Quota exhausted for this model/location -- retry or pick another region.")
    if "publishers/anthropic" in msg or model_id.startswith("claude"):
        hints.append(
            "The google-genai SDK only builds publishers/google paths, so Claude "
            "cannot be reached through it. Use the rawPredict path directly "
            "(see tests/test_proxy_vertex_native_integration.py)."
        )
    if not hints:
        hints.append(
            "Not a known provisioning condition -- suspect the proxy (path rewrite, "
            "body mangling, dropped auth header) and reproduce with curl straight "
            "against Vertex to confirm."
        )
    return msg + "\n    hint: " + "\n    hint: ".join(hints)


def run_demo(port: int, region: str, model_id: str) -> int:
    print("=" * 76)
    print(" Headroom E2E: google-genai SDK -> Headroom proxy -> Vertex")
    print("=" * 76)
    print(f" port={port} region={region} model={model_id}")
    print()

    print("[1/3] Spawning Headroom proxy ...")
    proxy = start_proxy(port=port, region=region)
    try:
        try:
            wait_for_proxy_ready(port=port, timeout_s=45.0)
        except Exception as e:
            print(f"  ! Proxy failed to start: {e}", file=sys.stderr)
            return 2
        print("  proxy ready.")

        print("\n[2/3] Configuring google-genai SDK for Vertex via Proxy")
        project_id = os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            print(
                "  ! GCP_PROJECT_ID is not set. Every probe below will 404.\n"
                "    export GCP_PROJECT_ID=$(gcloud config get-value project)",
                file=sys.stderr,
            )
            return 2

        # We enforce vertexai=True to hit standard Vertex boundaries.
        # Ensure your GCP ADC variables are valid / authorized if routing through real GCP APIs.
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region,
            http_options={"base_url": f"http://127.0.0.1:{port}"},
        )

        print("\n[3/3] Probes")
        print("\n  a. Standard Inference:")
        content = "Count to 5, listing each number separated by commas."

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=content,
            )
            print("  ✓ Standard response received successfully!")
            print(f"  > {response.text.strip()}")
        except Exception as e:
            print(f"  ! Standard inference failed: {explain_failure(e, region, model_id)}")
            return 1

        print("\n  b. Inference with Thinking Config:")
        # Configure thinking config (where supported).
        # For now, we just pass parameters and see if Headroom properly parses/forwards them.
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=128)
            )
            response = client.models.generate_content(
                model=model_id,
                contents="Think briefly. What is heavier: a kg of feathers or a kg of steel?",
                config=config,
            )
            print("  ✓ Thinking response received successfully!")
            print(f"  > {response.text.strip()}")
        except Exception as e:
            print(f"  ! Thinking inference failed: {explain_failure(e, region, model_id)}")
            return 1

        return 0

    finally:
        print("\n  shutting down proxy ...")
        stop_proxy(proxy)


def main() -> int:
    ap = argparse.ArgumentParser(description="google-genai SDK -> Headroom proxy -> Vertex")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    return run_demo(port=args.port, region=args.region, model_id=args.model)


if __name__ == "__main__":
    sys.exit(main())

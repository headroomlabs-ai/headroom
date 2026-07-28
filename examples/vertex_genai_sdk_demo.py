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
- ``pip install google-genai``

Run
---
    python examples/vertex_genai_sdk_demo.py
"""

from __future__ import annotations

import argparse
import sys
import os
import subprocess
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
DEFAULT_REGION = "us-central1"
DEFAULT_MODEL = "claude-sonnet-4-6" 
# Other options: gemini-flash-latest


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
        project_id = os.environ.get("GCP_PROJECT_ID", "dummy-project")
        
        # We enforce vertexai=True to hit standard Vertex boundaries.
        # Ensure your GCP ADC variables are valid / authorized if routing through real GCP APIs.
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region,
            http_options={'api_endpoint': f'127.0.0.1:{port}'}
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
            print(f"  ! Standard inference failed (ensure GCP credentials): {e}")

        print("\n  b. Inference with Thinking Config:")
        # Configure thinking config (where supported). 
        # For now, we just pass parameters and see if Headroom properly parses/forwards them.
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget_tokens=100
                ) 
            )
            response = client.models.generate_content(
                model=model_id,
                contents="Think briefly. What is heavier: a kg of feathers or a kg of steel?",
                config=config,
            )
            print("  ✓ Thinking response received successfully!")
            print(f"  > {response.text.strip()}")
        except Exception as e:
            print(f"  ! Thinking inference error: {e}")
            
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

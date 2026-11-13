#!/usr/bin/env python3
"""Live proof that a real OpenCode selects and authenticates an install-written
custom ``headroom/<id>`` model, for the persistent provider-scope install.

Only the upstream is a fixture. The chain is real throughout:

    real OpenCode  ->  real `headroom proxy`  ->  fixture upstream

No provider account or real API key is used. The credential is an inert
sentinel that the fixture asserts on.

Run with the repository's proxy dependencies installed:

    python plugins/opencode/scripts/install-credential-proof.py

Exits non-zero on the first failed step. Every assertion prints PASS/FAIL so
the transcript can be pasted next to the PR it proves.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from headroom.install.models import ConfigScope, DeploymentManifest  # noqa: E402
from headroom.providers.opencode.install import (  # noqa: E402
    apply_provider_scope,
    revert_provider_scope,
)

PROXY_PORT = 18787
UPSTREAM_PORT = 19101
API_KEY = "sk-proof-not-a-real-key"
CUSTOM_MODEL = "deepseek-chat"
UPSTREAM_URL = f"http://127.0.0.1:{UPSTREAM_PORT}/v1"

requests_seen: list[dict] = []


class MockUpstream(BaseHTTPRequestHandler):
    """Fixture upstream: records the auth it received and answers OpenAI-shaped."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # noqa: A003
        return

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {"object": "list", "data": [{"id": CUSTOM_MODEL, "object": "model"}]},
            )
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        payload = json.loads(raw.decode() or "{}")
        requests_seen.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "model": payload.get("model"),
            }
        )
        self._json(
            200,
            {
                "id": "chatcmpl-proof",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", CUSTOM_MODEL),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "PROOF_OK"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


def manifest() -> DeploymentManifest:
    """The deployment `install apply --scope provider --target opencode` builds."""
    return DeploymentManifest(
        profile="proof",
        preset="persistent-task",
        runtime_kind="python",
        supervisor_kind="none",
        scope=ConfigScope.PROVIDER.value,
        provider_mode="manual",
        targets=["opencode"],
        port=PROXY_PORT,
        host="127.0.0.1",
        backend="openai",
        # planner.py turns --env OPENAI_TARGET_API_URL into --openai-api-url too.
        proxy_args=[
            "--host",
            "127.0.0.1",
            "--port",
            str(PROXY_PORT),
            "--openai-api-url",
            UPSTREAM_URL,
        ],
        base_env={
            "OPENAI_TARGET_API_URL": UPSTREAM_URL,
            "HEADROOM_OPENCODE_EXTRA_MODELS": CUSTOM_MODEL,
        },
        tool_envs={},
    )


def find_opencode() -> str:
    """Return an absolute opencode executable (Windows needs the .cmd shim)."""
    if os.name == "nt":
        for name in ("opencode.cmd", "opencode.exe", "opencode"):
            found = shutil.which(name)
            if found:
                return found
    found = shutil.which("opencode")
    if found is None:
        raise SystemExit("opencode is not on PATH")
    return found


def opencode_env(root: Path) -> dict[str, str]:
    """A minimal, credential-free environment with isolated XDG directories."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root / "oc-home"),
        "TMPDIR": os.environ.get("TEMP", ""),
        "XDG_CONFIG_HOME": str(root / "oc-config"),
        "XDG_DATA_HOME": str(root / "oc-data"),
        "XDG_CACHE_HOME": str(root / "oc-cache"),
        "XDG_STATE_HOME": str(root / "oc-state"),
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    for key in ("SYSTEMROOT", "APPDATA", "LOCALAPPDATA", "COMSPEC", "PATHEXT", "WINDIR", "TEMP"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def run_opencode(args: list[str], root: Path, *, timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(
        [find_opencode(), *args],
        cwd=str(root / "work"),
        env=opencode_env(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def config_path(root: Path) -> Path:
    return root / "oc-config" / "opencode" / "opencode.json"


def headroom_provider(root: Path) -> dict:
    return json.loads(config_path(root).read_text(encoding="utf-8"))["provider"]["headroom"]


def headroom_ids(root: Path) -> list[str]:
    proc = run_opencode(["models", "--standalone"], root)
    return [
        line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("headroom/")
    ]


def report(step: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {step}: {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    opener = subprocess.run(
        [find_opencode(), "--version"], capture_output=True, text=True, timeout=120
    ).stdout.strip()
    root = Path(tempfile.mkdtemp(prefix="headroom-opencode-install-proof-"))
    for sub in ("oc-home", "oc-config/opencode", "oc-data", "oc-cache", "oc-state", "work"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    upstream = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), MockUpstream)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    proxy_env = dict(os.environ)
    proxy_env.update(
        {
            "HEADROOM_BEACON": "off",
            "HEADROOM_TELEMETRY": "off",
            "HEADROOM_UPDATE_CHECK": "off",
            "HEADROOM_REQUIRE_RUST_CORE": "false",
            "HEADROOM_ALLOWED_BASE_URLS": "127.0.0.1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "PYTHONPATH": str(REPO),
        }
    )
    proxy_log = (root / "proxy.log").open("w", encoding="utf-8")
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "headroom.cli",
            "proxy",
            "--host",
            "127.0.0.1",
            "--port",
            str(PROXY_PORT),
            "--openai-api-url",
            UPSTREAM_URL,
        ],
        cwd=str(REPO),
        env=proxy_env,
        stdout=proxy_log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        if proxy.poll() is not None:
            proxy_log.close()
            raise SystemExit((root / "proxy.log").read_text(encoding="utf-8", errors="replace"))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/readyz", timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise SystemExit("proxy never became ready")

    # The user's own config: a custom model plus the credential OpenCode must
    # send to the proxy, exactly as docs/content/docs/opencode-deepseek.mdx shows.
    config_path(root).write_text(
        json.dumps(
            {
                "provider": {
                    "headroom": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Headroom Proxy",
                        "options": {
                            "baseURL": f"http://127.0.0.1:{PROXY_PORT}/v1",
                            "apiKey": API_KEY,
                        },
                        "models": {
                            "gpt-4o": {
                                "name": "GPT-4o",
                                "limit": {"context": 128000, "output": 16384},
                            },
                            CUSTOM_MODEL: {
                                "name": "DeepSeek Chat",
                                "limit": {"context": 65536, "output": 16384},
                            },
                        },
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.environ["OPENCODE_CONFIG"] = str(config_path(root))
    m = manifest()

    print(f"OpenCode: {opener}")
    print(f"Proxy:    http://127.0.0.1:{PROXY_PORT} (real `headroom proxy`)")
    print(f"Upstream: {UPSTREAM_URL} (fixture)")
    print(f"Evidence: {root}\n")

    def live_turn(label: str) -> None:
        requests_seen.clear()
        proc = run_opencode(
            [
                "run",
                "--standalone",
                "--model",
                f"headroom/{CUSTOM_MODEL}",
                "--auto",
                "Reply with PROOF_OK",
            ],
            root,
        )
        if not requests_seen:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
        report(
            f"{label}: OpenCode requested the custom model through the proxy",
            bool(requests_seen) and requests_seen[0]["model"] == CUSTOM_MODEL,
            f"requests={len(requests_seen)} "
            f"model={requests_seen[0]['model'] if requests_seen else None}",
        )
        auth = requests_seen[0]["authorization"]
        report(
            f"{label}: the kept credential reached the upstream",
            bool(auth) and API_KEY in auth,
            f"authorization={'present' if auth else 'MISSING'}",
        )

    mutation = apply_provider_scope(m)
    assert mutation is not None

    provider = headroom_provider(root)
    report(
        "install kept the user credential",
        provider["options"].get("apiKey") == API_KEY,
        f"apiKey={'kept' if provider['options'].get('apiKey') == API_KEY else 'LOST'}",
    )
    report(
        "install wrote the custom model",
        CUSTOM_MODEL in provider["models"],
        f"models={sorted(provider['models'])}",
    )
    report(
        "OpenCode lists the custom id (first install)",
        f"headroom/{CUSTOM_MODEL}" in headroom_ids(root),
        f"headroom ids={headroom_ids(root)}",
    )
    live_turn("first install")

    # The deactivate/activate cycle is where the credential used to disappear.
    revert_provider_scope(mutation, m, restore_backup=False)
    apply_provider_scope(m)

    provider = headroom_provider(root)
    report(
        "re-apply kept the credential",
        provider["options"].get("apiKey") == API_KEY,
        f"apiKey={'kept' if provider['options'].get('apiKey') == API_KEY else 'LOST'}",
    )
    report(
        "OpenCode still lists the custom id (after re-apply)",
        f"headroom/{CUSTOM_MODEL}" in headroom_ids(root),
        f"headroom ids={headroom_ids(root)}",
    )
    live_turn("after re-apply")

    proxy_log.close()
    proxy.terminate()
    try:
        proxy.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proxy.kill()
    upstream.shutdown()
    upstream.server_close()

    print(f"\nPASS: real OpenCode {opener} selected and authenticated headroom/{CUSTOM_MODEL}")


if __name__ == "__main__":
    main()

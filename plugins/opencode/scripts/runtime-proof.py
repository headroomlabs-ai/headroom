#!/usr/bin/env python3
"""Exercise the built plugin with real OpenCode 2.x and Headroom processes.

Only the upstream is a fixture. No provider account or real API key is used.
Run with the repository's proxy dependencies installed in this Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

REPO = Path(__file__).resolve().parents[3]
PLUGIN = REPO / "plugins/opencode/dist-standalone"
TEXT = "HEADROOM_RUNTIME_OK"
MODEL = "claude-sonnet-4-6"
AUTH = "Bearer proof-token-not-a-secret"
KEY = "test-key-not-a-secret"


def require(condition, message: str):
    if not condition:
        raise RuntimeError(message)


def stop_process_group(process):
    # Both children are started in dedicated sessions; never target the caller's group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def events(model: str):
    yield (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_proof",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        },
    )
    yield (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": TEXT},
        },
    )
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
    )
    yield "message_stop", {"type": "message_stop"}


def run_case(opencode: str, output: Path, name: str, base_path: str, enabled: bool):
    case = output / name
    case.mkdir()
    records: list[dict] = []
    expected_path = (
        "/v1/messages" if enabled and base_path in ("", "/v1") else base_path + "/messages"
    )
    expected_proxy = enabled and base_path in ("", "/v1")

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {key.lower(): value for key, value in self.headers.items()}
            records.append(
                {
                    "path": self.path,
                    "headers": headers,
                    "model": body.get("model"),
                    "stream": body.get("stream"),
                }
            )
            # Do not let an unexpected endpoint pass just because this is a fixture.
            if self.path != expected_path:
                self.send_error(404, "Unexpected Messages path")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for event, data in events(body["model"]):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)

    # 127.0.0.2 is local but differs from the plugin's deliberately excluded
    # localhost/127.0.0.1 destinations, exercising remote-model routing safely.
    upstream = ThreadingHTTPServer(("127.0.0.2", 0), Upstream)
    upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        proxy_port = sock.getsockname()[1]
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    upstream_url = f"http://127.0.0.2:{upstream.server_port}"

    # Use isolated HOME/XDG directories and a minimal environment instead of
    # inheriting provider credentials/config. This is not a filesystem sandbox.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(case / "proxy-home"),
        "TMPDIR": str(case / "tmp"),
        "TZ": "UTC",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "PYTHONPATH": str(REPO),
        "HEADROOM_BEACON": "off",
        "HEADROOM_TELEMETRY": "off",
        "HEADROOM_UPDATE_CHECK": "off",
        "HEADROOM_REQUIRE_RUST_CORE": "false",
        "HEADROOM_ALLOWED_BASE_URLS": "127.0.0.2",
        "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    }
    for directory in (
        "proxy-home",
        "tmp",
        "work",
        "oc-home",
        "oc-data",
        "oc-config",
        "oc-cache",
        "oc-state",
    ):
        (case / directory).mkdir()
    cli_env = {
        "PATH": env["PATH"],
        "HOME": str(case / "oc-home"),
        "TMPDIR": env["TMPDIR"],
        "TZ": "UTC",
        "NO_PROXY": "*",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    for variable, directory in (
        ("DATA", "oc-data"),
        ("CONFIG", "oc-config"),
        ("CACHE", "oc-cache"),
        ("STATE", "oc-state"),
    ):
        cli_env[f"XDG_{variable}_HOME"] = str(case / directory)
    config = {
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "model": f"proof-copilot/{MODEL}",
        "providers": {
            "proof-copilot": {
                "name": "Controlled Copilot-shaped stand-in",
                "package": "@opencode/ai/providers/anthropic",
                "settings": {"baseURL": upstream_url + base_path, "apiKey": KEY},
                "headers": {"Authorization": AUTH},
                "models": {
                    MODEL: {
                        "name": "Claude proof",
                        "limit": {"context": 200000, "output": 512},
                        "capabilities": {"tools": False, "input": ["text"], "output": ["text"]},
                    }
                },
            }
        },
        # OpenCode 2.x local package entries point at a directory, not a file.
        "plugins": [
            {
                "package": str(PLUGIN),
                "options": {
                    "proxyUrl": proxy_url,
                    "project": name,
                    "debug": True,
                },
            }
        ]
        if enabled
        else [],
    }
    (case / "work/opencode.json").write_text(json.dumps(config, indent=2) + "\n")
    proxy = None
    try:
        with (case / "proxy.log").open("w") as proxy_log:
            proxy = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "headroom.cli",
                    "proxy",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(proxy_port),
                    "--no-optimize",
                    "--no-cache",
                ],
                cwd=REPO,
                env=env,
                stdout=proxy_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            opener = build_opener(ProxyHandler({}))
            for _ in range(60):
                if proxy.poll() is not None:
                    raise RuntimeError(f"Proxy exited; see {case / 'proxy.log'}")
                try:
                    with opener.open(proxy_url + "/livez", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Proxy did not become live")
            with (case / "opencode.log").open("w") as cli_log:
                cli = subprocess.Popen(
                    [
                        opencode,
                        "run",
                        "--standalone",
                        "--print-logs",
                        "--log-level",
                        "debug",
                        "--format",
                        "json",
                        f"Reply with {TEXT}.",
                    ],
                    cwd=case / "work",
                    env=cli_env,
                    stdout=cli_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = cli.wait(timeout=90)
                finally:
                    stop_process_group(cli)
            if returncode:
                raise RuntimeError(f"OpenCode exited {returncode}; see {case / 'opencode.log'}")
    finally:
        if proxy is not None:
            stop_process_group(proxy)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()
        (case / "upstream.json").write_text(json.dumps(records, indent=2) + "\n")

    cli_log = (case / "opencode.log").read_text()
    proxy_log = (case / "proxy.log").read_text()
    client_events = [
        json.loads(line) for line in cli_log.splitlines() if line.startswith('{"type":')
    ]
    text = "".join(event["part"]["text"] for event in client_events if event.get("type") == "text")
    require(text == TEXT, f"SSE text did not reach the client: {text!r}")
    require(
        any(event.get("type") == "step_finish" for event in client_events),
        "No completed client turn",
    )
    require(records, "No upstream request observed")
    require(
        all(record["path"] == expected_path for record in records),
        "Failed check: all((record['path'] == expected_path for record in records))",
    )
    for record in records:
        headers = record["headers"]
        require(headers.get("authorization") == AUTH, "Bearer auth changed")
        require(headers.get("x-api-key") == KEY, "API key changed")
        require(
            headers.get("anthropic-version") == "2023-06-01",
            "Failed check: headers.get('anthropic-version') == '2023-06-01'",
        )
        require(
            "opencode/" in headers.get("user-agent", ""),
            "Failed check: 'opencode/' in headers.get('user-agent', '')",
        )
        require(
            not any(key.startswith("x-headroom-") for key in headers), "Internal headers leaked"
        )
        require(
            record["model"] == MODEL and record["stream"] is True,
            "Failed check: record['model'] == MODEL and record['stream'] is True",
        )
    inbound = [
        line
        for line in proxy_log.splitlines()
        if "event=proxy_inbound_request" in line and "method=POST" in line
    ]
    require(bool(inbound) == expected_proxy, "Unexpected proxy routing")
    if expected_proxy:
        require(len(inbound) == len(records), "Proxy/upstream request counts differ")
        require(
            all(
                "path=/v1/messages" in line and '"x-headroom-base-url"' in line for line in inbound
            ),
            "Failed check: all(('path=/v1/messages' in line and '\"x-headroom-base-url\"' in line for line in inbound))",
        )
        require(
            f"path={upstream_url}/v1/messages" in proxy_log,
            "Failed check: f'path={upstream_url}/v1/messages' in proxy_log",
        )
        require("stripped_count=3" in proxy_log, "Failed check: 'stripped_count=3' in proxy_log")
    if enabled:
        require(
            'msg="loading plugin"' in cli_log and str(PLUGIN) in cli_log,
            "Built plugin was not loaded",
        )
    core_mode = "disabled" if "event=rust_core_disabled" in proxy_log else "loaded"
    require("event=rust_core_" + core_mode in proxy_log, "Rust core mode was not observed")
    summary = {
        "case": name,
        "rust_core": core_mode,
        "plugin": enabled,
        "base_path": base_path,
        "proxy_requests": len(inbound),
        "upstream_requests": len(records),
        "upstream_path": expected_path,
        "auth_preserved": True,
        "internal_headers_at_upstream": False,
        "client_text": text,
    }
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opencode", default=shutil.which("opencode"), help="OpenCode 2.x executable"
    )
    parser.add_argument(
        "--output", type=Path, help="New directory for logs (default: temporary directory)"
    )
    args = parser.parse_args()
    require(
        os.name == "posix", "This local proof requires POSIX process groups and loopback aliases"
    )
    if not args.opencode or not (PLUGIN / "index.js").is_file():
        parser.error("Install OpenCode 2.x and run npm run build:standalone first")
    opencode = str(Path(args.opencode).resolve())
    output = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="headroom-opencode-"))
    )
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    version_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(output),
        "XDG_DATA_HOME": str(output / "version-data"),
        "XDG_CONFIG_HOME": str(output / "version-config"),
        "XDG_CACHE_HOME": str(output / "version-cache"),
        "XDG_STATE_HOME": str(output / "version-state"),
    }
    version = subprocess.check_output(
        [opencode, "--version"], env=version_env, text=True, timeout=15
    ).strip()
    require(re.search(r"(?:^|\s)v?2\.", version), "This proof requires OpenCode 2.x")
    require(
        (PLUGIN / "index.js").read_bytes()
        == (REPO / "headroom/providers/opencode/_dist/index.js").read_bytes(),
        "Rebuilt plugin differs from the checked-in wheel bundle",
    )
    metadata = {
        "opencode": version,
        "python": sys.version.split()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "headroom_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "plugin_sha256": hashlib.sha256((PLUGIN / "index.js").read_bytes()).hexdigest(),
    }
    print(f"OpenCode: {version}\nEvidence: {output}", flush=True)
    results = [
        run_case(opencode, output, *case)
        for case in (
            ("baseline", "", False),
            ("copilot-shaped", "", True),
            ("v1-gateway", "/v1", True),
            ("prefixed-gateway", "/api/coding/paas/v4", True),
        )
    ]
    (output / "summary.json").write_text(
        json.dumps({**metadata, "results": results}, indent=2) + "\n"
    )
    print("PASS: all four real OpenCode / proxy transport scenarios", flush=True)


if __name__ == "__main__":
    main()

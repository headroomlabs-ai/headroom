from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

REPO_ROOT = Path("/workspace")
PLUGIN_DIR = REPO_ROOT / "plugins" / "openclaw"
SDK_DIR = REPO_ROOT / "sdk" / "typescript"
RTK_MARKER = "<!-- headroom:rtk-instructions -->"
PROXY_PORT = 28887
CODEX_PORT = 28888
AIDER_PORT = 28889
CURSOR_PORT = 28890
OPENCLAW_PROXY_PORT = 28891
# Phase G PR-G1: new wrap subcommands. Smoke-tested via --prepare-only since
# their CLIs may not exist on the e2e image and the wrap commands without
# --prepare-only block on the proxy. The wiring is otherwise covered by the
# unit tests in tests/test_cli/test_wrap_{cline,continue,goose,openhands}.py.
CLINE_PORT = 28892
CONTINUE_PORT = 28893
GOOSE_PORT = 28894
OPENHANDS_PORT = 28895
OPENCODE_PORT = 28896


def log(message: str) -> None:
    print(f"[wrap-e2e] {message}", flush=True)


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class MockOpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, MockOpenAIHandler)
        self.requests: list[dict[str, Any]] = []


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body: dict[str, Any] | None = None) -> None:
        server = self.server
        assert isinstance(server, MockOpenAIServer)
        server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-4o-mini",
                            "object": "model",
                            "owned_by": "openai",
                        }
                    ],
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        payload = json.loads(raw_body.decode("utf-8") or "{}")
        self._record(body=payload)
        if self.path == "/v1/chat/completions":
            self._write_json(
                200,
                {
                    "id": "chatcmpl-e2e",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "mock completion from upstream",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})


def wait_for_http(url: str, *, timeout: int = 30) -> httpx.Response:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code < 500:
                return response
        except Exception as exc:  # pragma: no cover - best effort retry surface
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def wait_for_output(proc: subprocess.Popen[str], text: str, *, timeout: int = 30) -> str:
    deadline = time.time() + timeout
    chunks: list[str] = []
    while time.time() < deadline:
        if proc.stdout is None:
            break
        line = proc.stdout.readline()
        if line:
            chunks.append(line)
            if text in "".join(chunks):
                return "".join(chunks)
        elif proc.poll() is not None:
            break
    output = "".join(chunks)
    raise RuntimeError(f"Timed out waiting for process output '{text}'. Output so far:\n{output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_shims(shim_dir: Path) -> None:
    generic_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        import urllib.request
        from pathlib import Path

        tool = Path(sys.argv[0]).name
        log_dir = Path(os.environ["HEADROOM_E2E_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "tool": tool,
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "env": {
                key: os.environ.get(key)
                for key in (
                    "OPENAI_BASE_URL",
                    "OPENAI_API_BASE",
                    "ANTHROPIC_BASE_URL",
                    "CODEX_HOME",
                    "OPENCODE_CONFIG_CONTENT",
                )
                if os.environ.get(key) is not None
            },
        }

        probes = []

        def fetch(url: str, *, headers: dict[str, str] | None = None) -> None:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=10) as response:
                probes.append({"url": url, "status": response.status})

        openai_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if openai_base:
            fetch(
                f"{openai_base.rstrip('/')}/models",
                headers={"Authorization": "Bearer test-key"},
            )

        anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")
        if anthropic_base:
            fetch(f"{anthropic_base.rstrip('/')}/health")

        record["probes"] = probes

        with (log_dir / f"{tool}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        print(f"{tool} shim executed")
        raise SystemExit(0)
        """
    )
    codex_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        import urllib.request
        from pathlib import Path

        tool = Path(sys.argv[0]).name
        log_dir = Path(os.environ["HEADROOM_E2E_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "tool": tool,
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "env": {
                key: os.environ.get(key)
                for key in (
                    "OPENAI_BASE_URL",
                    "OPENAI_API_BASE",
                    "ANTHROPIC_BASE_URL",
                    "CODEX_HOME",
                    "OPENCODE_CONFIG_CONTENT",
                )
                if os.environ.get(key) is not None
            },
        }

        probes = []

        def request_json(
            url: str,
            *,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, object]]:
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=data, headers=headers or {})
            if payload is not None:
                request.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                body = json.loads(raw.decode("utf-8") or "{}") if raw else {}
                return response.status, body

        openai_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if openai_base:
            auth_headers = {"Authorization": "Bearer test-key"}
            models_status, _models_body = request_json(
                f"{openai_base.rstrip('/')}/models",
                headers=auth_headers,
            )
            probes.append({"url": f"{openai_base.rstrip('/')}/models", "status": models_status})

            model_name = "headroom-wrap-e2e"
            chat_status, chat_body = request_json(
                f"{openai_base.rstrip('/')}/chat/completions",
                payload={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Confirm Headroom received this wrapped Codex message.",
                        }
                    ],
                },
                headers=auth_headers,
            )
            probes.append(
                {"url": f"{openai_base.rstrip('/')}/chat/completions", "status": chat_status}
            )
            record["chat_completion"] = (
                chat_body.get("choices", [{}])[0].get("message", {}).get("content")
            )

            stats_url = openai_base.rstrip("/").removesuffix("/v1") + "/stats"
            stats_status, stats_body = request_json(stats_url, headers=auth_headers)
            probes.append({"url": stats_url, "status": stats_status})
            requests = stats_body.get("requests", {})
            by_model = requests.get("by_model", {}) if isinstance(requests, dict) else {}
            record["headroom_request_total"] = (
                requests.get("total") if isinstance(requests, dict) else None
            )
            record["headroom_model_count"] = (
                by_model.get(model_name) if isinstance(by_model, dict) else None
            )

        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            session_config = Path(codex_home) / "config.toml"
            record["session_config"] = (
                session_config.read_text(encoding="utf-8") if session_config.exists() else None
            )

        record["probes"] = probes

        with (log_dir / f"{tool}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        print(f"{tool} shim executed")
        raise SystemExit(0)
        """
    )
    rtk_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import sys

        if "--version" in sys.argv:
            print("rtk e2e-shim")
        else:
            print("rtk shim")
        raise SystemExit(0)
        """
    )
    write_executable(shim_dir / "claude", generic_shim)
    write_executable(shim_dir / "codex", codex_shim)
    write_executable(shim_dir / "aider", generic_shim)
    write_executable(shim_dir / "opencode", generic_shim)
    write_executable(shim_dir / "rtk", rtk_shim)


def start_mock_server(port: int) -> tuple[MockOpenAIServer, threading.Thread]:
    server = MockOpenAIServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def start_proxy(port: int, env: dict[str, str]) -> subprocess.Popen[str]:
    log(f"Starting headroom proxy on port {port}")
    proc = subprocess.Popen(
        ["headroom", "proxy", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wait_for_http(f"http://127.0.0.1:{port}/health", timeout=30)
    return proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def wait_for_command_success(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    deadline = time.time() + timeout
    last_output = ""
    while time.time() < deadline:
        remaining = deadline - time.time()
        per_call_timeout = max(1.0, min(5.0, remaining))
        try:
            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=per_call_timeout,
            )
        except subprocess.TimeoutExpired:
            last_output = f"Command timed out after {per_call_timeout:.1f}s"
            time.sleep(1)
            continue
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout.rstrip(), flush=True)
            if result.stderr.strip():
                print(result.stderr.rstrip(), file=sys.stderr, flush=True)
            return result
        last_output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for command to succeed: {' '.join(cmd)}\nLast output:\n{last_output}"
    )


def start_openclaw_gateway(env: dict[str, str], cwd: Path) -> subprocess.Popen[str]:
    log("Starting OpenClaw gateway for e2e verification")
    return subprocess.Popen(
        ["openclaw", "gateway"],
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def stop_openclaw_gateway(env: dict[str, str], cwd: Path) -> None:
    log("Stopping OpenClaw gateway after e2e verification")
    run(["openclaw", "gateway", "stop"], env=env, cwd=cwd, timeout=60)


def verify_installs() -> None:
    log("Verifying installed packages and binaries")
    for tool in ("headroom", "codex", "aider", "openclaw"):
        assert_true(shutil.which(tool) is not None, f"Expected '{tool}' on PATH")
    run(["headroom", "--help"], timeout=30)
    run(["npm", "list", "-g", "--depth=0", "@openai/codex", "openclaw"], timeout=60)
    run(["/opt/aider-venv/bin/python", "-m", "pip", "show", "aider-chat"], timeout=60)


def prepare_local_openclaw_plugin(base_env: dict[str, str], tmp_dir: Path) -> Path:
    log("Preparing local TypeScript package for OpenClaw plugin build")
    sdk_dir = tmp_dir / "sdk-typescript"
    plugin_dir = tmp_dir / "openclaw-plugin"
    shutil.copytree(SDK_DIR, sdk_dir)
    shutil.copytree(PLUGIN_DIR, plugin_dir)

    plugin_lock = plugin_dir / "package-lock.json"
    if plugin_lock.exists():
        plugin_lock.unlink()

    run(["npm", "install"], env=base_env, cwd=sdk_dir, timeout=600)
    run(["npm", "run", "build"], env=base_env, cwd=sdk_dir, timeout=600)
    pack_result = run(["npm", "pack"], env=base_env, cwd=sdk_dir, timeout=600)
    tarball_name = pack_result.stdout.strip().splitlines()[-1].strip()
    tarball_path = sdk_dir / tarball_name
    assert_true(tarball_path.exists(), "Expected npm pack to produce a local SDK tarball")

    package_json_path = plugin_dir / "package.json"
    package_json = json.loads(package_json_path.read_text(encoding="utf-8"))
    package_json["dependencies"]["headroom-ai"] = f"file:{tarball_path.as_posix()}"
    package_json_path.write_text(f"{json.dumps(package_json, indent=2)}\n", encoding="utf-8")

    return plugin_dir


def verify_proxy_round_trip(base_env: dict[str, str], mock_server: MockOpenAIServer) -> None:
    proxy_port = PROXY_PORT
    proc = start_proxy(proxy_port, base_env)
    try:
        health = wait_for_http(f"http://127.0.0.1:{proxy_port}/health")
        assert_true(health.status_code == 200, "Proxy health check should return 200")

        models = httpx.get(
            f"http://127.0.0.1:{proxy_port}/v1/models",
            headers={"Authorization": "Bearer test-key"},
            timeout=10.0,
        )
        assert_true(models.status_code == 200, "Proxy should pass through /v1/models")
        assert_true(models.json()["data"][0]["id"] == "gpt-4o-mini", "Unexpected models payload")

        chat = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
            timeout=10.0,
        )
        assert_true(chat.status_code == 200, "Proxy should pass through chat completions")
        assert_true(
            chat.json()["choices"][0]["message"]["content"] == "mock completion from upstream",
            "Unexpected chat completion payload",
        )
        assert_true(
            any(item["path"] == "/v1/models" for item in mock_server.requests),
            "Mock upstream should receive /v1/models",
        )
        assert_true(
            any(item["path"] == "/v1/chat/completions" for item in mock_server.requests),
            "Mock upstream should receive /v1/chat/completions",
        )
    finally:
        stop_process(proc)


def verify_codex_wrap(
    base_env: dict[str, str], project_dir: Path, log_dir: Path, mock_server: MockOpenAIServer
) -> None:
    port = CODEX_PORT
    run(
        ["headroom", "wrap", "codex", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    # RTK guidance for Codex is global-only (#1240): it is injected into
    # ~/.codex/AGENTS.md, never a project-level AGENTS.md. A project AGENTS.md is
    # written only when `wrap codex --memory` is used (for memory guidance), which
    # this scenario does not exercise.
    global_agents = Path(base_env["HOME"]) / ".codex" / "AGENTS.md"
    assert_true(global_agents.exists(), "Codex wrap should create ~/.codex/AGENTS.md")
    assert_true(
        RTK_MARKER in global_agents.read_text(encoding="utf-8"), "Missing global RTK marker"
    )

    config_path = Path(base_env["HOME"]) / ".codex" / "config.toml"
    assert_true(
        not config_path.exists(),
        "Codex wrap should leave ~/.codex/config.toml untouched during normal launch",
    )

    entries = read_jsonl(log_dir / "codex.jsonl")
    assert_true(len(entries) > 0, "Codex shim should have been invoked")
    env_vars = entries[-1]["env"]
    session_home = env_vars.get("CODEX_HOME")
    assert_true(
        isinstance(session_home, str) and session_home,
        "Codex wrap should launch the child with a session-scoped CODEX_HOME",
    )
    assert_true(
        Path(session_home) != config_path.parent,
        "Codex wrap should not point the child at the real ~/.codex home",
    )
    config = entries[-1].get("session_config")
    assert_true(
        isinstance(config, str) and config,
        "Codex wrap should capture the session-scoped config during launch",
    )
    assert_true(
        f'openai_base_url = "http://127.0.0.1:{port}/v1"' in config,
        "Codex wrap should inject openai_base_url into the session config",
    )
    assert_true(
        f'base_url = "http://127.0.0.1:{port}/v1"' in config,
        "Codex wrap should inject the headroom provider base_url into the session config",
    )
    assert_true(
        'env_key = "OPENAI_API_KEY"' not in config,
        "Codex wrap should preserve OAuth and never inject env_key into the session config",
    )
    # Bug 3 (#406): requires_openai_auth must be absent from headroom provider blocks.
    assert_true(
        "requires_openai_auth" not in config,
        "Codex wrap must NOT inject requires_openai_auth into the session headroom provider block",
    )
    assert_true(
        "supports_websockets = true" in config,
        "Codex wrap missing 'supports_websockets = true' in the session config",
    )

    assert_true(
        env_vars.get("OPENAI_BASE_URL") == f"http://127.0.0.1:{port}/v1",
        "Codex wrap should set OPENAI_BASE_URL",
    )
    assert_true(
        entries[-1]["probes"]
        == [
            {"url": f"http://127.0.0.1:{port}/v1/models", "status": 200},
            {"url": f"http://127.0.0.1:{port}/v1/chat/completions", "status": 200},
            {"url": f"http://127.0.0.1:{port}/stats", "status": 200},
        ],
        "Codex shim should prove OPENAI_BASE_URL points at a live proxy and that Headroom logged the wrapped message",
    )
    assert_true(
        entries[-1].get("chat_completion") == "mock completion from upstream",
        "Codex wrap should receive the mock upstream completion through Headroom",
    )
    assert_true(
        entries[-1].get("headroom_model_count", 0) >= 1,
        "Codex wrap should appear in Headroom request stats",
    )
    assert_true(
        any(
            item["path"] == "/v1/chat/completions"
            and isinstance(item.get("body"), dict)
            and item["body"].get("model") == "headroom-wrap-e2e"
            for item in mock_server.requests
        ),
        "Codex wrap should forward the wrapped message upstream through Headroom",
    )


def verify_claude_wrap(base_env: dict[str, str], project_dir: Path, log_dir: Path) -> None:
    port = PROXY_PORT + 10
    run(
        ["headroom", "wrap", "claude", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    entries = read_jsonl(log_dir / "claude.jsonl")
    assert_true(len(entries) > 0, "Claude shim should have been invoked")
    env_vars = entries[-1]["env"]
    assert_true(
        env_vars.get("ANTHROPIC_BASE_URL") == f"http://127.0.0.1:{port}",
        "Claude wrap should set ANTHROPIC_BASE_URL",
    )
    assert_true(
        entries[-1]["probes"] == [{"url": f"http://127.0.0.1:{port}/health", "status": 200}],
        "Claude shim should prove ANTHROPIC_BASE_URL points at a live proxy",
    )


def verify_aider_wrap(base_env: dict[str, str], project_dir: Path, log_dir: Path) -> None:
    port = AIDER_PORT
    run(
        ["headroom", "wrap", "aider", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    conventions = project_dir / "CONVENTIONS.md"
    assert_true(conventions.exists(), "Aider wrap should create CONVENTIONS.md")
    assert_true(
        RTK_MARKER in conventions.read_text(encoding="utf-8"),
        "Aider wrap should inject RTK instructions",
    )

    entries = read_jsonl(log_dir / "aider.jsonl")
    assert_true(len(entries) > 0, "Aider shim should have been invoked")
    env_vars = entries[-1]["env"]
    # Aider cannot send custom headers, so its wrap embeds the launch
    # directory as a /p/<name> base-URL prefix for per-project savings;
    # the proxy strips it before routing, so the probes still succeed.
    project_prefix = f"/p/{quote(project_dir.name, safe='')}"
    assert_true(
        env_vars.get("OPENAI_API_BASE") == f"http://127.0.0.1:{port}{project_prefix}/v1",
        "Aider wrap should set OPENAI_API_BASE",
    )
    assert_true(
        env_vars.get("ANTHROPIC_BASE_URL") == f"http://127.0.0.1:{port}{project_prefix}",
        "Aider wrap should set ANTHROPIC_BASE_URL",
    )
    assert_true(
        entries[-1]["probes"]
        == [
            {"url": f"http://127.0.0.1:{port}{project_prefix}/v1/models", "status": 200},
            {"url": f"http://127.0.0.1:{port}{project_prefix}/health", "status": 200},
        ],
        "Aider shim should prove both configured base URLs point at a live proxy",
    )


def verify_cursor_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    port = CURSOR_PORT
    proc = subprocess.Popen(
        ["headroom", "wrap", "cursor", "--port", str(port)],
        env=base_env,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        output = wait_for_output(proc, "Press Ctrl+C to stop the proxy.", timeout=30)
        # Cursor setup lines embed the /p/<name> per-project prefix.
        cursor_prefix = f"/p/{quote(project_dir.name, safe='')}"
        assert_true(
            f"http://127.0.0.1:{port}{cursor_prefix}/v1" in output,
            "Cursor wrap should print the OpenAI base URL override",
        )
        assert_true(
            f"http://127.0.0.1:{port}" in output,
            "Cursor wrap should print the Anthropic base URL override",
        )
        wait_for_http(f"http://127.0.0.1:{port}/health", timeout=15)
        cursorrules = project_dir / ".cursorrules"
        assert_true(cursorrules.exists(), "Cursor wrap should create .cursorrules")
        assert_true(
            RTK_MARKER in cursorrules.read_text(encoding="utf-8"),
            "Cursor wrap should inject RTK instructions",
        )
    finally:
        stop_process(proc)


def verify_cline_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    """Smoke test: `wrap cline --prepare-only` writes RTK guidance to .clinerules."""
    run(
        ["headroom", "wrap", "cline", "--prepare-only", "--port", str(CLINE_PORT)],
        env=base_env,
        cwd=project_dir,
        timeout=60,
    )
    clinerules = project_dir / ".clinerules"
    assert_true(clinerules.exists(), "Cline wrap should create .clinerules")
    assert_true(
        RTK_MARKER in clinerules.read_text(encoding="utf-8"),
        "Cline wrap should inject RTK instructions",
    )


def verify_continue_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    """Smoke test: `wrap continue --prepare-only` injects RTK into .continue/config.json."""
    run(
        ["headroom", "wrap", "continue", "--prepare-only", "--port", str(CONTINUE_PORT)],
        env=base_env,
        cwd=project_dir,
        timeout=60,
    )
    config_file = project_dir / ".continue" / "config.json"
    assert_true(config_file.exists(), "Continue wrap should create .continue/config.json")
    data = json.loads(config_file.read_text(encoding="utf-8"))
    system_message = data.get("systemMessage", "")
    assert_true(
        RTK_MARKER in system_message,
        "Continue wrap should inject RTK instructions into systemMessage",
    )


def verify_goose_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    """Smoke test: `wrap goose --prepare-only` writes RTK guidance to .goosehints."""
    run(
        ["headroom", "wrap", "goose", "--prepare-only", "--port", str(GOOSE_PORT)],
        env=base_env,
        cwd=project_dir,
        timeout=60,
    )
    goosehints = project_dir / ".goosehints"
    assert_true(goosehints.exists(), "Goose wrap should create .goosehints")
    assert_true(
        RTK_MARKER in goosehints.read_text(encoding="utf-8"),
        "Goose wrap should inject RTK instructions",
    )


def verify_openhands_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    """Smoke test: `wrap openhands --prepare-only` exits clean and ensures rtk is present.

    OpenHands wires instructions via the OPENHANDS_INSTRUCTIONS env var at launch
    time (no on-disk artifact), so --prepare-only just exercises the rtk-binary
    setup path. The env-var wiring is covered by the unit tests.
    """
    run(
        ["headroom", "wrap", "openhands", "--prepare-only", "--port", str(OPENHANDS_PORT)],
        env=base_env,
        cwd=project_dir,
        timeout=60,
    )


def verify_openclaw_wrap(
    base_env: dict[str, str],
    project_dir: Path,
    plugin_dir: Path,
) -> None:
    port = OPENCLAW_PROXY_PORT
    gateway_proc: subprocess.Popen[str] | None = None
    run(
        [
            "headroom",
            "wrap",
            "openclaw",
            "--plugin-path",
            str(plugin_dir),
            "--proxy-port",
            str(port),
            "--startup-timeout-ms",
            # 5s is too tight for cold Python+pyo3 import on a busy CI runner.
            "30000",
            "--verbose",
        ],
        env=base_env,
        cwd=project_dir,
        timeout=600,
    )
    dist_index = plugin_dir / "dist" / "index.js"
    assert_true(dist_index.exists(), "OpenClaw plugin build should produce dist/index.js")

    config_file = run(["openclaw", "config", "file"], env=base_env, cwd=project_dir, timeout=60)
    config_path_str = config_file.stdout.strip().splitlines()[-1].strip()
    if config_path_str.startswith("~/"):
        config_path = Path(base_env["HOME"]) / config_path_str[2:]
    else:
        config_path = Path(config_path_str)
    assert_true(config_path.exists(), "OpenClaw should create a config file")

    state = json.loads(config_path.read_text(encoding="utf-8"))
    if state.get("gateway", {}).get("mode") != "local":
        run(
            [
                "openclaw",
                "config",
                "set",
                "gateway.mode",
                json.dumps("local"),
                "--strict-json",
            ],
            env=base_env,
            cwd=project_dir,
            timeout=60,
        )
        state = json.loads(config_path.read_text(encoding="utf-8"))

    try:
        try:
            wait_for_command_success(
                ["openclaw", "health"], env=base_env, cwd=project_dir, timeout=5
            )
        except RuntimeError:
            gateway_proc = start_openclaw_gateway(base_env, project_dir)
            try:
                wait_for_command_success(
                    ["openclaw", "health"], env=base_env, cwd=project_dir, timeout=30
                )
            except RuntimeError as exc:
                gateway_output = ""
                if gateway_proc.stdout is not None:
                    gateway_output = gateway_proc.stdout.read()
                raise RuntimeError(f"{exc}\nGateway output:\n{gateway_output}") from exc

        entry = state["plugins"]["entries"]["headroom"]
        assert_true(entry["enabled"] is True, "OpenClaw wrap should enable the plugin")
        assert_true(entry["config"]["proxyPort"] == port, "OpenClaw wrap should set proxy port")
        assert_true(
            entry["config"].get("autoStart", True) is True,
            "OpenClaw wrap should leave autoStart enabled",
        )
        assert_true(
            state["gateway"]["mode"] == "local",
            "OpenClaw e2e bootstrap should set gateway.mode=local",
        )
        assert_true(
            state["plugins"]["slots"]["contextEngine"] == "headroom",
            "OpenClaw wrap should set the context engine slot",
        )
    finally:
        if gateway_proc is not None:
            stop_process(gateway_proc)
        stop_openclaw_gateway(base_env, project_dir)

    run(["headroom", "unwrap", "openclaw"], env=base_env, cwd=project_dir, timeout=120)
    state = json.loads(config_path.read_text(encoding="utf-8"))
    assert_true(
        state["plugins"]["slots"]["contextEngine"] == "legacy",
        "OpenClaw unwrap should restore the context engine slot",
    )


def main() -> None:
    verify_installs()
    with tempfile.TemporaryDirectory(
        prefix="headroom-wrap-e2e-", ignore_cleanup_errors=True
    ) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        home_dir = tmp_dir / "home"
        project_dir = tmp_dir / "project"
        shim_dir = tmp_dir / "shim-bin"
        log_dir = tmp_dir / "logs"

        for path in (home_dir, project_dir, shim_dir, log_dir):
            path.mkdir(parents=True, exist_ok=True)
        create_shims(shim_dir)

        mock_server, mock_thread = start_mock_server(19001)
        base_env = os.environ.copy()
        base_env.update(
            {
                "HOME": str(home_dir),
                "PATH": f"{shim_dir}{os.pathsep}{base_env['PATH']}",
                "HEADROOM_E2E_LOG_DIR": str(log_dir),
                "OPENAI_TARGET_API_URL": "http://127.0.0.1:19001/v1",
            }
        )

        try:
            verify_proxy_round_trip(base_env, mock_server)
            verify_claude_wrap(base_env, project_dir, log_dir)
            verify_codex_wrap(base_env, project_dir, log_dir, mock_server)
            verify_aider_wrap(base_env, project_dir, log_dir)
            verify_cursor_wrap(base_env, project_dir)
            verify_cline_wrap(base_env, project_dir)
            verify_continue_wrap(base_env, project_dir)
            verify_goose_wrap(base_env, project_dir)
            verify_openhands_wrap(base_env, project_dir)
            verify_opencode_wrap(base_env, project_dir, log_dir)
            local_plugin_dir = prepare_local_openclaw_plugin(base_env, tmp_dir)
            verify_openclaw_wrap(base_env, project_dir, local_plugin_dir)
        finally:
            mock_server.shutdown()
            mock_thread.join(timeout=5)

    log("All Docker wrap e2e checks passed.")


def verify_opencode_wrap(base_env: dict[str, str], project_dir: Path, log_dir: Path) -> None:
    port = OPENCODE_PORT
    run(
        ["headroom", "wrap", "opencode", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    global_agents = Path(base_env["HOME"]) / ".config" / "opencode" / "AGENTS.md"
    project_agents = project_dir / "AGENTS.md"
    assert_true(global_agents.exists(), "Opencode wrap should create ~/.config/opencode/AGENTS.md")
    assert_true(project_agents.exists(), "Opencode wrap should create project AGENTS.md")
    assert_true(
        RTK_MARKER in global_agents.read_text(encoding="utf-8"),
        "Missing RTK marker in global AGENTS.md",
    )
    assert_true(
        RTK_MARKER in project_agents.read_text(encoding="utf-8"),
        "Missing RTK marker in project AGENTS.md",
    )

    entries = read_jsonl(log_dir / "opencode.jsonl")
    assert_true(len(entries) > 0, "Opencode shim should have been invoked")
    env_vars = entries[-1]["env"]
    assert_true(
        env_vars.get("OPENCODE_CONFIG_CONTENT") is not None,
        "Opencode wrap should set OPENCODE_CONFIG_CONTENT",
    )
    config = json.loads(env_vars["OPENCODE_CONFIG_CONTENT"])
    assert_true(
        config["provider"]["headroom"]["options"]["baseURL"] == f"http://127.0.0.1:{port}/v1",
        "Opencode wrap should inject headroom provider baseURL",
    )

    run(
        ["headroom", "unwrap", "opencode", "--port", str(port)],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    config_path = Path(base_env["HOME"]) / ".config" / "opencode" / "opencode.json"
    if config_path.exists():
        content = config_path.read_text(encoding="utf-8")
        assert_true(
            "headroom" not in content,
            "Opencode unwrap should remove headroom provider from config",
        )


if __name__ == "__main__":
    main()

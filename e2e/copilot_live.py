"""Credential-safe live E2E checks for Headroom's GitHub Copilot integrations.

Run from a source checkout after ``headroom copilot-auth login``::

    uv run --no-sync python e2e/copilot_live.py

The script never reads or prints a credential. It uses an isolated VS Code
settings file, verifies restoration byte-for-byte, and sends small live prompts
through both the Copilot CLI wrapper and the VS Code proxy route.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def run(command: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command[:4])}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def wait_for_health(port: int, timeout: float = 45) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:  # noqa: S310 - loopback E2E target
                return json.load(response)
        except (OSError, URLError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError(f"Headroom did not become healthy on port {port}")


def post_response(port: int, project: str, model: str) -> dict[str, object]:
    payload = json.dumps(
        {"model": model, "input": "Reply with exactly: HEADROOM_VSCODE_OK", "stream": False}
    ).encode()
    url = f"http://127.0.0.1:{port}/p/{quote(project, safe='')}/v1/responses"
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:  # noqa: S310 - loopback E2E target
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"live VS Code route returned HTTP {exc.code}: {body}") from exc


def request_count(port: int) -> float:
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:  # noqa: S310
        metrics = response.read().decode("utf-8", errors="replace")
    match = re.search(r"^headroom_requests_total ([0-9.eE+-]+)$", metrics, re.MULTILINE)
    if not match:
        raise AssertionError("Headroom metrics did not expose headroom_requests_total")
    return float(match.group(1))


def model_request_count(port: int, model: str) -> int:
    with urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as response:  # noqa: S310
        stats = json.load(response)
    requests = stats.get("requests", {})
    by_model = requests.get("by_model", {}) if isinstance(requests, dict) else {}
    value = by_model.get(model, 0) if isinstance(by_model, dict) else 0
    return int(value)


def stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    stdout, _ = process.communicate()
    return stdout


def assert_safe_settings(settings: str, proxy_url: str) -> None:
    required = (
        "github.copilot.advanced.debug.overrideProxyUrl",
        "github.copilot.advanced.debug.overrideCapiUrl",
    )
    for key in required:
        if key not in settings or proxy_url not in settings:
            raise AssertionError(f"VS Code settings did not contain {key}")
    forbidden = ("overrideAuthType", "token", "bearer", '"model"')
    for value in forbidden:
        if value.lower() in settings.lower():
            raise AssertionError(f"VS Code settings unexpectedly contained {value}")


def wait_for_settings(path: Path, timeout: float = 15) -> tuple[str, str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        settings = path.read_text(encoding="utf-8")
        if all(
            key in settings
            for key in (
                "github.copilot.advanced.debug.overrideProxyUrl",
                "github.copilot.advanced.debug.overrideCapiUrl",
            )
        ):
            urls = re.findall(
                r'github\.copilot\.advanced\.debug\.override(?:Proxy|Capi)Url"\s*:\s*"([^"]+)',
                settings,
            )
            if len(urls) != 2 or urls[0] != urls[1]:
                raise AssertionError("VS Code proxy and CAPI settings did not use the same URL")
            assert_safe_settings(settings, urls[0])
            return settings, urls[0]
        time.sleep(0.1)
    raise TimeoutError("Headroom did not finish writing the VS Code Copilot settings block")


def response_text(payload: dict[str, object]) -> str:
    """Return text fragments without depending on one Responses API SDK shape."""

    fragments: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"text", "output_text"} and isinstance(child, str):
                    fragments.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return "\n".join(fragments)


def model_matches_request(requested: str, returned: object) -> bool:
    """Allow GitHub's dated canonical name for an otherwise preserved alias."""

    return isinstance(returned, str) and (
        returned == requested or returned.startswith(f"{requested}-")
    )


def verify_real_vscode_extension(
    *, headroom: str, code: str, port: int, settings_path: Path
) -> None:
    """Drive the installed VS Code Copilot extension through ``code chat``."""

    if os.name != "nt":
        raise RuntimeError("--vscode-extension currently targets the Windows release gate")
    code_binary = shutil.which(code)
    if not code_binary:
        raise RuntimeError(f"VS Code executable not found on PATH: {code}")
    settings_existed = settings_path.exists()
    original = settings_path.read_bytes() if settings_existed else None
    if original and b"Headroom Copilot proxy" in original:
        raise RuntimeError("real VS Code settings already contain a Headroom-managed block")

    process = subprocess.Popen(
        [headroom, "wrap", "vscode", "--port", str(port)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    output = ""
    try:
        deadline = time.monotonic() + 45
        proxy_url = ""
        while time.monotonic() < deadline:
            if settings_path.exists():
                settings = settings_path.read_text(encoding="utf-8")
                match = re.search(r'overrideCapiUrl"\s*:\s*"(http://127\.0\.0\.1:\d+)', settings)
                if match:
                    proxy_url = match.group(1)
                    break
            if process.poll() is not None:
                raise RuntimeError("VS Code wrapper exited before configuring settings")
            time.sleep(0.1)
        if not proxy_url:
            raise TimeoutError("VS Code wrapper did not configure the real user settings")
        actual_port = int(proxy_url.rsplit(":", 1)[1])
        wait_for_health(actual_port)
        before = request_count(actual_port)
        run(
            [
                code_binary,
                "chat",
                "-m",
                "ask",
                "-r",
                "Reply with exactly: HEADROOM_VSCODE_EXTENSION_OK",
            ],
            timeout=30,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and request_count(actual_port) <= before:
            time.sleep(1)
        if request_count(actual_port) <= before:
            raise TimeoutError("the installed VS Code extension sent no request through Headroom")
    finally:
        output = stop_process(process)
        run([headroom, "unwrap", "vscode"])

    if settings_existed:
        if settings_path.read_bytes() != original:
            raise AssertionError("real VS Code settings were not restored byte-for-byte")
    elif settings_path.exists() and settings_path.read_bytes().strip() not in {b"", b"{}"}:
        raise AssertionError("VS Code settings were created but not restored to an empty state")
    if "remained running after shutdown" in output.lower():
        raise AssertionError("Windows wrapper orphaned its dedicated proxy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom", default="headroom", help="Headroom executable to test")
    parser.add_argument("--copilot", default="copilot", help="Copilot CLI executable to test")
    parser.add_argument("--model", action="append", default=[], help="Live model ID (repeatable)")
    parser.add_argument("--port", type=int, default=28787)
    parser.add_argument(
        "--vscode-extension",
        action="store_true",
        help="also modify real VS Code settings and drive the installed extension via `code chat`",
    )
    parser.add_argument("--code", default="code", help="VS Code executable to test")
    args = parser.parse_args()
    models = args.model or ["gpt-5-mini"]

    status = run([args.headroom, "copilot-auth", "status"])
    if "Status: logged in" not in status.stdout:
        raise RuntimeError("Headroom Copilot auth is missing; run `headroom copilot-auth login`")

    baseline = run(
        [args.copilot, "-p", "Reply with exactly: COPILOT_BASELINE_OK", "--model", models[0]]
    )
    if "COPILOT_BASELINE_OK" not in baseline.stdout:
        raise AssertionError("plain Copilot CLI did not return its sentinel")

    wrapped = run(
        [
            args.headroom,
            "wrap",
            "copilot",
            "--subscription",
            "--",
            "--model",
            models[0],
            "-p",
            "Reply with exactly: HEADROOM_CLI_OK",
        ]
    )
    if "HEADROOM_CLI_OK" not in wrapped.stdout:
        raise AssertionError("wrapped Copilot CLI did not return its sentinel")

    original = b'{\n  // preserved by Headroom\n  "editor.fontSize": 15,\n}\n'
    with tempfile.TemporaryDirectory(prefix="headroom-copilot-e2e-") as temp:
        settings_path = Path(temp) / "settings.json"
        settings_path.write_bytes(original)
        command = [
            args.headroom,
            "wrap",
            "vscode",
            "--port",
            str(args.port),
            "--settings-file",
            str(settings_path),
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", args.port))
        occupied.listen()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        output = ""
        try:
            _, proxy_url = wait_for_settings(settings_path)
            expected_suffix = f"/p/{quote(Path.cwd().name, safe='')}"
            if not proxy_url.endswith(expected_suffix):
                raise AssertionError(f"unexpected project-scoped proxy URL: {proxy_url}")
            actual_port = int(proxy_url.removesuffix(expected_suffix).rsplit(":", 1)[1])
            if actual_port == args.port:
                raise AssertionError("wrapper did not fall back from the occupied requested port")
            health = wait_for_health(actual_port)
            upstream = str(health.get("config", {}).get("openai_api_url", ""))
            if "githubcopilot.com" not in upstream:
                raise AssertionError(f"unexpected Copilot upstream host: {upstream}")
            project = Path.cwd().name
            for model in models:
                before = model_request_count(actual_port, model)
                response = post_response(actual_port, project, model)
                returned_model = response.get("model")
                if not model_matches_request(model, returned_model):
                    raise AssertionError(
                        f"model was not preserved: requested {model!r}, got {returned_model!r}"
                    )
                if "HEADROOM_VSCODE_OK" not in response_text(response):
                    raise AssertionError(f"model {model!r} did not return the expected sentinel")
                if model_request_count(actual_port, model) != before + 1:
                    raise AssertionError(f"Headroom traffic accounting missed model {model!r}")
        finally:
            output = stop_process(process)
            run([args.headroom, "unwrap", "vscode", "--settings-file", str(settings_path)])
            occupied.close()
        if settings_path.read_bytes() != original:
            raise AssertionError("VS Code settings were not restored byte-for-byte")
        if any(marker in output.lower() for marker in ("authorization: bearer", "github token")):
            raise AssertionError("wrapper output contained a credential-shaped marker")

    if args.vscode_extension:
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            raise RuntimeError("APPDATA is required to locate stable VS Code settings on Windows")
        verify_real_vscode_extension(
            headroom=args.headroom,
            code=args.code,
            port=args.port + 100,
            settings_path=Path(appdata) / "Code" / "User" / "settings.json",
        )

    print(
        "PASS: Copilot baseline, CLI wrap, port fallback, VS Code routing, "
        "model swaps/accounting, restore"
        + (", and installed VS Code extension" if args.vscode_extension else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

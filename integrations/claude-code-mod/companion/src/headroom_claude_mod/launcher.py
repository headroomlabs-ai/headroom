"""Launch Claude with explicit, isolated correlation; never edit user settings."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path

HEADER = "x-headroom-mod-session"
RESERVED = {"--session-id", "--resume", "-r", "--continue", "-c", "--fork-session"}
NON_INTERACTIVE = {
    "--bg",
    "--background",
    "--desktop",
    "--cloud",
    "--environment",
    "--exec",
    "--bare",
    "--safe-mode",
    "--disable-slash-commands",
    "--print",
    "-p",
}


def local_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port if parsed.port is not None else 8787
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or not 1 <= port <= 65535
        ):
            raise ValueError
        host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
        return f"http://{host}:{port}"
    except ValueError as exc:
        raise ValueError(
            "Use a local HTTP proxy origin, e.g. http://127.0.0.1:8787 (no path or credentials)"
        ) from exc


def child_environment(parent: Mapping[str, str], base: str, sid: str) -> dict[str, str]:
    base = local_url(base)
    sid = str(uuid.UUID(sid))
    inherited = parent.get("ANTHROPIC_BASE_URL")
    if inherited:
        try:
            same = local_url(inherited) == base
        except ValueError:
            same = False
        if not same:
            raise ValueError(
                "ANTHROPIC_BASE_URL already selects another upstream. Configure that upstream on Headroom, then explicitly unset it in this shell before launching; it was not overwritten."
            )
    for flag in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        if parent.get(flag, "").lower() not in {"", "0", "false"}:
            raise ValueError(
                f"{flag} bypasses this launcher. This release supports the Anthropic-compatible local proxy path only."
            )
    raw = parent.get("ANTHROPIC_CUSTOM_HEADERS", "")
    if len(raw) > 65_536:
        raise ValueError("ANTHROPIC_CUSTOM_HEADERS exceeds the safe limit")
    kept = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        key, separator, val = line.partition(":")
        if (
            not separator
            or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key)
            or any(ord(c) < 32 and c != "\t" for c in val)
        ):
            raise ValueError(
                "Malformed ANTHROPIC_CUSTOM_HEADERS; existing headers were not changed"
            )
        if key.lower() != HEADER:
            kept.append(line)
    env = dict(parent)
    env.update(
        {
            "ANTHROPIC_BASE_URL": base,
            "HEADROOM_MOD_URL": base,
            "HEADROOM_MOD_SESSION_ID": sid,
            "ANTHROPIC_CUSTOM_HEADERS": "\n".join([*kept, f"X-Headroom-Mod-Session: {sid}"]),
        }
    )
    # Sidebar startup must resolve the companion installed alongside this launcher,
    # including bundled runtimes invoked by an absolute executable path.
    env["PATH"] = sysconfig.get_path("scripts") + os.pathsep + env.get("PATH", "")
    return env


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def preflight(base: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        local_url(base) + "/headroom-mod/v1/health", headers={"Accept": "application/json"}
    )
    try:
        with opener.open(request, timeout=3) as response:
            raw = response.read(16_385)
            if len(raw) > 16_384:
                raise ValueError("Oversized companion response")
            data = json.loads(raw)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ValueError(
            "Headroom's claude_mod extension is unavailable. Install the companion in Headroom's environment and start: headroom proxy --proxy-extension claude_mod"
        ) from exc
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("service") != "headroom-claude-mod"
        or (data.get("read_only") is not True and data.get("session_controls") is not True)
    ):
        raise ValueError("The endpoint is not a compatible Headroom mod companion")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="Start a new/resumed Claude session through an already-running Headroom proxy"
    )
    run.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    run.add_argument(
        "--plugin-dir",
        type=Path,
        help="Local plugin directory; omit when installed from the bundled marketplace",
    )
    group = run.add_mutually_exclusive_group()
    group.add_argument("--session-id", type=uuid.UUID, help="Explicit new Claude session UUID")
    group.add_argument(
        "--resume", type=uuid.UUID, help="Resume this exact UUID, not the interactive selector"
    )
    run.add_argument("--claude-executable", default="claude")
    run.add_argument("claude_args", nargs=argparse.REMAINDER)
    check = sub.add_parser("doctor", help="Check the local companion without starting Claude")
    check.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    start = sub.add_parser(
        "start", help="Start the local Headroom companion without launching Claude"
    )
    start.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    args = parser.parse_args(argv)
    try:
        base = local_url(args.proxy_url)
        if args.command == "start":
            from .startup import start_proxy

            start_proxy(base, preflight)
            print(
                "Headroom is ready. Restart Claude with headroom-mod run to route this conversation through it."
            )
            return 0
        if args.command == "doctor":
            info = preflight(base)
            print(
                f"Headroom mod companion {info['version']}; capture={info.get('log_full_messages', False)}; session-controls={info.get('session_controls', False)}; schema=1"
            )
            return 0
        forwarded = args.claude_args
        if forwarded[:1] == ["--"]:
            forwarded = forwarded[1:]
        if any(a.split("=", 1)[0] in RESERVED for a in forwarded):
            raise ValueError(
                "Put --session-id/--resume BEFORE the -- separator. --continue and --fork-session are not supported; exact session correlation is required."
            )
        if any(a.split("=", 1)[0] in NON_INTERACTIVE for a in forwarded):
            raise ValueError(
                "This release requires an interactive local Claude session with plugins enabled; remote, Desktop handoff, background, bare and print modes are not supported by the launcher."
            )
        sid = str(args.resume or args.session_id or uuid.uuid4())
        env = child_environment(os.environ, base, sid)
        executable = shutil.which(args.claude_executable)
        if executable is None:
            raise ValueError(
                "Claude executable not found; install Claude Code 2.1.287+ or use --claude-executable"
            )
        if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
            raise ValueError(
                "Use the native Claude .exe with --claude-executable on Windows; batch shims are refused to avoid shell argument re-interpretation."
            )
        command = [executable]
        if args.plugin_dir:
            plugin = args.plugin_dir.expanduser().resolve()
            if not (plugin / ".claude-plugin" / "plugin.json").is_file():
                raise ValueError(
                    "--plugin-dir must name the directory containing .claude-plugin/plugin.json"
                )
            command += ["--plugin-dir", str(plugin)]
        command += ["--resume" if args.resume else "--session-id", sid, *forwarded]
        preflight(base)
        # No shell, settings writes, process detachment, proxy restarts, or credential printing.
        return subprocess.run(command, env=env, check=False).returncode
    except (ValueError, OSError) as exc:
        print(f"headroom-mod: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

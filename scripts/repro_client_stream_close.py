"""Reproduce client stream cleanup against a local SSE server.

Run the unread-stream proof with the checked-in defaults::

    uv run python scripts/repro_client_stream_close.py
    uv run python scripts/repro_client_stream_close.py --cycles 1001

Pass ``--started`` to consume the first event before closing each stream.
The baseline client method is loaded from ``--baseline-ref`` in a fresh child
process; it is never taken from the current checkout.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASELINE_REF = "3fef15d376d9899f77a28464e2e91f8b99514930"


def rss_bytes() -> int | None:
    """Return this process's RSS using stdlib APIs where available."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize)
        return None
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            pass
    return None


def pool_connections(client: object) -> int | None:
    try:
        return len(client._client._transport._pool.connections)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None


def load_baseline_method(module: object, ref: str) -> None:
    """Install exactly ``HeadroomClient._wrap_stream`` from ``git show``."""
    source = subprocess.check_output(["git", "show", f"{ref}:headroom/client.py"], text=True)
    tree = ast.parse(source)
    client_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HeadroomClient"
    )
    method = next(
        node
        for node in client_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_wrap_stream"
    )
    namespace = dict(vars(module))
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<baseline>", "exec"
    )
    exec(code, namespace)
    module.HeadroomClient._wrap_stream = namespace["_wrap_stream"]  # type: ignore[attr-defined]


class SSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", "1000000")
        self.end_headers()
        self.wfile.write(
            b'data: {"id":"local","object":"chat.completion.chunk",'
            b'"created":0,"model":"local","choices":[{"index":0,'
            b'"delta":{"content":"hi"},"finish_reason":null}]}'
            b"\n\n"
        )
        self.wfile.flush()
        self.connection.settimeout(120)
        try:
            self.connection.recv(1)
        except OSError:
            pass
        self.close_connection = True


def write_progress(path: Path, completed: int, samples: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"completed": completed, "samples": samples}),
        encoding="utf-8",
        newline="\n",
    )


def child(args: argparse.Namespace) -> None:
    import httpx
    from openai import OpenAI

    import headroom.client as client_module
    from headroom.providers.openai import OpenAIProvider

    if args.variant == "baseline":
        load_baseline_method(client_module, args.baseline_ref)

    progress_path = Path(args.progress)
    samples: list[dict[str, object]] = []
    with OpenAI(
        api_key="local-only",
        base_url=args.url,
        max_retries=0,
        http_client=httpx.Client(
            trust_env=False,
            limits=httpx.Limits(max_connections=1000),
            timeout=httpx.Timeout(10, pool=0.25),
        ),
    ) as original:
        client = client_module.HeadroomClient(
            original_client=original,
            provider=OpenAIProvider(),
            store_url=f"jsonl://{Path(args.store).resolve().as_posix()}",
            enable_cache_optimizer=False,
        )

        def request() -> None:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            if args.started:
                next(stream)
            stream.close()
            del stream

        for _ in range(5):
            warmup = original.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            next(warmup)
            warmup.close()
            del warmup
        warmup = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        next(warmup)
        warmup.close()
        del warmup
        gc.collect()
        import tracemalloc

        if not args.no_trace:
            tracemalloc.start(1)

        def sample(phase: str, cycles: int) -> None:
            heap, peak = tracemalloc.get_traced_memory()
            samples.append(
                {
                    "phase": phase,
                    "cycles": cycles,
                    "rss": rss_bytes(),
                    "heap": heap,
                    "heap_peak": peak,
                    "connections": pool_connections(original),
                }
            )
            write_progress(progress_path, completed, samples)

        completed = 0
        sample("baseline", 0)
        started_at = time.perf_counter()
        error: dict[str, object] | None = None
        for request_number in range(1, args.cycles + 1):
            try:
                request()
            except Exception as exc:  # includes httpx.PoolTimeout
                cause = exc.__cause__
                error = {
                    "type": type(exc).__name__,
                    "cause": type(cause).__name__ if cause is not None else None,
                    "request": request_number,
                    "completed": completed,
                    "pool_timeout": type(exc).__name__ == "PoolTimeout"
                    or type(cause).__name__ == "PoolTimeout",
                }
                break
            completed += 1
            if request_number in (10, 50, 100, 250, 500, 1000, args.cycles):
                sample("running", request_number)
        elapsed = time.perf_counter() - started_at
        sample("before_gc", completed)
        gc.collect()
        time.sleep(0.05)
        sample("after_gc", completed)

    print(
        json.dumps(
            {
                "variant": args.variant,
                "started": args.started,
                "cycles": args.cycles,
                "completed": completed,
                "error": error,
                "elapsed": elapsed,
                "gc_enabled": gc.isenabled(),
                "samples": samples,
            }
        ),
        flush=True,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def run_child(
    args: argparse.Namespace,
    server_port: int,
    variant: str,
    temporary: Path,
) -> dict[str, object]:
    progress = temporary / "progress.json"
    store = temporary / "metrics.jsonl"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--variant",
        variant,
        "--baseline-ref",
        args.baseline_ref,
        "--cycles",
        str(args.cycles),
        "--url",
        f"http://127.0.0.1:{server_port}/v1",
        "--progress",
        str(progress),
        "--store",
        str(store),
    ]
    if args.started:
        command.append("--started")
    if args.no_trace:
        command.append("--no-trace")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HEADROOM_OFFLINE": "1"},
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        stop_process(process)
        record: dict[str, object] = {
            "variant": variant,
            "started": args.started,
            "cycles": args.cycles,
            "completed": None,
            "error": {"type": "TimeoutExpired"},
        }
        if progress.exists():
            record.update(json.loads(progress.read_text(encoding="utf-8")))
        return record
    if process.returncode:
        raise RuntimeError(
            f"{variant} child exited {process.returncode}: {stdout[-1000:]} {stderr[-1000:]}"
        )
    return json.loads(stdout)


def parent(args: argparse.Namespace) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for repeat in range(1, args.repeats + 1):
            for variant in ("baseline", "fixed"):
                with tempfile.TemporaryDirectory(prefix="headroom-stream-close-") as directory:
                    record = run_child(args, server.server_port, variant, Path(directory))
                record["repeat"] = repeat
                print(json.dumps(record), flush=True)
    finally:
        server.shutdown()
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--started", action="store_true")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--variant", choices=("baseline", "fixed"), help=argparse.SUPPRESS)
    parser.add_argument("--url", help=argparse.SUPPRESS)
    parser.add_argument("--progress", help=argparse.SUPPRESS)
    parser.add_argument("--store", help=argparse.SUPPRESS)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.child:
        child(arguments)
    else:
        parent(arguments)

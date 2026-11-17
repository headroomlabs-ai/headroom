import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["HEADROOM_SKIP_UPSTREAM_CHECK"] = "1"
os.environ["HEADROOM_DISTRIBUTION"] = "local"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
state = tempfile.TemporaryDirectory(prefix="headroom-sdk-proof-")
os.environ["HEADROOM_WORKSPACE_DIR"] = state.name
os.environ["HEADROOM_CONFIG_DIR"] = str(Path(state.name) / "config")
sys.path.insert(0, str(root))
import uvicorn

from headroom.proxy.server import ProxyConfig, create_app

app = create_app(
    ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        disable_kompress=True,
        subscription_tracking_enabled=False,
        stateless=True,
    )
)
seen = []


@app.middleware("http")
async def count_compress(request, call_next):
    response = await call_next(request)
    if request.url.path == "/v1/compress":
        seen.append(response.status_code)
    return response


listener = socket.socket()
listener.bind(("127.0.0.1", 0))
port = listener.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
thread.start()
try:
    for _ in range(300):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("proxy stopped before startup")
        time.sleep(0.1)
    assert server.started
    env = dict(os.environ, HEADROOM_BASE_URL=f"http://127.0.0.1:{port}")
    command = ["node", str(Path(__file__).with_suffix(".mjs")), str(root / "sdk/typescript")]
    if len(sys.argv) > 2:
        command.append(sys.argv[2])
    result = subprocess.run(command, env=env)
    assert seen and all(status == 200 for status in seen), seen
    print("Real Headroom /v1/compress response:", seen, flush=True)
    raise SystemExit(result.returncode)
finally:
    server.should_exit = True
    thread.join(timeout=15)
    listener.close()
    state.cleanup()

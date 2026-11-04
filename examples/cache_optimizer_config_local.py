"""Actual OpenAI SDK HTTP requests: typed cache opt-out must preserve caller spaces."""

import hashlib
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory(prefix="headroom-cache-config-") as scratch:
        for key in ["HEADROOM_WORKSPACE_DIR", "HEADROOM_CONFIG_DIR", "XDG_CACHE_HOME"]:
            os.environ[key] = str(Path(scratch) / key.lower())
        # This isolated proof needs no persistent, process-global CCR database.
        # Keep the store in memory so Windows can remove the scratch directory.
        os.environ.update(
            HEADROOM_OFFLINE="1",
            HEADROOM_BEACON="off",
            DO_NOT_TRACK="1",
            HEADROOM_CCR_BACKEND="memory",
        )
        from openai import OpenAI

        from headroom import HeadroomClient, OpenAIProvider
        from headroom.config import HeadroomConfig, HeadroomMode

        requests = []
        violations = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                wire_dir = os.environ.get("HEADROOM_PROOF_WIRE_DIR")
                if wire_dir:
                    target = Path(wire_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"request-{len(requests)}.json").write_bytes(raw)
                requests.append(body)
                payload = json.dumps(
                    {
                        "id": "chatcmpl_fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "local fixture"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        original = OpenAI(
            api_key="local-fixture",
            base_url=f"http://127.0.0.1:{server.server_port}",
            max_retries=0,
        )
        try:
            for enabled in [False, True]:
                url = f"sqlite:///{scratch}/{enabled}.db"
                config = HeadroomConfig(
                    store_url=url,
                    default_mode=HeadroomMode.OPTIMIZE,
                    discover_pipeline_extensions=False,
                )
                config.smart_crusher.enabled = False
                config.cache_aligner.enabled = False
                config.cache_optimizer.enabled = enabled
                client = HeadroomClient(original, OpenAIProvider(), store_url=url, config=config)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Keep  exact   spaces.\n"},
                            {"role": "user", "content": "Hello."},
                        ],
                        max_tokens=8,
                    )
                    wire_system = requests[-1]["messages"][0]["content"]
                    print(
                        json.dumps(
                            {
                                "configured_enabled": enabled,
                                "runtime_optimizer": client.get_stats()["config"][
                                    "cache_optimizer"
                                ],
                                "wire_system": wire_system,
                                "wire_system_sha256": hashlib.sha256(
                                    wire_system.encode()
                                ).hexdigest(),
                                "response": response.choices[0].message.content,
                            }
                        )
                    )
                    if not enabled and wire_system != "Keep  exact   spaces.\n":
                        violations.append("Disabled cache optimizer modified the system message")
                finally:
                    client.close()
        finally:
            original.close()
            server.shutdown()
            server.server_close()

        assert not violations, violations


if __name__ == "__main__":
    main()

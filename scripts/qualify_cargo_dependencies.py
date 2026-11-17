#!/usr/bin/env python3
"""Reproduce the Cargo parser canary and credentialless native proxy proof.

Standard-library-only driver. Run from any directory; Cargo uses the repository's
pinned toolchain. The parser comparison compiles two temporary standalone crates
and uses the eight exact production grammar pins. Runtime proof requires an
already-built headroom-proxy binary and never enables model compression.
"""

import argparse
import hashlib
import http.server
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUST_CANARY = r"""
use serde_json::{json, Value};
fn visit(n: tree_sitter::Node, out: &mut Vec<Value>) {
    out.push(json!([
        n.kind(),
        n.start_position().row,
        n.start_position().column,
        n.end_position().row,
        n.end_position().column,
        n.is_error(),
        n.is_missing()
    ]));
    let mut c = n.walk();
    for child in n.children(&mut c) {
        visit(child, out);
    }
}
fn main() {
    let inputs: Vec<Value> = serde_json::from_str(include_str!("../inputs.json")).unwrap();
    let mut outputs = vec![];
    for input in inputs {
        let l = input["language"].as_str().unwrap();
        let language: tree_sitter::Language = match l {
            "python" => tree_sitter_python::LANGUAGE.into(),
            "javascript" => tree_sitter_javascript::LANGUAGE.into(),
            "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "go" => tree_sitter_go::LANGUAGE.into(),
            "rust" => tree_sitter_rust::LANGUAGE.into(),
            "java" => tree_sitter_java::LANGUAGE.into(),
            "c" => tree_sitter_c::LANGUAGE.into(),
            "cpp" => tree_sitter_cpp::LANGUAGE.into(),
            _ => panic!(),
        };
        let mut p = tree_sitter::Parser::new();
        p.set_language(&language).unwrap();
        let tree = p.parse(input["source"].as_str().unwrap(), None).unwrap();
        let mut nodes = vec![];
        visit(tree.root_node(), &mut nodes);
        outputs.push(json!({"language":l,"sample":input["sample"],"nodes":nodes}));
    }
    println!("{}", serde_json::to_string(&outputs).unwrap());
}
"""


def parser_canary(artifacts, target):
    crate = artifacts / "parser-canary"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for path in sorted((ROOT / "tests/parity/fixtures/code_aware_compressor").glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        sources = fixtures.setdefault(fixture["output"]["language"], [])
        if fixture["input"] not in sources:
            sources.append(fixture["input"])
    fixtures["c"] = [
        "struct Point { int x; int y; };\nint main(void) { struct Point p={1,2}; return p.x; }\n",
        "int apply(int (*fn)(int), int x) { return fn(x); }\nint square(int x) { return x*x; }\nint main(void) { return apply(square, 4); }\n",
    ]
    inputs = []
    for language in ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp"):
        original, second = fixtures[language][:2]
        samples = [
            original,
            "",
            original + "\n",
            second,
            original.replace("\n", "\r\n"),
            original[: len(original) // 2],
            original + "\n} ) ]",
            original + "\n" + original,
            original.replace("Processor", "Caf\u00e9")
            if "Processor" in original
            else original + "\n\u03bb\u4e2d\U0001f600",
        ]
        inputs.extend(
            {"language": language, "sample": i + 1, "source": source}
            for i, source in enumerate(samples)
        )
    (crate / "inputs.json").write_text(
        json.dumps(inputs, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (crate / "src/main.rs").write_text(RUST_CANARY, encoding="utf-8", newline="\n")
    manifest = """[package]
name="headroom-parser-qualification"
version="0.0.0"
edition="2021"
[workspace]
[dependencies]
tree-sitter="=VERSION"
serde_json="=1.0.154"
tree-sitter-python="=0.25.0"
tree-sitter-javascript="=0.25.0"
tree-sitter-typescript="=0.23.2"
tree-sitter-go="=0.25.0"
tree-sitter-rust="=0.24.2"
tree-sitter-java="=0.23.5"
tree-sitter-c="=0.24.2"
tree-sitter-cpp="=0.23.4"
"""
    outputs = []
    for version in ("0.27.0", "0.27.1"):
        (crate / "Cargo.toml").write_text(
            manifest.replace("VERSION", version), encoding="utf-8", newline="\n"
        )
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(crate / "Cargo.toml"),
                "--target-dir",
                str(target),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        (artifacts / f"parser-{version}.json").write_text(
            result.stdout, encoding="utf-8", newline="\n"
        )
        outputs.append(json.loads(result.stdout))
    assert len(outputs[0]) == len(outputs[1]) == 72
    for before, after in zip(*outputs, strict=True):
        assert before == after, (before["language"], before["sample"])
        print("AST", before["language"], before["sample"], "PASS", len(before["nodes"]), "nodes")
    print(
        "PASS 72/72 reconstructed samples: 8 pinned languages x 9 samples; historical corpus unavailable"
    )


def native_runtime(binary, artifacts):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = b""
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                while True:
                    size = int(self.rfile.readline().strip().split(b";")[0], 16)
                    if not size:
                        self.rfile.readline()
                        break
                    body += self.rfile.read(size)
                    self.rfile.read(2)
            else:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            seen.append((self.path, dict(self.headers), body))
            answer = b'{"content":[{"type":"text","text":"fixture-ok"}],"usage":{"input_tokens":3,"output_tokens":2}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{server.server_port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "HEADROOM_PROXY_"))}
    env.update(
        AWS_ACCESS_KEY_ID="AKIDEXAMPLE",
        AWS_SECRET_ACCESS_KEY="fake-local-test-secret",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_CONFIG_FILE=os.devnull,
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        AWS_REGION="us-east-1",
        HEADROOM_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        NO_PROXY="127.0.0.1,localhost",
        TRANSFORMERS_OFFLINE="1",
    )
    log = open(artifacts / "runtime-proxy.log", "w", encoding="utf-8", newline="\n")
    proc = subprocess.Popen(
        [
            str(binary),
            "--listen",
            f"127.0.0.1:{port}",
            "--upstream",
            base,
            "--enable-bedrock-native",
            "true",
            "--bedrock-endpoint",
            base,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(150):
            if proc.poll() is not None:
                raise RuntimeError(f"proxy exited {proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        payload = b'{"anthropic_version":"bedrock-2023-05-31","max_tokens":8,"messages":[{"role":"user","content":"local fixture"}]}'
        for path in ["/v1/messages", "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"]:
            with opener.open(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}" + path,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=15,
            ) as response:
                answer = json.load(response)
                assert answer["content"][0]["text"] == "fixture-ok"
                rid = uuid.UUID(response.headers["x-request-id"])
                assert rid.version == 4
                print(path, response.status, "UUIDv4", str(rid))
        assert len(seen) == 2, seen
        assert all(body == payload for _, _, body in seen)
        headers = {k.lower(): v for k, v in seen[-1][1].items()}
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
        assert headers["x-amz-content-sha256"] == hashlib.sha256(payload).hexdigest()
        print(
            "PASS: built binary passthrough + AWS environment provider/SigV4 local wire proof; body byte-equal; payload hash verified. No real AWS/SDK service response emulation."
        )
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)
        finally:
            try:
                server.shutdown()
            finally:
                try:
                    server.server_close()
                finally:
                    log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("canary", "runtime", "all"))
    parser.add_argument("--binary", type=Path, help="built headroom-proxy executable")
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument(
        "--canary-target-dir", type=Path, default=ROOT / "target/cargo-qualification"
    )
    args = parser.parse_args()
    if args.mode in ("runtime", "all") and not args.binary:
        parser.error("--binary is required for runtime/all")
    with tempfile.TemporaryDirectory(prefix="headroom-cargo-proof-") as temporary:
        artifacts = (args.artifacts_dir or Path(temporary)).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        if args.mode in ("canary", "all"):
            parser_canary(artifacts, args.canary_target_dir.resolve())
        if args.mode in ("runtime", "all"):
            native_runtime(args.binary.resolve(strict=True), artifacts)


if __name__ == "__main__":
    main()

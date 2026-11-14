# mypy: disable-error-code="no-untyped-def"
"""Exercise generated clients over a real loopback HTTP mock, without providers.

This does not start the Headroom proxy or qualify its middleware/lifecycle.
Use --fixture only when running the delivered standalone source-fixture bundle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .compiler import Compiler, load_sources
from .emit import render


def tool_command(name: str, *arguments: str) -> list[str]:
    """Return an executable argv, including Windows command shims."""
    executable = shutil.which(name)
    if executable is None:
        raise SystemExit(f"Required conformance tool not available: {name}")
    if os.name == "nt" and executable.lower().endswith((".cmd", ".bat")):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", executable, *arguments]
    return [executable, *arguments]


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[dict] = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith("/prefix/v1/retrieve/"):
            self.reply(unquote(self.path.split("/prefix/v1/retrieve/", 1)[1]))
        else:
            self.send_error(500, "unexpected path (possibly a followed redirect)")

    def do_POST(self):
        if self.path != "/prefix/v1/retrieve":
            self.send_error(500)
            return
        value = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.reply(value["hash"])

    def reply(self, key):
        self.seen.append({"method": self.command, "path": self.path, "key": key})
        if key.startswith("retry_probe_"):
            self.close_connection = True
            return
        if key == "slow":
            time.sleep(0.25)
        status = 200
        media = "application/json"
        value = {
            "hash": key,
            "original_content": '{"snake_case":"世界"}',
            "original_tokens": 100,
            "original_item_count": 10,
            "compressed_item_count": 2,
            "tool_name": None,
            "retrieval_count": 1,
            "future_extension": {"snake_case": "unchanged"},
        }
        if key == "missing":
            status = 404
            value = {"detail": "Entry missing"}
        if key == "missing_field":
            del value["tool_name"]
        if key == "wrong_type":
            value["original_tokens"] = "100"
        if key == "secret_wrong_type":
            value["original_tokens"] = "sdkgen-secret-marker"
        if key == "null_nonnullable":
            value["hash"] = None
        if key == "unsafe_int":
            value["original_tokens"] = 9007199254740993
        if key == "redirect":
            status = 302
            value = {"detail": "redirect"}
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        if key == "notjson":
            media = "text/html"
            raw = b"<h1>Not JSON</h1>"
        if key == "valid_nonjson":
            media = "text/html"
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Fixture", "true")
        if status == 302:
            self.send_header("Location", "/must-not-follow")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class FixtureServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(
            sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
        ):
            return
        super().handle_error(request, client_address)


def write_tree(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def load_python(root):
    p = root / "python"
    spec = importlib.util.spec_from_file_location(
        "wire_conformance", p / "__init__.py", submodule_search_locations=[str(p)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def python_tests(root, url):
    sdk = load_python(root)
    client = sdk.Client(url)
    count = 0

    def expect_error(fn, error):
        try:
            fn()
        except error as exc:
            return exc
        raise AssertionError(f"expected {error}")

    result = client.retrieve({"hash": "ok", "extra_request": {"snake_case": False}})
    assert result["tool_name"] is None and result["original_content"] == '{"snake_case":"世界"}'
    assert result["future_extension"] == {"snake_case": "unchanged"}
    count += 1
    value = "a/世界 ?#+'!*()"
    assert client.retrieve_get(value)["hash"] == value
    count += 1
    err = expect_error(lambda: client.retrieve({"hash": "missing"}), sdk.APIError)
    assert err.status == 404 and b"Entry missing" in err.body and "Entry missing" not in str(err)
    count += 1
    for key in ("missing_field", "wrong_type", "null_nonnullable", "notjson", "valid_nonjson"):
        expect_error(lambda key=key: client.retrieve({"hash": key}), sdk.ProtocolError)
        count += 1
    assert client.retrieve({"hash": "unsafe_int"})["original_tokens"] == 9007199254740993
    count += 1
    err = expect_error(lambda: client.retrieve({"hash": "redirect"}), sdk.APIError)
    assert err.status == 302
    count += 1
    expect_error(
        lambda: sdk.Client(url, max_response_bytes=16).retrieve({"hash": "ok"}), sdk.ProtocolError
    )
    count += 1
    expect_error(lambda: sdk.Client(url, timeout=0.05).retrieve({"hash": "slow"}), OSError)
    count += 1
    expect_error(lambda: client.retrieve({"hash": "retry_probe_python"}), OSError)
    count += 1
    expect_error(lambda: client.retrieve({}), sdk.ProtocolError)
    count += 1
    expect_error(lambda: client.retrieve_get(".."), ValueError)
    count += 1
    for base in ("file:///tmp", "http://user:pass@localhost", "http://localhost/?secret=1"):
        expect_error(lambda base=base: sdk.Client(base), ValueError)
    count += 1
    print(f"Python: {count} loopback HTTP conformance checks passed", flush=True)


def optional_model_tests(root, env):
    source = """
class Presence(TypedDict):
    required_nullable: str | None
    note: NotRequired[str | None]
    enabled: NotRequired[bool]
    count: NotRequired[int]
    labels: NotRequired[list[str]]
@app.post("/presence")
@sdk_operation(operation_id="check_presence", request=Presence, response=Presence, access="loopback")
async def presence(request: Request): pass
"""
    files = render(Compiler({"presence.py": source}).compile())
    write_tree(root, files)
    (root / "go/presence_test.go").write_text(
        r"""
package headroomwire
import ("encoding/json"; "testing")
func TestPresenceStates(t *testing.T) {
    for _, raw := range []string{`{"required_nullable":null}`, `{"required_nullable":null,"note":null}`, `{"required_nullable":null,"note":""}`, `{"required_nullable":"x","enabled":false,"count":0}`} {
        var value Presence
        if err := json.Unmarshal([]byte(raw), &value); err != nil { t.Fatal(err) }
        encoded, err := json.Marshal(value); if err != nil { t.Fatal(err) }
        var before, after map[string]json.RawMessage
        json.Unmarshal([]byte(raw), &before); json.Unmarshal(encoded, &after)
        if len(before) != len(after) { t.Fatalf("presence changed: %s -> %s", raw, encoded) }
        for key, data := range before { if string(after[key]) != string(data) { t.Fatalf("value changed: %s", key) } }
    }
    for _, raw := range []string{`{}`, `{"required_nullable":null,"enabled":null}`, `{"required_nullable":null,"labels":[null]}`} {
        var value Presence; if json.Unmarshal([]byte(raw), &value) == nil { t.Fatal("invalid presence accepted") }
    }
}
""",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["go", "test", "-v", "./..."], cwd=root / "go", env=env, check=True)
    subprocess.run(
        tool_command(
            "tsc", "-p", str(root / "typescript/tsconfig.json"), "--outDir", str(root / "tsbuild")
        ),
        env=env,
        check=True,
    )
    rust_tests = root / "rust/tests"
    rust_tests.mkdir(parents=True)
    (rust_tests / "presence.rs").write_text(
        r"""
use headroom_generated_pilot::Presence;
use serde_json::{Map, Value};

#[test]
fn presence_states_round_trip_without_collapsing_omission_and_null() {
    for raw in [
        r#"{"required_nullable":null}"#,
        r#"{"required_nullable":null,"note":null}"#,
        r#"{"required_nullable":null,"note":""}"#,
        r#"{"required_nullable":"x","enabled":false,"count":0}"#,
    ] {
        let value: Presence = serde_json::from_str(raw).expect("valid presence state");
        let before: Map<String, Value> = serde_json::from_str(raw).unwrap();
        let after: Map<String, Value> = serde_json::to_value(value).unwrap().as_object().unwrap().clone();
        assert_eq!(before, after);
    }
    for raw in [r#"{}"#, r#"{"required_nullable":null,"enabled":null}"#, r#"{"required_nullable":null,"labels":[null]}"#] {
        assert!(serde_json::from_str::<Presence>(raw).is_err());
    }
}
""",
        encoding="utf-8",
        newline="\n",
    )
    rust_env = {**env, "CARGO_TARGET_DIR": str(root / "rust-target")}
    subprocess.run(
        ["cargo", "test", "--manifest-path", str(root / "rust/Cargo.toml"), "--locked"],
        env=rust_env,
        check=True,
    )
    dotnet_check = root / "dotnet-presence"
    dotnet_check.mkdir()
    (dotnet_check / "PresenceCheck.csproj").write_text(
        f'''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings></PropertyGroup>
  <ItemGroup><ProjectReference Include="{(root / "dotnet/Headroom.GeneratedPilot.csproj").as_posix()}" /></ItemGroup>
</Project>
''',
        encoding="utf-8",
        newline="\n",
    )
    (dotnet_check / "Program.cs").write_text(
        r"""
using System.Text.Json;
using Headroom.GeneratedPilot;

foreach (var raw in new[] {
    "{\"required_nullable\":null}",
    "{\"required_nullable\":null,\"note\":null}",
    "{\"required_nullable\":null,\"note\":\"\"}",
    "{\"required_nullable\":\"x\",\"enabled\":false,\"count\":0}",
}) {
    var value = JsonSerializer.Deserialize<Presence>(raw) ?? throw new Exception("null model");
    value.ValidateModel();
    using var before = JsonDocument.Parse(raw);
    using var after = JsonDocument.Parse(JsonSerializer.Serialize(value));
    if (!JsonElement.DeepEquals(before.RootElement, after.RootElement)) throw new Exception($"presence changed: {raw}");
}
foreach (var raw in new[] { "{}", "{\"required_nullable\":null,\"enabled\":null}", "{\"required_nullable\":null,\"labels\":[null]}" }) {
    try {
        var value = JsonSerializer.Deserialize<Presence>(raw) ?? throw new ProtocolException("null model");
        value.ValidateModel();
        throw new Exception($"invalid presence accepted: {raw}");
    } catch (Exception error) when (error is JsonException or ProtocolException) { }
}
""",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(
        [
            "dotnet",
            "run",
            "--project",
            str(dotnet_check / "PresenceCheck.csproj"),
            "--configuration",
            "Release",
        ],
        env=env,
        check=True,
    )
    print("Optional/nullable generated-model checks passed in Go, Rust, and .NET", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    for binary in ("node", "tsc", "go", "cargo", "dotnet"):
        if not shutil.which(binary):
            raise SystemExit(f"Required conformance tool not available: {binary}")
    config = json.loads((root / "sdk/codegen/config.json").read_text())
    files = render(
        Compiler(
            load_sources(root, config["fixture_sources" if args.fixture else "sources"])
        ).compile()
    )
    FixtureHandler.seen = []
    server = FixtureServer(("127.0.0.1", 0), FixtureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/prefix"
    env = {**os.environ, "HEADROOM_WIRE_TEST_URL": url, "GOTOOLCHAIN": "local", "GOPROXY": "off"}
    try:
        with tempfile.TemporaryDirectory(prefix="headroom-wire-") as td:
            output = Path(td) / "pilot"
            write_tree(output, files)
            python_tests(output, url)
            tsbuild = Path(td) / "typescript-build"
            subprocess.run(
                tool_command(
                    "tsc", "-p", str(output / "typescript/tsconfig.json"), "--outDir", str(tsbuild)
                ),
                env=env,
                check=True,
            )
            (tsbuild / "package.json").write_text(
                '{"type":"module"}\n', encoding="utf-8", newline="\n"
            )
            subprocess.run(
                ["node", str(root / "tests/sdkgen/typescript_wire_test.mjs"), str(tsbuild)],
                env=env,
                check=True,
            )
            shutil.copyfile(root / "tests/sdkgen/go_wire_test.go", output / "go/wire_test.go")
            subprocess.run(["go", "test", "-v", "./..."], cwd=output / "go", env=env, check=True)
            subprocess.run(["go", "vet", "./..."], cwd=output / "go", env=env, check=True)
            rust_tests = output / "rust/tests"
            rust_tests.mkdir(parents=True)
            shutil.copyfile(root / "tests/sdkgen/rust_wire_test.rs", rust_tests / "wire.rs")
            rust_env = {**env, "CARGO_TARGET_DIR": str(Path(td) / "rust-target")}
            subprocess.run(
                ["cargo", "test", "--manifest-path", str(output / "rust/Cargo.toml"), "--locked"],
                env=rust_env,
                check=True,
            )
            dotnet_tests = Path(td) / "dotnet-wire"
            shutil.copytree(root / "tests/sdkgen/dotnet_wire", dotnet_tests)
            project = dotnet_tests / "Headroom.GeneratedPilot.WireTests.csproj"
            project.write_text(
                project.read_text(encoding="utf-8").replace(
                    "../../../sdk/generated-pilot/dotnet/Headroom.GeneratedPilot.csproj",
                    (output / "dotnet/Headroom.GeneratedPilot.csproj").as_posix(),
                ),
                encoding="utf-8",
                newline="\n",
            )
            subprocess.run(
                [
                    "dotnet",
                    "run",
                    "--project",
                    str(project),
                    "--configuration",
                    "Release",
                    "--artifacts-path",
                    str(Path(td) / "dotnet-artifacts"),
                ],
                env=env,
                check=True,
            )
            optional_model_tests(Path(td) / "presence", env)
            assert all("must-not-follow" not in request["path"] for request in FixtureHandler.seen)
            retry_counts = {
                key: sum(request["key"] == key for request in FixtureHandler.seen)
                for key in (
                    "retry_probe_python",
                    "retry_probe_typescript",
                    "retry_probe_go",
                    "retry_probe_rust",
                    "retry_probe_dotnet",
                )
            }
            assert all(count == 1 for count in retry_counts.values()), retry_counts
            print(
                f"PASS: five-language conformance; {len(FixtureHandler.seen)} observed HTTP requests; no provider calls",
                flush=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()

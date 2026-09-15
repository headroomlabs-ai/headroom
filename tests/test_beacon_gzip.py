"""Beacon upload compression, and the fallback that makes it safe to ship.

A schema-v2 event is ~8KB of repetitive JSON and gzips ~6x, which is worth more
than every schema trim available put together. The risk is not the compression;
it is that uploads are fire-and-forget, so an endpoint that cannot read a
gzipped body looks exactly like an endpoint that is working. These tests cover
the compression, and then the three ways it is allowed to stop.

The server here re-implements worker.js's sniff-and-inflate in Python. It is not
the Worker, and cannot prove `DecompressionStream` behaves — but it does pin the
wire contract the two halves have to agree on, which is the part that would
otherwise only be checked in production.
"""

from __future__ import annotations

import gzip
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from headroom.telemetry import session as S


class _Recorder(BaseHTTPRequestHandler):
    """Stands in for worker.js's fetch(): sniff magic bytes, inflate, parse."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        box = self.server.box  # type: ignore[attr-defined]

        if box.get("too_large"):
            self.send_response(413)
            self.end_headers()
            box["over"] += 1
            return
        if box.get("refuse_gzip") and raw[:2] == b"\x1f\x8b":
            # A Worker that predates the inflate path: JSON.parse throws on the
            # gzip bytes and it answers 400.
            self.send_response(400)
            self.end_headers()
            box["refused"] += 1
            return
        if box.get("fail_5xx"):
            self.send_response(503)
            self.end_headers()
            return

        body = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        box["events"].append(json.loads(body))
        box["encodings"].append(self.headers.get("content-encoding"))
        box["wire_bytes"].append(len(raw))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def collector(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    server.box = {  # type: ignore[attr-defined]
        "events": [],
        "encodings": [],
        "wire_bytes": [],
        "refused": 0,
        "over": 0,
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("HEADROOM_TELEMETRY_ENDPOINT", f"http://{host}:{port}/v1/logs")
    monkeypatch.setenv("HEADROOM_BEACON", "on")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    # gzip ships opt-in (_GZIP_DEFAULT is False for the staged rollout), so the
    # tests that exercise the transport have to turn it on explicitly. The
    # default itself is locked by test_gzip_is_opt_in_by_default below, which
    # deliberately does NOT use this fixture's environment.
    monkeypatch.setenv("HEADROOM_BEACON_GZIP", "1")
    # Compression disables itself process-wide on refusal; reset between tests.
    monkeypatch.setattr(S, "_gzip_supported", True)
    yield server.box  # type: ignore[attr-defined]
    server.shutdown()
    server.server_close()


class Outcome:
    provider = "anthropic"
    model = "gpt-4o"
    original_tokens = 120_000
    attempted_input_tokens = 40_000
    optimized_tokens = 90_000
    output_tokens = 1_200
    tokens_saved = 30_000
    cache_read_tokens = 60_000
    status_code = 200
    total_latency_ms = 4_200.0
    overhead_ms = 48.0
    ttfb_ms = 900.0
    num_messages = 140
    client = "claude-code"
    transforms_applied: tuple[str, ...] = ("crush",)
    tags: dict[str, Any] = {}
    waste_signals = {"reread": 900, "reread_compressed": 400}


def sample_payload() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append)
    for index in range(40):
        agg.record(Outcome(), now=1000.0 + index * 45)
    agg.flush_all()
    return rows[-1]


def test_payload_survives_the_round_trip_byte_for_byte(collector):
    """Compression must be invisible above the transport: what the collector
    parses has to be exactly what the aggregator built."""
    payload = sample_payload()
    S._post_blocking(payload)
    assert len(collector["events"]) == 1
    received = collector["events"][0]
    expected = S.build_otlp_logs(payload, S.resource_attributes())
    body = received["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["body"]
    want = expected["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["body"]
    assert body == want, "the payload changed in transit"


def test_it_is_actually_compressed_and_declared(collector):
    S._post_blocking(sample_payload())
    assert collector["encodings"] == ["gzip"], "Content-Encoding was not declared"
    plain = len(
        json.dumps(
            S.build_otlp_logs(sample_payload(), S.resource_attributes()), separators=(",", ":")
        ).encode()
    )
    sent = collector["wire_bytes"][0]
    assert sent < plain / 3, f"only compressed {plain / sent:.1f}x ({sent} vs {plain} B)"


def test_a_worker_that_cannot_inflate_costs_one_retry_not_the_data(collector):
    """The rollout-order safety net. Shipping this client against a Worker that
    predates the inflate path must not lose events — it must notice, fall back,
    and stop compressing."""
    collector["refuse_gzip"] = True
    S._post_blocking(sample_payload())
    assert collector["refused"] == 1, "the compressed attempt never happened"
    assert len(collector["events"]) == 1, "the event was lost instead of retried"
    assert collector["encodings"] == [None], "the retry was not uncompressed"

    # ...and it must not keep paying that retry for the rest of the process.
    S._post_blocking(sample_payload())
    assert collector["refused"] == 1, "it tried gzip again after being refused"
    assert len(collector["events"]) == 2
    assert collector["encodings"] == [None, None]


def test_a_5xx_does_not_disable_compression(collector):
    """A sick endpoint says nothing about whether it understands gzip. Treating
    a 503 as "no gzip here" would permanently give up the saving on the first
    upstream blip."""
    collector["fail_5xx"] = True
    S._post_blocking(sample_payload())
    assert S._gzip_enabled() is True

    collector["fail_5xx"] = False
    S._post_blocking(sample_payload())
    assert collector["encodings"] == ["gzip"]


def test_the_kill_switch_works(collector, monkeypatch):
    monkeypatch.setenv("HEADROOM_BEACON_GZIP", "off")
    S._post_blocking(sample_payload())
    assert collector["encodings"] == [None]
    assert len(collector["events"]) == 1


def test_a_dead_endpoint_still_never_raises(monkeypatch):
    monkeypatch.setenv("HEADROOM_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/v1/logs")
    monkeypatch.setenv("HEADROOM_BEACON", "on")
    S._post_blocking(sample_payload(), timeout=0.25)


def test_the_wire_contract_worker_js_must_satisfy():
    """What worker.js sniffs for. If this changes, the Worker's `bytes[0] ===
    0x1f && bytes[1] === 0x8b` check stops matching and every upload 400s."""
    body = json.dumps(
        S.build_otlp_logs(sample_payload(), S.resource_attributes()), separators=(",", ":")
    ).encode()
    compressed = gzip.compress(body, S._GZIP_LEVEL)
    assert compressed[0] == 0x1F and compressed[1] == 0x8B, "not gzip magic"
    assert gzip.decompress(compressed) == body, "not standard gzip"
    # And the Worker's arriving-body cap must still be comfortable.
    assert len(compressed) < 64 * 1024


def test_a_413_must_not_trigger_the_uncompressed_retry(collector):
    """413 means the body was too big. Answering that by re-sending the SAME
    payload uncompressed — six times larger — is guaranteed to 413 again, and
    would permanently switch off the compression that was the only thing
    keeping the request under the cap. Only "I cannot read this encoding"
    (400/415) may fall back."""
    collector["too_large"] = True
    S._post_blocking(sample_payload())
    assert collector["over"] == 1, "it should not have retried at all"
    assert S._gzip_enabled() is True, "a 413 disabled compression"


def test_415_does_fall_back(collector, monkeypatch):
    """The other honest 'I cannot read this body' answer, from a collector that
    rejects the encoding outright rather than failing to parse it."""

    class Refuse415(_Recorder):
        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            box = self.server.box
            if raw[:2] == b"\x1f\x8b":
                self.send_response(415)
                self.end_headers()
                box["refused"] += 1
                return
            box["events"].append(json.loads(raw))
            box["encodings"].append(self.headers.get("content-encoding"))
            self.send_response(204)
            self.end_headers()

    collector_server = HTTPServer(("127.0.0.1", 0), Refuse415)
    collector_server.box = {"events": [], "encodings": [], "refused": 0}
    threading.Thread(target=collector_server.serve_forever, daemon=True).start()
    host, port = collector_server.server_address
    monkeypatch.setenv("HEADROOM_TELEMETRY_ENDPOINT", f"http://{host}:{port}/v1/logs")
    try:
        S._post_blocking(sample_payload())
        assert collector_server.box["refused"] == 1
        assert collector_server.box["encodings"] == [None], "no uncompressed retry"
        assert S._gzip_enabled() is False
    finally:
        collector_server.shutdown()
        collector_server.server_close()


def test_the_gzip_gate_never_raises(monkeypatch):
    """`_gzip_enabled` is called before _post_blocking's try, from a bare
    daemon-thread target and from an atexit handler — both places where an
    escaping exception reaches the user's terminal."""
    import headroom.telemetry.beacon as beacon_module

    monkeypatch.delattr(beacon_module, "_OFF_VALUES", raising=False)
    assert S._gzip_enabled() is False, "it must degrade, not raise"
    # ...and the upload path still completes.
    monkeypatch.setenv("HEADROOM_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/v1/logs")
    S._post_blocking(sample_payload(), timeout=0.25)


def test_gzip_is_opt_in_by_default(monkeypatch):
    """The staged rollout: schema v2 ships without the new transport.

    Locks the default so enabling gzip is a deliberate edit to `_GZIP_DEFAULT`
    (or an operator setting the variable), never a side effect of touching this
    module. While it holds, the Worker's `inflate()` is unreachable -- the
    receiver sniffs magic bytes, and nothing produces them.
    """
    monkeypatch.delenv("HEADROOM_BEACON_GZIP", raising=False)
    monkeypatch.setattr(S, "_gzip_supported", True)
    assert S._GZIP_DEFAULT is False, "gzip must stay staged until v2 is proven"
    assert S._gzip_enabled() is False, "an unset variable must not compress"


def test_an_unrecognised_gzip_value_does_not_enable_it(monkeypatch):
    """A typo degrades to the proven transport, not to the new one."""
    monkeypatch.setattr(S, "_gzip_supported", True)
    for value in ("yess", "maybe", "2", " "):
        monkeypatch.setenv("HEADROOM_BEACON_GZIP", value)
        assert S._gzip_enabled() is False, f"{value!r} enabled compression"


def test_the_operator_can_opt_in(monkeypatch):
    """...and every documented on-value works, so the opt-in is discoverable."""
    monkeypatch.setattr(S, "_gzip_supported", True)
    for value in ("1", "on", "true", "yes", "enable", "enabled", "ON", " 1 "):
        monkeypatch.setenv("HEADROOM_BEACON_GZIP", value)
        assert S._gzip_enabled() is True, f"{value!r} did not enable compression"


# --------------------------------------------------------------- wire budget --
#
# The receiver answers an oversized body with 413, and 413 is deliberately not a
# reason to retry, so an event that exceeds the cap is lost outright -- v1
# counters included. These cover the client bounding itself instead.

_CONTENT = [f"ct{i:02d}" for i in range(10)]
_STRATEGIES = [f"st{i:02d}" for i in range(13)]


def _saturated(n_shape: int = 160, n_tool: int = 160, turns: int = 200) -> S._Session:
    """A session whose shape tables are at their caps."""
    sess = S._Session(sid="a" * 32, started=0.0, last_seen=0.0)
    sess.turns = turns
    sess.tokens_saved = 987_654
    combos = list(itertools.product(_CONTENT, _STRATEGIES))[:n_shape]
    for i, (content, strategy) in enumerate(combos):
        sess.shapes[(content, strategy)] = [i + 1, 222_222, 33_333]
    for i in range(n_tool):
        sess.tool_shapes[f"tool_shape_{i:03d}|d{i % 6}|w{i % 8}"] = [i + 1, 222_222, 33_333]
    sess.models.add("claude_sonnet_5")
    sess.providers.add("anthropic")
    sess.clients["claude_code"] = turns
    return sess


def _resource() -> dict[str, str]:
    res = S.resource_attributes(create_install_id=False)
    res.setdefault("headroom.install_id", "0" * 32)
    return res


def _body_of(raw: bytes) -> dict[str, Any]:
    """Unwrap one OTLP body back to plain JSON -- the inverse of _any_value."""

    def unwrap(value: dict[str, Any]) -> Any:
        if "kvlistValue" in value:
            return {kv["key"]: unwrap(kv["value"]) for kv in value["kvlistValue"]["values"]}
        if "arrayValue" in value:
            return [unwrap(v) for v in value["arrayValue"].get("values", [])]
        if "intValue" in value:
            return int(value["intValue"])
        for key in ("stringValue", "boolValue", "doubleValue"):
            if key in value:
                return value[key]
        return None

    record = json.loads(raw)["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    return unwrap(record["body"])


def test_an_oversized_plain_body_is_trimmed_to_fit():
    """Uncompressed schema v2 crosses 64KB at ~116 rows per table; caps are higher.

    Without this the busiest sessions -- the most informative ones -- would be
    exactly the ones the receiver refuses.
    """
    payload = _saturated().payload("heartbeat")
    raw = S._fit_to_wire(payload, _resource(), compress=False)
    assert len(raw) <= 64 * 1024, f"{len(raw)} B exceeds the 64KB receiver cap"
    assert _body_of(raw)["shapes"]["truncated"] is True


def test_trimming_never_costs_a_v1_counter():
    """The whole point of shedding shapes is that nothing else is shed."""
    payload = _saturated().payload("heartbeat")
    expected = payload["tokens"]["saved"]
    body = _body_of(S._fit_to_wire(payload, _resource(), compress=False))
    assert body["shapes"]["truncated"] is True, "nothing was shed; test proves nothing"
    assert body["tokens"]["saved"] == expected
    assert body["session"]["turns"] == 200


def test_the_trim_keeps_the_rows_carrying_the_most_evidence():
    """Rows are ranked by `n`, so a trimmed table loses resolution, not signal."""
    payload = _saturated().payload("heartbeat")
    kept = _body_of(S._fit_to_wire(payload, _resource(), compress=False))["shapes"]
    counts = [row["n"] for row in kept["by_content"]]
    assert counts, "everything was dropped"
    # 130 (content x strategy) rows carry n = 1..130. Keeping the top means the
    # thinnest survivor still beats what was dropped.
    assert min(counts) > 1, "the trim kept the least-used rows"


def test_a_trimmed_table_keeps_the_canonical_row_order():
    """Ranking is a selection rule, not a wire format."""
    payload = _saturated().payload("heartbeat")
    kept = _body_of(S._fit_to_wire(payload, _resource(), compress=False))["shapes"]
    assert kept["truncated"] is True, "nothing was shed; test proves nothing"
    by_content = [(r["content"], r["strategy"]) for r in kept["by_content"]]
    assert by_content == sorted(by_content), "row order depended on trimming"


def test_truncated_is_always_present_even_when_nothing_is_shed():
    """An optional key would give `shapes` two STRUCT layouts in the corpus."""
    payload = _saturated(n_shape=2, n_tool=2).payload("heartbeat")
    raw = S._fit_to_wire(payload, _resource(), compress=False)
    shapes = _body_of(raw)["shapes"]
    assert shapes["truncated"] is False
    assert len(shapes["by_content"]) == 2, "a body under budget was trimmed anyway"


def test_compression_makes_the_budget_unreachable():
    """Self-cancelling: once gzip ships, a saturated payload is ~4KB."""
    payload = _saturated().payload("heartbeat")
    raw = S._fit_to_wire(payload, _resource(), compress=True)
    shapes = _body_of(raw)["shapes"]
    assert shapes["truncated"] is False, "gzip should leave the tables intact"
    assert len(gzip.compress(raw, S._GZIP_LEVEL)) <= S._MAX_WIRE_BYTES


def test_the_gzip_fallback_rebudgets_before_resending(collector, monkeypatch):
    """The refusal path resends plain -- and plain is ~6x larger.

    Without re-measuring, the fallback added to SAVE an event from a Worker that
    cannot inflate would hand that Worker a body over its cap instead, turning a
    400 into a 413 and losing the event after all.
    """
    monkeypatch.setenv("HEADROOM_BEACON_GZIP", "1")
    monkeypatch.setattr(S, "_gzip_supported", True)
    collector["refuse_gzip"] = True
    S._post_blocking(_saturated().payload("heartbeat"), timeout=5.0)
    assert collector["refused"] == 1, "gzip was not attempted first"
    assert collector["events"], "the fallback never arrived"
    # 64KB is the cap on the OLDEST receiver still in service -- a literal on
    # purpose. Asserting against _MAX_WIRE_BYTES would restate the code under
    # test, and would keep passing if that budget were ever raised past what a
    # deployed Worker accepts.
    assert collector["wire_bytes"][-1] <= 64 * 1024, (
        f"fell back with {collector['wire_bytes'][-1]} B, over the 64KB receiver cap"
    )
    # The collector keeps the OTLP envelope; unwrap it back to the payload.
    sent = _body_of(json.dumps(collector["events"][-1]).encode())
    assert sent["tokens"]["saved"], "the fallback lost the v1 counters"
    assert sent["shapes"]["truncated"] is True, "the plain resend was not re-budgeted"

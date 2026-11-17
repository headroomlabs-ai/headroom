"""Tests for headroom.proxy_client_liveness -- extracted from wrap.py so both
the CLI and the proxy (workspace_registry.py) share one liveness/identity
source of truth rather than two independently-decaying ones.
"""

from __future__ import annotations

import json
import os

from headroom import proxy_client_liveness as liveness


def test_pid_alive_true_for_current_process():
    assert liveness.pid_alive(os.getpid()) is True


def test_pid_alive_false_for_pid_zero_or_negative():
    assert liveness.pid_alive(0) is False
    assert liveness.pid_alive(-1) is False


def test_identity_mismatch_conservative_false_for_legacy_record():
    """No start_src/start_time recorded (pre-identity marker format) -- must
    not claim a mismatch without proof."""
    assert liveness.identity_mismatch(None, None, os.getpid()) is False


def test_identity_mismatch_conservative_false_for_wrong_types():
    assert liveness.identity_mismatch(123, "not-a-number", os.getpid()) is False
    assert liveness.identity_mismatch("psutil", "not-a-number", os.getpid()) is False


def test_identity_mismatch_false_when_identity_matches_self():
    ident = liveness.proc_identity(os.getpid())
    if ident is None:
        return  # platform can't determine identity -- nothing to assert
    src, start_time = ident
    assert liveness.identity_mismatch(src, start_time, os.getpid()) is False


def test_marker_pid_reused_false_for_malformed_marker(tmp_path):
    marker = tmp_path / "123.json"
    marker.write_text("{not valid json", encoding="utf-8")
    assert liveness.marker_pid_reused(marker, os.getpid()) is False


def test_marker_pid_reused_false_for_missing_marker(tmp_path):
    marker = tmp_path / "does-not-exist.json"
    assert liveness.marker_pid_reused(marker, os.getpid()) is False


def test_marker_pid_reused_false_for_non_object_json(tmp_path):
    """Valid JSON that isn't an object (e.g. a bare list) must not crash via
    AttributeError on .get() -- same bug class as the malformed-syntax case,
    just a step further past json.loads succeeding."""
    marker = tmp_path / "123.json"
    marker.write_text("[1, 2, 3]", encoding="utf-8")
    assert liveness.marker_pid_reused(marker, os.getpid()) is False


def test_marker_pid_reused_false_when_recorded_identity_matches_self(tmp_path):
    ident = liveness.proc_identity(os.getpid())
    if ident is None:
        return
    src, start_time = ident
    marker = tmp_path / f"{os.getpid()}.json"
    marker.write_text(
        json.dumps({"pid": os.getpid(), "start_src": src, "start_time": start_time}),
        encoding="utf-8",
    )
    assert liveness.marker_pid_reused(marker, os.getpid()) is False

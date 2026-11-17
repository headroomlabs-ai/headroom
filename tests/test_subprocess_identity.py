"""Tests for headroom._subprocess proc_identity / identity_mismatch.

proc_identity has platform-exclusive legs (psutil vs /proc), so no single CI
runner reaches both; these tests drive each leg explicitly with fakes.
"""

from __future__ import annotations

import builtins
import io
import os
import sys
from types import SimpleNamespace

import pytest

from headroom import _subprocess as sub


class TestProcIdentity:
    def test_psutil_leg_with_stub(self, monkeypatch) -> None:
        # Deterministic: psutil is an optional dep, so drive the leg with a stub
        # instead of importorskip (line must be covered on psutil-less CI too).
        stub = SimpleNamespace(Process=lambda pid: SimpleNamespace(create_time=lambda: 1234.5))
        monkeypatch.setitem(sys.modules, "psutil", stub)
        assert sub.proc_identity(999) == ("psutil", 1234.5)

    def test_psutil_leg_real_when_installed(self) -> None:
        pytest.importorskip("psutil")
        ident = sub.proc_identity(os.getpid())
        assert ident is not None
        src, start = ident
        assert src == "psutil"
        assert isinstance(start, float) and start > 0

    def test_proc_leg_when_psutil_unavailable(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "psutil", None)  # import raises ImportError
        # /proc/<pid>/stat: fields after the final ")" are field 3 onwards, so
        # field 22 (starttime) is index 19 of the split — here 987654.
        stat = b"123 (python (test)) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 987654 21"
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                return io.BytesIO(stat)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert sub.proc_identity(123) == ("proc", 987654.0)

    def test_returns_none_when_both_sources_fail(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "psutil", None)
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise OSError("no /proc here")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)
        assert sub.proc_identity(123) is None


class TestIdentityMismatch:
    def test_legacy_record_never_mismatches(self) -> None:
        assert sub.identity_mismatch(None, None, os.getpid()) is False
        assert sub.identity_mismatch("psutil", "not-a-number", os.getpid()) is False

    def test_unknown_identity_never_mismatches(self) -> None:
        assert sub.identity_mismatch("psutil", 1.0, 123, identity_fn=lambda pid: None) is False

    def test_source_mismatch_never_mismatches(self) -> None:
        ident = lambda pid: ("proc", 1.0)  # noqa: E731
        assert sub.identity_mismatch("psutil", 1.0, 123, identity_fn=ident) is False

    def test_same_start_time_is_not_a_mismatch(self) -> None:
        ident = lambda pid: ("psutil", 1000.5)  # noqa: E731
        assert sub.identity_mismatch("psutil", 1000.2, 123, identity_fn=ident) is False

    def test_distant_start_time_proves_recycling(self) -> None:
        ident = lambda pid: ("psutil", 2000.0)  # noqa: E731
        assert sub.identity_mismatch("psutil", 1000.0, 123, identity_fn=ident) is True

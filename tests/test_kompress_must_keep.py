"""Tests for the must-keep token override in kompress_compressor."""
from __future__ import annotations

import os
import re

import pytest

from headroom.transforms.kompress_compressor import (
    _KOMPRESS_MUST_KEEP_ENV,
    _KOMPRESS_MUST_KEEP_RE,
)


class TestMustKeepRegex:
    def test_numbers(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("42")
        assert _KOMPRESS_MUST_KEEP_RE.search("3.14")
        assert _KOMPRESS_MUST_KEEP_RE.search("0x7fff2038")

    def test_allcaps(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("SIGILL")
        assert _KOMPRESS_MUST_KEEP_RE.search("HTTP")
        assert _KOMPRESS_MUST_KEEP_RE.search("EOF")

    def test_dotted_paths(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("libsystem_kernel.dylib")
        assert _KOMPRESS_MUST_KEEP_RE.search("torch.nn")

    def test_unix_paths(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("/usr/lib/python3")
        assert _KOMPRESS_MUST_KEEP_RE.search("/workspace/ultrawhale")

    def test_extensions(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("model.py")
        assert _KOMPRESS_MUST_KEEP_RE.search("weights.so")

    def test_flags(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("--verbose")
        assert _KOMPRESS_MUST_KEEP_RE.search("-n")

    def test_camelcase(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("IndexError")
        assert _KOMPRESS_MUST_KEEP_RE.search("EXC_BAD_INSTRUCTION")

    def test_plain_words_not_matched(self):
        assert not _KOMPRESS_MUST_KEEP_RE.search("the")
        assert not _KOMPRESS_MUST_KEEP_RE.search("process")
        assert not _KOMPRESS_MUST_KEEP_RE.search("raised")


class TestMustKeepEnvVar:
    def test_env_var_name(self):
        assert _KOMPRESS_MUST_KEEP_ENV == "HEADROOM_KOMPRESS_MUST_KEEP"

    def test_env_var_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        assert os.environ.get(_KOMPRESS_MUST_KEEP_ENV, "1") != "0"

    def test_env_var_can_disable(self, monkeypatch):
        monkeypatch.setenv(_KOMPRESS_MUST_KEEP_ENV, "0")
        assert os.environ.get(_KOMPRESS_MUST_KEEP_ENV, "1") == "0"

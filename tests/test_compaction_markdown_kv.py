"""Markdown-KV compaction formatter — opt-in serialization-aware output.

Covers the plumbing added for issue #858:

- ``headroom._core.SmartCrusher.with_compaction_format`` renders uniform
  arrays as Markdown-KV when the lossless gate passes.
- The high-level ``SmartCrusher`` exposes the knob via the
  ``compaction_format`` kwarg and the ``HEADROOM_COMPACTION_FORMAT`` env
  var, defaulting to the unchanged ``csv-schema`` path.
- Unknown format names fail loudly (``ValueError``) instead of silently
  falling back.

The formatter's rendering rules themselves (missing-cell omission,
string quoting, CCR marker contract, buckets) are covered by the Rust
unit tests in ``compaction/formatter.rs``.
"""

from __future__ import annotations

import json

import pytest

from headroom._core import SmartCrusher as RustSmartCrusher
from headroom._core import SmartCrusherConfig as RustSmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher
from headroom.transforms.smart_crusher import SmartCrusherConfig as PySmartCrusherConfig


def _tabular_json(n: int = 50) -> str:
    return json.dumps(
        [
            {
                "id": i,
                "name": f"user_{i}",
                "email": f"user_{i}@example.com",
                "status": "ok" if i % 3 == 0 else "pending",
            }
            for i in range(n)
        ]
    )


# ── Rust bridge: with_compaction_format ──


def test_markdown_kv_renders_kv_lines() -> None:
    # Lower the lossless gate: Markdown-KV repeats field names per row,
    # so its savings vs raw JSON are real but below the 30% default.
    cfg = RustSmartCrusherConfig(lossless_min_savings_ratio=0.01)
    crusher = RustSmartCrusher.with_compaction_format(cfg, "markdown-kv")
    result = crusher.crush(_tabular_json(), "", 1.0)
    assert result.was_modified
    assert "lossless" in result.strategy
    # Columns are schema-sorted, so `email` opens each row and `id`
    # renders as a continuation line.
    assert "- email: user_0@example.com" in result.compressed
    assert "id: 0" in result.compressed
    assert "name: user_1" in result.compressed
    # Declaration line survives (shared with the CSV formatter).
    assert "[50]{" in result.compressed


def test_csv_schema_format_name_matches_default() -> None:
    cfg = RustSmartCrusherConfig(lossless_min_savings_ratio=0.01)
    via_name = RustSmartCrusher.with_compaction_format(cfg, "csv-schema")
    via_default = RustSmartCrusher(cfg)
    content = _tabular_json()
    assert (
        via_name.crush(content, "", 1.0).compressed
        == via_default.crush(content, "", 1.0).compressed
    )


def test_unknown_format_name_raises() -> None:
    with pytest.raises(ValueError, match="markdown-kv"):
        RustSmartCrusher.with_compaction_format(None, "toon")


# ── High-level SmartCrusher knob ──


def test_default_format_is_csv_schema() -> None:
    crusher = SmartCrusher()
    assert crusher._compaction_format == "csv-schema"


def test_kwarg_opts_into_markdown_kv() -> None:
    crusher = SmartCrusher(compaction_format="markdown-kv")
    assert crusher._compaction_format == "markdown-kv"


def test_env_var_opts_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPACTION_FORMAT", "markdown-kv")
    crusher = SmartCrusher()
    assert crusher._compaction_format == "markdown-kv"


def test_kwarg_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPACTION_FORMAT", "json")
    crusher = SmartCrusher(compaction_format="markdown-kv")
    assert crusher._compaction_format == "markdown-kv"


def test_unknown_format_kwarg_raises() -> None:
    with pytest.raises(ValueError):
        SmartCrusher(compaction_format="bogus")


def test_without_compaction_ignores_format() -> None:
    crusher = SmartCrusher(with_compaction=False, compaction_format="markdown-kv")
    assert crusher._compaction_format is None


def test_without_compaction_still_validates_format() -> None:
    # An explicit bogus format is a misconfiguration even when the knob
    # is ignored on this path — fail loudly, don't silently accept.
    with pytest.raises(ValueError, match="bogus"):
        SmartCrusher(with_compaction=False, compaction_format="bogus")


def test_end_to_end_crush_emits_markdown_kv() -> None:
    # Through the high-level Python SmartCrusher, not the Rust bridge:
    # proves the kwarg changes crush() output, not just the stored
    # attribute. Same lowered gate as the bridge test — KV's savings on
    # minified JSON sit below the 30% default.
    config = PySmartCrusherConfig(lossless_min_savings_ratio=0.01)
    crusher = SmartCrusher(config=config, compaction_format="markdown-kv")
    result = crusher.crush(_tabular_json())
    assert result.was_modified
    assert "- email: user_0@example.com" in result.compressed
    assert "[50]{" in result.compressed


def test_default_output_unchanged_by_feature() -> None:
    # The default constructor path must stay byte-identical to an
    # explicit csv-schema opt-in — proves the gate is truly default-off.
    content = _tabular_json()
    default_out = SmartCrusher().crush(content)
    explicit_out = SmartCrusher(compaction_format="csv-schema").crush(content)
    assert default_out.compressed == explicit_out.compressed

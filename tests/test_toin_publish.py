"""PR-B5 acceptance tests for ``headroom.cli.toin_publish``.

Pins:

1. ``publish()`` writes a TOML file the stdlib ``tomllib`` can parse.
2. Slices below ``--min-observations`` are filtered out.
3. Rows include ``auth_mode``, ``model_family``, ``structure_hash``,
   ``strategy_hint``, ``confidence``, ``observations`` — the schema
   ``crates/headroom-core/src/transforms/recommendations.rs`` consumes.
4. The CLI entry point honors ``--output`` / ``--min-observations``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Python 3.11+ has tomllib in stdlib; otherwise tomli is shipped as a
# dependency by the project's pyproject.toml.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only hit on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from headroom.cli.toin_publish import main as publish_main
from headroom.cli.toin_publish import publish
from headroom.telemetry import (
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolSignature,
)


def _record(
    toin: ToolIntelligenceNetwork,
    *,
    items: list[dict[str, object]],
    n: int,
    auth_mode: str,
    model_family: str,
    strategy: str = "smart_crusher",
) -> ToolSignature:
    """Drive ``record_compression`` ``n`` times for the given slice."""
    sig = ToolSignature.from_items(items)
    for _ in range(n):
        toin.record_compression(
            tool_signature=sig,
            original_count=len(items),
            compressed_count=max(1, len(items) // 2),
            original_tokens=1000,
            compressed_tokens=500,
            strategy=strategy,
            auth_mode=auth_mode,
            model_family=model_family,
        )
    return sig


@pytest.fixture
def fresh_toin(tmp_path: Path) -> ToolIntelligenceNetwork:
    """Isolated TOIN handle so tests don't see each other's state."""
    return ToolIntelligenceNetwork(
        TOINConfig(
            storage_path=str(tmp_path / "toin_publish.json"),
            auto_save_interval=0,
        )
    )


def test_publish_command_writes_toml(fresh_toin: ToolIntelligenceNetwork, tmp_path: Path) -> None:
    """publish() emits a parseable TOML file with the expected schema."""
    items = [{"id": i, "status": "ok"} for i in range(20)]
    sig = _record(
        fresh_toin,
        items=items,
        n=60,
        auth_mode="payg",
        model_family="claude-3-5",
    )

    output = tmp_path / "recommendations.toml"
    rows_written = publish(
        output_path=output,
        min_observations=50,
        toin=fresh_toin,
    )
    assert rows_written == 1

    parsed = tomllib.loads(output.read_text(encoding="utf-8"))
    assert "recommendation" in parsed
    rec_list = parsed["recommendation"]
    assert isinstance(rec_list, list)
    assert len(rec_list) == 1

    row = rec_list[0]
    assert set(row.keys()) == {
        "auth_mode",
        "model_family",
        "structure_hash",
        "strategy_hint",
        "confidence",
        "observations",
    }
    assert row["auth_mode"] == "payg"
    assert row["model_family"] == "claude-3-5"
    assert row["structure_hash"] == sig.structure_hash
    assert row["strategy_hint"] == "smart_crusher"
    assert isinstance(row["confidence"], float)
    assert 0.0 <= row["confidence"] <= 1.0
    assert row["observations"] == 60


def test_publish_filters_below_min_observations(
    fresh_toin: ToolIntelligenceNetwork,
    tmp_path: Path,
) -> None:
    """Slices below the observation floor are dropped from the TOML."""
    eligible = [{"id": i} for i in range(10)]
    rare = [{"name": str(i)} for i in range(10)]

    _record(fresh_toin, items=eligible, n=60, auth_mode="payg", model_family="claude-3-5")
    _record(fresh_toin, items=rare, n=10, auth_mode="payg", model_family="claude-3-5")

    output = tmp_path / "recs.toml"
    rows_written = publish(output_path=output, min_observations=50, toin=fresh_toin)
    assert rows_written == 1

    parsed = tomllib.loads(output.read_text(encoding="utf-8"))
    rec_list = parsed["recommendation"]
    assert len(rec_list) == 1
    # The eligible signature wins; the rare one is filtered.
    assert rec_list[0]["observations"] == 60


def test_publish_emits_one_row_per_tenant_slice(
    fresh_toin: ToolIntelligenceNetwork, tmp_path: Path
) -> None:
    """Same tool-signature, different (auth_mode, model_family) ⇒ separate rows."""
    items = [{"id": i, "status": "ok"} for i in range(15)]
    _record(fresh_toin, items=items, n=60, auth_mode="payg", model_family="claude-3-5")
    _record(fresh_toin, items=items, n=60, auth_mode="oauth", model_family="claude-3-5")
    _record(fresh_toin, items=items, n=60, auth_mode="payg", model_family="gpt-4o")

    output = tmp_path / "recs.toml"
    rows_written = publish(output_path=output, min_observations=50, toin=fresh_toin)
    assert rows_written == 3

    parsed = tomllib.loads(output.read_text(encoding="utf-8"))
    rec_list = parsed["recommendation"]
    keys = sorted((r["auth_mode"], r["model_family"]) for r in rec_list)
    assert keys == [("oauth", "claude-3-5"), ("payg", "claude-3-5"), ("payg", "gpt-4o")]


def test_publish_writes_empty_file_with_no_eligible_rows(
    fresh_toin: ToolIntelligenceNetwork, tmp_path: Path
) -> None:
    """No qualifying patterns ⇒ valid empty TOML, not an exception."""
    output = tmp_path / "recs.toml"
    rows_written = publish(output_path=output, min_observations=50, toin=fresh_toin)
    assert rows_written == 0

    body = output.read_text(encoding="utf-8")
    parsed = tomllib.loads(body)
    assert parsed == {}
    # Header still shipped so ops can identify the file.
    assert body.startswith("# Auto-generated")


def test_publish_rows_are_deterministically_sorted(
    fresh_toin: ToolIntelligenceNetwork, tmp_path: Path
) -> None:
    """Rows sort by (auth_mode, model_family, structure_hash) for clean diffs.

    Use *structurally distinct* tool signatures so the hashes truly
    differ — `ToolSignature` keys off field names + types, not values.
    """
    one_field = [{"id": i} for i in range(8)]
    two_fields = [{"id": i, "code": 200 + i} for i in range(8)]

    _record(fresh_toin, items=one_field, n=60, auth_mode="payg", model_family="claude-3-5")
    _record(fresh_toin, items=two_fields, n=60, auth_mode="payg", model_family="claude-3-5")
    _record(fresh_toin, items=one_field, n=60, auth_mode="oauth", model_family="gpt-4o")

    output = tmp_path / "recs.toml"
    publish(output_path=output, min_observations=50, toin=fresh_toin)
    parsed = tomllib.loads(output.read_text(encoding="utf-8"))
    rec_list = parsed["recommendation"]
    # First sort key: auth_mode (oauth < payg).
    assert [r["auth_mode"] for r in rec_list] == ["oauth", "payg", "payg"]
    # And within payg, structure_hash sorts asc.
    payg_rows = [r for r in rec_list if r["auth_mode"] == "payg"]
    assert payg_rows == sorted(payg_rows, key=lambda r: r["structure_hash"])


def test_cli_entrypoint_writes_to_output_arg(tmp_path: Path, monkeypatch) -> None:
    """`python -m headroom.cli.toin_publish --output X --min-observations N`."""
    storage = tmp_path / "toin.json"
    monkeypatch.setenv("HEADROOM_TOIN_PATH", str(storage))

    # Prime the global TOIN singleton with eligible data.
    from headroom.telemetry.toin import get_toin, reset_toin

    reset_toin()
    try:
        toin = get_toin()
        _record(
            toin,
            items=[{"id": i} for i in range(10)],
            n=55,
            auth_mode="payg",
            model_family="claude-3-5",
        )
        toin.save()

        output = tmp_path / "out.toml"
        rc = publish_main(
            ["--output", str(output), "--min-observations", "50"],
        )
        assert rc == 0
        assert output.exists()
        parsed = tomllib.loads(output.read_text(encoding="utf-8"))
        assert len(parsed.get("recommendation", [])) == 1
    finally:
        reset_toin()


def test_cli_rejects_non_positive_min_observations(tmp_path: Path) -> None:
    """`--min-observations 0` is a CLI-level error."""
    output = tmp_path / "out.toml"
    with pytest.raises(SystemExit) as exc_info:
        publish_main(["--output", str(output), "--min-observations", "0"])
    assert exc_info.value.code != 0

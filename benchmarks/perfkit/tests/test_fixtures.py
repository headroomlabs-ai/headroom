"""Unit tests for :mod:`perfkit.fixtures` (T012).

Covers content-hash determinism, drift detection, and the immutability
policy required by ``contracts/mock-upstream-contract.md`` §4-§6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.perfkit.fixtures import (
    FixtureDriftError,
    canonical_bytes,
    fixture_hash,
    iter_fixtures,
    load_fixture,
    write_fixture_immutable,
)


def test_canonical_bytes_order_independent() -> None:
    a = {"a": 1, "b": [1, 2], "c": {"x": "y"}}
    b = {"c": {"x": "y"}, "b": [1, 2], "a": 1}
    assert canonical_bytes(a) == canonical_bytes(b)


def test_fixture_hash_is_stable_across_dict_order() -> None:
    a = {"alpha": 1, "beta": {"nested": "value"}, "list": [1, 2, 3]}
    b = {"list": [1, 2, 3], "beta": {"nested": "value"}, "alpha": 1}
    assert fixture_hash(a) == fixture_hash(b)


def test_fixture_hash_changes_when_content_changes() -> None:
    assert fixture_hash({"x": 1}) != fixture_hash({"x": 2})


def test_load_fixture_with_expected_hash_raises_on_drift(tmp_path: Path) -> None:
    p = tmp_path / "fx.json"
    p.write_text("""{"x": 1}""")
    # Pretend an older fixture_hash was recorded; the on-disk content now
    # disagrees with it, so we must trip the drift guard.
    with pytest.raises(FixtureDriftError):
        load_fixture(p, expected_hash="0" * 64)


def test_load_fixture_without_expected_hash_loads(tmp_path: Path) -> None:
    p = tmp_path / "fx.json"
    p.write_text("""{"x": 1}""")
    fx = load_fixture(p)
    assert fx.data == {"x": 1}
    assert len(fx.fixture_hash) == 64


def test_write_fixture_immutable_rejects_rewrite_with_new_content(tmp_path: Path) -> None:
    p = tmp_path / "fx.json"
    write_fixture_immutable(p, {"v": 1})
    with pytest.raises(FixtureDriftError):
        write_fixture_immutable(p, {"v": 2})


def test_write_fixture_immutable_allows_idempotent_rewrite(tmp_path: Path) -> None:
    p = tmp_path / "fx.json"
    write_fixture_immutable(p, {"v": 1})
    # Same content: no-op rewrite must succeed.
    write_fixture_immutable(p, {"v": 1})


def test_iter_fixtures_walks_directory(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "1.json").write_text("""{"v":1}""")
    (tmp_path / "b.json").write_text("""{"v":2}""")
    found = list(iter_fixtures(tmp_path))
    paths = sorted(str(f.path.relative_to(tmp_path)) for f in found)
    assert paths == ["a/1.json", "b.json"]


def test_iter_fixtures_on_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list(iter_fixtures(tmp_path / "does-not-exist")) == []

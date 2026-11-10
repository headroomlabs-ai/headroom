"""Immutable, content-hashed fixture loader for perfkit.

Per ``contracts/mock-upstream-contract.md`` §4-§6 + FR-012 (mock upstream
fixtures are pinned, immutable).

Every fixture carries a SHA-256 content hash computed over its canonical
JSON serialization (sorted keys, UTF-8, no whitespace). The hash is the
``fixture_hash`` join key in every artifact that consumes the fixture, so
changing fixture contents invalidates every baseline that referenced the
old hash — a deliberate "fixture drift" detection mechanism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

FIXTURE_HASH_VERSION = "sha256-v1"


@dataclass(slots=True, frozen=True)
class Fixture:
    """A loaded, content-addressed fixture."""

    path: Path
    data: Mapping[str, Any]
    fixture_hash: str
    byte_size: int

    @property
    def name(self) -> str:
        return self.path.name


def canonical_bytes(payload: Any) -> bytes:
    """Stable serialization for hashing: sorted keys, no whitespace, UTF-8."""

    return json.dumps(
        _strip_to_jsonable(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def fixture_hash(payload: Any) -> str:
    """Return the canonical SHA-256 hex digest of ``payload``.

    The hash is stable across runs/machines as long as the JSON structure
    is identical. ``None`` and ``datetime``-like objects are serialized via
    :func:`canonical_bytes`.
    """

    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def short_hash(payload: Any, n: int = 12) -> str:
    """First ``n`` hex chars of the canonical hash (filename-safe)."""

    return fixture_hash(payload)[:n]


def _strip_to_jsonable(obj: Any) -> Any:
    """Recursively coerce ``obj`` into JSON-serializable primitives."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _strip_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_strip_to_jsonable(v) for v in obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Filesystem loader with immutable policy.
# ---------------------------------------------------------------------------


class FixtureDriftError(RuntimeError):
    """Raised when a fixture's on-disk content no longer matches its hash."""


@dataclass(slots=True, frozen=True)
class LoadedFixture:
    """Result of loading one fixture from disk."""

    fixture: Fixture
    expected_hash: str | None


def load_fixture(
    path: Path | str,
    expected_hash: str | None = None,
) -> Fixture:
    """Load a fixture; raise :class:`FixtureDriftError` on content mismatch.

    ``expected_hash`` is checked against the on-disk content; pass ``None``
    to disable the check (useful for one-shot ad-hoc debugging).
    """

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    sha = fixture_hash(data)
    if expected_hash is not None and sha != expected_hash:
        raise FixtureDriftError(
            f"fixture {p.name!r} has hash {sha!r} but expected {expected_hash!r} "
            f"(fixture_hash join key changed; rerun perfkit to update baselines)"
        )
    return Fixture(path=p, data=data, fixture_hash=sha, byte_size=p.stat().st_size)


def write_fixture_immutable(
    path: Path | str,
    data: Any,
) -> Fixture:
    """Write a fixture to ``path``, then refuse any later in-place rewrite.

    The file is written once. A second ``write_fixture_immutable`` to the
    same path with different data raises :class:`FixtureDriftError` because
    the on-disk hash no longer matches what was originally recorded; this
    enforces the immutability policy required by the mock-upstream contract.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_hash = fixture_hash(data)
    if p.exists():
        existing = json.loads(p.read_text(encoding="utf-8"))
        if fixture_hash(existing) != new_hash:
            raise FixtureDriftError(
                f"refusing to rewrite fixture {p.name!r}; "
                f"immutable policy (FR-012 / contracts/mock-upstream-contract.md §4)"
            )
        return Fixture(path=p, data=existing, fixture_hash=fixture_hash(existing), byte_size=p.stat().st_size)

    p.write_text(
        json.dumps(_strip_to_jsonable(data), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return Fixture(path=p, data=data, fixture_hash=new_hash, byte_size=p.stat().st_size)


def iter_fixtures(root: Path | str) -> Iterable[Fixture]:
    """Yield every fixture below ``root`` (depth-first)."""

    r = Path(root)
    if not r.exists():
        return
    for p in sorted(r.rglob("*.json")):
        yield load_fixture(p)

"""Check a HuggingFace cache holds the pinned artifacts an offline host needs.

Backs ``headroom verify-hf-cache`` and the "Offline upgrade path" section of the
configuration guide. Run it after seeding and before copying a cache to an
air-gapped host, so a partially-seeded cache is caught on the networked side
rather than discovered later as a silent estimation fallback.

Lives in the package rather than ``scripts/`` because the audience is an
operator with an installed wheel, where ``scripts/`` does not exist.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = ["check"]


def _snapshot(cache: Path, repo: str, pin: str) -> Path:
    """Python's cache layout: ``models--owner--name/snapshots/<commit>/``."""
    return cache / f"models--{repo.replace('/', '--')}" / "snapshots" / pin


def check(
    cache: Path,
    *,
    model: str | None = None,
    tokenizer_repo: str | None = None,
    require_kompress: bool = True,
) -> list[str]:
    """Missing artifacts under ``cache``; empty when the cache is usable.

    ``model`` names the model the deployment actually serves (``Qwen/Qwen2.5-7B``,
    or the short alias). Checking it is what prevents the estimation fallback the
    guide describes: Kompress being seeded says nothing about whether token
    *counting* for that model can load its tokenizer.

    ``require_kompress=False`` checks the counting tokenizer alone, for
    deployments that do not use Kompress.

    Returns human-readable strings — a path, or a sentence when the problem is a
    missing pin rather than a missing file — so the caller can print them
    verbatim.
    """
    from headroom.onnx_runtime import _PINNED_REVISIONS, _resolve_revision
    from headroom.tokenizers.huggingface import (
        _TOKENIZER_REVISIONS,
        _revision_for,
        get_tokenizer_name,
    )

    missing: list[str] = []

    def want(
        repo: str,
        filename: str,
        pins: dict[str, str],
        resolve: Callable[[str], str | None],
    ) -> None:
        # Resolve, do not read the table: the effective revision can differ from
        # the shipped one (``HEADROOM_HF_TOKENIZER_REVISIONS``), and checking the
        # table would pass while the loader asks for a snapshot nobody seeded.
        pin = resolve(repo)
        if pin is None:
            if repo in pins:
                # The table has a pin, so resolution dropped it on purpose: the
                # operator turned pinning off and the loader floats on `main`.
                missing.append(
                    f"{repo}: pinning disabled (HEADROOM_HF_PIN), so there is no "
                    "fixed snapshot to verify"
                )
            else:
                # Unpinned is a real problem: the deployment floats on `main`, so
                # this check cannot say what to seed.
                missing.append(f"{repo}: no pinned revision in this build")
            return
        path = _snapshot(cache, repo, pin) / filename
        if not path.is_file():
            missing.append(str(path))

    if require_kompress:
        from headroom.transforms.kompress_compressor import (
            _DEFAULT_ONNX_FILENAMES,
            _MODERNBERT_TOKENIZER_REPO,
            HF_MODEL_ID,
        )

        # One supported candidate is enough: the loader tries them in order and
        # skips the absent ones. Requiring a specific file (as the guide's
        # original snippet did) rejects valid caches.
        def onnx_revision(repo: str) -> str | None:
            return _resolve_revision(repo, None)

        model_pin = onnx_revision(HF_MODEL_ID)
        if model_pin is None:
            missing.append(f"{HF_MODEL_ID}: no pinned revision in this build")
        else:
            snap = _snapshot(cache, HF_MODEL_ID, model_pin)
            candidates = list(_DEFAULT_ONNX_FILENAMES)
            if not any((snap / name).is_file() for name in candidates):
                missing.append(f"{HF_MODEL_ID} @ {model_pin}: needs one of {candidates}")
        want(_MODERNBERT_TOKENIZER_REPO, "tokenizer.json", _PINNED_REVISIONS, onnx_revision)

    # Mirror what the runtime will actually fetch, which is not the same
    # question as "is this repo pinned".
    #
    # A served model goes through `HuggingFaceTokenizer` -> `get_tokenizer_name`,
    # so a *bare alias* ("qwen2.5-7b") and a raw repo id ("Qwen/Qwen2.5-7B") can
    # resolve to different repositories — the latter matches a family prefix
    # first and lands on "Qwen/Qwen-7B", a different vocabulary. Verifying the
    # literal id would pass while the runtime asks for a snapshot this guide
    # never seeded, and falls back to estimation. So `model` resolves exactly as
    # the runtime does.
    if model:
        want(
            get_tokenizer_name(model),
            "tokenizer.json",
            _TOKENIZER_REVISIONS,
            _revision_for,
        )
    if tokenizer_repo:
        # This one names a repository outright, so it is taken literally; only
        # its revision comes from the same resolution the loader uses.
        want(tokenizer_repo, "tokenizer.json", _TOKENIZER_REVISIONS, _revision_for)

    return missing

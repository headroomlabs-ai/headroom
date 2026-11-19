"""Check a HuggingFace cache holds the pinned artifacts an offline host needs.

Backs ``headroom verify-hf-cache`` and the "Offline upgrade path" section of the
configuration guide. Run it after seeding and before copying a cache to an
air-gapped host, so a partially-seeded cache is caught on the networked side
rather than discovered later as a silent estimation fallback.

Lives in the package rather than ``scripts/`` because the audience is an
operator with an installed wheel, where ``scripts/`` does not exist.
"""

from __future__ import annotations

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
    from headroom.onnx_runtime import _PINNED_REVISIONS
    from headroom.tokenizers.huggingface import _TOKENIZER_REVISIONS, get_tokenizer_name

    missing: list[str] = []

    def want(repo: str, filename: str, pins: dict[str, str]) -> None:
        pin = pins.get(repo)
        if pin is None:
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
        model_pin = _PINNED_REVISIONS.get(HF_MODEL_ID)
        if model_pin is None:
            missing.append(f"{HF_MODEL_ID}: no pinned revision in this build")
        else:
            snap = _snapshot(cache, HF_MODEL_ID, model_pin)
            candidates = list(_DEFAULT_ONNX_FILENAMES)
            if not any((snap / name).is_file() for name in candidates):
                missing.append(f"{HF_MODEL_ID} @ {model_pin}: needs one of {candidates}")
        want(_MODERNBERT_TOKENIZER_REPO, "tokenizer.json", _PINNED_REVISIONS)

    for repo in (model, tokenizer_repo):
        if not repo:
            continue
        if "/" in repo:
            # A Hub repo id ("owner/name") is taken literally. It must never be
            # alias-resolved: `get_tokenizer_name` matches family prefixes
            # before the allowlist, so "Qwen/Qwen2.5-7B" would silently become
            # "Qwen/Qwen-7B" — a different vocabulary — and an id absent from
            # the pin table would quietly check some other repo's snapshot
            # instead of reporting that this build cannot pin it.
            want(repo, "tokenizer.json", _TOKENIZER_REVISIONS)
            continue
        # A bare alias ("qwen2.5-7b") is what the resolver is for.
        want(get_tokenizer_name(repo), "tokenizer.json", _TOKENIZER_REVISIONS)

    return missing

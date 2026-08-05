#!/usr/bin/env python3
"""Verify that CI can load the default embedding model offline.

The main test shards run with TRANSFORMERS_OFFLINE=1. If the Hugging Face cache
misses or is partially restored, many unrelated memory tests fail later with
network/cache errors. This preflight keeps that failure mode early and specific.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from headroom.models.config import ML_MODEL_DEFAULTS

    model_name = ML_MODEL_DEFAULTS.sentence_transformer
    expected_dim = ML_MODEL_DEFAULTS.sentence_transformer_dim

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True)
        embedding = model.encode(["headroom cache preflight"], convert_to_numpy=True)
    except Exception as exc:
        print(
            f"::error::Hugging Face offline model cache is not usable for {model_name!r}: {exc}",
            file=sys.stderr,
        )
        print(
            "The prefetch-model job or fallback download must populate "
            "~/.cache/huggingface before offline test shards run.",
            file=sys.stderr,
        )
        return 1

    actual_dim = int(embedding.shape[-1])
    if actual_dim != expected_dim:
        print(
            "::error::Loaded embedding model has unexpected dimension: "
            f"{actual_dim} != {expected_dim}",
            file=sys.stderr,
        )
        return 1

    print(f"offline Hugging Face model cache OK: {model_name} ({actual_dim} dims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

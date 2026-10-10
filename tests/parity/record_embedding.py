"""Record the embedding parity fixture for the default relevance model.

Embeds a fixed set of texts with the Python ``EmbeddingScorer``, which loads
the pinned Qdrant/bge-small-en-v1.5-onnx-Q snapshot, and writes them to
``fixtures/embedding/bge_small_en_v15_pinned.json``. Two gated tests compare
against it: ``tests/test_fastembed_pinned_snapshot.py`` (Python) and
``fastembed_matches_parity_fixture`` in
``crates/headroom-core/src/relevance/embedding.rs`` (Rust). Both reading one
fixture bounds the drift between the languages by ``MAX_ABS_DIFF``.

The languages run the same ONNX file, so the gap between them comes from the
ONNX Runtime build, not the pipeline. Measured when this fixture was recorded:
Rust and Python on the same ORT (1.31.0) agree within 1e-6 per component;
Python on ORT 1.23.2 vs 1.31.0 differs by up to 4.1e-4. ``MAX_ABS_DIFF``
covers the ORT range the project supports with ~2x headroom.

This bounds numeric drift; it does not identify the model. Xenova's fp32
export of the same model sits within 4.7e-4 of this fixture, inside the
tolerance. Which files load is guaranteed by the pinned SHA and the unit tests
on both sides, not by this fixture.

Re-record after bumping the pinned revision (needs network on a cold cache),
with an interpreter whose onnxruntime is >= 1.24, the runtime the Rust side
needs:
    python tests/parity/record_embedding.py
"""

from __future__ import annotations

import json
import os

from headroom.onnx_runtime import _PINNED_REVISIONS
from headroom.relevance.embedding import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REPO, EmbeddingScorer

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "embedding", "bge_small_en_v15_pinned.json"
)

# Largest allowed |embedding component - fixture|, on either side. See the
# module docstring for how it was chosen.
MAX_ABS_DIFF = 1e-3

TEXTS = [
    "login error",
    "authentication failed for user admin: invalid credentials",
    '{"status": "failed", "error": "connection refused", "retries": 3}',
    "def retry(fn, attempts=3):\n    for i in range(attempts):\n        return fn()",
    "Ünïcödé text, emoji 🚀 and CJK 中文 mixed in one line.",
    # Longer than the 512-token window, so truncation is covered too.
    " ".join(f"log line {i}: request handled in {i % 97} ms" for i in range(400)),
]


def main() -> None:
    scorer = EmbeddingScorer(DEFAULT_MODEL_NAME)
    embeddings = scorer._encode(TEXTS)
    fixture = {
        "model": DEFAULT_MODEL_NAME,
        "repo": DEFAULT_MODEL_REPO,
        "revision": _PINNED_REVISIONS[DEFAULT_MODEL_REPO],
        "max_abs_diff": MAX_ABS_DIFF,
        "texts": TEXTS,
        "embeddings": [[round(float(v), 8) for v in row] for row in embeddings],
    }
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {FIXTURE_PATH} ({len(TEXTS)} texts x {len(embeddings[0])} dims)")


if __name__ == "__main__":
    main()

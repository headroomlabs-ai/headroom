"""Deterministic scoring shared by the grep and rerank edges.

`Scorer` is the pluggable interface both edges score chunks/lines through.
`TokenOverlapScorer` is a stand-in for local wiring and tests only — it is
NOT a substitute for real next-step recall, which needs an actual reranker
(the protocol's own "Models to try on the A6000" list; on this project, a
0.6B reranker on the RTX 5070 rather than the A6000). `HttpRerankerScorer` is
the real implementation point: it calls a reranker service over HTTP with a
wall-clock timeout, since that's a real I/O boundary where a timeout is
meaningful (unlike the CPU-only paths — see cache_safety.cap_input).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Protocol

logger = logging.getLogger("headroom_edge_layer")

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class Scorer(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]:
        """Return one relevance score per candidate, higher = more relevant.

        Must be deterministic for a given (query, candidates) pair — greedy,
        no randomness, no clock — so a replay of the same request produces
        byte-identical output (cache-safety rule #2).
        """


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


class TokenOverlapScorer:
    """Deterministic, no-model fallback: Jaccard overlap of word sets.

    Stand-in only. Wire the real reranker (HttpRerankerScorer) before trusting
    any next-step-recall number produced with this scorer.
    """

    def score(self, query: str, candidates: list[str]) -> list[float]:
        query_words = _tokenize(query)
        if not query_words:
            return [0.0] * len(candidates)
        scores = []
        for candidate in candidates:
            candidate_words = _tokenize(candidate)
            if not candidate_words:
                scores.append(0.0)
                continue
            overlap = len(query_words & candidate_words)
            scores.append(overlap / len(query_words | candidate_words))
        return scores


class HttpRerankerScorer:
    """Calls a reranker service (e.g. bge-reranker-v2-m3 on the RTX 5070).

    POSTs {"query": ..., "candidates": [...]} and expects {"scores": [...]}.
    On any error, timeout, or malformed response, fails open by returning
    the fallback scorer's scores for this call only — never raises into the
    request path.
    """

    def __init__(self, url: str, *, timeout_seconds: float, fallback: Scorer | None = None):
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._fallback = fallback or TokenOverlapScorer()

    def score(self, query: str, candidates: list[str]) -> list[float]:
        payload = json.dumps({"query": query, "candidates": candidates}).encode()
        request = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read())
            scores = body["scores"]
            if len(scores) != len(candidates):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
                )
            return [float(s) for s in scores]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            logger.warning(
                "reranker call to %s failed (%s); failing open to TokenOverlapScorer",
                self._url,
                exc,
            )
            return self._fallback.score(query, candidates)


def build_scorer(*, reranker_url: str, timeout_seconds: float) -> Scorer:
    if reranker_url:
        return HttpRerankerScorer(reranker_url, timeout_seconds=timeout_seconds)
    return TokenOverlapScorer()

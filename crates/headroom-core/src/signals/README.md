# `signals/` — detection traits

Cross-cutting classifiers used by transforms. Lives at the crate root because the same classifier feeds many transforms; nesting under `transforms/` would imply ownership by one consumer.

## Trait family

| Trait | Granularity | Status |
|---|---|---|
| `LineImportanceDetector` | one line at a time | shipped (Phase 3e.1) |
| `ContentTypeDetector` | whole blob | future generalization of `transforms::detection` |
| `ItemImportanceDetector<I>` | `&[I]` ranking | future, for SmartCrusher cells / search hits |

## Tiering — composition, not inheritance

`Tiered<dyn Trait>` chains an ordered stack. The first tier whose signal exceeds `ESCALATE_THRESHOLD` (0.7) confidence wins; lower-confidence tiers fall through. If nothing crosses the threshold, the highest-confidence signal seen is returned so the caller still gets the best guess.

Today `KeywordDetector` is the only tier registered. The tier API is the seam where future ML detectors slot in.

## How to add a new detector

1. **Confirm granularity.** A line classifier implements `LineImportanceDetector`. A blob classifier gets a new trait. Don't shoehorn cross-granularity work into one trait.
2. **Implement `score(&self, ...) -> ImportanceSignal`.** Set `confidence` honestly: 0.7+ if the detector is the right authority for this input, lower if you want the next tier to override on disagreement.
3. **No silent fallbacks.** Per project conventions, return `ImportanceSignal::neutral()` when you have no information — never fabricate a positive answer with low confidence to "fail open".
4. **Wire into `Tiered` at the consumer**, not in this module. The detector itself doesn't know about other tiers.
5. **Add parity fixtures** if the detector replaces or augments an existing one. Mark divergence lines with `// fixed_in_<phase>` markers.

## Canonical future ML extension — BGE classifier head

The most likely next tier is a classification head on the existing `bge-small-en-v1.5` embedder loaded by `relevance::EmbeddingScorer`:

```rust
pub struct BgeClassifierDetector {
    embedder: Arc<dyn Embedder>,        // shared with relevance scoring
    classifier: LogisticRegression,     // 384-dim → 4-class softmax
    threshold: f32,                     // calibrated on validation set
}

impl LineImportanceDetector for BgeClassifierDetector { ... }
```

Why this is the cheapest path:

- The embedder is already loaded for SmartCrusher relevance scoring. A classification head adds ~1.5 KB of weights and ~1 ms inference per line (batchable).
- No new ONNX runtime, no new model file, no new download.
- Calibrated confidence lets the head short-circuit `KeywordDetector` on high-confidence positives but step aside on borderlines (where the keyword automaton is reliable anyway).

Two alternatives kept open in case BGE-head underfits:

- **Distilled tinyBERT (ONNX)** — more accurate, +10–20 MB model, +3 ms latency, new `ort` dependency.
- **Logistic regression on lexical features** — caps ratio, line length, structural markers, stack-frame heuristics. ~5 KB model, fastest of the three. Good A/B baseline.

The trait shape accepts all three without changes.

## What does NOT live here

- Concrete transforms — they go in `crates/headroom-core/src/transforms/`.
- Static keyword data tables — they're configuration for `KeywordDetector`, not detection logic. They live alongside the detector that consumes them (`signals/keyword_detector.rs::KeywordRegistry`).
- Tag protection (`<headroom:keep>` markers) — that's user intent, not classification.

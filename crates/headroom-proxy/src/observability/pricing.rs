//! Per-model price book for valuing savings in USD.
//!
//! The savings store records token counts; turning those into dollars needs
//! per-model prices, which Headroom does not hard-code. This module loads a
//! price table (the [models.dev] JSON schema, the same source the rest of the
//! ecosystem uses) and resolves a request's model id to its per-token prices.
//!
//! It is deliberately separate from [`super::stats`] so the store stays a pure
//! token aggregator and the (provider-specific, substring-resolving) price lookup
//! is tested in isolation. An empty price book is valid — savings are then
//! reported in tokens with USD left at zero rather than guessed.
//!
//! [models.dev]: https://models.dev

use std::collections::HashMap;

use serde::Deserialize;

/// Per-token USD prices for one model. Cache prices fall back to the input price
/// when a provider does not publish them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// A resolved set of per-model prices.
#[derive(Clone, Debug, Default)]
pub struct PriceBook {
    models: HashMap<String, ModelPrice>,
}

// ---- models.dev JSON shape (only the fields we use) ----------------------- #

#[derive(Deserialize)]
struct DevCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

#[derive(Deserialize)]
struct DevModel {
    cost: Option<DevCost>,
}

#[derive(Deserialize)]
struct DevProvider {
    models: Option<HashMap<String, DevModel>>,
}

/// Normalize a model id for lookup: lowercase and drop any version suffix after
/// a colon (`...-v1:0` → `...-v1`).
fn normalize(model: &str) -> String {
    let lower = model.trim().to_lowercase();
    match lower.split_once(':') {
        Some((head, _)) => head.to_string(),
        None => lower,
    }
}

impl PriceBook {
    /// An empty price book — every lookup misses; USD savings report as zero.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of priced models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the book has no entries.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Build from a [models.dev]-shaped JSON document. Prices in that schema are
    /// per **million** tokens; they are converted to per-token here. Malformed
    /// input yields an empty book rather than an error — pricing is best-effort.
    ///
    /// [models.dev]: https://models.dev
    pub fn from_models_dev_json(json: &str) -> Self {
        let parsed: HashMap<String, DevProvider> = match serde_json::from_str(json) {
            Ok(p) => p,
            Err(_) => return Self::empty(),
        };
        let mut models = HashMap::new();
        for provider in parsed.values() {
            let Some(provider_models) = &provider.models else {
                continue;
            };
            for (model_id, model) in provider_models {
                let Some(cost) = &model.cost else { continue };
                let (Some(input), Some(output)) = (cost.input, cost.output) else {
                    continue;
                };
                let per = 1_000_000.0;
                let input_pt = input / per;
                let price = ModelPrice {
                    input: input_pt,
                    output: output / per,
                    cache_read: cost.cache_read.map(|c| c / per).unwrap_or(input_pt),
                    cache_write: cost.cache_write.map(|c| c / per).unwrap_or(input_pt),
                };
                models.insert(normalize(model_id), price);
            }
        }
        Self { models }
    }

    /// Resolve a request model id to its prices, or `None` when unpriced.
    ///
    /// Resolution order (each over the normalized id):
    /// 1. exact match;
    /// 2. the longest stored id that the request id *contains* — so vendor/region
    ///    prefixes resolve correctly: Bedrock's `eu.anthropic.claude-haiku-4-5-…`
    ///    and LiteLLM's `copilot-gpt-5-mini` both contain the bare models.dev id.
    ///
    /// Only the request-contains-stored direction is matched. The reverse (a
    /// stored id containing the request) is deliberately NOT matched: a short or
    /// generic request id (e.g. a bare `claude`) would otherwise resolve to
    /// whichever stored id is longest (`claude-opus-…`), mispricing it. A request
    /// shorter than its canonical id simply misses → unpriced (USD 0), which is
    /// safer than guessing the wrong model's rate.
    pub fn lookup(&self, model: &str) -> Option<ModelPrice> {
        if self.models.is_empty() {
            return None;
        }
        let key = normalize(model);
        if let Some(p) = self.models.get(&key) {
            return Some(*p);
        }
        // Longest stored id contained in the request id. `max_by_key` keeps the
        // selection branch in std (deterministic, not a coverage region here).
        // Skip empty ids: a malformed price book with an empty model key would
        // otherwise `contains("")`-match (and misprice) every request.
        self.models
            .iter()
            .filter(|(id, _)| !id.is_empty() && key.contains(id.as_str()))
            .max_by_key(|(id, _)| id.len())
            .map(|(_, price)| *price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "anthropic": {
            "models": {
                "claude-haiku-4-5": {
                    "cost": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25}
                },
                "claude-opus-4-8": { "cost": {"input": 15.0, "output": 75.0} }
            }
        },
        "github-copilot": {
            "models": { "gpt-5-mini": { "cost": {"input": 0.25, "output": 2.0} } }
        },
        "broken": {
            "models": {
                "no-cost": {},
                "no-output": { "cost": {"input": 1.0} }
            }
        },
        "no-models": {}
    }"#;

    #[test]
    fn normalize_lowercases_and_strips_version() {
        assert_eq!(normalize("  Claude-Haiku-4-5-v1:0 "), "claude-haiku-4-5-v1");
        assert_eq!(normalize("gpt-5-mini"), "gpt-5-mini");
    }

    #[test]
    fn empty_book_misses_everything() {
        let book = PriceBook::empty();
        assert!(book.is_empty());
        assert_eq!(book.len(), 0);
        assert_eq!(book.lookup("claude-haiku-4-5"), None);
    }

    #[test]
    fn parses_models_dev_and_converts_per_million_to_per_token() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        let p = book.lookup("claude-haiku-4-5").unwrap();
        assert!((p.input - 1e-6).abs() < 1e-15);
        assert!((p.output - 5e-6).abs() < 1e-15);
        assert!((p.cache_read - 1e-7).abs() < 1e-15);
        assert!((p.cache_write - 1.25e-6).abs() < 1e-15);
    }

    #[test]
    fn cache_prices_default_to_input_when_absent() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        let p = book.lookup("claude-opus-4-8").unwrap();
        assert_eq!(p.cache_read, p.input);
        assert_eq!(p.cache_write, p.input);
    }

    #[test]
    fn skips_models_without_usable_cost() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        assert_eq!(book.lookup("no-cost"), None);
        assert_eq!(book.lookup("no-output"), None);
    }

    #[test]
    fn malformed_json_yields_empty_book() {
        let book = PriceBook::from_models_dev_json("{ not json ]");
        assert!(book.is_empty());
    }

    #[test]
    fn copilot_prefixed_id_resolves_via_substring() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        // LiteLLM names it `copilot-gpt-5-mini`; models.dev stores `gpt-5-mini`.
        let direct = book.lookup("gpt-5-mini").unwrap();
        let alias = book.lookup("copilot-gpt-5-mini").unwrap();
        assert_eq!(direct, alias);
    }

    #[test]
    fn substring_match_resolves_region_prefixed_bedrock_id() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        // Bedrock cross-region id contains the stored "claude-haiku-4-5".
        let p = book
            .lookup("eu.anthropic.claude-haiku-4-5-20251001-v1:0")
            .unwrap();
        assert!((p.input - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn substring_match_picks_longest_stored_id() {
        let json = r#"{"p":{"models":{
            "claude": {"cost":{"input":9.0,"output":9.0}},
            "claude-haiku-4-5": {"cost":{"input":1.0,"output":5.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        // "claude-haiku-4-5-v1" contains both "claude" and "claude-haiku-4-5";
        // the longer, more specific id wins.
        let p = book.lookup("claude-haiku-4-5-v1").unwrap();
        assert!((p.input - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn unknown_model_misses() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        assert_eq!(book.lookup("totally-unknown-model"), None);
    }

    #[test]
    fn empty_stored_model_id_never_matches() {
        // A malformed price book with an empty/whitespace model key must not
        // `contains("")`-match (and misprice) every request.
        let json = r#"{"p":{"models":{
            "   ": {"cost":{"input":99.0,"output":99.0}},
            "gpt-5-mini": {"cost":{"input":0.25,"output":2.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        // A request containing neither real id misses — not the empty-key entry.
        assert_eq!(book.lookup("totally-unrelated"), None);
        // Real models still resolve via substring.
        assert!(book.lookup("copilot-gpt-5-mini").is_some());
    }

    #[test]
    fn short_request_id_does_not_mismatch_to_a_longer_stored_id() {
        // The reverse (stored-contains-request) direction is intentionally NOT
        // matched: a bare `claude` must miss rather than resolve to the longest
        // stored `claude-*` and get mispriced.
        let book = PriceBook::from_models_dev_json(SAMPLE);
        assert_eq!(book.lookup("claude"), None);
        // `gpt` must not resolve to `gpt-5-mini` either.
        assert_eq!(book.lookup("gpt"), None);
    }
}

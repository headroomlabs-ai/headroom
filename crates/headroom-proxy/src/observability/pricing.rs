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
    /// Consensus (mode-reduced) price per model id, with a substring fallback in
    /// [`PriceBook::lookup`]. Used when no route-specific price is found.
    models: HashMap<String, ModelPrice>,
    /// Exact per-route price: request-provider (`anthropic`/`openai`/`bedrock`/
    /// `vertex`) → normalized model id → price, sourced from the matching
    /// models.dev provider, so a request is valued at the price of the backend it
    /// actually used rather than a cross-provider consensus.
    by_provider: HashMap<String, HashMap<String, ModelPrice>>,
}

/// Map a models.dev provider id to the request-provider label the recording lanes
/// attribute (`proxy::provider_for_endpoint` and the Bedrock/Vertex lanes), so a
/// route's exact price is reachable by the provider already in hand. `None` for
/// providers we don't serve directly (resellers/gateways) — those only feed the
/// cross-provider consensus (`models`) book.
fn request_provider_for(models_dev_id: &str) -> Option<&'static str> {
    Some(match models_dev_id {
        "anthropic" => "anthropic",
        "openai" => "openai",
        "amazon-bedrock" => "bedrock",
        // Vertex serves both Anthropic-on-Vertex and Gemini.
        "google-vertex" | "google-vertex-anthropic" | "google" => "vertex",
        _ => return None,
    })
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

/// Minimum length for a stored id to be eligible for the substring fallback in
/// [`PriceBook::lookup`]. The catalog's bare model *families* — the ones that
/// would swallow a *different* model and misprice it wildly (`gpt-4` $60/M,
/// `gpt-5`, `o1` $15/M, `glm-4` $15/M) — are all <= 5 chars. Requiring >= 6
/// blocks those while still resolving real 6-char *leaf* models (`gpt-4o`,
/// `grok-4`, `o1-pro`) when they arrive vendor/region-prefixed (`copilot-gpt-4o`,
/// `azure/gpt-4o`); a 6-char leaf's only residual fallback risk is a same-family
/// dated variant of itself, which prices about right. Tunable: raise it to be
/// stricter (more misses, fewer mispricings).
const MIN_SUBSTRING_MATCH_LEN: usize = 6;

/// If `id` starts with a Bedrock cross-region inference prefix (`global.`,
/// `eu.`, `us.`, `au.`, `jp.`), return the base id after stripping it.
/// These region prefixes identify the AWS routing zone for Bedrock's
/// cross-region inference profiles; the base id matches models.dev's
/// canonical entry (e.g. `global.anthropic.claude-haiku-…` → `anthropic.claude-haiku-…`).
fn strip_bedrock_region_prefix(id: &str) -> Option<&str> {
    crate::bedrock::vendor::GEO_PREFIXES
        .iter()
        .find_map(|p| id.strip_prefix(p))
}

/// Normalize a model id for lookup: lowercase, fold Vertex's `@`-version
/// separator to a hyphen (`claude-3-5-sonnet@20240620` → `...-20240620`, the form
/// models.dev stores), and drop any `:`-version suffix (`...-v1:0` → `...-v1`).
/// Applied to both stored ids and request ids, so the `@`→`-` fold stays
/// symmetric (Cloudflare `@cf/...` ids map to `-cf/...` on both sides and still
/// match each other).
fn normalize(model: &str) -> String {
    let lower = model.trim().to_lowercase().replace('@', "-");
    match lower.split_once(':') {
        Some((head, _)) => head.to_string(),
        None => lower,
    }
}

/// Pick the consensus price among the providers that list one model id. The
/// official rate is the one the bulk of resellers mirror, so the **modal**
/// per-token input price is the canonical price and rejects outliers — a gateway
/// markup, a discount reseller, or a $0 free-tier mirror. Deterministic, never
/// depending on map iteration order.
///
/// A genuine frequency tie (e.g. `gpt-4` listed as `openai: $30` ×2 and
/// `azure: $60` ×2) breaks first to a real (non-zero) price over a $0 mirror,
/// then to the **lower** price — the first-party rate, since the high side of
/// such a tie is the gateway markup, not the canonical cost.
fn modal_price(prices: &[ModelPrice]) -> ModelPrice {
    use std::cmp::Ordering::Equal;
    // Group by exact input price (bit pattern); keep a count + representative. The
    // representative's output/cache fields are kept DETERMINISTIC (lowest by
    // (output, cache_read, cache_write)) so the whole ModelPrice — not just the
    // winning input — is independent of map iteration order.
    let mut by_input: HashMap<u64, (usize, ModelPrice)> = HashMap::new();
    for p in prices {
        let entry = by_input.entry(p.input.to_bits()).or_insert((0, *p));
        entry.0 += 1;
        let fields = |m: &ModelPrice| (m.output, m.cache_read, m.cache_write);
        if fields(p).partial_cmp(&fields(&entry.1)) == Some(std::cmp::Ordering::Less) {
            entry.1 = *p;
        }
    }
    by_input
        .into_values()
        .max_by(|(ca, pa), (cb, pb)| {
            ca.cmp(cb)
                // count tie → prefer a real price over a $0 free-tier mirror
                .then((pa.input > 0.0).cmp(&(pb.input > 0.0)))
                // then the lower price (first-party rate, not a gateway markup)
                .then(pb.input.partial_cmp(&pa.input).unwrap_or(Equal))
        })
        .map(|(_, price)| price)
        .unwrap_or_default()
}

/// Vendored snapshot of the models.dev catalog (`https://models.dev/api.json`),
/// refreshed via `scripts/refresh_model_limits.sh`. ~2.4MB; embedded into the
/// binary so the proxy values USD savings out of the box with no startup network
/// dependency. Operators who want live prices pass `--pricebook <url>`.
const VENDORED_MODELS_DEV_JSON: &str = include_str!("../../data/models_dev.json");

impl PriceBook {
    /// An empty price book — every lookup misses; USD savings report as zero.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The compiled-in models.dev snapshot — the default price book when
    /// `--pricebook` is unset. The 2.4MB snapshot is parsed at most once per
    /// process (cached in a `OnceLock`, mirroring `compression::model_limits`)
    /// and cloned out; an unparseable snapshot fails open to empty.
    pub fn vendored() -> Self {
        static VENDORED: std::sync::OnceLock<PriceBook> = std::sync::OnceLock::new();
        VENDORED
            .get_or_init(|| Self::from_models_dev_json(VENDORED_MODELS_DEV_JSON))
            .clone()
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
        // models.dev lists the SAME model id under many providers (the first-party
        // vendor plus resellers/gateways) at DIFFERENT prices. Collect every
        // candidate per normalized id, then pick the modal price below — a plain
        // last-write-wins over a HashMap is non-deterministic across runs AND lets
        // a reseller outlier (or a $0 free-tier mirror) shadow the canonical rate.
        let mut candidates: HashMap<String, Vec<ModelPrice>> = HashMap::new();
        // Per-route candidates, scoped to the models.dev providers that back the
        // backends we serve, so a request can be valued at its actual route's rate.
        let mut route_candidates: HashMap<&'static str, HashMap<String, Vec<ModelPrice>>> =
            HashMap::new();
        for (provider_id, provider) in &parsed {
            let Some(provider_models) = &provider.models else {
                continue;
            };
            let route = request_provider_for(provider_id);
            for (model_id, model) in provider_models {
                let Some(cost) = &model.cost else { continue };
                // Require only `input`: compression savings are valued from the
                // input price, so a model with an input price but no published
                // output price (embedding/classifier models, promotional
                // `output: null`) is still usable. `output` defaults to 0 — it
                // is forward-schema (no lane counts output tokens yet).
                //
                // Reject a NEGATIVE input as invalid data: the compression USD
                // accrual is intentionally unclamped (hot path), so a negative
                // price would drive the headline savings *down*. A model with a
                // bad/negative input price is skipped → unpriced (USD 0), the safe
                // default. `0.0` is kept (genuinely free / promotional models).
                let Some(input) = cost.input.filter(|v| *v >= 0.0) else {
                    continue;
                };
                let input_pt = input / 1_000_000.0;
                // Per-token price for an optional field: drop a negative (invalid,
                // would inflate/decrement savings), convert per-million → per-token,
                // and fall back to `default` when absent or rejected.
                let per_token = |field: Option<f64>, default: f64| {
                    field
                        .filter(|v| *v >= 0.0)
                        .map(|v| v / 1_000_000.0)
                        .unwrap_or(default)
                };
                let price = ModelPrice {
                    input: input_pt,
                    output: per_token(cost.output, 0.0),
                    cache_read: per_token(cost.cache_read, input_pt),
                    cache_write: per_token(cost.cache_write, input_pt),
                };
                let key = normalize(model_id);
                candidates.entry(key.clone()).or_default().push(price);
                if let Some(route) = route {
                    let route_map = route_candidates.entry(route).or_default();
                    route_map.entry(key.clone()).or_default().push(price);
                    // For Bedrock, also index under the region-prefix-stripped base
                    // key so a `eu.`/`us.`/`au.`/`jp.` request can substring-match
                    // the base key in `lookup_with_provider` and get the route-
                    // specific price rather than falling back to consensus.
                    if route == "bedrock" {
                        if let Some(base) = strip_bedrock_region_prefix(&key) {
                            route_map.entry(base.to_string()).or_default().push(price);
                        }
                    }
                }
            }
        }
        let reduce = |m: HashMap<String, Vec<ModelPrice>>| -> HashMap<String, ModelPrice> {
            m.into_iter()
                .map(|(id, ps)| (id, modal_price(&ps)))
                .collect()
        };
        let models = reduce(candidates);
        let by_provider = route_candidates
            .into_iter()
            .map(|(route, models)| (route.to_string(), reduce(models)))
            .collect();
        Self {
            models,
            by_provider,
        }
    }

    /// Resolve a request model id to its prices, or `None` when unpriced.
    ///
    /// Resolution order (each over the normalized id):
    /// 1. exact match;
    /// 2. the longest stored id (>= [`MIN_SUBSTRING_MATCH_LEN`]) that the request
    ///    id *contains* — so vendor/region prefixes resolve correctly: Bedrock's
    ///    `eu.anthropic.claude-haiku-4-5-…` and LiteLLM's `copilot-gpt-5-mini`
    ///    both contain the bare models.dev id.
    ///
    /// Only the request-contains-stored direction is matched. The reverse (a
    /// stored id containing the request) is deliberately NOT matched: a short or
    /// generic request id (e.g. a bare `claude`) would otherwise resolve to
    /// whichever stored id is longest (`claude-opus-…`), mispricing it. A request
    /// shorter than its canonical id simply misses → unpriced (USD 0), which is
    /// safer than guessing the wrong model's rate.
    ///
    /// The minimum-length floor is the safety knob: the catalog holds bare model
    /// *families* (`gpt-4` at $60/M, `o1`, `gpt-5`, `glm-4`, …). Without the floor
    /// an uncatalogued variant like `gpt-4.5-preview` would substring-match legacy
    /// `gpt-4` and misprice ~20x. With it, such a variant safely misses (USD 0,
    /// the pre-pricing default) instead of inheriting a wrong sibling's rate.
    pub fn lookup(&self, model: &str) -> Option<ModelPrice> {
        self.lookup_key(&normalize(model))
    }

    /// Consensus resolution over an already-[`normalize`]d key (exact, then the
    /// longest-substring fallback). Shared by `lookup` and `lookup_with_provider`
    /// so neither re-normalizes.
    fn lookup_key(&self, key: &str) -> Option<ModelPrice> {
        if self.models.is_empty() {
            return None;
        }
        if let Some(p) = self.models.get(key) {
            return Some(*p);
        }
        // Longest stored id contained in the request id, ignoring ids too short to
        // be specific (see `MIN_SUBSTRING_MATCH_LEN`) — that also subsumes the
        // empty-id guard. Length ties break by id string so the result is
        // deterministic (HashMap iteration order is unspecified).
        self.models
            .iter()
            .filter(|(id, _)| id.len() >= MIN_SUBSTRING_MATCH_LEN && key.contains(id.as_str()))
            .max_by(|(a, _), (b, _)| a.len().cmp(&b.len()).then_with(|| a.as_str().cmp(b)))
            .map(|(_, price)| *price)
    }

    /// Resolve a request to its price, preferring the **exact price of the route
    /// that served it** — the price models.dev lists under the request's own
    /// provider (`anthropic`/`openai`/`bedrock`/`vertex`) — and only falling back
    /// to the cross-provider consensus [`lookup`] (incl. its substring fallback)
    /// when that provider doesn't list the model (a region-prefixed Bedrock id, an
    /// unmapped provider, or an empty book). This values e.g. an OpenAI-routed
    /// `gpt-4` at openai's real $30 instead of a consensus tie that could land on
    /// an azure markup.
    ///
    /// [`lookup`]: PriceBook::lookup
    pub fn lookup_with_provider(&self, provider: &str, model: &str) -> Option<ModelPrice> {
        let key = normalize(model);
        if let Some(provider_map) = self.by_provider.get(provider) {
            if let Some(price) = provider_map.get(&key) {
                return Some(*price);
            }
            // Substring fallback within this provider's map: a region-prefixed
            // request id (e.g. `eu.anthropic.claude-haiku-…`) contains the
            // base key (`anthropic.claude-haiku-…`) that was indexed from the
            // Bedrock catalog entry. Same longest-wins / deterministic ordering
            // as `lookup_key`.
            let route_price = provider_map
                .iter()
                .filter(|(id, _)| id.len() >= MIN_SUBSTRING_MATCH_LEN && key.contains(id.as_str()))
                .max_by(|(a, _), (b, _)| a.len().cmp(&b.len()).then_with(|| a.as_str().cmp(b)))
                .map(|(_, price)| *price);
            if route_price.is_some() {
                return route_price;
            }
        }
        self.lookup_key(&key)
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
                "no-output": { "cost": {"input": 1.0} },
                "no-input": { "cost": {"output": 5.0} }
            }
        },
        "no-models": {}
    }"#;

    #[test]
    fn conflicting_cross_provider_prices_resolve_to_the_modal_price() {
        // The same id appears under many providers at different rates; pick the
        // consensus (most common) price deterministically, ignoring a cheap
        // reseller and a $0 free-tier mirror. Under the old last-write-wins this
        // was non-deterministic (HashMap order) and could yield 0.43 or 0.0.
        let json = r#"{
            "anthropic":      {"models": {"claude-x": {"cost": {"input": 3.0}}}},
            "reseller-a":     {"models": {"claude-x": {"cost": {"input": 3.0}}}},
            "reseller-b":     {"models": {"claude-x": {"cost": {"input": 3.0}}}},
            "cheap-reseller": {"models": {"claude-x": {"cost": {"input": 0.43}}}},
            "free-mirror":    {"models": {"claude-x": {"cost": {"input": 0.0}}}}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        let p = book.lookup("claude-x").unwrap();
        // 3.0/M is the mode (3×) → 3e-6 per token, regardless of map order.
        assert!((p.input - 3e-6).abs() < 1e-15, "got {}", p.input);
    }

    #[test]
    fn modal_price_tie_breaks_to_lower_real_price_not_markup() {
        // Genuine 2-2 frequency tie (real gpt-4 shape): openai/llmgateway $30 vs
        // azure/azure-cog $60. Must pick the first-party $30, not the $60 markup.
        let json = r#"{
            "openai":   {"models": {"gpt-4": {"cost": {"input": 30.0}}}},
            "llmgw":    {"models": {"gpt-4": {"cost": {"input": 30.0}}}},
            "azure":    {"models": {"gpt-4": {"cost": {"input": 60.0}}}},
            "azure-cs": {"models": {"gpt-4": {"cost": {"input": 60.0}}}}
        }"#;
        let p = PriceBook::from_models_dev_json(json)
            .lookup("gpt-4")
            .unwrap();
        assert!((p.input - 30e-6).abs() < 1e-15, "got {}", p.input);
    }

    #[test]
    fn modal_price_tie_prefers_real_price_over_free_mirror() {
        // 2-2 tie between a $0 free-tier mirror and the real price → pick the real.
        let json = r#"{
            "github-models": {"models": {"x": {"cost": {"input": 0.0}}}},
            "free-mirror":   {"models": {"x": {"cost": {"input": 0.0}}}},
            "openai":        {"models": {"x": {"cost": {"input": 2.5}}}},
            "vendor":        {"models": {"x": {"cost": {"input": 2.5}}}}
        }"#;
        let p = PriceBook::from_models_dev_json(json).lookup("x").unwrap();
        assert!((p.input - 2.5e-6).abs() < 1e-15, "got {}", p.input);
    }

    #[test]
    fn vendored_catalog_populates_every_route() {
        // Guards `request_provider_for`'s hardcoded models.dev ids: a rename (e.g.
        // `amazon-bedrock` → something) would silently empty that route, degrading
        // it to the consensus book undetected (`len`/`is_empty` only see `models`).
        // Each served route must have a non-empty per-route price map.
        let book = PriceBook::vendored();
        for route in ["anthropic", "openai", "bedrock", "vertex"] {
            let m = book.by_provider.get(route).unwrap_or_else(|| {
                panic!("by_provider missing route `{route}` (models.dev rename?)")
            });
            assert!(!m.is_empty(), "route `{route}` has no priced models");
        }
    }

    #[test]
    fn provider_aware_lookup_uses_the_routes_exact_price() {
        // gpt-4: openai lists $30, azure (a non-route reseller) $60. An
        // OpenAI-routed request must be valued at openai's exact $30, never the
        // azure markup — and an unmapped route falls back to the consensus.
        let json = r#"{
            "openai": {"models": {"gpt-4": {"cost": {"input": 30.0}}}},
            "azure":  {"models": {"gpt-4": {"cost": {"input": 60.0}}}}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        assert!(
            (book.lookup_with_provider("openai", "gpt-4").unwrap().input - 30e-6).abs() < 1e-15
        );
        // Unknown route → consensus fallback (tie-break prefers the lower $30).
        assert!(
            (book.lookup_with_provider("bedrock", "gpt-4").unwrap().input - 30e-6).abs() < 1e-15
        );
    }

    #[test]
    fn provider_aware_route_price_can_differ_from_consensus() {
        // amazon-bedrock prices a model above the consensus; a Bedrock request
        // gets the bedrock rate, an Anthropic request the anthropic rate, and the
        // model-only consensus stays the mode ($3).
        let json = r#"{
            "anthropic":      {"models": {"claude-z": {"cost": {"input": 3.0}}}},
            "amazon-bedrock": {"models": {"claude-z": {"cost": {"input": 3.6}}}},
            "reseller":       {"models": {"claude-z": {"cost": {"input": 3.0}}}}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        let an = book.lookup_with_provider("anthropic", "claude-z").unwrap();
        let bd = book.lookup_with_provider("bedrock", "claude-z").unwrap();
        assert!((an.input - 3e-6).abs() < 1e-15);
        assert!((bd.input - 3.6e-6).abs() < 1e-15);
        assert!((book.lookup("claude-z").unwrap().input - 3e-6).abs() < 1e-15);
    }

    #[test]
    fn provider_aware_falls_back_to_substring_for_region_prefixed_bedrock() {
        // Bedrock's `global.`-prefixed catalog id doesn't exact-match a `eu.`-prefixed
        // request. The base key (`anthropic.claude-haiku-4-5-20251001-v1`) is indexed
        // alongside the full key at build time, so the provider-map substring fallback
        // in `lookup_with_provider` resolves it with the Bedrock-specific price.
        let json = r#"{
            "anthropic": {"models": {"claude-haiku-4-5-20251001": {"cost": {"input": 1.0}}}},
            "amazon-bedrock": {"models": {
                "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"cost": {"input": 1.0}}
            }}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        assert!(book
            .lookup_with_provider("bedrock", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
            .is_some());
    }

    #[test]
    fn strip_bedrock_region_prefix_covers_all_prefixes() {
        // Regression guard: all five Bedrock cross-region prefixes must be recognized.
        // A misspelling or accidental deletion would silently fall through to consensus
        // pricing for that region's traffic.
        for (input, expected_base) in [
            ("global.anthropic.claude-z", "anthropic.claude-z"),
            ("eu.anthropic.claude-z", "anthropic.claude-z"),
            ("us.anthropic.claude-z", "anthropic.claude-z"),
            ("au.anthropic.claude-z", "anthropic.claude-z"),
            ("jp.anthropic.claude-z", "anthropic.claude-z"),
            // Non-Bedrock id must pass through unchanged (no match → None).
            ("openai.gpt-4o", "openai.gpt-4o"),
        ] {
            let result = strip_bedrock_region_prefix(input);
            if expected_base == input {
                assert_eq!(result, None, "non-Bedrock id `{input}` should return None");
            } else {
                assert_eq!(
                    result,
                    Some(expected_base),
                    "prefix not stripped for `{input}`"
                );
            }
        }
    }

    #[test]
    fn lookup_with_provider_falls_to_consensus_when_provider_map_has_no_match() {
        // Provider map exists but the model isn't in it (neither exact nor substring):
        // must fall through to the consensus book rather than returning None.
        let json = r#"{
            "anthropic":      {"models": {"claude-haiku-4-5": {"cost": {"input": 1.0}}}},
            "amazon-bedrock": {"models": {"amazon.titan-text-express": {"cost": {"input": 0.13}}}}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        // Bedrock map exists but doesn't contain claude-haiku-4-5 → falls to consensus.
        let p = book
            .lookup_with_provider("bedrock", "claude-haiku-4-5")
            .expect("should fall through to consensus when provider map has no match");
        assert!((p.input - 1e-6).abs() < 1e-15, "got {}", p.input);
    }

    #[test]
    fn region_prefixed_bedrock_id_gets_bedrock_specific_price_not_consensus() {
        // A `eu.`-prefixed Bedrock request must resolve to the *Bedrock* rate, not
        // the Anthropic (consensus) rate. This guards the case where amazon-bedrock
        // prices a model differently from the direct Anthropic route.
        let json = r#"{
            "anthropic":      {"models": {"claude-z-20251001": {"cost": {"input": 3.0}}}},
            "amazon-bedrock": {"models": {
                "global.anthropic.claude-z-20251001-v1:0": {"cost": {"input": 3.6}}
            }}
        }"#;
        let book = PriceBook::from_models_dev_json(json);
        // Direct Anthropic request → anthropic rate ($3).
        let an = book
            .lookup_with_provider("anthropic", "claude-z-20251001")
            .unwrap();
        assert!(
            (an.input - 3e-6).abs() < 1e-15,
            "anthropic: got {}",
            an.input
        );
        // Region-prefixed Bedrock request → Bedrock rate ($3.6), not consensus.
        let bd = book
            .lookup_with_provider("bedrock", "eu.anthropic.claude-z-20251001-v1:0")
            .unwrap();
        assert!(
            (bd.input - 3.6e-6).abs() < 1e-15,
            "bedrock: got {}",
            bd.input
        );
    }

    #[test]
    fn vertex_at_date_id_resolves_to_models_dev_hyphen_form() {
        // Vertex Anthropic ids use `model@YYYYMMDD`; models.dev stores
        // `model-YYYYMMDD`. The `@`→`-` fold must bridge them so Vertex traffic is
        // priced (was silently USD 0).
        let json = r#"{"anthropic":{"models":{
            "claude-3-5-sonnet-20240620": {"cost":{"input":3.0,"output":15.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        let p = book
            .lookup("claude-3-5-sonnet@20240620")
            .expect("Vertex @-date id should resolve to the hyphen-date catalog entry");
        assert!((p.input - 3e-6).abs() < 1e-15);
    }

    #[test]
    fn normalize_lowercases_and_strips_version() {
        assert_eq!(normalize("  Claude-Haiku-4-5-v1:0 "), "claude-haiku-4-5-v1");
        assert_eq!(normalize("gpt-5-mini"), "gpt-5-mini");
    }

    #[test]
    fn vendored_snapshot_parses_and_prices_a_known_model() {
        // Guards the compiled-in models.dev snapshot: must be non-empty and
        // resolve a stable Anthropic model to a positive input price. Catches an
        // accidentally-truncated or schema-drifted vendored file at test time.
        let book = PriceBook::vendored();
        assert!(
            !book.is_empty(),
            "vendored snapshot parsed to an empty book"
        );
        let p = book
            .lookup("claude-haiku-4-5")
            .expect("vendored snapshot should price a stable Claude model");
        assert!(p.input > 0.0);
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
    fn skips_models_without_an_input_price_but_keeps_input_only_models() {
        let book = PriceBook::from_models_dev_json(SAMPLE);
        // No cost block / no input price → unusable, skipped.
        assert_eq!(book.lookup("no-cost"), None);
        assert_eq!(book.lookup("no-input"), None);
        // Input price but no output price → still valued (compression USD only
        // needs input); output defaults to 0.
        let p = book
            .lookup("no-output")
            .expect("input-only model should be priced");
        assert!((p.input - 1e-6).abs() < 1e-15);
        assert_eq!(p.output, 0.0);
    }

    #[test]
    fn negative_input_price_is_rejected_as_invalid() {
        // A negative input price would drive the (unclamped) compression USD total
        // negative — invalid data must be skipped → unpriced, not stored.
        let json = r#"{"p":{"models":{
            "bad":  {"cost":{"input":-1.0,"output":5.0}},
            "free": {"cost":{"input":0.0}},
            "good": {"cost":{"input":1.0,"cache_read":-0.5}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        assert_eq!(book.lookup("bad"), None, "negative input must be skipped");
        // 0.0 is valid (free model) and a negative cache field falls back to input.
        assert!(book.lookup("free").is_some());
        let good = book.lookup("good").unwrap();
        assert_eq!(
            good.cache_read, good.input,
            "negative cache_read falls back to input"
        );
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
    fn substring_fallback_rejects_short_generic_ids() {
        // A short generic stored id must not swallow an unrelated longer request.
        let json = r#"{"openai":{"models":{
            "gpt-4": {"cost":{"input":60.0,"output":120.0}},
            "claude-haiku-4-5": {"cost":{"input":1.0,"output":5.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        // Exact match still works for the bare family id.
        assert!(book.lookup("gpt-4").is_some());
        // An uncatalogued variant containing the short `gpt-4` misses (safe USD 0)
        // rather than mispricing to legacy gpt-4's $60/M.
        assert_eq!(book.lookup("gpt-4.5-preview"), None);
        assert_eq!(book.lookup("gpt-4-0613"), None);
        // A specific (>= 6-char) prefixed id still resolves via substring.
        assert!(book
            .lookup("eu.anthropic.claude-haiku-4-5-20251001-v1:0")
            .is_some());
    }

    #[test]
    fn substring_fallback_resolves_six_char_leaf_models_when_prefixed() {
        // A real 6-char leaf (gpt-4o) sent vendor-prefixed must still resolve —
        // the floor blocks <=5-char families, not specific leaves.
        let json = r#"{"openai":{"models":{
            "gpt-4o": {"cost":{"input":2.5,"output":10.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        let direct = book.lookup("gpt-4o").unwrap();
        let prefixed = book.lookup("copilot-gpt-4o").unwrap();
        assert_eq!(direct, prefixed);
    }

    #[test]
    fn substring_fallback_breaks_length_ties_deterministically() {
        // Two equal-length stored ids both contained in the request must resolve
        // deterministically (by id string), not by HashMap iteration order.
        let json = r#"{"p":{"models":{
            "aaaaaaa-x": {"cost":{"input":1.0}},
            "bbbbbbb-x": {"cost":{"input":2.0}}
        }}}"#;
        let book = PriceBook::from_models_dev_json(json);
        // Both ids are length 9 and contained; the lexicographically-greater id
        // ("bbbbbbb-x") wins every run → input 2.0/M = 2e-6 per token.
        let p = book.lookup("zzz-aaaaaaa-x-bbbbbbb-x").unwrap();
        assert!((p.input - 2e-6).abs() < 1e-15);
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

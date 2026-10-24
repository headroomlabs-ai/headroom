//! Per-token USD pricing — Phase H. Used to value spend and savings.
//!
//! Reads the LiteLLM price table already vendored for
//! [`crate::compression::model_limits`] — one embedded copy, two
//! consumers. No new data file, and no startup network dependency:
//! the table ships inside the binary and parses lazily on first
//! lookup.
//!
//! # Lookup discipline
//!
//! Lookup is exact-match over a small, deterministic candidate list
//! derived from the request's model id (lowercased, then: geo prefix
//! stripped, `:rev` suffix trimmed, provider path segment dropped).
//! We deliberately do NOT substring-scan the table: a short or
//! generic id must miss (and be logged) rather than misprice against
//! whichever stored id happens to contain it.
//!
//! **The unmodified id is always tried first**, because Bedrock
//! cross-region pricing is genuinely region-specific — e.g.
//! `eu.anthropic.claude-3-5-haiku-20241022-v1:0` bills at $0.25/M
//! against `anthropic.claude-3-5-haiku-20241022-v1:0`'s $0.80/M. Any
//! region the table tracks therefore gets its true price. The
//! geo-stripped candidate is only a *fallback* for a (region, model)
//! pair the table does not list, where the base price is a far
//! better estimate than $0 — approximate, but never silently
//! preferred over an exact regional entry.
//!
//! # Coverage
//!
//! The vendored snapshot tracks current model generations plus every
//! Bedrock/Vertex-prefixed id, but it does not carry every legacy
//! direct-API id (`claude-3-5-sonnet-20241022`, for one, is absent
//! while its Bedrock form is present). Those price at $0 with a WARN
//! until the table is refreshed — see the module note below.
//!
//! # Unknown models price at $0
//!
//! A miss returns `None`; the ledger then records $0 for every USD
//! field and this module logs one WARN naming the model. We
//! deliberately do NOT substitute a blended fallback rate: phantom
//! dollars on a cost dashboard are worse than an obvious zero, and
//! the WARN tells an operator exactly which id to add by refreshing
//! the vendored table (`scripts/refresh_model_limits.sh`).

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::bedrock::vendor::strip_geo_prefix;
use crate::compression::model_limits::VENDORED_JSON;

/// Per-token USD prices for one model. All fields are $/token
/// (LiteLLM stores them that way natively — no per-million
/// conversion happens here).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl ModelPrice {
    /// USD saved by removing `tokens_saved` input tokens before they
    /// were sent (compression savings).
    pub fn compression_savings_usd(&self, tokens_saved: u64) -> f64 {
        tokens_saved as f64 * self.input
    }

    /// NET USD saved by prompt caching for one request: the read
    /// discount (`reads × (input − cache_read)`) minus the write
    /// premium (`writes × (cache_write − input)` — cache writes bill
    /// ABOVE list price, typically 1.25×). Reporting the gross read
    /// discount alone overstates the benefit on write-heavy traffic,
    /// which is exactly the phantom-dollars failure mode this module
    /// exists to avoid. Floored at $0 per request: a warm-up turn
    /// that only writes is an investment, not negative savings —
    /// its real cost is already carried by [`Self::input_cost_usd`].
    ///
    /// Known bias, stated because it's the honest thing to state:
    /// the ledger sums these per-request values, and a floored
    /// request contributes 0 rather than its true negative, so
    /// `sum(max(0, net))` ≥ `max(0, sum(net))`. Aggregate cache
    /// savings therefore skew slightly high. The gap is one write
    /// premium per floored request, which in practice means the
    /// first turn of a session: once a prefix is cached, later turns
    /// read far more than they write and net positive well clear of
    /// the floor (100k reads + 5k writes on Sonnet nets +$0.266, vs
    /// a $0.004 premium). Fixing it properly means accumulating a
    /// signed net and flooring once at the aggregate — worth doing
    /// if cache-heavy traffic ever makes the skew material, but not
    /// worth a signed accumulator (and a persisted-negative
    /// migration) for single-digit thousandths of a cent today.
    pub fn cache_savings_usd(&self, cache_read_tokens: u64, cache_write_tokens: u64) -> f64 {
        let read_discount = (self.input - self.cache_read).max(0.0);
        let write_premium = (self.cache_write - self.input).max(0.0);
        let net =
            cache_read_tokens as f64 * read_discount - cache_write_tokens as f64 * write_premium;
        net.max(0.0)
    }

    /// USD spent on generated output for a request. Output bills at
    /// its own rate — typically 4-5x input — so a "spend" figure that
    /// omits it understates the bill by more than it reports on a
    /// short-prompt/long-answer turn.
    pub fn output_cost_usd(&self, output_tokens: u64) -> f64 {
        output_tokens as f64 * self.output
    }

    /// USD actually spent on input for a request, using the cache
    /// breakdown when any segment is non-zero (never adding the
    /// total on top of the breakdown — that would double-count).
    pub fn input_cost_usd(
        &self,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) -> f64 {
        uncached_input_tokens as f64 * self.input
            + cache_read_tokens as f64 * self.cache_read
            + cache_write_tokens as f64 * self.cache_write
    }
}

static BOOK: OnceLock<HashMap<String, ModelPrice>> = OnceLock::new();

/// Bounded once-per-model warn set so an unknown model logs one WARN,
/// not one per request. Capped so attacker-controlled model ids can't
/// grow it without bound; once full, further unknown models simply
/// stop logging (they still price at $0). RwLock so the steady-state
/// path for an already-known-unknown model (every miss after the
/// first) is a shared read, not serialized writes on the record path.
static WARNED: OnceLock<RwLock<std::collections::HashSet<String>>> = OnceLock::new();
const WARNED_CAP: usize = 256;

/// Parse the vendored table now instead of on the first recorded
/// request. Called once at startup (main.rs) so the ~1.4 MB JSON
/// parse never stalls tokio workers when a burst of first requests
/// lands together.
pub fn warm() {
    let _ = book();
}

fn book() -> &'static HashMap<String, ModelPrice> {
    BOOK.get_or_init(parse_vendored)
}

fn parse_vendored() -> HashMap<String, ModelPrice> {
    let raw: serde_json::Value = serde_json::from_str(VENDORED_JSON)
        .expect("vendored LiteLLM JSON must parse — same invariant as model_limits");
    let mut out = HashMap::new();
    let Some(obj) = raw.as_object() else {
        return out;
    };
    for (model_id, spec) in obj {
        if model_id == "sample_spec" {
            continue;
        }
        let Some(spec) = spec.as_object() else {
            continue;
        };
        let input = spec.get("input_cost_per_token").and_then(|v| v.as_f64());
        let output = spec.get("output_cost_per_token").and_then(|v| v.as_f64());
        let (Some(input), Some(output)) = (input, output) else {
            continue;
        };
        if !input.is_finite() || !output.is_finite() || input < 0.0 || output < 0.0 {
            continue;
        }
        let cache_read = spec
            .get("cache_read_input_token_cost")
            .and_then(|v| v.as_f64())
            .filter(|c| c.is_finite() && *c >= 0.0)
            .unwrap_or(input);
        let cache_write = spec
            .get("cache_creation_input_token_cost")
            .and_then(|v| v.as_f64())
            .filter(|c| c.is_finite() && *c >= 0.0)
            .unwrap_or(input);
        out.insert(
            model_id.to_ascii_lowercase(),
            ModelPrice {
                input,
                output,
                cache_read,
                cache_write,
            },
        );
    }
    out
}

/// Candidate keys tried, in order, for a request model id. Exact
/// matches only — see module doc for why there is no substring scan.
fn candidates(model: &str) -> Vec<String> {
    let lower = model.trim().to_ascii_lowercase();
    let mut out = Vec::with_capacity(6);
    let mut push = |s: String| {
        if !s.is_empty() && !out.contains(&s) {
            out.push(s);
        }
    };
    push(lower.clone());
    // Bedrock cross-region profile: `eu.anthropic.claude…` falls back
    // to `anthropic.claude…` when the region itself isn't in the
    // table. GEO_PREFIXES (shared with the compression vendor gate)
    // covers every geo the vendored table fronts, `us-gov.` included.
    let geo = strip_geo_prefix(&lower).to_string();
    push(geo.clone());
    // Bedrock revision suffix: `…-v1:0`. ONLY a numeric suffix is
    // trimmed. A non-numeric colon suffix is a routing/tier variant
    // (`:exacto`, `:free`, `:nitro`) that can genuinely bill at its
    // own rate — the vendored table lists
    // `openrouter/z-ai/glm-4.6` at $0.40/M and its `:exacto` variant
    // at $0.45/M. Trimming those would hand an uncatalogued variant
    // its sibling's price with no warning: a confident wrong number,
    // which this module exists to avoid. They miss and WARN instead.
    push_revision_trimmed(&mut push, &geo);
    push_revision_trimmed(&mut push, &lower);
    // Provider-routed ids (`openai/gpt-4o`, `github-copilot/claude…`):
    // the bare tail is how LiteLLM stores first-party models. Try the
    // tail both verbatim and `:rev`-trimmed, so a combined
    // `vendor/model:rev` id can still reach a fully-stripped entry.
    if let Some((_, tail)) = lower.rsplit_once('/') {
        push(tail.to_string());
        push_revision_trimmed(&mut push, tail);
    }
    out
}

/// Push `id` minus a trailing `:<digits>` revision, if it has one.
fn push_revision_trimmed(push: &mut impl FnMut(String), id: &str) {
    if let Some((head, rev)) = id.rsplit_once(':') {
        if !rev.is_empty() && rev.bytes().all(|b| b.is_ascii_digit()) {
            push(head.to_string());
        }
    }
}

/// Look up per-token pricing for a model id. `None` means "not in
/// the vendored table" — the caller records $0 for USD fields and
/// this module logs one WARN per distinct model id.
pub fn lookup(model: &str) -> Option<ModelPrice> {
    let table = book();
    for key in candidates(model) {
        if let Some(p) = table.get(&key) {
            return Some(*p);
        }
    }
    warn_once(model);
    None
}

fn warn_once(model: &str) {
    let set = WARNED.get_or_init(|| RwLock::new(std::collections::HashSet::new()));
    {
        let read = set
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if read.len() >= WARNED_CAP || read.contains(model) {
            return;
        }
    }
    let mut write = set
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if write.len() >= WARNED_CAP || !write.insert(model.to_string()) {
        return;
    }
    drop(write);
    tracing::warn!(
        event = "stats_price_unknown_model",
        model = %model,
        "model id not in the vendored price table; both spend AND savings \
         for it record as $0 — refresh data/model_prices_and_context_window.json \
         via scripts/refresh_model_limits.sh to add it"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_anthropic_direct_id_prices() {
        let p = lookup("claude-sonnet-4-5-20250929").expect("priced");
        assert!(p.input > 0.0 && p.output > p.input);
        assert!(p.cache_read < p.input, "cache reads are discounted");
        assert!(p.cache_write > p.input, "cache writes carry a premium");
    }

    /// Bedrock cross-region pricing is region-specific, so an exact
    /// regional entry must win over the geo-stripped base id.
    #[test]
    fn exact_regional_bedrock_entry_wins_over_base() {
        let base = lookup("anthropic.claude-3-5-haiku-20241022-v1:0").expect("base priced");
        let eu = lookup("eu.anthropic.claude-3-5-haiku-20241022-v1:0").expect("eu priced");
        assert!(
            (base.input - 8e-7).abs() < 1e-12,
            "base id prices at list rate, got {}",
            base.input
        );
        assert!(
            (eu.input - 2.5e-7).abs() < 1e-12,
            "eu id must use its own regional rate, got {}",
            eu.input
        );
        assert_ne!(
            base, eu,
            "geo-stripping must not clobber a tracked regional price"
        );
    }

    /// A region the table does NOT list falls back to the base id —
    /// approximate, but far better than $0.
    /// Every published geo prefix must fall back to the base entry
    /// when the table doesn't carry that exact (region, model) pair.
    /// Table-driven over GEO_PREFIXES itself, so adding a prefix
    /// without its pricing fallback fails here — the `au.`/`jp.`/
    /// `us-gov.` gap this PR fixes was exactly that.
    #[test]
    fn every_geo_prefix_falls_back_to_base_price() {
        const BASE: &str = "anthropic.claude-3-5-haiku-20241022-v1:0";
        let base = lookup(BASE).expect("base priced");
        for geo in crate::bedrock::vendor::GEO_PREFIXES {
            let id = format!("{geo}{BASE}");
            if book().get(&id).is_some() {
                continue; // table lists this region explicitly — priced on its own terms
            }
            assert_eq!(
                lookup(&id).unwrap_or_else(|| panic!("{id} must fall back")),
                base,
                "{geo} must strip to the base entry when unlisted"
            );
        }
    }

    #[test]
    fn provider_path_segment_falls_back_to_bare_tail() {
        let bare = lookup("gpt-4o").expect("bare id priced");
        let routed = lookup("openai/gpt-4o").expect("routed id priced");
        assert_eq!(bare, routed);
    }

    #[test]
    fn unknown_model_misses_instead_of_mispricing() {
        assert!(lookup("claude").is_none(), "generic id must not match");
        assert!(lookup("totally-unknown-model-xyz").is_none());
        assert!(lookup("").is_none());
    }

    #[test]
    fn case_insensitive_lookup() {
        assert_eq!(
            lookup("GPT-4o").expect("priced"),
            lookup("gpt-4o").expect("priced")
        );
    }

    /// A combined `vendor/model:rev` id must reach a table entry
    /// stored fully stripped — the slash-tail is also tried with its
    /// `:rev` suffix trimmed.
    #[test]
    fn provider_path_and_revision_compose() {
        let bare = lookup("gpt-4o").expect("bare id priced");
        let combined = lookup("openai/gpt-4o:2").expect("combined form priced");
        assert_eq!(bare, combined, "provider path + numeric revision compose");
    }

    /// A non-numeric colon suffix is a pricing-relevant variant, not a
    /// revision. It must NOT inherit the bare model's rate: the table
    /// prices `openrouter/z-ai/glm-4.6` at $0.40/M and `:exacto` at
    /// $0.45/M, so borrowing across that boundary invents dollars.
    #[test]
    fn variant_suffix_never_borrows_the_bare_models_price() {
        let bare = lookup("openrouter/z-ai/glm-4.6").expect("bare variant priced");
        let exacto = lookup("openrouter/z-ai/glm-4.6:exacto").expect("listed variant priced");
        assert_ne!(bare.input, exacto.input, "fixture guard: these must differ");
        // An UNLISTED variant must miss loudly, not silently take the
        // bare price.
        assert!(
            lookup("openrouter/z-ai/glm-4.6:priority").is_none(),
            "an uncatalogued variant must record $0 + WARN, never a guessed rate"
        );
    }

    #[test]
    fn usd_helpers_compute_list_price_math() {
        let p = ModelPrice {
            input: 3e-6,
            output: 15e-6,
            cache_read: 3e-7,
            cache_write: 3.75e-6,
        };
        assert!((p.compression_savings_usd(1_000_000) - 3.0).abs() < 1e-9);
        // Cache savings = read discount, NET of the write premium.
        assert!((p.cache_savings_usd(1_000_000, 0) - 2.7).abs() < 1e-9);
        // 1M reads save $2.70; 1M writes cost a $0.75 premium → net $1.95.
        assert!((p.cache_savings_usd(1_000_000, 1_000_000) - 1.95).abs() < 1e-9);
        // Write-only warm-up floors at $0, never negative savings
        // (the premium itself is billed via input_cost_usd).
        assert_eq!(p.cache_savings_usd(0, 1_000_000), 0.0);
        // Breakdown input cost: uncached + cache_read + cache_write.
        let usd = p.input_cost_usd(1_000_000, 1_000_000, 1_000_000);
        assert!((usd - (3.0 + 0.3 + 3.75)).abs() < 1e-9);
        // Inverted cache pricing clamps to $0, never negative savings.
        let inverted = ModelPrice {
            input: 1e-6,
            output: 1e-6,
            cache_read: 2e-6,
            cache_write: 1e-6,
        };
        assert_eq!(inverted.cache_savings_usd(1000, 0), 0.0);
    }
}

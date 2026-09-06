//! Canonical Bedrock vendor resolution.
//!
//! Bedrock model IDs are `<vendor>.<model>-<date>-<rev>` (e.g.
//! `anthropic.claude-3-haiku-20240307-v1:0`). Cross-region inference
//! profiles prepend a geo routing token before the vendor
//! (`eu.anthropic.…`, `us.anthropic.…`, `apac.anthropic.…`,
//! `global.anthropic.…`). Matching the bare `anthropic.` prefix alone
//! silently skips compression for those profiles, so we strip a known
//! geo prefix first, then take the leading dot-segment as the vendor.
//! Literal matching only — no regexes (project rule).

/// Cross-region inference-profile routing prefixes AWS prepends to the
/// vendor segment. Stripped (once) before vendor resolution. Kept to a
/// closed, known set so an unrelated `something.anthropic.x` id is not
/// mistaken for an Anthropic inference profile.
///
/// The set is the prefixes present in the vendored LiteLLM price
/// table (`data/model_prices_and_context_window.json`) — every one of
/// these fronts real `<geo>.anthropic.…` ids there. A missing entry
/// is not cosmetic: an unrecognized geo prefix makes
/// `is_anthropic_model_id` return false, silently skipping
/// compression for that region's inference profiles (`au.`/`jp.`/
/// `us-gov.` were missing until review of the stats work caught it).
pub(crate) const GEO_PREFIXES: [&str; 7] =
    ["eu.", "us.", "us-gov.", "apac.", "au.", "jp.", "global."];

/// Strip a cross-region inference-profile geo prefix, if present.
/// `eu.anthropic.claude-…` → `anthropic.claude-…`; ids without a
/// known geo prefix return unchanged. Shared with
/// `observability::pricing` so price lookups see the same canonical
/// id the vendor gate sees.
pub(crate) fn strip_geo_prefix(model_id: &str) -> &str {
    GEO_PREFIXES
        .iter()
        .find_map(|p| model_id.strip_prefix(p))
        .unwrap_or(model_id)
}

/// Resolve the canonical vendor of a Bedrock model id, stripping a
/// cross-region inference-profile geo prefix first.
///
/// `eu.anthropic.claude-…` → `anthropic`; `amazon.titan-…` → `amazon`;
/// `global.amazon.nova-…` → `amazon`.
pub fn canonical_vendor(model_id: &str) -> &str {
    let stripped = strip_geo_prefix(model_id);
    stripped.split('.').next().unwrap_or(stripped)
}

/// Whether the model id is an Anthropic-shape model — a foundation
/// model (`anthropic.…`) or a cross-region inference profile
/// (`<geo>.anthropic.…`) — i.e. eligible for the live-zone Anthropic
/// compression + envelope pipeline.
pub fn is_anthropic_model_id(model_id: &str) -> bool {
    canonical_vendor(model_id) == "anthropic"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_vendor_strips_known_cross_region_prefix() {
        assert_eq!(
            canonical_vendor("anthropic.claude-3-haiku-20240307-v1:0"),
            "anthropic"
        );
        assert_eq!(
            canonical_vendor("eu.anthropic.claude-haiku-4-5-20251001-v1:0"),
            "anthropic"
        );
        assert_eq!(canonical_vendor("amazon.titan-text-express-v1"), "amazon");
        assert_eq!(canonical_vendor("global.amazon.nova-lite-v1:0"), "amazon");
        // Unknown leading token is NOT stripped — stays as its own vendor.
        assert_eq!(canonical_vendor("random.anthropic.x"), "random");
    }

    #[test]
    fn anthropic_model_id_matches_foundation_and_inference_profiles() {
        assert!(is_anthropic_model_id(
            "anthropic.claude-3-haiku-20240307-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
        assert!(is_anthropic_model_id(
            "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        ));
        // au./jp./us-gov. are real AWS geo prefixes (present in the
        // vendored price table); they must gate as Anthropic too.
        assert!(is_anthropic_model_id(
            "au.anthropic.claude-sonnet-4-5-20250929-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
        ));
        // Non-Anthropic vendors, including geo-prefixed, stay false.
        assert!(!is_anthropic_model_id("amazon.titan-text-express-v1"));
        assert!(!is_anthropic_model_id("meta.llama3-70b-instruct-v1:0"));
        assert!(!is_anthropic_model_id("eu.amazon.nova-lite-v1:0"));
        assert!(!is_anthropic_model_id("mistral.voxtral-mini-3b-2507"));
    }
}

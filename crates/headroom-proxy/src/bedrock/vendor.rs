//! Canonical Bedrock vendor resolution.
//!
//! Bedrock model IDs are `<vendor>.<model>-<date>-<rev>` (e.g.
//! `anthropic.claude-3-haiku-20240307-v1:0`). Cross-region inference
//! profiles prepend a geo routing token before the vendor
//! (`eu.anthropic.…`, `us.anthropic.…`, `au.anthropic.…`,
//! `global.anthropic.…`). Matching the bare `anthropic.` prefix alone
//! silently skips compression for those profiles, so we strip a known
//! geo prefix first, then take the leading dot-segment as the vendor.
//! Literal matching only — no regexes (project rule).

/// Cross-region inference-profile routing prefixes AWS prepends to the
/// vendor segment. Stripped (once) before vendor resolution. Kept to a
/// closed, known set so an unrelated `something.anthropic.x` id is not
/// mistaken for an Anthropic inference profile.
pub(crate) const GEO_PREFIXES: [&str; 5] = ["eu.", "us.", "au.", "jp.", "global."];

/// Resolve the canonical vendor of a Bedrock model id, stripping a
/// cross-region inference-profile geo prefix first.
///
/// `eu.anthropic.claude-…` → `anthropic`; `amazon.titan-…` → `amazon`;
/// `global.amazon.nova-…` → `amazon`.
pub fn canonical_vendor(model_id: &str) -> &str {
    let stripped = GEO_PREFIXES
        .iter()
        .find_map(|p| model_id.strip_prefix(p))
        .unwrap_or(model_id);
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
            "au.anthropic.claude-3-5-sonnet-20240620-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
        ));
        assert!(is_anthropic_model_id(
            "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        ));
        // Non-Anthropic vendors, including geo-prefixed, stay false.
        assert!(!is_anthropic_model_id("amazon.titan-text-express-v1"));
        assert!(!is_anthropic_model_id("meta.llama3-70b-instruct-v1:0"));
        assert!(!is_anthropic_model_id("eu.amazon.nova-lite-v1:0"));
        assert!(!is_anthropic_model_id("mistral.voxtral-mini-3b-2507"));
    }
}

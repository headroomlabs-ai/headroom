//! Parsing contract of the `HEADROOM_LIVE_ZONE_DISABLE_ARMS` kill
//! switch, tested against the pure `parse_disabled_arms` rather than
//! through the process environment.
//!
//! The end-to-end wiring (env var → latched set → dispatcher no-op)
//! lives in `live_zone_disable_arms.rs`, which needs `std::env::set_var`
//! and therefore a dedicated single-test binary. Everything about how a
//! string becomes a set of arms is checkable here instead: no global
//! state, no isolation requirement, and these cases can run alongside
//! anything else.

use headroom_core::transforms::live_zone::parse_disabled_arms;
use headroom_core::transforms::ContentType;

#[test]
fn parses_both_spellings_and_trims_whitespace() {
    let parsed = parse_disabled_arms("source_code, plain_text");
    assert!(parsed.contains(&ContentType::SourceCode));
    assert!(parsed.contains(&ContentType::PlainText));
    assert_eq!(parsed.len(), 2);

    // `as_str()` tags parse to the same variants as the natural names.
    let tags = parse_disabled_arms("source_code,text");
    assert_eq!(parsed, tags, "both spellings must yield the same set");
}

#[test]
fn every_variant_can_be_named() {
    for content_type in ContentType::ALL {
        for spelling in [content_type.as_str(), content_type.natural_name()] {
            let parsed = parse_disabled_arms(spelling);
            assert!(
                parsed.contains(&content_type),
                "{spelling:?} must disable {content_type:?}"
            );
        }
    }
}

#[test]
fn unknown_and_blank_tokens_are_ignored_not_fatal() {
    // A typo in an operator's rollback switch must not take the proxy
    // down, and must not silently disable something else either.
    let parsed = parse_disabled_arms("bogus_type, , source_code,,   ");
    assert_eq!(
        parsed,
        std::iter::once(ContentType::SourceCode).collect(),
        "unknown and empty tokens drop out; valid ones survive"
    );
}

#[test]
fn empty_input_disables_nothing() {
    assert!(parse_disabled_arms("").is_empty());
    assert!(parse_disabled_arms("   ").is_empty());
    assert!(parse_disabled_arms(",,,").is_empty());
}

#[test]
fn repeated_tokens_collapse() {
    let parsed = parse_disabled_arms("plain_text,text,plain_text");
    assert_eq!(parsed.len(), 1);
    assert!(parsed.contains(&ContentType::PlainText));
}

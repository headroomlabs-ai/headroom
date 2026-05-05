//! Cache-control floor derivation helper.
//!
//! Phase B PR-B2 retired the passthrough stub `compress_anthropic_request`
//! that lived here through Phase A; it now lives in
//! [`super::live_zone_anthropic`] alongside the live-zone dispatcher.
//!
//! [`resolve_frozen_count`] stays in this module because it is the
//! cache-control-policy boundary used by both the live-zone path and
//! any future per-provider compressors that want to honour the same
//! gate.

use serde_json::Value;

use crate::config::CacheControlAutoFrozen;

/// Resolve the `frozen_message_count` floor for a parsed Anthropic
/// `/v1/messages` request body, honouring the
/// `cache_control_auto_frozen` config gate (PR-A4).
///
/// This is a thin wrapper around [`headroom_core::compute_frozen_count`]
/// that returns `0` when the operator has disabled automatic
/// derivation, regardless of the markers in `parsed`.
///
/// # Arguments
///
/// - `parsed`: parsed JSON body. The walker reads `messages`,
///   `system`, and `tools`; other fields are ignored. The function
///   itself does NOT mutate the value.
/// - `policy`: the resolved [`CacheControlAutoFrozen`] from `Config`.
///   `Disabled` short-circuits to `0` without inspecting the body.
/// - `request_id`: the per-request id used for log correlation,
///   matched against the proxy's existing `tracing` span fields.
///
/// # Returns
///
/// The frozen-count floor (smallest `N` such that `messages[i]` for
/// `i < N` is in the cache hot zone), or `0` when auto-derivation
/// is disabled. Phase B PR-B2's live-zone dispatcher refuses to
/// touch any index below this value.
pub fn resolve_frozen_count(
    parsed: &Value,
    policy: CacheControlAutoFrozen,
    request_id: &str,
) -> usize {
    if !policy.is_enabled() {
        tracing::debug!(
            request_id = %request_id,
            cache_control_auto_frozen = policy.as_str(),
            "cache_control auto-derivation disabled; floor=0"
        );
        return 0;
    }
    let count = headroom_core::compute_frozen_count(parsed);
    tracing::debug!(
        request_id = %request_id,
        cache_control_auto_frozen = policy.as_str(),
        frozen_count = count,
        "cache_control auto-derivation result"
    );
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn disabled_policy_yields_zero_regardless_of_markers() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        });
        assert_eq!(
            resolve_frozen_count(&body, CacheControlAutoFrozen::Disabled, "rid"),
            0
        );
    }

    #[test]
    fn enabled_policy_walks_to_compute_count() {
        // No markers → count is 0 even with policy enabled.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(
            resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "rid"),
            0
        );
    }
}

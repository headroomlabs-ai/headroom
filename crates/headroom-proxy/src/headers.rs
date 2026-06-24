//! Header filtering and X-Forwarded-* injection.
//!
//! Hop-by-hop headers per RFC 7230 §6.1 must not be forwarded.

use http::header::{HeaderMap, HeaderName, HeaderValue};
use std::net::IpAddr;

/// Hop-by-hop header names that must be stripped (RFC 7230 §6.1).
/// Note: `Upgrade` is hop-by-hop in general, but for WebSocket the upgrade is
/// handled by the websocket module (axum's WebSocketUpgrade extracts it before
/// we ever try to forward it as plain HTTP).
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

/// Some additional headers that are typically managed by the HTTP client and
/// must not be copied across (reqwest/hyper sets them itself).
const CLIENT_MANAGED: &[&str] = &["host", "content-length"];

/// Internal-header prefix dropped from upstream-bound requests when
/// `Config::strip_internal_headers == StripInternalHeaders::Enabled` (PR-A5,
/// fixes P5-49). Case-insensitive prefix match. The Rust path mirrors the
/// Python `_strip_internal_headers` helper.
///
/// Response-side `X-Headroom-*` injection (e.g. `x-headroom-tokens-saved`)
/// is intentionally untouched — that direction is the proxy describing its
/// own work to the client and never crosses an upstream boundary.
pub const INTERNAL_HEADER_PREFIX: &str = "x-headroom-";

/// Returns true if `name` is hop-by-hop and must be stripped.
pub fn is_hop_by_hop(name: &HeaderName) -> bool {
    let n = name.as_str();
    HOP_BY_HOP.iter().any(|h| h.eq_ignore_ascii_case(n))
}

/// Headers we additionally drop before forwarding the request to the upstream
/// (Host is rebuilt by reqwest for the upstream URL; Content-Length recomputed).
pub fn is_request_drop(name: &HeaderName) -> bool {
    if is_hop_by_hop(name) {
        return true;
    }
    let n = name.as_str();
    CLIENT_MANAGED.iter().any(|h| h.eq_ignore_ascii_case(n))
}

/// Returns true when `name` matches the internal `x-headroom-*` prefix
/// (case-insensitive). Pure function, no regex.
pub fn is_internal_header(name: &HeaderName) -> bool {
    name.as_str()
        .to_ascii_lowercase()
        .starts_with(INTERNAL_HEADER_PREFIX)
}

/// Headers we drop on the response side. Same hop-by-hop set; we don't touch
/// content-length since the response body length is known and we want clients
/// to see it.
pub fn is_response_drop(name: &HeaderName) -> bool {
    is_hop_by_hop(name)
}

/// Headers listed inside Connection: must be stripped too. Returns the lower-cased names.
pub fn connection_listed_headers(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(http::header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Append `addr` to an existing X-Forwarded-For header, or set it.
pub fn append_xff(headers: &mut HeaderMap, addr: IpAddr) {
    let xff = HeaderName::from_static("x-forwarded-for");
    let new_value = match headers.get(&xff) {
        Some(existing) => match existing.to_str() {
            Ok(s) => format!("{s}, {addr}"),
            Err(_) => addr.to_string(),
        },
        None => addr.to_string(),
    };
    if let Ok(v) = HeaderValue::from_str(&new_value) {
        headers.insert(xff, v);
    }
}

/// Set `name` to `value`, replacing any prior value.
pub fn set_single(headers: &mut HeaderMap, name: HeaderName, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        headers.insert(name, v);
    }
}

/// Remove every header whose name starts with `INTERNAL_HEADER_PREFIX`
/// (case-insensitive). Mutates in place. Returns the number of header
/// entries removed for structured logging.
pub fn strip_internal_headers(headers: &mut HeaderMap) -> usize {
    let to_remove: Vec<HeaderName> = headers
        .keys()
        .filter(|n| is_internal_header(n))
        .cloned()
        .collect();
    let mut removed = 0usize;
    for name in to_remove {
        // `remove` returns the first value; multi-valued internal headers are
        // not expected in practice but we drain them all for safety.
        while headers.remove(&name).is_some() {
            removed += 1;
        }
    }
    removed
}

/// Build a fresh HeaderMap suitable for forwarding to the upstream:
///   - hop-by-hop and connection-listed headers stripped
///   - Host/Content-Length removed (rebuilt by client)
///   - X-Forwarded-For appended
///   - X-Forwarded-Proto, X-Forwarded-Host set
///   - X-Request-Id ensured
///   - When `strip_internal == true`, `x-headroom-*` headers stripped
///     (PR-A5, fixes P5-49). Operators can disable via
///     `HEADROOM_PROXY_STRIP_INTERNAL_HEADERS=disabled` for diagnostic
///     shadow tracing.
pub fn build_forward_request_headers(
    incoming: &HeaderMap,
    client_addr: IpAddr,
    forwarded_proto: &str,
    forwarded_host: Option<&str>,
    request_id: &str,
    strip_internal: bool,
) -> HeaderMap {
    let connection_listed = connection_listed_headers(incoming);
    let mut out = HeaderMap::new();
    for (name, value) in incoming.iter() {
        if is_request_drop(name) {
            continue;
        }
        if connection_listed.iter().any(|h| h == name.as_str()) {
            continue;
        }
        if strip_internal && is_internal_header(name) {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    append_xff(&mut out, client_addr);
    set_single(
        &mut out,
        HeaderName::from_static("x-forwarded-proto"),
        forwarded_proto,
    );
    if let Some(host) = forwarded_host {
        set_single(&mut out, HeaderName::from_static("x-forwarded-host"), host);
    }
    set_single(
        &mut out,
        HeaderName::from_static("x-request-id"),
        request_id,
    );
    out
}

/// Filter the upstream response headers before passing to the client.
pub fn filter_response_headers(incoming: &HeaderMap) -> HeaderMap {
    let connection_listed = connection_listed_headers(incoming);
    let mut out = HeaderMap::new();
    for (name, value) in incoming.iter() {
        if is_response_drop(name) {
            continue;
        }
        if connection_listed.iter().any(|h| h == name.as_str()) {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::header::HeaderValue;

    #[test]
    fn hop_by_hop_detection() {
        assert!(is_hop_by_hop(&HeaderName::from_static("connection")));
        assert!(is_hop_by_hop(&HeaderName::from_static("transfer-encoding")));
        assert!(is_hop_by_hop(&HeaderName::from_static("upgrade")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("authorization")));
    }

    #[test]
    fn xff_appends() {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
        append_xff(&mut h, "5.6.7.8".parse().unwrap());
        assert_eq!(h.get("x-forwarded-for").unwrap(), "1.2.3.4, 5.6.7.8");
    }

    #[test]
    fn connection_listed_strip() {
        let mut h = HeaderMap::new();
        h.insert("connection", HeaderValue::from_static("close, x-foo"));
        h.insert("x-foo", HeaderValue::from_static("bar"));
        h.insert("x-keep", HeaderValue::from_static("yes"));
        let listed = connection_listed_headers(&h);
        assert!(listed.contains(&"x-foo".to_string()));
        assert!(listed.contains(&"close".to_string()));
    }

    #[test]
    fn internal_header_detected_case_insensitive() {
        assert!(is_internal_header(&HeaderName::from_static(
            "x-headroom-bypass"
        )));
        // HeaderName normalizes to lowercase internally, so any cased input
        // ends up matching the lowercase prefix.
        assert!(is_internal_header(
            &HeaderName::from_bytes(b"X-Headroom-Mode").unwrap()
        ));
        assert!(is_internal_header(
            &HeaderName::from_bytes(b"X-HEADROOM-FOO").unwrap()
        ));
        assert!(!is_internal_header(&HeaderName::from_static(
            "x-request-id"
        )));
        assert!(!is_internal_header(&HeaderName::from_static(
            "authorization"
        )));
    }

    #[test]
    fn strip_internal_headers_removes_only_internal_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer x"));
        h.insert("x-headroom-bypass", HeaderValue::from_static("true"));
        h.insert("x-headroom-mode", HeaderValue::from_static("passthrough"));
        h.insert("x-request-id", HeaderValue::from_static("req-1"));
        let removed = strip_internal_headers(&mut h);
        assert_eq!(removed, 2);
        assert!(h.get("x-headroom-bypass").is_none());
        assert!(h.get("x-headroom-mode").is_none());
        assert!(h.get("authorization").is_some());
        assert!(h.get("x-request-id").is_some());
    }

    #[test]
    fn build_forward_strips_internal_when_enabled() {
        let mut incoming = HeaderMap::new();
        incoming.insert("authorization", HeaderValue::from_static("Bearer x"));
        incoming.insert("x-headroom-bypass", HeaderValue::from_static("true"));
        let out = build_forward_request_headers(
            &incoming,
            "127.0.0.1".parse().unwrap(),
            "http",
            Some("h"),
            "req-1",
            true,
        );
        assert!(out.get("authorization").is_some());
        assert!(out.get("x-headroom-bypass").is_none());
    }

    #[test]
    fn build_forward_keeps_internal_when_disabled() {
        let mut incoming = HeaderMap::new();
        incoming.insert("authorization", HeaderValue::from_static("Bearer x"));
        incoming.insert("x-headroom-bypass", HeaderValue::from_static("true"));
        let out = build_forward_request_headers(
            &incoming,
            "127.0.0.1".parse().unwrap(),
            "http",
            Some("h"),
            "req-1",
            false,
        );
        assert!(out.get("authorization").is_some());
        assert!(out.get("x-headroom-bypass").is_some());
    }
}

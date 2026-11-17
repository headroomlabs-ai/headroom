//! Inbound `HEADROOM_PROXY_TOKEN` gate and startup bind policy.
//!
//! The Rust front connects to the Python proxy from `127.0.0.1`, and the
//! Python proxy exempts loopback peers from its token gate and trusts them
//! on `/admin`, `/debug` and `/settings`. So the front must apply the same
//! model itself, or every peer that reaches it is a "local" caller. Names,
//! credential rules and exemptions mirror `headroom/proxy/server.py`
//! (`_security_gate`, `read_proxy_token`) and `headroom/proxy/bind_policy.py`
//! so operators configure one token model for both processes.

use std::fmt;
use std::net::{IpAddr, SocketAddr};

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// The inbound credential. Same variable as the Python proxy.
pub const PROXY_TOKEN_ENV: &str = "HEADROOM_PROXY_TOKEN";
/// Explicit acknowledgement of an unauthenticated non-loopback bind.
/// Same variable and truthy values as the Python bind policy.
pub const OPEN_BIND_ACK_ENV: &str = "HEADROOM_ALLOW_UNAUTHENTICATED_BIND";
const PROXY_TOKEN_HEADER: &str = "x-headroom-proxy-token";

/// A configured proxy token. `Debug` never prints the value, because
/// `Config` derives `Debug`.
#[derive(Clone)]
pub struct ProxyToken(String);

impl ProxyToken {
    /// `None` for an empty or whitespace-only value, which the Python bind
    /// policy also treats as "not configured". A real value is kept
    /// verbatim, as the Python gate compares it.
    pub fn new(value: &str) -> Option<Self> {
        (!value.trim().is_empty()).then(|| Self(value.to_owned()))
    }

    fn matches(&self, provided: &str) -> bool {
        constant_time_eq(provided.as_bytes(), self.0.as_bytes())
    }
}

impl fmt::Debug for ProxyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProxyToken(<redacted>)")
    }
}

// Like Python's hmac.compare_digest: time does not depend on where the
// bytes differ (the length is not hidden, as there).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn is_loopback(addr: &SocketAddr) -> bool {
    addr.ip().to_canonical().is_loopback()
}

/// Outcome of the startup bind check that lets the process start.
#[derive(Debug, PartialEq, Eq)]
pub enum BindPolicy {
    /// Loopback bind, or a token is configured.
    Ok,
    /// Non-loopback bind with no token, acknowledged via [`OPEN_BIND_ACK_ENV`].
    OpenAcknowledged,
}

/// Refuse a non-loopback `listen` with no token unless acknowledged.
/// `ack` is the raw value of [`OPEN_BIND_ACK_ENV`]. The error text names
/// both remedies.
pub fn check_bind_policy(
    listen: SocketAddr,
    token_configured: bool,
    ack: Option<&str>,
) -> Result<BindPolicy, String> {
    if token_configured || is_loopback(&listen) {
        return Ok(BindPolicy::Ok);
    }
    let acknowledged = ack.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    });
    if acknowledged {
        return Ok(BindPolicy::OpenAcknowledged);
    }
    Err(format!(
        "Refusing to listen on {listen}: with no {PROXY_TOKEN_ENV} this proxy would relay \
         every network peer to the upstream Headroom proxy from 127.0.0.1, where it is \
         treated as a trusted local caller. Either set {PROXY_TOKEN_ENV} (e.g. \
         `openssl rand -hex 32`) so non-loopback callers must present it, listen on a \
         loopback address (--listen 127.0.0.1:8787), or, only when the runtime already \
         restricts who can reach the port (such as a container published on 127.0.0.1), \
         set {OPEN_BIND_ACK_ENV}=1 to acknowledge the open bind."
    ))
}

/// The caller-supplied token, read exactly as Python's `read_proxy_token`:
/// the explicit header wins when present (even empty, then it is no
/// credential); otherwise a `Bearer` Authorization value.
fn read_proxy_token(headers: &HeaderMap) -> Option<&str> {
    if let Some(raw) = headers.get(PROXY_TOKEN_HEADER) {
        return raw.to_str().ok().filter(|v| !v.is_empty());
    }
    bearer(headers)
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let auth = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let scheme = auth.get(..7)?;
    if !scheme.eq_ignore_ascii_case("bearer ") {
        return None;
    }
    Some(auth[7..].trim()).filter(|v| !v.is_empty())
}

/// Python exempts `GET /health`, `/livez` and `/readyz`; `/healthz` and
/// `/healthz/upstream` are this proxy's own probe routes.
fn is_health_probe(method: &Method, path: &str) -> bool {
    *method == Method::GET
        && matches!(
            path,
            "/healthz" | "/healthz/upstream" | "/health" | "/livez" | "/readyz"
        )
}

/// The Anthropic SDK's `GET`/`HEAD /api/hello` reachability probe (also
/// under `/p/<project>`). It never carries custom headers, so it cannot
/// hold the token.
fn is_hello_probe(method: &Method, path: &str) -> bool {
    let hello = path == "/api/hello"
        || path
            .strip_prefix("/p/")
            .and_then(|rest| rest.split_once('/'))
            .is_some_and(|(project, rest)| !project.is_empty() && rest == "api/hello");
    (*method == Method::GET || *method == Method::HEAD) && hello
}

/// Request extension: the caller is not local (not a loopback peer with a
/// loopback `Host`). `forward_http` then never sends it upstream with a
/// loopback `Host`, so the upstream's local-only routes (`/admin`,
/// `/debug`, `/settings`, ...) do not treat a relayed caller as local.
/// Without it this process's own loopback connection and rewritten `Host`
/// would pass both checks those routes make.
#[derive(Clone, Copy, Debug)]
pub struct NonLocalCaller;

/// `Host` sent upstream in place of one that may name loopback, for a
/// [`NonLocalCaller`]. A reserved name (RFC 2606), never the caller's own
/// value: the caller controls that, and the upstream's parser may accept
/// loopback forms this one does not.
pub const NON_LOCAL_HOST: &str = "headroom-remote.invalid";

/// For a [`NonLocalCaller`]: [`NON_LOCAL_HOST`] when the `Host` that would
/// be sent may name loopback, else `None` (keep it). That `Host` is the
/// explicit outgoing one (`--no-rewrite-host`), else the upstream URL's
/// authority. Judged on the value, not on the upstream address, so an
/// upstream name that resolves to loopback cannot be paired with a
/// loopback `Host`.
pub(crate) fn host_override_for_non_local(
    outgoing: Option<&HeaderValue>,
    upstream: &url::Url,
) -> Option<HeaderValue> {
    let effective = match outgoing {
        Some(h) => h.to_str().ok().map(str::to_owned),
        None => upstream.host_str().map(|h| match upstream.port() {
            Some(port) => format!("{h}:{port}"),
            None => h.to_owned(),
        }),
    };
    may_name_loopback(effective.as_deref()).then(|| HeaderValue::from_static(NON_LOCAL_HOST))
}

/// Conservative superset of what the Python proxy treats as a loopback
/// `Host`: anything [`is_loopback_host_header`] accepts, an unreadable
/// value, and any value with an IPv6 zone (`%`), since Python's
/// `ipaddress` accepts scoped literals such as `[::1%lo]`.
fn may_name_loopback(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(v) => v.contains('%') || is_loopback_host_header(Some(v)),
    }
}

/// Same rules as Python's `loopback_guard.is_loopback_host_header`:
/// `localhost`, or a loopback IP literal, with an optional port and
/// brackets around IPv6. Missing or malformed is not loopback.
fn is_loopback_host_header(value: Option<&str>) -> bool {
    let Some(candidate) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return false;
    };
    let host = if let Some(rest) = candidate.strip_prefix('[') {
        match rest.split_once(']') {
            Some((h, _)) => h,
            None => return false,
        }
    } else if candidate.matches(':').count() == 1 {
        candidate.rsplit_once(':').map_or(candidate, |(h, _)| h)
    } else {
        candidate
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.to_canonical().is_loopback())
}

/// Router-wide middleware. Marks every caller that is not local with
/// [`NonLocalCaller`]. When a token is configured, a non-loopback (or
/// unknown) peer must also present it or gets 401 with no upstream
/// contact, and the credential is removed before forwarding.
pub async fn inbound_gate(
    State(token): State<Option<ProxyToken>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // No ConnectInfo is no evidence of a local caller: fail closed, as
    // Python's is_loopback_host(None) does.
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    let loopback_peer = peer.as_ref().is_some_and(is_loopback);
    if let Some(token) = &token {
        // Answered here and empty, never relayed upstream without a
        // credential: the same answer the Python proxy gives it.
        if is_hello_probe(req.method(), req.uri().path()) {
            return StatusCode::OK.into_response();
        }
        if !loopback_peer && !is_health_probe(req.method(), req.uri().path()) {
            let provided = read_proxy_token(req.headers());
            if !provided.is_some_and(|p| token.matches(p)) {
                tracing::warn!(
                    event = "proxy_auth_rejected",
                    method = %req.method(),
                    path = req.uri().path(),
                    peer = ?peer,
                    reason = if provided.is_none() { "missing_token" } else { "bad_token" },
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({ "error": "unauthorized" })),
                )
                    .into_response();
            }
        }
        let headers = req.headers_mut();
        headers.remove(PROXY_TOKEN_HEADER);
        if bearer(headers).is_some_and(|b| token.matches(b)) {
            headers.remove(header::AUTHORIZATION);
        }
    }
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .or_else(|| req.uri().authority().map(|a| a.as_str()));
    if !(loopback_peer && is_loopback_host_header(host)) {
        req.extensions_mut().insert(NonLocalCaller);
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn bind_policy_refuses_open_non_loopback_bind() {
        let err = check_bind_policy(addr("0.0.0.0:8787"), false, None).unwrap_err();
        assert!(err.contains(PROXY_TOKEN_ENV) && err.contains(OPEN_BIND_ACK_ENV));
        assert!(check_bind_policy(addr("[::]:8787"), false, Some("0")).is_err());
        assert!(check_bind_policy(addr("192.0.2.1:8787"), false, Some("")).is_err());
    }

    #[test]
    fn bind_policy_allows_loopback_token_or_ack() {
        assert_eq!(
            check_bind_policy(addr("127.0.0.1:8787"), false, None),
            Ok(BindPolicy::Ok)
        );
        assert_eq!(
            check_bind_policy(addr("[::1]:8787"), false, None),
            Ok(BindPolicy::Ok)
        );
        assert_eq!(
            check_bind_policy(addr("0.0.0.0:8787"), true, None),
            Ok(BindPolicy::Ok)
        );
        for ack in ["1", "TRUE", " yes ", "on"] {
            assert_eq!(
                check_bind_policy(addr("0.0.0.0:8787"), false, Some(ack)),
                Ok(BindPolicy::OpenAcknowledged)
            );
        }
    }

    #[test]
    fn token_rules_match_python() {
        assert!(ProxyToken::new("  ").is_none());
        let mut h = HeaderMap::new();
        h.insert(header::AUTHORIZATION, "BEARER  abc ".parse().unwrap());
        assert_eq!(read_proxy_token(&h), Some("abc"));
        h.insert(PROXY_TOKEN_HEADER, "".parse().unwrap());
        assert_eq!(read_proxy_token(&h), None, "present-but-empty header wins");
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(format!("{:?}", ProxyToken::new("abc").unwrap()).contains("redacted"));
    }

    #[test]
    fn loopback_host_header_matches_python() {
        for ok in [
            "127.0.0.1:8787",
            "localhost",
            "LOCALHOST:1",
            "[::1]:8787",
            "::1",
            "127.1.2.3",
        ] {
            assert!(is_loopback_host_header(Some(ok)), "{ok}");
        }
        for bad in [
            "",
            "evil.example",
            "evil.example:8787",
            "10.0.0.1:8787",
            "[::1",
            "localhost.evil.example",
        ] {
            assert!(!is_loopback_host_header(Some(bad)), "{bad}");
        }
        assert!(!is_loopback_host_header(None));
    }

    #[test]
    fn non_local_caller_never_gets_a_loopback_host_upstream() {
        let up = |u: &str| url::Url::parse(u).unwrap();
        let hv = |h: &str| HeaderValue::from_str(h).unwrap();
        let alias = up("http://headroom-upstream.internal:8788");
        let loopback = up("http://127.0.0.1:8788");
        // --no-rewrite-host: the caller's Host is what would be sent.
        for host in [
            "localhost",
            "127.0.0.1:8787",
            "[::1]:8787",
            "[::1%lo]:8787",
            "::1%lo",
        ] {
            assert_eq!(
                host_override_for_non_local(Some(&hv(host)), &alias).unwrap(),
                NON_LOCAL_HOST,
                "{host}"
            );
        }
        // Rewritten Host on a loopback upstream: the reserved name.
        assert_eq!(
            host_override_for_non_local(None, &loopback).unwrap(),
            NON_LOCAL_HOST
        );
        // Not loopback: left alone.
        assert!(host_override_for_non_local(None, &alias).is_none());
        assert!(host_override_for_non_local(Some(&hv("proxy.example")), &loopback).is_none());
    }

    #[test]
    fn exemptions_match_python() {
        assert!(is_health_probe(&Method::GET, "/healthz"));
        assert!(is_health_probe(&Method::GET, "/readyz"));
        assert!(!is_health_probe(&Method::POST, "/readyz"));
        assert!(!is_health_probe(&Method::HEAD, "/readyz"));
        assert!(!is_health_probe(&Method::GET, "/admin/upstream"));
        assert!(!is_health_probe(&Method::GET, "/rollout/status"));
        assert!(is_hello_probe(&Method::HEAD, "/api/hello"));
        assert!(is_hello_probe(&Method::GET, "/p/demo/api/hello"));
        assert!(!is_hello_probe(&Method::GET, "/p//api/hello"));
        assert!(!is_hello_probe(&Method::POST, "/api/hello"));
    }
}

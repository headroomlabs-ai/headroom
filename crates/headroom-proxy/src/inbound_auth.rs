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
/// Proof, sent upstream, that this front authenticated the caller with the
/// token. Same name and derivation as Python's `FRONT_RELAY_PROOF_HEADER` /
/// `front_relay_proof`: the Python proxy behind the front accepts it in
/// place of the token this front strips. Removed from every inbound request.
pub const RELAY_PROOF_HEADER: &str = "x-headroom-front-proof";

/// Set to `1` on every request relayed for a [`NonLocalCaller`], whatever
/// `Host` is sent and in every mode, and removed from what clients send.
/// Not a secret: it only takes trust away. The Python proxy never treats a
/// request carrying it as local (loopback or container-alias `Host`), so a
/// remote caller cannot borrow a local name it would trust from this
/// process's loopback connection.
pub const FRONT_REMOTE_HEADER: &str = "x-headroom-front-remote";
const RELAY_PROOF_CONTEXT: &[u8] = b"headroom-front-relay-v1\0";

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

    /// The [`RELAY_PROOF_HEADER`] value for this token: lowercase hex
    /// SHA-256 of a fixed context and the token. Derived, so the token
    /// itself never leaves this process.
    pub fn relay_proof(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(RELAY_PROOF_CONTEXT);
        hasher.update(self.0.as_bytes());
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
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

/// Request extension: the [`RELAY_PROOF_HEADER`] value to send upstream,
/// set only for a non-local caller that presented the token here. Sent
/// only to a loopback upstream (see [`upstream_takes_relay_proof`]).
#[derive(Clone, Debug)]
pub struct RelayProof(pub HeaderValue);

/// `Host` sent upstream in place of one that may name loopback, for a
/// [`NonLocalCaller`]. A reserved name (RFC 2606), never the caller's own
/// value: the caller controls that, and the upstream's parser may accept
/// loopback forms this one does not.
pub const NON_LOCAL_HOST: &str = "headroom-remote.invalid";

/// What the front tells its upstream about a caller it relays, on HTTP and
/// WebSocket alike: the remote marker for a [`NonLocalCaller`], and, when
/// that caller presented the token and the upstream is a `--python-upstream`
/// address that may take it, the relay proof standing in for the stripped
/// token. A local caller's request is left as it is.
pub(crate) fn insert_relay_identity(
    headers: &mut HeaderMap,
    extensions: &axum::http::Extensions,
    upstream: &url::Url,
    python_upstream: bool,
) {
    if extensions.get::<NonLocalCaller>().is_none() {
        return;
    }
    headers.insert(FRONT_REMOTE_HEADER, HeaderValue::from_static("1"));
    if let Some(proof) = extensions.get::<RelayProof>() {
        if python_upstream && upstream_takes_relay_proof(upstream) {
            headers.insert(RELAY_PROOF_HEADER, proof.0.clone());
        }
    }
}

/// Whether a `--python-upstream` upstream may receive the relay proof: the
/// Python proxy on this machine, addressed as a loopback IP literal,
/// `localhost`, or (from a container) a container host alias.
pub(crate) fn upstream_takes_relay_proof(upstream: &url::Url) -> bool {
    match upstream.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.to_canonical().is_loopback(),
        Some(url::Host::Domain(name)) => {
            name.eq_ignore_ascii_case("localhost") || names_container_host_alias(name)
        }
        None => false,
    }
}

/// For a [`NonLocalCaller`]: [`NON_LOCAL_HOST`] when the upstream is the
/// Python proxy (`--python-upstream`), which tells relayed callers apart by
/// that name, or when the `Host` that would be sent may name loopback; else
/// `None` (keep it, so a generic backend can route on the caller's `Host`). That `Host` is the
/// explicit outgoing one (`--no-rewrite-host`), else the upstream URL's
/// authority. Judged on the value, not on the upstream address, so an
/// upstream name that resolves to loopback cannot be paired with a
/// loopback `Host`.
pub(crate) fn host_override_for_non_local(
    outgoing: Option<&HeaderValue>,
    upstream: &url::Url,
    python_upstream: bool,
) -> Option<HeaderValue> {
    if python_upstream {
        return Some(HeaderValue::from_static(NON_LOCAL_HOST));
    }
    let replace = match outgoing {
        // A Host the caller chose (`--no-rewrite-host`): never one the
        // Python proxy would trust as local, container aliases included.
        Some(h) => {
            let value = h.to_str().ok();
            may_name_loopback(value) || value.is_some_and(names_container_host_alias)
        }
        // The upstream URL's own authority: a generic backend configured as
        // `http://host.docker.internal:4000` keeps the name it routes on.
        None => may_name_loopback(
            upstream
                .host_str()
                .map(|h| match upstream.port() {
                    Some(port) => format!("{h}:{port}"),
                    None => h.to_owned(),
                })
                .as_deref(),
        ),
    };
    replace.then(|| HeaderValue::from_static(NON_LOCAL_HOST))
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

/// Python's `_CONTAINER_HOST_ALIASES`, on any port; case and a trailing
/// dot ignored. The Python proxy trusts them from a loopback peer (Docker
/// Desktop delivers container callers to the host from 127.0.0.1); this
/// front does not, they only decide which caller-chosen `Host` it relays.
fn names_container_host_alias(value: &str) -> bool {
    let host = value.trim();
    let host = host.rsplit_once(':').map_or(host, |(h, _)| h);
    let host = host.strip_suffix('.').unwrap_or(host);
    ["host.docker.internal", "host.containers.internal"]
        .iter()
        .any(|alias| host.eq_ignore_ascii_case(alias))
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
/// [`NonLocalCaller`]. A loopback peer under a `Host` that does not name
/// loopback is the shape of a browser page whose hostname was pointed at
/// 127.0.0.1, so it is treated like a remote peer: with a token configured
/// it must present it (401 otherwise), and without one it is refused (403).
/// A non-local caller that presented the token is forwarded with a
/// [`RelayProof`]; the credential itself is removed before forwarding.
pub async fn inbound_gate(
    State(token): State<Option<ProxyToken>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // Only this process may vouch for a caller upstream, or mark it remote.
    req.headers_mut().remove(RELAY_PROOF_HEADER);
    req.headers_mut().remove(FRONT_REMOTE_HEADER);
    // No ConnectInfo is no evidence of a local caller: fail closed, as
    // Python's is_loopback_host(None) does.
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    let loopback_peer = peer.as_ref().is_some_and(is_loopback);
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .or_else(|| req.uri().authority().map(|a| a.as_str()));
    // A browser always sends a Host, so a loopback peer without one (a raw
    // HTTP/1.0 client on this machine) is not the rebound-page shape: it is
    // local, like a loopback Host.
    let local = loopback_peer && (host.is_none() || is_loopback_host_header(host));
    let mut authenticated = false;
    if let Some(token) = &token {
        // Answered here and empty, never relayed upstream without a
        // credential: the same answer the Python proxy gives it.
        if is_hello_probe(req.method(), req.uri().path()) {
            return StatusCode::OK.into_response();
        }
        if !local && !is_health_probe(req.method(), req.uri().path()) {
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
            authenticated = true;
        }
        let headers = req.headers_mut();
        headers.remove(PROXY_TOKEN_HEADER);
        if bearer(headers).is_some_and(|b| token.matches(b)) {
            headers.remove(header::AUTHORIZATION);
        }
    } else if loopback_peer && !local && !is_health_probe(req.method(), req.uri().path()) {
        tracing::warn!(
            event = "proxy_loopback_host_rejected",
            method = %req.method(),
            path = req.uri().path(),
        );
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": "request to a loopback listener under a non-loopback Host rejected",
                "hint": "use 127.0.0.1 or localhost, or set HEADROOM_PROXY_TOKEN and send it",
            })),
        )
            .into_response();
    }
    if !local {
        req.extensions_mut().insert(NonLocalCaller);
        if let (true, Some(token)) = (authenticated, &token) {
            if let Ok(proof) = HeaderValue::from_str(&token.relay_proof()) {
                req.extensions_mut().insert(RelayProof(proof));
            }
        }
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
    fn relay_proof_goes_only_to_a_loopback_upstream() {
        for ok in [
            "http://127.0.0.1:8788",
            "http://[::1]:8788",
            "http://LOCALHOST:1",
            "http://host.docker.internal:8788",
            "http://HOST.CONTAINERS.INTERNAL:8788",
        ] {
            assert!(upstream_takes_relay_proof(&ok.parse().unwrap()), "{ok}");
        }
        for no in [
            "https://api.anthropic.com",
            "http://10.0.0.5:8788",
            "http://localhost.example.com",
            "http://host.docker.internal.example.com",
        ] {
            assert!(!upstream_takes_relay_proof(&no.parse().unwrap()), "{no}");
        }
    }

    #[test]
    fn relay_identity_for_a_container_alias_upstream() {
        // CI has no container DNS, so the alias upstream is checked here: a
        // front in a container dialling the Python proxy as
        // host.docker.internal sends the marker and the proof Python derives
        // from the same token (see `relay_proof_matches_python`).
        let token = ProxyToken::new("abc").unwrap();
        let proof = HeaderValue::from_str(&token.relay_proof()).unwrap();
        let identity = |upstream: &str, python: bool, remote: bool, proved: bool| {
            let mut ext = axum::http::Extensions::new();
            if remote {
                ext.insert(NonLocalCaller);
            }
            if proved {
                ext.insert(RelayProof(proof.clone()));
            }
            let mut headers = HeaderMap::new();
            insert_relay_identity(&mut headers, &ext, &upstream.parse().unwrap(), python);
            (
                headers.get(FRONT_REMOTE_HEADER).cloned(),
                headers.get(RELAY_PROOF_HEADER).cloned(),
            )
        };
        let marker = Some(HeaderValue::from_static("1"));
        for upstream in [
            "http://host.docker.internal:8788",
            "http://host.containers.internal:8788",
        ] {
            assert_eq!(
                identity(upstream, true, true, true),
                (marker.clone(), Some(proof.clone()))
            );
            // Without --python-upstream the marker still goes, the proof never.
            assert_eq!(
                identity(upstream, false, true, true),
                (marker.clone(), None)
            );
            // A remote caller that did not present the token: marker only.
            assert_eq!(
                identity(upstream, true, true, false),
                (marker.clone(), None)
            );
            // A local caller: nothing.
            assert_eq!(identity(upstream, true, false, false), (None, None));
        }
        assert_eq!(
            identity("http://proxy.internal.example:8788", true, true, true),
            (marker, None)
        );
    }

    #[test]
    fn relay_proof_matches_python() {
        // headroom.proxy.server.front_relay_proof("abc")
        assert_eq!(
            ProxyToken::new("abc").unwrap().relay_proof(),
            "33d9db219abd5b185118a8980948fe91930384fdf1a44fa3a964fcdb670be2e3"
        );
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
                host_override_for_non_local(Some(&hv(host)), &alias, false).unwrap(),
                NON_LOCAL_HOST,
                "{host}"
            );
        }
        // Rewritten Host on a loopback upstream: the reserved name.
        assert_eq!(
            host_override_for_non_local(None, &loopback, false).unwrap(),
            NON_LOCAL_HOST
        );
        // Not loopback: left alone.
        assert!(host_override_for_non_local(None, &alias, false).is_none());
        // The container host aliases the Python proxy trusts from loopback.
        for host in [
            "host.docker.internal",
            "HOST.DOCKER.INTERNAL.:8787",
            "host.containers.internal:443",
        ] {
            assert_eq!(
                host_override_for_non_local(Some(&hv(host)), &loopback, false).unwrap(),
                NON_LOCAL_HOST,
                "{host}"
            );
        }
        // A generic upstream addressed by a container name keeps it (default
        // Host rewriting); only a caller-chosen alias is replaced.
        let container = up("http://host.docker.internal:4000");
        assert!(host_override_for_non_local(None, &container, false).is_none());
        // A generic loopback backend keeps a public Host it may route on.
        assert!(
            host_override_for_non_local(Some(&hv("proxy.example")), &loopback, false).is_none()
        );
        // The Python proxy (--python-upstream) always gets the reserved name.
        assert_eq!(
            host_override_for_non_local(Some(&hv("proxy.example")), &loopback, true).unwrap(),
            NON_LOCAL_HOST
        );
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

//! Bedrock-route auth-mode middleware — Phase D PR-D3.
//!
//! # Why a dedicated middleware (vs inlining the classify call)?
//!
//! The Bedrock invoke + invoke-streaming handlers don't run through
//! `proxy::forward_http`'s catch-all (which is where Phase F PR-F1
//! classifies and stores the value in request extensions for
//! `/v1/messages`, `/v1/chat/completions`, `/v1/responses`). To
//! preserve the same downstream contract — Phase F PR-F2 and PR-F3
//! will read the [`AuthMode`] back out of `request.extensions()` and
//! gate compression policy on it — every Bedrock route applies this
//! middleware. The middleware:
//!
//! 1. Classifies the inbound headers via
//!    [`headroom_core::auth_mode::classify`].
//! 2. **Asserts** the result is [`AuthMode::OAuth`] under the Bedrock
//!    policy matrix. AWS SigV4 is an `Authorization` value that
//!    isn't `Bearer ...` — F1's classifier already routes that to
//!    OAuth (see `auth_mode.rs` decision rule 5). We additionally
//!    catch the no-Authorization case (Bedrock SDK signs DOWNSTREAM
//!    of our proxy on some setups, leaving the inbound request
//!    unsigned), classify it, and if F1 returned `Payg` for it we
//!    still force `OAuth` while emitting
//!    `event = bedrock_auth_mode_unexpected` at WARN. Per the
//!    realignment build constraint "no silent fallbacks", we
//!    NEVER silently coerce — the divergence is loud.
//! 3. Stores the resolved [`AuthMode`] in `request.extensions()`
//!    so downstream handlers can read it without re-classifying.
//! 4. Emits a structured info-level log with
//!    `event = bedrock_auth_mode_classified` for ops correlation.
//!
//! # Performance
//!
//! `classify` is a pure function with one short owned `String` for
//! the lowercase UA copy; benched <10us. Inserting into request
//! extensions is `O(1)`. Total per-request overhead well under 1us
//! (excluding the classify call itself, which is shared with the
//! main proxy path).
//!
//! # Where the middleware is mounted
//!
//! See `crate::proxy::build_app` — the Bedrock router branch wraps
//! the three Bedrock POST routes
//! (`/model/:model_id/invoke`, `/converse`, and
//! `/invoke-with-response-stream`) with this layer using
//! `axum::middleware::from_fn`. The catch-all and the other
//! provider routes already classify in `forward_http`, so this
//! middleware does NOT apply to them.

use axum::body::Body;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

use headroom_core::auth_mode::{classify, AuthMode};

/// Inspect the inbound headers, classify the auth mode under
/// Bedrock policy (always [`AuthMode::OAuth`], with a loud WARN
/// when F1's classifier disagrees), and attach the resolved value
/// to `request.extensions()`.
///
/// Mounted as `axum::middleware::from_fn(classify_and_attach_auth_mode)`
/// so it composes with axum's standard router. The middleware is
/// infallible — it never short-circuits the request, never returns
/// an error response, and never panics. Worst case it logs and
/// proceeds.
pub async fn classify_and_attach_auth_mode(mut req: Request<Body>, next: Next) -> Response {
    let raw_classification = classify(req.headers());

    // Bedrock policy: always OAuth-equivalent. SigV4 IAM is an
    // OAuth-class signal under our policy matrix. F1 already
    // returns OAuth for non-Bearer Authorization (rule 5) and for
    // sk-ant-oat-* Bearer tokens (rule 2); the only paths that
    // wouldn't return OAuth are:
    //
    //   - empty headers entirely (test setups, or AWS SDK that
    //     signs after our hop) → F1 returns Payg.
    //   - x-api-key set (Anthropic key on a Bedrock URL — wrong
    //     surface, but possible in misconfigured setups) → F1
    //     returns Payg.
    //
    // In both cases we coerce to OAuth (defence-in-depth: Bedrock
    // is OAuth-class regardless of the inbound surface) but log
    // loudly so operators see the misclassification. NO SILENT
    // FALLBACK — the divergence is the whole reason for the warn.
    let resolved = if raw_classification == AuthMode::OAuth {
        AuthMode::OAuth
    } else {
        tracing::warn!(
            event = "bedrock_auth_mode_unexpected",
            raw = raw_classification.as_str(),
            resolved = AuthMode::OAuth.as_str(),
            path = %req.uri().path(),
            "Bedrock route received headers that classified as non-OAuth; \
             coercing to OAuth per Bedrock policy and logging the divergence \
             so operators can investigate the source"
        );
        AuthMode::OAuth
    };

    tracing::info!(
        event = "bedrock_auth_mode_classified",
        mode = resolved.as_str(),
        raw = raw_classification.as_str(),
        path = %req.uri().path(),
        "bedrock route classified inbound auth mode"
    );

    req.extensions_mut().insert(resolved);
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::Extension;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use http::HeaderValue;
    use tower::util::ServiceExt;

    /// Probe handler: returns the AuthMode it sees in extensions.
    async fn probe(Extension(auth_mode): Extension<AuthMode>) -> String {
        auth_mode.as_str().to_string()
    }

    fn router() -> Router {
        Router::new()
            .route("/probe", post(probe))
            .layer(axum::middleware::from_fn(classify_and_attach_auth_mode))
    }

    #[tokio::test]
    async fn empty_headers_classify_as_oauth_for_bedrock() {
        // No Authorization, no x-api-key, no UA — F1 returns Payg
        // by default. The middleware coerces to OAuth (with a
        // WARN logged at the call site) so downstream sees OAuth
        // and the policy matrix is consistent.
        let app = router();
        let req = HttpRequest::builder()
            .method("POST")
            .uri("/probe")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 64).await.unwrap();
        assert_eq!(&body[..], b"oauth");
    }

    #[tokio::test]
    async fn sigv4_authorization_classifies_as_oauth() {
        // Real Bedrock SDK does sign before reaching us in some
        // setups — `Authorization: AWS4-HMAC-SHA256 ...` is
        // routed to OAuth by F1's rule 5 directly. No coercion
        // needed; no WARN.
        let app = router();
        let mut req = HttpRequest::builder()
            .method("POST")
            .uri("/probe")
            .body(Body::empty())
            .unwrap();
        req.headers_mut().insert(
            "authorization",
            HeaderValue::from_static(
                "AWS4-HMAC-SHA256 Credential=AKIA.../20260101/us-east-1/bedrock/aws4_request",
            ),
        );
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 64).await.unwrap();
        assert_eq!(&body[..], b"oauth");
    }

    #[tokio::test]
    async fn x_api_key_inbound_is_coerced_to_oauth_loudly() {
        // x-api-key on the Bedrock surface is misconfigured but
        // possible. F1 returns Payg; we coerce to OAuth.
        let app = router();
        let mut req = HttpRequest::builder()
            .method("POST")
            .uri("/probe")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert("x-api-key", HeaderValue::from_static("sk-ant-api-fake"));
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 64).await.unwrap();
        assert_eq!(&body[..], b"oauth");
    }
}

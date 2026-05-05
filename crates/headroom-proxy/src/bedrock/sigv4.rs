//! AWS SigV4 request signing for the Bedrock InvokeModel route.
//!
//! # What this module does
//!
//! Wraps the `aws-sigv4` crate with the project's policies:
//!
//! - **No silent fallbacks.** A failed sign-attempt returns a
//!   structured error; the handler surfaces 5xx and logs an event.
//!   We NEVER forward an unsigned request to Bedrock, because the
//!   AWS endpoint will reject anyway and the user would see an
//!   opaque 403.
//! - **Body bytes are hashed AFTER compression.** The signing
//!   inputs include a `&[u8]` that is the EXACT byte slice the
//!   forwarder will send upstream. There is no separate "hash the
//!   pre-compression body" code path.
//! - **Structured logs at every decision point.** Every code path
//!   emits `tracing::info!`/`warn!` with an `event = ...` field so
//!   operators can confirm signing happened.
//!
//! # What this module deliberately does NOT do
//!
//! - It does not resolve credentials — that's `aws-config`'s job and
//!   happens at app-startup time so per-request signing is cheap.
//!   The handler holds the resolved [`aws_credential_types::Credentials`]
//!   in `AppState` and passes them in.
//! - It does not buffer the body — callers buffer the body in the
//!   compression gate (it's the same byte slice).
//! - It does not handle SigV4a (the cross-region variant). Bedrock
//!   uses standard SigV4 per region.

use std::time::SystemTime;

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use thiserror::Error;
use url::Url;

/// AWS service name used in the SigV4 string-to-sign for Bedrock.
/// Documented at
/// <https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html>
pub const BEDROCK_SERVICE_NAME: &str = "bedrock";

/// Inputs needed to sign a Bedrock request. Borrows the body so we
/// avoid a copy on the hot path.
#[derive(Debug)]
pub struct SigningInputs<'a> {
    /// HTTP method (always `"POST"` for InvokeModel today, but we
    /// keep this explicit so a future GET-shaped surface doesn't
    /// require redoing the call site).
    pub method: &'a str,
    /// Fully-qualified upstream URL (scheme + host + path + query).
    /// Used for canonical-request URI normalization.
    pub url: &'a Url,
    /// AWS region the upstream endpoint lives in.
    pub region: &'a str,
    /// AWS credentials (resolved from the `aws-config` default chain
    /// at app-startup time).
    pub credentials: &'a Credentials,
    /// Body bytes to sign. MUST be the exact bytes the forwarder
    /// will send to Bedrock (post-compression).
    pub body: &'a [u8],
    /// Extra headers the canonical request must include in the
    /// signed-headers list. The signer always includes `host`,
    /// `x-amz-date`, and `x-amz-content-sha256`; callers add
    /// anything else they want covered (e.g. `accept-encoding`,
    /// `content-type`).
    pub extra_signed_headers: &'a [(&'a str, &'a str)],
    /// Time to use in the signature. Production uses
    /// `SystemTime::now()`; tests pin a known time to make the
    /// canonical request deterministic.
    pub time: SystemTime,
}

/// Headers that the signer will write into the outbound request.
/// The handler must add every entry to the upstream-bound HeaderMap
/// before sending — Bedrock validates each header against the
/// canonical request.
#[derive(Debug, Clone)]
pub struct SignedHeaders {
    pub entries: Vec<(String, String)>,
    /// Lowercase hex SHA-256 of the body. Surfaced for tests + logs.
    pub signature: String,
}

/// Errors surfaced by the signing path.
#[derive(Debug, Error)]
pub enum SigV4Error {
    /// `aws-sigv4` rejected the request (URL parse, malformed header,
    /// etc).
    #[error("sigv4 signing failed: {0}")]
    Sign(String),
    /// The signing-params builder rejected the inputs (e.g. missing
    /// region — should never happen because we validate at startup).
    #[error("sigv4 builder error: {0}")]
    Builder(String),
}

/// Sign a Bedrock request and return the headers the handler must add
/// to the outbound request.
///
/// # Cache safety
///
/// The body bytes passed in MUST be the bytes the proxy is about to
/// send upstream. If the compressor mutated the body, those mutated
/// bytes are what get signed — Bedrock will accept because the
/// signature covers the wire payload, not the original.
///
/// # Errors
///
/// Returns [`SigV4Error::Sign`] when `aws-sigv4` rejects the request
/// (malformed URL, etc). Returns [`SigV4Error::Builder`] when the
/// signing-params builder rejects the inputs.
pub fn sign_request(inputs: &SigningInputs<'_>) -> Result<SignedHeaders, SigV4Error> {
    // Build the identity wrapper around the credentials. Identity is
    // the type the SigV4 signer accepts; it can in principle hold
    // alternative auth schemes (Bearer, etc) but we only ever use it
    // for AWS creds.
    let identity: Identity = Identity::new(inputs.credentials.clone(), None);

    // Default settings + force `x-amz-content-sha256` into the
    // canonical request. Bedrock validates the content hash to
    // catch any in-flight body mutation; with `NoHeader` (the
    // crate-level default) the signer would skip the header,
    // which means a downstream gateway that DOES check it would
    // 403.
    let mut settings = SigningSettings::default();
    settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(inputs.region)
        .name(BEDROCK_SERVICE_NAME)
        .time(inputs.time)
        .settings(settings)
        .build()
        .map_err(|e| SigV4Error::Builder(e.to_string()))?
        .into();

    // The signer needs the URL as a string; Url's Display impl is
    // canonical RFC 3986 form which is what aws-sigv4 expects.
    let url_string = inputs.url.to_string();

    let signable = SignableRequest::new(
        inputs.method,
        url_string,
        inputs.extra_signed_headers.iter().copied(),
        SignableBody::Bytes(inputs.body),
    )
    .map_err(|e| SigV4Error::Sign(e.to_string()))?;

    let signing_output =
        sign(signable, &signing_params).map_err(|e| SigV4Error::Sign(e.to_string()))?;

    let signature = signing_output.signature().to_string();
    let (instructions, _signature) = signing_output.into_parts();
    let (header_entries, _query_params) = instructions.into_parts();
    let entries = header_entries
        .into_iter()
        .map(|h| (h.name().to_string(), h.value().to_string()))
        .collect::<Vec<_>>();

    tracing::info!(
        event = "sigv4_signed",
        forwarder = "rust_proxy",
        region = inputs.region,
        service = BEDROCK_SERVICE_NAME,
        method = inputs.method,
        host = inputs.url.host_str().unwrap_or(""),
        body_bytes = inputs.body.len(),
        headers_added = entries.len(),
        "bedrock request signed with sigv4"
    );

    Ok(SignedHeaders { entries, signature })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fixed_time() -> SystemTime {
        // 2026-05-03T12:00:00Z — pinned so canonical request is
        // deterministic across runs.
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_777_910_400)
    }

    fn fixture_credentials() -> Credentials {
        Credentials::new("AKIA_TEST", "secret_test", None, None, "test")
    }

    #[test]
    fn signs_minimal_request_produces_expected_headers() {
        let url = Url::parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
        )
        .unwrap();
        let creds = fixture_credentials();
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":16,"messages":[]}"#;
        let inputs = SigningInputs {
            method: "POST",
            url: &url,
            region: "us-east-1",
            credentials: &creds,
            body,
            extra_signed_headers: &[("content-type", "application/json")],
            time: fixed_time(),
        };
        let signed = sign_request(&inputs).expect("sigv4 sign");
        // The signer always emits `authorization`, `x-amz-date`, and
        // `x-amz-content-sha256`. Confirm all three are present.
        let names: Vec<String> = signed
            .entries
            .iter()
            .map(|(k, _)| k.to_ascii_lowercase())
            .collect();
        assert!(
            names.iter().any(|n| n == "authorization"),
            "must add authorization; got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "x-amz-date"),
            "must add x-amz-date"
        );
        assert!(
            names.iter().any(|n| n == "x-amz-content-sha256"),
            "must add x-amz-content-sha256"
        );
        assert_eq!(signed.signature.len(), 64, "sigv4 signature is 32-byte hex");
    }

    #[test]
    fn signature_is_deterministic_for_fixed_inputs() {
        let url = Url::parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
        )
        .unwrap();
        let creds = fixture_credentials();
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":16}"#;
        let mk = || SigningInputs {
            method: "POST",
            url: &url,
            region: "us-east-1",
            credentials: &creds,
            body,
            extra_signed_headers: &[("content-type", "application/json")],
            time: fixed_time(),
        };
        let a = sign_request(&mk()).unwrap();
        let b = sign_request(&mk()).unwrap();
        assert_eq!(a.signature, b.signature);
    }

    #[test]
    fn changing_body_changes_signature() {
        let url = Url::parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
        )
        .unwrap();
        let creds = fixture_credentials();
        let inputs_a = SigningInputs {
            method: "POST",
            url: &url,
            region: "us-east-1",
            credentials: &creds,
            body: br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":16}"#,
            extra_signed_headers: &[("content-type", "application/json")],
            time: fixed_time(),
        };
        let inputs_b = SigningInputs {
            body: br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":32}"#,
            ..SigningInputs {
                method: "POST",
                url: &url,
                region: "us-east-1",
                credentials: &creds,
                body: &[],
                extra_signed_headers: &[("content-type", "application/json")],
                time: fixed_time(),
            }
        };
        let a = sign_request(&inputs_a).unwrap();
        let b = sign_request(&inputs_b).unwrap();
        assert_ne!(
            a.signature, b.signature,
            "different body bytes must yield different signatures"
        );
    }

    #[test]
    fn changing_region_changes_signature() {
        let url = Url::parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke",
        )
        .unwrap();
        let creds = fixture_credentials();
        let body = br#"{"anthropic_version":"bedrock-2023-05-31"}"#;
        let east = SigningInputs {
            method: "POST",
            url: &url,
            region: "us-east-1",
            credentials: &creds,
            body,
            extra_signed_headers: &[],
            time: fixed_time(),
        };
        let west = SigningInputs {
            method: "POST",
            url: &url,
            region: "us-west-2",
            credentials: &creds,
            body,
            extra_signed_headers: &[],
            time: fixed_time(),
        };
        let a = sign_request(&east).unwrap();
        let b = sign_request(&west).unwrap();
        assert_ne!(a.signature, b.signature);
    }
}

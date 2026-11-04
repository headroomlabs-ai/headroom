//! Upstream path construction that never rewrites what the client asked for.
//!
//! A reverse proxy has one invariant on the request line: the path it sends
//! upstream is the operator-configured base prefix followed by exactly the
//! path the client sent. Two things break that invariant silently:
//!
//! * [`url::Url::set_path`] normalises dot segments (`.` and `..`, in any
//!   percent-encoded spelling) and folds `\` into `/`. A client could walk
//!   out of a tenant prefix (`https://gw/tenant-a` + `/../tenant-b/...`) or,
//!   on the provider routes, move a SigV4- or ADC-signed request to a
//!   different resource (`/model/../../other/invoke`).
//! * Route parameters arrive percent-decoded from axum. Interpolating them
//!   back into a path string re-parses `/` as a segment separator, so a model
//!   id that legitimately contains `/` (Bedrock inference-profile ARNs) is
//!   split into two segments, and a hostile one gains segments.
//!
//! Every upstream URL in the proxy goes through this module: raw request
//! paths through [`join_request_path`], decoded route parameters through
//! [`append_segments`]. Anything the module would have to rewrite is
//! rejected instead, and the caller answers the client with 400.

use url::Url;

/// Why a path could not be forwarded unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    /// A `.` or `..` segment (including `%2e` spellings). The proxy does not
    /// resolve these; the client must send the path it means.
    #[error("path contains a dot segment {0:?}; the proxy does not normalise paths")]
    DotSegment(String),
    /// A backslash, which the URL standard folds into `/` for http(s).
    #[error("path contains a backslash; the proxy does not normalise paths")]
    Backslash,
    /// A decoded route parameter was empty.
    #[error("path segment is empty")]
    EmptySegment,
    /// A decoded route parameter carried a control character.
    #[error("path segment contains a control character")]
    ControlCharacter,
    /// Post-condition failure: the joined path no longer sits under the
    /// configured base path. Guards against future changes to the join
    /// logic; unreachable while [`validate_request_path`] runs first.
    #[error("joined path {joined:?} escapes the configured base path {base:?}")]
    EscapesBase { base: String, joined: String },
    /// The upstream URL is not hierarchical (`mailto:`-style), so it cannot
    /// carry path segments. Not reachable with http(s)/ws(s) upstreams.
    #[error("upstream URL cannot carry a path")]
    CannotBeABase,
}

/// True when a raw (still percent-encoded) segment is a dot segment under
/// the WHATWG URL standard, which treats `%2e` as `.` for this purpose.
fn is_dot_segment(segment: &str) -> bool {
    matches!(
        segment.to_ascii_lowercase().as_str(),
        "." | ".." | "%2e" | ".%2e" | "%2e." | "%2e%2e"
    )
}

/// Reject a raw request path that `Url::set_path` would rewrite.
///
/// `path` is the request-target path exactly as received (percent-encoded,
/// no query). Only structural rewrites are rejected — dot segments and
/// backslashes; ordinary percent-encoding of reserved characters is fine
/// because it does not change which resource the path names.
pub fn validate_request_path(path: &str) -> Result<(), PathError> {
    if path.contains('\\') {
        return Err(PathError::Backslash);
    }
    if let Some(bad) = path.split('/').find(|segment| is_dot_segment(segment)) {
        return Err(PathError::DotSegment(bad.to_string()));
    }
    Ok(())
}

/// Append the client's raw request path (and query) to the configured
/// upstream base, preserving any base path prefix.
///
/// `"http://x:1/api"` + `"/v1/foo"` → `"http://x:1/api/v1/foo"`. The result
/// is guaranteed to start with the base path: the input is validated first
/// and the output is checked afterwards.
pub fn join_request_path(base: &Url, path: &str, query: Option<&str>) -> Result<Url, PathError> {
    validate_request_path(path)?;

    let mut joined = base.clone();
    let base_path = joined.path().trim_end_matches('/').to_string();
    let combined = if path.is_empty() || path == "/" {
        if base_path.is_empty() {
            "/".to_string()
        } else {
            base_path.clone()
        }
    } else if base_path.is_empty() {
        path.to_string()
    } else {
        format!("{base_path}{path}")
    };
    joined.set_path(&combined);
    joined.set_query(query);

    if !base_path.is_empty() {
        let got = joined.path();
        if got != base_path && !got.starts_with(&format!("{base_path}/")) {
            return Err(PathError::EscapesBase {
                base: base_path,
                joined: got.to_string(),
            });
        }
    }
    Ok(joined)
}

/// Reject a decoded route parameter that cannot be a single path segment.
///
/// `/` is allowed: [`append_segments`] percent-encodes it, so a Bedrock
/// inference-profile ARN stays one segment on the wire (`%2F`), exactly as
/// the AWS SDKs send it.
pub fn validate_segment(segment: &str) -> Result<(), PathError> {
    if segment.is_empty() {
        return Err(PathError::EmptySegment);
    }
    if matches!(segment, "." | "..") {
        return Err(PathError::DotSegment(segment.to_string()));
    }
    if segment.chars().any(char::is_control) {
        return Err(PathError::ControlCharacter);
    }
    Ok(())
}

/// Append already-decoded segments to `base` as individual path segments,
/// percent-encoding each one so it stays a single segment on the wire.
///
/// Keeps any path prefix on `base` (`https://gw/prefix/` + `["model", id,
/// "invoke"]` → `/prefix/model/<id>/invoke`), never produces a double slash,
/// and refuses segments that would change the path's shape.
pub fn append_segments<'a, I>(base: &Url, segments: I) -> Result<Url, PathError>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut joined = base.clone();
    {
        let mut path = joined
            .path_segments_mut()
            .map_err(|()| PathError::CannotBeABase)?;
        path.pop_if_empty();
        for segment in segments {
            validate_segment(segment)?;
            path.push(segment);
        }
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(s: &str) -> Url {
        Url::parse(s).expect("test base url")
    }

    #[test]
    fn join_preserves_base_prefix_and_query() {
        let out =
            join_request_path(&base("http://up:8080/api"), "/v1/messages", Some("a=1")).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/api/v1/messages?a=1");
    }

    #[test]
    fn join_root_forms() {
        assert_eq!(
            join_request_path(&base("http://up:8080/"), "/", None)
                .unwrap()
                .as_str(),
            "http://up:8080/"
        );
        assert_eq!(
            join_request_path(&base("http://up:8080/api/"), "/", None)
                .unwrap()
                .as_str(),
            "http://up:8080/api"
        );
        assert_eq!(
            join_request_path(&base("http://up:8080"), "/v1", None)
                .unwrap()
                .as_str(),
            "http://up:8080/v1"
        );
    }

    #[test]
    fn join_rejects_every_dot_segment_spelling() {
        for path in [
            "/../tenant-b/v1",
            "/v1/./x",
            "/v1/%2e%2e/x",
            "/v1/%2E%2E/x",
            "/v1/.%2e/x",
            "/v1/%2e./x",
            "/v1/%2e/x",
            "/..",
        ] {
            let err = join_request_path(&base("http://gw/tenant-a"), path, None).unwrap_err();
            assert!(matches!(err, PathError::DotSegment(_)), "{path}: {err:?}");
        }
    }

    #[test]
    fn join_rejects_backslash() {
        let err =
            join_request_path(&base("http://gw/tenant-a"), "/..\\tenant-b", None).unwrap_err();
        assert_eq!(err, PathError::Backslash);
    }

    #[test]
    fn join_rejects_dot_segments_even_without_a_base_prefix() {
        // No prefix to escape, but the upstream would still receive a
        // different path than the client sent (`/x` instead of `/a/../x`).
        let err = join_request_path(&base("http://up"), "/a/../x", None).unwrap_err();
        assert!(matches!(err, PathError::DotSegment(_)));
    }

    #[test]
    fn join_keeps_encoded_slashes_as_one_segment() {
        // `..%2F..` is not a dot segment; it stays a literal segment upstream.
        let out = join_request_path(&base("http://gw/tenant-a"), "/v1/..%2F..%2Fx", None).unwrap();
        assert_eq!(out.path(), "/tenant-a/v1/..%2F..%2Fx");
    }

    #[test]
    fn append_encodes_slashes_inside_a_segment() {
        let arn =
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5";
        let out = append_segments(
            &base("https://bedrock-runtime.us-east-1.amazonaws.com/"),
            ["model", arn, "invoke"],
        )
        .unwrap();
        assert_eq!(
            out.as_str(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Fus.anthropic.claude-3-5/invoke"
        );
    }

    #[test]
    fn append_keeps_base_prefix_without_double_slash() {
        let out = append_segments(&base("https://gw/prefix/"), ["model", "m", "invoke"]).unwrap();
        assert_eq!(out.path(), "/prefix/model/m/invoke");
        let out = append_segments(&base("https://gw"), ["model", "m", "invoke"]).unwrap();
        assert_eq!(out.path(), "/model/m/invoke");
    }

    #[test]
    fn append_rejects_traversal_and_malformed_segments() {
        let b = base("https://gw/");
        // What `..%2F..%2Fother` decodes to: a `/`-containing segment is
        // fine (it is re-encoded and stays one segment); a bare `..` is not.
        assert_eq!(
            append_segments(&b, ["model", "../../other", "invoke"])
                .unwrap()
                .path(),
            "/model/..%2F..%2Fother/invoke"
        );
        assert!(matches!(
            append_segments(&b, ["model", "..", "invoke"]).unwrap_err(),
            PathError::DotSegment(_)
        ));
        assert!(matches!(
            append_segments(&b, ["model", ".", "invoke"]).unwrap_err(),
            PathError::DotSegment(_)
        ));
        assert_eq!(
            append_segments(&b, ["model", "", "invoke"]).unwrap_err(),
            PathError::EmptySegment
        );
        assert_eq!(
            append_segments(&b, ["model", "m\r\nX-Injected: 1", "invoke"]).unwrap_err(),
            PathError::ControlCharacter
        );
    }

    #[test]
    fn append_does_not_alter_ordinary_model_ids() {
        let out = append_segments(
            &base("https://bedrock-runtime.us-west-2.amazonaws.com/"),
            [
                "model",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "converse",
            ],
        )
        .unwrap();
        assert_eq!(
            out.as_str(),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse"
        );
    }
}

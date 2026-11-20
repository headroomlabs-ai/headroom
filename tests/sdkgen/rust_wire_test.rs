use headroom_generated_pilot::{Client, Error, Options, RetrieveRequest};
use serde_json::{Map, Value};
use std::time::Duration;

#[test]
fn additional_properties_cannot_shadow_declared_request_fields() {
    let request = RetrieveRequest {
        hash: "ok".into(),
        additional_properties: Map::from_iter([("hash".into(), Value::String("other".into()))]),
    };
    assert!(serde_json::to_string(&request).is_err(), "extension shadowed hash");
}

fn client() -> Client {
    client_with(Options::default())
}

fn client_with(options: Options) -> Client {
    Client::new(
        &std::env::var("HEADROOM_WIRE_TEST_URL").expect("HEADROOM_WIRE_TEST_URL"),
        options,
    )
    .expect("valid fixture URL")
}

#[tokio::test]
async fn post_and_get_preserve_wire_values() {
    let response = client()
        .retrieve(&RetrieveRequest {
            hash: "ok".into(),
            additional_properties: Map::from_iter([(
                "extra_request".into(),
                Value::String("unchanged".into()),
            )]),
        })
        .await
        .expect("POST retrieval succeeds");
    assert_eq!(response.tool_name, None);
    assert_eq!(response.original_content, r#"{"snake_case":"世界"}"#);
    assert_eq!(response.additional_properties["future_extension"]["snake_case"], "unchanged");

    let key = "a/世界 ?#+'!*()";
    let by_hash = client().retrieve_get(key).await.expect("GET retrieval succeeds");
    assert_eq!(by_hash.hash, key);
}

#[tokio::test]
async fn http_error_retains_body_without_displaying_it() {
    let error = client()
        .retrieve(&RetrieveRequest {
            hash: "missing".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect_err("missing entry returns an API error");
    match error {
        Error::API(api) => {
            assert_eq!(api.status, 404);
            assert!(String::from_utf8_lossy(&api.body).contains("Entry missing"));
            assert!(!api.to_string().contains("Entry missing"));
        }
        other => panic!("expected API error, got {other}"),
    }
}

#[tokio::test]
async fn malformed_redirect_and_bounded_responses_fail_closed() {
    for key in ["missing_field", "wrong_type", "null_nonnullable", "notjson", "valid_nonjson"] {
        let error = client()
            .retrieve(&RetrieveRequest {
                hash: key.into(),
                additional_properties: Map::new(),
            })
            .await
            .expect_err("malformed response must fail");
        assert!(matches!(error, Error::Protocol(_)), "unexpected error: {error}");
    }

    let secret = client()
        .retrieve(&RetrieveRequest { hash: "secret_wrong_type".into(), additional_properties: Map::new() })
        .await
        .expect_err("wrong type must fail");
    assert!(!secret.to_string().contains("sdkgen-secret-marker"));

    let retry = client()
        .retrieve(&RetrieveRequest { hash: "retry_probe_rust".into(), additional_properties: Map::new() })
        .await;
    assert!(retry.is_err(), "closed connection must not be retried");

    let wide = client()
        .retrieve(&RetrieveRequest {
            hash: "unsafe_int".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect("i64 wire value is supported");
    assert_eq!(wide.original_tokens, 9_007_199_254_740_993);

    let redirect = client()
        .retrieve(&RetrieveRequest {
            hash: "redirect".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect_err("redirect must not be followed");
    assert!(matches!(redirect, Error::API(error) if error.status == 302));

    let bounded = client_with(Options {
        timeout: Duration::from_secs(30),
        max_response_bytes: 16,
    });
    let oversized = bounded
        .retrieve(&RetrieveRequest {
            hash: "ok".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect_err("oversized response must fail");
    assert!(matches!(oversized, Error::Protocol(_)));
}

#[tokio::test]
async fn invalid_paths_base_urls_and_timeout_fail_closed() {
    let dot = client().retrieve_get("..").await.expect_err("dot segment must fail");
    assert!(matches!(dot, Error::Protocol(_)));

    for base in ["file:///tmp", "http://user:pass@localhost", "http://localhost/?secret=1"] {
        assert!(Client::new(base, Options::default()).is_err(), "accepted {base}");
    }

    let timed = client_with(Options {
        timeout: Duration::from_millis(50),
        max_response_bytes: 1024,
    });
    let timeout = timed
        .retrieve(&RetrieveRequest {
            hash: "slow".into(),
            additional_properties: Map::new(),
        })
        .await
        .expect_err("slow response must time out");
    assert!(matches!(timeout, Error::Transport(_)));

    let cancelled = tokio::time::timeout(
        Duration::from_millis(25),
        client().retrieve(&RetrieveRequest { hash: "slow".into(), additional_properties: Map::new() }),
    ).await;
    assert!(cancelled.is_err(), "caller cancellation did not interrupt the request");
}

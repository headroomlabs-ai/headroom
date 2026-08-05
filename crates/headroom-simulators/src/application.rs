use bytes::Bytes;

use crate::config::{ConfiguredResponse, SimulatorConfig, StubRule};
use crate::domain::{default_response, RequestFacts, SimulatedResponse};

#[derive(Debug, Clone)]
pub struct Simulator {
    config: SimulatorConfig,
}

impl Simulator {
    pub fn new(config: SimulatorConfig) -> Self {
        Self { config }
    }

    pub fn simulate(&self, facts: &RequestFacts) -> SimulatedResponse {
        if let Some(rule) = self.config.stubs.iter().find(|rule| rule.matches(facts)) {
            tracing::info!(
                event = "simulator_stub_matched",
                stub = %rule.name,
                path = %facts.path,
                provider_path = facts.provider_path.label(),
                "configured simulator stub matched request"
            );
            return configured_response(&rule.response);
        }
        tracing::info!(
            event = "simulator_default_response",
            path = %facts.path,
            provider_path = facts.provider_path.label(),
            "using bottled simulator response"
        );
        default_response(facts)
    }
}

trait MatchesRequest {
    fn matches(&self, facts: &RequestFacts) -> bool;
}

impl MatchesRequest for StubRule {
    fn matches(&self, facts: &RequestFacts) -> bool {
        if let Some(method) = &self.method {
            if !method.eq_ignore_ascii_case(&facts.method) {
                return false;
            }
        }
        if self.path != facts.path {
            return false;
        }
        if let Some(needle) = &self.body_contains {
            let haystack = String::from_utf8_lossy(&facts.body);
            if !haystack.contains(needle) {
                return false;
            }
        }
        if let Some(pointer_match) = &self.body_json_pointer {
            let Some(json) = &facts.json else {
                return false;
            };
            if json.pointer(&pointer_match.pointer) != Some(&pointer_match.equals) {
                return false;
            }
        }
        true
    }
}

fn configured_response(config: &ConfiguredResponse) -> SimulatedResponse {
    let mut response = if !config.sse.is_empty() {
        let mut body = String::new();
        for frame in &config.sse {
            if let Some(event) = &frame.event {
                body.push_str("event: ");
                body.push_str(event);
                body.push('\n');
            }
            body.push_str("data: ");
            body.push_str(&serde_json::to_string(&frame.data).expect("stub data serializes"));
            body.push_str("\n\n");
        }
        SimulatedResponse::text(config.status, "text/event-stream", body)
    } else if let Some(json) = &config.json {
        SimulatedResponse::json(config.status, json.clone())
    } else {
        SimulatedResponse::text(
            config.status,
            "text/plain; charset=utf-8",
            Bytes::from(config.body.clone().unwrap_or_default()),
        )
    };
    response.headers.extend(config.headers.clone());
    response
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;
    use serde_json::json;
    use serde_json::Value;

    use super::*;
    use crate::config::{ConfiguredResponse, JsonPointerMatch, StubRule};

    #[test]
    fn exact_stub_overrides_bottled_default() {
        let simulator = Simulator::new(SimulatorConfig {
            stubs: vec![StubRule {
                name: "chat-ping".to_string(),
                method: Some("POST".to_string()),
                path: "/v1/chat/completions".to_string(),
                body_contains: None,
                body_json_pointer: Some(JsonPointerMatch {
                    pointer: "/messages/0/content".to_string(),
                    equals: json!("ping"),
                }),
                response: ConfiguredResponse {
                    status: 202,
                    headers: Default::default(),
                    json: Some(json!({"configured": true})),
                    body: None,
                    sse: vec![],
                },
            }],
        });
        let facts = RequestFacts::new(
            "POST",
            "/v1/chat/completions",
            &HeaderMap::new(),
            Bytes::from_static(br#"{"messages":[{"content":"ping"}]}"#),
        );
        let response = simulator.simulate(&facts);
        assert_eq!(response.status, 202);
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body, json!({"configured": true}));
    }
}

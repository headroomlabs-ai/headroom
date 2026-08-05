use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{HeaderName, Request, Response, StatusCode};
use axum::routing::{any, get};
use axum::Router;

use crate::application::Simulator;
use crate::domain::{RequestFacts, SimulatedResponse};

const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

pub fn build_app(simulator: Simulator) -> Router {
    let state = Arc::new(simulator);
    Router::new()
        .route("/healthz", get(healthz))
        .fallback(any(simulate))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn simulate(State(simulator): State<Arc<Simulator>>, req: Request<Body>) -> Response<Body> {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let headers = req.headers().clone();
    let body = match to_bytes(req.into_body(), MAX_BODY_BYTES).await {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!(event = "simulator_body_read_failed", error = %err);
            return response_from(SimulatedResponse::text(
                413,
                "application/json",
                r#"{"error":"request body too large or unreadable"}"#,
            ));
        }
    };
    let facts = RequestFacts::new(&method, &path, &headers, body);
    response_from(simulator.simulate(&facts))
}

fn response_from(simulated: SimulatedResponse) -> Response<Body> {
    let status = StatusCode::from_u16(simulated.status).unwrap_or(StatusCode::OK);
    let mut builder = Response::builder()
        .status(status)
        .header(http::header::CONTENT_TYPE, simulated.content_type)
        .header("x-headroom-simulator", "true");
    for (name, value) in simulated.headers {
        match HeaderName::from_bytes(name.as_bytes()) {
            Ok(header_name) => {
                builder = builder.header(header_name, value);
            }
            Err(err) => {
                tracing::warn!(
                    event = "simulator_invalid_response_header",
                    header = %name,
                    error = %err,
                    "configured simulator header ignored"
                );
            }
        }
    }
    builder
        .body(Body::from(simulated.body))
        .expect("simulator response builds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulatorConfig;

    #[tokio::test]
    async fn health_route_returns_ok() {
        let app = build_app(Simulator::new(SimulatorConfig::default()));
        let response = tower::ServiceExt::oneshot(
            app,
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}

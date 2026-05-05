//! headroom-proxy: transparent reverse proxy binary.
//!
//! Drops in front of the existing Python proxy. End-users hit the public
//! port; this binary forwards every HTTP/SSE/WebSocket request verbatim to
//! `--upstream`. See RUST_DEV.md for the operator runbook.

use std::net::SocketAddr;

use clap::Parser;
use headroom_proxy::config::CliArgs;
use headroom_proxy::{build_app, AppState, Config};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = CliArgs::parse();
    let config = Config::from_cli(args);

    init_tracing(&config.log_level);

    tracing::info!(
        listen = %config.listen,
        upstream = %config.upstream,
        upstream_timeout_s = config.upstream_timeout.as_secs(),
        upstream_connect_timeout_s = config.upstream_connect_timeout.as_secs(),
        max_body_bytes = config.max_body_bytes,
        rewrite_host = config.rewrite_host,
        graceful_shutdown_timeout_s = config.graceful_shutdown_timeout.as_secs(),
        "headroom-proxy starting"
    );

    let mut state = AppState::new(config.clone())?;

    // PR-D1: resolve AWS credentials at startup via the `aws-config`
    // default chain. Loaded once so per-request signing is cheap.
    // Failure is NOT fatal — the proxy may run in front of a non-AWS
    // upstream — but the Bedrock invoke handler refuses to forward
    // unsigned requests when `bedrock_credentials` is `None`
    // (see `bedrock::invoke::handle_invoke`).
    if config.enable_bedrock_native {
        match load_bedrock_credentials(&config).await {
            Ok(creds) => {
                state = state.with_bedrock_credentials(creds);
                tracing::info!(
                    event = "bedrock_credentials_loaded",
                    region = %config.bedrock_region,
                    profile = ?config.aws_profile,
                    "AWS credentials resolved for Bedrock SigV4 signing"
                );
            }
            Err(e) => {
                tracing::warn!(
                    event = "bedrock_credentials_unavailable",
                    region = %config.bedrock_region,
                    profile = ?config.aws_profile,
                    error = %e,
                    "AWS credentials not available at startup; Bedrock invoke will 5xx until creds are configured"
                );
            }
        }
    }

    let app = build_app(state).into_make_service_with_connect_info::<SocketAddr>();

    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    tracing::info!(addr = %listener.local_addr()?, "listening");

    let grace = config.graceful_shutdown_timeout;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            tracing::info!(
                timeout_s = grace.as_secs(),
                "draining in-flight requests before exit"
            );
            tokio::time::sleep(grace).await;
        })
        .await?;

    Ok(())
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(false)
        .with_span_list(false);
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(json_layer)
        .try_init();
}

/// PR-D1: resolve AWS credentials for Bedrock SigV4 signing.
///
/// Uses the `aws-config` default chain (env vars → shared profile
/// file → IMDS / ECS task role). Honours `Config::aws_profile` when
/// set; otherwise the chain picks up `AWS_PROFILE` from the
/// environment automatically.
async fn load_bedrock_credentials(
    config: &Config,
) -> Result<aws_credential_types::Credentials, Box<dyn std::error::Error + Send + Sync>> {
    use aws_config::BehaviorVersion;
    use aws_credential_types::provider::ProvideCredentials;

    let mut loader = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new(config.bedrock_region.clone()));
    if let Some(profile) = config.aws_profile.as_deref() {
        loader = loader.profile_name(profile);
    }
    let aws_config = loader.load().await;
    let creds_provider = aws_config
        .credentials_provider()
        .ok_or("no credentials provider configured")?;
    let creds = creds_provider.provide_credentials().await?;
    Ok(creds)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}

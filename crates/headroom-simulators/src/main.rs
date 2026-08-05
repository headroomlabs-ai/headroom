use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use headroom_simulators::{build_app, load_config, Simulator};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Deterministic local upstream simulators for Headroom tests")]
struct Cli {
    #[arg(
        long,
        env = "HEADROOM_SIMULATOR_LISTEN",
        default_value = "127.0.0.1:8789"
    )]
    listen: SocketAddr,
    #[arg(long, env = "HEADROOM_SIMULATOR_CONFIG")]
    config: Option<PathBuf>,
    #[arg(long, env = "HEADROOM_SIMULATOR_LOG", default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    let config = load_config(cli.config.as_deref())?;
    let simulator = Simulator::new(config);
    let app = build_app(simulator);
    let listener = tokio::net::TcpListener::bind(cli.listen).await?;
    tracing::info!(addr = %listener.local_addr()?, "headroom simulator listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
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
}

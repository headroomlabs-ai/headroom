//! Entry point. Bootstrap only — all logic lives in the library crate.
use creative_tui::app;
use creative_tui::types::AppConfig;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let app = app::App::new();
    let config = AppConfig::default();

    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (shell_tx, shell_rx) = mpsc::channel(32);

    app::spawn_shell_worker(app.uniforms.clone(), cmd_rx, shell_tx);
    app::spawn_graphics(&app, &config);
    app::run_tui(&app, cmd_tx, shell_rx).await;
}

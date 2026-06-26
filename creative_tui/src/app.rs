//! Application orchestrator — wires layers together.
//!
//! # Entry points from the user's perspective
//!
//! ```text
//! ┌──────────────┐    typed commands     ┌──────────────┐
//! │   Terminal   │ ──────────────────►   │   Shell      │
//! │   (tui)      │ ◄──────────────────   │   (shell)    │
//! └──────┬───────┘    log responses       └──────┬───────┘
//!        │                                       │
//!        │ Arc<Mutex<Uniforms>>                  │ mutates
//!        │                                       │
//! ┌──────▼───────┐                        ┌──────▼───────┐
//! │   Graphics   │  reads uniforms every  │   Uniforms   │
//! │   (gfx)      │  frame via Arc<Mutex>  │   (types)    │
//! └──────────────┘                        └──────────────┘
//! ```
//!
//! # Data flow
//!
//! 1. User types into TUI shell prompt → `KeyAction::Submit(text)`
//! 2. `Command::parse(text)` creates typed command
//! 3. `shell::dispatch(command, &mut uniforms)` mutates shared state
//! 4. Response logged to TUI; GPU thread picks up new uniforms next frame
//!
//! # Thread model
//!
//! - **Main thread** — tokio runtime, TUI render loop, input polling
//! - **Graphics thread** — winit event loop, wgpu rendering
//! - **Command worker** — tokio task, processes commands from channel

use std::sync::{Arc, Mutex};

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use tokio::sync::mpsc;

use crate::shell;
use crate::tui;
use crate::types::{AppConfig, Command, LogBuffer, ShellResponse, Uniforms};

/// Shared application state that crosses thread boundaries.
pub struct App {
    pub uniforms: Arc<Mutex<Uniforms>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            uniforms: Arc::new(Mutex::new(Uniforms::default())),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the TUI loop on the main thread. Returns when the user quits.
///
/// Channels:
/// - `cmd_tx` — send parsed commands to the shell worker
/// - `shell_rx` — receive responses from the shell worker
pub async fn run_tui(
    app: &App,
    cmd_tx: mpsc::Sender<Command>,
    mut shell_rx: mpsc::Receiver<ShellResponse>,
) {
    let mut terminal = tui::terminal::TerminalUi::new().expect("terminal setup");
    let mut input = String::new();
    let mut log = LogBuffer::new(256);
    log.push("creative-tui v0.2 -- shader + tui + shell + nix");
    log.push("commands: speed N | color R G B | time");

    loop {
        // ── draw frame (no clone — LogBuffer view is borrowed) ──
        terminal
            .draw(|f| {
                draw_frame(f, &app.uniforms, &log, &input);
            })
            .expect("terminal draw");

        // ── drain shell responses ──
        while let Ok(msg) = shell_rx.try_recv() {
            log.push(&msg);
        }

        let key = tui::input::poll_input();
        match key {
            Some(tui::input::KeyAction::Quit) => break,
            Some(tui::input::KeyAction::Append(c)) => input.push(c),
            Some(tui::input::KeyAction::Backspace) => {
                input.pop();
            }
            _ => {}
        }

        // Enter key handling (not in input poller — needs command context)
        if crossterm::event::poll(std::time::Duration::from_millis(16)).unwrap_or(false) {
            if let Ok(crossterm::event::Event::Key(key)) = crossterm::event::read() {
                if key.kind == crossterm::event::KeyEventKind::Press
                    && key.code == crossterm::event::KeyCode::Enter
                    && !input.is_empty()
                {
                    if let Some(cmd) = Command::parse(&input) {
                        log.push(format_args!("> {}", input));
                        let _ = cmd_tx.send(cmd).await;
                        input.clear();
                    }
                }
            }
        }
    }

    terminal.teardown().expect("terminal teardown");
}

/// Render one TUI frame.
/// `log` is borrowed — no allocation per frame.
fn draw_frame(
    f: &mut ratatui::Frame,
    uniforms: &Arc<Mutex<Uniforms>>,
    log: &LogBuffer,
    input: &str,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Min(1),
                Constraint::Length(3),
            ]
            .as_ref(),
        )
        .split(f.size());

    let u = uniforms.lock().unwrap();

    // status bar
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                " creative-tui -- shaders + shell + nix ",
                Style::default().fg(Color::Yellow),
            )),
            Line::from(format!(
                " speed: {:.1} | color: ({:.2}, {:.2}, {:.2}) | time: {:.2}",
                u.speed, u.color_r, u.color_g, u.color_b, u.time
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title(" status ")),
        chunks[0],
    );

    // log (view last 20 entries, no allocation)
    let lines: Vec<Line> = log.view_recent(20).map(Line::from).collect();
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" log ")),
        chunks[1],
    );

    // shell prompt
    f.render_widget(
        Paragraph::new(format!("> {}", input))
            .block(Block::default().borders(Borders::ALL).title(" shell ")),
        chunks[2],
    );
}

/// Run the shell command worker as a tokio task.
/// Reads commands from `cmd_rx`, dispatches to `shell::dispatch`,
/// sends responses to `shell_tx`.
pub fn spawn_shell_worker(
    uniforms: Arc<Mutex<Uniforms>>,
    mut cmd_rx: mpsc::Receiver<Command>,
    shell_tx: mpsc::Sender<ShellResponse>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            let response = {
                let mut u = uniforms.lock().unwrap();
                shell::dispatch(cmd, &mut u)
            };
            let _ = shell_tx.send(response).await;
        }
    })
}

/// Boot the graphics thread. Spawns an OS thread that runs the winit
/// event loop + wgpu rendering.
pub fn spawn_graphics(app: &App, config: &AppConfig) {
    let uniforms = Arc::clone(&app.uniforms);
    let cfg = config.clone();
    std::thread::spawn(move || crate::gfx_thread::run(uniforms, cfg));
}

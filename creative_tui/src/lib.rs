//! creative-tui — creative coding shader canvas with terminal shell control.
//!
//! # Architecture
//!
//! ```text
//! main.rs (bootstrap, ~6 LOC)
//!   └─ app.rs (orchestrator)
//!        ├─ tui::terminal  (ratatui draw loop)
//!        ├─ tui::input     (keyboard polling)
//!        ├─ shell          (command dispatch, pure fn)
//!        ├─ types          (Uniforms, Command, Response, Config)
//!        ├─ gfx_thread     (winit event loop)
//!        │    ├─ gfx::resources  (GpuDevice)
//!        │    ├─ gfx::pipeline   (PipelineBuilder)
//!        │    └─ gfx::renderer   (Renderer)
//!        └─ gfx::*
//! ```
//!
//! # User-facing entry points
//!
//! | Entry point | What it does |
//! |------------|-------------|
//! | `main()` in `main.rs` | Boots app, spawns graphics thread, runs TUI loop |
//! | `app::run_tui()` | Terminal UI loop with keyboard input + log |
//! | `app::spawn_graphics()` | Spawns OS thread for shader window |
//! | `app::spawn_shell_worker()` | tokio task for command dispatch |
//! | `shell::dispatch()` | Pure function: Command → Uniforms mutation + Response |
//! | `gfx_thread::run()` | winit event loop + wgpu frame rendering |

pub mod app;
pub mod gfx;
mod gfx_thread;
pub mod shell;
pub mod tui;
pub mod types;

//! TUI layer — crossterm terminal + ratatui rendering.
//!
//! [`TerminalUi`] owns setup, draw loop, and teardown. The caller
//! supplies a callback that receives a [`ratatui::Frame`] and the
//! current app snapshot.

use std::io;

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;

/// Wraps a ratatui terminal with crossterm lifecycle management.
pub struct TerminalUi {
    terminal: ratatui::Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalUi {
    /// Enter raw mode + alternate screen. Returns the ready terminal.
    pub fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = ratatui::Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Draw one frame. The closure receives `&mut Frame` — keep it fast.
    pub fn draw(&mut self, render: impl FnOnce(&mut ratatui::Frame)) -> io::Result<()> {
        self.terminal.draw(render)?;
        Ok(())
    }

    /// Teardown: leave alternate screen, restore raw mode, show cursor.
    pub fn teardown(&mut self) -> io::Result<()> {
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    /// Mutable access to the backend (for raw-mode cursor etc.).
    pub fn backend_mut(&mut self) -> &mut CrosstermBackend<io::Stdout> {
        self.terminal.backend_mut()
    }
}

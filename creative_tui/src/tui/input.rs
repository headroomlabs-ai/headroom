//! TUI layer — keyboard input polling.

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::Duration;

/// Decoded keyboard action the caller should act on.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyAction {
    Submit(String),
    Quit,
    Append(char),
    Backspace,
    None,
}

/// Poll for keyboard input with a 16ms timeout.
/// Returns the first action found (or `None` if no input).
pub fn poll_input() -> Option<KeyAction> {
    if !event::poll(Duration::from_millis(16)).unwrap_or(false) {
        return None;
    }
    let Ok(Event::Key(key)) = event::read() else {
        return None;
    };
    if key.kind != KeyEventKind::Press {
        return None;
    }
    match key.code {
        KeyCode::Enter => None, // caller handles Submit separately
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Some(KeyAction::Quit)
        }
        KeyCode::Char(c) => Some(KeyAction::Append(c)),
        KeyCode::Backspace => Some(KeyAction::Backspace),
        _ => None,
    }
}

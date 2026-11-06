//! Per-CLI transcript adapters.
//!
//! Phase 1: only Claude Code (`claude_code.rs`).
//! Phase 2 will add Codex and Gemini CLI adapters here.

pub mod claude_code;

/// A normalized transcript across coding-agent CLIs.
#[derive(Debug, Default)]
pub struct Transcript {
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone)]
pub struct Block {
    /// e.g., "Bash", "Read", "Edit", "Grep", "Glob", "Agent",
    /// "mcp__codebase-memory-mcp__search_graph", "<system-reminder>", etc.
    pub tool: String,
    /// Raw text content of the block (untokenized).
    pub text: String,
}

impl Transcript {
    pub fn push(&mut self, tool: impl Into<String>, text: impl Into<String>) {
        self.blocks.push(Block {
            tool: tool.into(),
            text: text.into(),
        });
    }
}

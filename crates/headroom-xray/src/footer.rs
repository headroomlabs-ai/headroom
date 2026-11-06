//! Render the Headroom footer block.

use std::collections::HashMap;
use std::path::PathBuf;

const RULE: &str = "─────────────────────────────────────────────────────────────";

/// Context for rendering the footer — what scope CodeBurn was asked for,
/// and what scope the footer actually covers. The footer is honest about
/// the mismatch.
#[derive(Debug, Default)]
pub struct FooterContext {
    /// The Claude Code session JSONL the footer analyzed, if any. `None`
    /// means we didn't find a Claude Code session for this cwd (e.g., the
    /// user is in a Codex/Gemini project, or just hasn't used Claude Code
    /// here yet).
    pub session_path: Option<PathBuf>,

    /// True when CodeBurn was invoked with a fleet/aggregate query
    /// (`report`, `compare`, `month`, `optimize`, or no args = default report).
    /// In that case CodeBurn's output spans many sessions; the Headroom
    /// footer covers only one, so we add a caveat line.
    pub aggregate_query: bool,
}

fn human_tokens(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

/// Render the footer.
///
/// Phase 1 reports raw per-tool-type token counts only — no compression
/// claims, no genre-based "this looks CCR-compressible" guesses. Actual
/// compressibility is measured by `headroom xray replay` (Phase 2).
///
/// Three outcomes:
/// - No Claude Code session was found for cwd → emit a one-line "no CC
///   session here" notice (loud, not silent).
/// - Found a session but it has no scorable tool calls → returns empty.
/// - Normal case: rule-bracketed block with descriptive header,
///   top-3 tool-type token consumers, optional aggregate-query caveat,
///   and a single forward-looking line about Phase 2.
pub fn render(counts: &HashMap<String, usize>, ctx: &FooterContext) -> String {
    // Case 1: no Claude Code session was discovered for cwd. Emit a loud
    // note rather than silently skipping — Phase 1 footer is Claude-Code-only
    // by design; we owe the user a sign of that.
    if ctx.session_path.is_none() {
        let mut out = String::new();
        out.push_str(RULE);
        out.push('\n');
        out.push_str("Headroom footer: no Claude Code session detected for this directory.\n");
        out.push_str("  → Phase 1 footer scans Claude Code transcripts only.\n");
        out.push_str("    Codex / Gemini CLI / Cursor coverage arrives in Phase 2.\n");
        out.push_str(RULE);
        out.push('\n');
        return out;
    }

    if counts.is_empty() {
        return String::new();
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return String::new();
    }

    let session_name = ctx
        .session_path
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    let mut entries: Vec<(&String, &usize)> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let top: Vec<_> = entries.into_iter().take(3).collect();

    let mut out = String::new();
    out.push_str(RULE);
    out.push('\n');
    out.push_str(&format!(
        "Headroom: top tool types by token usage\n  (claude-code · {session_name})\n"
    ));
    for (tool, &count) in &top {
        let pct = (count * 100) / total.max(1);
        out.push_str(&format!(
            "  ▸ {tool:<28} {tokens:>6} tokens ({pct:>2}%)\n",
            tool = tool,
            tokens = human_tokens(count),
            pct = pct,
        ));
    }
    if ctx.aggregate_query {
        out.push_str(
            "  ⚠ CodeBurn report above spans many sessions; this view shows one session.\n",
        );
    }
    out.push_str("  → `headroom xray replay` (Phase 2) measures actual compression savings.\n");
    out.push_str(RULE);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_session(name: &str) -> FooterContext {
        FooterContext {
            session_path: Some(PathBuf::from(format!("/tmp/.claude/projects/-foo/{name}"))),
            aggregate_query: false,
        }
    }

    #[test]
    fn no_session_emits_loud_note() {
        let counts = HashMap::new();
        let ctx = FooterContext::default(); // session_path = None
        let out = render(&counts, &ctx);
        assert!(out.contains("no Claude Code session detected"));
        assert!(out.contains("Phase 2"));
        assert!(out.contains("────"));
    }

    #[test]
    fn empty_counts_with_session_returns_empty_string() {
        let counts = HashMap::new();
        let ctx = ctx_with_session("session-id.jsonl");
        assert_eq!(render(&counts, &ctx), "");
    }

    #[test]
    fn renders_top_three_sorted_descending() {
        let mut counts = HashMap::new();
        counts.insert("Bash".to_string(), 53_000);
        counts.insert("Read".to_string(), 28_000);
        counts.insert("Grep".to_string(), 100);
        counts.insert("Edit".to_string(), 4_000);
        let ctx = ctx_with_session("session-id.jsonl");
        let r = render(&counts, &ctx);
        let bash = r.find("Bash").unwrap();
        let read = r.find("Read").unwrap();
        let edit = r.find("Edit").unwrap();
        assert!(bash < read);
        assert!(read < edit);
        assert!(!r.contains("Grep"));
        assert!(r.contains("top tool types by token usage"));
        assert!(r.contains("session-id.jsonl"));
        // No compression claims in Phase 1.
        assert!(!r.contains("CCR-compressible"));
        assert!(!r.contains("ContentRouter"));
        assert!(!r.contains("opportunities"));
        // No aggregate caveat when not requested.
        assert!(!r.contains("spans many sessions"));
    }

    #[test]
    fn aggregate_query_adds_scope_caveat() {
        let mut counts = HashMap::new();
        counts.insert("Bash".to_string(), 10_000);
        let ctx = FooterContext {
            session_path: Some(PathBuf::from("/tmp/.claude/projects/-foo/sess.jsonl")),
            aggregate_query: true,
        };
        let r = render(&counts, &ctx);
        assert!(r.contains("spans many sessions"));
    }

    #[test]
    fn defers_compression_to_phase_2() {
        let mut counts = HashMap::new();
        counts.insert("Bash".to_string(), 1_000);
        let ctx = ctx_with_session("session-id.jsonl");
        let r = render(&counts, &ctx);
        // The only forward-looking line should defer measurement, not promise
        // it's available now.
        assert!(r.contains("Phase 2"));
        assert!(r.contains("measures actual compression savings"));
        assert!(!r.contains("coming soon"));
        assert!(!r.contains("exact savings"));
    }
}

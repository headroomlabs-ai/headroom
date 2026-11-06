//! `headroom-xray` binary entrypoint.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "headroom-xray", version, about, long_about = None)]
struct Cli {
    /// Suppress the Headroom footer (CodeBurn output only).
    #[arg(long, env = "HEADROOM_XRAY_NO_FOOTER")]
    no_footer: bool,

    /// Emit debug logs about the footer pipeline to stderr.
    #[arg(long)]
    xray_debug: bool,

    /// Show CodeBurn's own --help (not headroom-xray's wrapper help).
    #[arg(long, conflicts_with_all = ["no_footer", "codeburn_args"])]
    help_codeburn: bool,

    /// All arguments forwarded to CodeBurn (e.g., `report`, `today`, `optimize`).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    codeburn_args: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.xray_debug {
        tracing_subscriber::fmt()
            .with_env_filter("headroom_xray=debug")
            .with_writer(std::io::stderr)
            .init();
    }

    if let Err(e) = headroom_xray::node::check() {
        eprintln!("{e}");
        let code = match e {
            headroom_xray::node::NodeError::NotFound => 127,
            _ => 1,
        };
        std::process::exit(code);
    }

    let args: Vec<String> = if cli.help_codeburn {
        vec!["--help".to_string()]
    } else {
        cli.codeburn_args.clone()
    };

    let code = headroom_xray::codeburn::run(&args, None)
        .await
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            1
        });

    // Footer pipeline (best-effort, never breaks the main flow).
    if !cli.no_footer && code == 0 {
        let aggregate = is_aggregate_codeburn_query(&cli.codeburn_args);
        if let Err(e) = print_footer(aggregate).await {
            if cli.xray_debug {
                eprintln!("[xray-debug] footer suppressed: {e}");
            }
        }
    }

    std::process::exit(code);
}

/// Detect whether the CodeBurn invocation produces a fleet/aggregate view
/// (spanning many sessions) vs. a single-session view. The Headroom footer
/// only ever shows one session, so we surface a scope caveat when the
/// CodeBurn output above is broader.
///
/// CodeBurn's default (no args) is `report` = 30-day fleet, so empty args
/// count as aggregate. The only common single-session-ish CodeBurn
/// subcommand is `today`. Everything else (report, month, compare,
/// optimize, yield, models, by-task, status, export) spans many sessions.
/// We err on the side of marking things aggregate — a spurious caveat is
/// cheaper than a missing one.
fn is_aggregate_codeburn_query(args: &[String]) -> bool {
    !args.iter().any(|a| a == "today")
}

async fn print_footer(aggregate_query: bool) -> Result<()> {
    use headroom_xray::footer::{self, FooterContext};
    use headroom_xray::tokenize::count_by_tool;
    use headroom_xray::transcripts::claude_code;

    let session_path = claude_code::latest_session_for_cwd();

    // No Claude Code session in this cwd: render the "no CC session" notice
    // (Phase 1 footer is Claude-Code-only — say so loudly, don't skip silently).
    let Some(session) = session_path.clone() else {
        let ctx = FooterContext {
            session_path: None,
            aggregate_query,
        };
        let rendered = footer::render(&Default::default(), &ctx);
        if !rendered.is_empty() {
            print!("\n{rendered}");
        }
        return Ok(());
    };

    let transcript = claude_code::parse(&session)?;
    let counts = count_by_tool(&transcript)?;
    let ctx = FooterContext {
        session_path: Some(session),
        aggregate_query,
    };
    let rendered = footer::render(&counts, &ctx);
    if !rendered.is_empty() {
        print!("\n{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_aggregate_codeburn_query;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn empty_args_are_aggregate() {
        assert!(is_aggregate_codeburn_query(&args(&[])));
    }

    #[test]
    fn report_is_aggregate() {
        assert!(is_aggregate_codeburn_query(&args(&["report"])));
        assert!(is_aggregate_codeburn_query(&args(&[
            "report", "--since", "7d"
        ])));
    }

    #[test]
    fn today_is_not_aggregate() {
        assert!(!is_aggregate_codeburn_query(&args(&["today"])));
    }

    #[test]
    fn status_is_aggregate() {
        // `status` shows recent activity across sessions, not one session.
        assert!(is_aggregate_codeburn_query(&args(&["status"])));
    }

    #[test]
    fn compare_is_aggregate() {
        assert!(is_aggregate_codeburn_query(&args(&["compare", "a", "b"])));
    }

    #[test]
    fn flag_only_is_aggregate_default() {
        // `--format json` with no subcommand falls back to default (report).
        assert!(is_aggregate_codeburn_query(&args(&["--format", "json"])));
    }

    #[test]
    fn today_with_flag_value_still_not_aggregate() {
        // The `today` token must be present anywhere in args, even alongside flags.
        assert!(!is_aggregate_codeburn_query(&args(&[
            "today", "--format", "json"
        ])));
    }
}

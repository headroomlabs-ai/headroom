//! `parity-run` CLI: drive the parity harness from the command line.
//!
//! Exit-code semantics (PR-I6):
//! - `Diff` always produces a non-zero exit so CI fails on real divergence.
//! - `Skipped` produces zero exit by default — this lets stub comparators
//!   (e.g. `ccr`, `log_compressor`, `cache_aligner` during Phase 0) land
//!   without blocking PRs.
//! - `--strict` (also `HEADROOM_PARITY_STRICT=1`) flips `Skipped` to
//!   non-zero. Used once Phase B/I5 promotes those stubs to real
//!   comparators so the gate refuses to regress to a stub.
//!
//! All knobs are configurable via CLI flag → env var → default.
//! Hardcodes are intentionally absent.

use anyhow::Result;
use clap::{Parser, Subcommand};
use headroom_parity::{builtin_comparators, run_comparator, RunSummary};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "parity-run",
    about = "Run Headroom Rust-vs-Python parity checks"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run all built-in comparators against fixtures under --fixtures.
    Run {
        /// Fixture root. Each transform reads from `<fixtures>/<transform>/`.
        /// Configurable via `HEADROOM_PARITY_FIXTURE_DIR` env var.
        #[arg(
            long,
            env = "HEADROOM_PARITY_FIXTURE_DIR",
            default_value = "tests/parity/fixtures"
        )]
        fixtures: PathBuf,
        /// Only run this comparator (by transform name).
        #[arg(long)]
        only: Option<String>,
        /// When set, treat `Skipped` outcomes as failures (non-zero exit).
        /// Default off so stub comparators don't block CI.
        /// Configurable via `HEADROOM_PARITY_STRICT` env var.
        #[arg(long, env = "HEADROOM_PARITY_STRICT", default_value_t = false)]
        strict: bool,
    },
    /// List the transforms the harness knows about.
    List,
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::List => {
            for c in builtin_comparators() {
                println!("{}", c.name());
            }
            Ok(())
        }
        Cmd::Run {
            fixtures,
            only,
            strict,
        } => {
            tracing::info!(
                fixtures = %fixtures.display(),
                strict,
                only = ?only,
                "parity-run starting"
            );

            let mut summary = RunSummary::default();
            for comparator in builtin_comparators() {
                if let Some(ref filt) = only {
                    if filt != comparator.name() {
                        continue;
                    }
                }
                let started = Instant::now();
                let report = run_comparator(&fixtures, comparator.as_ref())?;
                let duration_ms = started.elapsed().as_millis() as u64;

                let outcome = report_outcome(&report);
                tracing::info!(
                    transform = comparator.name(),
                    outcome = outcome,
                    matched = report.matched,
                    skipped = report.skipped.len(),
                    diffed = report.diffed.len(),
                    total = report.total(),
                    duration_ms,
                    "parity comparator run"
                );

                println!(
                    "[{:<16}] total={} matched={} skipped={} diffed={}",
                    comparator.name(),
                    report.total(),
                    report.matched,
                    report.skipped.len(),
                    report.diffed.len()
                );
                for (path, reason) in &report.skipped {
                    println!("  skipped {}: {}", path.display(), reason);
                }
                for (path, expected, actual) in &report.diffed {
                    tracing::error!(
                        transform = comparator.name(),
                        fixture = %path.display(),
                        "parity diff detected"
                    );
                    println!("  DIFF {}", path.display());
                    println!("    expected: {}", first_line(expected));
                    println!("    actual  : {}", first_line(actual));
                }

                summary.matched += report.matched;
                summary.diffed += report.diffed.len();
                summary.skipped += report.skipped.len();
            }

            let exit_code = summary.exit_code(strict);
            tracing::info!(
                matched = summary.matched,
                diffed = summary.diffed,
                skipped = summary.skipped,
                strict,
                exit_code,
                "parity-run finished"
            );
            std::process::exit(exit_code);
        }
    }
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn report_outcome(report: &headroom_parity::Report) -> &'static str {
    if !report.diffed.is_empty() {
        "diff"
    } else if !report.skipped.is_empty() && report.matched == 0 {
        "skipped"
    } else if report.matched == 0 {
        "empty"
    } else if !report.skipped.is_empty() {
        "matched_with_skips"
    } else {
        "matched"
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

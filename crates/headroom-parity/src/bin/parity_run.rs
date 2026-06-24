//! `parity-run` CLI: drive the parity harness from the command line.

use anyhow::Result;
use clap::{Parser, Subcommand};
use headroom_parity::{builtin_comparators, run_comparator};
use std::path::PathBuf;

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
        #[arg(long, default_value = "tests/parity/fixtures")]
        fixtures: PathBuf,
        /// Only run this comparator (by transform name).
        #[arg(long)]
        only: Option<String>,
    },
    /// List the transforms the harness knows about.
    List,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::List => {
            for c in builtin_comparators() {
                println!("{}", c.name());
            }
            Ok(())
        }
        Cmd::Run { fixtures, only } => {
            let mut any_diffs = false;
            for comparator in builtin_comparators() {
                if let Some(ref filt) = only {
                    if filt != comparator.name() {
                        continue;
                    }
                }
                let report = run_comparator(&fixtures, comparator.as_ref())?;
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
                    any_diffs = true;
                    println!("  DIFF {}", path.display());
                    println!("    expected: {}", first_line(expected));
                    println!("    actual  : {}", first_line(actual));
                }
            }
            if any_diffs {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

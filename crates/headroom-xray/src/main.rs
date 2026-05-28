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
        if let Err(e) = print_footer().await {
            if cli.xray_debug {
                eprintln!("[xray-debug] footer suppressed: {e}");
            }
        }
    }

    std::process::exit(code);
}

async fn print_footer() -> Result<()> {
    use headroom_xray::footer;
    use headroom_xray::tokenize::count_by_tool;
    use headroom_xray::transcripts::claude_code;

    let session = match claude_code::latest_session_for_cwd() {
        Some(p) => p,
        None => return Ok(()), // no session here — silently skip
    };
    let transcript = claude_code::parse(&session)?;
    let counts = count_by_tool(&transcript)?;
    let rendered = footer::render(&counts);
    if !rendered.is_empty() {
        print!("\n{rendered}");
    }
    Ok(())
}

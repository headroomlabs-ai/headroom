//! Diagnostic: load one parity fixture and print expected vs actual.
//!
//! Usage: cargo run -p headroom-parity --example diff_fixture -- <path-to-fixture.json>

use anyhow::{bail, Context, Result};
use headroom_parity::{builtin_comparators, Fixture};
use std::env;
use std::fs;

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .context("usage: diff_fixture <fixture.json>")?;
    let bytes = fs::read(&path).context("reading fixture")?;
    let fixture: Fixture = serde_json::from_slice(&bytes).context("parsing fixture")?;

    let comparator = builtin_comparators()
        .into_iter()
        .find(|c| c.name() == fixture.transform)
        .with_context(|| format!("no comparator named {}", fixture.transform))?;

    let actual = match comparator.run(&fixture.input, &fixture.config) {
        Ok(v) => v,
        Err(e) => bail!("comparator failed: {e}"),
    };

    let expected_pretty = serde_json::to_string_pretty(&fixture.output)?;
    let actual_pretty = serde_json::to_string_pretty(&actual)?;

    println!("=== Expected (Python) ===");
    println!("{expected_pretty}");
    println!("\n=== Actual (Rust) ===");
    println!("{actual_pretty}");

    if actual == fixture.output {
        println!("\n=== MATCH ===");
    } else {
        println!("\n=== DIFFER ===");
        // Field-by-field for objects
        if let (Some(exp_obj), Some(act_obj)) = (fixture.output.as_object(), actual.as_object()) {
            for key in exp_obj
                .keys()
                .chain(act_obj.keys())
                .collect::<std::collections::BTreeSet<_>>()
            {
                let e = exp_obj.get(key);
                let a = act_obj.get(key);
                if e != a {
                    println!("  field {key}:");
                    println!(
                        "    expected: {}",
                        e.map(|v| serde_json::to_string(v).unwrap())
                            .unwrap_or_default()
                    );
                    println!(
                        "    actual  : {}",
                        a.map(|v| serde_json::to_string(v).unwrap())
                            .unwrap_or_default()
                    );
                }
            }
        }
    }

    Ok(())
}

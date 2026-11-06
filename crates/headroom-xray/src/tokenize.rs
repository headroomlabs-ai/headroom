//! Token counting using tiktoken-rs (cl100k_base).
//!
//! Phase 1 uses a single tokenizer for all blocks (cl100k_base). This is
//! pessimistic for Anthropic models (which use a slightly different
//! tokenizer) but stable across runs and good enough for ranking. Phase 2
//! can pick a model-aware tokenizer when we know the session's model.

use std::collections::HashMap;
use tiktoken_rs::cl100k_base;

use crate::transcripts::Transcript;

/// Aggregate token count per tool name.
pub fn count_by_tool(transcript: &Transcript) -> Result<HashMap<String, usize>, anyhow::Error> {
    let bpe = cl100k_base()?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for block in &transcript.blocks {
        let n = bpe.encode_with_special_tokens(&block.text).len();
        *counts.entry(block.tool.clone()).or_insert(0) += n;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Transcript;

    #[test]
    fn counts_aggregate_per_tool() {
        let mut t = Transcript::default();
        t.push("Bash", "hello world\n".repeat(100));
        t.push("Bash", "another bash output\n".repeat(50));
        t.push("Read", "file content here\n".repeat(20));
        let counts = count_by_tool(&t).unwrap();
        assert!(
            counts.get("Bash").copied().unwrap_or(0) > 100,
            "Bash should have substantial tokens"
        );
        assert!(
            counts.get("Read").copied().unwrap_or(0) > 0,
            "Read should be present"
        );
        assert!(
            counts.get("Bash").unwrap() > counts.get("Read").unwrap(),
            "Bash > Read"
        );
    }

    #[test]
    fn empty_transcript_yields_empty_counts() {
        let t = Transcript::default();
        let counts = count_by_tool(&t).unwrap();
        assert!(counts.is_empty());
    }
}

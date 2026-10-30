//! Dump BM25 scores, rankings and matched terms so the query-preparation
//! change can be diffed byte-for-byte against its baseline.
//!
//! Scores print as raw IEEE-754 bits, not decimals: the change alters when
//! query terms are ordered, not the order itself, so accumulation must stay
//! bit-identical rather than merely close.
//!
//! `cargo run --release --example bm25_query_prep_parity > out.txt` on each
//! side, then `diff`.

use headroom_core::relevance::{BM25Scorer, RelevanceScorer};

/// xorshift64*, so both sides generate the identical corpus without
/// depending on a rand version.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

const VOCAB: [&str; 16] = [
    "handler",
    "parse",
    "timeout",
    "cache",
    "retry",
    "index",
    "error",
    "550e8400-e29b-41d4-a716-446655440000",
    "9f8e7d6c-1234-4321-abcd-0123456789ab",
    "12345",
    "987654",
    "a",
    "zz",
    "Mixed_Case_Token",
    "UPPER",
    "repeated",
];

fn text(rng: &mut Rng, words: usize) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(words);
    for _ in 0..words {
        parts.push(VOCAB[rng.below(VOCAB.len() as u64) as usize]);
    }
    parts.join(" ")
}

fn dump(name: &str, scorer: &BM25Scorer, items: &[String], context: &str) {
    let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
    let batch = scorer.score_batch(&refs, context);

    println!(
        "===== {name} (n={}, context={:?}) =====",
        items.len(),
        context
    );
    for (i, s) in batch.iter().enumerate() {
        // Raw bits: a reordered summation would change these even when the
        // rounded decimal looks identical.
        println!(
            "  batch[{i}] bits={:016x} reason={:?} matched={:?}",
            s.score.to_bits(),
            s.reason,
            s.matched_terms
        );
    }

    // Ranking is what callers actually consume, so compare it explicitly.
    let mut ranked: Vec<usize> = (0..batch.len()).collect();
    ranked.sort_by(|a, b| {
        batch[*b]
            .score
            .partial_cmp(&batch[*a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    println!("  ranking={ranked:?}");

    // The single-item path shares bm25_score, so verify it agrees with the
    // batch path item for item (reasons and caps differ by design).
    for (i, item) in items.iter().enumerate().take(20) {
        let single = scorer.score(item, context);
        println!(
            "  single[{i}] bits={:016x} reason={:?} matched={:?}",
            single.score.to_bits(),
            single.reason,
            single.matched_terms
        );
    }
    println!();
}

fn main() {
    let scorer = BM25Scorer::default();
    let unnormalized = BM25Scorer::new(1.5, 0.75, false, 10.0);
    let tuned = BM25Scorer::new(0.5, 0.0, true, 3.0);

    // Degenerate shapes first.
    dump("empty-batch", &scorer, &[], "error handler");
    dump("empty-context", &scorer, &["error here".into()], "");
    dump(
        "empty-docs",
        &scorer,
        &[String::new(), String::new()],
        "error",
    );
    dump("single-doc", &scorer, &["error handler".into()], "error");
    dump(
        "no-overlap",
        &scorer,
        &["aaa bbb ccc".into(), "ddd eee".into()],
        "zzz yyy",
    );

    // Repeated query terms: query frequency > 1 multiplies each term score,
    // so a dropped or double-counted key shows up immediately.
    dump(
        "repeated-query-terms",
        &scorer,
        &[
            "error error handler".into(),
            "handler parse".into(),
            "error".into(),
        ],
        "error error error handler",
    );

    // Exact ties: identical documents must keep identical scores and a
    // stable ranking.
    dump(
        "exact-ties",
        &scorer,
        &vec!["error handler parse".to_string(); 6],
        "error parse",
    );

    // Long tokens trip the >=8 char bonus in finalize_score.
    dump(
        "long-token-bonus",
        &scorer,
        &[
            "550e8400-e29b-41d4-a716-446655440000 tail".into(),
            "short a b".into(),
        ],
        "550e8400-e29b-41d4-a716-446655440000 short",
    );

    // Vary document count and query length across configs.
    for (label, s) in [
        ("default", &scorer),
        ("unnormalized", &unnormalized),
        ("tuned", &tuned),
    ] {
        for docs in [1usize, 2, 17, 64, 250] {
            for query_words in [1usize, 4, 32] {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (docs as u64) << 8 ^ query_words as u64);
                let items: Vec<String> = (0..docs)
                    .map(|_| {
                        let words = 1 + rng.below(40) as usize;
                        text(&mut rng, words)
                    })
                    .collect();
                let context = text(&mut rng, query_words);
                dump(
                    &format!("{label}-d{docs}-q{query_words}"),
                    s,
                    &items,
                    &context,
                );
            }
        }
    }
}

//! Criterion benchmark for `BM25Scorer::score_batch`.
//!
//! Query preparation used to happen once per document: every `bm25_score`
//! call collected and sorted the same query keys. Cost therefore scales with
//! documents x query length, so both are varied here.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use headroom_core::relevance::{BM25Scorer, RelevanceScorer};

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
    "resolve",
    "connection",
    "serialize",
    "upstream",
    "budget",
];

fn text(seed: usize, words: usize) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(words);
    for i in 0..words {
        parts.push(VOCAB[(seed * 7 + i * 3) % VOCAB.len()]);
    }
    parts.join(" ")
}

fn bench_score_batch(c: &mut Criterion) {
    let scorer = BM25Scorer::default();
    let mut group = c.benchmark_group("bm25/score_batch");

    for docs in [10usize, 100, 1000] {
        for query_words in [4usize, 16, 64] {
            let items: Vec<String> = (0..docs).map(|d| text(d, 60)).collect();
            let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
            let context = text(9_999, query_words);

            group.throughput(Throughput::Elements(docs as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{docs}d_q{query_words}w")),
                &(refs, context),
                |b, (refs, context)| {
                    b.iter(|| black_box(scorer.score_batch(black_box(refs), black_box(context))))
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_score_batch);
criterion_main!(benches);

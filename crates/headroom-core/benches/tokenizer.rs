//! Throughput benchmark for `headroom_core::tokenizer`.
//!
//! Measures the tiktoken-rs–backed counter on a small / medium / large input.
//! Used as a baseline; future stages can compare against this to catch
//! regressions when we change tokenizer backends or add caching layers.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use headroom_core::tokenizer::{TiktokenCounter, Tokenizer};

fn bench_count_text(c: &mut Criterion) {
    let counter = TiktokenCounter::for_model("gpt-4o-mini").expect("init");

    // Small: a typical short prompt.
    let small = "Reply with exactly: PONG";
    // Medium: a typical chat turn (~1KB).
    let medium = "the quick brown fox jumps over the lazy dog\n".repeat(25);
    // Large: a long context (~64KB) — stresses BPE inner loops.
    let large = "the quick brown fox jumps over the lazy dog\n".repeat(1500);

    let mut group = c.benchmark_group("tokenizer/count_text");
    group.throughput(Throughput::Bytes(small.len() as u64));
    group.bench_function("small", |b| {
        b.iter_batched(
            || small,
            |s| black_box(counter.count_text(s)),
            BatchSize::SmallInput,
        )
    });

    group.throughput(Throughput::Bytes(medium.len() as u64));
    group.bench_function("medium", |b| {
        b.iter_batched(
            || medium.as_str(),
            |s| black_box(counter.count_text(s)),
            BatchSize::SmallInput,
        )
    });

    group.throughput(Throughput::Bytes(large.len() as u64));
    group.bench_function("large", |b| {
        b.iter_batched(
            || large.as_str(),
            |s| black_box(counter.count_text(s)),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_count_text);
criterion_main!(benches);

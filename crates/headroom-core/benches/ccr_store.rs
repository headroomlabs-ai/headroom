//! CCR store throughput benchmark — single-threaded and multi-threaded.
//!
//! Pins the win from PR9: replacing the single-`Mutex<HashMap>` design
//! with a `DashMap`-backed sharded store. The single-threaded numbers
//! should be roughly comparable (DashMap has a small per-op shard-hash
//! overhead vs a raw Mutex), but the multi-threaded numbers should
//! diverge sharply — distinct keys hit distinct shards and never
//! contend.
//!
//! Run with:
//!     cargo bench -p headroom-core --bench ccr_store
//!
//! The critical numbers to watch are the `mt/N=8` rows: with the
//! Mutex design, all 8 threads serialize on one lock, so throughput
//! is ~1× the single-threaded figure. With DashMap, throughput should
//! scale near-linearly with cores.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use headroom_core::ccr::{CcrStore, InMemoryCcrStore};

// ─── Baseline: the old single-Mutex<HashMap> design ────────────────
//
// Inlined here so the bench is self-contained and shows the
// before/after gap directly. Same trait, same semantics; the only
// difference is "all ops serialize on one Mutex" vs "DashMap-sharded".

struct LegacyMutexStore {
    inner: Mutex<LegacyInner>,
    ttl: Duration,
    capacity: usize,
}

struct LegacyInner {
    map: HashMap<String, LegacyEntry>,
    order: VecDeque<String>,
}

struct LegacyEntry {
    payload: String,
    inserted: Instant,
}

impl LegacyMutexStore {
    fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(LegacyInner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
            capacity,
        }
    }
}

impl CcrStore for LegacyMutexStore {
    fn put(&self, hash: &str, payload: &str) {
        let mut g = self.inner.lock().unwrap();
        if g.map.contains_key(hash) {
            g.map.insert(
                hash.to_string(),
                LegacyEntry {
                    payload: payload.to_string(),
                    inserted: Instant::now(),
                },
            );
            return;
        }
        while g.map.len() >= self.capacity {
            let Some(oldest) = g.order.pop_front() else {
                break;
            };
            g.map.remove(&oldest);
        }
        g.map.insert(
            hash.to_string(),
            LegacyEntry {
                payload: payload.to_string(),
                inserted: Instant::now(),
            },
        );
        g.order.push_back(hash.to_string());
    }

    fn get(&self, hash: &str) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        let expired = match g.map.get(hash) {
            Some(e) => e.inserted.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            g.map.remove(hash);
            return None;
        }
        g.map.get(hash).map(|e| e.payload.clone())
    }

    fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }
}

fn bench_put_single_threaded(c: &mut Criterion) {
    let store = InMemoryCcrStore::new();
    let payload = "x".repeat(512); // typical CCR payload size

    let mut group = c.benchmark_group("ccr_store/put_st");
    group.throughput(Throughput::Elements(1));
    group.bench_function("new_keys", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("k{i:012x}");
            store.put(black_box(&key), black_box(&payload));
            i += 1;
        });
    });
    group.bench_function("same_key_overwrite", |b| {
        b.iter(|| {
            store.put(black_box("hot_key"), black_box(&payload));
        });
    });
    group.finish();
}

fn bench_get_single_threaded(c: &mut Criterion) {
    let store = InMemoryCcrStore::new();
    let payload = "y".repeat(512);
    for i in 0..1000u32 {
        let key = format!("k{i:08x}");
        store.put(&key, &payload);
    }

    let mut group = c.benchmark_group("ccr_store/get_st");
    group.throughput(Throughput::Elements(1));
    group.bench_function("hit", |b| {
        let mut i = 0u32;
        b.iter(|| {
            let key = format!("k{:08x}", i % 1000);
            let _ = black_box(store.get(black_box(&key)));
            i = i.wrapping_add(1);
        });
    });
    group.bench_function("miss", |b| {
        let mut i = 0u32;
        b.iter(|| {
            let key = format!("absent_{i}");
            let _ = black_box(store.get(black_box(&key)));
            i = i.wrapping_add(1);
        });
    });
    group.finish();
}

fn run_mt_workload(store: Arc<dyn CcrStore>, threads: usize, n: u64) -> Duration {
    const ITERS_PER_THREAD: usize = 200;
    let payload = Arc::new("z".repeat(256));
    for i in 0..256u32 {
        store.put(&format!("warm_{i:08x}"), &payload);
    }
    let start = Instant::now();
    for _ in 0..n {
        thread::scope(|scope| {
            for tid in 0..threads {
                let s = store.clone();
                let p = payload.clone();
                scope.spawn(move || {
                    for i in 0..ITERS_PER_THREAD {
                        if i & 1 == 0 {
                            let k = format!("t{tid}_k{i:08x}");
                            s.put(&k, &p);
                        } else {
                            let k = format!("warm_{:08x}", i % 256);
                            let _ = s.get(&k);
                        }
                    }
                });
            }
        });
    }
    start.elapsed()
}

/// Multi-threaded mixed put/get — direct A/B between the legacy
/// `Mutex<HashMap>` design and the new DashMap-backed store. The
/// legacy version serializes every op on one lock; the new version
/// shards across keys so distinct hashes never contend.
fn bench_mixed_multi_threaded(c: &mut Criterion) {
    let mut group = c.benchmark_group("ccr_store/mt_mixed");
    group.throughput(Throughput::Elements(1));

    for &threads in &[1usize, 2, 4, 8] {
        // DashMap-backed (current design).
        let label = format!("dashmap/threads={threads}");
        group.bench_function(&label, |b| {
            b.iter_custom(|n| {
                let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
                run_mt_workload(store, threads, n)
            });
        });
        // Legacy Mutex<HashMap> (the design PR9 replaces).
        let label = format!("legacy_mutex/threads={threads}");
        group.bench_function(&label, |b| {
            b.iter_custom(|n| {
                let store: Arc<dyn CcrStore> =
                    Arc::new(LegacyMutexStore::new(1000, Duration::from_secs(300)));
                run_mt_workload(store, threads, n)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_put_single_threaded,
    bench_get_single_threaded,
    bench_mixed_multi_threaded,
);
criterion_main!(benches);

//! In-memory CCR backend.
//!
//! Process-local store backed by [`DashMap`] (sharded concurrent hash
//! map). Distinct keys never contend on the read path; capacity-bound
//! eviction is the only globally-serialized step.
//!
//! This is the **test-default** backend. Production deployments use
//! [`super::sqlite::SqliteCcrStore`] or [`super::redis::RedisCcrStore`]
//! which are persistent across worker restarts and shareable across
//! workers (see `RUST_DEV.md` "Multi-worker deployment").

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::ccr::{max_lifetime_for, CcrStore, DEFAULT_CAPACITY, DEFAULT_TTL};

/// In-memory CCR store backed by [`DashMap`] for sharded concurrent
/// access.
///
/// - **TTL**: 30 minutes by default, treated as an **idle window** —
///   every successful `get` restarts the entry's clock (#2604), bounded
///   by an absolute max lifetime of 8x the idle TTL measured from
///   insertion. Entries past their window are dropped on the next `get`
///   (lazy expiry — no background reaper thread).
/// - **Capacity**: 1000 entries by default. When `put` would push us
///   past capacity, the oldest entry (per insertion order) is evicted.
/// - **Concurrency**: gets and puts on distinct keys do not contend.
///   The only serialization point is the insertion-order queue used
///   for capacity eviction; that mutex is held for an O(1) push or a
///   small sweep.
pub struct InMemoryCcrStore {
    map: DashMap<String, Entry>,
    /// FIFO insertion order as `(hash, seq)`. A queue item is live only
    /// while `map[hash].seq == seq`: an entry removed by TTL expiry and
    /// stored again later gets a new `seq`, so its old queue item can
    /// neither evict the fresh entry nor pin it. Stale items are skipped
    /// by eviction and dropped by `compact_order` once the queue grows
    /// past twice the capacity.
    order: Mutex<VecDeque<(String, u64)>>,
    next_seq: AtomicU64,
    /// Test-only pause point between inserting a fresh entry and
    /// recording it in `order`, to reproduce writer interleavings.
    #[cfg(test)]
    after_insert_hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    ttl: Duration,
    max_lifetime: Duration,
    capacity: usize,
}

#[derive(Clone)]
struct Entry {
    payload: String,
    seq: u64,
    inserted: Instant,
    last_accessed: Instant,
}

impl Entry {
    /// Expired when idle past `ttl` OR older (since insertion) than
    /// `max_lifetime` — the absolute ceiling that keeps constant access
    /// from pinning an entry forever.
    fn is_expired(&self, ttl: Duration, max_lifetime: Duration) -> bool {
        self.last_accessed.elapsed() > ttl || self.inserted.elapsed() > max_lifetime
    }
}

impl InMemoryCcrStore {
    /// Default: 1000 entries, 30-minute idle TTL (8x max lifetime).
    pub fn new() -> Self {
        Self::with_capacity_and_ttl(DEFAULT_CAPACITY, DEFAULT_TTL)
    }

    /// `ttl` is the idle window; the absolute max lifetime defaults to
    /// 8x that (see [`crate::ccr::DEFAULT_MAX_LIFETIME_MULTIPLIER`]).
    pub fn with_capacity_and_ttl(capacity: usize, ttl: Duration) -> Self {
        Self::with_capacity_and_ttls(capacity, ttl, max_lifetime_for(ttl))
    }

    /// Full-control constructor: idle window and absolute max lifetime
    /// specified independently.
    pub fn with_capacity_and_ttls(capacity: usize, ttl: Duration, max_lifetime: Duration) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            order: Mutex::new(VecDeque::with_capacity(capacity)),
            next_seq: AtomicU64::new(0),
            #[cfg(test)]
            after_insert_hook: Mutex::new(None),
            ttl,
            max_lifetime,
            capacity,
        }
    }

    /// Evict real entries, oldest first, until `map.len() < capacity`.
    /// The caller holds the `order` lock, which serializes every
    /// fresh-key insert, so no entry can be in the map without its
    /// queue item while this runs.
    fn evict_until_under_capacity(&self, guard: &mut VecDeque<(String, u64)>) {
        while self.map.len() >= self.capacity {
            let Some((oldest, seq)) = guard.pop_front() else {
                break;
            };
            // Remove only the entry this queue item was recorded for. A
            // stale item (the entry expired, or was stored again since)
            // removes nothing, and the loop moves on to the next one.
            self.map.remove_if(&oldest, |_, entry| entry.seq == seq);
        }
    }

    /// Drop queue items whose entry is gone or was re-stored, so the
    /// queue stays within a constant factor of the live entries even
    /// when entries keep expiring through `get` without any eviction.
    fn compact_order(&self, order: &mut VecDeque<(String, u64)>) {
        if order.len() > self.capacity.max(16) * 2 {
            order.retain(|(hash, seq)| self.map.get(hash).is_some_and(|e| e.seq == *seq));
        }
    }
}

impl Default for InMemoryCcrStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CcrStore for InMemoryCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        // Idempotent re-store fast-path: same hash → overwrite payload
        // in place, leave the order queue alone. Common when the same
        // tool output flows through multiple times in a session.
        if let Some(mut existing) = self.map.get_mut(hash) {
            let now = Instant::now();
            existing.payload = payload.to_string();
            existing.inserted = now;
            existing.last_accessed = now;
            return;
        }

        // New entry. The capacity check, the insert and the queue append
        // happen under one `order` lock, so a concurrent fresh insert can
        // never sit in the map without its queue item (which would let
        // eviction run out of queue while the map is still full). Lock
        // order is always `order` then a map shard; no path takes `order`
        // while holding a shard guard.
        let mut order = self.order.lock().expect("ccr order mutex poisoned");
        if let Some(mut existing) = self.map.get_mut(hash) {
            // Another writer stored this hash while we waited for the lock.
            let now = Instant::now();
            existing.payload = payload.to_string();
            existing.inserted = now;
            existing.last_accessed = now;
            return;
        }
        if self.map.len() >= self.capacity {
            self.evict_until_under_capacity(&mut order);
        }
        let now = Instant::now();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let entry = Entry {
            payload: payload.to_string(),
            seq,
            inserted: now,
            last_accessed: now,
        };
        self.map.insert(hash.to_string(), entry);
        #[cfg(test)]
        if let Some(hook) = self.after_insert_hook.lock().expect("hook mutex").take() {
            hook();
        }
        order.push_back((hash.to_string(), seq));
        self.compact_order(&mut order);
    }

    fn get(&self, hash: &str) -> Option<String> {
        // Hit path: shard write-lock (get_mut), check the idle window +
        // max-lifetime ceiling, refresh `last_accessed`, clone payload
        // out. The TTL is a sliding idle window (#2604): every hit
        // restarts the clock, so an entry a session keeps touching does
        // not expire mid-burst. Distinct hashes hash to distinct shards
        // and never contend.
        //
        // Lazy expiry uses DashMap's `remove_if` so the check-and-remove
        // is atomic on the shard. An earlier 2-step (drop read lock,
        // then `remove`) had a TOCTOU race: between dropping the read
        // lock and calling `remove`, a concurrent `put()` of the same
        // hash with a fresh timestamp could land — and our `remove`
        // would then wipe that fresh entry. Under multi-worker proxy
        // load this manifested as "I just stored it; why is it gone?"
        // `remove_if` closes the window because the shard write lock
        // is held across both the predicate evaluation and the removal.
        if let Some(mut entry) = self.map.get_mut(hash) {
            if !entry.is_expired(self.ttl, self.max_lifetime) {
                entry.last_accessed = Instant::now();
                return Some(entry.payload.clone());
            }
        } else {
            return None;
        }
        // Out-of-band path: the entry exists and looks expired. Re-check
        // under the shard write lock; if it's still expired, evict.
        // Otherwise (a concurrent `put` refreshed it) leave it alone
        // and re-fetch its payload.
        let was_removed = self
            .map
            .remove_if(hash, |_, entry| {
                entry.is_expired(self.ttl, self.max_lifetime)
            })
            .is_some();
        if was_removed {
            None
        } else {
            // Concurrent refresh — return the fresh payload.
            self.map.get(hash).map(|e| e.payload.clone())
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_returns_payload() {
        let store = InMemoryCcrStore::new();
        store.put("abc123", r#"[{"id":1}]"#);
        assert_eq!(store.get("abc123"), Some(r#"[{"id":1}]"#.to_string()));
    }

    #[test]
    fn missing_hash_returns_none() {
        let store = InMemoryCcrStore::new();
        assert_eq!(store.get("never_stored"), None);
    }

    #[test]
    fn put_overwrites_under_same_hash() {
        let store = InMemoryCcrStore::new();
        store.put("h", "first");
        store.put("h", "second");
        assert_eq!(store.get("h"), Some("second".to_string()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(2, DEFAULT_TTL);
        store.put("a", "1");
        store.put("b", "2");
        store.put("c", "3");
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a"), None);
        assert_eq!(store.get("b"), Some("2".to_string()));
        assert_eq!(store.get("c"), Some("3".to_string()));
    }

    #[test]
    fn expired_entries_are_dropped_on_get() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(10));
        store.put("a", "1");
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(store.get("a"), None);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn reput_after_expiry_keeps_the_newest_entry() {
        // a expires and is dropped by get(); b is stored; a is stored again
        // (now the newest entry). At capacity, the oldest live entry is b,
        // so storing c must evict b and keep the fresh a.
        let store = InMemoryCcrStore::with_capacity_and_ttl(2, Duration::from_millis(30));
        store.put("a", "old");
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(store.get("a"), None);
        store.put("b", "2");
        store.put("a", "fresh");
        store.put("c", "3");
        let a = store.get("a");
        assert_eq!(store.get("b"), None, "b was the oldest live entry");
        assert_eq!(
            a.as_deref(),
            Some("fresh"),
            "fresh a was evicted instead of b"
        );
    }

    #[test]
    fn order_queue_stays_bounded_when_entries_expire_through_get() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(4, Duration::from_millis(1));
        for i in 0..2000 {
            let k = format!("k{i}");
            store.put(&k, "v");
            std::thread::sleep(Duration::from_millis(2));
            assert_eq!(store.get(&k), None);
        }
        let n = store.order.lock().unwrap().len();
        assert!(
            n <= 32,
            "order queue grew to {n} while the map held {}",
            store.len()
        );
    }

    #[test]
    fn concurrent_fresh_inserts_never_exceed_capacity() {
        use std::sync::{mpsc, Arc};
        // Capacity 1. Writer A stores x and pauses after the map insert,
        // before x is recorded in the order queue. Writer C then stores y.
        // If C could run in that gap it would find no queue item to evict
        // and leave both x and y in a capacity-1 store.
        let store = Arc::new(InMemoryCcrStore::with_capacity_and_ttl(1, DEFAULT_TTL));
        let (paused_tx, paused_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        *store.after_insert_hook.lock().unwrap() = Some(Box::new(move || {
            paused_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        }));
        let a = {
            let s = Arc::clone(&store);
            std::thread::spawn(move || s.put("x", "1"))
        };
        paused_rx.recv().unwrap();
        let c = {
            let s = Arc::clone(&store);
            std::thread::spawn(move || s.put("y", "2"))
        };
        // Give C time to run into the gap if it can.
        std::thread::sleep(Duration::from_millis(200));
        release_tx.send(()).unwrap();
        a.join().unwrap();
        c.join().unwrap();
        assert!(
            store.len() <= 1,
            "capacity 1 store holds {} entries",
            store.len()
        );
        assert_eq!(store.get("y").as_deref(), Some("2"));
    }

    #[test]
    fn store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryCcrStore>();
    }

    #[test]
    fn trait_object_is_usable() {
        let store: Box<dyn CcrStore> = Box::new(InMemoryCcrStore::new());
        store.put("h", "v");
        assert_eq!(store.get("h"), Some("v".to_string()));
        assert!(!store.is_empty());
    }

    #[test]
    fn concurrent_puts_and_gets_do_not_corrupt() {
        // Smoke test for the concurrent design — N threads each do
        // P puts and P gets against distinct keys. Every key written
        // must be readable afterwards.
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemoryCcrStore::with_capacity_and_ttl(10_000, DEFAULT_TTL));
        let n_threads = 8;
        let per_thread = 200;

        let mut handles = Vec::new();
        for tid in 0..n_threads {
            let s = store.clone();
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let key = format!("t{tid}_k{i}");
                    let val = format!("v{tid}_{i}");
                    s.put(&key, &val);
                }
                for i in 0..per_thread {
                    let key = format!("t{tid}_k{i}");
                    let got = s.get(&key);
                    assert_eq!(got, Some(format!("v{tid}_{i}")));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(store.len(), n_threads * per_thread);
    }

    #[test]
    fn expired_get_does_not_wipe_concurrent_refresh() {
        // Regression for the TOCTOU race fixed in the audit-cleanup PR.
        // Two threads contend on the SAME key:
        //   - Thread A: stores fresh value, then `get` it many times.
        //   - Thread B: keeps re-storing the same key with FRESH
        //     timestamps in a tight loop (simulating a second worker
        //     touching the same payload).
        // With the old 2-step check-then-remove, A's `get` could see
        // an "expired" entry, drop the read lock, and remove B's
        // freshly-inserted entry between drop and remove. With
        // `remove_if`, the predicate runs under the shard write lock,
        // so the race window is closed.
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemoryCcrStore::with_capacity_and_ttl(
            64,
            Duration::from_millis(20),
        ));
        let key = "shared_key";
        let payload = "fresh";

        // Seed.
        store.put(key, payload);

        let writer = {
            let s = store.clone();
            thread::spawn(move || {
                // 200 fresh re-stores, racing the reader.
                for _ in 0..200 {
                    s.put(key, payload);
                }
            })
        };

        let reader = {
            let s = store.clone();
            thread::spawn(move || {
                let mut hits = 0;
                for _ in 0..200 {
                    if s.get(key).as_deref() == Some(payload) {
                        hits += 1;
                    }
                }
                hits
            })
        };

        writer.join().unwrap();
        let hits = reader.join().unwrap();
        // The entry must be live at the end (writer's last put won).
        assert_eq!(store.get(key).as_deref(), Some(payload));
        // Reader should have observed the live entry the vast majority
        // of the time. Allow some misses on first iterations / TTL
        // transitions but require strong majority.
        assert!(
            hits > 100,
            "reader should mostly observe live entry, hits={hits}"
        );
    }
}

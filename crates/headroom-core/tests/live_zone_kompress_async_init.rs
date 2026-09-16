//! Non-blocking Kompress initialization — dedicated integration test file.
//!
//! The Kompress slot in `live_zone.rs` is process-global and latches on
//! first use, so proving anything about its *virgin* state needs a test
//! process that does nothing else first — the same reason
//! `live_zone_disable_arms.rs` and `live_zone_kompress_absent.rs` are
//! single-test binaries. Do NOT add more tests to this file.
//!
//! Contract under test (#3227 review): activating the PlainText arm
//! because model artifacts happen to be cache-resident must not make the
//! first qualifying request pay — or wait on — the ~261 MB ONNX session
//! build. Dispatches that race a virgin slot return promptly as NoOp;
//! the expensive construction runs exactly once, off the request path.
//!
//! Two vacuity notes:
//!
//! 1. The timing assertion discriminates only on a warm HuggingFace
//!    cache, where the synchronous build costs whole seconds; on a cold
//!    cache the loader returns `None` quickly and even a blocking
//!    implementation passes the bound. The exactly-once and NoOp
//!    assertions hold on any host.
//! 2. The NoOp assertion cannot be raced into `Modified` by a straggler
//!    thread observing a *completed* init: every thread dispatches
//!    barrier-synchronized within microseconds of the CAS that starts
//!    the initializer, and the build it would have to lose to costs
//!    seconds (tokenizer load + 261 MB session commit).

mod common;

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use common::{body_with_tool_result, dispatch, plain_prose, tool_result_action};
use headroom_core::transforms::live_zone;
use headroom_core::transforms::{BlockAction, LiveZoneOutcome};

#[test]
fn first_dispatch_never_blocks_on_model_init() {
    // > 5120 bytes so the PlainText byte threshold is cleared and the
    // dispatcher actually attempts the arm rather than short-circuiting
    // at `BelowByteThreshold`.
    let prose = plain_prose(5_200);
    assert!(
        prose.len() > 5120,
        "fixture must clear the PlainText byte threshold (5120); got {} bytes",
        prose.len()
    );
    let (body, _) = body_with_tool_result(&prose);

    // Phase 1 — virgin slot, concurrent first requests. Every dispatch
    // must return promptly (never waiting on a model build another
    // thread started) and must leave the body untouched.
    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let body = body.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let started = Instant::now();
                let out = dispatch(&body);
                (started.elapsed(), out)
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let (elapsed, out) = handle.join().expect("dispatch thread panicked");
        assert!(
            elapsed < Duration::from_millis(500),
            "thread {i}: dispatch must not wait on model initialization; took {elapsed:?}"
        );
        assert!(
            matches!(out, LiveZoneOutcome::NoChange { .. }),
            "thread {i}: a dispatch racing a virgin model slot must be a NoOp \
             while initialization is in flight, not a rewrite"
        );
    }

    // Phase 2 — however many threads raced the virgin slot, the
    // expensive construction ran exactly once; a blocking warmup (the
    // proxy's startup path) settles the slot and reports readiness.
    let ready = live_zone::warm_live_zone_compressors();
    let expected_runs = usize::from(cfg!(feature = "ml"));
    assert_eq!(
        live_zone::kompress_init_runs(),
        expected_runs,
        "the model construction must run exactly once per process"
    );

    // Phase 3 — with the slot settled, the arm compresses iff the model
    // actually loaded; readiness comes from the warmup itself, so this
    // cannot silently skip on a host where production would compress.
    if ready {
        match dispatch(&body) {
            LiveZoneOutcome::Modified { manifest, .. } => match tool_result_action(&manifest) {
                BlockAction::Compressed { strategy, .. } => assert_eq!(
                    strategy, "kompress",
                    "post-warmup PlainText dispatch must compress via Kompress"
                ),
                other => panic!("expected Compressed via kompress, got {other:?}"),
            },
            LiveZoneOutcome::NoChange { manifest } => panic!(
                "warmup reported the model ready, so a settled dispatch must compress \
                 this above-threshold prose block; manifest: {manifest:?}"
            ),
        }
    } else {
        eprintln!(
            "NOTE: cold model cache (or a no-ml build) — the non-blocking and \
             exactly-once contracts above are covered; the post-warmup compression \
             check could not run on this host."
        );
    }
}

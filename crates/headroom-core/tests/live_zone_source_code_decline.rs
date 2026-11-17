//! A declined source-code block never reaches the Kompress slot — dedicated
//! integration test file.
//!
//! When the code compressor declines a block (exact passthrough), the
//! dispatcher used to hand it to Kompress, a prose model that drops
//! keywords such as `for`, `in` and `if` (#4214). The Kompress slot is
//! process-global and latches on first use, so the cleanest observable of
//! "Kompress was consulted" is that the model construction never starts.
//! That holds on any host, warm cache or not, which is why this does not
//! need the model. It needs a virgin process, so this is a single-test
//! binary like `live_zone_kompress_async_init.rs`. Do NOT add more tests
//! to this file.
//!
//! Vacuity note: without the `ml` feature there is no Kompress slot and
//! `kompress_init_runs()` is always 0, so the init assertion is trivially
//! true there; the byte-identical assertion still holds.

mod common;

use std::time::{Duration, Instant};

use common::{body_with_tool_result, dispatch, go_handlers_source};
use headroom_core::transforms::live_zone;
use headroom_core::transforms::LiveZoneOutcome;

#[test]
fn declined_source_code_never_starts_kompress() {
    let code = go_handlers_source(30);
    assert!(
        code.len() > 2048,
        "fixture must clear the SourceCode byte threshold (2048); got {} bytes",
        code.len()
    );
    let (body, _) = body_with_tool_result(&code);

    assert!(
        matches!(dispatch(&body), LiveZoneOutcome::NoChange { .. }),
        "a block the code compressor declines must be forwarded unchanged"
    );

    // The slot's initializer runs on a background thread started by the
    // dispatch that first consults it. Give a would-be start ample time to
    // show up before asserting it never did.
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline && live_zone::kompress_init_runs() == 0 {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        live_zone::kompress_init_runs(),
        0,
        "Kompress must not be consulted for code the code compressor declined"
    );
}

//! OSS default `Observer` implementation.
//!
//! Ships [`TracingObserver`] which writes each `CrushEvent` to the
//! `tracing` crate at `debug` level. Subscribers that filter `debug`
//! out (typical production config) pay nothing — `tracing` stops
//! evaluation at the level check before constructing the event
//! fields. Subscribers that retain `debug` get a structured per-crush
//! event suitable for log analytics.
//!
//! Enterprise consumers ship richer observers — `AuditObserver` for
//! SOC2/HIPAA decision logs, `MetricsObserver` for Datadog/Atlas
//! gauges, `LoopTrainingObserver` to stream events to Headroom Loop —
//! all on the same trait.

use super::traits::{CrushEvent, Observer};

/// Writes each `CrushEvent` to the `tracing` crate at `debug` level.
/// Zero-cost when the subscriber filters `debug` out.
#[derive(Debug, Default, Clone, Copy)]
pub struct TracingObserver;

impl Observer for TracingObserver {
    fn name(&self) -> &str {
        "tracing"
    }

    fn on_event(&self, event: &CrushEvent) {
        // `tracing::debug!` is a macro; the level check happens before
        // the fields are evaluated, so this is essentially free at
        // higher log levels.
        tracing::debug!(
            target: "headroom::smart_crusher",
            strategy = %event.strategy,
            input_bytes = event.input_bytes,
            output_bytes = event.output_bytes,
            elapsed_ns = event.elapsed_ns,
            was_modified = event.was_modified,
            "smart_crusher.crush emitted",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_observer_does_not_panic_on_event() {
        // We can't easily assert what `tracing` did without a
        // subscriber set up — but the call must not panic, and the
        // event fields must be readable.
        let event = CrushEvent {
            strategy: "passthrough".to_string(),
            input_bytes: 100,
            output_bytes: 100,
            elapsed_ns: 0,
            was_modified: false,
        };
        TracingObserver.on_event(&event);
    }

    #[test]
    fn tracing_observer_name_is_stable() {
        assert_eq!(TracingObserver.name(), "tracing");
    }
}

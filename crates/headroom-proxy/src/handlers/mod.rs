//! Per-endpoint POST handlers wired explicitly into the router.
//!
//! Most paths flow through the catch-all `forward_http` which gates
//! compression on `compression::is_compressible_path` + content-type.
//! The handlers in this module exist for endpoints whose request
//! shape needs explicit routing for clarity (PR-C2 onward) — the
//! actual forwarding logic still goes through `forward_http`. The
//! gate in `forward_http` runs the per-provider live-zone dispatcher
//! based on `classify_compressible_path`.

pub mod chat_completions;
pub mod conversations;
pub mod responses;

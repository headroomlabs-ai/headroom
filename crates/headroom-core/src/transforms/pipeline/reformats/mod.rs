//! Reformat transforms — pack denser without dropping any information.
//!
//! Every transform here implements [`super::traits::ReformatTransform`].
//! Output bytes are semantically equivalent to input bytes (the LLM
//! reading them gets the same data, just with fewer characters). No
//! CCR involvement; no marker emission.

pub mod json_minifier;
pub mod log_template;

pub use json_minifier::JsonMinifier;
pub use log_template::LogTemplate;

//! Graphics layer — wgpu device, pipeline, and surface rendering.
//!
//! # Public API
//!
//! - [`resources::init`] — async: creates adapter, device, queue, uniform buffer
//! - [`pipeline::build_pipeline`] — builds shader + render pipeline + bind group
//! - [`renderer::Renderer`] — owns everything, drives per-frame draw

pub mod pipeline;
pub mod renderer;
pub mod resources;

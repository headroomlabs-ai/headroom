//! GPU device + resources — adapter, device, queue, uniform buffer.
//!
//! Returns ownership to the caller — nothing is cloned.

use crate::types::Uniforms;

/// Result of initialising the wgpu device.
pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub uniform_buffer: wgpu::Buffer,
}

/// Create adapter + device + queue + uniform buffer.
/// Requires a surface for adapter selection.
pub async fn init(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Option<GpuContext> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("creative_tui"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .ok()?;

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Some(GpuContext {
        adapter,
        device,
        queue,
        uniform_buffer,
    })
}

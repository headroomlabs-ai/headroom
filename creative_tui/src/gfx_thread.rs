//! Graphics thread — winit event loop + wgpu rendering.

use std::sync::{Arc, Mutex};

use winit::{
    dpi::LogicalSize,
    event::{Event as WinitEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowAttributes,
};

use crate::gfx::renderer::Renderer;
use crate::types::{AppConfig, Uniforms};

/// Entry point for the graphics OS thread.
pub fn run(uniforms: Arc<Mutex<Uniforms>>, config: AppConfig) {
    let event_loop = EventLoop::new().expect("event loop");

    #[allow(deprecated)]
    let window = Arc::new(
        event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title(config.window_title)
                    .with_inner_size(LogicalSize::new(
                        config.window_size.width,
                        config.window_size.height,
                    )),
            )
            .expect("create window"),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let mut renderer: Option<Renderer> = None;
    let start = std::time::Instant::now();

    #[allow(deprecated)]
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            WinitEvent::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),

            WinitEvent::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } if new_size.width > 0 && new_size.height > 0 => {
                if let Some(ref mut r) = renderer {
                    r.resize(new_size.width, new_size.height);
                }
            }

            WinitEvent::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let r = match renderer.as_mut() {
                    Some(r) => r,
                    None => {
                        let surface = instance
                            .create_surface(Arc::clone(&window))
                            .expect("create surface");
                        let r = pollster::block_on(Renderer::new(
                            &instance,
                            surface,
                            config.window_size,
                        ))
                        .expect("gpu init");
                        renderer = Some(r);
                        renderer.as_mut().unwrap()
                    }
                };

                let elapsed = start.elapsed().as_secs_f32();
                let u = {
                    let guard = uniforms.lock().unwrap();
                    let mut u = *guard;
                    u.set_time(elapsed);
                    u
                };
                r.write_uniforms(&u);
                r.draw_frame();
            }

            WinitEvent::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })
    .expect("event loop");
}

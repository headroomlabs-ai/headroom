struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

struct Uniforms {
    time: f32,
    speed: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = u.time * u.speed;
    let r = sin(in.uv.x * 8.0 + t) * cos(in.uv.y * 6.0 - t * 0.7) * 0.5 + 0.5;
    let g = sin(in.uv.y * 7.0 + t * 0.9) * cos(in.uv.x * 5.0 + t * 0.6) * 0.5 + 0.5;
    let b = sin((in.uv.x + in.uv.y) * 4.0 * t * 0.4) * 0.5 + 0.5;
    return vec4(r * u.color_r, g * u.color_g, b * u.color_b, 1.0);
}

// SNS Mesh Shader
// Renders receptor meshes with activation heatmap overlay

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var activation_texture: texture_2d<f32>;
@group(1) @binding(1)
var activation_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = in.position;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample activation texture
    let activation = textureSample(activation_texture, activation_sampler, in.uv);

    // Base tissue color (pinkish)
    let base_color = vec3<f32>(0.9, 0.75, 0.72);

    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, -1.0, 0.8));
    let normal = normalize(in.normal);
    let n_dot_l = max(dot(normal, light_dir), 0.0);

    // Ambient + diffuse
    let ambient = 0.3;
    let diffuse = 0.7 * n_dot_l;
    let lighting = ambient + diffuse;

    // Mix base color with activation overlay
    let lit_base = base_color * lighting;

    // Blend activation on top (using activation alpha)
    let final_color = mix(lit_base, activation.rgb, activation.a * 0.8);

    return vec4<f32>(final_color, 1.0);
}

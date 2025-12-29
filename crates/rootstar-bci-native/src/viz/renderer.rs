//! Native wgpu Renderer for SNS Visualization
//!
//! Provides GPU-accelerated rendering of sensory receptor meshes
//! with activation heatmap overlays using native window surfaces.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use super::mesh::{MeshData, MeshId, ReceptorPosition, Vertex};

/// Colormap for activation visualization
#[derive(Clone, Debug, PartialEq)]
pub enum Colormap {
    /// Perceptually uniform, good for scientific data
    Viridis,
    /// Purple to yellow, perceptually uniform
    Plasma,
    /// Black to yellow through red
    Inferno,
    /// Rainbow colormap
    Turbo,
    /// Diverging blue-white-red for +/- values
    CoolWarm,
}

impl Colormap {
    /// Sample the colormap at parameter t (0.0 to 1.0)
    #[must_use]
    pub fn sample(&self, t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => Self::sample_viridis(t),
            Colormap::Plasma => Self::sample_plasma(t),
            Colormap::Inferno => Self::sample_inferno(t),
            Colormap::Turbo => Self::sample_turbo(t),
            Colormap::CoolWarm => Self::sample_coolwarm(t),
        }
    }

    fn sample_viridis(t: f32) -> [u8; 4] {
        let r = (68.0 + t * (49.0 - 68.0 + t * (253.0 - 49.0))).clamp(0.0, 255.0) as u8;
        let g = (1.0 + t * (104.0 - 1.0 + t * (231.0 - 104.0))).clamp(0.0, 255.0) as u8;
        let b = (84.0 + t * (142.0 - 84.0 + t * (37.0 - 142.0))).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_plasma(t: f32) -> [u8; 4] {
        let r = (13.0 + t * (240.0 - 13.0)).clamp(0.0, 255.0) as u8;
        let g = (8.0 + t * t * 240.0).clamp(0.0, 255.0) as u8;
        let b = (135.0 + t * (50.0 - 135.0)).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_inferno(t: f32) -> [u8; 4] {
        let r = (t * 255.0).clamp(0.0, 255.0) as u8;
        let g = (t * t * 200.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - t) * 128.0 * (1.0 - t * t)).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_turbo(t: f32) -> [u8; 4] {
        use std::f32::consts::PI;
        let r = (34.0 + 120.0 * (PI * (t - 0.3)).sin().max(0.0) * 2.0 + 135.0 * t * t)
            .clamp(0.0, 255.0) as u8;
        let g = (30.0 + 220.0 * (PI * (t - 0.5) * 2.0).sin().max(0.0)).clamp(0.0, 255.0) as u8;
        let b = (130.0 + 125.0 * (PI * (t + 0.2)).sin().max(0.0) - 200.0 * t * t)
            .clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_coolwarm(t: f32) -> [u8; 4] {
        let (r, g, b) = if t < 0.5 {
            let s = t * 2.0;
            (59.0 + s * 196.0, 76.0 + s * 179.0, 192.0 + s * 63.0)
        } else {
            let s = (t - 0.5) * 2.0;
            (255.0, 255.0 - s * 155.0, 255.0 - s * 195.0)
        };
        [r as u8, g as u8, b as u8, 255]
    }
}

impl Default for Colormap {
    fn default() -> Self {
        Colormap::Viridis
    }
}

/// Camera uniform buffer layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position (for lighting)
    pub camera_pos: [f32; 4],
}

/// 3D camera for scene viewing
#[derive(Clone, Debug)]
pub struct Camera3D {
    /// Eye position
    pub position: [f32; 3],
    /// Look-at target
    pub target: [f32; 3],
    /// Up vector
    pub up: [f32; 3],
    /// Field of view (radians)
    pub fov: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
}

impl Camera3D {
    /// Create a new camera with default settings
    #[must_use]
    pub fn new(aspect: f32) -> Self {
        Self {
            position: [0.0, -5.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            aspect,
        }
    }

    /// Compute view matrix
    #[must_use]
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let f = normalize(sub(self.target, self.position));
        let s = normalize(cross(f, self.up));
        let u = cross(s, f);

        [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [
                -dot(s, self.position),
                -dot(u, self.position),
                dot(f, self.position),
                1.0,
            ],
        ]
    }

    /// Compute projection matrix (perspective)
    #[must_use]
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (self.fov / 2.0).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) * nf, -1.0],
            [0.0, 0.0, 2.0 * self.far * self.near * nf, 0.0],
        ]
    }

    /// Orbit camera around target
    pub fn orbit(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        let rel = sub(self.position, self.target);
        let r = length(rel);
        let theta = rel[0].atan2(rel[1]) + delta_azimuth;
        let phi = (rel[2] / r).asin().clamp(-1.4, 1.4) + delta_elevation;

        self.position = [
            self.target[0] + r * phi.cos() * theta.sin(),
            self.target[1] + r * phi.cos() * theta.cos(),
            self.target[2] + r * phi.sin(),
        ];
    }

    /// Zoom camera (dolly)
    pub fn zoom(&mut self, factor: f32) {
        let rel = sub(self.position, self.target);
        let new_rel = scale(rel, factor.clamp(0.1, 10.0));
        self.position = add(self.target, new_rel);
    }

    /// Pan camera
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let f = normalize(sub(self.target, self.position));
        let s = normalize(cross(f, self.up));
        let u = cross(s, f);

        let offset = add(scale(s, delta_x), scale(u, delta_y));
        self.position = add(self.position, offset);
        self.target = add(self.target, offset);
    }

    /// Create camera uniform
    #[must_use]
    pub fn uniform(&self) -> CameraUniform {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = mat4_mul(proj, view);

        CameraUniform {
            view_proj,
            camera_pos: [self.position[0], self.position[1], self.position[2], 1.0],
        }
    }
}

// Vector math helpers
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = length(v);
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// GPU-resident mesh with activation texture
pub struct GpuMesh {
    /// Vertex buffer
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer
    pub index_buffer: wgpu::Buffer,
    /// Number of indices
    pub index_count: u32,
    /// Receptor positions (for activation mapping)
    pub receptor_positions: Vec<ReceptorPosition>,
    /// Bind group (activation texture + sampler)
    pub bind_group: wgpu::BindGroup,
    /// Activation texture
    pub activation_texture: wgpu::Texture,
}

/// Native wgpu renderer for SNS visualization
pub struct SnsRenderer {
    /// GPU device
    device: wgpu::Device,
    /// Command queue
    queue: wgpu::Queue,
    /// Surface for rendering
    surface: wgpu::Surface<'static>,
    /// Surface configuration
    surface_config: wgpu::SurfaceConfiguration,
    /// Mesh rendering pipeline
    mesh_pipeline: wgpu::RenderPipeline,
    /// Camera uniform buffer
    camera_buffer: wgpu::Buffer,
    /// Camera bind group
    camera_bind_group: wgpu::BindGroup,
    /// Heatmap texture sampler
    sampler: wgpu::Sampler,
    /// Bind group layout for meshes
    mesh_bind_group_layout: wgpu::BindGroupLayout,
    /// Background color
    background_color: [f32; 4],
    /// Depth texture
    depth_texture: wgpu::Texture,
    /// Depth texture view
    depth_view: wgpu::TextureView,
    /// GPU meshes
    gpu_meshes: HashMap<MeshId, GpuMesh>,
    /// Colormap for heatmap
    colormap: Colormap,
    /// Smoothing radius for heatmap
    smoothing_radius: f32,
    /// Activation range
    activation_range: (f32, f32),
}

impl SnsRenderer {
    /// Create a new native renderer
    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("Surface creation failed: {e}"))?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No suitable GPU adapter found")?;

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SNS Renderer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Device request failed: {e}"))?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create depth texture
        let (depth_texture, depth_view) =
            Self::create_depth_texture(&device, size.width.max(1), size.height.max(1));

        // Create camera uniform buffer
        let camera_uniform = CameraUniform {
            view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create camera bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create camera bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Create mesh bind group layout
        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mesh Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Activation Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create shader module with embedded WGSL
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SNS Shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SNS Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &mesh_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            mesh_pipeline,
            camera_buffer,
            camera_bind_group,
            sampler,
            mesh_bind_group_layout,
            background_color: [0.1, 0.1, 0.12, 1.0],
            depth_texture,
            depth_view,
            gpu_meshes: HashMap::new(),
            colormap: Colormap::Viridis,
            smoothing_radius: 0.02,
            activation_range: (0.0, 100.0),
        })
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Add a mesh to the renderer
    pub fn add_mesh(&mut self, id: MeshId, mesh_data: &MeshData) {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh_data.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh_data.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let texture_size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        let activation_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Activation Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let activation_view = activation_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &self.mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&activation_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.gpu_meshes.insert(
            id,
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh_data.indices.len() as u32,
                receptor_positions: mesh_data.receptor_uvs.clone(),
                bind_group,
                activation_texture,
            },
        );
    }

    /// Update receptor activations for a mesh
    pub fn update_activations(&self, id: &MeshId, activations: &[f32]) {
        if let Some(mesh) = self.gpu_meshes.get(id) {
            self.update_activation_texture(mesh, activations);
        }
    }

    fn normalize_activation(&self, value: f32) -> f32 {
        ((value - self.activation_range.0) / (self.activation_range.1 - self.activation_range.0))
            .clamp(0.0, 1.0)
    }

    fn update_activation_texture(&self, mesh: &GpuMesh, activations: &[f32]) {
        let size = 256u32;
        let mut pixels = vec![0u8; (size * size * 4) as usize];

        for (i, &activation) in activations.iter().enumerate() {
            if i >= mesh.receptor_positions.len() {
                break;
            }

            let uv = &mesh.receptor_positions[i].uv;
            let normalized = self.normalize_activation(activation);
            let color = self.colormap.sample(normalized);

            let center_x = (uv[0] * size as f32) as i32;
            let center_y = (uv[1] * size as f32) as i32;
            let radius = (self.smoothing_radius * size as f32) as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = center_x + dx;
                    let py = center_y + dy;

                    if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 {
                        let dist = ((dx * dx + dy * dy) as f32).sqrt();
                        let sigma = self.smoothing_radius * size as f32;
                        let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();

                        let idx = ((py as u32 * size + px as u32) * 4) as usize;
                        for c in 0..3 {
                            let existing = pixels[idx + c] as f32;
                            let new_val = color[c] as f32;
                            pixels[idx + c] = (existing * (1.0 - weight) + new_val * weight) as u8;
                        }
                        pixels[idx + 3] = (pixels[idx + 3] as f32 + weight * 255.0).min(255.0) as u8;
                    }
                }
            }
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &mesh.activation_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Update camera
    pub fn update_camera(&self, camera: &Camera3D) {
        let uniform = camera.uniform();
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Render all meshes
    pub fn render(&mut self) -> Result<(), String> {
        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| format!("Surface texture error: {e}"))?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SNS Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.background_color[0] as f64,
                            g: self.background_color[1] as f64,
                            b: self.background_color[2] as f64,
                            a: self.background_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.mesh_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for mesh in self.gpu_meshes.values() {
                render_pass.set_bind_group(1, &mesh.bind_group, &[]);
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);

            let (depth_texture, depth_view) = Self::create_depth_texture(&self.device, width, height);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    /// Set colormap
    pub fn set_colormap(&mut self, colormap: Colormap) {
        self.colormap = colormap;
    }

    /// Set activation range
    pub fn set_activation_range(&mut self, min: f32, max: f32) {
        self.activation_range = (min, max.max(min + 0.001));
    }

    /// Get mesh IDs
    pub fn mesh_ids(&self) -> impl Iterator<Item = &MeshId> {
        self.gpu_meshes.keys()
    }

    /// Check if mesh exists
    pub fn has_mesh(&self, id: &MeshId) -> bool {
        self.gpu_meshes.contains_key(id)
    }
}

/// Embedded WGSL shader for mesh rendering
const MESH_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var activation_tex: texture_2d<f32>;

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
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.world_normal = in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample activation texture
    let activation_color = textureSample(activation_tex, activation_sampler, in.uv);

    // Base mesh color (light gray)
    let base_color = vec3<f32>(0.7, 0.7, 0.7);

    // Simple lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(in.world_normal);
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.6 + 0.4;

    // Blend base with activation
    let final_color = mix(base_color * diffuse, activation_color.rgb, activation_color.a);

    return vec4<f32>(final_color, 1.0);
}
"#;

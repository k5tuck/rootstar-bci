//! wgpu Renderer for SNS Visualization
//!
//! Provides GPU-accelerated rendering of sensory receptor meshes
//! with activation heatmap overlays.

use std::collections::HashMap;
use wgpu::util::DeviceExt;

use super::{Camera3D, CameraUniform, SnsScene, SnsVizConfig};
use super::meshes::{GpuMesh, MeshData, MeshId, Vertex};
use super::heatmap::Colormap;

/// wgpu renderer for SNS visualization
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
    /// Configuration
    config: SnsVizConfig,
    /// Depth texture
    depth_texture: wgpu::Texture,
    /// Depth texture view
    depth_view: wgpu::TextureView,
}

impl SnsRenderer {
    /// Create a new renderer
    pub async fn new(canvas_id: &str, config: SnsVizConfig) -> Result<Self, String> {
        // Get the canvas element
        let window = web_sys::window().ok_or("No window")?;
        let document = window.document().ok_or("No document")?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or(format!("Canvas '{}' not found", canvas_id))?;
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into()
            .map_err(|_| "Element is not a canvas")?;

        // Set canvas size
        canvas.set_width(config.width);
        canvas.set_height(config.height);

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("Surface creation failed: {}", e))?;

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
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Device request failed: {}", e))?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: config.width,
            height: config.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(&device, &config);

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
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create camera bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        // Create mesh bind group layout (for activation texture)
        let mesh_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SNS Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
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
            config,
            depth_texture,
            depth_view,
        })
    }

    fn create_depth_texture(device: &wgpu::Device, config: &SnsVizConfig) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
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

    /// Create a GPU mesh from mesh data
    pub fn create_gpu_mesh(&self, mesh_data: &MeshData) -> Result<GpuMesh, String> {
        // Create vertex buffer
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh_data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&mesh_data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create activation texture (256x256)
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

        // Create bind group
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

        Ok(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: mesh_data.indices.len() as u32,
            receptor_positions: mesh_data.receptor_uvs.clone(),
            bind_group,
            activation_texture,
            activation_view,
        })
    }

    /// Render the scene
    pub fn render(&mut self, scene: &SnsScene) -> Result<(), String> {
        // Update camera uniform
        let camera_uniform = CameraUniform::from_camera(scene.camera());
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Update activation textures
        for mesh_id in scene.mesh_ids() {
            if let (Some(mesh), Some(activations)) = (scene.get_mesh(mesh_id), scene.get_activations(mesh_id)) {
                self.update_activation_texture(mesh, activations, scene.heatmap());
            }
        }

        // Get surface texture
        let output = self.surface.get_current_texture()
            .map_err(|e| format!("Surface texture error: {}", e))?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Begin render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SNS Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.config.background_color[0] as f64,
                            g: self.config.background_color[1] as f64,
                            b: self.config.background_color[2] as f64,
                            a: self.config.background_color[3] as f64,
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

            // Draw each mesh
            for mesh_id in scene.mesh_ids() {
                if let Some(mesh) = scene.get_mesh(mesh_id) {
                    render_pass.set_bind_group(1, &mesh.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update_activation_texture(&self, mesh: &GpuMesh, activations: &[f32], heatmap: &super::ActivationHeatmap) {
        let size = 256u32;
        let mut pixels = vec![0u8; (size * size * 4) as usize];

        // Render each receptor as a Gaussian blob
        for (i, &activation) in activations.iter().enumerate() {
            if i >= mesh.receptor_positions.len() {
                break;
            }

            let uv = &mesh.receptor_positions[i].uv;
            let color = heatmap.colormap().sample(heatmap.normalize(activation));

            let center_x = (uv[0] * size as f32) as i32;
            let center_y = (uv[1] * size as f32) as i32;
            let radius = (heatmap.smoothing_radius() * size as f32) as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = center_x + dx;
                    let py = center_y + dy;

                    if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 {
                        let dist = ((dx * dx + dy * dy) as f32).sqrt();
                        let sigma = heatmap.smoothing_radius() * size as f32;
                        let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();

                        let idx = ((py as u32 * size + px as u32) * 4) as usize;
                        // Alpha blend
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

    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);

        let (depth_texture, depth_view) = Self::create_depth_texture(&self.device, &self.config);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }
}

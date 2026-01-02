//! SNS Mesh Generation
//!
//! Procedural mesh generation for sensory organ visualization:
//! - Skin surface with mechanoreceptor positions
//! - Cochlea with hair cell positions
//! - Tongue with papillae and taste bud positions
//! - Retina with photoreceptor and ganglion cell positions
//! - Olfactory epithelium and bulb with receptor neuron positions

pub mod cochlea;
pub mod olfactory;
pub mod retina;
pub mod skin;
pub mod tongue;

use rootstar_bci_core::sns::types::{BodyRegion, Ear, Finger};

pub use olfactory::{OdorantReceptorClass, OlfactoryView};
pub use retina::{Eye, RetinaView};

/// Vertex format for SNS meshes
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Position (x, y, z)
    pub position: [f32; 3],
    /// Normal vector (x, y, z)
    pub normal: [f32; 3],
    /// Texture/UV coordinates (u, v)
    pub uv: [f32; 2],
}

impl Vertex {
    /// Get the vertex buffer layout for wgpu
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Receptor position on a mesh
#[derive(Clone, Debug)]
pub struct ReceptorPosition {
    /// UV coordinates on the mesh
    pub uv: [f32; 2],
    /// Receptor type identifier
    pub receptor_type: ReceptorType,
    /// Additional data (frequency for auditory, taste quality for gustatory)
    pub data: f32,
}

/// Receptor type for mesh positioning
#[derive(Clone, Debug, PartialEq)]
pub enum ReceptorType {
    // Tactile
    Meissner,
    Merkel,
    Pacinian,
    Ruffini,
    FreeNerve,

    // Auditory
    InnerHairCell,
    OuterHairCell,

    // Gustatory
    FungiformPapilla,
    FoliatePapilla,
    CircumvallatePapilla,

    // Visual - Photoreceptors
    Rod,
    LCone,
    MCone,
    SCone,
    // Visual - Ganglion cells
    GanglionOn,
    GanglionOff,
    // Visual - V1 cortex electrode sites
    V1Electrode,

    // Olfactory - Epithelium
    OlfactoryNeuron,
    // Olfactory - Bulb
    Glomerulus,
    Mitral,
    Tufted,
    Granule,
}

/// Mesh ID for scene management
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeshId {
    /// Skin patch for a body region
    SkinPatch { region: BodyRegion },
    /// Cochlea for an ear
    Cochlea { ear: Ear },
    /// Tongue surface
    Tongue,
    /// Retina (visual)
    Retina { eye: Eye, view: RetinaView },
    /// Olfactory structure
    Olfactory { view: OlfactoryView },
}

impl MeshId {
    /// Convert to u32 for WASM interop (simplified encoding)
    pub fn as_u32(&self) -> u32 {
        match self {
            MeshId::SkinPatch { .. } => 0,
            MeshId::Cochlea { ear } => match ear {
                Ear::Left => 1,
                Ear::Right => 2,
            },
            MeshId::Tongue => 3,
            MeshId::Retina { eye, view } => {
                let eye_offset = match eye {
                    Eye::Left => 0,
                    Eye::Right => 3,
                };
                let view_offset = match view {
                    RetinaView::Flat => 0,
                    RetinaView::Curved => 1,
                    RetinaView::Cortex => 2,
                };
                10 + eye_offset + view_offset
            }
            MeshId::Olfactory { view } => match view {
                OlfactoryView::Epithelium => 20,
                OlfactoryView::Bulb => 21,
                OlfactoryView::Combined => 22,
            },
        }
    }

    /// Create from u32 for WASM interop (simplified decoding)
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => MeshId::SkinPatch {
                region: BodyRegion::Fingertip(Finger::Index),
            },
            1 => MeshId::Cochlea { ear: Ear::Left },
            2 => MeshId::Cochlea { ear: Ear::Right },
            3 => MeshId::Tongue,
            10 => MeshId::Retina { eye: Eye::Left, view: RetinaView::Flat },
            11 => MeshId::Retina { eye: Eye::Left, view: RetinaView::Curved },
            12 => MeshId::Retina { eye: Eye::Left, view: RetinaView::Cortex },
            13 => MeshId::Retina { eye: Eye::Right, view: RetinaView::Flat },
            14 => MeshId::Retina { eye: Eye::Right, view: RetinaView::Curved },
            15 => MeshId::Retina { eye: Eye::Right, view: RetinaView::Cortex },
            20 => MeshId::Olfactory { view: OlfactoryView::Epithelium },
            21 => MeshId::Olfactory { view: OlfactoryView::Bulb },
            22 => MeshId::Olfactory { view: OlfactoryView::Combined },
            _ => MeshId::Tongue,
        }
    }
}

/// Complete mesh data (vertices, indices, receptor positions)
#[derive(Clone, Debug)]
pub struct MeshData {
    /// Vertex data
    pub vertices: Vec<Vertex>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Receptor positions (UV coordinates + type)
    pub receptor_uvs: Vec<ReceptorPosition>,
}

impl MeshData {
    /// Create empty mesh data
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            receptor_uvs: Vec::new(),
        }
    }

    /// Get the number of triangles
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get the number of receptors
    pub fn receptor_count(&self) -> usize {
        self.receptor_uvs.len()
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self::new()
    }
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
    /// Activation texture view
    pub activation_view: wgpu::TextureView,
}

/// Surface curvature types
#[derive(Clone, Debug)]
pub enum SurfaceCurvature {
    /// Flat surface
    Flat,
    /// Cylindrical (like a finger)
    Cylindrical { radius: f32, axis: [f32; 3] },
    /// Spherical (like fingertip)
    Spherical { radius: f32, center: [f32; 3] },
    /// Custom height map
    HeightMap { heights: Vec<f32>, width: usize },
}

impl SurfaceCurvature {
    /// Apply curvature to a UV position
    pub fn apply(&self, u: f32, v: f32, width: f32, height: f32) -> [f32; 3] {
        match self {
            SurfaceCurvature::Flat => {
                [(u - 0.5) * width, (v - 0.5) * height, 0.0]
            }
            SurfaceCurvature::Cylindrical { radius, axis } => {
                let angle = (u - 0.5) * std::f32::consts::PI;
                let y = (v - 0.5) * height;

                // Rotate around the axis
                let x = radius * angle.sin();
                let z = radius * (1.0 - angle.cos());

                // Simple case: axis is Y
                [x * axis[0].abs().max(1.0), y, z * axis[2].abs().max(1.0)]
            }
            SurfaceCurvature::Spherical { radius, center } => {
                let theta = (u - 0.5) * std::f32::consts::PI;
                let phi = (v - 0.5) * std::f32::consts::PI * 0.5;

                [
                    center[0] + radius * phi.cos() * theta.sin(),
                    center[1] + radius * phi.sin(),
                    center[2] + radius * phi.cos() * theta.cos(),
                ]
            }
            SurfaceCurvature::HeightMap { heights, width: map_width } => {
                let x = (u - 0.5) * width;
                let y = (v - 0.5) * height;

                let map_u = (u * (*map_width - 1) as f32) as usize;
                let map_v = (v * (heights.len() / map_width - 1) as f32) as usize;
                let idx = map_v * map_width + map_u;
                let z = heights.get(idx).copied().unwrap_or(0.0);

                [x, y, z]
            }
        }
    }

    /// Compute normal at a UV position
    pub fn normal_at(&self, u: f32, v: f32) -> [f32; 3] {
        match self {
            SurfaceCurvature::Flat => [0.0, 0.0, 1.0],
            SurfaceCurvature::Cylindrical { radius: _, axis: _ } => {
                let angle = (u - 0.5) * std::f32::consts::PI;
                [angle.sin(), 0.0, angle.cos()]
            }
            SurfaceCurvature::Spherical { radius: _, center: _ } => {
                let theta = (u - 0.5) * std::f32::consts::PI;
                let phi = (v - 0.5) * std::f32::consts::PI * 0.5;
                [
                    phi.cos() * theta.sin(),
                    phi.sin(),
                    phi.cos() * theta.cos(),
                ]
            }
            SurfaceCurvature::HeightMap { heights, width: map_width } => {
                // Compute normal from height gradient
                let eps = 0.01;
                let p0 = self.apply(u, v, 1.0, 1.0);
                let pu = self.apply((u + eps).min(1.0), v, 1.0, 1.0);
                let pv = self.apply(u, (v + eps).min(1.0), 1.0, 1.0);

                let du = [pu[0] - p0[0], pu[1] - p0[1], pu[2] - p0[2]];
                let dv = [pv[0] - p0[0], pv[1] - p0[1], pv[2] - p0[2]];

                // Cross product
                let n = [
                    du[1] * dv[2] - du[2] * dv[1],
                    du[2] * dv[0] - du[0] * dv[2],
                    du[0] * dv[1] - du[1] * dv[0],
                ];

                // Normalize
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-6 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
        }
    }
}

/// Generate a simple grid mesh
pub fn generate_grid_mesh(
    width: f32,
    height: f32,
    resolution_x: u32,
    resolution_y: u32,
    curvature: &SurfaceCurvature,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..=resolution_y {
        for x in 0..=resolution_x {
            let u = x as f32 / resolution_x as f32;
            let v = y as f32 / resolution_y as f32;

            let position = curvature.apply(u, v, width, height);
            let normal = curvature.normal_at(u, v);

            vertices.push(Vertex {
                position,
                normal,
                uv: [u, v],
            });
        }
    }

    let row_verts = resolution_x + 1;
    for y in 0..resolution_y {
        for x in 0..resolution_x {
            let tl = y * row_verts + x;
            let tr = tl + 1;
            let bl = tl + row_verts;
            let br = bl + 1;

            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    (vertices, indices)
}

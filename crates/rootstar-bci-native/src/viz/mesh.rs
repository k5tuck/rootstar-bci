//! Native SNS Mesh Generation
//!
//! Provides mesh generation for skin, cochlea, and tongue visualization.
//! This is a native-compatible version without wasm-bindgen dependencies.

use rootstar_bci_core::sns::types::{BodyRegion, Ear, Finger};

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
    /// Meissner corpuscle (rapid-adapting, light touch)
    Meissner,
    /// Merkel disc (slow-adapting, texture/edges)
    Merkel,
    /// Pacinian corpuscle (rapid-adapting, vibration)
    Pacinian,
    /// Ruffini ending (slow-adapting, stretch)
    Ruffini,
    /// Free nerve ending (pain/temperature)
    FreeNerve,
    /// Inner hair cell (auditory)
    InnerHairCell,
    /// Outer hair cell (auditory amplification)
    OuterHairCell,
    /// Fungiform papilla (gustatory)
    FungiformPapilla,
    /// Foliate papilla (gustatory)
    FoliatePapilla,
    /// Circumvallate papilla (gustatory)
    CircumvallatePapilla,
}

/// Mesh ID for scene management
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeshId {
    /// Skin patch for a body region
    SkinPatch {
        /// The body region for the skin patch
        region: BodyRegion,
    },
    /// Cochlea for an ear
    Cochlea {
        /// Which ear
        ear: Ear,
    },
    /// Tongue surface
    Tongue,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            receptor_uvs: Vec::new(),
        }
    }

    /// Get the number of triangles
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get the number of receptors
    #[must_use]
    pub fn receptor_count(&self) -> usize {
        self.receptor_uvs.len()
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self::new()
    }
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
    #[must_use]
    pub fn apply(&self, u: f32, v: f32, width: f32, height: f32) -> [f32; 3] {
        match self {
            SurfaceCurvature::Flat => [(u - 0.5) * width, (v - 0.5) * height, 0.0],
            SurfaceCurvature::Cylindrical { radius, axis } => {
                let angle = (u - 0.5) * std::f32::consts::PI;
                let y = (v - 0.5) * height;
                let x = radius * angle.sin();
                let z = radius * (1.0 - angle.cos());
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
            SurfaceCurvature::HeightMap {
                heights,
                width: map_width,
            } => {
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
    #[must_use]
    pub fn normal_at(&self, u: f32, v: f32) -> [f32; 3] {
        match self {
            SurfaceCurvature::Flat => [0.0, 0.0, 1.0],
            SurfaceCurvature::Cylindrical { .. } => {
                let angle = (u - 0.5) * std::f32::consts::PI;
                [angle.sin(), 0.0, angle.cos()]
            }
            SurfaceCurvature::Spherical { .. } => {
                let theta = (u - 0.5) * std::f32::consts::PI;
                let phi = (v - 0.5) * std::f32::consts::PI * 0.5;
                [phi.cos() * theta.sin(), phi.sin(), phi.cos() * theta.cos()]
            }
            SurfaceCurvature::HeightMap { .. } => {
                let eps = 0.01;
                let p0 = self.apply(u, v, 1.0, 1.0);
                let pu = self.apply((u + eps).min(1.0), v, 1.0, 1.0);
                let pv = self.apply(u, (v + eps).min(1.0), 1.0, 1.0);

                let du = [pu[0] - p0[0], pu[1] - p0[1], pu[2] - p0[2]];
                let dv = [pv[0] - p0[0], pv[1] - p0[1], pv[2] - p0[2]];

                let n = [
                    du[1] * dv[2] - du[2] * dv[1],
                    du[2] * dv[0] - du[0] * dv[2],
                    du[0] * dv[1] - du[1] * dv[0],
                ];

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
#[must_use]
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

/// Receptor density (per cm²) for different body regions
#[derive(Clone, Debug)]
pub struct ReceptorDensity {
    pub meissner: f32,
    pub merkel: f32,
    pub pacinian: f32,
    pub ruffini: f32,
}

impl ReceptorDensity {
    /// Get receptor density for a body region
    #[must_use]
    pub fn for_region(region: BodyRegion) -> Self {
        match region {
            BodyRegion::Fingertip(_) => Self {
                meissner: 150.0,
                merkel: 70.0,
                pacinian: 20.0,
                ruffini: 10.0,
            },
            BodyRegion::Palm(_) => Self {
                meissner: 60.0,
                merkel: 30.0,
                pacinian: 10.0,
                ruffini: 8.0,
            },
            BodyRegion::Forearm(_) => Self {
                meissner: 10.0,
                merkel: 5.0,
                pacinian: 2.0,
                ruffini: 5.0,
            },
            BodyRegion::UpperArm(_) => Self {
                meissner: 5.0,
                merkel: 3.0,
                pacinian: 1.0,
                ruffini: 3.0,
            },
            _ => Self {
                meissner: 20.0,
                merkel: 10.0,
                pacinian: 5.0,
                ruffini: 5.0,
            },
        }
    }

    /// Total receptor density
    #[must_use]
    pub fn total(&self) -> f32 {
        self.meissner + self.merkel + self.pacinian + self.ruffini
    }
}

/// Get dimensions for a body region (width cm, height cm)
fn region_dimensions(region: BodyRegion) -> (f32, f32, SurfaceCurvature) {
    match region {
        BodyRegion::Fingertip(_) => (
            1.5,
            1.5,
            SurfaceCurvature::Spherical {
                radius: 0.6,
                center: [0.0, 0.0, -0.3],
            },
        ),
        BodyRegion::Palm(_) => (8.0, 10.0, SurfaceCurvature::Flat),
        BodyRegion::Forearm(_) => (
            6.0,
            15.0,
            SurfaceCurvature::Cylindrical {
                radius: 3.0,
                axis: [0.0, 1.0, 0.0],
            },
        ),
        BodyRegion::UpperArm(_) => (
            8.0,
            15.0,
            SurfaceCurvature::Cylindrical {
                radius: 4.0,
                axis: [0.0, 1.0, 0.0],
            },
        ),
        _ => (5.0, 5.0, SurfaceCurvature::Flat),
    }
}

/// Generate receptor positions based on density
fn generate_receptor_positions(area_cm2: f32, density: &ReceptorDensity) -> Vec<ReceptorPosition> {
    let mut positions = Vec::new();
    let mut rng_state = 12345u64;

    let mut rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    let n_meissner = (density.meissner * area_cm2) as usize;
    for _ in 0..n_meissner {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Meissner,
            data: 0.0,
        });
    }

    let n_merkel = (density.merkel * area_cm2) as usize;
    for _ in 0..n_merkel {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Merkel,
            data: 0.0,
        });
    }

    let n_pacinian = (density.pacinian * area_cm2) as usize;
    for _ in 0..n_pacinian {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Pacinian,
            data: 0.0,
        });
    }

    let n_ruffini = (density.ruffini * area_cm2) as usize;
    for _ in 0..n_ruffini {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Ruffini,
            data: 0.0,
        });
    }

    positions
}

/// Generate skin surface mesh with receptor positions
#[must_use]
pub fn generate_skin_mesh(region: BodyRegion, resolution: u32) -> MeshData {
    let (width_cm, height_cm, curvature) = region_dimensions(region);
    let density = ReceptorDensity::for_region(region);

    let (vertices, indices) =
        generate_grid_mesh(width_cm, height_cm, resolution, resolution, &curvature);

    let area_cm2 = width_cm * height_cm;
    let receptor_uvs = generate_receptor_positions(area_cm2, &density);

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

/// Cochlea physical parameters
const COCHLEA_LENGTH_MM: f32 = 35.0;
const COCHLEA_TURNS: f32 = 2.5;
const BASE_RADIUS_MM: f32 = 4.0;
const APEX_RADIUS_MM: f32 = 0.5;

/// Generate cochlea mesh
#[must_use]
pub fn generate_cochlea_mesh(ear: Ear, unrolled: bool) -> MeshData {
    let segments = 256u32;
    let radial_segments = 16u32;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut receptor_uvs = Vec::new();

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let _length_mm = t * COCHLEA_LENGTH_MM;
        let freq_hz = 165.4 * (10.0_f32.powf(2.1 * (1.0 - t)) - 0.88);

        let (center_x, center_y, center_z) = if unrolled {
            (t * COCHLEA_LENGTH_MM / 10.0, 0.0, 0.0)
        } else {
            let angle = t * COCHLEA_TURNS * 2.0 * std::f32::consts::PI;
            let radius = BASE_RADIUS_MM - t * (BASE_RADIUS_MM - APEX_RADIUS_MM);
            let x = match ear {
                Ear::Left => radius * angle.cos(),
                Ear::Right => -radius * angle.cos(),
            };
            (x / 10.0, radius * angle.sin() / 10.0, t * 0.5)
        };

        let width = (0.5 + t * 1.0) / 10.0;

        for j in 0..=radial_segments {
            let theta = j as f32 / radial_segments as f32 * 2.0 * std::f32::consts::PI;
            let local_x = width * theta.cos();
            let local_y = width * theta.sin();
            let normal = [theta.cos(), theta.sin(), 0.0];

            vertices.push(Vertex {
                position: [center_x + local_x, center_y + local_y, center_z],
                normal,
                uv: [t, j as f32 / radial_segments as f32],
            });
        }

        receptor_uvs.push(ReceptorPosition {
            uv: [t, 0.0],
            receptor_type: ReceptorType::InnerHairCell,
            data: freq_hz,
        });

        for row in 0..3 {
            receptor_uvs.push(ReceptorPosition {
                uv: [t, 0.2 + row as f32 * 0.15],
                receptor_type: ReceptorType::OuterHairCell,
                data: freq_hz,
            });
        }
    }

    let ring_verts = radial_segments + 1;
    for i in 0..segments {
        for j in 0..radial_segments {
            let current = i * ring_verts + j;
            let next_ring = (i + 1) * ring_verts + j;
            indices.push(current);
            indices.push(next_ring);
            indices.push(current + 1);
            indices.push(current + 1);
            indices.push(next_ring);
            indices.push(next_ring + 1);
        }
    }

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

/// Tongue dimensions
const TONGUE_LENGTH_CM: f32 = 10.0;
const TONGUE_WIDTH_CM: f32 = 5.0;
const FUNGIFORM_COUNT: usize = 200;
const CIRCUMVALLATE_COUNT: usize = 9;
const FOLIATE_COUNT: usize = 20;

/// Generate tongue surface mesh with papillae
#[must_use]
pub fn generate_tongue_mesh() -> MeshData {
    let resolution = 50u32;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut receptor_uvs = Vec::new();

    for y in 0..=resolution {
        for x in 0..=resolution {
            let u = x as f32 / resolution as f32;
            let v = y as f32 / resolution as f32;

            let width_factor = 0.3 + 0.7 * v;
            let actual_x = (u - 0.5) * TONGUE_WIDTH_CM * width_factor;
            let actual_y = v * TONGUE_LENGTH_CM;

            let center_dist = (u - 0.5).abs() * 2.0;
            let z = 0.5 * (1.0 - center_dist * center_dist);

            let bump_freq = 30.0;
            let bump = 0.02 * ((u * bump_freq).sin() * (v * bump_freq).cos());

            let position = [actual_x, actual_y - TONGUE_LENGTH_CM / 2.0, z + bump];

            let normal = [-0.2 * (u - 0.5), -0.1, 1.0];
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let normal = [normal[0] / len, normal[1] / len, normal[2] / len];

            vertices.push(Vertex {
                position,
                normal,
                uv: [u, v],
            });
        }
    }

    let row_verts = resolution + 1;
    for y in 0..resolution {
        for x in 0..resolution {
            let tl = y * row_verts + x;
            let tr = tl + 1;
            let bl = tl + row_verts;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    let mut rng_state = 54321u64;
    let mut rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    for _ in 0..FUNGIFORM_COUNT {
        let u = 0.15 + rand() * 0.7;
        let v = rand() * 0.66;
        receptor_uvs.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::FungiformPapilla,
            data: 0.0,
        });
    }

    for i in 0..CIRCUMVALLATE_COUNT {
        let u = 0.2 + (i as f32 / (CIRCUMVALLATE_COUNT - 1) as f32) * 0.6;
        let v_base = 0.72;
        let v_offset =
            0.08 * (1.0 - (2.0 * (i as f32 / (CIRCUMVALLATE_COUNT - 1) as f32) - 1.0).abs());
        let v = v_base + v_offset;
        receptor_uvs.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::CircumvallatePapilla,
            data: 0.0,
        });
    }

    for side in [0.1f32, 0.9f32] {
        for i in 0..FOLIATE_COUNT / 2 {
            let v = 0.3 + (i as f32 / (FOLIATE_COUNT / 2) as f32) * 0.4;
            receptor_uvs.push(ReceptorPosition {
                uv: [side, v],
                receptor_type: ReceptorType::FoliatePapilla,
                data: 0.0,
            });
        }
    }

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::sns::types::Side;

    #[test]
    fn test_skin_mesh_generation() {
        let mesh = generate_skin_mesh(BodyRegion::Fingertip(Finger::Index), 10);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());
    }

    #[test]
    fn test_cochlea_mesh_generation() {
        let mesh = generate_cochlea_mesh(Ear::Left, false);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());
    }

    #[test]
    fn test_tongue_mesh_generation() {
        let mesh = generate_tongue_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(
            mesh.receptor_uvs.len(),
            FUNGIFORM_COUNT + CIRCUMVALLATE_COUNT + FOLIATE_COUNT
        );
    }
}

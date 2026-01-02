//! Olfactory Epithelium and Bulb Mesh Generation
//!
//! Procedural generation of olfactory structures for smell visualization.
//!
//! # Anatomy Modeled
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Olfactory System                                 │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    Olfactory Epithelium                          │   │
//! │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │   │
//! │  │  │ OR1 │ │ OR2 │ │ OR3 │ │ OR1 │ │ OR4 │ │ OR2 │ │ OR5 │ ...  │   │
//! │  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘      │   │
//! │  └─────│───────│───────│───────│───────│───────│───────│──────────┘   │
//! │        │       │       │       │       │       │       │              │
//! │        ▼       ▼       ▼       ▼       ▼       ▼       ▼              │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                     Olfactory Bulb                               │   │
//! │  │    ┌───┐     ┌───┐     ┌───┐     ┌───┐     ┌───┐               │   │
//! │  │    │G1 │     │G2 │     │G3 │     │G4 │     │G5 │  Glomeruli    │   │
//! │  │    └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘               │   │
//! │  │      │         │         │         │         │                  │   │
//! │  │    [M/T]     [M/T]     [M/T]     [M/T]     [M/T]  Mitral/Tufted │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Mesh Types
//!
//! - **Epithelium**: Flat patch representing olfactory receptor neuron locations
//! - **Bulb**: Spheroid with glomerular layer showing convergence zones
//! - **Combined**: Shows both structures and their connections

use super::{generate_grid_mesh, MeshData, ReceptorPosition, ReceptorType, SurfaceCurvature, Vertex};

/// Olfactory view type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OlfactoryView {
    /// Olfactory epithelium (receptor neurons)
    Epithelium,
    /// Olfactory bulb (glomeruli and output neurons)
    Bulb,
    /// Combined view showing both
    Combined,
}

/// Odorant receptor class (simplified, ~8 major classes)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdorantReceptorClass {
    /// Class 1: Fruity/ester receptors
    Fruity,
    /// Class 2: Floral/terpene receptors
    Floral,
    /// Class 3: Woody/sesquiterpene receptors
    Woody,
    /// Class 4: Minty/menthol receptors
    Minty,
    /// Class 5: Sweet/vanillin receptors
    Sweet,
    /// Class 6: Pungent/irritant receptors
    Pungent,
    /// Class 7: Musky/macrocyclic receptors
    Musky,
    /// Class 8: Sulfurous/thiol receptors
    Sulfurous,
}

impl OdorantReceptorClass {
    /// Get color for visualization
    pub fn color(&self) -> [f32; 3] {
        match self {
            Self::Fruity => [1.0, 0.6, 0.2],    // Orange
            Self::Floral => [1.0, 0.4, 0.8],    // Pink
            Self::Woody => [0.6, 0.4, 0.2],     // Brown
            Self::Minty => [0.4, 1.0, 0.8],     // Cyan
            Self::Sweet => [1.0, 0.9, 0.5],     // Yellow
            Self::Pungent => [1.0, 0.2, 0.2],   // Red
            Self::Musky => [0.5, 0.3, 0.5],     // Purple
            Self::Sulfurous => [0.8, 0.8, 0.2], // Yellow-green
        }
    }

    /// Get from index
    pub fn from_index(idx: usize) -> Self {
        match idx % 8 {
            0 => Self::Fruity,
            1 => Self::Floral,
            2 => Self::Woody,
            3 => Self::Minty,
            4 => Self::Sweet,
            5 => Self::Pungent,
            6 => Self::Musky,
            _ => Self::Sulfurous,
        }
    }
}

/// Generate olfactory mesh
///
/// # Arguments
/// * `view` - Which structure to visualize
/// * `detail` - Level of detail (1-10)
pub fn generate_olfactory_mesh(view: OlfactoryView, detail: u32) -> MeshData {
    let detail = detail.clamp(1, 10);

    match view {
        OlfactoryView::Epithelium => generate_epithelium(detail),
        OlfactoryView::Bulb => generate_bulb(detail),
        OlfactoryView::Combined => generate_combined(detail),
    }
}

/// Generate olfactory epithelium mesh
fn generate_epithelium(detail: u32) -> MeshData {
    // Olfactory epithelium is ~5 cm² per side, irregular patch
    // We model it as a slightly curved surface in the nasal cavity
    let width = 25.0;  // mm
    let height = 20.0; // mm
    let resolution = 20 + detail * 5;

    // Slight concave curvature (inside nasal cavity)
    let curvature = SurfaceCurvature::Spherical {
        radius: 50.0,
        center: [0.0, 0.0, 50.0],
    };

    let (vertices, indices) = generate_grid_mesh(width, height, resolution, resolution, &curvature);

    // Generate olfactory receptor neuron positions
    let receptors = generate_orn_positions(detail);

    MeshData {
        vertices,
        indices,
        receptor_uvs: receptors,
    }
}

/// Generate olfactory bulb mesh
fn generate_bulb(detail: u32) -> MeshData {
    let resolution = 24 + detail * 6;

    // Olfactory bulb is roughly ellipsoidal, ~8mm × 4mm × 3mm
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let radius_x = 4.0; // mm
    let radius_y = 2.0;
    let radius_z = 1.5;

    for lat in 0..=resolution {
        for lon in 0..=resolution {
            let u = lon as f32 / resolution as f32;
            let v = lat as f32 / resolution as f32;

            let theta = u * std::f32::consts::TAU;
            let phi = (v - 0.5) * std::f32::consts::PI;

            let x = radius_x * phi.cos() * theta.cos();
            let y = radius_y * phi.cos() * theta.sin();
            let z = radius_z * phi.sin();

            // Normal for ellipsoid
            let nx = x / (radius_x * radius_x);
            let ny = y / (radius_y * radius_y);
            let nz = z / (radius_z * radius_z);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();

            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx / len, ny / len, nz / len],
                uv: [u, v],
            });
        }
    }

    // Generate indices
    let row_size = resolution + 1;
    for lat in 0..resolution {
        for lon in 0..resolution {
            let tl = lat * row_size + lon;
            let tr = tl + 1;
            let bl = tl + row_size;
            let br = bl + 1;

            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    // Generate glomerulus and mitral/tufted cell positions
    let receptors = generate_bulb_positions(detail);

    MeshData {
        vertices,
        indices,
        receptor_uvs: receptors,
    }
}

/// Generate combined epithelium + bulb mesh
fn generate_combined(detail: u32) -> MeshData {
    let mut combined = MeshData::new();

    // Generate epithelium (offset upward)
    let epithelium = generate_epithelium(detail);
    let epithelium_offset = [0.0, 0.0, 10.0];

    for mut v in epithelium.vertices {
        v.position[0] += epithelium_offset[0];
        v.position[1] += epithelium_offset[1];
        v.position[2] += epithelium_offset[2];
        combined.vertices.push(v);
    }
    combined.indices.extend(epithelium.indices);
    combined.receptor_uvs.extend(epithelium.receptor_uvs);

    // Generate bulb (below)
    let bulb = generate_bulb(detail);
    let bulb_vertex_offset = combined.vertices.len() as u32;
    let bulb_receptor_offset = combined.receptor_uvs.len();

    for v in bulb.vertices {
        combined.vertices.push(v);
    }
    for idx in bulb.indices {
        combined.indices.push(idx + bulb_vertex_offset);
    }

    // Adjust bulb receptor UVs to differentiate from epithelium
    for mut r in bulb.receptor_uvs {
        // Offset UV v-coordinate to place in lower half
        r.uv[1] = r.uv[1] * 0.4 + 0.6;
        combined.receptor_uvs.push(r);
    }

    combined
}

/// Generate olfactory receptor neuron positions
fn generate_orn_positions(detail: u32) -> Vec<ReceptorPosition> {
    let mut receptors = Vec::new();

    // ORNs are randomly distributed but clustered by receptor type
    // We create patches of each receptor class

    let patches_per_class = 2 + detail / 3;
    let orns_per_patch = 8 + detail * 2;

    for class_idx in 0..8 {
        let class = OdorantReceptorClass::from_index(class_idx);

        for patch in 0..patches_per_class {
            // Random patch center (deterministic based on indices)
            let seed = class_idx * 100 + patch as usize;
            let center_u = 0.1 + 0.8 * pseudo_random(seed);
            let center_v = 0.1 + 0.8 * pseudo_random(seed + 1);

            for i in 0..orns_per_patch {
                // Scatter around center
                let angle = pseudo_random(seed + i as usize * 2 + 10) * std::f32::consts::TAU;
                let radius = pseudo_random(seed + i as usize * 2 + 11) * 0.08;

                let u = (center_u + radius * angle.cos()).clamp(0.01, 0.99);
                let v = (center_v + radius * angle.sin()).clamp(0.01, 0.99);

                receptors.push(ReceptorPosition {
                    uv: [u, v],
                    receptor_type: ReceptorType::OlfactoryNeuron,
                    data: class_idx as f32, // Store class as data
                });
            }
        }
    }

    receptors
}

/// Generate olfactory bulb cell positions (glomeruli, mitral, tufted)
fn generate_bulb_positions(detail: u32) -> Vec<ReceptorPosition> {
    let mut receptors = Vec::new();

    // Glomeruli are arranged on the surface of the bulb
    // Each glomerulus corresponds to one receptor type
    // ~2000 glomeruli in humans, we show a representative sample

    let glom_per_class = 3 + detail;

    for class_idx in 0..8 {
        for g in 0..glom_per_class {
            let seed = class_idx * 50 + g as usize;

            // Position on bulb surface (UV coordinates)
            let u = (class_idx as f32 + 0.5) / 8.0 + pseudo_random(seed) * 0.08 - 0.04;
            let v = 0.3 + pseudo_random(seed + 1) * 0.4;

            // Glomerulus
            receptors.push(ReceptorPosition {
                uv: [u.clamp(0.05, 0.95), v],
                receptor_type: ReceptorType::Glomerulus,
                data: class_idx as f32,
            });

            // Mitral cell (below glomerulus)
            receptors.push(ReceptorPosition {
                uv: [u.clamp(0.05, 0.95), v + 0.05],
                receptor_type: ReceptorType::Mitral,
                data: class_idx as f32,
            });

            // Tufted cell (slightly offset)
            if g % 2 == 0 {
                receptors.push(ReceptorPosition {
                    uv: [(u + 0.02).clamp(0.05, 0.95), v + 0.03],
                    receptor_type: ReceptorType::Tufted,
                    data: class_idx as f32,
                });
            }
        }
    }

    // Granule cells (inhibitory, between glomeruli)
    let granule_count = 10 + detail * 3;
    for i in 0..granule_count {
        let u = 0.1 + 0.8 * pseudo_random(i as usize + 500);
        let v = 0.5 + 0.3 * pseudo_random(i as usize + 501);

        receptors.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::Granule,
            data: -1.0, // No specific class
        });
    }

    receptors
}

/// Simple deterministic pseudo-random for reproducible layouts
fn pseudo_random(seed: usize) -> f32 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 16) & 0x7fff) as f32 / 32767.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epithelium_generation() {
        let mesh = generate_olfactory_mesh(OlfactoryView::Epithelium, 3);

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // Should have ORN receptors
        let has_orns = mesh.receptor_uvs.iter()
            .any(|r| r.receptor_type == ReceptorType::OlfactoryNeuron);
        assert!(has_orns);
    }

    #[test]
    fn test_bulb_generation() {
        let mesh = generate_olfactory_mesh(OlfactoryView::Bulb, 3);

        assert!(!mesh.vertices.is_empty());

        // Should have glomeruli, mitral, tufted, and granule cells
        let has_glom = mesh.receptor_uvs.iter()
            .any(|r| r.receptor_type == ReceptorType::Glomerulus);
        let has_mitral = mesh.receptor_uvs.iter()
            .any(|r| r.receptor_type == ReceptorType::Mitral);
        let has_granule = mesh.receptor_uvs.iter()
            .any(|r| r.receptor_type == ReceptorType::Granule);

        assert!(has_glom);
        assert!(has_mitral);
        assert!(has_granule);
    }

    #[test]
    fn test_combined_generation() {
        let mesh = generate_olfactory_mesh(OlfactoryView::Combined, 2);

        // Combined should have more vertices than either alone
        let epithelium = generate_olfactory_mesh(OlfactoryView::Epithelium, 2);
        let bulb = generate_olfactory_mesh(OlfactoryView::Bulb, 2);

        assert_eq!(
            mesh.vertices.len(),
            epithelium.vertices.len() + bulb.vertices.len()
        );
    }

    #[test]
    fn test_receptor_class_distribution() {
        let mesh = generate_olfactory_mesh(OlfactoryView::Epithelium, 5);

        // Check that all 8 receptor classes are represented
        let mut class_counts = [0usize; 8];
        for r in &mesh.receptor_uvs {
            if r.receptor_type == ReceptorType::OlfactoryNeuron {
                let class = r.data as usize;
                if class < 8 {
                    class_counts[class] += 1;
                }
            }
        }

        // Each class should have some receptors
        for count in class_counts {
            assert!(count > 0, "Each odorant class should be represented");
        }
    }
}

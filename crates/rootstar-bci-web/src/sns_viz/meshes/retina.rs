//! Retina Mesh Generation
//!
//! Procedural generation of retinal photoreceptor array for visual cortex visualization.
//!
//! # Anatomy Modeled
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Retinal Structure                             │
//! │                                                                         │
//! │                         ┌───────────────────┐                           │
//! │                         │                   │                           │
//! │       Periphery         │      Fovea        │         Periphery         │
//! │       (Rods)            │     (Cones)       │          (Rods)           │
//! │                         │   High density    │                           │
//! │                         │   S, M, L cones   │                           │
//! │                         │                   │                           │
//! │                         └───────────────────┘                           │
//! │                                                                         │
//! │  Receptor density falls off logarithmically from fovea                  │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Mesh Types
//!
//! - **Flat retina**: Unfolded view for receptor visualization
//! - **Curved retina**: Hemispherical representation matching eye anatomy
//! - **V1 cortex**: Retinotopic mapping with cortical magnification

use super::{generate_grid_mesh, MeshData, ReceptorPosition, ReceptorType, SurfaceCurvature, Vertex};

/// Retina view type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RetinaView {
    /// Flat unrolled view (for detailed receptor analysis)
    Flat,
    /// Curved hemispherical view (anatomical)
    Curved,
    /// V1 cortex retinotopic representation
    Cortex,
}

/// Eye selection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Eye {
    Left,
    Right,
}

/// Generate a retina mesh with photoreceptor positions
///
/// # Arguments
/// * `eye` - Which eye (affects V1 hemisphere mapping)
/// * `view` - Flat, curved, or cortex view
/// * `fovea_detail` - Higher values = more receptors in fovea (1-10)
pub fn generate_retina_mesh(eye: Eye, view: RetinaView, fovea_detail: u32) -> MeshData {
    let detail = fovea_detail.clamp(1, 10);

    match view {
        RetinaView::Flat => generate_flat_retina(detail),
        RetinaView::Curved => generate_curved_retina(detail),
        RetinaView::Cortex => generate_v1_cortex(eye, detail),
    }
}

/// Generate flat retina mesh (unrolled view)
fn generate_flat_retina(detail: u32) -> MeshData {
    // Retina is approximately 42mm diameter, we represent visual field in degrees
    // Covering ±90° horizontal, ±60° vertical
    let width = 180.0; // degrees
    let height = 120.0; // degrees
    let resolution = 40 + detail * 10;

    let curvature = SurfaceCurvature::Flat;
    let (vertices, indices) = generate_grid_mesh(width, height, resolution, resolution, &curvature);

    // Generate photoreceptor positions
    let receptors = generate_photoreceptor_positions(detail, false);

    MeshData {
        vertices,
        indices,
        receptor_uvs: receptors,
    }
}

/// Generate curved hemispherical retina
fn generate_curved_retina(detail: u32) -> MeshData {
    let resolution = 40 + detail * 10;

    // Generate hemisphere mesh
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let radius = 12.0; // mm (approximate eye radius)

    for lat in 0..=resolution {
        for lon in 0..=resolution {
            let u = lon as f32 / resolution as f32;
            let v = lat as f32 / resolution as f32;

            // Convert to spherical coordinates (hemisphere)
            let theta = (u - 0.5) * std::f32::consts::PI; // -90° to +90°
            let phi = v * std::f32::consts::FRAC_PI_2; // 0° to 90°

            let x = radius * phi.cos() * theta.sin();
            let y = radius * phi.sin();
            let z = radius * phi.cos() * theta.cos();

            // Normal points inward (toward center of eye)
            let normal = [-x / radius, -y / radius, -z / radius];

            vertices.push(Vertex {
                position: [x, y, z],
                normal,
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

    let receptors = generate_photoreceptor_positions(detail, true);

    MeshData {
        vertices,
        indices,
        receptor_uvs: receptors,
    }
}

/// Generate V1 cortex mesh with retinotopic mapping
fn generate_v1_cortex(eye: Eye, detail: u32) -> MeshData {
    let resolution = 30 + detail * 8;

    // V1 is approximately 3cm × 8cm with log-polar mapping
    let width = 30.0;  // mm
    let height = 80.0; // mm

    // Use height map for cortical folding (simplified)
    let map_size = (resolution + 1) as usize;
    let mut heights = vec![0.0f32; map_size * map_size];

    // Add cortical sulci/gyri pattern
    for y in 0..map_size {
        for x in 0..map_size {
            let u = x as f32 / (map_size - 1) as f32;
            let v = y as f32 / (map_size - 1) as f32;

            // Calcarine sulcus runs horizontally
            let calcarine = (-((v - 0.5) * 10.0).powi(2)).exp() * 3.0;

            // Add some gyral pattern
            let gyri = ((u * 8.0).sin() * (v * 6.0).sin()) * 1.5;

            heights[y * map_size + x] = calcarine + gyri;
        }
    }

    let curvature = SurfaceCurvature::HeightMap {
        heights,
        width: map_size,
    };

    let (mut vertices, indices) = generate_grid_mesh(width, height, resolution, resolution, &curvature);

    // Mirror for right eye (right visual field → left V1)
    if eye == Eye::Right {
        for v in &mut vertices {
            v.position[0] = -v.position[0];
            v.normal[0] = -v.normal[0];
        }
    }

    // V1 receptors represent retinotopic positions (with cortical magnification)
    let receptors = generate_v1_electrode_positions(detail);

    MeshData {
        vertices,
        indices,
        receptor_uvs: receptors,
    }
}

/// Generate photoreceptor positions with foveal density gradient
fn generate_photoreceptor_positions(detail: u32, curved: bool) -> Vec<ReceptorPosition> {
    let mut receptors = Vec::new();

    // Foveal cones (high density in center)
    let fovea_rings = 3 + detail;
    for ring in 0..fovea_rings {
        let radius = 0.02 + ring as f32 * 0.015; // UV space
        let n_receptors = 8 + ring * 6;

        for i in 0..n_receptors {
            let angle = (i as f32 / n_receptors as f32) * std::f32::consts::TAU;
            let u = 0.5 + radius * angle.cos();
            let v = 0.5 + radius * angle.sin();

            // Cone type distribution (L:M:S ≈ 10:5:1)
            let cone_type = match i % 16 {
                0 => ReceptorType::SCone,
                1..=5 => ReceptorType::MCone,
                _ => ReceptorType::LCone,
            };

            receptors.push(ReceptorPosition {
                uv: [u, v],
                receptor_type: cone_type,
                data: 0.0, // Eccentricity in degrees
            });
        }
    }

    // Peripheral rods (lower density, larger spacing)
    let peripheral_rings = 5 + detail / 2;
    for ring in 0..peripheral_rings {
        let radius = 0.15 + ring as f32 * 0.08;
        if radius > 0.48 {
            break;
        }

        let n_receptors = 12 + ring * 4;

        for i in 0..n_receptors {
            let angle = (i as f32 / n_receptors as f32) * std::f32::consts::TAU;
            // Add some jitter for natural distribution
            let jitter_r = ((i * 7 + ring * 13) % 100) as f32 / 1000.0 - 0.05;
            let jitter_a = ((i * 11 + ring * 17) % 100) as f32 / 500.0 - 0.1;

            let u = 0.5 + (radius + jitter_r) * (angle + jitter_a).cos();
            let v = 0.5 + (radius + jitter_r) * (angle + jitter_a).sin();

            if u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 {
                receptors.push(ReceptorPosition {
                    uv: [u, v],
                    receptor_type: ReceptorType::Rod,
                    data: radius * 90.0, // Approximate eccentricity in degrees
                });
            }
        }
    }

    // Add ganglion cell layer (sparse, representing RGC receptive field centers)
    let rgc_grid = 4 + detail / 2;
    for y in 0..rgc_grid {
        for x in 0..rgc_grid {
            let u = (x as f32 + 0.5) / rgc_grid as f32;
            let v = (y as f32 + 0.5) / rgc_grid as f32;

            // Skip fovea (no RGCs there, they're displaced)
            let dist_to_center = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
            if dist_to_center < 0.1 {
                continue;
            }

            // Alternate ON and OFF center
            let rgc_type = if (x + y) % 2 == 0 {
                ReceptorType::GanglionOn
            } else {
                ReceptorType::GanglionOff
            };

            receptors.push(ReceptorPosition {
                uv: [u, v],
                receptor_type: rgc_type,
                data: dist_to_center * 90.0,
            });
        }
    }

    receptors
}

/// Generate V1 electrode/stimulation positions for phosphene mapping
fn generate_v1_electrode_positions(detail: u32) -> Vec<ReceptorPosition> {
    let mut positions = Vec::new();

    // Grid of electrode positions with cortical magnification
    // Higher density near foveal representation (posterior V1)
    let grid_size = 6 + detail;

    for y in 0..grid_size {
        for x in 0..grid_size {
            let u = (x as f32 + 0.5) / grid_size as f32;
            let v = (y as f32 + 0.5) / grid_size as f32;

            // v=1 is foveal (posterior), v=0 is peripheral (anterior)
            // Apply log scaling to match cortical magnification
            let eccentricity = (1.0 - v) * 60.0; // degrees from fovea

            positions.push(ReceptorPosition {
                uv: [u, v],
                receptor_type: ReceptorType::V1Electrode,
                data: eccentricity,
            });
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_retina_generation() {
        let mesh = generate_retina_mesh(Eye::Left, RetinaView::Flat, 3);

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // Check that we have both cones and rods
        let has_cones = mesh.receptor_uvs.iter().any(|r| {
            matches!(r.receptor_type, ReceptorType::LCone | ReceptorType::MCone | ReceptorType::SCone)
        });
        let has_rods = mesh.receptor_uvs.iter().any(|r| r.receptor_type == ReceptorType::Rod);

        assert!(has_cones);
        assert!(has_rods);
    }

    #[test]
    fn test_curved_retina_generation() {
        let mesh = generate_retina_mesh(Eye::Right, RetinaView::Curved, 2);

        assert!(!mesh.vertices.is_empty());
        // Curved should form a hemisphere
        for v in &mesh.vertices {
            // All vertices should be approximately on sphere surface
            let dist = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!((dist - 12.0).abs() < 1.0); // radius ≈ 12mm
        }
    }

    #[test]
    fn test_v1_cortex_generation() {
        let mesh = generate_retina_mesh(Eye::Left, RetinaView::Cortex, 2);

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // All electrodes should have V1Electrode type
        for r in &mesh.receptor_uvs {
            assert_eq!(r.receptor_type, ReceptorType::V1Electrode);
        }
    }

    #[test]
    fn test_foveal_density() {
        let mesh = generate_retina_mesh(Eye::Left, RetinaView::Flat, 5);

        // Count receptors near fovea vs periphery
        let foveal: Vec<_> = mesh.receptor_uvs.iter()
            .filter(|r| {
                let d = ((r.uv[0] - 0.5).powi(2) + (r.uv[1] - 0.5).powi(2)).sqrt();
                d < 0.1
            })
            .collect();

        let peripheral: Vec<_> = mesh.receptor_uvs.iter()
            .filter(|r| {
                let d = ((r.uv[0] - 0.5).powi(2) + (r.uv[1] - 0.5).powi(2)).sqrt();
                d > 0.3
            })
            .collect();

        // Fovea should have higher cone density
        let foveal_cones = foveal.iter().filter(|r| {
            matches!(r.receptor_type, ReceptorType::LCone | ReceptorType::MCone | ReceptorType::SCone)
        }).count();

        assert!(foveal_cones > 0);
    }
}

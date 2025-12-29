//! Skin Mesh Generation for Tactile Visualization
//!
//! Generates skin surface meshes with mechanoreceptor positions
//! based on anatomical receptor density distributions.

use rootstar_bci_core::sns::types::{BodyRegion, Finger};

use super::{generate_grid_mesh, MeshData, ReceptorPosition, ReceptorType, SurfaceCurvature};

/// Receptor density (per cm²) for different body regions
#[derive(Clone, Debug)]
pub struct ReceptorDensity {
    /// Meissner corpuscle density
    pub meissner: f32,
    /// Merkel disc density
    pub merkel: f32,
    /// Pacinian corpuscle density
    pub pacinian: f32,
    /// Ruffini ending density
    pub ruffini: f32,
}

impl ReceptorDensity {
    /// Get receptor density for a body region
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
            // Default for other regions
            _ => Self {
                meissner: 20.0,
                merkel: 10.0,
                pacinian: 5.0,
                ruffini: 5.0,
            },
        }
    }

    /// Total receptor density
    pub fn total(&self) -> f32 {
        self.meissner + self.merkel + self.pacinian + self.ruffini
    }
}

/// Get dimensions for a body region (width cm, height cm)
fn region_dimensions(region: BodyRegion) -> (f32, f32, SurfaceCurvature) {
    match region {
        BodyRegion::Fingertip(_) => (
            1.5, // 1.5 cm width
            1.5, // 1.5 cm height
            SurfaceCurvature::Spherical {
                radius: 0.6,
                center: [0.0, 0.0, -0.3],
            },
        ),
        BodyRegion::Palm(_) => (
            8.0, // 8 cm width
            10.0, // 10 cm height
            SurfaceCurvature::Flat,
        ),
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

/// Generate skin surface mesh with receptor positions
pub fn generate_skin_mesh(region: BodyRegion, resolution: u32) -> MeshData {
    let (width_cm, height_cm, curvature) = region_dimensions(region);
    let density = ReceptorDensity::for_region(region);

    // Generate surface mesh
    let (vertices, indices) = generate_grid_mesh(
        width_cm,
        height_cm,
        resolution,
        resolution,
        &curvature,
    );

    // Generate receptor positions
    let area_cm2 = width_cm * height_cm;
    let receptor_uvs = generate_receptor_positions(area_cm2, &density);

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

/// Generate receptor positions based on density
fn generate_receptor_positions(area_cm2: f32, density: &ReceptorDensity) -> Vec<ReceptorPosition> {
    let mut positions = Vec::new();
    let mut rng_state = 12345u64; // Simple PRNG state

    // Helper for pseudo-random numbers
    let mut rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    // Generate Meissner corpuscles (superficial, small RF)
    let n_meissner = (density.meissner * area_cm2) as usize;
    for _ in 0..n_meissner {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Meissner,
            data: 0.0,
        });
    }

    // Generate Merkel discs (superficial, small RF)
    let n_merkel = (density.merkel * area_cm2) as usize;
    for _ in 0..n_merkel {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Merkel,
            data: 0.0,
        });
    }

    // Generate Pacinian corpuscles (deep, large RF - more spread out)
    let n_pacinian = (density.pacinian * area_cm2) as usize;
    for _ in 0..n_pacinian {
        positions.push(ReceptorPosition {
            uv: [rand(), rand()],
            receptor_type: ReceptorType::Pacinian,
            data: 0.0,
        });
    }

    // Generate Ruffini endings (deep, large RF)
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

/// Generate a detailed fingertip mesh with dermal ridges
pub fn generate_fingertip_mesh(resolution: u32) -> MeshData {
    let width_cm = 1.5;
    let height_cm = 1.5;

    // Create height map for dermal ridges
    let ridge_resolution = 64usize;
    let mut heights = vec![0.0f32; ridge_resolution * ridge_resolution];

    for y in 0..ridge_resolution {
        for x in 0..ridge_resolution {
            let u = x as f32 / ridge_resolution as f32;
            let v = y as f32 / ridge_resolution as f32;

            // Fingerprint-like pattern (simplified)
            let ridge_freq = 20.0;
            let dist_from_center = ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)).sqrt();
            let angle = (v - 0.5).atan2(u - 0.5);

            // Concentric circles with some variation
            let ridge = (dist_from_center * ridge_freq + angle * 0.5).sin();

            heights[y * ridge_resolution + x] = ridge * 0.02; // 0.02 cm ridge height
        }
    }

    let curvature = SurfaceCurvature::HeightMap {
        heights,
        width: ridge_resolution,
    };

    let (vertices, indices) = generate_grid_mesh(width_cm, height_cm, resolution, resolution, &curvature);

    // High density receptor distribution for fingertip
    let density = ReceptorDensity::for_region(BodyRegion::Fingertip(Finger::Index));
    let area_cm2 = width_cm * height_cm;
    let receptor_uvs = generate_receptor_positions(area_cm2, &density);

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skin_mesh_generation() {
        let mesh = generate_skin_mesh(BodyRegion::Fingertip, 10);

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // Check that all UVs are in valid range
        for receptor in &mesh.receptor_uvs {
            assert!(receptor.uv[0] >= 0.0 && receptor.uv[0] <= 1.0);
            assert!(receptor.uv[1] >= 0.0 && receptor.uv[1] <= 1.0);
        }
    }

    #[test]
    fn test_receptor_density() {
        let fingertip = ReceptorDensity::for_region(BodyRegion::Fingertip(Finger::Index));
        let forearm = ReceptorDensity::for_region(BodyRegion::Forearm(Side::Right));

        // Fingertip should have much higher density
        assert!(fingertip.total() > forearm.total() * 5.0);
    }
}

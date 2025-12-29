//! Cochlea Mesh Generation for Auditory Visualization
//!
//! Generates cochlea (inner ear) mesh with hair cell positions
//! mapped to characteristic frequency (tonotopic organization).

use rootstar_bci_core::sns::types::Ear;

use super::{MeshData, ReceptorPosition, ReceptorType, Vertex};

/// Cochlea physical parameters
const COCHLEA_LENGTH_MM: f32 = 35.0;
const COCHLEA_TURNS: f32 = 2.5;
const BASE_RADIUS_MM: f32 = 4.0;
const APEX_RADIUS_MM: f32 = 0.5;

/// Generate cochlea mesh
///
/// # Arguments
/// * `ear` - Which ear (affects orientation)
/// * `unrolled` - If true, shows flat representation; if false, shows spiral
pub fn generate_cochlea_mesh(ear: Ear, unrolled: bool) -> MeshData {
    let segments = 256u32;
    let radial_segments = 16u32;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut receptor_uvs = Vec::new();

    // Generate tube along cochlea
    for i in 0..=segments {
        let t = i as f32 / segments as f32;

        // Position along basilar membrane
        let length_mm = t * COCHLEA_LENGTH_MM;

        // Characteristic frequency at this position (Greenwood function)
        let freq_hz = 165.4 * (10.0_f32.powf(2.1 * (1.0 - t)) - 0.88);

        // Spiral or unrolled coordinates
        let (center_x, center_y, center_z) = if unrolled {
            (length_mm / 10.0, 0.0, 0.0) // Scale down for visualization
        } else {
            let angle = t * COCHLEA_TURNS * 2.0 * std::f32::consts::PI;
            let radius = BASE_RADIUS_MM - t * (BASE_RADIUS_MM - APEX_RADIUS_MM);

            // Mirror for right ear
            let x = match ear {
                Ear::Left => radius * angle.cos(),
                Ear::Right => -radius * angle.cos(),
            };

            (x / 10.0, radius * angle.sin() / 10.0, t * 0.5)
        };

        // Width varies along cochlea (wider at apex)
        let width = (0.5 + t * 1.0) / 10.0;

        // Generate cross-section ring
        for j in 0..=radial_segments {
            let theta = j as f32 / radial_segments as f32 * 2.0 * std::f32::consts::PI;

            let local_x = width * theta.cos();
            let local_y = width * theta.sin();

            // Compute tangent direction for proper normal orientation
            let tangent = if unrolled {
                [1.0, 0.0, 0.0]
            } else {
                let angle = t * COCHLEA_TURNS * 2.0 * std::f32::consts::PI;
                [-angle.sin(), angle.cos(), 0.1]
            };

            // Normal points outward from tube center
            let normal = [theta.cos(), theta.sin(), 0.0];

            vertices.push(Vertex {
                position: [center_x + local_x, center_y + local_y, center_z],
                normal,
                uv: [t, j as f32 / radial_segments as f32],
            });
        }

        // Add hair cell positions at this location
        // Inner hair cells (1 row)
        receptor_uvs.push(ReceptorPosition {
            uv: [t, 0.0],
            receptor_type: ReceptorType::InnerHairCell,
            data: freq_hz,
        });

        // Outer hair cells (3 rows)
        for row in 0..3 {
            receptor_uvs.push(ReceptorPosition {
                uv: [t, 0.2 + row as f32 * 0.15],
                receptor_type: ReceptorType::OuterHairCell,
                data: freq_hz,
            });
        }
    }

    // Generate indices connecting rings
    let ring_verts = radial_segments + 1;
    for i in 0..segments {
        for j in 0..radial_segments {
            let current = i * ring_verts + j;
            let next_ring = (i + 1) * ring_verts + j;

            // Two triangles per quad
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

/// Generate a simplified organ of Corti cross-section
pub fn generate_organ_of_corti_mesh() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut receptor_uvs = Vec::new();

    // Create a simplified cross-section showing IHC and OHC positions
    let width = 1.0f32;
    let height = 0.5f32;
    let resolution = 20u32;

    // Generate surface
    for y in 0..=resolution {
        for x in 0..=resolution {
            let u = x as f32 / resolution as f32;
            let v = y as f32 / resolution as f32;

            // Create slight arch shape
            let arch = 0.1 * (1.0 - (2.0 * u - 1.0).powi(2));

            vertices.push(Vertex {
                position: [(u - 0.5) * width, (v - 0.5) * height, arch],
                normal: [0.0, 0.0, 1.0],
                uv: [u, v],
            });
        }
    }

    // Generate indices
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

    // Add receptor positions (IHC and OHC)
    // IHC: single row
    for i in 0..10 {
        let u = 0.3;
        let v = (i as f32 + 0.5) / 10.0;
        receptor_uvs.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::InnerHairCell,
            data: 1000.0 + i as f32 * 500.0, // Frequency
        });
    }

    // OHC: three rows
    for row in 0..3 {
        let u = 0.5 + row as f32 * 0.1;
        for i in 0..10 {
            let v = (i as f32 + 0.5) / 10.0;
            receptor_uvs.push(ReceptorPosition {
                uv: [u, v],
                receptor_type: ReceptorType::OuterHairCell,
                data: 1000.0 + i as f32 * 500.0,
            });
        }
    }

    MeshData {
        vertices,
        indices,
        receptor_uvs,
    }
}

/// Convert frequency to position along basilar membrane (0-1)
pub fn frequency_to_position(freq_hz: f32) -> f32 {
    // Inverse Greenwood function
    const A: f32 = 165.4;
    const ALPHA: f32 = 2.1;
    const K: f32 = 0.88;

    let freq_clamped = freq_hz.clamp(20.0, 20000.0);
    let x = libm::log10f((freq_clamped / A) + K) / ALPHA;

    1.0 - x.clamp(0.0, 1.0)
}

/// Convert position along basilar membrane to frequency
pub fn position_to_frequency(t: f32) -> f32 {
    // Greenwood function
    const A: f32 = 165.4;
    const ALPHA: f32 = 2.1;
    const K: f32 = 0.88;

    let t_clamped = t.clamp(0.0, 1.0);
    A * (libm::powf(10.0, ALPHA * (1.0 - t_clamped)) - K)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cochlea_mesh_generation() {
        let mesh = generate_cochlea_mesh(Ear::Left, false);

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // Should have IHC and OHC receptors
        let ihc_count = mesh.receptor_uvs.iter()
            .filter(|r| r.receptor_type == ReceptorType::InnerHairCell)
            .count();
        let ohc_count = mesh.receptor_uvs.iter()
            .filter(|r| r.receptor_type == ReceptorType::OuterHairCell)
            .count();

        assert!(ihc_count > 0);
        assert!(ohc_count > ihc_count); // Should have ~3x more OHC
    }

    #[test]
    fn test_frequency_position_mapping() {
        // Base (high freq) should be at position 0
        let base_pos = frequency_to_position(20000.0);
        assert!(base_pos < 0.1);

        // Apex (low freq) should be at position 1
        let apex_pos = frequency_to_position(20.0);
        assert!(apex_pos > 0.9);

        // Round-trip
        let freq = 1000.0;
        let pos = frequency_to_position(freq);
        let freq_back = position_to_frequency(pos);
        assert!((freq_back - freq).abs() < 50.0);
    }
}

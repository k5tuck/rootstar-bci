//! Tongue Mesh Generation for Gustatory Visualization
//!
//! Generates tongue surface mesh with papillae and taste bud positions
//! based on anatomical distribution.

use super::{generate_grid_mesh, MeshData, ReceptorPosition, ReceptorType, SurfaceCurvature, Vertex};

/// Tongue dimensions
const TONGUE_LENGTH_CM: f32 = 10.0;
const TONGUE_WIDTH_CM: f32 = 5.0;

/// Papilla counts by type
const FUNGIFORM_COUNT: usize = 200;
const CIRCUMVALLATE_COUNT: usize = 9;
const FOLIATE_COUNT: usize = 20;

/// Generate tongue surface mesh with papillae
pub fn generate_tongue_mesh() -> MeshData {
    let resolution = 50u32;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut receptor_uvs = Vec::new();

    // Generate tongue surface with natural curvature
    for y in 0..=resolution {
        for x in 0..=resolution {
            let u = x as f32 / resolution as f32;
            let v = y as f32 / resolution as f32;

            // Tongue shape: wider at back, pointed at front
            let width_factor = 0.3 + 0.7 * v;
            let actual_x = (u - 0.5) * TONGUE_WIDTH_CM * width_factor;
            let actual_y = v * TONGUE_LENGTH_CM;

            // Surface curvature (convex)
            let center_dist = (u - 0.5).abs() * 2.0;
            let z = 0.5 * (1.0 - center_dist * center_dist);

            // Add some texture/bumps for papillae
            let bump_freq = 30.0;
            let bump = 0.02 * ((u * bump_freq).sin() * (v * bump_freq).cos());

            let position = [actual_x, actual_y - TONGUE_LENGTH_CM / 2.0, z + bump];

            // Normal (simplified - pointing up with slight variations)
            let normal = [
                -0.2 * (u - 0.5),
                -0.1,
                1.0,
            ];
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let normal = [normal[0] / len, normal[1] / len, normal[2] / len];

            vertices.push(Vertex {
                position,
                normal,
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

    // Generate papilla/taste bud positions
    let mut rng_state = 54321u64;
    let mut rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    // Fungiform papillae: scattered on anterior 2/3
    for _ in 0..FUNGIFORM_COUNT {
        let u = 0.15 + rand() * 0.7;
        let v = rand() * 0.66; // Anterior 2/3

        receptor_uvs.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::FungiformPapilla,
            data: 0.0, // Could encode taste sensitivity
        });
    }

    // Circumvallate papillae: V-shaped row at posterior
    for i in 0..CIRCUMVALLATE_COUNT {
        // V-shape arrangement
        let u = 0.2 + (i as f32 / (CIRCUMVALLATE_COUNT - 1) as f32) * 0.6;
        let v_base = 0.72;
        let v_offset = 0.08 * (1.0 - (2.0 * (i as f32 / (CIRCUMVALLATE_COUNT - 1) as f32) - 1.0).abs());
        let v = v_base + v_offset;

        receptor_uvs.push(ReceptorPosition {
            uv: [u, v],
            receptor_type: ReceptorType::CircumvallatePapilla,
            data: 0.0,
        });
    }

    // Foliate papillae: lateral edges
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

/// Generate a taste map showing regional sensitivities
///
/// Note: The "tongue map" showing distinct taste regions is a myth.
/// All taste qualities can be sensed across the tongue, but there are
/// slight regional variations in sensitivity.
pub fn generate_taste_sensitivity_map() -> [[f32; 5]; 4] {
    // [region][taste_quality]
    // Regions: Tip, Sides, Back, Center
    // Qualities: Sweet, Salty, Sour, Bitter, Umami
    [
        // Tip: slightly more sensitive to sweet and salty
        [1.1, 1.1, 0.9, 0.8, 1.0],
        // Sides: slightly more sensitive to sour
        [0.9, 1.0, 1.2, 0.9, 1.0],
        // Back: slightly more sensitive to bitter
        [0.8, 0.9, 0.9, 1.2, 1.0],
        // Center: moderate sensitivity to all
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]
}

/// Get the papilla type at a UV position
pub fn papilla_type_at(u: f32, v: f32) -> Option<ReceptorType> {
    // Check regions
    if v > 0.7 {
        // Posterior - circumvallate region
        Some(ReceptorType::CircumvallatePapilla)
    } else if u < 0.15 || u > 0.85 {
        // Lateral edges - foliate
        Some(ReceptorType::FoliatePapilla)
    } else if v < 0.66 {
        // Anterior - fungiform
        Some(ReceptorType::FungiformPapilla)
    } else {
        None
    }
}

/// Taste quality enum (for reference)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TasteQuality {
    Sweet = 0,
    Salty = 1,
    Sour = 2,
    Bitter = 3,
    Umami = 4,
}

impl TasteQuality {
    /// Get color for visualization
    pub fn color(&self) -> [u8; 4] {
        match self {
            TasteQuality::Sweet => [255, 182, 193, 255], // Pink
            TasteQuality::Salty => [173, 216, 230, 255], // Light blue
            TasteQuality::Sour => [255, 255, 0, 255],    // Yellow
            TasteQuality::Bitter => [144, 238, 144, 255], // Light green
            TasteQuality::Umami => [221, 160, 221, 255], // Plum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tongue_mesh_generation() {
        let mesh = generate_tongue_mesh();

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.receptor_uvs.is_empty());

        // Check papilla types are present
        let fungiform = mesh.receptor_uvs.iter()
            .filter(|r| r.receptor_type == ReceptorType::FungiformPapilla)
            .count();
        let circumvallate = mesh.receptor_uvs.iter()
            .filter(|r| r.receptor_type == ReceptorType::CircumvallatePapilla)
            .count();
        let foliate = mesh.receptor_uvs.iter()
            .filter(|r| r.receptor_type == ReceptorType::FoliatePapilla)
            .count();

        assert_eq!(fungiform, FUNGIFORM_COUNT);
        assert_eq!(circumvallate, CIRCUMVALLATE_COUNT);
        assert_eq!(foliate, FOLIATE_COUNT);
    }

    #[test]
    fn test_papilla_type_at() {
        // Tip should be fungiform
        assert_eq!(papilla_type_at(0.5, 0.1), Some(ReceptorType::FungiformPapilla));

        // Back should be circumvallate
        assert_eq!(papilla_type_at(0.5, 0.8), Some(ReceptorType::CircumvallatePapilla));

        // Edge should be foliate
        assert_eq!(papilla_type_at(0.1, 0.5), Some(ReceptorType::FoliatePapilla));
    }
}

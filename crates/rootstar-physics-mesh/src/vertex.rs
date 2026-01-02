//! Mesh vertex types for body representation.

use serde::{Deserialize, Serialize};

use rootstar_physics_core::types::Velocity;

/// A vertex in the body mesh.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MeshVertex {
    /// Vertex index.
    pub index: u32,
    /// 3D position.
    pub position: Velocity,
    /// Surface normal.
    pub normal: Velocity,
    /// UV texture coordinate.
    pub uv: (f32, f32),
}

impl MeshVertex {
    /// Create a new mesh vertex.
    #[must_use]
    pub fn new(index: u32, position: Velocity, normal: Velocity, uv: (f32, f32)) -> Self {
        Self {
            index,
            position,
            normal,
            uv,
        }
    }

    /// Create a simple vertex with position only.
    #[must_use]
    pub fn simple(index: u32, x: f32, y: f32, z: f32, uv_x: f32, uv_y: f32) -> Self {
        Self {
            index,
            position: Velocity::new(x, y, z),
            normal: Velocity::new(0.0, 0.0, 1.0), // Default forward
            uv: (uv_x, uv_y),
        }
    }
}

/// Buffer of mesh vertices for efficient processing.
#[derive(Debug)]
pub struct VertexBuffer {
    vertices: Vec<MeshVertex>,
}

impl VertexBuffer {
    /// Create a new empty vertex buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(capacity),
        }
    }

    /// Add a vertex to the buffer.
    pub fn push(&mut self, vertex: MeshVertex) {
        self.vertices.push(vertex);
    }

    /// Get the number of vertices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Get a vertex by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&MeshVertex> {
        self.vertices.get(index)
    }

    /// Iterate over vertices.
    pub fn iter(&self) -> impl Iterator<Item = &MeshVertex> {
        self.vertices.iter()
    }

    /// Create a simple humanoid body mesh for testing.
    ///
    /// Creates a basic 2D projection mesh with ~1000 vertices.
    #[must_use]
    pub fn create_test_body() -> Self {
        let mut buffer = Self::with_capacity(1000);

        // Create grid of vertices mapped to body regions
        let resolution = 32;
        let mut index = 0u32;

        for y in 0..resolution {
            for x in 0..resolution {
                let uv_x = x as f32 / (resolution - 1) as f32;
                let uv_y = y as f32 / (resolution - 1) as f32;

                // Map UV to approximate body shape
                let (pos_x, pos_y, pos_z) = uv_to_body_position(uv_x, uv_y);

                buffer.push(MeshVertex::new(
                    index,
                    Velocity::new(pos_x, pos_y, pos_z),
                    Velocity::new(0.0, 0.0, 1.0), // Forward facing
                    (uv_x, uv_y),
                ));

                index += 1;
            }
        }

        buffer
    }
}

impl Default for VertexBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Map UV coordinates to approximate body position.
///
/// Creates a simplified T-pose body shape.
fn uv_to_body_position(uv_x: f32, uv_y: f32) -> (f32, f32, f32) {
    // Y coordinate maps to height (feet at 0, head at 1)
    let height = uv_y * 2.0 - 1.0; // -1 to 1

    // X coordinate maps to width, but modulated by body shape
    let center = 0.5;
    let x_offset = uv_x - center;

    // Body width varies by height
    let body_width = if uv_y > 0.8 {
        // Head - narrow
        0.3
    } else if uv_y > 0.4 {
        // Torso and arms - wide
        if x_offset.abs() < 0.2 {
            0.4 // Torso
        } else {
            0.8 // Arms extended
        }
    } else {
        // Legs
        0.5
    };

    let width = x_offset * body_width;

    // Depth is mostly flat for front-facing mesh
    let depth = 0.0;

    (width, height, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation() {
        let v = MeshVertex::simple(0, 1.0, 2.0, 3.0, 0.5, 0.5);
        assert_eq!(v.index, 0);
        assert!((v.position.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vertex_buffer() {
        let mut buffer = VertexBuffer::new();
        buffer.push(MeshVertex::simple(0, 0.0, 0.0, 0.0, 0.5, 0.5));
        buffer.push(MeshVertex::simple(1, 1.0, 0.0, 0.0, 0.6, 0.5));

        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_body_mesh_creation() {
        let mesh = VertexBuffer::create_test_body();
        assert_eq!(mesh.len(), 32 * 32);

        // Check first vertex (bottom-left)
        let v = mesh.get(0).unwrap();
        assert!((v.uv.0).abs() < 0.01);
        assert!((v.uv.1).abs() < 0.01);

        // Check last vertex (top-right)
        let v = mesh.get(32 * 32 - 1).unwrap();
        assert!((v.uv.0 - 1.0).abs() < 0.01);
        assert!((v.uv.1 - 1.0).abs() < 0.01);
    }
}

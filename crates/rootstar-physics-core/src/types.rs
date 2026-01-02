//! Core physics types for VR neural mapping.
//!
//! These types are `no_std` compatible for embedded VR headsets.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A 3D velocity vector in meters per second.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    /// X component (m/s)
    pub x: f32,
    /// Y component (m/s)
    pub y: f32,
    /// Z component (m/s)
    pub z: f32,
}

impl Velocity {
    /// Create a new velocity vector.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create a zero velocity.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Get the magnitude (speed) of the velocity.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        libm::sqrtf(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Normalize the velocity to a unit vector.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag > 1e-8 {
            Self::new(self.x / mag, self.y / mag, self.z / mag)
        } else {
            Self::zero()
        }
    }

    /// Convert to nalgebra Vector3.
    #[must_use]
    pub fn to_vector3(&self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Create from nalgebra Vector3.
    #[must_use]
    pub fn from_vector3(v: &Vector3<f32>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Dot product with another velocity.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Normalized intensity value (0.0 to 1.0).
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Intensity(f32);

impl Intensity {
    /// Create a new intensity, clamping to [0.0, 1.0].
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Create zero intensity.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0.0)
    }

    /// Create maximum intensity.
    #[must_use]
    pub const fn max() -> Self {
        Self(1.0)
    }

    /// Get the raw value.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Scale by a factor.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        Self::new(self.0 * factor)
    }

    /// Blend with another intensity.
    #[must_use]
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::new(self.0 * (1.0 - t) + other.0 * t)
    }
}

impl From<f32> for Intensity {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Intensity> for f32 {
    fn from(intensity: Intensity) -> Self {
        intensity.0
    }
}

/// Frame timing information for synchronized physics.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FrameTime {
    /// Frame number since simulation start.
    pub frame_number: u64,
    /// Time since simulation start (seconds).
    pub elapsed_s: f32,
    /// Delta time since last frame (seconds).
    pub delta_s: f32,
    /// Target frame rate (Hz).
    pub target_hz: f32,
}

impl FrameTime {
    /// Create a new frame time.
    #[must_use]
    pub fn new(frame_number: u64, elapsed_s: f32, delta_s: f32, target_hz: f32) -> Self {
        Self {
            frame_number,
            elapsed_s,
            delta_s,
            target_hz,
        }
    }

    /// Create the first frame at time zero.
    #[must_use]
    pub fn first(target_hz: f32) -> Self {
        Self {
            frame_number: 0,
            elapsed_s: 0.0,
            delta_s: 1.0 / target_hz,
            target_hz,
        }
    }

    /// Advance to the next frame.
    #[must_use]
    pub fn next(&self) -> Self {
        Self {
            frame_number: self.frame_number + 1,
            elapsed_s: self.elapsed_s + self.delta_s,
            delta_s: self.delta_s,
            target_hz: self.target_hz,
        }
    }

    /// Check if we're running slower than target.
    #[must_use]
    pub fn is_lagging(&self) -> bool {
        self.delta_s > (1.0 / self.target_hz) * 1.1 // 10% tolerance
    }
}

impl Default for FrameTime {
    fn default() -> Self {
        Self::first(60.0)
    }
}

/// A complete physics frame with computed effects.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhysicsFrame {
    /// Frame timing.
    pub time: FrameTime,
    /// List of affected vertices with intensities.
    pub affected_vertices: heapless::Vec<AffectedVertex, 1024>,
}

impl PhysicsFrame {
    /// Create a new physics frame.
    #[must_use]
    pub fn new(target_hz: f32) -> Self {
        Self {
            time: FrameTime::first(target_hz),
            affected_vertices: heapless::Vec::new(),
        }
    }

    /// Create from frame time.
    #[must_use]
    pub fn from_time(time: FrameTime) -> Self {
        Self {
            time,
            affected_vertices: heapless::Vec::new(),
        }
    }

    /// Add an affected vertex.
    pub fn add_vertex(&mut self, vertex: AffectedVertex) -> Result<(), AffectedVertex> {
        self.affected_vertices.push(vertex)
    }

    /// Get the number of affected vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.affected_vertices.len()
    }

    /// Clear all affected vertices.
    pub fn clear(&mut self) {
        self.affected_vertices.clear();
    }

    /// Advance to next frame.
    pub fn advance(&mut self) {
        self.time = self.time.next();
        self.affected_vertices.clear();
    }
}

/// A vertex affected by an environmental effect.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AffectedVertex {
    /// Vertex index in the mesh.
    pub index: u32,
    /// UV coordinate X (for body region lookup).
    pub uv_x: f32,
    /// UV coordinate Y (for body region lookup).
    pub uv_y: f32,
    /// Effect intensity at this vertex.
    pub intensity: Intensity,
    /// Effect type that affected this vertex.
    pub effect_type: EffectType,
}

impl AffectedVertex {
    /// Create a new affected vertex.
    #[must_use]
    pub fn new(index: u32, uv_x: f32, uv_y: f32, intensity: Intensity, effect_type: EffectType) -> Self {
        Self {
            index,
            uv_x,
            uv_y,
            intensity,
            effect_type,
        }
    }
}

/// Type of environmental effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectType {
    /// Wind/air movement.
    Wind,
    /// Temperature (hot or cold).
    Temperature,
    /// Pressure/touch.
    Pressure,
    /// Vibration.
    Vibration,
    /// Combined sensation.
    Composite,
}

impl EffectType {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Wind => "wind",
            Self::Temperature => "temperature",
            Self::Pressure => "pressure",
            Self::Vibration => "vibration",
            Self::Composite => "composite",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_magnitude() {
        let v = Velocity::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_normalized() {
        let v = Velocity::new(0.0, 10.0, 0.0);
        let n = v.normalized();
        assert!((n.y - 1.0).abs() < 0.001);
        assert!((n.magnitude() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_intensity_clamping() {
        assert!((Intensity::new(1.5).value() - 1.0).abs() < 0.001);
        assert!((Intensity::new(-0.5).value()).abs() < 0.001);
        assert!((Intensity::new(0.5).value() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_frame_advance() {
        let frame = PhysicsFrame::new(60.0);
        assert_eq!(frame.time.frame_number, 0);

        let mut frame2 = frame;
        frame2.advance();
        assert_eq!(frame2.time.frame_number, 1);
    }
}

//! Wind simulation for VR neural mapping.
//!
//! Simulates wind vectors with direction, velocity, turbulence, and temperature.

use serde::{Deserialize, Serialize};

use crate::types::{EffectType, Intensity, Velocity};
use crate::MAX_WIND_VELOCITY;

/// Turbulence level (0.0 = laminar, 1.0 = chaotic).
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Turbulence(f32);

impl Turbulence {
    /// Create a new turbulence value, clamping to [0.0, 1.0].
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Laminar flow (no turbulence).
    #[must_use]
    pub const fn laminar() -> Self {
        Self(0.0)
    }

    /// Moderate turbulence.
    #[must_use]
    pub const fn moderate() -> Self {
        Self(0.5)
    }

    /// Chaotic turbulence.
    #[must_use]
    pub const fn chaotic() -> Self {
        Self(1.0)
    }

    /// Get the raw value.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Apply turbulence variation to a velocity.
    ///
    /// Uses a simple noise approximation for embedded compatibility.
    #[must_use]
    pub fn apply(&self, velocity: &Velocity, time_s: f32) -> Velocity {
        if self.0 < 0.01 {
            return *velocity;
        }

        // Simple pseudo-random variation based on time
        let phase = time_s * 10.0;
        let noise_x = libm::sinf(phase * 1.1) * libm::cosf(phase * 0.7);
        let noise_y = libm::sinf(phase * 0.9) * libm::cosf(phase * 1.3);
        let noise_z = libm::sinf(phase * 1.2) * libm::cosf(phase * 0.8);

        let variation = self.0 * 0.3; // Max 30% variation at full turbulence
        Velocity::new(
            velocity.x * (1.0 + noise_x * variation),
            velocity.y * (1.0 + noise_y * variation),
            velocity.z * (1.0 + noise_z * variation),
        )
    }
}

/// Wind simulation parameters.
///
/// Represents a wind vector with associated environmental properties.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WindVector {
    /// Base velocity (direction and speed in m/s).
    pub velocity: Velocity,
    /// Turbulence level.
    pub turbulence: Turbulence,
    /// Temperature in Celsius.
    pub temperature_c: f32,
    /// Humidity percentage (0-100).
    pub humidity_pct: f32,
}

impl WindVector {
    /// Create a new wind vector from direction components.
    ///
    /// # Arguments
    ///
    /// * `x` - X velocity component (m/s)
    /// * `y` - Y velocity component (m/s)
    /// * `z` - Z velocity component (m/s)
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            velocity: Velocity::new(x, y, z),
            turbulence: Turbulence::laminar(),
            temperature_c: 20.0, // Room temperature
            humidity_pct: 50.0,
        }
    }

    /// Create still air (no wind).
    #[must_use]
    pub fn still() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create a gentle breeze (5 m/s).
    #[must_use]
    pub fn gentle_breeze(direction: Velocity) -> Self {
        let normalized = direction.normalized();
        Self::new(normalized.x * 5.0, normalized.y * 5.0, normalized.z * 5.0)
            .with_turbulence(Turbulence::new(0.1))
    }

    /// Create a strong wind (15 m/s).
    #[must_use]
    pub fn strong_wind(direction: Velocity) -> Self {
        let normalized = direction.normalized();
        Self::new(
            normalized.x * 15.0,
            normalized.y * 15.0,
            normalized.z * 15.0,
        )
        .with_turbulence(Turbulence::moderate())
    }

    /// Set the turbulence level.
    #[must_use]
    pub fn with_turbulence(mut self, turbulence: Turbulence) -> Self {
        self.turbulence = turbulence;
        self
    }

    /// Set the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature_c: f32) -> Self {
        self.temperature_c = temperature_c.clamp(-20.0, 50.0);
        self
    }

    /// Set the humidity.
    #[must_use]
    pub fn with_humidity(mut self, humidity_pct: f32) -> Self {
        self.humidity_pct = humidity_pct.clamp(0.0, 100.0);
        self
    }

    /// Get the wind speed (magnitude of velocity) in m/s.
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.velocity.magnitude()
    }

    /// Get the normalized wind direction.
    #[must_use]
    pub fn direction(&self) -> Velocity {
        self.velocity.normalized()
    }

    /// Get the effective velocity at a given time (with turbulence).
    #[must_use]
    pub fn effective_velocity(&self, time_s: f32) -> Velocity {
        self.turbulence.apply(&self.velocity, time_s)
    }

    /// Calculate intensity at a surface based on angle of incidence.
    ///
    /// # Arguments
    ///
    /// * `surface_normal` - Normal vector of the surface
    ///
    /// # Returns
    ///
    /// Intensity scaled by the cosine of the angle of incidence.
    #[must_use]
    pub fn intensity_at_surface(&self, surface_normal: &Velocity) -> Intensity {
        let wind_dir = self.direction();
        let normal = surface_normal.normalized();

        // Dot product gives cosine of angle
        // Negate because wind hitting the surface is opposite to normal
        let cos_angle = -wind_dir.dot(&normal);

        // Only positive (facing) surfaces are affected
        let facing_factor = cos_angle.max(0.0);

        // Scale by wind speed (normalized to max)
        let speed_factor = (self.speed() / MAX_WIND_VELOCITY).min(1.0);

        Intensity::new(facing_factor * speed_factor)
    }

    /// Get the effect type for this wind.
    #[must_use]
    pub const fn effect_type(&self) -> EffectType {
        EffectType::Wind
    }

    /// Check if this is effectively still air.
    #[must_use]
    pub fn is_still(&self) -> bool {
        self.speed() < 0.1
    }

    /// Calculate perceived temperature with wind chill.
    ///
    /// Uses a simplified wind chill formula.
    #[must_use]
    pub fn perceived_temperature(&self) -> f32 {
        let speed_kmh = self.speed() * 3.6; // Convert m/s to km/h
        let t = self.temperature_c;

        if t > 10.0 || speed_kmh < 4.8 {
            // Wind chill only applies below 10°C with significant wind
            t
        } else {
            // Simplified wind chill formula
            13.12 + 0.6215 * t - 11.37 * libm::powf(speed_kmh, 0.16)
                + 0.3965 * t * libm::powf(speed_kmh, 0.16)
        }
    }
}

impl Default for WindVector {
    fn default() -> Self {
        Self::still()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_speed() {
        let wind = WindVector::new(3.0, 4.0, 0.0);
        assert!((wind.speed() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_wind_direction() {
        let wind = WindVector::new(0.0, 10.0, 0.0);
        let dir = wind.direction();
        assert!((dir.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_intensity_facing() {
        let wind = WindVector::new(0.0, 0.0, -10.0); // Wind coming from +Z
        let normal = Velocity::new(0.0, 0.0, 1.0); // Surface facing +Z

        let intensity = wind.intensity_at_surface(&normal);
        assert!(intensity.value() > 0.1); // Should be affected
    }

    #[test]
    fn test_intensity_away() {
        let wind = WindVector::new(0.0, 0.0, 10.0); // Wind going to +Z
        let normal = Velocity::new(0.0, 0.0, 1.0); // Surface facing +Z

        let intensity = wind.intensity_at_surface(&normal);
        assert!(intensity.value() < 0.001); // Should not be affected (wind from behind)
    }

    #[test]
    fn test_turbulence() {
        let velocity = Velocity::new(10.0, 0.0, 0.0);
        let turbulence = Turbulence::chaotic();

        let v1 = turbulence.apply(&velocity, 0.0);
        let v2 = turbulence.apply(&velocity, 0.1);

        // Velocities should be different due to turbulence
        assert!((v1.x - v2.x).abs() > 0.01 || (v1.y - v2.y).abs() > 0.01);
    }

    #[test]
    fn test_gentle_breeze() {
        let breeze = WindVector::gentle_breeze(Velocity::new(1.0, 0.0, 0.0));
        assert!((breeze.speed() - 5.0).abs() < 0.001);
    }
}

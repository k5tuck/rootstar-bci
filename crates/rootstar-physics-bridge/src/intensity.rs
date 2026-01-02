//! Intensity calculation for VR to neural stimulation mapping.
//!
//! Implements the intensity formula from the VR Neural Mapping spec:
//! ```text
//! intensity = (affected_vertices / total_region_vertices) *
//!             (wind_velocity / max_velocity) *
//!             cos(angle_of_incidence) *
//!             temporal_smoothing_factor
//! ```

use rootstar_physics_core::types::{Intensity, Velocity};
use rootstar_physics_core::wind::WindVector;
use rootstar_physics_core::MAX_WIND_VELOCITY;
use rootstar_physics_mesh::BodyRegion;

/// Calculator for stimulation intensity.
pub struct IntensityCalculator {
    /// Maximum wind velocity for normalization.
    max_velocity: f32,
    /// Temporal smoothing factor.
    smoothing_factor: f32,
}

impl IntensityCalculator {
    /// Create a new intensity calculator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_velocity: MAX_WIND_VELOCITY,
            smoothing_factor: 0.8,
        }
    }

    /// Set custom max velocity.
    #[must_use]
    pub fn with_max_velocity(mut self, velocity: f32) -> Self {
        self.max_velocity = velocity.max(1.0);
        self
    }

    /// Set temporal smoothing factor.
    #[must_use]
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing_factor = factor.clamp(0.1, 1.0);
        self
    }

    /// Calculate wind intensity at a body region.
    ///
    /// Uses the formula:
    /// ```text
    /// intensity = coverage_ratio * velocity_factor * facing_factor * smoothing
    /// ```
    #[must_use]
    pub fn calculate_wind_intensity(
        &self,
        region: &BodyRegion,
        wind: &WindVector,
        surface_normal: &Velocity,
        affected_vertex_count: u32,
    ) -> Intensity {
        // Coverage ratio: affected vertices / total vertices
        let coverage = region.coverage_ratio(affected_vertex_count);

        // Velocity factor: wind speed / max speed
        let velocity_factor = (wind.speed() / self.max_velocity).min(1.0);

        // Facing factor: cos(angle) between wind direction and surface
        let wind_dir = wind.direction();
        let cos_angle = -wind_dir.dot(&surface_normal.normalized());
        let facing_factor = cos_angle.max(0.0); // Only facing surfaces

        // Combined intensity
        let raw_intensity = coverage * velocity_factor * facing_factor * self.smoothing_factor;

        Intensity::new(raw_intensity)
    }

    /// Calculate temperature intensity at a body region.
    ///
    /// Based on deviation from body temperature (37°C).
    #[must_use]
    pub fn calculate_temperature_intensity(
        &self,
        temperature_c: f32,
        affected_vertex_count: u32,
        total_vertices: u32,
    ) -> Intensity {
        // Coverage factor
        let coverage = if total_vertices == 0 {
            0.0
        } else {
            (affected_vertex_count as f32 / total_vertices as f32).min(1.0)
        };

        // Temperature deviation from body temp
        let deviation = (temperature_c - 37.0).abs();

        // Normalize: 20°C deviation = full intensity
        let temp_factor = (deviation / 20.0).min(1.0);

        Intensity::new(coverage * temp_factor * self.smoothing_factor)
    }

    /// Calculate combined intensity for multiple effects.
    #[must_use]
    pub fn combine_intensities(&self, intensities: &[Intensity]) -> Intensity {
        if intensities.is_empty() {
            return Intensity::zero();
        }

        // Use RMS (root mean square) for combination
        let sum_squares: f32 = intensities.iter().map(|i| i.value() * i.value()).sum();
        let rms = libm::sqrtf(sum_squares / intensities.len() as f32);

        Intensity::new(rms)
    }

    /// Apply temporal smoothing between frames.
    #[must_use]
    pub fn smooth(&self, previous: Intensity, current: Intensity, alpha: f32) -> Intensity {
        let alpha = alpha.clamp(0.0, 1.0);
        let smoothed = previous.value() * (1.0 - alpha) + current.value() * alpha;
        Intensity::new(smoothed)
    }
}

impl Default for IntensityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_physics_mesh::{BodyRegionId, SensitivityLevel};

    fn create_test_region() -> BodyRegion {
        BodyRegion::new(
            BodyRegionId::ArmLeftHand,
            0.05,
            0.15,
            0.35,
            0.45,
            500,
            SensitivityLevel::high(),
        )
    }

    #[test]
    fn test_wind_intensity() {
        let calc = IntensityCalculator::new();
        let region = create_test_region();
        let wind = WindVector::new(25.0, 0.0, 0.0); // 25 m/s = 50% of max
        let normal = Velocity::new(-1.0, 0.0, 0.0); // Facing into wind

        let intensity = calc.calculate_wind_intensity(&region, &wind, &normal, 250); // 50% coverage

        // Expected: 0.5 (coverage) * 0.5 (velocity) * 1.0 (facing) * 0.8 (smoothing) = 0.2
        assert!(intensity.value() > 0.15);
        assert!(intensity.value() < 0.25);
    }

    #[test]
    fn test_wind_not_facing() {
        let calc = IntensityCalculator::new();
        let region = create_test_region();
        let wind = WindVector::new(25.0, 0.0, 0.0);
        let normal = Velocity::new(1.0, 0.0, 0.0); // Facing away from wind

        let intensity = calc.calculate_wind_intensity(&region, &wind, &normal, 250);

        // Should be near zero since surface faces away
        assert!(intensity.value() < 0.01);
    }

    #[test]
    fn test_temperature_intensity() {
        let calc = IntensityCalculator::new();

        // Cold: 17°C (20° below body temp)
        let cold_intensity = calc.calculate_temperature_intensity(17.0, 100, 200);
        assert!(cold_intensity.value() > 0.3); // Should be significant

        // Body temp: should be near zero
        let neutral_intensity = calc.calculate_temperature_intensity(37.0, 100, 200);
        assert!(neutral_intensity.value() < 0.01);
    }

    #[test]
    fn test_combine_intensities() {
        let calc = IntensityCalculator::new();

        let intensities = vec![Intensity::new(0.5), Intensity::new(0.5), Intensity::new(0.5)];

        let combined = calc.combine_intensities(&intensities);
        // RMS of [0.5, 0.5, 0.5] = 0.5
        assert!((combined.value() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_temporal_smoothing() {
        let calc = IntensityCalculator::new();

        let prev = Intensity::new(0.0);
        let curr = Intensity::new(1.0);

        // 50% smoothing
        let smoothed = calc.smooth(prev, curr, 0.5);
        assert!((smoothed.value() - 0.5).abs() < 0.01);

        // 0% smoothing = keep previous
        let no_change = calc.smooth(prev, curr, 0.0);
        assert!(no_change.value() < 0.01);

        // 100% smoothing = use current
        let full_change = calc.smooth(prev, curr, 1.0);
        assert!((full_change.value() - 1.0).abs() < 0.01);
    }
}

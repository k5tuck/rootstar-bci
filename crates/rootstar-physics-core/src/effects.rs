//! Environmental effects for VR neural mapping.
//!
//! Includes temperature fields and composite sensations.

use serde::{Deserialize, Serialize};

use crate::types::{EffectType, Intensity, Velocity};
use crate::wind::WindVector;
use crate::TEMP_RANGE;

/// Temperature field for thermal sensations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TemperatureField {
    /// Base temperature in Celsius.
    pub temperature_c: f32,
    /// Temperature gradient direction (warmer in this direction).
    pub gradient: Velocity,
    /// Gradient strength (°C per meter).
    pub gradient_strength: f32,
}

impl TemperatureField {
    /// Create a uniform temperature field.
    #[must_use]
    pub fn uniform(temperature_c: f32) -> Self {
        Self {
            temperature_c: temperature_c.clamp(TEMP_RANGE.0, TEMP_RANGE.1),
            gradient: Velocity::zero(),
            gradient_strength: 0.0,
        }
    }

    /// Create a temperature with gradient.
    #[must_use]
    pub fn with_gradient(temperature_c: f32, direction: Velocity, strength: f32) -> Self {
        Self {
            temperature_c: temperature_c.clamp(TEMP_RANGE.0, TEMP_RANGE.1),
            gradient: direction.normalized(),
            gradient_strength: strength,
        }
    }

    /// Room temperature (20°C).
    #[must_use]
    pub fn room_temperature() -> Self {
        Self::uniform(20.0)
    }

    /// Cold environment (-5°C).
    #[must_use]
    pub fn cold() -> Self {
        Self::uniform(-5.0)
    }

    /// Hot environment (35°C).
    #[must_use]
    pub fn hot() -> Self {
        Self::uniform(35.0)
    }

    /// Get temperature at a position relative to origin.
    #[must_use]
    pub fn temperature_at(&self, position: &Velocity) -> f32 {
        let offset = position.dot(&self.gradient) * self.gradient_strength;
        (self.temperature_c + offset).clamp(TEMP_RANGE.0, TEMP_RANGE.1)
    }

    /// Calculate intensity based on deviation from body temperature (37°C).
    #[must_use]
    pub fn sensation_intensity(&self, position: &Velocity) -> Intensity {
        let temp = self.temperature_at(position);
        let deviation = (temp - 37.0).abs(); // Body temperature is 37°C

        // Normalize: 20°C deviation = full intensity
        Intensity::new(deviation / 20.0)
    }

    /// Determine if sensation is cold (true) or hot (false).
    #[must_use]
    pub fn is_cold(&self, position: &Velocity) -> bool {
        self.temperature_at(position) < 37.0
    }

    /// Get the effect type.
    #[must_use]
    pub const fn effect_type(&self) -> EffectType {
        EffectType::Temperature
    }
}

impl Default for TemperatureField {
    fn default() -> Self {
        Self::room_temperature()
    }
}

/// Combined environmental effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentalEffect {
    /// Wind component.
    pub wind: Option<WindVector>,
    /// Temperature component.
    pub temperature: Option<TemperatureField>,
    /// Pressure/touch component (normalized 0-1).
    pub pressure: Option<Intensity>,
    /// Vibration frequency (Hz).
    pub vibration_hz: Option<f32>,
}

impl EnvironmentalEffect {
    /// Create an empty effect.
    #[must_use]
    pub fn none() -> Self {
        Self {
            wind: None,
            temperature: None,
            pressure: None,
            vibration_hz: None,
        }
    }

    /// Create a wind-only effect.
    #[must_use]
    pub fn wind(wind: WindVector) -> Self {
        Self {
            wind: Some(wind),
            temperature: None,
            pressure: None,
            vibration_hz: None,
        }
    }

    /// Create a temperature-only effect.
    #[must_use]
    pub fn temperature(temp: TemperatureField) -> Self {
        Self {
            wind: None,
            temperature: Some(temp),
            pressure: None,
            vibration_hz: None,
        }
    }

    /// Create a cold wind effect.
    #[must_use]
    pub fn cold_wind(velocity: Velocity, temperature_c: f32) -> Self {
        Self {
            wind: Some(
                WindVector::new(velocity.x, velocity.y, velocity.z)
                    .with_temperature(temperature_c)
                    .with_turbulence(crate::wind::Turbulence::moderate()),
            ),
            temperature: Some(TemperatureField::uniform(temperature_c)),
            pressure: None,
            vibration_hz: None,
        }
    }

    /// Add pressure to the effect.
    #[must_use]
    pub fn with_pressure(mut self, pressure: Intensity) -> Self {
        self.pressure = Some(pressure);
        self
    }

    /// Add vibration to the effect.
    #[must_use]
    pub fn with_vibration(mut self, frequency_hz: f32) -> Self {
        self.vibration_hz = Some(frequency_hz.clamp(0.0, 1000.0));
        self
    }

    /// Check if any effect is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.wind.is_some()
            || self.temperature.is_some()
            || self.pressure.is_some()
            || self.vibration_hz.is_some()
    }

    /// Get the primary effect type.
    #[must_use]
    pub fn primary_effect_type(&self) -> EffectType {
        if self.wind.is_some() && self.temperature.is_some() {
            EffectType::Composite
        } else if self.wind.is_some() {
            EffectType::Wind
        } else if self.temperature.is_some() {
            EffectType::Temperature
        } else if self.pressure.is_some() {
            EffectType::Pressure
        } else if self.vibration_hz.is_some() {
            EffectType::Vibration
        } else {
            EffectType::Composite
        }
    }

    /// Calculate combined intensity at a surface.
    #[must_use]
    pub fn combined_intensity(&self, surface_normal: &Velocity, position: &Velocity) -> Intensity {
        let mut total = 0.0f32;
        let mut count = 0;

        if let Some(ref wind) = self.wind {
            total += wind.intensity_at_surface(surface_normal).value();
            count += 1;
        }

        if let Some(ref temp) = self.temperature {
            total += temp.sensation_intensity(position).value();
            count += 1;
        }

        if let Some(ref pressure) = self.pressure {
            total += pressure.value();
            count += 1;
        }

        if count > 0 {
            Intensity::new(total / count as f32)
        } else {
            Intensity::zero()
        }
    }
}

impl Default for EnvironmentalEffect {
    fn default() -> Self {
        Self::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_sensation() {
        let cold = TemperatureField::uniform(17.0); // 20°C below body temp
        let intensity = cold.sensation_intensity(&Velocity::zero());
        assert!(intensity.value() > 0.9); // Should be near full intensity

        let room = TemperatureField::room_temperature();
        let intensity = room.sensation_intensity(&Velocity::zero());
        assert!(intensity.value() > 0.5); // Still noticeable (17°C below body temp)
    }

    #[test]
    fn test_cold_wind() {
        let effect = EnvironmentalEffect::cold_wind(Velocity::new(10.0, 0.0, 0.0), 5.0);
        assert!(effect.wind.is_some());
        assert!(effect.temperature.is_some());
        assert_eq!(effect.primary_effect_type(), EffectType::Composite);
    }

    #[test]
    fn test_gradient_temperature() {
        let field = TemperatureField::with_gradient(20.0, Velocity::new(1.0, 0.0, 0.0), 2.0);

        let origin_temp = field.temperature_at(&Velocity::zero());
        let far_temp = field.temperature_at(&Velocity::new(5.0, 0.0, 0.0));

        assert!((origin_temp - 20.0).abs() < 0.01);
        assert!((far_temp - 30.0).abs() < 0.01); // 5m * 2°C/m = +10°C
    }
}

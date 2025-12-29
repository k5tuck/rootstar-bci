//! Tactile Receptor Models
//!
//! Phenomenological models for cutaneous mechanoreceptors:
//! - **Meissner corpuscle**: Rapid adaptation (RA-I), light touch, flutter
//! - **Merkel disc**: Slow adaptation (SA-I), pressure, edges
//! - **Pacinian corpuscle**: Very rapid adaptation (RA-II), vibration
//! - **Ruffini ending**: Slow adaptation (SA-II), skin stretch
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::tactile::MeissnerReceptor;
//!
//! let mut receptor = MeissnerReceptor::default();
//! let rate = receptor.compute_rate(50.0, 1.0);
//! assert!(rate > receptor.baseline_rate);
//! ```

use core::f32::consts::PI;

use serde::{Deserialize, Serialize};

use super::types::{BodyRegion, ReceptorType};

// ============================================================================
// Tactile Receptor Trait
// ============================================================================

/// Common interface for tactile receptor models
pub trait TactileReceptor {
    /// Compute instantaneous firing rate from stimulus
    ///
    /// # Arguments
    /// * `stimulus` - Stimulus intensity (force in mN or indentation in µm)
    /// * `dt_ms` - Time step in milliseconds
    ///
    /// # Returns
    /// Firing rate in Hz
    fn compute_rate(&mut self, stimulus: f32, dt_ms: f32) -> f32;

    /// Reset the receptor state to baseline
    fn reset(&mut self);

    /// Get the receptor type
    fn receptor_type(&self) -> ReceptorType;

    /// Get baseline firing rate (Hz)
    fn baseline_rate(&self) -> f32;

    /// Get maximum firing rate (Hz)
    fn max_rate(&self) -> f32;

    /// Get adaptation time constant (ms)
    fn tau_adapt(&self) -> f32;

    /// Get sensitivity to velocity (for RA types)
    fn velocity_sensitivity(&self) -> f32 {
        0.0
    }
}

// ============================================================================
// Meissner Corpuscle (RA-I)
// ============================================================================

/// Meissner corpuscle model - rapid adaptation type I
///
/// Located in dermal papillae of glabrous (hairless) skin.
/// Responds to light touch and flutter vibration (5-50 Hz).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeissnerReceptor {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Half-activation stimulus intensity
    pub k_half: f32,
    /// Hill coefficient (response steepness)
    pub n_hill: f32,
    /// Adaptation time constant (ms) - rapid for Meissner
    pub tau_adapt: f32,
    /// Velocity sensitivity coefficient
    pub velocity_sensitivity: f32,
    /// Current adaptation state (internal)
    adaptation_state: f32,
    /// Previous stimulus for velocity calculation
    prev_stimulus: f32,
}

impl MeissnerReceptor {
    /// Create a new Meissner receptor with specified parameters
    #[must_use]
    pub fn new(
        baseline_rate: f32,
        max_rate: f32,
        k_half: f32,
        tau_adapt: f32,
    ) -> Self {
        Self {
            baseline_rate,
            max_rate,
            k_half,
            n_hill: 2.0,
            tau_adapt,
            velocity_sensitivity: 0.5,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }

    /// Create for a specific body region
    #[must_use]
    pub fn for_region(region: BodyRegion) -> Self {
        let density_factor = region.receptor_density() as f32 / 240.0; // Normalize to fingertip
        let sensitivity = 1.0 / density_factor.max(0.1); // Higher density = more sensitive

        Self {
            baseline_rate: 2.0,
            max_rate: 120.0 * density_factor,
            k_half: 10.0 * sensitivity,
            n_hill: 2.0,
            tau_adapt: 50.0,
            velocity_sensitivity: 0.6,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }
}

impl TactileReceptor for MeissnerReceptor {
    fn compute_rate(&mut self, stimulus: f32, dt_ms: f32) -> f32 {
        // Calculate stimulus velocity (derivative)
        let velocity = (stimulus - self.prev_stimulus) / dt_ms.max(0.001);
        self.prev_stimulus = stimulus;

        // Meissner responds to both position and velocity
        let effective_stimulus = stimulus + self.velocity_sensitivity * velocity.abs();

        // Hill equation for steady-state response
        let s_n = libm::powf(effective_stimulus.max(0.0), self.n_hill);
        let k_n = libm::powf(self.k_half, self.n_hill);
        let steady_state = if s_n + k_n > 0.0001 {
            (self.max_rate - self.baseline_rate) * (s_n / (k_n + s_n))
        } else {
            0.0
        };

        // Rapid adaptation: quick decay toward baseline
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        self.adaptation_state += alpha * (steady_state - self.adaptation_state);

        // Only respond during changes (rapid adaptation)
        let response = self.adaptation_state * (1.0 + velocity.abs() * 0.1);

        self.baseline_rate + response.max(0.0)
    }

    fn reset(&mut self) {
        self.adaptation_state = 0.0;
        self.prev_stimulus = 0.0;
    }

    fn receptor_type(&self) -> ReceptorType {
        ReceptorType::Meissner
    }

    fn baseline_rate(&self) -> f32 {
        self.baseline_rate
    }

    fn max_rate(&self) -> f32 {
        self.max_rate
    }

    fn tau_adapt(&self) -> f32 {
        self.tau_adapt
    }

    fn velocity_sensitivity(&self) -> f32 {
        self.velocity_sensitivity
    }
}

impl Default for MeissnerReceptor {
    fn default() -> Self {
        Self {
            baseline_rate: 2.0,
            max_rate: 120.0,
            k_half: 15.0,
            n_hill: 2.0,
            tau_adapt: 50.0,
            velocity_sensitivity: 0.6,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }
}

// ============================================================================
// Merkel Disc (SA-I)
// ============================================================================

/// Merkel disc model - slow adaptation type I
///
/// Located at the base of the epidermis.
/// Responds to sustained pressure and edge detection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MerkelReceptor {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Half-activation stimulus intensity
    pub k_half: f32,
    /// Hill coefficient (response steepness)
    pub n_hill: f32,
    /// Adaptation time constant (ms) - slow for Merkel
    pub tau_adapt: f32,
    /// Dynamic response component weight
    pub dynamic_weight: f32,
    /// Current adaptation state
    adaptation_state: f32,
    /// Previous stimulus
    prev_stimulus: f32,
}

impl MerkelReceptor {
    /// Create a new Merkel receptor
    #[must_use]
    pub fn new(
        baseline_rate: f32,
        max_rate: f32,
        k_half: f32,
        tau_adapt: f32,
    ) -> Self {
        Self {
            baseline_rate,
            max_rate,
            k_half,
            n_hill: 1.5,
            tau_adapt,
            dynamic_weight: 0.3,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }

    /// Create for a specific body region
    #[must_use]
    pub fn for_region(region: BodyRegion) -> Self {
        let density_factor = region.receptor_density() as f32 / 240.0;

        Self {
            baseline_rate: 5.0,
            max_rate: 80.0 * density_factor.max(0.5),
            k_half: 20.0,
            n_hill: 1.5,
            tau_adapt: 500.0,
            dynamic_weight: 0.3,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }
}

impl TactileReceptor for MerkelReceptor {
    fn compute_rate(&mut self, stimulus: f32, dt_ms: f32) -> f32 {
        // Calculate dynamic component (response to change)
        let delta = stimulus - self.prev_stimulus;
        self.prev_stimulus = stimulus;

        // Hill equation for static response
        let s_n = libm::powf(stimulus.max(0.0), self.n_hill);
        let k_n = libm::powf(self.k_half, self.n_hill);
        let static_response = if s_n + k_n > 0.0001 {
            (self.max_rate - self.baseline_rate) * (s_n / (k_n + s_n))
        } else {
            0.0
        };

        // Dynamic component (transient boost)
        let dynamic_response = delta.abs() * self.dynamic_weight * (self.max_rate - self.baseline_rate);

        // Slow adaptation
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        let target = static_response + dynamic_response;
        self.adaptation_state += alpha * (target - self.adaptation_state);

        // SA-I maintains sustained response
        self.baseline_rate + self.adaptation_state.max(0.0)
    }

    fn reset(&mut self) {
        self.adaptation_state = 0.0;
        self.prev_stimulus = 0.0;
    }

    fn receptor_type(&self) -> ReceptorType {
        ReceptorType::Merkel
    }

    fn baseline_rate(&self) -> f32 {
        self.baseline_rate
    }

    fn max_rate(&self) -> f32 {
        self.max_rate
    }

    fn tau_adapt(&self) -> f32 {
        self.tau_adapt
    }
}

impl Default for MerkelReceptor {
    fn default() -> Self {
        Self {
            baseline_rate: 5.0,
            max_rate: 80.0,
            k_half: 20.0,
            n_hill: 1.5,
            tau_adapt: 500.0,
            dynamic_weight: 0.3,
            adaptation_state: 0.0,
            prev_stimulus: 0.0,
        }
    }
}

// ============================================================================
// Pacinian Corpuscle (RA-II)
// ============================================================================

/// Pacinian corpuscle model - very rapid adaptation type II
///
/// Located in deep dermis and subcutaneous tissue.
/// Responds to high-frequency vibration (40-400 Hz).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PacinianReceptor {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Half-activation velocity
    pub k_half: f32,
    /// Preferred vibration frequency (Hz)
    pub preferred_freq: f32,
    /// Frequency tuning bandwidth (Q factor)
    pub q_factor: f32,
    /// Adaptation time constant (ms) - very rapid
    pub tau_adapt: f32,
    /// Bandpass filter state (high-pass)
    hp_state: f32,
    /// Bandpass filter state (low-pass)
    lp_state: f32,
    /// Previous stimulus
    prev_stimulus: f32,
    /// Adaptation state
    adaptation_state: f32,
}

impl PacinianReceptor {
    /// Create a new Pacinian receptor
    #[must_use]
    pub fn new(preferred_freq: f32, max_rate: f32) -> Self {
        Self {
            baseline_rate: 0.0,
            max_rate,
            k_half: 5.0,
            preferred_freq,
            q_factor: 1.5,
            tau_adapt: 5.0,
            hp_state: 0.0,
            lp_state: 0.0,
            prev_stimulus: 0.0,
            adaptation_state: 0.0,
        }
    }

    /// Compute frequency response gain
    fn frequency_gain(&self, freq_hz: f32) -> f32 {
        // Bandpass characteristic centered on preferred frequency
        let ratio = freq_hz / self.preferred_freq;
        let log_ratio = libm::logf(ratio);
        libm::expf(-log_ratio * log_ratio * self.q_factor)
    }
}

impl TactileReceptor for PacinianReceptor {
    fn compute_rate(&mut self, stimulus: f32, dt_ms: f32) -> f32 {
        // High-pass filter to extract velocity/acceleration
        let hp_cutoff = 30.0; // Hz
        let hp_alpha = 1.0 - libm::expf(-dt_ms * hp_cutoff / 1000.0 * 2.0 * PI);
        let hp_input = stimulus - self.prev_stimulus;
        self.hp_state += hp_alpha * (hp_input - self.hp_state);
        self.prev_stimulus = stimulus;

        // Low-pass filter to remove very high frequencies
        let lp_cutoff = 500.0; // Hz
        let lp_alpha = 1.0 - libm::expf(-dt_ms * lp_cutoff / 1000.0 * 2.0 * PI);
        self.lp_state += lp_alpha * (self.hp_state - self.lp_state);

        // Rectify and compute magnitude
        let magnitude = self.lp_state.abs();

        // Very rapid adaptation
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        self.adaptation_state += alpha * (magnitude * 10.0 - self.adaptation_state);

        // Sigmoidal response
        let response = if self.adaptation_state > 0.0 {
            self.max_rate * (1.0 - libm::expf(-self.adaptation_state / self.k_half))
        } else {
            0.0
        };

        self.baseline_rate + response.max(0.0)
    }

    fn reset(&mut self) {
        self.hp_state = 0.0;
        self.lp_state = 0.0;
        self.prev_stimulus = 0.0;
        self.adaptation_state = 0.0;
    }

    fn receptor_type(&self) -> ReceptorType {
        ReceptorType::Pacinian
    }

    fn baseline_rate(&self) -> f32 {
        self.baseline_rate
    }

    fn max_rate(&self) -> f32 {
        self.max_rate
    }

    fn tau_adapt(&self) -> f32 {
        self.tau_adapt
    }

    fn velocity_sensitivity(&self) -> f32 {
        1.0 // Maximum velocity sensitivity
    }
}

impl Default for PacinianReceptor {
    fn default() -> Self {
        Self::new(250.0, 300.0)
    }
}

// ============================================================================
// Ruffini Ending (SA-II)
// ============================================================================

/// Ruffini ending model - slow adaptation type II
///
/// Located in the dermis, aligned with collagen fibers.
/// Responds to sustained skin stretch and finger position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuffiniReceptor {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Half-activation stretch (normalized)
    pub k_half: f32,
    /// Hill coefficient
    pub n_hill: f32,
    /// Adaptation time constant (ms) - slow
    pub tau_adapt: f32,
    /// Preferred stretch direction (radians)
    pub preferred_direction: f32,
    /// Directional tuning width
    pub direction_tuning: f32,
    /// Current adaptation state
    adaptation_state: f32,
}

impl RuffiniReceptor {
    /// Create a new Ruffini receptor
    #[must_use]
    pub fn new(preferred_direction: f32, max_rate: f32) -> Self {
        Self {
            baseline_rate: 10.0,
            max_rate,
            k_half: 0.3,
            n_hill: 1.2,
            tau_adapt: 1000.0,
            preferred_direction,
            direction_tuning: 0.5,
            adaptation_state: 0.0,
        }
    }

    /// Compute directional gain for stretch
    #[must_use]
    pub fn direction_gain(&self, stretch_direction: f32) -> f32 {
        let diff = (stretch_direction - self.preferred_direction).abs();
        let wrapped = if diff > PI { 2.0 * PI - diff } else { diff };
        libm::expf(-wrapped * wrapped / (2.0 * self.direction_tuning * self.direction_tuning))
    }

    /// Compute response to stretch magnitude and direction
    pub fn compute_rate_directional(&mut self, stretch_magnitude: f32, stretch_direction: f32, dt_ms: f32) -> f32 {
        let dir_gain = self.direction_gain(stretch_direction);
        let effective_stretch = stretch_magnitude * dir_gain;
        self.compute_rate(effective_stretch, dt_ms)
    }
}

impl TactileReceptor for RuffiniReceptor {
    fn compute_rate(&mut self, stimulus: f32, dt_ms: f32) -> f32 {
        // Hill equation for stretch response
        let s_n = libm::powf(stimulus.max(0.0), self.n_hill);
        let k_n = libm::powf(self.k_half, self.n_hill);
        let steady_state = if s_n + k_n > 0.0001 {
            (self.max_rate - self.baseline_rate) * (s_n / (k_n + s_n))
        } else {
            0.0
        };

        // Very slow adaptation (essentially tonic response)
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        self.adaptation_state += alpha * (steady_state - self.adaptation_state);

        // SA-II maintains sustained response
        self.baseline_rate + self.adaptation_state.max(0.0)
    }

    fn reset(&mut self) {
        self.adaptation_state = 0.0;
    }

    fn receptor_type(&self) -> ReceptorType {
        ReceptorType::Ruffini
    }

    fn baseline_rate(&self) -> f32 {
        self.baseline_rate
    }

    fn max_rate(&self) -> f32 {
        self.max_rate
    }

    fn tau_adapt(&self) -> f32 {
        self.tau_adapt
    }
}

impl Default for RuffiniReceptor {
    fn default() -> Self {
        Self::new(0.0, 60.0)
    }
}

// ============================================================================
// Tactile Population Builder
// ============================================================================

/// Builder for creating tactile receptor populations
#[derive(Clone, Debug)]
pub struct TactilePopulationBuilder {
    /// Region of the body
    region: BodyRegion,
    /// Meissner receptor density factor
    meissner_density: f32,
    /// Merkel receptor density factor
    merkel_density: f32,
    /// Pacinian receptor density factor
    pacinian_density: f32,
    /// Ruffini receptor density factor
    ruffini_density: f32,
}

impl TactilePopulationBuilder {
    /// Create a new builder for a body region
    #[must_use]
    pub fn new(region: BodyRegion) -> Self {
        Self {
            region,
            meissner_density: 1.0,
            merkel_density: 1.0,
            pacinian_density: 0.5,
            ruffini_density: 0.3,
        }
    }

    /// Set Meissner density factor
    #[must_use]
    pub fn meissner_density(mut self, density: f32) -> Self {
        self.meissner_density = density;
        self
    }

    /// Set Merkel density factor
    #[must_use]
    pub fn merkel_density(mut self, density: f32) -> Self {
        self.merkel_density = density;
        self
    }

    /// Calculate total receptor count for given area
    pub fn receptor_count(&self, area_mm2: f32) -> TactileReceptorCounts {
        let base_density = self.region.receptor_density() as f32;
        let area_cm2 = area_mm2 / 100.0;

        TactileReceptorCounts {
            meissner: (base_density * area_cm2 * 0.43 * self.meissner_density) as u16,
            merkel: (base_density * area_cm2 * 0.25 * self.merkel_density) as u16,
            pacinian: (base_density * area_cm2 * 0.02 * self.pacinian_density) as u16,
            ruffini: (base_density * area_cm2 * 0.30 * self.ruffini_density) as u16,
        }
    }
}

/// Receptor counts by type
#[derive(Clone, Debug, Default)]
pub struct TactileReceptorCounts {
    /// Meissner corpuscle count
    pub meissner: u16,
    /// Merkel disc count
    pub merkel: u16,
    /// Pacinian corpuscle count
    pub pacinian: u16,
    /// Ruffini ending count
    pub ruffini: u16,
}

impl TactileReceptorCounts {
    /// Total receptor count
    #[must_use]
    pub fn total(&self) -> u16 {
        self.meissner + self.merkel + self.pacinian + self.ruffini
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meissner_rapid_adaptation() {
        let mut receptor = MeissnerReceptor::default();

        // Apply constant stimulus
        let mut rates = [0.0f32; 20];
        for i in 0..20 {
            rates[i] = receptor.compute_rate(50.0, 10.0);
        }

        // Initial response should be higher than final (adaptation)
        assert!(rates[0] > rates[19], "Meissner should adapt rapidly");
        assert!(rates[19] >= receptor.baseline_rate, "Should stay above baseline");
    }

    #[test]
    fn test_merkel_sustained_response() {
        let mut receptor = MerkelReceptor::default();

        // Apply constant stimulus for a long time
        let mut final_rate = 0.0;
        for _ in 0..100 {
            final_rate = receptor.compute_rate(50.0, 10.0);
        }

        // SA-I should maintain elevated response
        assert!(final_rate > receptor.baseline_rate * 2.0, "Merkel should maintain response");
    }

    #[test]
    fn test_pacinian_vibration_response() {
        let mut receptor = PacinianReceptor::default();

        // Apply vibration (sinusoidal at ~250 Hz)
        let mut max_rate = 0.0f32;
        for i in 0..100 {
            let t = i as f32 * 0.001; // 1ms steps
            let stimulus = 10.0 * libm::sinf(2.0 * PI * 250.0 * t);
            let rate = receptor.compute_rate(stimulus, 1.0);
            max_rate = max_rate.max(rate);
        }

        // Should respond to vibration
        assert!(max_rate > receptor.baseline_rate, "Pacinian should respond to vibration");
    }

    #[test]
    fn test_ruffini_directional_tuning() {
        let mut receptor = RuffiniReceptor::new(0.0, 60.0);

        // Stretch in preferred direction
        let rate_aligned = receptor.compute_rate_directional(0.5, 0.0, 10.0);
        receptor.reset();

        // Stretch perpendicular to preferred direction
        let rate_perp = receptor.compute_rate_directional(0.5, PI / 2.0, 10.0);

        assert!(rate_aligned > rate_perp, "Ruffini should be direction-tuned");
    }

    #[test]
    fn test_population_builder() {
        let builder = TactilePopulationBuilder::new(BodyRegion::Fingertip(super::super::types::Finger::Index));
        let counts = builder.receptor_count(100.0); // 1 cm²

        assert!(counts.total() > 0);
        assert!(counts.meissner > counts.pacinian);
    }
}

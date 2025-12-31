//! Auditory Receptor Models
//!
//! Phenomenological models for cochlear hair cells:
//! - **Inner Hair Cells (IHC)**: Primary transduction to auditory nerve
//! - **Outer Hair Cells (OHC)**: Cochlear amplification
//! - **Basilar Membrane**: Tonotopic frequency decomposition
//!
//! # Tonotopic Organization
//!
//! The cochlea maps frequency to position along the basilar membrane:
//! - Base (near oval window): High frequencies (20 kHz)
//! - Apex: Low frequencies (20 Hz)
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::auditory::{InnerHairCell, BasilarMembrane};
//!
//! let bm = BasilarMembrane::new();
//! let position = bm.frequency_to_position(1000.0); // 1 kHz
//!
//! let mut ihc = InnerHairCell::at_frequency(1000.0);
//! let rate = ihc.compute_rate(0.5, 0.1);
//! ```

use serde::{Deserialize, Serialize};

use super::types::Ear;

// ============================================================================
// Physical Constants
// ============================================================================

/// Human audible frequency range
pub const MIN_FREQUENCY_HZ: f32 = 20.0;
/// Maximum audible frequency
pub const MAX_FREQUENCY_HZ: f32 = 20000.0;

/// Cochlea length in millimeters
pub const COCHLEA_LENGTH_MM: f32 = 35.0;

/// Reference sound pressure (20 µPa)
pub const REFERENCE_PRESSURE_PA: f32 = 20e-6;

/// Damage threshold in dB SPL (sustained)
pub const DAMAGE_THRESHOLD_DB: f32 = 85.0;

/// Peak damage threshold in dB SPL
pub const PEAK_DAMAGE_DB: f32 = 120.0;

// ============================================================================
// Hair Cell Type
// ============================================================================

/// Type of cochlear hair cell
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HairCellType {
    /// Inner hair cell - primary sensory transduction
    Inner,
    /// Outer hair cell - cochlear amplifier
    Outer,
}

impl HairCellType {
    /// Get the number of cells per row
    #[inline]
    #[must_use]
    pub const fn cells_per_row(self) -> u8 {
        match self {
            Self::Inner => 1, // Single row
            Self::Outer => 3, // Three rows
        }
    }

    /// Get the approximate total count along cochlea
    #[inline]
    #[must_use]
    pub const fn total_count(self) -> u16 {
        match self {
            Self::Inner => 3500,
            Self::Outer => 12000,
        }
    }
}

// ============================================================================
// Cochlear Position
// ============================================================================

/// Position along the cochlea
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CochlearPosition {
    /// Distance from apex in millimeters (0 = apex, 35 = base)
    pub distance_mm: f32,
    /// Corresponding characteristic frequency in Hz
    pub frequency_hz: f32,
    /// Which ear
    pub ear: Ear,
}

impl CochlearPosition {
    /// Create a position from distance
    #[must_use]
    pub fn from_distance(distance_mm: f32, ear: Ear) -> Self {
        let freq = BasilarMembrane::position_to_frequency(distance_mm);
        Self { distance_mm, frequency_hz: freq, ear }
    }

    /// Create a position from frequency
    #[must_use]
    pub fn from_frequency(frequency_hz: f32, ear: Ear) -> Self {
        let dist = BasilarMembrane::frequency_to_position(frequency_hz);
        Self { distance_mm: dist, frequency_hz, ear }
    }
}

// ============================================================================
// Basilar Membrane
// ============================================================================

/// Basilar membrane model with tonotopic mapping
///
/// Uses the Greenwood function to map frequency to position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BasilarMembrane {
    /// Length in millimeters
    pub length_mm: f32,
    /// Greenwood constant A (Hz)
    pub greenwood_a: f32,
    /// Greenwood constant alpha
    pub greenwood_alpha: f32,
    /// Greenwood constant K
    pub greenwood_k: f32,
    /// Which ear
    pub ear: Ear,
}

impl BasilarMembrane {
    /// Create a new basilar membrane model
    #[must_use]
    pub fn new() -> Self {
        Self::for_ear(Ear::Left)
    }

    /// Create for specific ear
    #[must_use]
    pub fn for_ear(ear: Ear) -> Self {
        Self {
            length_mm: COCHLEA_LENGTH_MM,
            greenwood_a: 165.4,
            greenwood_alpha: 2.1,
            greenwood_k: 0.88,
            ear,
        }
    }

    /// Convert frequency (Hz) to position (mm from apex)
    ///
    /// Uses the Greenwood function: x = (1/α) × log₁₀((f/A) + k)
    #[must_use]
    pub fn frequency_to_position(frequency_hz: f32) -> f32 {
        const A: f32 = 165.4;
        const ALPHA: f32 = 2.1;
        const K: f32 = 0.88;

        let freq_clamped = frequency_hz.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
        let position_normalized = libm::log10f((freq_clamped / A) + K) / ALPHA;

        // Convert to mm from apex (invert so apex = 0, base = 35)
        COCHLEA_LENGTH_MM * (1.0 - position_normalized.clamp(0.0, 1.0))
    }

    /// Convert position (mm from apex) to frequency (Hz)
    ///
    /// Inverse Greenwood function: f = A × (10^(α×(L-x)/L) - k)
    #[must_use]
    pub fn position_to_frequency(distance_mm: f32) -> f32 {
        const A: f32 = 165.4;
        const ALPHA: f32 = 2.1;
        const K: f32 = 0.88;

        let dist_clamped = distance_mm.clamp(0.0, COCHLEA_LENGTH_MM);
        let position_normalized = 1.0 - (dist_clamped / COCHLEA_LENGTH_MM);

        A * (libm::powf(10.0, ALPHA * position_normalized) - K)
    }

    /// Get the Equivalent Rectangular Bandwidth (ERB) at a frequency
    ///
    /// ERB(f) = 24.7 × (4.37f/1000 + 1)
    #[must_use]
    pub fn erb_hz(frequency_hz: f32) -> f32 {
        24.7 * (4.37 * frequency_hz / 1000.0 + 1.0)
    }

    /// Get the critical band number (Bark scale)
    #[must_use]
    pub fn bark(frequency_hz: f32) -> f32 {
        let f_khz = frequency_hz / 1000.0;
        let ratio = f_khz / 7.5;
        13.0 * libm::atanf(0.76 * f_khz) + 3.5 * libm::atanf(ratio * ratio)
    }

    /// Compute basilar membrane displacement at a position for a given frequency
    #[must_use]
    pub fn displacement(&self, input_freq_hz: f32, position_mm: f32, amplitude: f32) -> f32 {
        let cf = Self::position_to_frequency(position_mm);
        let erb = Self::erb_hz(cf);

        // Gammatone-like frequency response
        let freq_diff = (input_freq_hz - cf).abs();
        let normalized_diff = freq_diff / erb;

        // Response falls off with distance from CF
        amplitude * libm::expf(-normalized_diff * normalized_diff / 2.0)
    }
}

impl Default for BasilarMembrane {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Inner Hair Cell (IHC)
// ============================================================================

/// Inner hair cell model - primary auditory transduction
///
/// IHCs convert basilar membrane motion into neural signals.
/// They are responsible for ~95% of auditory nerve fibers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InnerHairCell {
    /// Characteristic frequency (Hz)
    pub cf_hz: f32,
    /// Spontaneous firing rate (Hz)
    pub spontaneous_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Threshold in dB SPL
    pub threshold_db: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Adaptation time constant (ms)
    pub tau_adapt: f32,
    /// Current adaptation state
    adaptation_state: f32,
    /// Vesicle release probability
    release_prob: f32,
}

impl InnerHairCell {
    /// Create an IHC at a specific characteristic frequency
    #[must_use]
    pub fn at_frequency(cf_hz: f32) -> Self {
        // Threshold varies with frequency (audiogram shape)
        let threshold = Self::threshold_at_frequency(cf_hz);

        Self {
            cf_hz,
            spontaneous_rate: 20.0,
            max_rate: 300.0,
            threshold_db: threshold,
            dynamic_range_db: 30.0,
            tau_adapt: 5.0,
            adaptation_state: 0.0,
            release_prob: 0.0,
        }
    }

    /// Create a high spontaneous rate IHC (most common)
    #[must_use]
    pub fn high_sr(cf_hz: f32) -> Self {
        let mut ihc = Self::at_frequency(cf_hz);
        ihc.spontaneous_rate = 50.0;
        ihc.threshold_db -= 20.0; // More sensitive
        ihc
    }

    /// Create a low spontaneous rate IHC
    #[must_use]
    pub fn low_sr(cf_hz: f32) -> Self {
        let mut ihc = Self::at_frequency(cf_hz);
        ihc.spontaneous_rate = 2.0;
        ihc.threshold_db += 20.0; // Less sensitive but wider dynamic range
        ihc.dynamic_range_db = 50.0;
        ihc
    }

    /// Get threshold at a given frequency (simplified audiogram)
    fn threshold_at_frequency(freq_hz: f32) -> f32 {
        // U-shaped audiogram with minimum around 1-4 kHz
        let log_freq = libm::log10f(freq_hz);
        let optimal = 3.3; // log10(2000)

        let diff = log_freq - optimal;
        10.0 + 20.0 * diff * diff // Minimum at ~2 kHz
    }

    /// Compute firing rate from basilar membrane displacement
    ///
    /// # Arguments
    /// * `displacement` - Normalized basilar membrane displacement (0-1)
    /// * `dt_ms` - Time step in milliseconds
    #[must_use]
    pub fn compute_rate(&mut self, displacement: f32, dt_ms: f32) -> f32 {
        // Convert displacement to dB-like measure
        let level = 20.0 * libm::log10f(displacement.max(1e-10));

        // Rate-level function (sigmoid)
        let level_above_threshold = level - self.threshold_db;
        let normalized = level_above_threshold / self.dynamic_range_db;
        let sigmoid = 1.0 / (1.0 + libm::expf(-5.0 * (normalized - 0.5)));

        let steady_state = self.spontaneous_rate +
            (self.max_rate - self.spontaneous_rate) * sigmoid;

        // Adaptation
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        self.adaptation_state += alpha * (steady_state - self.adaptation_state);

        // Update vesicle release probability
        self.release_prob = (self.adaptation_state / self.max_rate).clamp(0.0, 1.0);

        self.adaptation_state.max(self.spontaneous_rate)
    }

    /// Get current vesicle release probability
    #[inline]
    #[must_use]
    pub fn release_probability(&self) -> f32 {
        self.release_prob
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.adaptation_state = 0.0;
        self.release_prob = 0.0;
    }
}

impl Default for InnerHairCell {
    fn default() -> Self {
        Self::at_frequency(1000.0)
    }
}

// ============================================================================
// Outer Hair Cell (OHC)
// ============================================================================

/// Outer hair cell model - cochlear amplifier
///
/// OHCs provide active mechanical amplification of basilar membrane motion.
/// They are the most vulnerable to noise damage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuterHairCell {
    /// Characteristic frequency (Hz)
    pub cf_hz: f32,
    /// Amplification gain at low levels (dB)
    pub max_gain_db: f32,
    /// Compression ratio (output/input at high levels)
    pub compression_ratio: f32,
    /// Compression knee point (dB SPL)
    pub compression_knee_db: f32,
    /// Health factor (0 = dead, 1 = fully functional)
    pub health: f32,
    /// Electromotility state
    motility_state: f32,
}

impl OuterHairCell {
    /// Create an OHC at a specific characteristic frequency
    #[must_use]
    pub fn at_frequency(cf_hz: f32) -> Self {
        Self {
            cf_hz,
            max_gain_db: 50.0,
            compression_ratio: 0.2,
            compression_knee_db: 40.0,
            health: 1.0,
            motility_state: 0.0,
        }
    }

    /// Compute amplification gain for a given input level
    ///
    /// # Arguments
    /// * `input_db` - Input level in dB SPL
    ///
    /// # Returns
    /// Gain in dB (0 to max_gain_db)
    #[must_use]
    pub fn compute_gain(&self, input_db: f32) -> f32 {
        if input_db < self.compression_knee_db {
            // Linear region: full gain
            self.max_gain_db * self.health
        } else {
            // Compression region
            let excess = input_db - self.compression_knee_db;
            let compressed_gain = self.max_gain_db - excess * (1.0 - self.compression_ratio);
            (compressed_gain * self.health).max(0.0)
        }
    }

    /// Compute electromotility response
    ///
    /// OHCs physically change length in response to stimulation.
    pub fn compute_motility(&mut self, membrane_potential: f32, dt_ms: f32) -> f32 {
        // Boltzmann function for motility
        let v_half = -0.5; // Half-activation voltage (normalized)
        let slope = 0.1;

        let activation = 1.0 / (1.0 + libm::expf(-(membrane_potential - v_half) / slope));

        // Low-pass filter the motility response
        let tau_motility = 0.1; // Fast response
        let alpha = 1.0 - libm::expf(-dt_ms / tau_motility);
        self.motility_state += alpha * (activation - self.motility_state);

        // Return length change (-1 to +1, normalized)
        (self.motility_state - 0.5) * 2.0 * self.health
    }

    /// Simulate noise damage
    pub fn apply_damage(&mut self, exposure_db: f32, duration_hours: f32) {
        if exposure_db > DAMAGE_THRESHOLD_DB {
            let excess = exposure_db - DAMAGE_THRESHOLD_DB;
            let damage_rate = 0.01 * libm::powf(2.0, excess / 3.0); // 3 dB trading ratio
            let damage = (damage_rate * duration_hours).min(1.0);
            self.health = (self.health - damage).max(0.0);
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.motility_state = 0.0;
    }
}

impl Default for OuterHairCell {
    fn default() -> Self {
        Self::at_frequency(1000.0)
    }
}

// ============================================================================
// Auditory Receptor (Combined Model)
// ============================================================================

/// Combined auditory receptor model (IHC + OHC at same CF)
#[derive(Clone, Debug)]
pub struct AuditoryReceptor {
    /// Basilar membrane model
    pub basilar_membrane: BasilarMembrane,
    /// Inner hair cell
    pub ihc: InnerHairCell,
    /// Outer hair cells (3 rows)
    pub ohc: [OuterHairCell; 3],
    /// Position along cochlea
    pub position: CochlearPosition,
    /// Current basilar membrane displacement
    displacement: f32,
}

impl AuditoryReceptor {
    /// Create a receptor at a specific frequency
    #[must_use]
    pub fn at_frequency(cf_hz: f32, ear: Ear) -> Self {
        Self {
            basilar_membrane: BasilarMembrane::for_ear(ear),
            ihc: InnerHairCell::at_frequency(cf_hz),
            ohc: [
                OuterHairCell::at_frequency(cf_hz),
                OuterHairCell::at_frequency(cf_hz),
                OuterHairCell::at_frequency(cf_hz),
            ],
            position: CochlearPosition::from_frequency(cf_hz, ear),
            displacement: 0.0,
        }
    }

    /// Process an audio sample at this receptor's position
    ///
    /// # Arguments
    /// * `frequency_hz` - Frequency of the input signal
    /// * `amplitude_db` - Amplitude in dB SPL
    /// * `dt_ms` - Time step in milliseconds
    ///
    /// # Returns
    /// Firing rate in Hz
    pub fn process(&mut self, frequency_hz: f32, amplitude_db: f32, dt_ms: f32) -> f32 {
        // Compute OHC amplification (average of 3 cells)
        let ohc_gain: f32 = self.ohc.iter()
            .map(|o| o.compute_gain(amplitude_db))
            .sum::<f32>() / 3.0;

        // Amplified level
        let amplified_db = amplitude_db + ohc_gain;

        // Convert to linear amplitude
        let amplitude_linear = libm::powf(10.0, amplified_db / 20.0) / 1e6; // Normalize

        // Basilar membrane displacement
        self.displacement = self.basilar_membrane.displacement(
            frequency_hz,
            self.position.distance_mm,
            amplitude_linear,
        );

        // IHC response
        self.ihc.compute_rate(self.displacement, dt_ms)
    }

    /// Get the mean OHC health
    #[must_use]
    pub fn ohc_health(&self) -> f32 {
        self.ohc.iter().map(|o| o.health).sum::<f32>() / 3.0
    }

    /// Apply noise damage to OHCs
    pub fn apply_damage(&mut self, exposure_db: f32, duration_hours: f32) {
        for ohc in &mut self.ohc {
            ohc.apply_damage(exposure_db, duration_hours);
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.ihc.reset();
        for ohc in &mut self.ohc {
            ohc.reset();
        }
        self.displacement = 0.0;
    }
}

// ============================================================================
// Auditory Population Builder
// ============================================================================

/// Builder for auditory receptor populations
#[derive(Clone, Debug)]
pub struct AuditoryPopulationBuilder {
    /// Ear selection
    ear: Ear,
    /// Minimum frequency to include (Hz)
    min_freq: f32,
    /// Maximum frequency to include (Hz)
    max_freq: f32,
    /// Spacing in ERB units
    erb_spacing: f32,
}

impl AuditoryPopulationBuilder {
    /// Create a new builder with default settings (left ear)
    #[must_use]
    pub fn new() -> Self {
        Self {
            ear: Ear::Left,
            min_freq: 100.0,
            max_freq: 8000.0,
            erb_spacing: 0.5,
        }
    }

    /// Create a new builder for specified ear
    #[must_use]
    pub fn for_ear(ear: Ear) -> Self {
        Self {
            ear,
            min_freq: 100.0,
            max_freq: 8000.0,
            erb_spacing: 0.5,
        }
    }

    /// Set frequency range
    #[must_use]
    pub fn frequency_range(mut self, min_hz: f32, max_hz: f32) -> Self {
        self.min_freq = min_hz.max(MIN_FREQUENCY_HZ);
        self.max_freq = max_hz.min(MAX_FREQUENCY_HZ);
        self
    }

    /// Set ERB spacing
    #[must_use]
    pub fn erb_spacing(mut self, spacing: f32) -> Self {
        self.erb_spacing = spacing.max(0.1);
        self
    }

    /// Calculate the number of channels needed
    #[must_use]
    pub fn channel_count(&self) -> usize {
        let erb_min = BasilarMembrane::erb_hz(self.min_freq);
        let erb_max = BasilarMembrane::erb_hz(self.max_freq);

        // Approximate number of ERBs in range
        let n_erbs = (erb_max - erb_min) / self.erb_spacing;
        (libm::ceilf(n_erbs) as usize).max(1)
    }

    /// Get center frequencies for the population
    pub fn center_frequencies(&self) -> heapless::Vec<f32, 128> {
        let mut freqs = heapless::Vec::new();

        let mut freq = self.min_freq;
        while freq <= self.max_freq && freqs.len() < 128 {
            let _ = freqs.push(freq);
            let erb = BasilarMembrane::erb_hz(freq);
            freq += erb * self.erb_spacing;
        }

        freqs
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greenwood_function() {
        // Base should be high frequency
        let freq_base = BasilarMembrane::position_to_frequency(35.0);
        let freq_apex = BasilarMembrane::position_to_frequency(0.0);

        assert!(freq_base > 10000.0, "Base should be >10kHz");
        assert!(freq_apex < 500.0, "Apex should be <500Hz");

        // Round-trip
        let pos = BasilarMembrane::frequency_to_position(1000.0);
        let freq_back = BasilarMembrane::position_to_frequency(pos);
        assert!((freq_back - 1000.0).abs() < 10.0);
    }

    #[test]
    fn test_erb() {
        let erb_1k = BasilarMembrane::erb_hz(1000.0);
        let erb_4k = BasilarMembrane::erb_hz(4000.0);

        assert!(erb_1k > 100.0 && erb_1k < 200.0);
        assert!(erb_4k > erb_1k); // ERB increases with frequency
    }

    #[test]
    fn test_ihc_rate_level() {
        let mut ihc = InnerHairCell::high_sr(1000.0);

        // Low stimulus
        let rate_low = ihc.compute_rate(0.001, 1.0);
        ihc.reset();

        // High stimulus
        let rate_high = ihc.compute_rate(0.1, 1.0);

        assert!(rate_high > rate_low);
        assert!(rate_low >= ihc.spontaneous_rate * 0.9);
    }

    #[test]
    fn test_ohc_compression() {
        let ohc = OuterHairCell::at_frequency(1000.0);

        let gain_low = ohc.compute_gain(30.0);
        let gain_high = ohc.compute_gain(80.0);

        assert!(gain_low > gain_high); // Compression at high levels
        assert!(gain_low > 40.0); // Near full gain at low levels
    }

    #[test]
    fn test_population_builder() {
        let builder = AuditoryPopulationBuilder::for_ear(Ear::Left)
            .frequency_range(200.0, 4000.0);

        let count = builder.channel_count();
        assert!(count > 10 && count < 100);

        let freqs = builder.center_frequencies();
        assert!(!freqs.is_empty());
        assert!(*freqs.first().unwrap() >= 200.0);
    }
}

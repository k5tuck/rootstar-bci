//! fNIRS signal processing
//!
//! Implements the Modified Beer-Lambert Law for computing hemoglobin
//! concentration changes from optical density measurements.

use rootstar_bci_core::math::constants;
use rootstar_bci_core::types::{Fixed24_8, FnirsSample, HemodynamicSample, Wavelength};

/// fNIRS signal processor
#[derive(Clone, Debug)]
pub struct FnirsProcessor {
    /// Source-detector distance in cm
    distance_cm: f64,
    /// DPF at 760nm
    dpf_760: f64,
    /// DPF at 850nm
    dpf_850: f64,
    /// Baseline log intensity at 760nm
    baseline_log_760: f64,
    /// Baseline log intensity at 850nm
    baseline_log_850: f64,
    /// Precomputed inverse matrix elements (scaled by 1e6 for µM output)
    inv_matrix: [[f64; 2]; 2],
}

impl FnirsProcessor {
    /// Create a new processor with given parameters
    ///
    /// # Arguments
    ///
    /// * `separation_mm` - Source-detector distance in mm
    /// * `age_years` - Subject age for DPF lookup
    #[must_use]
    pub fn new(separation_mm: u8, age_years: u8) -> Self {
        let distance_cm = f64::from(separation_mm) / 10.0;
        let dpf_760 = dpf_for_age(age_years, Wavelength::Nm760);
        let dpf_850 = dpf_for_age(age_years, Wavelength::Nm850);

        let mut processor = Self {
            distance_cm,
            dpf_760,
            dpf_850,
            baseline_log_760: 0.0,
            baseline_log_850: 0.0,
            inv_matrix: [[0.0; 2]; 2],
        };

        processor.compute_inverse_matrix();
        processor
    }

    /// Create with default adult DPF values
    #[must_use]
    pub fn with_adult_dpf(separation_mm: u8) -> Self {
        let distance_cm = f64::from(separation_mm) / 10.0;

        let mut processor = Self {
            distance_cm,
            dpf_760: f64::from(constants::DPF_760NM_ADULT),
            dpf_850: f64::from(constants::DPF_850NM_ADULT),
            baseline_log_760: 0.0,
            baseline_log_850: 0.0,
            inv_matrix: [[0.0; 2]; 2],
        };

        processor.compute_inverse_matrix();
        processor
    }

    /// Set baseline intensities for ΔOD calculation
    pub fn set_baseline(&mut self, intensity_760: u16, intensity_850: u16) {
        self.baseline_log_760 = f64::from(intensity_760).log10();
        self.baseline_log_850 = f64::from(intensity_850).log10();
    }

    /// Compute the inverse matrix for Beer-Lambert Law
    fn compute_inverse_matrix(&mut self) {
        let path_760 = self.distance_cm * self.dpf_760;
        let path_850 = self.distance_cm * self.dpf_850;

        // Matrix: [ε_HbO2_λ * L_λ, ε_HbR_λ * L_λ]
        let a11 = constants::E_HBO2_760 * path_760;
        let a12 = constants::E_HBR_760 * path_760;
        let a21 = constants::E_HBO2_850 * path_850;
        let a22 = constants::E_HBR_850 * path_850;

        let det = a11 * a22 - a12 * a21;

        // Inverse matrix scaled by 1e6 (mol/L → µM)
        self.inv_matrix[0][0] = a22 / det * 1e6;
        self.inv_matrix[0][1] = -a12 / det * 1e6;
        self.inv_matrix[1][0] = -a21 / det * 1e6;
        self.inv_matrix[1][1] = a11 / det * 1e6;
    }

    /// Process an fNIRS sample to compute hemodynamic changes
    pub fn process(&self, sample: &FnirsSample) -> HemodynamicSample {
        // Calculate optical density changes
        // ΔOD = log10(I0) - log10(I) = -log10(I/I0)
        let od_760 = if sample.intensity_760 > 0 {
            self.baseline_log_760 - f64::from(sample.intensity_760).log10()
        } else {
            0.0
        };

        let od_850 = if sample.intensity_850 > 0 {
            self.baseline_log_850 - f64::from(sample.intensity_850).log10()
        } else {
            0.0
        };

        // Apply inverse matrix to get concentration changes
        let delta_hbo2 = self.inv_matrix[0][0] * od_760 + self.inv_matrix[0][1] * od_850;
        let delta_hbr = self.inv_matrix[1][0] * od_760 + self.inv_matrix[1][1] * od_850;

        HemodynamicSample::new(
            sample.timestamp_us,
            sample.channel,
            Fixed24_8::from_f32(delta_hbo2 as f32),
            Fixed24_8::from_f32(delta_hbr as f32),
        )
    }
}

/// Get DPF based on age and wavelength
///
/// Uses empirical formulas from Duncan et al. (1996)
fn dpf_for_age(age_years: u8, wavelength: Wavelength) -> f64 {
    let base = match wavelength {
        Wavelength::Nm760 => 5.13,
        Wavelength::Nm850 => 4.99,
    };

    // DPF increases slightly with age
    base + 0.07 * (f64::from(age_years) / 50.0)
}

/// Hemodynamic filter for smoothing
#[derive(Clone, Debug)]
pub struct HemodynamicFilter {
    alpha: f64,
    state_hbo2: f64,
    state_hbr: f64,
}

impl HemodynamicFilter {
    /// Create a new filter with given time constant
    ///
    /// # Arguments
    ///
    /// * `sample_rate_hz` - fNIRS sample rate
    /// * `time_constant_s` - Filter time constant in seconds
    #[must_use]
    pub fn new(sample_rate_hz: f64, time_constant_s: f64) -> Self {
        let dt = 1.0 / sample_rate_hz;
        let alpha = dt / (time_constant_s + dt);

        Self {
            alpha,
            state_hbo2: 0.0,
            state_hbr: 0.0,
        }
    }

    /// Filter a hemodynamic sample
    pub fn filter(&mut self, sample: &HemodynamicSample) -> HemodynamicSample {
        let hbo2 = f64::from(sample.delta_hbo2.to_f32());
        let hbr = f64::from(sample.delta_hbr.to_f32());

        self.state_hbo2 = self.alpha * hbo2 + (1.0 - self.alpha) * self.state_hbo2;
        self.state_hbr = self.alpha * hbr + (1.0 - self.alpha) * self.state_hbr;

        HemodynamicSample::new(
            sample.timestamp_us,
            sample.channel,
            Fixed24_8::from_f32(self.state_hbo2 as f32),
            Fixed24_8::from_f32(self.state_hbr as f32),
        )
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state_hbo2 = 0.0;
        self.state_hbr = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnirs_processor() {
        let mut processor = FnirsProcessor::with_adult_dpf(30);

        // Set baseline
        processor.set_baseline(40000, 38000);

        // Process a sample with slightly lower intensity (simulating activation)
        let sample = FnirsSample::new(
            1000000,
            FnirsChannel::new(0, 0, 30),
            39500, // Slightly lower at 760nm
            37600, // Slightly lower at 850nm
            1,
        );

        let result = processor.process(&sample);

        // Check that we get reasonable values
        assert!(result.delta_hbo2.to_f32().abs() < 100.0);
        assert!(result.delta_hbr.to_f32().abs() < 100.0);
    }
}

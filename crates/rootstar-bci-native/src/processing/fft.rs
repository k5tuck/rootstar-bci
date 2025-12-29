//! FFT-based spectral analysis
//!
//! Provides band power extraction and spectral features.

use rustfft::{num_complex::Complex, FftPlanner};

use rootstar_bci_core::types::EegBand;

/// FFT-based spectral analyzer
pub struct SpectralAnalyzer {
    fft_size: usize,
    sample_rate: f64,
    planner: FftPlanner<f64>,
    window: Vec<f64>,
    buffer: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer
    ///
    /// # Arguments
    ///
    /// * `fft_size` - FFT size (should be power of 2)
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(fft_size: usize, sample_rate: f64) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        Self {
            fft_size,
            sample_rate,
            planner,
            window: hann_window(fft_size),
            buffer: vec![Complex::new(0.0, 0.0); fft_size],
            scratch: vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()],
        }
    }

    /// Frequency resolution (Hz per bin)
    #[must_use]
    pub fn frequency_resolution(&self) -> f64 {
        self.sample_rate / self.fft_size as f64
    }

    /// Compute power spectrum from time-domain samples
    ///
    /// Returns power spectral density (magnitude squared)
    pub fn compute_psd(&mut self, samples: &[f64]) -> Vec<f64> {
        assert!(samples.len() >= self.fft_size, "Not enough samples for FFT");

        // Apply window and copy to buffer
        for (i, (&s, &w)) in samples.iter().zip(self.window.iter()).enumerate() {
            self.buffer[i] = Complex::new(s * w, 0.0);
        }

        // Perform FFT
        let fft = self.planner.plan_fft_forward(self.fft_size);
        fft.process_with_scratch(&mut self.buffer, &mut self.scratch);

        // Compute power (magnitude squared), only positive frequencies
        let n_freqs = self.fft_size / 2 + 1;
        let norm = 1.0 / (self.fft_size as f64).powi(2);

        self.buffer[..n_freqs]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) * norm)
            .collect()
    }

    /// Extract band power for a frequency range
    #[must_use]
    pub fn band_power(&self, psd: &[f64], low_hz: f64, high_hz: f64) -> f64 {
        let freq_res = self.frequency_resolution();
        let start_bin = (low_hz / freq_res).floor() as usize;
        let end_bin = (high_hz / freq_res).ceil() as usize;

        let end_bin = end_bin.min(psd.len() - 1);

        psd[start_bin..=end_bin].iter().sum()
    }

    /// Extract power for a standard EEG band
    #[must_use]
    pub fn eeg_band_power(&self, psd: &[f64], band: EegBand) -> f64 {
        let (low, high) = band.range_hz();
        self.band_power(psd, f64::from(low), f64::from(high))
    }

    /// Extract all standard EEG band powers
    #[must_use]
    pub fn all_band_powers(&self, psd: &[f64]) -> BandPowers {
        BandPowers {
            delta: self.eeg_band_power(psd, EegBand::Delta),
            theta: self.eeg_band_power(psd, EegBand::Theta),
            alpha: self.eeg_band_power(psd, EegBand::Alpha),
            beta: self.eeg_band_power(psd, EegBand::Beta),
            gamma: self.eeg_band_power(psd, EegBand::Gamma),
        }
    }

    /// Compute relative band powers (normalized to total power)
    #[must_use]
    pub fn relative_band_powers(&self, psd: &[f64]) -> BandPowers {
        let powers = self.all_band_powers(psd);
        let total = powers.total();

        if total > 0.0 {
            BandPowers {
                delta: powers.delta / total,
                theta: powers.theta / total,
                alpha: powers.alpha / total,
                beta: powers.beta / total,
                gamma: powers.gamma / total,
            }
        } else {
            BandPowers::default()
        }
    }
}

/// EEG band powers container
#[derive(Clone, Copy, Debug, Default)]
pub struct BandPowers {
    /// Delta band power (0.5-4 Hz)
    pub delta: f64,
    /// Theta band power (4-8 Hz)
    pub theta: f64,
    /// Alpha band power (8-13 Hz)
    pub alpha: f64,
    /// Beta band power (13-30 Hz)
    pub beta: f64,
    /// Gamma band power (30-100 Hz)
    pub gamma: f64,
}

impl BandPowers {
    /// Total power across all bands
    #[must_use]
    pub fn total(&self) -> f64 {
        self.delta + self.theta + self.alpha + self.beta + self.gamma
    }

    /// Alpha/theta ratio (attention/relaxation metric)
    #[must_use]
    pub fn alpha_theta_ratio(&self) -> f64 {
        if self.theta > 0.0 {
            self.alpha / self.theta
        } else {
            0.0
        }
    }

    /// Beta/alpha ratio (engagement metric)
    #[must_use]
    pub fn beta_alpha_ratio(&self) -> f64 {
        if self.alpha > 0.0 {
            self.beta / self.alpha
        } else {
            0.0
        }
    }
}

/// Generate Hann window coefficients
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analyzer() {
        let mut analyzer = SpectralAnalyzer::new(256, 250.0);

        // Generate 10 Hz sine wave
        let samples: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 250.0).sin())
            .collect();

        let psd = analyzer.compute_psd(&samples);

        // Peak should be near 10 Hz
        let alpha_power = analyzer.eeg_band_power(&psd, EegBand::Alpha);
        let total_power: f64 = psd.iter().sum();

        // Alpha should contain most of the power for a 10 Hz signal
        assert!(alpha_power > total_power * 0.5);
    }
}

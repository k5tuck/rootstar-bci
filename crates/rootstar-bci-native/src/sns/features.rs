//! SNS-Specific Feature Extraction
//!
//! Extract features from EEG/fNIRS for sensory decoding.
//! Implements FFT-based spectral analysis for band power extraction.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::{Deserialize, Serialize};

use rootstar_bci_core::types::{EegChannel, EegSample, FnirsChannel, HemodynamicSample};

use super::cortical_map::{AepComponents, GepComponents, SepComponents};
use super::error::{DecoderError, DecoderResult};

// ============================================================================
// Spectral Analyzer (FFT-based)
// ============================================================================

/// FFT-based spectral analyzer for EEG band power extraction.
///
/// Uses Welch's method with overlapping windows for robust power spectral
/// density estimation.
#[derive(Clone)]
pub struct SpectralAnalyzer {
    /// Sample rate in Hz
    sample_rate_hz: f64,
    /// FFT size (must be power of 2)
    fft_size: usize,
    /// Forward FFT instance
    fft: Arc<dyn Fft<f64>>,
    /// Hanning window coefficients
    window: Vec<f64>,
    /// Window normalization factor
    window_power: f64,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer.
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sampling rate in Hz
    /// * `fft_size` - FFT size (will be rounded up to power of 2)
    #[must_use]
    pub fn new(sample_rate_hz: f64, fft_size: usize) -> Self {
        // Round up to power of 2
        let fft_size = fft_size.next_power_of_two();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // Create Hanning window
        let window: Vec<f64> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (fft_size - 1) as f64).cos()))
            .collect();

        // Calculate window power for normalization
        let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>() / fft_size as f64;

        Self {
            sample_rate_hz,
            fft_size,
            fft,
            window,
            window_power,
        }
    }

    /// Compute power spectral density using FFT.
    ///
    /// Returns power values for each frequency bin from 0 to Nyquist.
    pub fn compute_psd(&self, signal: &[f64]) -> Vec<f64> {
        if signal.len() < self.fft_size {
            // Zero-pad if signal is shorter than FFT size
            let mut padded = vec![0.0; self.fft_size];
            padded[..signal.len()].copy_from_slice(signal);
            return self.compute_psd_internal(&padded);
        }

        // Use Welch's method with 50% overlapping windows
        let hop_size = self.fft_size / 2;
        let n_windows = (signal.len() - self.fft_size) / hop_size + 1;

        if n_windows == 0 {
            return self.compute_psd_internal(&signal[..self.fft_size]);
        }

        // Average PSD across windows
        let mut avg_psd = vec![0.0; self.fft_size / 2 + 1];
        for w in 0..n_windows {
            let start = w * hop_size;
            let window_psd = self.compute_psd_internal(&signal[start..start + self.fft_size]);
            for (i, p) in window_psd.iter().enumerate() {
                avg_psd[i] += p;
            }
        }

        for p in &mut avg_psd {
            *p /= n_windows as f64;
        }

        avg_psd
    }

    fn compute_psd_internal(&self, segment: &[f64]) -> Vec<f64> {
        // Apply window and convert to complex
        let mut buffer: Vec<Complex<f64>> = segment
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Perform FFT
        self.fft.process(&mut buffer);

        // Compute one-sided power spectral density
        let n = self.fft_size;
        let scale = 2.0 / (self.sample_rate_hz * self.window_power * n as f64);

        let mut psd = Vec::with_capacity(n / 2 + 1);

        // DC component (no doubling)
        psd.push(buffer[0].norm_sqr() * scale / 2.0);

        // Positive frequencies (doubled for one-sided PSD)
        for i in 1..n / 2 {
            psd.push(buffer[i].norm_sqr() * scale);
        }

        // Nyquist component (no doubling)
        psd.push(buffer[n / 2].norm_sqr() * scale / 2.0);

        psd
    }

    /// Get frequency bin for a given frequency in Hz.
    #[must_use]
    pub fn freq_to_bin(&self, freq_hz: f64) -> usize {
        let bin = (freq_hz * self.fft_size as f64 / self.sample_rate_hz).round() as usize;
        bin.min(self.fft_size / 2)
    }

    /// Get frequency in Hz for a given bin index.
    #[must_use]
    pub fn bin_to_freq(&self, bin: usize) -> f64 {
        bin as f64 * self.sample_rate_hz / self.fft_size as f64
    }

    /// Calculate band power between two frequencies (inclusive).
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `low_hz` - Lower frequency bound
    /// * `high_hz` - Upper frequency bound
    ///
    /// # Returns
    /// Integrated power in the specified band (µV²/Hz)
    pub fn band_power(&self, signal: &[f64], low_hz: f64, high_hz: f64) -> f64 {
        let psd = self.compute_psd(signal);

        let low_bin = self.freq_to_bin(low_hz);
        let high_bin = self.freq_to_bin(high_hz).min(psd.len() - 1);

        if low_bin > high_bin || low_bin >= psd.len() {
            return 0.0;
        }

        // Integrate power in band (trapezoidal rule)
        let freq_resolution = self.sample_rate_hz / self.fft_size as f64;
        let mut power = 0.0;

        for i in low_bin..=high_bin {
            power += psd[i] * freq_resolution;
        }

        power
    }

    /// Calculate power at a specific frequency (for ASSR analysis).
    ///
    /// Uses a narrow band around the target frequency.
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `freq_hz` - Target frequency
    ///
    /// # Returns
    /// Power at the target frequency and phase in radians
    pub fn power_at_frequency(&self, signal: &[f64], freq_hz: f64) -> (f64, f64) {
        if signal.len() < self.fft_size {
            let mut padded = vec![0.0; self.fft_size];
            padded[..signal.len()].copy_from_slice(signal);
            return self.power_at_frequency_internal(&padded, freq_hz);
        }

        self.power_at_frequency_internal(&signal[..self.fft_size], freq_hz)
    }

    fn power_at_frequency_internal(&self, segment: &[f64], freq_hz: f64) -> (f64, f64) {
        // Apply window and convert to complex
        let mut buffer: Vec<Complex<f64>> = segment
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Perform FFT
        self.fft.process(&mut buffer);

        // Find the target bin
        let bin = self.freq_to_bin(freq_hz);
        if bin >= buffer.len() {
            return (0.0, 0.0);
        }

        let complex_val = buffer[bin];
        let power = complex_val.norm_sqr();
        let phase = complex_val.arg();

        (power, phase)
    }

    /// Calculate relative band power (percentage of total power).
    pub fn relative_band_power(&self, signal: &[f64], low_hz: f64, high_hz: f64) -> f64 {
        let psd = self.compute_psd(signal);

        let low_bin = self.freq_to_bin(low_hz);
        let high_bin = self.freq_to_bin(high_hz).min(psd.len() - 1);

        let band_power: f64 = psd[low_bin..=high_bin].iter().sum();
        let total_power: f64 = psd.iter().sum();

        if total_power > 1e-10 {
            band_power / total_power
        } else {
            0.0
        }
    }

    /// Extract all canonical EEG band powers.
    ///
    /// Returns (delta, theta, alpha, beta, gamma) powers.
    pub fn canonical_bands(&self, signal: &[f64]) -> [f64; 5] {
        [
            self.band_power(signal, 0.5, 4.0),   // Delta
            self.band_power(signal, 4.0, 8.0),   // Theta
            self.band_power(signal, 8.0, 13.0),  // Alpha
            self.band_power(signal, 13.0, 30.0), // Beta
            self.band_power(signal, 30.0, 100.0), // Gamma
        ]
    }
}

impl std::fmt::Debug for SpectralAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralAnalyzer")
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("fft_size", &self.fft_size)
            .finish()
    }
}

/// Feature extraction configuration
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    /// Sample rate in Hz
    pub sample_rate_hz: f64,
    /// Window size for analysis in samples
    pub window_size: usize,
    /// Overlap between windows (0-1)
    pub overlap: f64,
    /// Whether to apply baseline correction
    pub baseline_correction: bool,
    /// Baseline window in milliseconds
    pub baseline_window_ms: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 250.0,
            window_size: 256,
            overlap: 0.5,
            baseline_correction: true,
            baseline_window_ms: 100.0,
        }
    }
}

/// SEP Feature Extractor
#[derive(Clone)]
pub struct SepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// Spectral analyzer for FFT-based band power
    spectral: SpectralAnalyzer,
    /// Expected N20 latency window (min, max) in ms
    n20_window_ms: (f64, f64),
    /// Expected P25 latency window
    p25_window_ms: (f64, f64),
    /// Expected N30 latency window
    n30_window_ms: (f64, f64),
}

impl std::fmt::Debug for SepFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SepFeatureExtractor")
            .field("config", &self.config)
            .field("n20_window_ms", &self.n20_window_ms)
            .field("p25_window_ms", &self.p25_window_ms)
            .field("n30_window_ms", &self.n30_window_ms)
            .finish()
    }
}

impl SepFeatureExtractor {
    /// Create a new SEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        let spectral = SpectralAnalyzer::new(config.sample_rate_hz, config.window_size);
        Self {
            config,
            spectral,
            n20_window_ms: (18.0, 25.0),
            p25_window_ms: (23.0, 30.0),
            n30_window_ms: (28.0, 35.0),
        }
    }

    /// Extract SEP components from EEG window
    pub fn extract(
        &self,
        eeg_window: &[EegSample],
        stimulus_time_us: u64,
        channel: EegChannel,
    ) -> DecoderResult<SepComponents> {
        if eeg_window.len() < 10 {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: 10,
            });
        }

        // Convert to f64 array for processing
        let values: Vec<f64> = eeg_window
            .iter()
            .map(|s| s.channel(channel).to_f32() as f64)
            .collect();

        let times_ms: Vec<f64> = eeg_window
            .iter()
            .map(|s| (s.timestamp_us - stimulus_time_us) as f64 / 1000.0)
            .collect();

        // Apply baseline correction
        let values = if self.config.baseline_correction {
            self.baseline_correct(&values, &times_ms)
        } else {
            values
        };

        // Find N20 (negative peak around 20ms)
        let (n20_latency, n20_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.n20_window_ms.0,
            self.n20_window_ms.1,
            true, // negative
        );

        // Find P25 (positive peak around 25ms)
        let (p25_latency, p25_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.p25_window_ms.0,
            self.p25_window_ms.1,
            false, // positive
        );

        // Find N30 (negative peak around 30ms)
        let (n30_latency, n30_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.n30_window_ms.0,
            self.n30_window_ms.1,
            true, // negative
        );

        // Calculate mu and beta band powers using FFT
        let mu_suppression = self.spectral.band_power(&values, 8.0, 12.0);
        let beta_rebound = self.spectral.band_power(&values, 15.0, 25.0);

        Ok(SepComponents {
            n20_latency_ms: n20_latency,
            n20_amplitude_uv: n20_amplitude,
            p25_latency_ms: p25_latency,
            p25_amplitude_uv: p25_amplitude,
            n30_latency_ms: n30_latency,
            n30_amplitude_uv: n30_amplitude,
            mu_suppression,
            beta_rebound,
        })
    }

    fn baseline_correct(&self, values: &[f64], times_ms: &[f64]) -> Vec<f64> {
        // Find baseline samples (before stimulus)
        let baseline_samples: Vec<f64> = values
            .iter()
            .zip(times_ms.iter())
            .filter(|&(_, t)| *t < 0.0 && *t > -self.config.baseline_window_ms)
            .map(|(&v, _)| v)
            .collect();

        let baseline = if baseline_samples.is_empty() {
            0.0
        } else {
            baseline_samples.iter().sum::<f64>() / baseline_samples.len() as f64
        };

        values.iter().map(|&v| v - baseline).collect()
    }

    fn find_peak(
        &self,
        values: &[f64],
        times_ms: &[f64],
        start_ms: f64,
        end_ms: f64,
        negative: bool,
    ) -> (f64, f64) {
        let mut best_idx = 0;
        let mut best_val = if negative { f64::MAX } else { f64::MIN };

        for (i, (&v, &t)) in values.iter().zip(times_ms.iter()).enumerate() {
            if t >= start_ms && t <= end_ms {
                let is_better = if negative { v < best_val } else { v > best_val };
                if is_better {
                    best_val = v;
                    best_idx = i;
                }
            }
        }

        let latency = times_ms.get(best_idx).copied().unwrap_or(0.0);
        (latency, best_val)
    }
}

impl Default for SepFeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// AEP Feature Extractor
#[derive(Clone)]
pub struct AepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// Spectral analyzer for FFT-based band power
    spectral: SpectralAnalyzer,
    /// N1 window (ms)
    n1_window_ms: (f64, f64),
    /// P2 window (ms)
    p2_window_ms: (f64, f64),
    /// N2 window (ms)
    n2_window_ms: (f64, f64),
}

impl std::fmt::Debug for AepFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AepFeatureExtractor")
            .field("config", &self.config)
            .field("n1_window_ms", &self.n1_window_ms)
            .field("p2_window_ms", &self.p2_window_ms)
            .field("n2_window_ms", &self.n2_window_ms)
            .finish()
    }
}

impl AepFeatureExtractor {
    /// Create a new AEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        let spectral = SpectralAnalyzer::new(config.sample_rate_hz, config.window_size);
        Self {
            config,
            spectral,
            n1_window_ms: (80.0, 120.0),
            p2_window_ms: (150.0, 250.0),
            n2_window_ms: (200.0, 350.0),
        }
    }

    /// Extract AEP components from EEG window
    pub fn extract(
        &self,
        eeg_window: &[EegSample],
        stimulus_time_us: u64,
        channel: EegChannel,
    ) -> DecoderResult<AepComponents> {
        if eeg_window.len() < 50 {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: 50,
            });
        }

        let values: Vec<f64> = eeg_window
            .iter()
            .map(|s| s.channel(channel).to_f32() as f64)
            .collect();

        let times_ms: Vec<f64> = eeg_window
            .iter()
            .map(|s| (s.timestamp_us - stimulus_time_us) as f64 / 1000.0)
            .collect();

        // Find N1 (negative around 100ms)
        let (n1_latency, n1_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.n1_window_ms.0,
            self.n1_window_ms.1,
            true,
        );

        // Find P2 (positive around 200ms)
        let (p2_latency, p2_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.p2_window_ms.0,
            self.p2_window_ms.1,
            false,
        );

        // Find N2 (negative around 250ms)
        let (n2_latency, n2_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.n2_window_ms.0,
            self.n2_window_ms.1,
            true,
        );

        // ASSR and gamma power using proper FFT analysis
        let (assr_power, assr_phase) = self.spectral.power_at_frequency(&values, 40.0);
        let gamma_power = self.spectral.band_power(&values, 30.0, 100.0);

        Ok(AepComponents {
            n1_latency_ms: n1_latency,
            n1_amplitude_uv: n1_amplitude,
            p2_latency_ms: p2_latency,
            p2_amplitude_uv: p2_amplitude,
            n2_latency_ms: n2_latency,
            n2_amplitude_uv: n2_amplitude,
            assr_power,
            assr_phase,
            gamma_power,
        })
    }

    fn find_peak(
        &self,
        values: &[f64],
        times_ms: &[f64],
        start_ms: f64,
        end_ms: f64,
        negative: bool,
    ) -> (f64, f64) {
        let mut best_idx = 0;
        let mut best_val = if negative { f64::MAX } else { f64::MIN };

        for (i, (&v, &t)) in values.iter().zip(times_ms.iter()).enumerate() {
            if t >= start_ms && t <= end_ms {
                let is_better = if negative { v < best_val } else { v > best_val };
                if is_better {
                    best_val = v;
                    best_idx = i;
                }
            }
        }

        let latency = times_ms.get(best_idx).copied().unwrap_or(0.0);
        (latency, best_val)
    }
}

impl Default for AepFeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// GEP Feature Extractor
#[derive(Clone)]
pub struct GepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// Spectral analyzer for FFT-based band power
    spectral: SpectralAnalyzer,
    /// P1 window (ms)
    p1_window_ms: (f64, f64),
    /// N1 window (ms)
    n1_window_ms: (f64, f64),
}

impl std::fmt::Debug for GepFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GepFeatureExtractor")
            .field("config", &self.config)
            .field("p1_window_ms", &self.p1_window_ms)
            .field("n1_window_ms", &self.n1_window_ms)
            .finish()
    }
}

impl GepFeatureExtractor {
    /// Create a new GEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        let spectral = SpectralAnalyzer::new(config.sample_rate_hz, config.window_size);
        Self {
            config,
            spectral,
            p1_window_ms: (100.0, 160.0),
            n1_window_ms: (150.0, 220.0),
        }
    }

    /// Extract GEP components from EEG window
    pub fn extract(
        &self,
        eeg_window: &[EegSample],
        stimulus_time_us: u64,
        channels: &[EegChannel],
    ) -> DecoderResult<GepComponents> {
        if eeg_window.len() < 50 {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: 50,
            });
        }

        // Average across frontal channels
        let values: Vec<f64> = eeg_window
            .iter()
            .map(|s| {
                channels
                    .iter()
                    .map(|&ch| s.channel(ch).to_f32() as f64)
                    .sum::<f64>()
                    / channels.len().max(1) as f64
            })
            .collect();

        let times_ms: Vec<f64> = eeg_window
            .iter()
            .map(|s| (s.timestamp_us - stimulus_time_us) as f64 / 1000.0)
            .collect();

        // Find P1 (positive around 130ms)
        let (p1_latency, p1_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.p1_window_ms.0,
            self.p1_window_ms.1,
            false,
        );

        // Find N1 (negative around 180ms)
        let (n1_latency, n1_amplitude) = self.find_peak(
            &values,
            &times_ms,
            self.n1_window_ms.0,
            self.n1_window_ms.1,
            true,
        );

        // Frontal theta using FFT
        let frontal_theta = self.spectral.band_power(&values, 4.0, 8.0);

        Ok(GepComponents {
            p1_latency_ms: p1_latency,
            p1_amplitude_uv: p1_amplitude,
            n1_latency_ms: n1_latency,
            n1_amplitude_uv: n1_amplitude,
            frontal_theta,
        })
    }

    fn find_peak(
        &self,
        values: &[f64],
        times_ms: &[f64],
        start_ms: f64,
        end_ms: f64,
        negative: bool,
    ) -> (f64, f64) {
        let mut best_idx = 0;
        let mut best_val = if negative { f64::MAX } else { f64::MIN };

        for (i, (&v, &t)) in values.iter().zip(times_ms.iter()).enumerate() {
            if t >= start_ms && t <= end_ms {
                let is_better = if negative { v < best_val } else { v > best_val };
                if is_better {
                    best_val = v;
                    best_idx = i;
                }
            }
        }

        let latency = times_ms.get(best_idx).copied().unwrap_or(0.0);
        (latency, best_val)
    }
}

impl Default for GepFeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// fNIRS feature extraction for S1/A1/gustatory cortex
#[derive(Clone, Debug)]
pub struct FnirsFeatureExtractor {
    /// Hemodynamic delay in milliseconds
    pub hemodynamic_delay_ms: f64,
    /// Analysis window in milliseconds
    pub window_ms: f64,
}

impl FnirsFeatureExtractor {
    /// Create a new fNIRS feature extractor
    #[must_use]
    pub fn new() -> Self {
        Self {
            hemodynamic_delay_ms: 5000.0,
            window_ms: 10000.0,
        }
    }

    /// Extract activation from fNIRS samples
    pub fn extract_activation(
        &self,
        samples: &[HemodynamicSample],
        channel: FnirsChannel,
    ) -> DecoderResult<FnirsActivation> {
        let channel_samples: Vec<&HemodynamicSample> = samples
            .iter()
            .filter(|s| s.channel == channel)
            .collect();

        if channel_samples.is_empty() {
            return Err(DecoderError::InsufficientData { got: 0, need: 1 });
        }

        // Calculate mean activation
        let mean_hbo2: f64 = channel_samples
            .iter()
            .map(|s| s.delta_hbo2.to_f32() as f64)
            .sum::<f64>()
            / channel_samples.len() as f64;

        let mean_hbr: f64 = channel_samples
            .iter()
            .map(|s| s.delta_hbr.to_f32() as f64)
            .sum::<f64>()
            / channel_samples.len() as f64;

        let mean_hbt = mean_hbo2 + mean_hbr.abs();

        Ok(FnirsActivation {
            delta_hbo2: mean_hbo2,
            delta_hbr: mean_hbr,
            delta_hbt: mean_hbt,
            oxygenation_index: if mean_hbt > 0.001 {
                mean_hbo2 / mean_hbt
            } else {
                0.5
            },
        })
    }

    /// Extract asymmetry between left and right channels
    pub fn extract_asymmetry(
        &self,
        samples: &[HemodynamicSample],
        left_channel: FnirsChannel,
        right_channel: FnirsChannel,
    ) -> DecoderResult<f64> {
        let left = self.extract_activation(samples, left_channel)?;
        let right = self.extract_activation(samples, right_channel)?;

        let total = left.delta_hbo2.abs() + right.delta_hbo2.abs();
        if total > 0.001 {
            Ok((left.delta_hbo2 - right.delta_hbo2) / total)
        } else {
            Ok(0.0)
        }
    }
}

impl Default for FnirsFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// fNIRS activation features
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FnirsActivation {
    /// Change in oxygenated hemoglobin (µM)
    pub delta_hbo2: f64,
    /// Change in deoxygenated hemoglobin (µM)
    pub delta_hbr: f64,
    /// Change in total hemoglobin (µM)
    pub delta_hbt: f64,
    /// Oxygenation index (0-1)
    pub oxygenation_index: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::types::Fixed24_8;

    #[test]
    fn test_sep_extractor() {
        let extractor = SepFeatureExtractor::default();

        // Create mock EEG samples
        let samples: Vec<EegSample> = (0..100)
            .map(|i| {
                let mut sample = EegSample::new(i as u64 * 1000, i as u32);
                // Simulate N20 peak at ~20ms
                let t_ms = i as f64;
                let value = if (t_ms - 20.0).abs() < 3.0 { -5.0 } else { 0.0 };
                sample.set_channel(EegChannel::C3, Fixed24_8::from_f32(value as f32));
                sample
            })
            .collect();

        let result = extractor.extract(&samples, 0, EegChannel::C3);
        assert!(result.is_ok());

        let components = result.unwrap();
        assert!(components.n20_amplitude_uv < 0.0); // Should be negative
    }

    #[test]
    fn test_fnirs_extractor() {
        let extractor = FnirsFeatureExtractor::new();

        let channel = FnirsChannel::new(0, 0, 30);
        let samples: Vec<HemodynamicSample> = (0..10)
            .map(|i| {
                HemodynamicSample::new(
                    i as u64 * 100_000,
                    channel,
                    Fixed24_8::from_f32(1.0),
                    Fixed24_8::from_f32(-0.5),
                )
            })
            .collect();

        let result = extractor.extract_activation(&samples, channel);
        assert!(result.is_ok());

        let activation = result.unwrap();
        assert!(activation.delta_hbo2 > 0.0);
        assert!(activation.delta_hbr < 0.0);
    }

    #[test]
    fn test_spectral_analyzer_sinusoid() {
        // Create a 10 Hz sine wave at 250 Hz sample rate
        let sample_rate = 250.0;
        let frequency = 10.0;
        let n_samples = 512;

        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * frequency * t).sin()
            })
            .collect();

        let analyzer = SpectralAnalyzer::new(sample_rate, 256);
        let psd = analyzer.compute_psd(&signal);

        // Find the bin with maximum power
        let max_bin = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_freq = analyzer.bin_to_freq(max_bin);

        // Peak should be near 10 Hz
        assert!(
            (peak_freq - frequency).abs() < 2.0,
            "Expected peak near {frequency} Hz, got {peak_freq} Hz"
        );
    }

    #[test]
    fn test_spectral_analyzer_band_power() {
        let sample_rate = 250.0;
        let analyzer = SpectralAnalyzer::new(sample_rate, 256);

        // Create signal with alpha (10 Hz) component
        let n_samples = 512;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // 10 Hz (alpha) + small amount of noise
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        // Alpha power should be higher than theta and beta
        let theta_power = analyzer.band_power(&signal, 4.0, 8.0);
        let alpha_power = analyzer.band_power(&signal, 8.0, 13.0);
        let beta_power = analyzer.band_power(&signal, 13.0, 30.0);

        assert!(
            alpha_power > theta_power,
            "Alpha should dominate: alpha={alpha_power}, theta={theta_power}"
        );
        assert!(
            alpha_power > beta_power,
            "Alpha should dominate: alpha={alpha_power}, beta={beta_power}"
        );
    }

    #[test]
    fn test_spectral_analyzer_canonical_bands() {
        let sample_rate = 500.0;
        let analyzer = SpectralAnalyzer::new(sample_rate, 512);

        // Create signal with 30 Hz gamma component
        let n_samples = 1024;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * 35.0 * t).sin() // 35 Hz gamma
            })
            .collect();

        let bands = analyzer.canonical_bands(&signal);
        let [delta, theta, alpha, beta, gamma] = bands;

        // Gamma should be the dominant band
        assert!(gamma > delta, "Gamma should dominate delta");
        assert!(gamma > theta, "Gamma should dominate theta");
        assert!(gamma > alpha, "Gamma should dominate alpha");
        assert!(gamma > beta, "Gamma should dominate beta");
    }

    #[test]
    fn test_spectral_analyzer_power_at_frequency() {
        let sample_rate = 250.0;
        let target_freq = 40.0;
        let n_samples = 256;

        // Create 40 Hz ASSR stimulus
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * target_freq * t).sin()
            })
            .collect();

        let analyzer = SpectralAnalyzer::new(sample_rate, 256);
        let (power, phase) = analyzer.power_at_frequency(&signal, target_freq);

        // Power should be significant
        assert!(power > 0.0, "Power at 40 Hz should be positive");

        // Phase should be in valid range
        assert!(
            phase >= -std::f64::consts::PI && phase <= std::f64::consts::PI,
            "Phase should be in [-π, π]"
        );
    }

    #[test]
    fn test_spectral_analyzer_relative_band_power() {
        let sample_rate = 250.0;
        let analyzer = SpectralAnalyzer::new(sample_rate, 256);

        // Create pure 10 Hz signal
        let n_samples = 512;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        let relative_alpha = analyzer.relative_band_power(&signal, 8.0, 13.0);

        // Most power should be in alpha band for a 10 Hz signal
        assert!(
            relative_alpha > 0.5,
            "Relative alpha power should be >50%, got {relative_alpha}"
        );
    }
}

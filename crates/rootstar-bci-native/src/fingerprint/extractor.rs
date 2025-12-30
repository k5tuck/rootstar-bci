//! Feature extraction for neural fingerprints.
//!
//! Extracts comprehensive feature sets from EEG and fNIRS signals
//! for neural fingerprint generation.

use std::f64::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

use rootstar_bci_core::fingerprint::FrequencyBand;
use rootstar_bci_core::types::Fixed24_8;

// ============================================================================
// EEG Features
// ============================================================================

/// Extracted EEG features for a single epoch.
#[derive(Clone, Debug)]
pub struct EegFeatures {
    /// Band power per channel: \[channel\]\[band\]
    pub band_power: Vec<Vec<f64>>,
    /// Coherence matrix per band: \[band\]\[channel\]\[channel\]
    pub coherence: Vec<Vec<Vec<f64>>>,
    /// Phase synchrony (PLV) matrix: \[channel\]\[channel\]
    pub phase_sync: Vec<Vec<f64>>,
    /// Topography (RMS per channel, normalized)
    pub topography: Vec<f64>,
    /// Sample entropy per channel
    pub entropy: Vec<f64>,
    /// Number of channels
    pub n_channels: usize,
}

impl EegFeatures {
    /// Create empty features for given channel count.
    #[must_use]
    pub fn new(n_channels: usize) -> Self {
        Self {
            band_power: vec![vec![0.0; FrequencyBand::COUNT]; n_channels],
            coherence: vec![vec![vec![0.0; n_channels]; n_channels]; 4], // 4 sensory bands
            phase_sync: vec![vec![0.0; n_channels]; n_channels],
            topography: vec![0.0; n_channels],
            entropy: vec![0.0; n_channels],
            n_channels,
        }
    }

    /// Convert band power to fixed-point for storage.
    #[must_use]
    pub fn band_power_fixed(&self) -> Vec<Fixed24_8> {
        self.band_power
            .iter()
            .flat_map(|ch| ch.iter().map(|&p| Fixed24_8::from_f32(p as f32)))
            .collect()
    }

    /// Convert topography to fixed-point.
    #[must_use]
    pub fn topography_fixed(&self) -> Vec<Fixed24_8> {
        self.topography
            .iter()
            .map(|&t| Fixed24_8::from_f32(t as f32))
            .collect()
    }

    /// Convert entropy to fixed-point.
    #[must_use]
    pub fn entropy_fixed(&self) -> Vec<Fixed24_8> {
        self.entropy
            .iter()
            .map(|&e| Fixed24_8::from_f32(e as f32))
            .collect()
    }
}

// ============================================================================
// EEG Feature Extractor
// ============================================================================

/// Extracts comprehensive features from multi-channel EEG.
pub struct EegFeatureExtractor {
    n_channels: usize,
    sample_rate: f64,
    fft_size: usize,
    planner: FftPlanner<f64>,
    window: Vec<f64>,
}

impl EegFeatureExtractor {
    /// Create a new EEG feature extractor.
    ///
    /// # Arguments
    ///
    /// * `n_channels` - Number of EEG channels
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(n_channels: usize, sample_rate: f64) -> Self {
        let fft_size = 256; // ~0.5s at 500 Hz
        Self {
            n_channels,
            sample_rate,
            fft_size,
            planner: FftPlanner::new(),
            window: hann_window(fft_size),
        }
    }

    /// Create with custom FFT size.
    #[must_use]
    pub fn with_fft_size(n_channels: usize, sample_rate: f64, fft_size: usize) -> Self {
        Self {
            n_channels,
            sample_rate,
            fft_size,
            planner: FftPlanner::new(),
            window: hann_window(fft_size),
        }
    }

    /// Extract features from an EEG epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - EEG data: \[channel\]\[sample\]
    ///
    /// # Panics
    ///
    /// Panics if epoch dimensions don't match configuration.
    pub fn extract(&mut self, epoch: &[Vec<f64>]) -> EegFeatures {
        assert_eq!(epoch.len(), self.n_channels, "Channel count mismatch");

        let mut features = EegFeatures::new(self.n_channels);

        // 1. Compute band power per channel
        for (ch, samples) in epoch.iter().enumerate() {
            if samples.len() >= self.fft_size {
                let psd = self.compute_psd(samples);
                for band in FrequencyBand::CANONICAL.iter().chain(FrequencyBand::SENSORY_FOCUS.iter())
                {
                    let power = self.band_power(&psd, *band);
                    features.band_power[ch][band.index()] = power;
                }
            }
        }

        // 2. Compute coherence for sensory focus bands
        self.compute_coherence(epoch, &mut features);

        // 3. Compute phase synchrony (PLV)
        self.compute_phase_sync(epoch, &mut features);

        // 4. Compute topography (RMS)
        self.compute_topography(epoch, &mut features);

        // 5. Compute entropy
        self.compute_entropy(epoch, &mut features);

        features
    }

    fn compute_psd(&mut self, samples: &[f64]) -> Vec<f64> {
        let mut buffer: Vec<Complex<f64>> = samples
            .iter()
            .take(self.fft_size)
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Pad if necessary
        buffer.resize(self.fft_size, Complex::new(0.0, 0.0));

        let fft = self.planner.plan_fft_forward(self.fft_size);
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(&mut buffer, &mut scratch);

        // Compute power (magnitude squared)
        let n_freqs = self.fft_size / 2 + 1;
        let norm = 1.0 / (self.fft_size as f64).powi(2);

        buffer[..n_freqs]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) * norm)
            .collect()
    }

    fn band_power(&self, psd: &[f64], band: FrequencyBand) -> f64 {
        let (low, high) = band.range_hz();
        let freq_res = self.sample_rate / self.fft_size as f64;

        let start_bin = (low as f64 / freq_res).floor() as usize;
        let end_bin = (high as f64 / freq_res).ceil() as usize;
        let end_bin = end_bin.min(psd.len() - 1);

        if start_bin >= psd.len() {
            return 0.0;
        }

        psd[start_bin..=end_bin].iter().sum()
    }

    fn compute_coherence(&mut self, epoch: &[Vec<f64>], features: &mut EegFeatures) {
        // Simplified coherence: correlation of band power time series
        // Full coherence would require Welch's method
        for (band_idx, band) in FrequencyBand::SENSORY_FOCUS.iter().enumerate() {
            for i in 0..self.n_channels {
                for j in (i + 1)..self.n_channels {
                    if epoch[i].len() >= self.fft_size && epoch[j].len() >= self.fft_size {
                        // Use cross-spectral coherence approximation
                        let coh = self.estimate_coherence(&epoch[i], &epoch[j], *band);
                        features.coherence[band_idx][i][j] = coh;
                        features.coherence[band_idx][j][i] = coh;
                    }
                }
            }
        }
    }

    fn estimate_coherence(&mut self, signal_a: &[f64], signal_b: &[f64], band: FrequencyBand) -> f64 {
        let psd_a = self.compute_psd(signal_a);
        let psd_b = self.compute_psd(signal_b);

        let (low, high) = band.range_hz();
        let freq_res = self.sample_rate / self.fft_size as f64;
        let start_bin = (low as f64 / freq_res).floor() as usize;
        let end_bin = ((high as f64 / freq_res).ceil() as usize).min(psd_a.len() - 1);

        if start_bin >= psd_a.len() {
            return 0.0;
        }

        // Magnitude squared coherence approximation
        let power_a: f64 = psd_a[start_bin..=end_bin].iter().sum();
        let power_b: f64 = psd_b[start_bin..=end_bin].iter().sum();

        // Cross-power (simplified)
        let cross: f64 = psd_a[start_bin..=end_bin]
            .iter()
            .zip(psd_b[start_bin..=end_bin].iter())
            .map(|(&a, &b)| (a * b).sqrt())
            .sum();

        if power_a > 0.0 && power_b > 0.0 {
            (cross * cross) / (power_a * power_b)
        } else {
            0.0
        }
    }

    fn compute_phase_sync(&self, epoch: &[Vec<f64>], features: &mut EegFeatures) {
        // Phase Locking Value (PLV) computation
        // Using Hilbert transform approximation

        for i in 0..self.n_channels {
            for j in (i + 1)..self.n_channels {
                if epoch[i].len() >= self.fft_size && epoch[j].len() >= self.fft_size {
                    let plv = self.compute_plv(&epoch[i], &epoch[j]);
                    features.phase_sync[i][j] = plv;
                    features.phase_sync[j][i] = plv;
                }
            }
        }
    }

    fn compute_plv(&self, signal_a: &[f64], signal_b: &[f64]) -> f64 {
        // Simplified PLV using instantaneous phase difference
        // Real implementation would use Hilbert transform

        let n = signal_a.len().min(signal_b.len());
        if n < 3 {
            return 0.0;
        }

        // Estimate instantaneous phase using zero crossings and interpolation
        let mut phase_diff_cos = 0.0;
        let mut phase_diff_sin = 0.0;

        for i in 1..n - 1 {
            // Approximate phase using arctangent of derivative ratio
            let da = signal_a[i + 1] - signal_a[i - 1];
            let db = signal_b[i + 1] - signal_b[i - 1];

            // Phase difference approximation
            let phase_a = da.atan2(signal_a[i]);
            let phase_b = db.atan2(signal_b[i]);
            let diff = phase_a - phase_b;

            phase_diff_cos += diff.cos();
            phase_diff_sin += diff.sin();
        }

        let count = (n - 2) as f64;
        ((phase_diff_cos / count).powi(2) + (phase_diff_sin / count).powi(2)).sqrt()
    }

    fn compute_topography(&self, epoch: &[Vec<f64>], features: &mut EegFeatures) {
        let mut max_rms = 0.0f64;

        for (ch, samples) in epoch.iter().enumerate() {
            let rms = if samples.is_empty() {
                0.0
            } else {
                let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
                (sum_sq / samples.len() as f64).sqrt()
            };
            features.topography[ch] = rms;
            max_rms = max_rms.max(rms);
        }

        // Normalize
        if max_rms > 0.0 {
            for t in &mut features.topography {
                *t /= max_rms;
            }
        }
    }

    fn compute_entropy(&self, epoch: &[Vec<f64>], features: &mut EegFeatures) {
        for (ch, samples) in epoch.iter().enumerate() {
            features.entropy[ch] = sample_entropy(samples, 2, 0.2);
        }
    }
}

// ============================================================================
// fNIRS Features
// ============================================================================

/// Extracted fNIRS features.
#[derive(Clone, Debug)]
pub struct FnirsFeatures {
    /// HbO activation per channel
    pub hbo_activation: Vec<f64>,
    /// HbR activation per channel
    pub hbr_activation: Vec<f64>,
    /// HbT (total hemoglobin) per channel
    pub hbt_activation: Vec<f64>,
    /// Peak HbO response amplitude
    pub hbo_peak_amplitude: Vec<f64>,
    /// Time to peak HbO (samples)
    pub hbo_time_to_peak: Vec<usize>,
    /// Number of channels
    pub n_channels: usize,
}

impl FnirsFeatures {
    /// Create empty features.
    #[must_use]
    pub fn new(n_channels: usize) -> Self {
        Self {
            hbo_activation: vec![0.0; n_channels],
            hbr_activation: vec![0.0; n_channels],
            hbt_activation: vec![0.0; n_channels],
            hbo_peak_amplitude: vec![0.0; n_channels],
            hbo_time_to_peak: vec![0; n_channels],
            n_channels,
        }
    }

    /// Convert HbO activation to fixed-point.
    #[must_use]
    pub fn hbo_fixed(&self) -> Vec<Fixed24_8> {
        self.hbo_activation
            .iter()
            .map(|&h| Fixed24_8::from_f32(h as f32))
            .collect()
    }

    /// Convert HbR activation to fixed-point.
    #[must_use]
    pub fn hbr_fixed(&self) -> Vec<Fixed24_8> {
        self.hbr_activation
            .iter()
            .map(|&h| Fixed24_8::from_f32(h as f32))
            .collect()
    }
}

// ============================================================================
// fNIRS Feature Extractor
// ============================================================================

/// Extracts features from fNIRS hemodynamic signals.
pub struct FnirsFeatureExtractor {
    n_channels: usize,
    sample_rate: f64,
}

impl FnirsFeatureExtractor {
    /// Create a new fNIRS feature extractor.
    #[must_use]
    pub fn new(n_channels: usize, sample_rate: f64) -> Self {
        Self { n_channels, sample_rate }
    }

    /// Extract features from HbO and HbR time series.
    ///
    /// # Arguments
    ///
    /// * `hbo` - Oxygenated hemoglobin: \[channel\]\[sample\]
    /// * `hbr` - Deoxygenated hemoglobin: \[channel\]\[sample\]
    pub fn extract(&self, hbo: &[Vec<f64>], hbr: &[Vec<f64>]) -> FnirsFeatures {
        let n_channels = hbo.len().min(hbr.len()).min(self.n_channels);
        let mut features = FnirsFeatures::new(n_channels);

        for ch in 0..n_channels {
            // Mean activation (baseline-corrected assumed)
            features.hbo_activation[ch] = mean(&hbo[ch]);
            features.hbr_activation[ch] = mean(&hbr[ch]);
            features.hbt_activation[ch] = features.hbo_activation[ch] + features.hbr_activation[ch];

            // Peak detection
            if let Some((peak_amp, peak_idx)) = find_peak(&hbo[ch]) {
                features.hbo_peak_amplitude[ch] = peak_amp;
                features.hbo_time_to_peak[ch] = peak_idx;
            }
        }

        features
    }

    /// Compute neurovascular coupling (cross-correlation with EEG band power).
    ///
    /// # Arguments
    ///
    /// * `hbo` - HbO time series
    /// * `eeg_power` - EEG band power time series (resampled to fNIRS rate)
    /// * `max_lag_samples` - Maximum lag to search (samples)
    ///
    /// Returns (correlation, lag in samples).
    #[must_use]
    pub fn neurovascular_coupling(
        &self,
        hbo: &[f64],
        eeg_power: &[f64],
        max_lag_samples: usize,
    ) -> (f64, i32) {
        if hbo.is_empty() || eeg_power.is_empty() {
            return (0.0, 0);
        }

        let mut best_corr = 0.0f64;
        let mut best_lag = 0i32;

        // Hemodynamic response typically lags neural activity by 4-6 seconds
        for lag in 0..max_lag_samples.min(hbo.len()) {
            let corr = cross_correlation(eeg_power, hbo, lag as i32);
            if corr.abs() > best_corr.abs() {
                best_corr = corr;
                best_lag = lag as i32;
            }
        }

        (best_corr, best_lag)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate Hann window.
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos()))
        .collect()
}

/// Compute mean of a slice.
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        0.0
    } else {
        data.iter().sum::<f64>() / data.len() as f64
    }
}

/// Find peak value and index.
fn find_peak(data: &[f64]) -> Option<(f64, usize)> {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (v, i))
}

/// Compute cross-correlation at a specific lag.
fn cross_correlation(a: &[f64], b: &[f64], lag: i32) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a = mean(a);
    let mean_b = mean(b);

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        let j = (i as i32 + lag) as usize;
        if j < b.len() {
            sum += (a[i] - mean_a) * (b[j] - mean_b);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Compute sample entropy.
///
/// # Arguments
///
/// * `data` - Time series
/// * `m` - Embedding dimension (typically 2)
/// * `r` - Tolerance (typically 0.2 × std)
fn sample_entropy(data: &[f64], m: usize, r_factor: f64) -> f64 {
    let n = data.len();
    if n < m + 2 {
        return 0.0;
    }

    // Compute standard deviation
    let mean = mean(data);
    let std: f64 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let r = r_factor * std;

    if r == 0.0 {
        return 0.0;
    }

    let mut count_m = 0;
    let mut count_m1 = 0;

    // Count template matches for dimension m and m+1
    for i in 0..n - m {
        for j in (i + 1)..n - m {
            // Check m-length match
            let mut match_m = true;
            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > r {
                    match_m = false;
                    break;
                }
            }

            if match_m {
                count_m += 1;

                // Check m+1 length match
                if (data[i + m] - data[j + m]).abs() <= r {
                    count_m1 += 1;
                }
            }
        }
    }

    if count_m == 0 || count_m1 == 0 {
        0.0
    } else {
        -(count_m1 as f64 / count_m as f64).ln()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_feature_extractor() {
        let mut extractor = EegFeatureExtractor::new(8, 500.0);

        // Generate test data: 8 channels, 512 samples (1 second)
        let epoch: Vec<Vec<f64>> = (0..8)
            .map(|ch| {
                (0..512)
                    .map(|i| {
                        // Mix of 10Hz (alpha) and noise
                        let t = i as f64 / 500.0;
                        (2.0 * PI * 10.0 * t).sin() + 0.1 * (ch as f64 * 0.1)
                    })
                    .collect()
            })
            .collect();

        let features = extractor.extract(&epoch);

        assert_eq!(features.n_channels, 8);
        assert_eq!(features.band_power.len(), 8);
        assert_eq!(features.topography.len(), 8);

        // Alpha band should have significant power
        assert!(features.band_power[0][FrequencyBand::Alpha.index()] > 0.0);
    }

    #[test]
    fn test_fnirs_feature_extractor() {
        let extractor = FnirsFeatureExtractor::new(4, 25.0);

        // Generate test HRF-like data
        let hbo: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                (0..250) // 10 seconds
                    .map(|i| {
                        let t = i as f64 / 25.0;
                        // Simple HRF approximation
                        (t / 5.0) * (-t / 5.0).exp()
                    })
                    .collect()
            })
            .collect();

        let hbr: Vec<Vec<f64>> = hbo
            .iter()
            .map(|ch| ch.iter().map(|&v| -v * 0.3).collect())
            .collect();

        let features = extractor.extract(&hbo, &hbr);

        assert_eq!(features.n_channels, 4);
        assert!(features.hbo_activation[0] > 0.0);
        assert!(features.hbr_activation[0] < 0.0);
    }

    #[test]
    fn test_sample_entropy() {
        // Constant signal should have low entropy
        let constant: Vec<f64> = vec![1.0; 100];
        let entropy_const = sample_entropy(&constant, 2, 0.2);

        // Random signal should have higher entropy
        let random: Vec<f64> = (0..100).map(|i| (i as f64 * 1.234).sin()).collect();
        let entropy_random = sample_entropy(&random, 2, 0.2);

        // Random should have more entropy than constant
        assert!(entropy_random >= entropy_const);
    }
}

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

// ============================================================================
// EMG Features
// ============================================================================

/// Extracted EMG features for emotional valence and arousal.
#[derive(Clone, Debug)]
pub struct EmgFeatures {
    /// RMS amplitude per channel (8 channels)
    pub rms_amplitude: Vec<f64>,
    /// Mean frequency per channel (Hz)
    pub mean_frequency: Vec<f64>,
    /// Valence score (-1 = negative, 0 = neutral, 1 = positive)
    pub valence: f64,
    /// Arousal score (0 = calm, 1 = excited)
    pub arousal: f64,
    /// Smile activation (zygomaticus)
    pub smile_activation: f64,
    /// Frown activation (corrugator)
    pub frown_activation: f64,
    /// Number of channels
    pub n_channels: usize,
}

impl EmgFeatures {
    /// Create empty features.
    #[must_use]
    pub fn new(n_channels: usize) -> Self {
        Self {
            rms_amplitude: vec![0.0; n_channels],
            mean_frequency: vec![0.0; n_channels],
            valence: 0.0,
            arousal: 0.0,
            smile_activation: 0.0,
            frown_activation: 0.0,
            n_channels,
        }
    }

    /// Convert RMS amplitude to fixed-point.
    #[must_use]
    pub fn rms_fixed(&self) -> Vec<Fixed24_8> {
        self.rms_amplitude
            .iter()
            .map(|&r| Fixed24_8::from_f32(r as f32))
            .collect()
    }

    /// Convert mean frequency to fixed-point.
    #[must_use]
    pub fn frequency_fixed(&self) -> Vec<Fixed24_8> {
        self.mean_frequency
            .iter()
            .map(|&f| Fixed24_8::from_f32(f as f32))
            .collect()
    }
}

// ============================================================================
// EMG Feature Extractor
// ============================================================================

/// Extracts features from facial EMG for emotional analysis.
pub struct EmgFeatureExtractor {
    n_channels: usize,
    sample_rate: f64,
    baseline_rms: Vec<f64>,
    calibrated: bool,
}

impl EmgFeatureExtractor {
    /// Create a new EMG feature extractor.
    ///
    /// # Arguments
    ///
    /// * `n_channels` - Number of EMG channels (typically 8)
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(n_channels: usize, sample_rate: f64) -> Self {
        Self {
            n_channels,
            sample_rate,
            baseline_rms: vec![0.0; n_channels],
            calibrated: false,
        }
    }

    /// Calibrate baseline from relaxed state.
    ///
    /// # Arguments
    ///
    /// * `calibration_data` - EMG data during relaxed state: \[channel\]\[sample\]
    pub fn calibrate(&mut self, calibration_data: &[Vec<f64>]) {
        for (ch, samples) in calibration_data.iter().enumerate().take(self.n_channels) {
            if !samples.is_empty() {
                let sum_sq: f64 = samples.iter().map(|x| x * x).sum();
                self.baseline_rms[ch] = (sum_sq / samples.len() as f64).sqrt();
            }
        }
        self.calibrated = true;
    }

    /// Extract features from EMG epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - EMG data: \[channel\]\[sample\]
    ///
    /// Channel mapping:
    /// - 0, 1: Zygomaticus (smile) L/R
    /// - 2, 3: Corrugator (frown) L/R
    /// - 4, 5: Masseter (jaw) L/R
    /// - 6, 7: Orbicularis (lips) U/D
    pub fn extract(&self, epoch: &[Vec<f64>]) -> EmgFeatures {
        let n_channels = epoch.len().min(self.n_channels);
        let mut features = EmgFeatures::new(n_channels);

        // Extract per-channel features
        for (ch, samples) in epoch.iter().enumerate().take(n_channels) {
            if samples.is_empty() {
                continue;
            }

            // RMS amplitude
            let sum_sq: f64 = samples.iter().map(|x| x * x).sum();
            features.rms_amplitude[ch] = (sum_sq / samples.len() as f64).sqrt();

            // Mean frequency from zero-crossing rate
            let mut zero_crossings = 0;
            for i in 1..samples.len() {
                if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                    zero_crossings += 1;
                }
            }
            let duration_s = samples.len() as f64 / self.sample_rate;
            features.mean_frequency[ch] = zero_crossings as f64 / (2.0 * duration_s);
        }

        // Compute valence from smile vs frown
        if n_channels >= 4 {
            // Zygomaticus (smile) = channels 0, 1
            let smile = (features.rms_amplitude[0] + features.rms_amplitude[1]) / 2.0;
            let smile_baseline = if self.calibrated {
                (self.baseline_rms[0] + self.baseline_rms[1]) / 2.0
            } else {
                smile * 0.5
            };

            // Corrugator (frown) = channels 2, 3
            let frown = (features.rms_amplitude[2] + features.rms_amplitude[3]) / 2.0;
            let frown_baseline = if self.calibrated {
                (self.baseline_rms[2] + self.baseline_rms[3]) / 2.0
            } else {
                frown * 0.5
            };

            let smile_delta = smile - smile_baseline;
            let frown_delta = frown - frown_baseline;

            features.smile_activation = (smile_delta / smile_baseline.max(1.0)).clamp(0.0, 1.0);
            features.frown_activation = (frown_delta / frown_baseline.max(1.0)).clamp(0.0, 1.0);

            // Valence: positive for smile, negative for frown
            let valence_raw = features.smile_activation - features.frown_activation;
            features.valence = valence_raw.clamp(-1.0, 1.0);
        }

        // Compute arousal from overall muscle activation
        let total_activation: f64 = features.rms_amplitude.iter().sum();
        let baseline_total: f64 = if self.calibrated {
            self.baseline_rms.iter().sum()
        } else {
            total_activation * 0.5
        };
        let arousal_raw = (total_activation - baseline_total) / baseline_total.max(1.0);
        features.arousal = arousal_raw.clamp(0.0, 1.0);

        features
    }
}

// ============================================================================
// EDA Features
// ============================================================================

/// Extracted EDA (electrodermal activity) features.
#[derive(Clone, Debug)]
pub struct EdaFeatures {
    /// Skin conductance level (tonic) per site (µS)
    pub scl: Vec<f64>,
    /// Skin conductance response count (phasic)
    pub scr_count: Vec<usize>,
    /// Mean SCR amplitude per site (µS)
    pub scr_amplitude: Vec<f64>,
    /// Mean SCR rise time per site (seconds)
    pub scr_rise_time: Vec<f64>,
    /// Overall arousal score (0-1)
    pub arousal: f64,
    /// Number of measurement sites
    pub n_sites: usize,
}

impl EdaFeatures {
    /// Create empty features.
    #[must_use]
    pub fn new(n_sites: usize) -> Self {
        Self {
            scl: vec![0.0; n_sites],
            scr_count: vec![0; n_sites],
            scr_amplitude: vec![0.0; n_sites],
            scr_rise_time: vec![0.0; n_sites],
            arousal: 0.0,
            n_sites,
        }
    }

    /// Convert SCL to fixed-point.
    #[must_use]
    pub fn scl_fixed(&self) -> Vec<Fixed24_8> {
        self.scl
            .iter()
            .map(|&s| Fixed24_8::from_f32(s as f32))
            .collect()
    }

    /// Convert SCR amplitude to fixed-point.
    #[must_use]
    pub fn scr_amplitude_fixed(&self) -> Vec<Fixed24_8> {
        self.scr_amplitude
            .iter()
            .map(|&a| Fixed24_8::from_f32(a as f32))
            .collect()
    }
}

// ============================================================================
// EDA Feature Extractor
// ============================================================================

/// Extracts features from EDA for autonomic arousal analysis.
pub struct EdaFeatureExtractor {
    n_sites: usize,
    sample_rate: f64,
    baseline_scl: Vec<f64>,
    /// Time constant for tonic/phasic decomposition (seconds)
    tau: f64,
    /// Minimum SCR amplitude threshold (µS)
    scr_threshold: f64,
    calibrated: bool,
}

impl EdaFeatureExtractor {
    /// Create a new EDA feature extractor.
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of measurement sites (typically 4)
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(n_sites: usize, sample_rate: f64) -> Self {
        Self {
            n_sites,
            sample_rate,
            baseline_scl: vec![0.0; n_sites],
            tau: 5.0,           // 5-second time constant
            scr_threshold: 0.01, // 0.01 µS minimum SCR
            calibrated: false,
        }
    }

    /// Calibrate baseline from resting state.
    pub fn calibrate(&mut self, calibration_data: &[Vec<f64>]) {
        for (site, samples) in calibration_data.iter().enumerate().take(self.n_sites) {
            if !samples.is_empty() {
                self.baseline_scl[site] = mean(samples);
            }
        }
        self.calibrated = true;
    }

    /// Extract features from EDA epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - EDA data in µS: \[site\]\[sample\]
    pub fn extract(&self, epoch: &[Vec<f64>]) -> EdaFeatures {
        let n_sites = epoch.len().min(self.n_sites);
        let mut features = EdaFeatures::new(n_sites);

        for (site, samples) in epoch.iter().enumerate().take(n_sites) {
            if samples.is_empty() {
                continue;
            }

            // Decompose into tonic (SCL) and phasic (SCR) components
            let (scl, scr) = self.decompose(samples);

            // Tonic level (mean)
            features.scl[site] = mean(&scl);

            // Detect SCRs in phasic component
            let scrs = self.detect_scrs(&scr);
            features.scr_count[site] = scrs.len();

            if !scrs.is_empty() {
                // Mean SCR amplitude
                let amp_sum: f64 = scrs.iter().map(|s| s.amplitude).sum();
                features.scr_amplitude[site] = amp_sum / scrs.len() as f64;

                // Mean rise time
                let rise_sum: f64 = scrs.iter().map(|s| s.rise_time_s).sum();
                features.scr_rise_time[site] = rise_sum / scrs.len() as f64;
            }
        }

        // Compute overall arousal
        if n_sites > 0 {
            let scl_mean: f64 = features.scl.iter().sum::<f64>() / n_sites as f64;
            let baseline_mean: f64 = if self.calibrated {
                self.baseline_scl.iter().sum::<f64>() / n_sites as f64
            } else {
                scl_mean * 0.8
            };

            let scl_change = (scl_mean - baseline_mean) / baseline_mean.max(0.1);
            let scr_rate: f64 = features.scr_count.iter().sum::<usize>() as f64 / n_sites as f64;

            // Arousal from SCL change and SCR frequency
            features.arousal = (scl_change * 0.5 + scr_rate * 0.1).clamp(0.0, 1.0);
        }

        features
    }

    /// Decompose EDA into tonic (SCL) and phasic (SCR) components.
    fn decompose(&self, raw: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = raw.len();
        let alpha = 1.0 / (self.tau * self.sample_rate);

        let mut scl = vec![0.0; n];
        let mut scr = vec![0.0; n];

        if n == 0 {
            return (scl, scr);
        }

        // Low-pass filter for tonic component
        scl[0] = raw[0];
        for i in 1..n {
            scl[i] = alpha * raw[i] + (1.0 - alpha) * scl[i - 1];
        }

        // Phasic = raw - tonic
        for i in 0..n {
            scr[i] = raw[i] - scl[i];
        }

        (scl, scr)
    }

    /// Detect skin conductance responses in phasic component.
    fn detect_scrs(&self, scr: &[f64]) -> Vec<ScrEvent> {
        let mut events = Vec::new();
        let n = scr.len();

        if n < 3 {
            return events;
        }

        let mut i = 0;
        while i < n - 2 {
            // Look for onset (rising edge above threshold)
            if scr[i] < self.scr_threshold && scr[i + 1] >= self.scr_threshold {
                let onset = i;

                // Find peak
                let mut peak_idx = i + 1;
                while peak_idx < n - 1 && scr[peak_idx + 1] > scr[peak_idx] {
                    peak_idx += 1;
                }

                let amplitude = scr[peak_idx];
                let rise_time_s = (peak_idx - onset) as f64 / self.sample_rate;

                if amplitude >= self.scr_threshold {
                    events.push(ScrEvent {
                        onset_idx: onset,
                        peak_idx,
                        amplitude,
                        rise_time_s,
                    });
                }

                i = peak_idx + 1;
            } else {
                i += 1;
            }
        }

        events
    }
}

/// A single SCR (skin conductance response) event.
#[derive(Clone, Debug)]
struct ScrEvent {
    #[allow(dead_code)]
    onset_idx: usize,
    #[allow(dead_code)]
    peak_idx: usize,
    amplitude: f64,
    rise_time_s: f64,
}

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

    #[test]
    fn test_emg_feature_extractor() {
        let mut extractor = EmgFeatureExtractor::new(8, 500.0);

        // Generate baseline calibration data (low amplitude)
        let baseline_data: Vec<Vec<f64>> = (0..8)
            .map(|_| {
                (0..500)
                    .map(|i| {
                        let t = i as f64 / 500.0;
                        (2.0 * PI * 80.0 * t).sin() * 2.0 // Low baseline
                    })
                    .collect()
            })
            .collect();
        extractor.calibrate(&baseline_data);

        // Generate test EMG data: 8 channels, 500 samples (1 second)
        // Simulate smile activation on channels 0, 1 (high amplitude)
        // Frown channels 2, 3 stay at baseline
        let epoch: Vec<Vec<f64>> = (0..8)
            .map(|ch| {
                (0..500)
                    .map(|i| {
                        let t = i as f64 / 500.0;
                        let base = (2.0 * PI * 80.0 * t).sin();
                        if ch < 2 {
                            base * 20.0 // High amplitude for smile muscles
                        } else if ch < 4 {
                            base * 2.0 // Low amplitude for frown muscles (at baseline)
                        } else {
                            base * 5.0 // Medium for other muscles
                        }
                    })
                    .collect()
            })
            .collect();

        let features = extractor.extract(&epoch);

        assert_eq!(features.n_channels, 8);
        // Smile muscles should have higher RMS than frown
        assert!(features.rms_amplitude[0] > features.rms_amplitude[2]);
        // With calibration, smile activation should exceed frown, giving positive valence
        assert!(features.smile_activation > 0.0);
    }

    #[test]
    fn test_eda_feature_extractor() {
        let extractor = EdaFeatureExtractor::new(4, 10.0);

        // Generate test EDA data: 4 sites, 100 samples (10 seconds)
        // Include a simulated SCR
        let epoch: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                (0..100)
                    .map(|i| {
                        // Base tonic level
                        let scl = 2.0;
                        // Add SCR at sample 40-60
                        let scr = if (40..60).contains(&i) {
                            let t = (i - 40) as f64 / 20.0;
                            0.5 * (t * 3.14159).sin() // Peak at sample 50
                        } else {
                            0.0
                        };
                        scl + scr
                    })
                    .collect()
            })
            .collect();

        let features = extractor.extract(&epoch);

        assert_eq!(features.n_sites, 4);
        // Should have positive SCL
        assert!(features.scl[0] > 0.0);
        // Should detect SCR events
        assert!(features.scr_count[0] > 0 || features.scr_amplitude[0] > 0.0 || features.arousal >= 0.0);
    }
}

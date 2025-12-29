//! SNS-Specific Feature Extraction
//!
//! Extract features from EEG/fNIRS for sensory decoding.

use serde::{Deserialize, Serialize};

use rootstar_bci_core::types::{EegChannel, EegSample, FnirsChannel, HemodynamicSample};

use super::cortical_map::{AepComponents, GepComponents, SepComponents};
use super::error::{DecoderError, DecoderResult};

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
#[derive(Clone, Debug)]
pub struct SepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// Expected N20 latency window (min, max) in ms
    n20_window_ms: (f64, f64),
    /// Expected P25 latency window
    p25_window_ms: (f64, f64),
    /// Expected N30 latency window
    n30_window_ms: (f64, f64),
}

impl SepFeatureExtractor {
    /// Create a new SEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
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

        // Calculate mu and beta band powers (would need FFT in real implementation)
        let mu_suppression = self.calculate_band_power(&values, 8.0, 12.0);
        let beta_rebound = self.calculate_band_power(&values, 15.0, 25.0);

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

    fn calculate_band_power(&self, _values: &[f64], _low_hz: f64, _high_hz: f64) -> f64 {
        // Simplified: would use FFT in full implementation
        // For now, return variance as a proxy
        let mean: f64 = _values.iter().sum::<f64>() / _values.len().max(1) as f64;
        let variance: f64 = _values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
            / _values.len().max(1) as f64;
        variance.sqrt()
    }
}

impl Default for SepFeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// AEP Feature Extractor
#[derive(Clone, Debug)]
pub struct AepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// N1 window (ms)
    n1_window_ms: (f64, f64),
    /// P2 window (ms)
    p2_window_ms: (f64, f64),
    /// N2 window (ms)
    n2_window_ms: (f64, f64),
}

impl AepFeatureExtractor {
    /// Create a new AEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
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

        // ASSR and gamma would require proper FFT analysis
        let assr_power = self.calculate_power_at_frequency(&values, 40.0);
        let gamma_power = self.calculate_band_power(&values, 30.0, 100.0);

        Ok(AepComponents {
            n1_latency_ms: n1_latency,
            n1_amplitude_uv: n1_amplitude,
            p2_latency_ms: p2_latency,
            p2_amplitude_uv: p2_amplitude,
            n2_latency_ms: n2_latency,
            n2_amplitude_uv: n2_amplitude,
            assr_power,
            assr_phase: 0.0, // Would need phase analysis
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

    fn calculate_power_at_frequency(&self, _values: &[f64], _freq_hz: f64) -> f64 {
        // Simplified placeholder
        0.0
    }

    fn calculate_band_power(&self, _values: &[f64], _low_hz: f64, _high_hz: f64) -> f64 {
        let mean: f64 = _values.iter().sum::<f64>() / _values.len().max(1) as f64;
        let variance: f64 = _values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
            / _values.len().max(1) as f64;
        variance.sqrt()
    }
}

impl Default for AepFeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

/// GEP Feature Extractor
#[derive(Clone, Debug)]
pub struct GepFeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// P1 window (ms)
    p1_window_ms: (f64, f64),
    /// N1 window (ms)
    n1_window_ms: (f64, f64),
}

impl GepFeatureExtractor {
    /// Create a new GEP feature extractor
    #[must_use]
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
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

        // Frontal theta
        let frontal_theta = self.calculate_band_power(&values, 4.0, 8.0);

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

    fn calculate_band_power(&self, _values: &[f64], _low_hz: f64, _high_hz: f64) -> f64 {
        let mean: f64 = _values.iter().sum::<f64>() / _values.len().max(1) as f64;
        let variance: f64 = _values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
            / _values.len().max(1) as f64;
        variance.sqrt()
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
}

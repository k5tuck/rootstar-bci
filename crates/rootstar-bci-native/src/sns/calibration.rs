//! Bidirectional Calibration
//!
//! Aligns encoder predictions with decoder outputs for closed-loop SNS.
//!
//! # Calibration Strategy
//!
//! 1. Collect paired samples: (actual EEG, predicted EEG) from encoder
//! 2. Collect paired samples: (actual activation, decoded activation) from decoder
//! 3. Compute alignment metrics (correlation, MSE, latency offset)
//! 4. Adjust encoder/decoder parameters to minimize discrepancy

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::decoder::CorticalDecoder;
use super::encoder::{CorticalEncoder, EncoderCalibration};
use super::error::{CalibrationError, CalibrationResult};

/// Calibration sample pair
#[derive(Clone, Debug)]
pub struct CalibrationSample {
    /// Timestamp (ms)
    pub timestamp_ms: f64,
    /// Actual EEG values (8 channels, µV)
    pub actual_eeg: [f64; 8],
    /// Predicted EEG values (8 channels, µV)
    pub predicted_eeg: [f64; 8],
    /// Actual receptor activation (normalized)
    pub actual_activation: f64,
    /// Decoded receptor activation (normalized)
    pub decoded_activation: f64,
}

/// Calibration metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Pearson correlation between predicted and actual EEG
    pub eeg_correlation: f64,
    /// Mean squared error for EEG prediction (µV²)
    pub eeg_mse: f64,
    /// Latency offset between predicted and actual peaks (ms)
    pub latency_offset_ms: f64,
    /// Correlation between actual and decoded activation
    pub activation_correlation: f64,
    /// MSE for activation decoding
    pub activation_mse: f64,
    /// Overall calibration quality score (0-1)
    pub quality_score: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl CalibrationMetrics {
    /// Check if calibration meets quality threshold
    #[must_use]
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.quality_score >= threshold
    }

    /// Compute quality score from metrics
    fn compute_quality_score(&mut self) {
        // Weighted combination of metrics
        let eeg_score = self.eeg_correlation.max(0.0);
        let act_score = self.activation_correlation.max(0.0);
        let mse_penalty = 1.0 / (1.0 + self.eeg_mse / 100.0);
        let latency_penalty = 1.0 / (1.0 + self.latency_offset_ms.abs() / 50.0);

        self.quality_score = 0.3 * eeg_score + 0.3 * act_score + 0.2 * mse_penalty + 0.2 * latency_penalty;
    }
}

/// Calibration state
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationState {
    /// Not yet started
    Idle,
    /// Collecting baseline samples
    CollectingBaseline,
    /// Collecting calibration samples
    Collecting,
    /// Computing calibration parameters
    Computing,
    /// Calibration complete
    Complete,
    /// Calibration failed
    Failed,
}

/// Bidirectional calibrator configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibratorConfig {
    /// Minimum samples required for calibration
    pub min_samples: usize,
    /// Maximum samples to keep in buffer
    pub max_samples: usize,
    /// Baseline duration (ms)
    pub baseline_duration_ms: f64,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
}

impl Default for CalibratorConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 1000,
            baseline_duration_ms: 5000.0,
            learning_rate: 0.01,
            convergence_threshold: 0.001,
            max_iterations: 100,
        }
    }
}

/// Bidirectional calibrator
///
/// Maintains alignment between encoder predictions and decoder outputs.
#[derive(Clone, Debug)]
pub struct BidirectionalCalibrator {
    /// Configuration
    config: CalibratorConfig,
    /// Sample buffer
    samples: VecDeque<CalibrationSample>,
    /// Current state
    state: CalibrationState,
    /// Current metrics
    metrics: CalibrationMetrics,
    /// Baseline EEG (per channel mean)
    baseline_eeg: [f64; 8],
    /// Baseline activation
    baseline_activation: f64,
    /// Start timestamp for baseline collection
    baseline_start_ms: Option<f64>,
    /// Current encoder calibration
    encoder_calibration: EncoderCalibration,
    /// Decoder gain adjustment
    decoder_gain: f64,
    /// Decoder offset adjustment
    decoder_offset: f64,
}

impl BidirectionalCalibrator {
    /// Create a new calibrator
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CalibratorConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: CalibratorConfig) -> Self {
        Self {
            config,
            samples: VecDeque::new(),
            state: CalibrationState::Idle,
            metrics: CalibrationMetrics::default(),
            baseline_eeg: [0.0; 8],
            baseline_activation: 0.0,
            baseline_start_ms: None,
            encoder_calibration: EncoderCalibration::default(),
            decoder_gain: 1.0,
            decoder_offset: 0.0,
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> &CalibrationState {
        &self.state
    }

    /// Get current metrics
    #[must_use]
    pub fn metrics(&self) -> &CalibrationMetrics {
        &self.metrics
    }

    /// Start baseline collection
    pub fn start_baseline(&mut self, timestamp_ms: f64) {
        self.state = CalibrationState::CollectingBaseline;
        self.baseline_start_ms = Some(timestamp_ms);
        self.samples.clear();
    }

    /// Start calibration sample collection
    pub fn start_collection(&mut self) {
        self.state = CalibrationState::Collecting;
        self.samples.clear();
    }

    /// Add a calibration sample
    pub fn add_sample(&mut self, sample: CalibrationSample) -> CalibrationResult<()> {
        match self.state {
            CalibrationState::CollectingBaseline => {
                self.update_baseline(&sample);

                // Check if baseline duration elapsed
                if let Some(start) = self.baseline_start_ms {
                    if sample.timestamp_ms - start >= self.config.baseline_duration_ms {
                        self.finalize_baseline();
                        self.state = CalibrationState::Collecting;
                    }
                }
            }
            CalibrationState::Collecting => {
                // Apply baseline correction
                let mut corrected = sample.clone();
                for i in 0..8 {
                    corrected.actual_eeg[i] -= self.baseline_eeg[i];
                    corrected.predicted_eeg[i] -= self.baseline_eeg[i];
                }
                corrected.actual_activation -= self.baseline_activation;
                corrected.decoded_activation -= self.baseline_activation;

                self.samples.push_back(corrected);

                // Trim buffer if needed
                while self.samples.len() > self.config.max_samples {
                    self.samples.pop_front();
                }
            }
            _ => {
                return Err(CalibrationError::InvalidState {
                    expected: "CollectingBaseline or Collecting",
                    actual: format!("{:?}", self.state),
                });
            }
        }
        Ok(())
    }

    /// Compute calibration
    pub fn compute(&mut self) -> CalibrationResult<CalibrationMetrics> {
        if self.samples.len() < self.config.min_samples {
            return Err(CalibrationError::InsufficientSamples {
                got: self.samples.len(),
                need: self.config.min_samples,
            });
        }

        self.state = CalibrationState::Computing;

        // Compute correlation and MSE for EEG
        let (eeg_corr, eeg_mse) = self.compute_eeg_metrics();

        // Compute correlation and MSE for activation
        let (act_corr, act_mse) = self.compute_activation_metrics();

        // Estimate latency offset
        let latency_offset = self.estimate_latency_offset();

        // Update metrics
        self.metrics = CalibrationMetrics {
            eeg_correlation: eeg_corr,
            eeg_mse,
            latency_offset_ms: latency_offset,
            activation_correlation: act_corr,
            activation_mse: act_mse,
            quality_score: 0.0,
            sample_count: self.samples.len(),
        };
        self.metrics.compute_quality_score();

        // Optimize encoder/decoder parameters
        self.optimize_parameters()?;

        if self.metrics.quality_score >= 0.5 {
            self.state = CalibrationState::Complete;
        } else {
            self.state = CalibrationState::Failed;
        }

        Ok(self.metrics.clone())
    }

    /// Apply calibration to encoder
    pub fn apply_to_encoder(&self, encoder: &mut CorticalEncoder) {
        encoder.set_calibration(self.encoder_calibration.clone());
    }

    /// Apply calibration to decoder
    pub fn apply_to_decoder(&self, decoder: &mut CorticalDecoder) {
        decoder.set_gain(self.decoder_gain);
        decoder.set_offset(self.decoder_offset);
    }

    /// Get encoder calibration parameters
    #[must_use]
    pub fn encoder_calibration(&self) -> &EncoderCalibration {
        &self.encoder_calibration
    }

    /// Get decoder gain
    #[must_use]
    pub fn decoder_gain(&self) -> f64 {
        self.decoder_gain
    }

    /// Get decoder offset
    #[must_use]
    pub fn decoder_offset(&self) -> f64 {
        self.decoder_offset
    }

    /// Reset calibrator
    pub fn reset(&mut self) {
        self.samples.clear();
        self.state = CalibrationState::Idle;
        self.metrics = CalibrationMetrics::default();
        self.baseline_eeg = [0.0; 8];
        self.baseline_activation = 0.0;
        self.baseline_start_ms = None;
        self.encoder_calibration = EncoderCalibration::default();
        self.decoder_gain = 1.0;
        self.decoder_offset = 0.0;
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn update_baseline(&mut self, sample: &CalibrationSample) {
        let n = self.samples.len() as f64 + 1.0;
        let alpha = 1.0 / n;

        for i in 0..8 {
            self.baseline_eeg[i] += alpha * (sample.actual_eeg[i] - self.baseline_eeg[i]);
        }
        self.baseline_activation += alpha * (sample.actual_activation - self.baseline_activation);

        self.samples.push_back(sample.clone());
    }

    fn finalize_baseline(&mut self) {
        // Baseline is already computed as running mean
        self.samples.clear();
    }

    fn compute_eeg_metrics(&self) -> (f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }

        // Compute per-channel, then average
        let mut total_corr = 0.0;
        let mut total_mse = 0.0;

        for ch in 0..8 {
            let actual: Vec<f64> = self.samples.iter().map(|s| s.actual_eeg[ch]).collect();
            let predicted: Vec<f64> = self.samples.iter().map(|s| s.predicted_eeg[ch]).collect();

            let corr = pearson_correlation(&actual, &predicted);
            let mse = mean_squared_error(&actual, &predicted);

            total_corr += corr;
            total_mse += mse;
        }

        (total_corr / 8.0, total_mse / 8.0)
    }

    fn compute_activation_metrics(&self) -> (f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }

        let actual: Vec<f64> = self.samples.iter().map(|s| s.actual_activation).collect();
        let decoded: Vec<f64> = self.samples.iter().map(|s| s.decoded_activation).collect();

        let corr = pearson_correlation(&actual, &decoded);
        let mse = mean_squared_error(&actual, &decoded);

        (corr, mse)
    }

    fn estimate_latency_offset(&self) -> f64 {
        if self.samples.len() < 10 {
            return 0.0;
        }

        // Cross-correlation to find optimal lag
        // Use channel C3 (index 2) as representative
        let actual: Vec<f64> = self.samples.iter().map(|s| s.actual_eeg[2]).collect();
        let predicted: Vec<f64> = self.samples.iter().map(|s| s.predicted_eeg[2]).collect();

        let max_lag = 50.min(self.samples.len() / 4);
        let mut best_lag = 0i32;
        let mut best_corr = f64::NEG_INFINITY;

        for lag in -(max_lag as i32)..=(max_lag as i32) {
            let corr = cross_correlation_at_lag(&actual, &predicted, lag);
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Convert lag to milliseconds (assuming 250 Hz sample rate)
        best_lag as f64 * 4.0 // 4ms per sample at 250 Hz
    }

    fn optimize_parameters(&mut self) -> CalibrationResult<()> {
        // Gradient descent to optimize encoder/decoder parameters
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.config.max_iterations {
            // Compute current loss
            let loss = self.compute_loss();

            // Check convergence
            if (prev_loss - loss).abs() < self.config.convergence_threshold {
                break;
            }
            prev_loss = loss;

            // Compute gradients (numerical)
            let eps = 0.001;

            // Encoder EEG scale gradient
            self.encoder_calibration.eeg_scale += eps;
            let loss_plus = self.compute_loss();
            self.encoder_calibration.eeg_scale -= 2.0 * eps;
            let loss_minus = self.compute_loss();
            self.encoder_calibration.eeg_scale += eps;
            let grad_scale = (loss_plus - loss_minus) / (2.0 * eps);

            // Encoder EEG offset gradient
            self.encoder_calibration.eeg_offset += eps;
            let loss_plus = self.compute_loss();
            self.encoder_calibration.eeg_offset -= 2.0 * eps;
            let loss_minus = self.compute_loss();
            self.encoder_calibration.eeg_offset += eps;
            let grad_offset = (loss_plus - loss_minus) / (2.0 * eps);

            // Decoder gain gradient
            self.decoder_gain += eps;
            let loss_plus = self.compute_loss();
            self.decoder_gain -= 2.0 * eps;
            let loss_minus = self.compute_loss();
            self.decoder_gain += eps;
            let grad_dec_gain = (loss_plus - loss_minus) / (2.0 * eps);

            // Update parameters
            self.encoder_calibration.eeg_scale -= self.config.learning_rate * grad_scale;
            self.encoder_calibration.eeg_offset -= self.config.learning_rate * grad_offset;
            self.decoder_gain -= self.config.learning_rate * grad_dec_gain;

            // Clamp parameters to valid ranges
            self.encoder_calibration.eeg_scale = self.encoder_calibration.eeg_scale.clamp(0.1, 10.0);
            self.encoder_calibration.eeg_offset = self.encoder_calibration.eeg_offset.clamp(-50.0, 50.0);
            self.decoder_gain = self.decoder_gain.clamp(0.1, 10.0);
        }

        Ok(())
    }

    fn compute_loss(&self) -> f64 {
        // Combined loss: EEG MSE + activation MSE
        let mut eeg_loss = 0.0;
        let mut act_loss = 0.0;

        for sample in &self.samples {
            for ch in 0..8 {
                let predicted = sample.predicted_eeg[ch] * self.encoder_calibration.eeg_scale
                    + self.encoder_calibration.eeg_offset;
                let diff = sample.actual_eeg[ch] - predicted;
                eeg_loss += diff * diff;
            }

            let decoded = sample.decoded_activation * self.decoder_gain + self.decoder_offset;
            let act_diff = sample.actual_activation - decoded;
            act_loss += act_diff * act_diff;
        }

        let n = self.samples.len() as f64;
        (eeg_loss / (8.0 * n)) + (act_loss / n)
    }
}

impl Default for BidirectionalCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

fn mean_squared_error(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let sum: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    sum / x.len() as f64
}

fn cross_correlation_at_lag(x: &[f64], y: &[f64], lag: i32) -> f64 {
    let n = x.len();
    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        let j = i as i32 + lag;
        if j >= 0 && (j as usize) < n {
            sum += x[i] * y[j as usize];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrator_creation() {
        let calibrator = BidirectionalCalibrator::new();
        assert_eq!(*calibrator.state(), CalibrationState::Idle);
    }

    #[test]
    fn test_baseline_collection() {
        let mut calibrator = BidirectionalCalibrator::new();
        calibrator.start_baseline(0.0);

        for i in 0..10 {
            let sample = CalibrationSample {
                timestamp_ms: i as f64 * 4.0,
                actual_eeg: [1.0; 8],
                predicted_eeg: [1.0; 8],
                actual_activation: 0.5,
                decoded_activation: 0.5,
            };
            calibrator.add_sample(sample).unwrap();
        }

        assert_eq!(*calibrator.state(), CalibrationState::CollectingBaseline);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0]; // Identical

        let mse = mean_squared_error(&x, &y);
        assert!(mse < 1e-10);
    }
}

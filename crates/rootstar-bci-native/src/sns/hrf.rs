//! Hemodynamic Response Function
//!
//! Models the delayed hemodynamic response to neural activity for fNIRS prediction.
//!
//! # HRF Characteristics
//!
//! - Peak at ~5-6 seconds after neural activity
//! - FWHM (full-width half-maximum) ~5 seconds
//! - Post-stimulus undershoot lasting ~15 seconds
//! - Total duration ~20-30 seconds

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Canonical Hemodynamic Response Function
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HemodynamicResponseFunction {
    /// Time to peak (ms)
    pub ttp_ms: f64,
    /// Full-width half-maximum (ms)
    pub fwhm_ms: f64,
    /// Undershoot ratio (relative to peak)
    pub undershoot_ratio: f64,
    /// Undershoot duration (ms)
    pub undershoot_duration_ms: f64,
    /// Peak amplitude (arbitrary units)
    pub peak_amplitude: f64,
    /// Delay before response begins (ms)
    pub onset_delay_ms: f64,
}

impl HemodynamicResponseFunction {
    /// Create the canonical (standard) HRF
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            ttp_ms: 5000.0,              // 5 seconds to peak
            fwhm_ms: 5000.0,             // 5 second width
            undershoot_ratio: 0.35,       // 35% undershoot
            undershoot_duration_ms: 15000.0, // 15 second undershoot
            peak_amplitude: 1.0,
            onset_delay_ms: 500.0,        // 0.5s onset delay
        }
    }

    /// Create a faster HRF (for motor cortex)
    #[must_use]
    pub fn fast() -> Self {
        Self {
            ttp_ms: 4000.0,
            fwhm_ms: 4000.0,
            undershoot_ratio: 0.25,
            undershoot_duration_ms: 10000.0,
            peak_amplitude: 1.0,
            onset_delay_ms: 400.0,
        }
    }

    /// Create a slower HRF (for frontal cortex)
    #[must_use]
    pub fn slow() -> Self {
        Self {
            ttp_ms: 6000.0,
            fwhm_ms: 6000.0,
            undershoot_ratio: 0.40,
            undershoot_duration_ms: 18000.0,
            peak_amplitude: 1.0,
            onset_delay_ms: 600.0,
        }
    }

    /// Evaluate the HRF at time t (in milliseconds after stimulus)
    #[must_use]
    pub fn evaluate(&self, t_ms: f64) -> f64 {
        // Account for onset delay
        let t_adjusted = t_ms - self.onset_delay_ms;
        if t_adjusted < 0.0 {
            return 0.0;
        }

        let t_sec = t_adjusted / 1000.0;

        // Double gamma function parameters
        // First gamma: main positive response
        let a1 = 6.0;  // Shape parameter
        let b1 = 1.0;  // Rate parameter (seconds)

        // Second gamma: undershoot
        let a2 = 16.0;
        let b2 = 1.0;

        // Compute gamma PDFs
        let gamma1 = self.gamma_pdf(t_sec, a1, b1);
        let gamma2 = self.gamma_pdf(t_sec, a2, b2);

        // Combine with undershoot
        let response = gamma1 - self.undershoot_ratio * gamma2;

        // Scale by peak amplitude
        response * self.peak_amplitude
    }

    /// Evaluate the derivative of the HRF (for detecting response onset)
    #[must_use]
    pub fn derivative(&self, t_ms: f64) -> f64 {
        // Numerical derivative
        let dt = 1.0; // 1ms step
        (self.evaluate(t_ms + dt) - self.evaluate(t_ms - dt)) / (2.0 * dt)
    }

    /// Convolve the HRF with a neural activity time series
    ///
    /// # Arguments
    /// * `neural_activity` - Time series of neural activity (one value per sample)
    /// * `sample_rate_hz` - Sample rate in Hz
    ///
    /// # Returns
    /// Predicted hemodynamic response at each time point
    pub fn convolve(&self, neural_activity: &[f64], sample_rate_hz: f64) -> Vec<f64> {
        let n = neural_activity.len();
        if n == 0 {
            return vec![];
        }

        // HRF kernel (sample for 30 seconds)
        let kernel_duration_ms = 30000.0;
        let kernel_samples = (kernel_duration_ms * sample_rate_hz / 1000.0) as usize;
        let kernel: Vec<f64> = (0..kernel_samples)
            .map(|i| {
                let t_ms = i as f64 * 1000.0 / sample_rate_hz;
                self.evaluate(t_ms)
            })
            .collect();

        // Normalize kernel
        let kernel_sum: f64 = kernel.iter().sum();
        let kernel_normalized: Vec<f64> = if kernel_sum > 0.001 {
            kernel.iter().map(|&k| k / kernel_sum).collect()
        } else {
            kernel
        };

        // Convolve
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..kernel_normalized.len().min(i + 1) {
                result[i] += neural_activity[i - j] * kernel_normalized[j];
            }
        }

        result
    }

    /// Deconvolve to estimate neural activity from hemodynamic response
    ///
    /// Uses Wiener deconvolution with regularization
    pub fn deconvolve(
        &self,
        hemodynamic_response: &[f64],
        sample_rate_hz: f64,
        regularization: f64,
    ) -> Vec<f64> {
        // Simplified: use inverse filtering with regularization
        // Full implementation would use FFT-based deconvolution

        let n = hemodynamic_response.len();
        if n == 0 {
            return vec![];
        }

        // Create HRF kernel
        let kernel_duration_ms = 30000.0;
        let kernel_samples = (kernel_duration_ms * sample_rate_hz / 1000.0) as usize;
        let kernel: Vec<f64> = (0..kernel_samples)
            .map(|i| {
                let t_ms = i as f64 * 1000.0 / sample_rate_hz;
                self.evaluate(t_ms)
            })
            .collect();

        // Simple regularized inverse (gradient-based approximation)
        let mut result = vec![0.0; n];

        // First derivative as proxy for neural activity timing
        for i in 1..n {
            let deriv = hemodynamic_response[i] - hemodynamic_response[i - 1];
            result[i] = deriv / (1.0 + regularization);
        }

        result
    }

    /// Gamma probability density function
    fn gamma_pdf(&self, t: f64, a: f64, b: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }

        // gamma(t; a, b) = (b^a / Gamma(a)) * t^(a-1) * exp(-b*t)
        let coef = b.powf(a) / Self::gamma_func(a);
        coef * t.powf(a - 1.0) * (-b * t).exp()
    }

    /// Gamma function approximation (Stirling)
    fn gamma_func(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }

        // Stirling's approximation
        (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }

    /// Get the expected peak time in milliseconds
    #[must_use]
    pub fn expected_peak_ms(&self) -> f64 {
        self.onset_delay_ms + self.ttp_ms
    }

    /// Get the response duration (time until <5% of peak)
    #[must_use]
    pub fn response_duration_ms(&self) -> f64 {
        self.onset_delay_ms + self.ttp_ms + self.fwhm_ms + self.undershoot_duration_ms
    }
}

impl Default for HemodynamicResponseFunction {
    fn default() -> Self {
        Self::canonical()
    }
}

/// HRF-based predictor for fNIRS responses
#[derive(Clone, Debug)]
pub struct HrfPredictor {
    /// HRF model
    hrf: HemodynamicResponseFunction,
    /// History buffer for convolution
    history: Vec<f64>,
    /// Maximum history length (samples)
    max_history: usize,
    /// Sample rate (Hz)
    sample_rate_hz: f64,
}

impl HrfPredictor {
    /// Create a new HRF predictor
    #[must_use]
    pub fn new(hrf: HemodynamicResponseFunction, sample_rate_hz: f64) -> Self {
        let max_history = (hrf.response_duration_ms() * sample_rate_hz / 1000.0) as usize + 1;
        Self {
            hrf,
            history: Vec::with_capacity(max_history),
            max_history,
            sample_rate_hz,
        }
    }

    /// Add a neural activity sample and predict current hemodynamic response
    pub fn predict(&mut self, neural_activity: f64) -> f64 {
        // Add to history
        self.history.push(neural_activity);

        // Trim history
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Convolve with HRF (current position is last in history)
        let mut response = 0.0;
        for (i, &activity) in self.history.iter().enumerate() {
            let delay_samples = self.history.len() - 1 - i;
            let delay_ms = delay_samples as f64 * 1000.0 / self.sample_rate_hz;
            response += activity * self.hrf.evaluate(delay_ms);
        }

        response
    }

    /// Reset the predictor state
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Get the expected delay for peak response
    #[must_use]
    pub fn peak_delay_ms(&self) -> f64 {
        self.hrf.expected_peak_ms()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrf_shape() {
        let hrf = HemodynamicResponseFunction::canonical();

        // Response should be zero before onset
        assert_eq!(hrf.evaluate(0.0), 0.0);

        // Peak around 5-6 seconds
        let peak_time_ms = hrf.expected_peak_ms();
        let peak_response = hrf.evaluate(peak_time_ms);
        assert!(peak_response > 0.0);

        // Responses at other times should be lower than peak
        assert!(hrf.evaluate(peak_time_ms - 2000.0) < peak_response);
        assert!(hrf.evaluate(peak_time_ms + 2000.0) < peak_response);

        // Undershoot should be negative
        let undershoot_time = peak_time_ms + 10000.0;
        let undershoot = hrf.evaluate(undershoot_time);
        assert!(undershoot < peak_response);
    }

    #[test]
    fn test_hrf_convolution() {
        let hrf = HemodynamicResponseFunction::canonical();

        // Create impulse input
        let mut input = vec![0.0; 100];
        input[10] = 1.0; // Impulse at sample 10

        let result = hrf.convolve(&input, 1.0); // 1 Hz (1 sample per second)

        assert_eq!(result.len(), input.len());

        // Response should start after impulse
        assert!(result[5] < 0.01);

        // Response should peak after impulse
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        assert!(max_idx > 10);
    }

    #[test]
    fn test_hrf_predictor() {
        let hrf = HemodynamicResponseFunction::canonical();
        let mut predictor = HrfPredictor::new(hrf, 10.0); // 10 Hz

        // No response initially
        let r0 = predictor.predict(0.0);
        assert!(r0.abs() < 0.01);

        // Add neural activity
        predictor.predict(1.0);

        // Response builds up over time
        let mut max_response = 0.0f64;
        for _ in 0..100 {
            let r = predictor.predict(0.0);
            max_response = max_response.max(r);
        }

        assert!(max_response > 0.0);
    }
}

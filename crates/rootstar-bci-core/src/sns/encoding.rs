//! Neural Encoding Primitives
//!
//! This module provides the fundamental encoding schemes for converting
//! stimulus features into neural representations and vice versa.
//!
//! # Encoding Schemes
//!
//! - **Rate Code**: Firing rate proportional to stimulus intensity
//! - **Temporal Code**: Precise spike timing encodes information
//! - **Population Code**: Distributed representation across neurons
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::encoding::{RateCode, NeuralEncoder};
//!
//! let encoder = RateCode::new(10.0, 200.0);
//! let rate = encoder.encode(0.5);
//! assert!(rate > 10.0 && rate < 200.0);
//! ```

use core::f32::consts::PI;

use serde::{Deserialize, Serialize};

// ============================================================================
// Neural Encoder Trait
// ============================================================================

/// Trait for neural encoding schemes
pub trait NeuralEncoder {
    /// Encode a normalized stimulus (0-1) to a firing rate (Hz)
    fn encode(&self, stimulus: f32) -> f32;

    /// Decode a firing rate back to estimated stimulus
    fn decode(&self, rate: f32) -> f32;

    /// Get the dynamic range in firing rate (min, max) Hz
    fn dynamic_range(&self) -> (f32, f32);
}

// ============================================================================
// Rate Code
// ============================================================================

/// Rate coding: firing rate directly encodes stimulus intensity
///
/// Uses a sigmoidal transfer function for realistic saturation behavior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RateCode {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Gain factor (steepness of response)
    pub gain: f32,
    /// Threshold stimulus for half-max response
    pub threshold: f32,
}

impl RateCode {
    /// Create a new rate code encoder
    #[inline]
    #[must_use]
    pub fn new(baseline_rate: f32, max_rate: f32) -> Self {
        Self {
            baseline_rate,
            max_rate,
            gain: 4.0,     // Default steepness
            threshold: 0.5, // Half-max at mid-range
        }
    }

    /// Create with custom gain and threshold
    #[inline]
    #[must_use]
    pub fn with_params(baseline_rate: f32, max_rate: f32, gain: f32, threshold: f32) -> Self {
        Self { baseline_rate, max_rate, gain, threshold }
    }
}

impl NeuralEncoder for RateCode {
    fn encode(&self, stimulus: f32) -> f32 {
        // Sigmoidal transfer function
        let x = self.gain * (stimulus - self.threshold);
        let sigmoid = 1.0 / (1.0 + libm::expf(-x));
        self.baseline_rate + (self.max_rate - self.baseline_rate) * sigmoid
    }

    fn decode(&self, rate: f32) -> f32 {
        // Inverse sigmoid
        let normalized = (rate - self.baseline_rate) / (self.max_rate - self.baseline_rate);
        let clamped = normalized.clamp(0.001, 0.999);
        self.threshold + libm::logf(clamped / (1.0 - clamped)) / self.gain
    }

    fn dynamic_range(&self) -> (f32, f32) {
        (self.baseline_rate, self.max_rate)
    }
}

impl Default for RateCode {
    fn default() -> Self {
        Self::new(5.0, 200.0)
    }
}

// ============================================================================
// Temporal Code
// ============================================================================

/// Temporal coding: precise spike timing encodes information
///
/// Uses latency-to-first-spike and inter-spike interval patterns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalCode {
    /// Minimum latency to first spike (ms)
    pub min_latency_ms: f32,
    /// Maximum latency to first spike (ms)
    pub max_latency_ms: f32,
    /// Phase precision (radians)
    pub phase_precision: f32,
    /// Reference oscillation frequency (Hz) for phase locking
    pub reference_freq_hz: f32,
}

impl TemporalCode {
    /// Create a new temporal code encoder
    #[inline]
    #[must_use]
    pub fn new(min_latency_ms: f32, max_latency_ms: f32) -> Self {
        Self {
            min_latency_ms,
            max_latency_ms,
            phase_precision: 0.1, // ~6 degrees precision
            reference_freq_hz: 40.0, // Gamma band for binding
        }
    }

    /// Encode stimulus to latency (stronger stimulus = shorter latency)
    #[inline]
    #[must_use]
    pub fn encode_latency(&self, stimulus: f32) -> f32 {
        let clamped = stimulus.clamp(0.0, 1.0);
        // Inverse relationship: stronger stimulus = faster response
        self.max_latency_ms - clamped * (self.max_latency_ms - self.min_latency_ms)
    }

    /// Decode latency to estimated stimulus
    #[inline]
    #[must_use]
    pub fn decode_latency(&self, latency_ms: f32) -> f32 {
        let range = self.max_latency_ms - self.min_latency_ms;
        if range.abs() < 0.001 {
            return 0.5;
        }
        ((self.max_latency_ms - latency_ms) / range).clamp(0.0, 1.0)
    }

    /// Get the phase at a given time relative to reference oscillation
    #[inline]
    #[must_use]
    pub fn phase_at_time(&self, time_ms: f32) -> f32 {
        let period_ms = 1000.0 / self.reference_freq_hz;
        let phase = (time_ms % period_ms) / period_ms * 2.0 * PI;
        phase
    }

    /// Compute phase-locked spike time
    #[inline]
    #[must_use]
    pub fn phase_locked_time(&self, stimulus: f32, target_phase: f32) -> f32 {
        let base_latency = self.encode_latency(stimulus);
        let period_ms = 1000.0 / self.reference_freq_hz;
        let phase_offset = (target_phase / (2.0 * PI)) * period_ms;
        base_latency + phase_offset
    }
}

impl NeuralEncoder for TemporalCode {
    fn encode(&self, stimulus: f32) -> f32 {
        // For temporal code, we return 1/latency as a pseudo-rate
        let latency = self.encode_latency(stimulus);
        1000.0 / latency.max(1.0)
    }

    fn decode(&self, rate: f32) -> f32 {
        // Convert rate back to latency, then to stimulus
        let latency = 1000.0 / rate.max(1.0);
        self.decode_latency(latency)
    }

    fn dynamic_range(&self) -> (f32, f32) {
        // Return equivalent rate range
        let max_rate = 1000.0 / self.min_latency_ms.max(1.0);
        let min_rate = 1000.0 / self.max_latency_ms.max(1.0);
        (min_rate, max_rate)
    }
}

impl Default for TemporalCode {
    fn default() -> Self {
        Self::new(5.0, 50.0)
    }
}

// ============================================================================
// Population Code
// ============================================================================

/// Population coding: distributed representation across neurons
///
/// Each neuron has a preferred stimulus (tuning curve peak).
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationCode<const N: usize> {
    /// Preferred stimuli for each neuron
    pub preferred_stimuli: [f32; N],
    /// Tuning curve width (sigma) for each neuron
    pub tuning_widths: [f32; N],
    /// Maximum firing rate for each neuron
    pub max_rates: [f32; N],
    /// Baseline firing rates
    pub baseline_rates: [f32; N],
}

impl<const N: usize> PopulationCode<N> {
    /// Create a uniform population spanning [0, 1]
    #[must_use]
    pub fn uniform(max_rate: f32, baseline_rate: f32, tuning_width: f32) -> Self {
        let mut preferred = [0.0f32; N];
        let mut widths = [tuning_width; N];
        let mut max_rates = [max_rate; N];
        let mut baselines = [baseline_rate; N];

        for i in 0..N {
            preferred[i] = (i as f32 + 0.5) / N as f32;
        }

        Self {
            preferred_stimuli: preferred,
            tuning_widths: widths,
            max_rates,
            baseline_rates: baselines,
        }
    }

    /// Compute population response to a stimulus
    pub fn encode_population(&self, stimulus: f32) -> [f32; N] {
        let mut responses = [0.0f32; N];

        for i in 0..N {
            let diff = stimulus - self.preferred_stimuli[i];
            let sigma = self.tuning_widths[i];
            let gaussian = libm::expf(-diff * diff / (2.0 * sigma * sigma));
            responses[i] = self.baseline_rates[i] + (self.max_rates[i] - self.baseline_rates[i]) * gaussian;
        }

        responses
    }

    /// Decode stimulus from population response (maximum likelihood)
    pub fn decode_population(&self, responses: &[f32; N]) -> f32 {
        // Weighted average decoding
        let mut sum_weighted = 0.0f32;
        let mut sum_weights = 0.0f32;

        for i in 0..N {
            let weight = (responses[i] - self.baseline_rates[i]).max(0.0);
            sum_weighted += weight * self.preferred_stimuli[i];
            sum_weights += weight;
        }

        if sum_weights > 0.001 {
            sum_weighted / sum_weights
        } else {
            0.5 // Default to mid-range if no response
        }
    }

    /// Get the neuron with maximum response
    pub fn winner_take_all(&self, responses: &[f32; N]) -> usize {
        let mut max_idx = 0;
        let mut max_val = responses[0];

        for i in 1..N {
            if responses[i] > max_val {
                max_val = responses[i];
                max_idx = i;
            }
        }

        max_idx
    }
}

impl<const N: usize> Default for PopulationCode<N> {
    fn default() -> Self {
        Self::uniform(100.0, 5.0, 0.15)
    }
}

// ============================================================================
// Spike Pattern
// ============================================================================

/// Spike pattern representation
#[derive(Clone, Debug, PartialEq)]
pub struct SpikePattern<const MAX_SPIKES: usize> {
    /// Spike times in microseconds
    pub spike_times: heapless::Vec<u64, MAX_SPIKES>,
    /// Neuron ID that generated this pattern
    pub neuron_id: u16,
    /// Start time of the pattern window (us)
    pub window_start_us: u64,
    /// End time of the pattern window (us)
    pub window_end_us: u64,
}

impl<const MAX_SPIKES: usize> SpikePattern<MAX_SPIKES> {
    /// Create a new empty spike pattern
    #[inline]
    #[must_use]
    pub fn new(neuron_id: u16, window_start_us: u64, window_end_us: u64) -> Self {
        Self {
            spike_times: heapless::Vec::new(),
            neuron_id,
            window_start_us,
            window_end_us,
        }
    }

    /// Add a spike to the pattern
    #[inline]
    pub fn add_spike(&mut self, time_us: u64) -> Result<(), u64> {
        self.spike_times.push(time_us)
    }

    /// Get the number of spikes
    #[inline]
    #[must_use]
    pub fn spike_count(&self) -> usize {
        self.spike_times.len()
    }

    /// Calculate mean firing rate in Hz
    #[inline]
    #[must_use]
    pub fn mean_rate_hz(&self) -> f32 {
        let duration_s = (self.window_end_us - self.window_start_us) as f32 / 1_000_000.0;
        if duration_s > 0.0 {
            self.spike_count() as f32 / duration_s
        } else {
            0.0
        }
    }

    /// Calculate inter-spike intervals in microseconds
    pub fn inter_spike_intervals(&self) -> heapless::Vec<u64, MAX_SPIKES> {
        let mut isis = heapless::Vec::new();
        if self.spike_times.len() >= 2 {
            for i in 1..self.spike_times.len() {
                let isi = self.spike_times[i] - self.spike_times[i - 1];
                let _ = isis.push(isi);
            }
        }
        isis
    }

    /// Calculate coefficient of variation of ISIs
    #[must_use]
    pub fn isi_cv(&self) -> f32 {
        let isis = self.inter_spike_intervals();
        if isis.len() < 2 {
            return 0.0;
        }

        let mean: f32 = isis.iter().map(|&x| x as f32).sum::<f32>() / isis.len() as f32;
        if mean < 0.001 {
            return 0.0;
        }

        let variance: f32 = isis.iter()
            .map(|&x| {
                let diff = x as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / isis.len() as f32;

        libm::sqrtf(variance) / mean
    }

    /// Get the latency to first spike in microseconds
    #[inline]
    #[must_use]
    pub fn first_spike_latency(&self) -> Option<u64> {
        self.spike_times.first().map(|&t| t - self.window_start_us)
    }
}

impl<const MAX_SPIKES: usize> Default for SpikePattern<MAX_SPIKES> {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

// ============================================================================
// Spike Encoder
// ============================================================================

/// Encoder that converts continuous signals to spike patterns
#[derive(Clone, Debug)]
pub struct SpikeEncoder {
    /// Rate code for baseline conversion
    rate_code: RateCode,
    /// Temporal code for latency encoding
    temporal_code: TemporalCode,
    /// Refractory period in microseconds
    refractory_us: u64,
    /// Last spike time
    last_spike_us: u64,
    /// Accumulated probability (for Poisson spiking)
    accumulated_prob: f32,
}

impl SpikeEncoder {
    /// Create a new spike encoder
    #[inline]
    #[must_use]
    pub fn new(max_rate_hz: f32, refractory_us: u64) -> Self {
        Self {
            rate_code: RateCode::new(0.0, max_rate_hz),
            temporal_code: TemporalCode::default(),
            refractory_us,
            last_spike_us: 0,
            accumulated_prob: 0.0,
        }
    }

    /// Reset the encoder state
    pub fn reset(&mut self) {
        self.last_spike_us = 0;
        self.accumulated_prob = 0.0;
    }

    /// Check if a spike should be generated (Poisson process)
    ///
    /// Returns Some(spike_time) if a spike should be generated, None otherwise.
    pub fn step(&mut self, stimulus: f32, current_time_us: u64, dt_us: u64) -> Option<u64> {
        // Check refractory period
        if current_time_us < self.last_spike_us + self.refractory_us {
            return None;
        }

        // Get instantaneous firing rate
        let rate_hz = self.rate_code.encode(stimulus);

        // Probability of spike in this time step (Poisson)
        let dt_s = dt_us as f32 / 1_000_000.0;
        let prob = rate_hz * dt_s;

        // Accumulate probability
        self.accumulated_prob += prob;

        // Generate spike if accumulated probability exceeds 1
        if self.accumulated_prob >= 1.0 {
            self.accumulated_prob -= 1.0;
            self.last_spike_us = current_time_us;
            Some(current_time_us)
        } else {
            None
        }
    }

    /// Generate a complete spike train for a stimulus time series
    pub fn encode_train<const MAX_SPIKES: usize>(
        &mut self,
        stimuli: &[f32],
        start_time_us: u64,
        sample_interval_us: u64,
    ) -> SpikePattern<MAX_SPIKES> {
        self.reset();

        let end_time_us = start_time_us + (stimuli.len() as u64) * sample_interval_us;
        let mut pattern = SpikePattern::new(0, start_time_us, end_time_us);

        for (i, &stim) in stimuli.iter().enumerate() {
            let time_us = start_time_us + (i as u64) * sample_interval_us;
            if let Some(spike_time) = self.step(stim, time_us, sample_interval_us) {
                let _ = pattern.add_spike(spike_time);
            }
        }

        pattern
    }
}

impl Default for SpikeEncoder {
    fn default() -> Self {
        Self::new(200.0, 2000) // 200 Hz max, 2ms refractory
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_code_encode_decode() {
        let encoder = RateCode::new(5.0, 100.0);

        // Test encoding
        let rate_low = encoder.encode(0.1);
        let rate_mid = encoder.encode(0.5);
        let rate_high = encoder.encode(0.9);

        assert!(rate_low < rate_mid);
        assert!(rate_mid < rate_high);
        assert!(rate_low >= 5.0);
        assert!(rate_high <= 100.0);

        // Test decoding
        let decoded = encoder.decode(rate_mid);
        assert!((decoded - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_temporal_code_latency() {
        let encoder = TemporalCode::new(5.0, 50.0);

        // Strong stimulus = short latency
        let latency_strong = encoder.encode_latency(1.0);
        let latency_weak = encoder.encode_latency(0.0);

        assert!(latency_strong < latency_weak);
        assert!((latency_strong - 5.0).abs() < 0.01);
        assert!((latency_weak - 50.0).abs() < 0.01);

        // Test decode
        let decoded = encoder.decode_latency(27.5);
        assert!((decoded - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_population_code() {
        const N: usize = 8;
        let pop = PopulationCode::<N>::uniform(100.0, 5.0, 0.1);

        // Encode a stimulus
        let responses = pop.encode_population(0.5);

        // Middle neurons should have highest response
        let max_idx = pop.winner_take_all(&responses);
        assert!(max_idx == 3 || max_idx == 4);

        // Decode should recover stimulus
        let decoded = pop.decode_population(&responses);
        assert!((decoded - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_spike_pattern_metrics() {
        let mut pattern: SpikePattern<32> = SpikePattern::new(0, 0, 1_000_000);

        // Add spikes at regular intervals (100 Hz)
        for i in 0..10 {
            pattern.add_spike((i as u64) * 10_000).ok();
        }

        assert_eq!(pattern.spike_count(), 10);
        assert!((pattern.mean_rate_hz() - 10.0).abs() < 0.1);

        let isis = pattern.inter_spike_intervals();
        assert_eq!(isis.len(), 9);
        assert!(isis.iter().all(|&isi| isi == 10_000));
    }

    #[test]
    fn test_spike_encoder() {
        let mut encoder = SpikeEncoder::new(100.0, 2000);

        // High stimulus should produce spikes
        let stimuli = [1.0f32; 100];
        let pattern: SpikePattern<100> = encoder.encode_train(&stimuli, 0, 1000);

        // Should have some spikes
        assert!(pattern.spike_count() > 0);
        assert!(pattern.spike_count() < 100); // Limited by refractory period
    }
}

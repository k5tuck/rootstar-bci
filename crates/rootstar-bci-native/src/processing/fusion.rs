//! EEG + fNIRS data fusion
//!
//! Provides temporal alignment and feature fusion for hybrid BCI.

use std::collections::VecDeque;

use rootstar_bci_core::math::constants;
use rootstar_bci_core::types::{EegSample, HemodynamicSample};

/// Aligned EEG + fNIRS sample pair
#[derive(Clone, Debug)]
pub struct AlignedSample {
    /// Timestamp of the EEG sample
    pub timestamp_us: u64,
    /// EEG sample
    pub eeg: EegSample,
    /// Corresponding hemodynamic sample (accounting for neurovascular delay)
    pub hemodynamic: HemodynamicSample,
}

/// Temporal aligner for EEG and fNIRS data
///
/// Accounts for the neurovascular coupling delay (~5 seconds) between
/// neural activity (measured by EEG) and hemodynamic response (measured by fNIRS).
pub struct TemporalAligner {
    eeg_buffer: VecDeque<EegSample>,
    fnirs_buffer: VecDeque<HemodynamicSample>,
    window_us: u64,
    hemodynamic_lag_us: u64,
    max_buffer_size: usize,
}

impl TemporalAligner {
    /// Create a new temporal aligner
    ///
    /// # Arguments
    ///
    /// * `window_ms` - Alignment window in milliseconds
    /// * `hemodynamic_lag_ms` - Expected hemodynamic delay in milliseconds
    #[must_use]
    pub fn new(window_ms: u32, hemodynamic_lag_ms: u32) -> Self {
        Self {
            eeg_buffer: VecDeque::with_capacity(1000),
            fnirs_buffer: VecDeque::with_capacity(100),
            window_us: u64::from(window_ms) * 1000,
            hemodynamic_lag_us: u64::from(hemodynamic_lag_ms) * 1000,
            max_buffer_size: 10000,
        }
    }

    /// Create with default neurovascular coupling delay (~5 seconds)
    #[must_use]
    pub fn with_default_lag(window_ms: u32) -> Self {
        Self::new(window_ms, constants::HEMODYNAMIC_DELAY_MS)
    }

    /// Add an EEG sample to the buffer
    pub fn push_eeg(&mut self, sample: EegSample) {
        self.eeg_buffer.push_back(sample);
        self.prune_old();
    }

    /// Add an fNIRS/hemodynamic sample to the buffer
    pub fn push_fnirs(&mut self, sample: HemodynamicSample) {
        self.fnirs_buffer.push_back(sample);
        self.prune_old();
    }

    /// Get time-aligned sample pairs
    ///
    /// For each EEG sample, finds the corresponding fNIRS sample
    /// accounting for the hemodynamic delay.
    pub fn get_aligned(&self) -> Vec<AlignedSample> {
        let mut aligned = Vec::new();

        for eeg in &self.eeg_buffer {
            // Find fNIRS sample that corresponds to this EEG sample
            // Account for hemodynamic lag (neural activity precedes blood flow)
            let target_time = eeg.timestamp_us.saturating_sub(self.hemodynamic_lag_us);

            if let Some(fnirs) = self.find_nearest_fnirs(target_time) {
                aligned.push(AlignedSample {
                    timestamp_us: eeg.timestamp_us,
                    eeg: *eeg,
                    hemodynamic: fnirs,
                });
            }
        }

        aligned
    }

    /// Get the most recent aligned sample
    #[must_use]
    pub fn get_latest_aligned(&self) -> Option<AlignedSample> {
        let eeg = self.eeg_buffer.back()?;
        let target_time = eeg.timestamp_us.saturating_sub(self.hemodynamic_lag_us);
        let fnirs = self.find_nearest_fnirs(target_time)?;

        Some(AlignedSample {
            timestamp_us: eeg.timestamp_us,
            eeg: *eeg,
            hemodynamic: fnirs,
        })
    }

    fn find_nearest_fnirs(&self, target_us: u64) -> Option<HemodynamicSample> {
        self.fnirs_buffer
            .iter()
            .min_by_key(|s| (s.timestamp_us as i64 - target_us as i64).unsigned_abs())
            .copied()
    }

    fn prune_old(&mut self) {
        let now = self.eeg_buffer.back()
            .map(|s| s.timestamp_us)
            .unwrap_or(0);

        // Prune EEG buffer
        while let Some(front) = self.eeg_buffer.front() {
            if now - front.timestamp_us > self.window_us || self.eeg_buffer.len() > self.max_buffer_size {
                self.eeg_buffer.pop_front();
            } else {
                break;
            }
        }

        // Prune fNIRS buffer (keep longer due to hemodynamic lag)
        let fnirs_window = self.window_us + self.hemodynamic_lag_us;
        while let Some(front) = self.fnirs_buffer.front() {
            if now - front.timestamp_us > fnirs_window || self.fnirs_buffer.len() > self.max_buffer_size / 10 {
                self.fnirs_buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get EEG buffer size
    #[must_use]
    pub fn eeg_buffer_len(&self) -> usize {
        self.eeg_buffer.len()
    }

    /// Get fNIRS buffer size
    #[must_use]
    pub fn fnirs_buffer_len(&self) -> usize {
        self.fnirs_buffer.len()
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        self.eeg_buffer.clear();
        self.fnirs_buffer.clear();
    }
}

/// Hybrid BCI features combining EEG and fNIRS
#[derive(Clone, Debug, Default)]
pub struct HybridFeatures {
    /// EEG band powers per channel [channel][band]
    pub eeg_band_powers: [[f64; 5]; 8],
    /// fNIRS HbO₂ level (µM)
    pub hbo2_level: f64,
    /// fNIRS HbR level (µM)
    pub hbr_level: f64,
    /// HbO₂ trend (µM/s)
    pub hbo2_slope: f64,
    /// Oxygenation index (0-1)
    pub oxygenation_index: f64,
    /// Frontal alpha asymmetry (R-L)/(R+L)
    pub frontal_asymmetry: f64,
    /// Timestamp
    pub timestamp_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::types::{Fixed24_8, FnirsChannel};

    #[test]
    fn test_temporal_aligner() {
        let mut aligner = TemporalAligner::new(10000, 5000); // 10s window, 5s lag

        // Add EEG samples
        for i in 0..100 {
            aligner.push_eeg(EegSample::new(i * 4000, i as u32)); // 250 Hz
        }

        // Add fNIRS samples (earlier, to account for lag)
        let ch = FnirsChannel::new(0, 0, 30);
        for i in 0..20 {
            let timestamp = i * 100000; // 10 Hz, starting earlier
            let sample = HemodynamicSample::new(
                timestamp,
                ch,
                Fixed24_8::from_f32(1.0),
                Fixed24_8::from_f32(-0.5),
            );
            aligner.push_fnirs(sample);
        }

        // Should be able to get aligned samples
        let aligned = aligner.get_aligned();
        assert!(!aligned.is_empty());
    }
}

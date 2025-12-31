//! EEG + fNIRS + EMG + EDA data fusion
//!
//! Provides temporal alignment and feature fusion for hybrid BCI with
//! multi-modal biosignal integration.

use std::collections::VecDeque;

use rootstar_bci_core::math::constants;
use rootstar_bci_core::types::{EegSample, EdaSample, EmgSample, HemodynamicSample};

/// Aligned multi-modal sample (EEG + fNIRS + EMG + EDA)
#[derive(Clone, Debug)]
pub struct AlignedSample {
    /// Timestamp of the primary (EEG) sample
    pub timestamp_us: u64,
    /// EEG sample
    pub eeg: EegSample,
    /// Corresponding hemodynamic sample (accounting for neurovascular delay)
    pub hemodynamic: HemodynamicSample,
    /// EMG sample (if available)
    pub emg: Option<EmgSample>,
    /// EDA sample (if available)
    pub eda: Option<EdaSample>,
}

/// Temporal aligner for multi-modal biosignal data
///
/// Accounts for the neurovascular coupling delay (~5 seconds) between
/// neural activity (measured by EEG) and hemodynamic response (measured by fNIRS).
/// EMG and EDA are aligned with minimal delay as they respond faster.
pub struct TemporalAligner {
    eeg_buffer: VecDeque<EegSample>,
    fnirs_buffer: VecDeque<HemodynamicSample>,
    emg_buffer: VecDeque<EmgSample>,
    eda_buffer: VecDeque<EdaSample>,
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
            emg_buffer: VecDeque::with_capacity(500),
            eda_buffer: VecDeque::with_capacity(50),
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

    /// Add an EMG sample to the buffer
    pub fn push_emg(&mut self, sample: EmgSample) {
        self.emg_buffer.push_back(sample);
        self.prune_old();
    }

    /// Add an EDA sample to the buffer
    pub fn push_eda(&mut self, sample: EdaSample) {
        self.eda_buffer.push_back(sample);
        self.prune_old();
    }

    /// Get time-aligned sample tuples
    ///
    /// For each EEG sample, finds the corresponding fNIRS, EMG, and EDA samples
    /// accounting for the hemodynamic delay (fNIRS) and minimal delay (EMG/EDA).
    pub fn get_aligned(&self) -> Vec<AlignedSample> {
        let mut aligned = Vec::new();

        for eeg in &self.eeg_buffer {
            // Find fNIRS sample that corresponds to this EEG sample
            // Account for hemodynamic lag (neural activity precedes blood flow)
            let fnirs_target_time = eeg.timestamp_us.saturating_sub(self.hemodynamic_lag_us);

            if let Some(fnirs) = self.find_nearest_fnirs(fnirs_target_time) {
                aligned.push(AlignedSample {
                    timestamp_us: eeg.timestamp_us,
                    eeg: *eeg,
                    hemodynamic: fnirs,
                    emg: self.find_nearest_emg(eeg.timestamp_us),
                    eda: self.find_nearest_eda(eeg.timestamp_us),
                });
            }
        }

        aligned
    }

    /// Get the most recent aligned sample
    #[must_use]
    pub fn get_latest_aligned(&self) -> Option<AlignedSample> {
        let eeg = self.eeg_buffer.back()?;
        let fnirs_target_time = eeg.timestamp_us.saturating_sub(self.hemodynamic_lag_us);
        let fnirs = self.find_nearest_fnirs(fnirs_target_time)?;

        Some(AlignedSample {
            timestamp_us: eeg.timestamp_us,
            eeg: *eeg,
            hemodynamic: fnirs,
            emg: self.find_nearest_emg(eeg.timestamp_us),
            eda: self.find_nearest_eda(eeg.timestamp_us),
        })
    }

    fn find_nearest_fnirs(&self, target_us: u64) -> Option<HemodynamicSample> {
        self.fnirs_buffer
            .iter()
            .min_by_key(|s| (s.timestamp_us as i64 - target_us as i64).unsigned_abs())
            .copied()
    }

    fn find_nearest_emg(&self, target_us: u64) -> Option<EmgSample> {
        const MAX_EMG_OFFSET_US: u64 = 10_000; // 10ms max offset
        self.emg_buffer
            .iter()
            .filter(|s| {
                let diff = (s.timestamp_us as i64 - target_us as i64).unsigned_abs();
                diff < MAX_EMG_OFFSET_US
            })
            .min_by_key(|s| (s.timestamp_us as i64 - target_us as i64).unsigned_abs())
            .cloned()
    }

    fn find_nearest_eda(&self, target_us: u64) -> Option<EdaSample> {
        const MAX_EDA_OFFSET_US: u64 = 100_000; // 100ms max offset (EDA is slower)
        self.eda_buffer
            .iter()
            .filter(|s| {
                let diff = (s.timestamp_us as i64 - target_us as i64).unsigned_abs();
                diff < MAX_EDA_OFFSET_US
            })
            .min_by_key(|s| (s.timestamp_us as i64 - target_us as i64).unsigned_abs())
            .cloned()
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

        // Prune EMG buffer (same window as EEG)
        while let Some(front) = self.emg_buffer.front() {
            if now - front.timestamp_us > self.window_us || self.emg_buffer.len() > self.max_buffer_size / 2 {
                self.emg_buffer.pop_front();
            } else {
                break;
            }
        }

        // Prune EDA buffer (slower rate, keep longer)
        let eda_window = self.window_us * 2; // Double window for slow EDA
        while let Some(front) = self.eda_buffer.front() {
            if now - front.timestamp_us > eda_window || self.eda_buffer.len() > self.max_buffer_size / 20 {
                self.eda_buffer.pop_front();
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

    /// Get EMG buffer size
    #[must_use]
    pub fn emg_buffer_len(&self) -> usize {
        self.emg_buffer.len()
    }

    /// Get EDA buffer size
    #[must_use]
    pub fn eda_buffer_len(&self) -> usize {
        self.eda_buffer.len()
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        self.eeg_buffer.clear();
        self.fnirs_buffer.clear();
        self.emg_buffer.clear();
        self.eda_buffer.clear();
    }
}

/// Hybrid BCI features combining EEG, fNIRS, EMG, and EDA
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

    // EMG features
    /// EMG RMS amplitude per channel (µV)
    pub emg_rms: [f64; 8],
    /// Emotional valence from facial EMG (-1 to 1)
    pub emg_valence: f64,
    /// Emotional arousal from facial EMG (0 to 1)
    pub emg_arousal: f64,

    // EDA features
    /// Skin conductance level per site (µS)
    pub eda_scl: [f64; 4],
    /// Skin conductance response count
    pub eda_scr_count: u32,
    /// Autonomic arousal from EDA (0 to 1)
    pub eda_arousal: f64,

    /// Timestamp
    pub timestamp_us: u64,
}

impl HybridFeatures {
    /// Create new features from aligned sample
    pub fn from_aligned(sample: &AlignedSample) -> Self {
        let mut features = Self::default();
        features.timestamp_us = sample.timestamp_us;

        // Extract fNIRS features
        features.hbo2_level = sample.hemodynamic.delta_hbo2.to_f32() as f64;
        features.hbr_level = sample.hemodynamic.delta_hbr.to_f32() as f64;
        let hbt = features.hbo2_level + features.hbr_level.abs();
        if hbt > 0.0 {
            features.oxygenation_index = features.hbo2_level / hbt;
        }

        // Extract EMG features if available
        if let Some(ref emg) = sample.emg {
            for (i, ch) in emg.channels.iter().enumerate().take(8) {
                features.emg_rms[i] = ch.to_f32().abs() as f64;
            }
            // Compute valence from zygomaticus vs corrugator
            let smile = (features.emg_rms[0] + features.emg_rms[1]) / 2.0;
            let frown = (features.emg_rms[2] + features.emg_rms[3]) / 2.0;
            features.emg_valence = ((smile - frown) / (smile + frown + 1.0)).clamp(-1.0, 1.0);
            features.emg_arousal = (features.emg_rms.iter().sum::<f64>() / 800.0).clamp(0.0, 1.0);
        }

        // Extract EDA features if available
        if let Some(ref eda) = sample.eda {
            for (i, conductance) in eda.conductance.iter().enumerate().take(4) {
                features.eda_scl[i] = conductance.to_f32() as f64;
            }
            features.eda_arousal = (features.eda_scl.iter().sum::<f64>() / 40.0).clamp(0.0, 1.0);
        }

        features
    }

    /// Combined arousal score from EMG and EDA (0-1)
    pub fn combined_arousal(&self) -> f64 {
        // Weight EMG and EDA equally
        (self.emg_arousal + self.eda_arousal) / 2.0
    }

    /// Overall emotional state description
    pub fn emotional_state(&self) -> &'static str {
        match (self.emg_valence, self.combined_arousal()) {
            (v, a) if v > 0.3 && a > 0.5 => "excited/happy",
            (v, a) if v > 0.3 && a <= 0.5 => "calm/content",
            (v, a) if v < -0.3 && a > 0.5 => "stressed/angry",
            (v, a) if v < -0.3 && a <= 0.5 => "sad/tired",
            (_, a) if a > 0.5 => "alert",
            _ => "neutral",
        }
    }
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

//! Multimodal fusion for neural fingerprint generation.
//!
//! Combines EEG and fNIRS features into unified neural fingerprints.

use rootstar_bci_core::fingerprint::{
    FingerprintId, FingerprintMetadata, FrequencyBand, NeuralFingerprint, QualityMetrics,
    SensoryModality,
};
use rootstar_bci_core::types::Fixed24_8;

use super::extractor::{EegFeatures, FnirsFeatures};

// ============================================================================
// Fusion Configuration
// ============================================================================

/// Configuration for fingerprint fusion.
#[derive(Clone, Debug)]
pub struct FusionConfig {
    /// EEG sampling rate (Hz)
    pub eeg_sample_rate: f64,
    /// fNIRS sampling rate (Hz)
    pub fnirs_sample_rate: f64,
    /// Hemodynamic response lag range (seconds)
    pub hrf_lag_range: (f64, f64),
    /// Minimum signal quality for acceptance (0-1)
    pub min_quality: f64,
    /// Weight for EEG features in combined embedding
    pub eeg_weight: f64,
    /// Weight for fNIRS features in combined embedding
    pub fnirs_weight: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            eeg_sample_rate: 500.0,
            fnirs_sample_rate: 25.0,
            hrf_lag_range: (-5.0, 15.0), // Typical HRF peaks at 4-6s
            min_quality: 0.5,
            eeg_weight: 0.7,
            fnirs_weight: 0.3,
        }
    }
}

// ============================================================================
// Temporal Alignment
// ============================================================================

/// Aligned multimodal data sample.
#[derive(Clone, Debug)]
pub struct AlignedData {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// EEG features at this timepoint
    pub eeg: EegFeatures,
    /// fNIRS features (upsampled to EEG rate)
    pub fnirs: FnirsFeatures,
    /// Neurovascular coupling strength per channel
    pub nv_coupling: Vec<f64>,
    /// Neurovascular coupling lag per channel (ms)
    pub nv_lag_ms: Vec<i16>,
}

/// Temporal aligner for EEG and fNIRS data streams.
pub struct TemporalAligner {
    config: FusionConfig,
    eeg_buffer: Vec<(u64, EegFeatures)>,
    fnirs_buffer: Vec<(u64, FnirsFeatures)>,
}

impl TemporalAligner {
    /// Create a new temporal aligner.
    #[must_use]
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            eeg_buffer: Vec::new(),
            fnirs_buffer: Vec::new(),
        }
    }

    /// Add EEG features with timestamp.
    pub fn add_eeg(&mut self, timestamp_us: u64, features: EegFeatures) {
        self.eeg_buffer.push((timestamp_us, features));

        // Keep buffer size reasonable (last 30 seconds)
        let cutoff = timestamp_us.saturating_sub(30_000_000);
        self.eeg_buffer.retain(|(t, _)| *t >= cutoff);
    }

    /// Add fNIRS features with timestamp.
    pub fn add_fnirs(&mut self, timestamp_us: u64, features: FnirsFeatures) {
        self.fnirs_buffer.push((timestamp_us, features));

        // Keep buffer size reasonable
        let cutoff = timestamp_us.saturating_sub(30_000_000);
        self.fnirs_buffer.retain(|(t, _)| *t >= cutoff);
    }

    /// Get aligned data at a specific timestamp.
    ///
    /// Interpolates fNIRS to match EEG timestamp.
    pub fn get_aligned(&self, timestamp_us: u64) -> Option<AlignedData> {
        // Find nearest EEG sample
        let eeg = self
            .eeg_buffer
            .iter()
            .min_by_key(|(t, _)| (*t as i64 - timestamp_us as i64).unsigned_abs())?;

        // Find nearest fNIRS samples for interpolation
        let fnirs_before = self
            .fnirs_buffer
            .iter()
            .filter(|(t, _)| *t <= timestamp_us)
            .max_by_key(|(t, _)| *t);

        let fnirs_after = self
            .fnirs_buffer
            .iter()
            .filter(|(t, _)| *t > timestamp_us)
            .min_by_key(|(t, _)| *t);

        // Use nearest fNIRS sample (or interpolate)
        let fnirs = match (fnirs_before, fnirs_after) {
            (Some((_, f)), _) => f.clone(),
            (None, Some((_, f))) => f.clone(),
            (None, None) => return None,
        };

        // Compute neurovascular coupling
        let n_channels = fnirs.n_channels;
        let nv_coupling = vec![0.0; n_channels]; // Placeholder - would compute from buffers
        let nv_lag_ms = vec![0i16; n_channels];

        Some(AlignedData {
            timestamp_us,
            eeg: eeg.1.clone(),
            fnirs,
            nv_coupling,
            nv_lag_ms,
        })
    }
}

// ============================================================================
// Fingerprint Fusion
// ============================================================================

/// Fuses multimodal data into neural fingerprints.
pub struct FingerprintFusion {
    config: FusionConfig,
    id_counter: u64,
}

impl FingerprintFusion {
    /// Create a new fingerprint fusion processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: FusionConfig::default(),
            id_counter: 0,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: FusionConfig) -> Self {
        Self { config, id_counter: 0 }
    }

    /// Generate a neural fingerprint from features.
    pub fn generate(
        &mut self,
        eeg: &EegFeatures,
        fnirs: &FnirsFeatures,
        modality: SensoryModality,
        stimulus_label: &str,
        subject_id: &str,
        timestamp_us: u64,
    ) -> NeuralFingerprint {
        // Generate fingerprint ID
        let id = self.generate_id(timestamp_us);

        // Create metadata
        let mut metadata = FingerprintMetadata::new(
            id,
            modality,
            stimulus_label,
            subject_id,
            timestamp_us,
        );
        metadata.eeg_channels = eeg.n_channels as u16;
        metadata.fnirs_channels = fnirs.n_channels as u16;

        // Compute quality metrics
        metadata.quality = self.compute_quality(eeg, fnirs);
        let overall_score = metadata.quality.overall_score;

        // Create fingerprint
        let mut fingerprint = NeuralFingerprint::new(metadata);

        // Copy EEG band power
        for (ch, bands) in eeg.band_power.iter().enumerate() {
            for (band_idx, &power) in bands.iter().enumerate() {
                if let Some(band) = FrequencyBand::CANONICAL
                    .iter()
                    .chain(FrequencyBand::SENSORY_FOCUS.iter())
                    .nth(band_idx)
                {
                    fingerprint.set_band_power(ch, *band, Fixed24_8::from_f32(power as f32));
                }
            }
        }

        // Copy topography
        for &t in &eeg.topography {
            let _ = fingerprint.eeg_topography.push(Fixed24_8::from_f32(t as f32));
        }

        // Copy entropy
        for &e in &eeg.entropy {
            let _ = fingerprint.eeg_entropy.push(Fixed24_8::from_f32(e as f32));
        }

        // Copy fNIRS activation
        for &h in &fnirs.hbo_activation {
            let _ = fingerprint.fnirs_hbo_activation.push(Fixed24_8::from_f32(h as f32));
        }
        for &h in &fnirs.hbr_activation {
            let _ = fingerprint.fnirs_hbr_activation.push(Fixed24_8::from_f32(h as f32));
        }

        // Set confidence based on quality
        fingerprint.set_confidence_f32(overall_score as f32 / 100.0);

        fingerprint
    }

    /// Generate a fingerprint from aligned data.
    pub fn generate_from_aligned(
        &mut self,
        aligned: &AlignedData,
        modality: SensoryModality,
        stimulus_label: &str,
        subject_id: &str,
    ) -> NeuralFingerprint {
        let mut fp = self.generate(
            &aligned.eeg,
            &aligned.fnirs,
            modality,
            stimulus_label,
            subject_id,
            aligned.timestamp_us,
        );

        // Add neurovascular coupling lag
        for &lag in &aligned.nv_lag_ms {
            let _ = fp.nv_coupling_lag_ms.push(lag);
        }

        fp
    }

    /// Average multiple fingerprints of the same stimulus.
    ///
    /// Used for creating robust reference fingerprints from multiple trials.
    pub fn average(&mut self, fingerprints: &[NeuralFingerprint]) -> Option<NeuralFingerprint> {
        if fingerprints.is_empty() {
            return None;
        }

        let first = &fingerprints[0];
        let n = fingerprints.len() as f32;

        // Create averaged metadata
        let id = self.generate_id(first.metadata.timestamp_us);
        let mut metadata = FingerprintMetadata::new(
            id,
            first.metadata.modality,
            first.metadata.stimulus_label(),
            first.metadata.subject_id(),
            first.metadata.timestamp_us,
        );
        metadata.eeg_channels = first.metadata.eeg_channels;
        metadata.fnirs_channels = first.metadata.fnirs_channels;

        // Average quality scores
        let avg_quality: f32 = fingerprints
            .iter()
            .map(|fp| fp.metadata.quality.overall_score as f32)
            .sum::<f32>()
            / n;
        metadata.quality.overall_score = avg_quality as u8;

        let mut result = NeuralFingerprint::new(metadata);

        // Average band power
        let bp_len = fingerprints
            .iter()
            .map(|fp| fp.eeg_band_power.len())
            .min()
            .unwrap_or(0);

        for i in 0..bp_len {
            let avg: f32 = fingerprints
                .iter()
                .map(|fp| fp.eeg_band_power.get(i).map_or(0.0, |f| f.to_f32()))
                .sum::<f32>()
                / n;
            let _ = result.eeg_band_power.push(Fixed24_8::from_f32(avg));
        }

        // Average topography
        let topo_len = fingerprints
            .iter()
            .map(|fp| fp.eeg_topography.len())
            .min()
            .unwrap_or(0);

        for i in 0..topo_len {
            let avg: f32 = fingerprints
                .iter()
                .map(|fp| fp.eeg_topography.get(i).map_or(0.0, |f| f.to_f32()))
                .sum::<f32>()
                / n;
            let _ = result.eeg_topography.push(Fixed24_8::from_f32(avg));
        }

        // Average entropy
        let ent_len = fingerprints
            .iter()
            .map(|fp| fp.eeg_entropy.len())
            .min()
            .unwrap_or(0);

        for i in 0..ent_len {
            let avg: f32 = fingerprints
                .iter()
                .map(|fp| fp.eeg_entropy.get(i).map_or(0.0, |f| f.to_f32()))
                .sum::<f32>()
                / n;
            let _ = result.eeg_entropy.push(Fixed24_8::from_f32(avg));
        }

        // Average fNIRS activation
        let hbo_len = fingerprints
            .iter()
            .map(|fp| fp.fnirs_hbo_activation.len())
            .min()
            .unwrap_or(0);

        for i in 0..hbo_len {
            let avg: f32 = fingerprints
                .iter()
                .map(|fp| fp.fnirs_hbo_activation.get(i).map_or(0.0, |f| f.to_f32()))
                .sum::<f32>()
                / n;
            let _ = result.fnirs_hbo_activation.push(Fixed24_8::from_f32(avg));
        }

        let hbr_len = fingerprints
            .iter()
            .map(|fp| fp.fnirs_hbr_activation.len())
            .min()
            .unwrap_or(0);

        for i in 0..hbr_len {
            let avg: f32 = fingerprints
                .iter()
                .map(|fp| fp.fnirs_hbr_activation.get(i).map_or(0.0, |f| f.to_f32()))
                .sum::<f32>()
                / n;
            let _ = result.fnirs_hbr_activation.push(Fixed24_8::from_f32(avg));
        }

        // Compute confidence based on consistency across trials
        let mut similarity_sum = 0.0;
        for i in 0..fingerprints.len() {
            for j in (i + 1)..fingerprints.len() {
                similarity_sum += fingerprints[i].similarity(&fingerprints[j]);
            }
        }
        let n_pairs = (fingerprints.len() * (fingerprints.len() - 1) / 2) as f32;
        let avg_similarity = if n_pairs > 0.0 {
            similarity_sum / n_pairs
        } else {
            1.0
        };

        result.set_confidence_f32(avg_similarity as f32);

        Some(result)
    }

    fn generate_id(&mut self, timestamp_us: u64) -> FingerprintId {
        self.id_counter += 1;

        let mut bytes = [0u8; 16];

        // Use timestamp and counter for uniqueness
        bytes[0..8].copy_from_slice(&timestamp_us.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.id_counter.to_le_bytes());

        FingerprintId::from_bytes(bytes)
    }

    fn compute_quality(&self, eeg: &EegFeatures, fnirs: &FnirsFeatures) -> QualityMetrics {
        let mut quality = QualityMetrics::unknown();

        // Estimate SNR from topography variance
        let topo_mean: f64 = eeg.topography.iter().sum::<f64>() / eeg.topography.len() as f64;
        let topo_var: f64 = eeg
            .topography
            .iter()
            .map(|t| (t - topo_mean).powi(2))
            .sum::<f64>()
            / eeg.topography.len() as f64;

        if topo_var > 0.0 {
            let snr_db = 10.0 * (topo_mean.powi(2) / topo_var).log10();
            quality.snr_db = Fixed24_8::from_f32(snr_db.clamp(-20.0, 60.0) as f32);
        }

        // Count channels with reasonable signal
        let active_channels = eeg
            .topography
            .iter()
            .filter(|&&t| t > 0.05 && t < 0.95)
            .count();
        let active_ratio = active_channels as f64 / eeg.n_channels as f64;

        // fNIRS contribution
        let fnirs_active = fnirs
            .hbo_activation
            .iter()
            .filter(|&&h| h.abs() > 0.001)
            .count();
        let fnirs_ratio = fnirs_active as f64 / fnirs.n_channels.max(1) as f64;

        // Combined quality score
        let score = ((active_ratio * 0.6 + fnirs_ratio * 0.4) * 100.0) as u8;
        quality.overall_score = score.max(10); // Minimum 10%
        quality.artifact_free_pct = (active_ratio * 100.0) as u8;

        quality
    }
}

impl Default for FingerprintFusion {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_eeg_features() -> EegFeatures {
        let mut features = EegFeatures::new(8);
        for ch in 0..8 {
            for band in 0..FrequencyBand::COUNT {
                features.band_power[ch][band] = (ch + band) as f64 * 0.1;
            }
            features.topography[ch] = 0.5 + ch as f64 * 0.05;
            features.entropy[ch] = 0.3 + ch as f64 * 0.02;
        }
        features
    }

    fn create_test_fnirs_features() -> FnirsFeatures {
        let mut features = FnirsFeatures::new(4);
        for ch in 0..4 {
            features.hbo_activation[ch] = 0.1 * (ch + 1) as f64;
            features.hbr_activation[ch] = -0.03 * (ch + 1) as f64;
            features.hbt_activation[ch] = features.hbo_activation[ch] + features.hbr_activation[ch];
        }
        features
    }

    #[test]
    fn test_fusion_generate() {
        let mut fusion = FingerprintFusion::new();
        let eeg = create_test_eeg_features();
        let fnirs = create_test_fnirs_features();

        let fp = fusion.generate(
            &eeg,
            &fnirs,
            SensoryModality::Gustatory,
            "apple_taste",
            "subject01",
            1_000_000,
        );

        assert_eq!(fp.metadata.modality, SensoryModality::Gustatory);
        assert_eq!(fp.metadata.stimulus_label(), "apple_taste");
        assert_eq!(fp.metadata.eeg_channels, 8);
        assert_eq!(fp.metadata.fnirs_channels, 4);
        assert!(!fp.eeg_band_power.is_empty());
        assert!(!fp.fnirs_hbo_activation.is_empty());
    }

    #[test]
    fn test_fusion_average() {
        let mut fusion = FingerprintFusion::new();
        let eeg = create_test_eeg_features();
        let fnirs = create_test_fnirs_features();

        // Create multiple similar fingerprints
        let fps: Vec<_> = (0..5)
            .map(|i| {
                fusion.generate(
                    &eeg,
                    &fnirs,
                    SensoryModality::Olfactory,
                    "rose_smell",
                    "subject01",
                    i * 1_000_000,
                )
            })
            .collect();

        let avg = fusion.average(&fps);
        assert!(avg.is_some());

        let avg = avg.unwrap();
        assert_eq!(avg.metadata.modality, SensoryModality::Olfactory);
        assert!(avg.confidence_f32() > 0.9); // Similar fingerprints should have high consistency
    }

    #[test]
    fn test_temporal_aligner() {
        let config = FusionConfig::default();
        let mut aligner = TemporalAligner::new(config);

        // Add EEG at 500 Hz
        for i in 0..100 {
            let ts = i * 2000; // 2ms intervals
            aligner.add_eeg(ts, create_test_eeg_features());
        }

        // Add fNIRS at 25 Hz
        for i in 0..5 {
            let ts = i * 40000; // 40ms intervals
            aligner.add_fnirs(ts, create_test_fnirs_features());
        }

        // Get aligned at a middle timestamp
        let aligned = aligner.get_aligned(100_000);
        assert!(aligned.is_some());
    }
}

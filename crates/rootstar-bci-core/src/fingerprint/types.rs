//! Core types for Neural Fingerprint system.
//!
//! This module defines the fundamental data structures for capturing,
//! storing, and comparing neural fingerprints across sensory modalities.

use core::fmt;

use heapless::Vec as HeaplessVec;
use serde::{Deserialize, Serialize};

use crate::types::Fixed24_8;

// ============================================================================
// Sensory Modality
// ============================================================================

/// Sensory modality classification for neural fingerprints.
///
/// Each modality has characteristic neural patterns in specific brain regions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    /// Gustatory (taste) - frontal-insular region
    Gustatory,
    /// Olfactory (smell) - orbitofrontal region
    Olfactory,
    /// Somatosensory (touch) - parietal cortex
    Tactile,
    /// Auditory (sound) - temporal cortex
    Auditory,
    /// Visual (sight) - occipital cortex
    Visual,
}

impl SensoryModality {
    /// Get the primary cortical region for this modality.
    #[inline]
    #[must_use]
    pub const fn primary_region(self) -> &'static str {
        match self {
            Self::Gustatory => "Frontal-Insular (F7, F8, FT7, FT8)",
            Self::Olfactory => "Orbitofrontal (Fp1, Fp2, AF7, AF8)",
            Self::Tactile => "Somatosensory (C3, C4, CP3, CP4)",
            Self::Auditory => "Temporal (T7, T8, TP7, TP8)",
            Self::Visual => "Occipital (O1, O2, PO7, PO8)",
        }
    }

    /// Get the characteristic EEG frequency band for this modality.
    #[inline]
    #[must_use]
    pub const fn characteristic_band(self) -> FrequencyBand {
        match self {
            Self::Gustatory => FrequencyBand::GustatoryGamma,
            Self::Olfactory => FrequencyBand::OlfactoryTheta,
            Self::Tactile => FrequencyBand::Beta,
            Self::Auditory => FrequencyBand::Gamma,
            Self::Visual => FrequencyBand::Alpha,
        }
    }

    /// Get the modality name.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gustatory => "gustatory",
            Self::Olfactory => "olfactory",
            Self::Tactile => "tactile",
            Self::Auditory => "auditory",
            Self::Visual => "visual",
        }
    }
}

impl fmt::Display for SensoryModality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SensoryModality {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

// ============================================================================
// Frequency Bands
// ============================================================================

/// EEG frequency bands for neural fingerprint extraction.
///
/// Includes both canonical bands and sensory-specific sub-bands.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrequencyBand {
    /// Delta: 0.5-4 Hz (deep sleep, unconscious processing)
    Delta,
    /// Theta: 4-8 Hz (memory, emotional processing)
    Theta,
    /// Alpha: 8-13 Hz (relaxed awareness, inhibition)
    Alpha,
    /// Beta: 13-30 Hz (active thinking, motor planning)
    Beta,
    /// Gamma: 30-100 Hz (sensory binding, perception)
    Gamma,
    /// High Gamma: 70-150 Hz (fine sensory discrimination)
    HighGamma,
    /// Gustatory Theta: 4-7 Hz (taste memory retrieval)
    GustatoryTheta,
    /// Gustatory Gamma: 35-45 Hz (taste discrimination)
    GustatoryGamma,
    /// Olfactory Theta: 4-8 Hz (odor identification)
    OlfactoryTheta,
    /// Olfactory Gamma: 40-80 Hz (odor-object binding)
    OlfactoryGamma,
}

impl FrequencyBand {
    /// All canonical frequency bands.
    pub const CANONICAL: [Self; 6] = [
        Self::Delta,
        Self::Theta,
        Self::Alpha,
        Self::Beta,
        Self::Gamma,
        Self::HighGamma,
    ];

    /// Sensory-specific frequency bands.
    pub const SENSORY_FOCUS: [Self; 4] = [
        Self::GustatoryTheta,
        Self::GustatoryGamma,
        Self::OlfactoryTheta,
        Self::OlfactoryGamma,
    ];

    /// Get the frequency range (low, high) in Hz.
    #[inline]
    #[must_use]
    pub const fn range_hz(self) -> (f32, f32) {
        match self {
            Self::Delta => (0.5, 4.0),
            Self::Theta => (4.0, 8.0),
            Self::Alpha => (8.0, 13.0),
            Self::Beta => (13.0, 30.0),
            Self::Gamma => (30.0, 100.0),
            Self::HighGamma => (70.0, 150.0),
            Self::GustatoryTheta => (4.0, 7.0),
            Self::GustatoryGamma => (35.0, 45.0),
            Self::OlfactoryTheta => (4.0, 8.0),
            Self::OlfactoryGamma => (40.0, 80.0),
        }
    }

    /// Get the band name.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Delta => "delta",
            Self::Theta => "theta",
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Gamma => "gamma",
            Self::HighGamma => "high_gamma",
            Self::GustatoryTheta => "gustatory_theta",
            Self::GustatoryGamma => "gustatory_gamma",
            Self::OlfactoryTheta => "olfactory_theta",
            Self::OlfactoryGamma => "olfactory_gamma",
        }
    }

    /// Get the index for array storage.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Delta => 0,
            Self::Theta => 1,
            Self::Alpha => 2,
            Self::Beta => 3,
            Self::Gamma => 4,
            Self::HighGamma => 5,
            Self::GustatoryTheta => 6,
            Self::GustatoryGamma => 7,
            Self::OlfactoryTheta => 8,
            Self::OlfactoryGamma => 9,
        }
    }

    /// Total number of frequency bands.
    pub const COUNT: usize = 10;
}

#[cfg(feature = "defmt")]
impl defmt::Format for FrequencyBand {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

// ============================================================================
// Electrode Positions
// ============================================================================

/// Extended 10-20 system electrode positions for high-density EEG.
///
/// Supports 8-channel basic montage up to 256-channel high-density arrays.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElectrodePosition {
    /// Electrode name (e.g., "Fp1", "F7", "FT7")
    name: [u8; 4],
    /// X coordinate in normalized head space (-1 to 1)
    pub x: i8,
    /// Y coordinate in normalized head space (-1 to 1)
    pub y: i8,
    /// Z coordinate in normalized head space (-1 to 1)
    pub z: i8,
    /// Channel index in hardware array
    pub channel_index: u16,
}

impl ElectrodePosition {
    /// Create a new electrode position.
    #[inline]
    #[must_use]
    pub const fn new(name: &[u8; 4], x: i8, y: i8, z: i8, channel_index: u16) -> Self {
        Self { name: *name, x, y, z, channel_index }
    }

    /// Get the electrode name as a string slice.
    #[inline]
    pub fn name(&self) -> &str {
        let len = self.name.iter().position(|&b| b == 0).unwrap_or(4);
        core::str::from_utf8(&self.name[..len]).unwrap_or("")
    }

    /// Calculate Euclidean distance to another electrode.
    #[inline]
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = (self.x - other.x) as f32;
        let dy = (self.y - other.y) as f32;
        let dz = (self.z - other.z) as f32;
        libm::sqrtf(dx * dx + dy * dy + dz * dz)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for ElectrodePosition {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}[{}]", self.name(), self.channel_index);
    }
}

// ============================================================================
// Fingerprint Identification
// ============================================================================

/// Unique identifier for a neural fingerprint.
///
/// 16-byte UUID-like identifier for database storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FingerprintId([u8; 16]);

impl FingerprintId {
    /// Create a fingerprint ID from raw bytes.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Create a null/empty fingerprint ID.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self([0u8; 16])
    }

    /// Check if this is a null ID.
    #[inline]
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl Default for FingerprintId {
    fn default() -> Self {
        Self::null()
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for FingerprintId {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}...",
            self.0[0],
            self.0[1],
            self.0[2],
            self.0[3]
        );
    }
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Quality metrics for a neural fingerprint recording.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB (Q8.8 fixed point)
    pub snr_db: Fixed24_8,
    /// Electrode impedance quality (0-100%)
    pub impedance_quality: u8,
    /// Percentage of artifact-free samples (0-100%)
    pub artifact_free_pct: u8,
    /// Number of bad channels detected
    pub bad_channels: u8,
    /// Overall quality score (0-100%)
    pub overall_score: u8,
}

impl QualityMetrics {
    /// Minimum acceptable quality score for storage.
    pub const MIN_QUALITY: u8 = 50;

    /// Create quality metrics with default (unknown) values.
    #[inline]
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            snr_db: Fixed24_8::ZERO,
            impedance_quality: 0,
            artifact_free_pct: 0,
            bad_channels: 0,
            overall_score: 0,
        }
    }

    /// Check if quality meets minimum threshold.
    #[inline]
    #[must_use]
    pub const fn is_acceptable(&self) -> bool {
        self.overall_score >= Self::MIN_QUALITY
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::unknown()
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for QualityMetrics {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "Quality({}%)", self.overall_score);
    }
}

// ============================================================================
// Fingerprint Metadata
// ============================================================================

/// Metadata for a neural fingerprint recording.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintMetadata {
    /// Unique fingerprint identifier
    pub id: FingerprintId,
    /// Sensory modality (taste, smell, etc.)
    pub modality: SensoryModality,
    /// Stimulus label (e.g., "apple_taste", "rose_smell")
    /// Stored as fixed-size array for no_std compatibility
    stimulus_label: [u8; 32],
    stimulus_label_len: u8,
    /// Subject identifier
    subject_id: [u8; 16],
    subject_id_len: u8,
    /// Timestamp in microseconds since epoch
    pub timestamp_us: u64,
    /// Quality metrics
    pub quality: QualityMetrics,
    /// Number of EEG channels used
    pub eeg_channels: u16,
    /// Number of fNIRS channels used
    pub fnirs_channels: u16,
}

impl FingerprintMetadata {
    /// Create new metadata with given parameters.
    #[must_use]
    pub fn new(
        id: FingerprintId,
        modality: SensoryModality,
        stimulus_label: &str,
        subject_id: &str,
        timestamp_us: u64,
    ) -> Self {
        let mut label = [0u8; 32];
        let label_bytes = stimulus_label.as_bytes();
        let label_len = label_bytes.len().min(32);
        label[..label_len].copy_from_slice(&label_bytes[..label_len]);

        let mut subject = [0u8; 16];
        let subject_bytes = subject_id.as_bytes();
        let subject_len = subject_bytes.len().min(16);
        subject[..subject_len].copy_from_slice(&subject_bytes[..subject_len]);

        Self {
            id,
            modality,
            stimulus_label: label,
            stimulus_label_len: label_len as u8,
            subject_id: subject,
            subject_id_len: subject_len as u8,
            timestamp_us,
            quality: QualityMetrics::unknown(),
            eeg_channels: 8,
            fnirs_channels: 4,
        }
    }

    /// Get stimulus label as string slice.
    #[inline]
    pub fn stimulus_label(&self) -> &str {
        core::str::from_utf8(&self.stimulus_label[..self.stimulus_label_len as usize])
            .unwrap_or("")
    }

    /// Get subject ID as string slice.
    #[inline]
    pub fn subject_id(&self) -> &str {
        core::str::from_utf8(&self.subject_id[..self.subject_id_len as usize]).unwrap_or("")
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for FingerprintMetadata {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "Meta({}, {})", self.modality, self.stimulus_label());
    }
}

// ============================================================================
// Neural Fingerprint
// ============================================================================

/// Maximum number of EEG channels in compact representation.
const MAX_EEG_CHANNELS: usize = 256;

/// Maximum number of fNIRS channels in compact representation.
const MAX_FNIRS_CHANNELS: usize = 128;

/// Maximum number of EMG channels
const MAX_EMG_CHANNELS: usize = 8;

/// Maximum number of EDA sites
const MAX_EDA_SITES: usize = 4;

/// Complete neural fingerprint for a specific sensory experience.
///
/// This is a compact representation suitable for embedded storage.
/// For high-density configurations, use streaming or chunked transfer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralFingerprint {
    /// Fingerprint metadata
    pub metadata: FingerprintMetadata,

    /// EEG band power per channel: [channel][band]
    /// Stored as flat array for no_std: channel * N_BANDS + band
    /// Values in dB (Q24.8 fixed point)
    pub eeg_band_power: HeaplessVec<Fixed24_8, 2560>, // 256 channels × 10 bands

    /// EEG topography (RMS per channel, normalized)
    pub eeg_topography: HeaplessVec<Fixed24_8, 256>,

    /// EEG entropy per channel (sample entropy)
    pub eeg_entropy: HeaplessVec<Fixed24_8, 256>,

    /// fNIRS HbO spatial activation pattern
    pub fnirs_hbo_activation: HeaplessVec<Fixed24_8, 128>,

    /// fNIRS HbR spatial activation pattern
    pub fnirs_hbr_activation: HeaplessVec<Fixed24_8, 128>,

    /// Neurovascular coupling lag in milliseconds per channel
    pub nv_coupling_lag_ms: HeaplessVec<i16, 128>,

    // ========================================================================
    // EMG Features (Facial Muscle Activity)
    // ========================================================================

    /// EMG RMS activation per channel (µV)
    /// 8 channels: Zygomaticus (L/R), Corrugator (L/R), Masseter (L/R), Orbicularis (U/D)
    pub emg_rms_activation: HeaplessVec<Fixed24_8, 8>,

    /// EMG mean frequency per channel (Hz)
    /// Higher frequency indicates more intense contraction
    pub emg_mean_frequency: HeaplessVec<Fixed24_8, 8>,

    /// EMG emotional valence score (-1.0 to 1.0)
    /// Computed from smile vs frown muscle activity ratio
    /// Positive = pleasure, Negative = displeasure
    pub emg_valence_score: Fixed24_8,

    /// EMG arousal score (0.0 to 1.0)
    /// Overall facial muscle activation level
    pub emg_arousal_score: Fixed24_8,

    // ========================================================================
    // EDA Features (Electrodermal Activity / Skin Conductance)
    // ========================================================================

    /// EDA skin conductance level (SCL) per site in µS
    /// Tonic level representing baseline arousal
    pub eda_scl: HeaplessVec<Fixed24_8, 4>,

    /// EDA skin conductance response (SCR) count per site
    /// Number of phasic responses detected during recording
    pub eda_scr_count: HeaplessVec<u8, 4>,

    /// EDA mean SCR amplitude per site (µS)
    /// Average amplitude of detected responses
    pub eda_scr_amplitude: HeaplessVec<Fixed24_8, 4>,

    /// EDA autonomic arousal score (0.0 to 1.0)
    /// Derived from SCL and SCR metrics
    pub eda_arousal_score: Fixed24_8,

    /// Confidence score (0-255 mapped to 0.0-1.0)
    pub confidence: u8,
}

impl NeuralFingerprint {
    /// Create an empty fingerprint with given metadata.
    #[must_use]
    pub fn new(metadata: FingerprintMetadata) -> Self {
        Self {
            metadata,
            eeg_band_power: HeaplessVec::new(),
            eeg_topography: HeaplessVec::new(),
            eeg_entropy: HeaplessVec::new(),
            fnirs_hbo_activation: HeaplessVec::new(),
            fnirs_hbr_activation: HeaplessVec::new(),
            nv_coupling_lag_ms: HeaplessVec::new(),
            // EMG features
            emg_rms_activation: HeaplessVec::new(),
            emg_mean_frequency: HeaplessVec::new(),
            emg_valence_score: Fixed24_8::ZERO,
            emg_arousal_score: Fixed24_8::ZERO,
            // EDA features
            eda_scl: HeaplessVec::new(),
            eda_scr_count: HeaplessVec::new(),
            eda_scr_amplitude: HeaplessVec::new(),
            eda_arousal_score: Fixed24_8::ZERO,
            confidence: 0,
        }
    }

    /// Get band power for a specific channel and band.
    ///
    /// Returns None if indices are out of range.
    #[inline]
    pub fn get_band_power(&self, channel: usize, band: FrequencyBand) -> Option<Fixed24_8> {
        let n_channels = self.metadata.eeg_channels as usize;
        if channel >= n_channels {
            return None;
        }
        let idx = channel * FrequencyBand::COUNT + band.index();
        self.eeg_band_power.get(idx).copied()
    }

    /// Set band power for a specific channel and band.
    ///
    /// Returns false if indices are out of range.
    #[inline]
    pub fn set_band_power(&mut self, channel: usize, band: FrequencyBand, value: Fixed24_8) -> bool {
        let n_channels = self.metadata.eeg_channels as usize;
        if channel >= n_channels {
            return false;
        }
        let idx = channel * FrequencyBand::COUNT + band.index();

        // Ensure vector is large enough
        while self.eeg_band_power.len() <= idx {
            if self.eeg_band_power.push(Fixed24_8::ZERO).is_err() {
                return false;
            }
        }

        self.eeg_band_power[idx] = value;
        true
    }

    /// Compute cosine similarity with another fingerprint.
    ///
    /// Returns a value between -1.0 and 1.0.
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Compare band power vectors
        let len = self.eeg_band_power.len().min(other.eeg_band_power.len());
        for i in 0..len {
            let a = self.eeg_band_power[i].to_f32();
            let b = other.eeg_band_power[i].to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare topography
        let len = self.eeg_topography.len().min(other.eeg_topography.len());
        for i in 0..len {
            let a = self.eeg_topography[i].to_f32();
            let b = other.eeg_topography[i].to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare fNIRS activation
        let len = self.fnirs_hbo_activation.len().min(other.fnirs_hbo_activation.len());
        for i in 0..len {
            let a = self.fnirs_hbo_activation[i].to_f32();
            let b = other.fnirs_hbo_activation[i].to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare EMG RMS activation
        let len = self.emg_rms_activation.len().min(other.emg_rms_activation.len());
        for i in 0..len {
            let a = self.emg_rms_activation[i].to_f32();
            let b = other.emg_rms_activation[i].to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare EMG valence and arousal scores
        {
            let a = self.emg_valence_score.to_f32();
            let b = other.emg_valence_score.to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        {
            let a = self.emg_arousal_score.to_f32();
            let b = other.emg_arousal_score.to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare EDA SCL
        let len = self.eda_scl.len().min(other.eda_scl.len());
        for i in 0..len {
            let a = self.eda_scl[i].to_f32();
            let b = other.eda_scl[i].to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        // Compare EDA arousal score
        {
            let a = self.eda_arousal_score.to_f32();
            let b = other.eda_arousal_score.to_f32();
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        let norm_product = libm::sqrtf(norm_a) * libm::sqrtf(norm_b);
        if norm_product > 1e-8 {
            dot_product / norm_product
        } else {
            0.0
        }
    }

    /// Get confidence as f32 (0.0 to 1.0).
    #[inline]
    #[must_use]
    pub fn confidence_f32(&self) -> f32 {
        self.confidence as f32 / 255.0
    }

    /// Set confidence from f32 (0.0 to 1.0).
    #[inline]
    pub fn set_confidence_f32(&mut self, conf: f32) {
        self.confidence = (conf.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for NeuralFingerprint {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "Fingerprint({}, {} ch, conf={}%)",
            self.metadata.modality,
            self.metadata.eeg_channels,
            (self.confidence as u16 * 100) / 255
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory_modality_regions() {
        assert!(SensoryModality::Gustatory.primary_region().contains("Frontal"));
        assert!(SensoryModality::Olfactory.primary_region().contains("Orbitofrontal"));
        assert!(SensoryModality::Visual.primary_region().contains("Occipital"));
    }

    #[test]
    fn test_frequency_band_ranges() {
        for band in FrequencyBand::CANONICAL {
            let (low, high) = band.range_hz();
            assert!(low < high);
            assert!(low >= 0.0);
        }
    }

    #[test]
    fn test_fingerprint_id() {
        let id = FingerprintId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert!(!id.is_null());

        let null_id = FingerprintId::null();
        assert!(null_id.is_null());
    }

    #[test]
    fn test_quality_metrics() {
        let mut q = QualityMetrics::unknown();
        assert!(!q.is_acceptable());

        q.overall_score = 75;
        assert!(q.is_acceptable());
    }

    #[test]
    fn test_fingerprint_similarity() {
        let meta1 = FingerprintMetadata::new(
            FingerprintId::null(),
            SensoryModality::Gustatory,
            "apple",
            "subject1",
            0,
        );

        let mut fp1 = NeuralFingerprint::new(meta1.clone());
        let mut fp2 = NeuralFingerprint::new(meta1);

        // Add identical data
        for i in 0..8 {
            let val = Fixed24_8::from_f32(i as f32);
            fp1.eeg_topography.push(val).unwrap();
            fp2.eeg_topography.push(val).unwrap();
        }

        // Identical fingerprints should have similarity ~1.0
        let sim = fp1.similarity(&fp2);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_band_power_access() {
        let meta = FingerprintMetadata::new(
            FingerprintId::null(),
            SensoryModality::Gustatory,
            "test",
            "subject",
            0,
        );

        let mut fp = NeuralFingerprint::new(meta);

        // Set and get band power
        let val = Fixed24_8::from_f32(10.5);
        assert!(fp.set_band_power(0, FrequencyBand::Alpha, val));

        let retrieved = fp.get_band_power(0, FrequencyBand::Alpha);
        assert!(retrieved.is_some());
        assert!((retrieved.unwrap().to_f32() - 10.5).abs() < 0.1);
    }
}

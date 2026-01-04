//! Cortical Decoders
//!
//! Decode intended/perceived stimuli from EEG and fNIRS signals.
//!
//! # Modality-Specific Decoders
//!
//! - **Tactile**: SEP features (N20, P25, N30) + mu/beta suppression + S1 fNIRS
//! - **Auditory**: AEP features (N1, P2, N2) + ASSR + A1 fNIRS
//! - **Gustatory**: GEP features + frontal theta + insular fNIRS

use rootstar_bci_core::sns::types::{BodyRegion, TasteQualitySimple};
use rootstar_bci_core::sns::SensoryModality;
use rootstar_bci_core::types::{EegSample, HemodynamicSample};
use serde::{Deserialize, Serialize};

use super::cortical_map::{AepComponents, CorticalChannelMap, GepComponents, SepComponents};
use super::error::{DecoderError, DecoderResult};
use super::features::{AepFeatureExtractor, FnirsFeatureExtractor, GepFeatureExtractor, SepFeatureExtractor, FnirsActivation};

/// Decoder configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Temporal integration window (ms)
    pub window_ms: u32,
    /// Minimum samples required
    pub min_samples: usize,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Use fNIRS data if available
    pub use_fnirs: bool,
    /// Sample rate (Hz)
    pub sample_rate_hz: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            window_ms: 500,
            min_samples: 50,
            confidence_threshold: 0.6,
            use_fnirs: true,
            sample_rate_hz: 250.0,
        }
    }
}

// ============================================================================
// Calibration System
// ============================================================================

/// Per-channel calibration parameters.
///
/// Calibrated values are computed as: `output = (input - offset) * gain`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelCalibration {
    /// Gain multiplier (default 1.0)
    pub gain: f64,
    /// Offset to subtract before gain (default 0.0)
    pub offset: f64,
    /// Channel is enabled for decoding
    pub enabled: bool,
    /// Quality indicator from calibration (0-1)
    pub quality: f64,
}

impl Default for ChannelCalibration {
    fn default() -> Self {
        Self {
            gain: 1.0,
            offset: 0.0,
            enabled: true,
            quality: 1.0,
        }
    }
}

impl ChannelCalibration {
    /// Create calibration with specified gain and offset.
    #[must_use]
    pub fn new(gain: f64, offset: f64) -> Self {
        Self {
            gain,
            offset,
            enabled: true,
            quality: 1.0,
        }
    }

    /// Apply calibration to a raw value.
    #[must_use]
    pub fn apply(&self, raw_value: f64) -> f64 {
        if self.enabled {
            (raw_value - self.offset) * self.gain
        } else {
            0.0
        }
    }
}

/// Modality-specific calibration parameters.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModalityCalibration {
    /// EEG channel calibrations (indexed by channel number)
    pub eeg_channels: Vec<ChannelCalibration>,
    /// fNIRS channel calibrations (indexed by channel number)
    pub fnirs_channels: Vec<ChannelCalibration>,
    /// Global amplitude scaling for this modality
    pub amplitude_scale: f64,
    /// Latency offset in milliseconds (for timing correction)
    pub latency_offset_ms: f64,
    /// Confidence adjustment factor
    pub confidence_scale: f64,
}

impl ModalityCalibration {
    /// Create default calibration for a given number of channels.
    #[must_use]
    pub fn new(n_eeg: usize, n_fnirs: usize) -> Self {
        Self {
            eeg_channels: vec![ChannelCalibration::default(); n_eeg],
            fnirs_channels: vec![ChannelCalibration::default(); n_fnirs],
            amplitude_scale: 1.0,
            latency_offset_ms: 0.0,
            confidence_scale: 1.0,
        }
    }

    /// Get calibration for an EEG channel.
    #[must_use]
    pub fn get_eeg(&self, channel_idx: usize) -> &ChannelCalibration {
        self.eeg_channels.get(channel_idx).unwrap_or(&DEFAULT_CALIBRATION)
    }

    /// Get calibration for an fNIRS channel.
    #[must_use]
    pub fn get_fnirs(&self, channel_idx: usize) -> &ChannelCalibration {
        self.fnirs_channels.get(channel_idx).unwrap_or(&DEFAULT_CALIBRATION)
    }

    /// Set EEG channel calibration.
    pub fn set_eeg(&mut self, channel_idx: usize, cal: ChannelCalibration) {
        if channel_idx >= self.eeg_channels.len() {
            self.eeg_channels.resize(channel_idx + 1, ChannelCalibration::default());
        }
        self.eeg_channels[channel_idx] = cal;
    }

    /// Set fNIRS channel calibration.
    pub fn set_fnirs(&mut self, channel_idx: usize, cal: ChannelCalibration) {
        if channel_idx >= self.fnirs_channels.len() {
            self.fnirs_channels.resize(channel_idx + 1, ChannelCalibration::default());
        }
        self.fnirs_channels[channel_idx] = cal;
    }
}

/// Static default calibration for fallback.
static DEFAULT_CALIBRATION: ChannelCalibration = ChannelCalibration {
    gain: 1.0,
    offset: 0.0,
    enabled: true,
    quality: 1.0,
};

/// Complete decoder calibration state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecoderCalibration {
    /// Tactile modality calibration
    pub tactile: ModalityCalibration,
    /// Auditory modality calibration
    pub auditory: ModalityCalibration,
    /// Gustatory modality calibration
    pub gustatory: ModalityCalibration,
    /// Olfactory modality calibration
    pub olfactory: ModalityCalibration,
    /// Visual modality calibration
    pub visual: ModalityCalibration,
    /// Calibration timestamp (Unix epoch microseconds)
    pub timestamp_us: u64,
    /// Subject ID for this calibration
    pub subject_id: String,
    /// Whether calibration has been performed
    pub is_calibrated: bool,
}

impl DecoderCalibration {
    /// Create new uncalibrated state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tactile: ModalityCalibration::new(8, 4),
            auditory: ModalityCalibration::new(8, 2),
            gustatory: ModalityCalibration::new(8, 2),
            olfactory: ModalityCalibration::new(8, 2),
            visual: ModalityCalibration::new(8, 2),
            timestamp_us: 0,
            subject_id: String::new(),
            is_calibrated: false,
        }
    }

    /// Get modality calibration.
    #[must_use]
    pub fn get_modality(&self, modality: SensoryModality) -> &ModalityCalibration {
        match modality {
            SensoryModality::Tactile => &self.tactile,
            SensoryModality::Auditory => &self.auditory,
            SensoryModality::Gustatory => &self.gustatory,
            SensoryModality::Olfactory => &self.olfactory,
            SensoryModality::Visual => &self.visual,
        }
    }

    /// Get mutable modality calibration.
    pub fn get_modality_mut(&mut self, modality: SensoryModality) -> &mut ModalityCalibration {
        match modality {
            SensoryModality::Tactile => &mut self.tactile,
            SensoryModality::Auditory => &mut self.auditory,
            SensoryModality::Gustatory => &mut self.gustatory,
            SensoryModality::Olfactory => &mut self.olfactory,
            SensoryModality::Visual => &mut self.visual,
        }
    }

    /// Mark calibration as complete.
    pub fn set_calibrated(&mut self, subject_id: &str, timestamp_us: u64) {
        self.subject_id = subject_id.to_string();
        self.timestamp_us = timestamp_us;
        self.is_calibrated = true;
    }

    /// Serialize calibration to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize calibration from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Tactile prediction result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TactilePrediction {
    /// Predicted body location
    pub location: BodyRegion,
    /// Predicted intensity (0-1)
    pub intensity: f64,
    /// Predicted texture class
    pub texture_class: TextureClass,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// SEP features used
    pub sep_features: SepComponents,
    /// fNIRS activation if available
    pub fnirs_activation: Option<FnirsActivation>,
}

/// Texture classification
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextureClass {
    /// Smooth surface
    Smooth,
    /// Rough surface
    Rough,
    /// Textured surface (e.g., fabric)
    Textured,
    /// Vibrating surface
    Vibrating,
    /// Unknown
    Unknown,
}

/// Auditory prediction result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditoryPrediction {
    /// Frequency band estimate
    pub frequency_band: FrequencyBand,
    /// Intensity in dB SPL
    pub intensity_db: f64,
    /// Spatial location (azimuth in degrees, -180 to 180)
    pub azimuth_deg: f64,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// AEP features used
    pub aep_features: AepComponents,
    /// fNIRS activation if available
    pub fnirs_activation: Option<FnirsActivation>,
}

/// Frequency band classification
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrequencyBand {
    /// Low frequency (< 500 Hz)
    Low,
    /// Mid frequency (500-2000 Hz)
    Mid,
    /// High frequency (> 2000 Hz)
    High,
    /// Broadband/noise
    Broadband,
}

/// Gustatory prediction result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GustatoryPrediction {
    /// Predicted taste quality
    pub taste_quality: TasteQualitySimple,
    /// Predicted intensity (0-1)
    pub intensity: f64,
    /// Hedonic value (pleasantness: -1 to 1)
    pub hedonic_value: f64,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// GEP features used
    pub gep_features: GepComponents,
    /// fNIRS activation if available
    pub fnirs_activation: Option<FnirsActivation>,
}

/// Cortical decoder for all modalities
#[derive(Clone, Debug)]
pub struct CorticalDecoder {
    /// Configuration
    config: DecoderConfig,
    /// Channel mapping
    channel_map: CorticalChannelMap,
    /// Calibration parameters
    calibration: DecoderCalibration,
    /// SEP feature extractor
    sep_extractor: SepFeatureExtractor,
    /// AEP feature extractor
    aep_extractor: AepFeatureExtractor,
    /// GEP feature extractor
    gep_extractor: GepFeatureExtractor,
    /// fNIRS feature extractor
    fnirs_extractor: FnirsFeatureExtractor,
}

impl CorticalDecoder {
    /// Create a new decoder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DecoderConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: DecoderConfig) -> Self {
        Self {
            config,
            channel_map: CorticalChannelMap::default(),
            calibration: DecoderCalibration::new(),
            sep_extractor: SepFeatureExtractor::default(),
            aep_extractor: AepFeatureExtractor::default(),
            gep_extractor: GepFeatureExtractor::default(),
            fnirs_extractor: FnirsFeatureExtractor::default(),
        }
    }

    /// Set custom channel mapping
    pub fn set_channel_map(&mut self, map: CorticalChannelMap) {
        self.channel_map = map;
    }

    /// Get the current calibration state.
    #[must_use]
    pub fn calibration(&self) -> &DecoderCalibration {
        &self.calibration
    }

    /// Set calibration parameters.
    pub fn set_calibration(&mut self, calibration: DecoderCalibration) {
        self.calibration = calibration;
    }

    /// Load calibration from JSON string.
    pub fn load_calibration(&mut self, json: &str) -> Result<(), serde_json::Error> {
        self.calibration = DecoderCalibration::from_json(json)?;
        Ok(())
    }

    /// Export current calibration as JSON.
    #[must_use]
    pub fn export_calibration(&self) -> String {
        self.calibration.to_json()
    }

    /// Decode tactile sensation from cortical activity
    pub fn decode_tactile(
        &self,
        eeg_window: &[EegSample],
        fnirs_window: &[HemodynamicSample],
        stimulus_time_us: u64,
    ) -> DecoderResult<TactilePrediction> {
        // Validate input
        if eeg_window.len() < self.config.min_samples {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: self.config.min_samples,
            });
        }

        // Extract SEP features from C3 (right body) and C4 (left body)
        let sep_c3 = self.sep_extractor.extract(
            eeg_window,
            stimulus_time_us,
            self.channel_map.tactile.c3,
        )?;

        let sep_c4 = self.sep_extractor.extract(
            eeg_window,
            stimulus_time_us,
            self.channel_map.tactile.c4,
        )?;

        // Determine laterality from asymmetry
        let c3_power = sep_c3.n20_amplitude_uv.abs() + sep_c3.p25_amplitude_uv.abs();
        let c4_power = sep_c4.n20_amplitude_uv.abs() + sep_c4.p25_amplitude_uv.abs();
        let total_power = c3_power + c4_power;
        let asymmetry = if total_power > 0.1 {
            (c3_power - c4_power) / total_power
        } else {
            0.0
        };

        // Determine body side from asymmetry (contralateral representation)
        let (location, primary_sep) = if asymmetry > 0.2 {
            // Stronger C3 response → left body
            use rootstar_bci_core::sns::types::{Finger, Hand};
            (BodyRegion::Fingertip(Finger::Index), sep_c3.clone())
        } else if asymmetry < -0.2 {
            // Stronger C4 response → right body
            use rootstar_bci_core::sns::types::{Finger, Hand};
            (BodyRegion::Palm(Hand::Right), sep_c4.clone())
        } else {
            // Bilateral or midline
            use rootstar_bci_core::sns::types::Finger;
            (BodyRegion::Fingertip(Finger::Index), sep_c3.clone())
        };

        // Estimate intensity from SEP amplitude
        let intensity = self.estimate_tactile_intensity(&primary_sep);

        // Estimate texture from frequency content
        let texture = self.estimate_texture(&primary_sep);

        // Extract fNIRS activation if available
        let fnirs_activation = if self.config.use_fnirs && !fnirs_window.is_empty() {
            self.channel_map.tactile.fnirs_s1.first()
                .and_then(|&ch| self.fnirs_extractor.extract_activation(fnirs_window, ch).ok())
        } else {
            None
        };

        // Calculate confidence
        let eeg_confidence = self.calculate_tactile_confidence(&primary_sep);
        let fnirs_confidence = fnirs_activation.as_ref()
            .map(|a| (a.delta_hbo2.abs() / 2.0).min(1.0))
            .unwrap_or(0.0);
        let confidence = eeg_confidence * 0.7 + fnirs_confidence * 0.3;

        Ok(TactilePrediction {
            location,
            intensity,
            texture_class: texture,
            confidence,
            sep_features: primary_sep,
            fnirs_activation,
        })
    }

    /// Decode auditory perception
    pub fn decode_auditory(
        &self,
        eeg_window: &[EegSample],
        fnirs_window: &[HemodynamicSample],
        stimulus_time_us: u64,
    ) -> DecoderResult<AuditoryPrediction> {
        if eeg_window.len() < self.config.min_samples {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: self.config.min_samples,
            });
        }

        let electrodes = self.channel_map.auditory.primary_electrodes();
        if electrodes.is_empty() {
            return Err(DecoderError::UndefinedChannelMapping {
                modality: SensoryModality::Auditory,
            });
        }

        // Extract AEP from primary auditory electrode
        let aep = self.aep_extractor.extract(
            eeg_window,
            stimulus_time_us,
            electrodes[0],
        )?;

        // Estimate frequency band from N1 latency (shorter for higher frequencies)
        let freq_band = if aep.n1_latency_ms < 90.0 {
            FrequencyBand::High
        } else if aep.n1_latency_ms < 110.0 {
            FrequencyBand::Mid
        } else {
            FrequencyBand::Low
        };

        // Estimate intensity from N1 amplitude
        let intensity_db = 40.0 + aep.n1_amplitude_uv.abs() * 2.0;

        // Estimate laterality from inter-aural differences
        let azimuth = if electrodes.len() >= 2 {
            let left = self.aep_extractor.extract(eeg_window, stimulus_time_us, electrodes[0]).ok();
            let right = self.aep_extractor.extract(eeg_window, stimulus_time_us, electrodes[1]).ok();

            match (left, right) {
                (Some(l), Some(r)) => {
                    let diff = l.n1_amplitude_uv - r.n1_amplitude_uv;
                    diff.clamp(-1.0, 1.0) * 90.0
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        // fNIRS activation
        let fnirs_activation = if self.config.use_fnirs && !fnirs_window.is_empty() {
            self.channel_map.auditory.fnirs_a1.first()
                .and_then(|&ch| self.fnirs_extractor.extract_activation(fnirs_window, ch).ok())
        } else {
            None
        };

        // Confidence
        let confidence = (aep.n1_amplitude_uv.abs() / 10.0).clamp(0.3, 0.95);

        Ok(AuditoryPrediction {
            frequency_band: freq_band,
            intensity_db,
            azimuth_deg: azimuth,
            confidence,
            aep_features: aep,
            fnirs_activation,
        })
    }

    /// Decode gustatory perception
    pub fn decode_gustatory(
        &self,
        eeg_window: &[EegSample],
        fnirs_window: &[HemodynamicSample],
        stimulus_time_us: u64,
    ) -> DecoderResult<GustatoryPrediction> {
        if eeg_window.len() < self.config.min_samples {
            return Err(DecoderError::InsufficientData {
                got: eeg_window.len(),
                need: self.config.min_samples,
            });
        }

        let channels = self.channel_map.gustatory.primary_electrodes();

        // Extract GEP
        let gep = self.gep_extractor.extract(
            eeg_window,
            stimulus_time_us,
            &channels,
        )?;

        // Determine taste quality from frontal patterns
        // (This is highly simplified - real implementation would use ML)
        let taste_quality = self.estimate_taste_quality(&gep);

        // Intensity from GEP amplitude
        let intensity = (gep.p1_amplitude_uv.abs() / 20.0).clamp(0.0, 1.0);

        // Hedonic value from theta power and asymmetry
        let hedonic = self.estimate_hedonic_value(&gep);

        // fNIRS (critical for gustatory given deep insula)
        let fnirs_activation = if self.config.use_fnirs && !fnirs_window.is_empty() {
            self.channel_map.gustatory.fnirs_gustatory.first()
                .and_then(|&ch| self.fnirs_extractor.extract_activation(fnirs_window, ch).ok())
        } else {
            None
        };

        // Confidence (lower for gustatory due to signal challenges)
        let eeg_conf = (gep.p1_amplitude_uv.abs() / 15.0).clamp(0.2, 0.7);
        let fnirs_conf = fnirs_activation.as_ref()
            .map(|a| (a.delta_hbo2.abs() / 1.0).min(0.8))
            .unwrap_or(0.0);
        let confidence = eeg_conf * 0.4 + fnirs_conf * 0.6;

        Ok(GustatoryPrediction {
            taste_quality,
            intensity,
            hedonic_value: hedonic,
            confidence,
            gep_features: gep,
            fnirs_activation,
        })
    }

    // Helper methods

    fn estimate_tactile_intensity(&self, sep: &SepComponents) -> f64 {
        // Intensity correlates with SEP amplitude
        let amplitude = sep.n20_amplitude_uv.abs() + sep.p25_amplitude_uv.abs();
        (amplitude / 20.0).clamp(0.0, 1.0)
    }

    fn estimate_texture(&self, sep: &SepComponents) -> TextureClass {
        // High beta suggests vibration, high mu suppression suggests pressure
        if sep.beta_rebound > 5.0 {
            TextureClass::Vibrating
        } else if sep.mu_suppression > 3.0 {
            TextureClass::Rough
        } else {
            TextureClass::Smooth
        }
    }

    fn calculate_tactile_confidence(&self, sep: &SepComponents) -> f64 {
        // Confidence based on SEP clarity
        let amplitude_score = (sep.n20_amplitude_uv.abs() / 5.0).clamp(0.0, 1.0);
        let latency_score = if sep.n20_latency_ms > 15.0 && sep.n20_latency_ms < 30.0 {
            1.0
        } else {
            0.5
        };
        amplitude_score * 0.7 + latency_score * 0.3
    }

    fn estimate_taste_quality(&self, gep: &GepComponents) -> TasteQualitySimple {
        // Highly simplified - real implementation would use trained classifier
        if gep.frontal_theta > 5.0 {
            TasteQualitySimple::Bitter
        } else if gep.p1_amplitude_uv > 5.0 {
            TasteQualitySimple::Sweet
        } else if gep.p1_amplitude_uv < -5.0 {
            TasteQualitySimple::Sour
        } else {
            TasteQualitySimple::Umami
        }
    }

    fn estimate_hedonic_value(&self, gep: &GepComponents) -> f64 {
        // Positive theta often correlates with pleasant tastes
        (gep.frontal_theta / 10.0 - 0.5).clamp(-1.0, 1.0)
    }

    /// Set global gain for all channels in a modality.
    ///
    /// The gain is applied during feature extraction as: output = (input - offset) * gain
    pub fn set_gain(&mut self, modality: SensoryModality, gain: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        cal.amplitude_scale = gain;
    }

    /// Set global offset for all channels in a modality.
    ///
    /// The offset is subtracted before gain: output = (input - offset) * gain
    pub fn set_offset(&mut self, modality: SensoryModality, offset: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        cal.latency_offset_ms = offset;
    }

    /// Set gain for a specific EEG channel.
    pub fn set_channel_gain(&mut self, modality: SensoryModality, channel_idx: usize, gain: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        if channel_idx < cal.eeg_channels.len() {
            cal.eeg_channels[channel_idx].gain = gain;
        } else {
            cal.set_eeg(channel_idx, ChannelCalibration::new(gain, 0.0));
        }
    }

    /// Set offset for a specific EEG channel.
    pub fn set_channel_offset(&mut self, modality: SensoryModality, channel_idx: usize, offset: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        if channel_idx < cal.eeg_channels.len() {
            cal.eeg_channels[channel_idx].offset = offset;
        } else {
            cal.set_eeg(channel_idx, ChannelCalibration::new(1.0, offset));
        }
    }

    /// Set gain for a specific fNIRS channel.
    pub fn set_fnirs_channel_gain(&mut self, modality: SensoryModality, channel_idx: usize, gain: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        if channel_idx < cal.fnirs_channels.len() {
            cal.fnirs_channels[channel_idx].gain = gain;
        } else {
            cal.set_fnirs(channel_idx, ChannelCalibration::new(gain, 0.0));
        }
    }

    /// Set offset for a specific fNIRS channel.
    pub fn set_fnirs_channel_offset(&mut self, modality: SensoryModality, channel_idx: usize, offset: f64) {
        let cal = self.calibration.get_modality_mut(modality);
        if channel_idx < cal.fnirs_channels.len() {
            cal.fnirs_channels[channel_idx].offset = offset;
        } else {
            cal.set_fnirs(channel_idx, ChannelCalibration::new(1.0, offset));
        }
    }

    /// Enable or disable a specific EEG channel.
    pub fn set_channel_enabled(&mut self, modality: SensoryModality, channel_idx: usize, enabled: bool) {
        let cal = self.calibration.get_modality_mut(modality);
        if channel_idx < cal.eeg_channels.len() {
            cal.eeg_channels[channel_idx].enabled = enabled;
        }
    }

    /// Run automatic calibration procedure for a modality.
    ///
    /// Uses baseline data to compute optimal gain and offset for each channel.
    /// Returns the number of channels successfully calibrated.
    pub fn auto_calibrate(
        &mut self,
        modality: SensoryModality,
        baseline_eeg: &[EegSample],
        baseline_fnirs: &[HemodynamicSample],
        subject_id: &str,
    ) -> usize {
        let mut calibrated_count = 0;

        // Compute per-channel statistics from baseline
        if !baseline_eeg.is_empty() {
            let n_channels = 8usize; // EEG channels
            let cal = self.calibration.get_modality_mut(modality);

            for ch_idx in 0..n_channels {
                // Collect values for this channel
                let values: Vec<f64> = baseline_eeg
                    .iter()
                    .map(|s| {
                        use rootstar_bci_core::types::EegChannel;
                        let ch = EegChannel::ALL.get(ch_idx).copied().unwrap_or(EegChannel::Fp1);
                        s.channel(ch).to_f32() as f64
                    })
                    .collect();

                if values.len() < 10 {
                    continue;
                }

                // Compute mean (offset) and std (for gain normalization)
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                let std_dev = variance.sqrt();

                // Set offset to remove DC bias
                let offset = mean;

                // Set gain to normalize to unit variance (if std > 0)
                let gain = if std_dev > 1e-6 { 1.0 / std_dev } else { 1.0 };

                // Compute quality based on signal variance
                let quality = (std_dev / 100.0).clamp(0.0, 1.0);

                cal.set_eeg(ch_idx, ChannelCalibration {
                    gain,
                    offset,
                    enabled: quality > 0.1, // Disable very low quality channels
                    quality,
                });

                calibrated_count += 1;
            }
        }

        // Calibrate fNIRS channels similarly
        if !baseline_fnirs.is_empty() {
            let cal = self.calibration.get_modality_mut(modality);
            let n_fnirs = cal.fnirs_channels.len();

            for ch_idx in 0..n_fnirs {
                // Collect HbO2 values for this channel
                let values: Vec<f64> = baseline_fnirs
                    .iter()
                    .filter(|s| s.channel.idx() == ch_idx as u8)
                    .map(|s| s.delta_hbo2.to_f32() as f64)
                    .collect();

                if values.len() < 5 {
                    continue;
                }

                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                let std_dev = variance.sqrt();

                let offset = mean;
                let gain = if std_dev > 1e-6 { 1.0 / std_dev } else { 1.0 };
                let quality = (std_dev / 1.0).clamp(0.0, 1.0); // fNIRS has different scale

                cal.set_fnirs(ch_idx, ChannelCalibration {
                    gain,
                    offset,
                    enabled: quality > 0.05,
                    quality,
                });

                calibrated_count += 1;
            }
        }

        // Mark calibration as complete
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        self.calibration.set_calibrated(subject_id, timestamp);

        calibrated_count
    }

    /// Check if decoder has been calibrated.
    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_calibrated
    }

    /// Apply calibration to a raw EEG value.
    #[must_use]
    pub fn apply_calibration(&self, modality: SensoryModality, channel_idx: usize, raw_value: f64) -> f64 {
        let cal = self.calibration.get_modality(modality);
        let ch_cal = cal.get_eeg(channel_idx);
        ch_cal.apply(raw_value) * cal.amplitude_scale
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        // Reset extractors if they maintain state
        self.sep_extractor = SepFeatureExtractor::default();
        self.aep_extractor = AepFeatureExtractor::default();
        self.gep_extractor = GepFeatureExtractor::default();
        self.fnirs_extractor = FnirsFeatureExtractor::default();
    }

    /// Reset decoder state and calibration
    pub fn reset_with_calibration(&mut self) {
        self.reset();
        self.calibration = DecoderCalibration::new();
    }
}

impl Default for CorticalDecoder {
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
    use rootstar_bci_core::types::Fixed24_8;

    fn create_test_samples(n: usize, base_time: u64) -> Vec<EegSample> {
        (0..n)
            .map(|i| {
                let mut sample = EegSample::new(base_time + i as u64 * 1000, i as u32);
                for ch in rootstar_bci_core::types::EegChannel::ALL {
                    sample.set_channel(ch, Fixed24_8::from_f32((i as f32).sin() * 5.0));
                }
                sample
            })
            .collect()
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = CorticalDecoder::new();
        assert_eq!(decoder.config.min_samples, 50);
    }

    #[test]
    fn test_tactile_decode_insufficient_data() {
        let decoder = CorticalDecoder::new();
        let samples = create_test_samples(10, 0);

        let result = decoder.decode_tactile(&samples, &[], 0);
        assert!(matches!(result, Err(DecoderError::InsufficientData { .. })));
    }

    #[test]
    fn test_tactile_decode() {
        let decoder = CorticalDecoder::with_config(DecoderConfig {
            min_samples: 10,
            ..Default::default()
        });
        let samples = create_test_samples(100, 0);

        let result = decoder.decode_tactile(&samples, &[], 0);
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn test_channel_calibration() {
        let cal = ChannelCalibration::new(2.0, 10.0);

        // Test calibration formula: output = (input - offset) * gain
        assert_eq!(cal.apply(20.0), 20.0); // (20 - 10) * 2 = 20
        assert_eq!(cal.apply(10.0), 0.0);  // (10 - 10) * 2 = 0
        assert_eq!(cal.apply(15.0), 10.0); // (15 - 10) * 2 = 10
    }

    #[test]
    fn test_channel_calibration_disabled() {
        let mut cal = ChannelCalibration::new(2.0, 10.0);
        cal.enabled = false;

        // Disabled channels return 0
        assert_eq!(cal.apply(100.0), 0.0);
    }

    #[test]
    fn test_modality_calibration() {
        let mut cal = ModalityCalibration::new(8, 4);

        // Set calibration for channel 3
        cal.set_eeg(3, ChannelCalibration::new(1.5, 5.0));

        let ch_cal = cal.get_eeg(3);
        assert_eq!(ch_cal.gain, 1.5);
        assert_eq!(ch_cal.offset, 5.0);

        // Default calibration for unset channels
        let default_cal = cal.get_eeg(0);
        assert_eq!(default_cal.gain, 1.0);
        assert_eq!(default_cal.offset, 0.0);
    }

    #[test]
    fn test_decoder_calibration_json() {
        let mut cal = DecoderCalibration::new();
        cal.tactile.set_eeg(0, ChannelCalibration::new(2.0, 1.0));
        cal.set_calibrated("test_subject", 1234567890);

        let json = cal.to_json();
        assert!(json.contains("test_subject"));
        assert!(json.contains("1234567890"));

        // Deserialize
        let loaded = DecoderCalibration::from_json(&json).unwrap();
        assert!(loaded.is_calibrated);
        assert_eq!(loaded.subject_id, "test_subject");
        assert_eq!(loaded.tactile.get_eeg(0).gain, 2.0);
    }

    #[test]
    fn test_decoder_set_gain_offset() {
        let mut decoder = CorticalDecoder::new();

        decoder.set_gain(SensoryModality::Tactile, 1.5);
        decoder.set_offset(SensoryModality::Tactile, 10.0);

        assert_eq!(decoder.calibration().tactile.amplitude_scale, 1.5);
        assert_eq!(decoder.calibration().tactile.latency_offset_ms, 10.0);
    }

    #[test]
    fn test_decoder_set_channel_gain() {
        let mut decoder = CorticalDecoder::new();

        decoder.set_channel_gain(SensoryModality::Auditory, 2, 3.0);
        decoder.set_channel_offset(SensoryModality::Auditory, 2, 5.0);

        let ch_cal = decoder.calibration().auditory.get_eeg(2);
        assert_eq!(ch_cal.gain, 3.0);
        assert_eq!(ch_cal.offset, 5.0);
    }

    #[test]
    fn test_auto_calibrate() {
        let mut decoder = CorticalDecoder::new();

        // Create baseline samples with varying values
        let samples: Vec<EegSample> = (0..100)
            .map(|i| {
                let mut sample = EegSample::new(i as u64 * 1000, i as u32);
                for ch in rootstar_bci_core::types::EegChannel::ALL {
                    // Add some variation with offset
                    let value = 50.0 + (i as f32) * 0.5 + (i as f32).sin() * 10.0;
                    sample.set_channel(ch, Fixed24_8::from_f32(value));
                }
                sample
            })
            .collect();

        assert!(!decoder.is_calibrated());

        let count = decoder.auto_calibrate(
            SensoryModality::Tactile,
            &samples,
            &[],
            "test_subject"
        );

        assert!(count > 0);
        assert!(decoder.is_calibrated());
        assert_eq!(decoder.calibration().subject_id, "test_subject");

        // Verify that channels have non-default calibration
        let ch0 = decoder.calibration().tactile.get_eeg(0);
        assert!(ch0.offset.abs() > 1.0); // Should have detected the ~75 offset
    }

    #[test]
    fn test_apply_calibration() {
        let mut decoder = CorticalDecoder::new();

        decoder.set_channel_gain(SensoryModality::Tactile, 0, 2.0);
        decoder.set_channel_offset(SensoryModality::Tactile, 0, 10.0);
        decoder.set_gain(SensoryModality::Tactile, 1.5); // Global scale

        // (100 - 10) * 2.0 * 1.5 = 270
        let result = decoder.apply_calibration(SensoryModality::Tactile, 0, 100.0);
        assert_eq!(result, 270.0);
    }

    #[test]
    fn test_export_load_calibration() {
        let mut decoder = CorticalDecoder::new();

        decoder.set_channel_gain(SensoryModality::Gustatory, 1, 2.5);
        decoder.calibration.set_calibrated("subject_123", 9999);

        let json = decoder.export_calibration();

        // Create new decoder and load calibration
        let mut decoder2 = CorticalDecoder::new();
        decoder2.load_calibration(&json).unwrap();

        assert!(decoder2.is_calibrated());
        assert_eq!(decoder2.calibration().subject_id, "subject_123");
        assert_eq!(decoder2.calibration().gustatory.get_eeg(1).gain, 2.5);
    }

    #[test]
    fn test_reset_with_calibration() {
        let mut decoder = CorticalDecoder::new();

        decoder.calibration.set_calibrated("test", 1234);
        assert!(decoder.is_calibrated());

        decoder.reset_with_calibration();
        assert!(!decoder.is_calibrated());
    }
}

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

    /// Set decoder gain
    pub fn set_gain(&mut self, _gain: f64) {
        // Gain adjustment would be applied in feature extraction
        // This is a placeholder for calibration interface
    }

    /// Set decoder offset
    pub fn set_offset(&mut self, _offset: f64) {
        // Offset adjustment would be applied in feature extraction
        // This is a placeholder for calibration interface
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        // Reset extractors if they maintain state
        self.sep_extractor = SepFeatureExtractor::default();
        self.aep_extractor = AepFeatureExtractor::default();
        self.gep_extractor = GepFeatureExtractor::default();
        self.fnirs_extractor = FnirsFeatureExtractor::default();
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
}

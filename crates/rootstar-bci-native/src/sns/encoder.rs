//! Cortical Encoder
//!
//! Predict expected EEG and fNIRS responses from simulated receptor activation.
//!
//! # Forward Model
//!
//! The encoder uses a simplified thalamocortical model to predict:
//! - SEP/AEP/GEP components from receptor firing patterns
//! - fNIRS responses via HRF convolution

use rootstar_bci_core::sns::types::{BodyRegion, SensoryModality, TactilePopulationState};
use rootstar_bci_core::types::{EegChannel, Fixed24_8};
use serde::{Deserialize, Serialize};

use super::cortical_map::{AepComponents, GepComponents, SepComponents, CorticalChannelMap};
use super::error::{EncoderError, EncoderResult};
use super::hrf::{HemodynamicResponseFunction, HrfPredictor};

/// Encoder configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Sample rate (Hz)
    pub sample_rate_hz: f64,
    /// Forward model scaling factor
    pub scaling_factor: f64,
    /// Use conduction velocity delays
    pub use_conduction_delays: bool,
    /// Thalamocortical delay (ms)
    pub thalamocortical_delay_ms: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 250.0,
            scaling_factor: 1.0,
            use_conduction_delays: true,
            thalamocortical_delay_ms: 15.0,
        }
    }
}

/// Predicted EEG response
#[derive(Clone, Debug)]
pub struct PredictedEeg {
    /// Channel voltages (µV)
    pub channels: [f64; 8],
    /// Timestamp (ms)
    pub timestamp_ms: f64,
    /// Predicted SEP/AEP/GEP components
    pub predicted_components: PredictedComponents,
}

impl PredictedEeg {
    /// Get value for a specific channel
    #[must_use]
    pub fn channel(&self, ch: EegChannel) -> f64 {
        self.channels[ch.index()]
    }
}

/// Predicted evoked potential components
#[derive(Clone, Debug, Default)]
pub struct PredictedComponents {
    /// Expected N20/N1/P1 amplitude (µV)
    pub early_amplitude: f64,
    /// Expected N20/N1/P1 latency (ms)
    pub early_latency: f64,
    /// Expected mu/beta/theta suppression
    pub band_suppression: f64,
}

/// Predicted fNIRS response
#[derive(Clone, Debug)]
pub struct PredictedFnirs {
    /// Delta HbO2 (µM)
    pub delta_hbo2: f64,
    /// Delta HbR (µM)
    pub delta_hbr: f64,
    /// Timestamp (ms)
    pub timestamp_ms: f64,
}

/// Thalamocortical relay model
#[derive(Clone, Debug)]
struct ThalamocorticalRelay {
    /// Relay gain
    gain: f64,
    /// Temporal integration (ms)
    tau_ms: f64,
    /// Current state
    state: f64,
}

impl ThalamocorticalRelay {
    fn new(gain: f64, tau_ms: f64) -> Self {
        Self { gain, tau_ms, state: 0.0 }
    }

    fn relay(&mut self, input: f64, dt_ms: f64) -> f64 {
        // Leaky integrator
        let alpha = 1.0 - (-dt_ms / self.tau_ms).exp();
        self.state += alpha * (input * self.gain - self.state);
        self.state
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }
}

/// Cortical encoder
#[derive(Clone, Debug)]
pub struct CorticalEncoder {
    /// Configuration
    config: EncoderConfig,
    /// Channel mapping
    channel_map: CorticalChannelMap,
    /// Thalamocortical relay for somatosensory (VPL)
    somatosensory_relay: ThalamocorticalRelay,
    /// Thalamocortical relay for auditory (MGN)
    auditory_relay: ThalamocorticalRelay,
    /// Thalamocortical relay for gustatory (VPMpc)
    gustatory_relay: ThalamocorticalRelay,
    /// HRF predictor
    hrf_predictor: HrfPredictor,
    /// Calibration parameters
    calibration: EncoderCalibration,
}

/// Encoder calibration parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderCalibration {
    /// EEG amplitude scaling
    pub eeg_scale: f64,
    /// EEG offset
    pub eeg_offset: f64,
    /// HRF time-to-peak adjustment
    pub hrf_ttp_adjustment: f64,
    /// HRF amplitude scaling
    pub hrf_amplitude_scale: f64,
}

impl Default for EncoderCalibration {
    fn default() -> Self {
        Self {
            eeg_scale: 1.0,
            eeg_offset: 0.0,
            hrf_ttp_adjustment: 0.0,
            hrf_amplitude_scale: 1.0,
        }
    }
}

impl CorticalEncoder {
    /// Create a new encoder
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: EncoderConfig) -> Self {
        let hrf = HemodynamicResponseFunction::canonical();
        Self {
            config: config.clone(),
            channel_map: CorticalChannelMap::default(),
            somatosensory_relay: ThalamocorticalRelay::new(0.8, 5.0),
            auditory_relay: ThalamocorticalRelay::new(0.9, 3.0),
            gustatory_relay: ThalamocorticalRelay::new(0.6, 10.0),
            hrf_predictor: HrfPredictor::new(hrf, config.sample_rate_hz),
            calibration: EncoderCalibration::default(),
        }
    }

    /// Set calibration parameters
    pub fn set_calibration(&mut self, calibration: EncoderCalibration) {
        self.calibration = calibration;
    }

    /// Get mutable reference to calibration
    pub fn calibration_mut(&mut self) -> &mut EncoderCalibration {
        &mut self.calibration
    }

    /// Encode tactile receptor activation to predicted EEG
    pub fn encode_tactile<const N: usize>(
        &mut self,
        population_state: &TactilePopulationState<N>,
        simulation_time_ms: f64,
    ) -> EncoderResult<PredictedEeg> {
        if population_state.firing_rates.is_empty() {
            return Err(EncoderError::EmptyPopulation {
                modality: SensoryModality::Tactile,
            });
        }

        // Calculate mean population activity
        let mean_rate = population_state.mean_rate();

        // Conduction delay (from periphery to thalamus)
        let conduction_delay_ms = if self.config.use_conduction_delays {
            // Approximate delay based on body region (arm ~20ms, leg ~30ms)
            20.0
        } else {
            0.0
        };

        // Thalamocortical relay
        let dt_ms = 1000.0 / self.config.sample_rate_hz;
        let thalamic_output = self.somatosensory_relay.relay(mean_rate as f64, dt_ms);

        // Cortical column model (simplified)
        let cortical_response = self.s1_column_model(thalamic_output);

        // Forward model to EEG
        let predicted = self.forward_model_s1(cortical_response, &population_state.region)?;

        // Apply calibration
        let mut channels = predicted.channels;
        for ch in &mut channels {
            *ch = *ch * self.calibration.eeg_scale + self.calibration.eeg_offset;
        }

        // Calculate expected latency
        let expected_latency = conduction_delay_ms + self.config.thalamocortical_delay_ms;

        Ok(PredictedEeg {
            channels,
            timestamp_ms: simulation_time_ms,
            predicted_components: PredictedComponents {
                early_amplitude: thalamic_output * 5.0, // N20 amplitude
                early_latency: expected_latency,
                band_suppression: mean_rate as f64 * 0.1,
            },
        })
    }

    /// Encode receptor activation to predicted fNIRS
    ///
    /// Takes a history of activations for proper HRF convolution
    pub fn encode_fnirs(
        &mut self,
        neural_activity: f64,
        current_time_ms: f64,
    ) -> EncoderResult<PredictedFnirs> {
        // Predict hemodynamic response via HRF convolution
        let hrf_response = self.hrf_predictor.predict(neural_activity);

        // Scale by calibration
        let scaled_response = hrf_response * self.calibration.hrf_amplitude_scale;

        // HbO2 and HbR have opposite-ish responses
        let delta_hbo2 = scaled_response;
        let delta_hbr = -scaled_response * 0.3; // HbR typically smaller, opposite sign

        Ok(PredictedFnirs {
            delta_hbo2,
            delta_hbr,
            timestamp_ms: current_time_ms,
        })
    }

    /// Encode auditory activation to predicted EEG
    pub fn encode_auditory(
        &mut self,
        firing_rates: &[f64],
        simulation_time_ms: f64,
    ) -> EncoderResult<PredictedEeg> {
        if firing_rates.is_empty() {
            return Err(EncoderError::EmptyPopulation {
                modality: SensoryModality::Auditory,
            });
        }

        let mean_rate: f64 = firing_rates.iter().sum::<f64>() / firing_rates.len() as f64;

        let dt_ms = 1000.0 / self.config.sample_rate_hz;
        let thalamic_output = self.auditory_relay.relay(mean_rate, dt_ms);

        // A1 forward model (bilateral with right ear → left hemisphere preference)
        let mut channels = [0.0; 8];

        // Temporal electrodes would get strongest response
        // Using P3/P4 as approximation for T7/T8
        channels[EegChannel::P3.index()] = thalamic_output * 8.0;
        channels[EegChannel::P4.index()] = thalamic_output * 8.0;

        // Some spread to frontal
        channels[EegChannel::Fp1.index()] = thalamic_output * 2.0;
        channels[EegChannel::Fp2.index()] = thalamic_output * 2.0;

        Ok(PredictedEeg {
            channels,
            timestamp_ms: simulation_time_ms,
            predicted_components: PredictedComponents {
                early_amplitude: thalamic_output * 8.0,
                early_latency: 100.0, // N1 latency
                band_suppression: mean_rate * 0.05,
            },
        })
    }

    /// Encode gustatory activation to predicted EEG
    pub fn encode_gustatory(
        &mut self,
        taste_activation: &[f64; 5], // [sweet, salty, sour, bitter, umami]
        simulation_time_ms: f64,
    ) -> EncoderResult<PredictedEeg> {
        let total_activation: f64 = taste_activation.iter().sum();

        if total_activation < 0.001 {
            return Err(EncoderError::EmptyPopulation {
                modality: SensoryModality::Gustatory,
            });
        }

        let dt_ms = 1000.0 / self.config.sample_rate_hz;
        let thalamic_output = self.gustatory_relay.relay(total_activation, dt_ms);

        // Gustatory cortex is deep (insula) - limited EEG visibility
        let mut channels = [0.0; 8];

        // Frontal electrodes get weak projection
        channels[EegChannel::Fp1.index()] = thalamic_output * 3.0;
        channels[EegChannel::Fp2.index()] = thalamic_output * 3.0;

        Ok(PredictedEeg {
            channels,
            timestamp_ms: simulation_time_ms,
            predicted_components: PredictedComponents {
                early_amplitude: thalamic_output * 3.0,
                early_latency: 130.0, // GEP P1 latency
                band_suppression: total_activation * 0.02,
            },
        })
    }

    /// S1 cortical column model (simplified)
    fn s1_column_model(&self, thalamic_input: f64) -> f64 {
        // Simple gain model - real implementation would model layer 4 → 2/3 dynamics
        thalamic_input * 0.7
    }

    /// Forward model from S1 to EEG electrodes
    fn forward_model_s1(
        &self,
        cortical_response: f64,
        region: &BodyRegion,
    ) -> EncoderResult<PredictedEeg> {
        let mut channels = [0.0; 8];

        // Determine primary electrode from region
        let primary = self.channel_map.tactile.electrode_for_region(*region);

        // Primary electrode gets strongest signal
        channels[primary.index()] = cortical_response * 10.0;

        // Contralateral gets minimal
        channels[primary.contralateral().index()] = cortical_response * 1.0;

        // Some spread to parietal
        channels[EegChannel::P3.index()] = cortical_response * 3.0;
        channels[EegChannel::P4.index()] = cortical_response * 3.0;

        Ok(PredictedEeg {
            channels,
            timestamp_ms: 0.0, // Will be set by caller
            predicted_components: PredictedComponents::default(),
        })
    }

    /// Reset encoder state
    pub fn reset(&mut self) {
        self.somatosensory_relay.reset();
        self.auditory_relay.reset();
        self.gustatory_relay.reset();
        self.hrf_predictor.reset();
    }
}

impl Default for CorticalEncoder {
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
    use rootstar_bci_core::sns::types::Finger;

    #[test]
    fn test_encoder_creation() {
        let encoder = CorticalEncoder::new();
        assert_eq!(encoder.config.sample_rate_hz, 250.0);
    }

    #[test]
    fn test_tactile_encoding() {
        let mut encoder = CorticalEncoder::new();

        let mut state: TactilePopulationState<16> =
            TactilePopulationState::new(BodyRegion::Fingertip(Finger::Index), 0.0);

        // Add some firing rates
        for _ in 0..10 {
            state.firing_rates.push(50.0).ok();
        }

        let result = encoder.encode_tactile(&state, 100.0);
        assert!(result.is_ok());

        let predicted = result.unwrap();
        assert!(predicted.channels[EegChannel::C3.index()].abs() > 0.0);
    }

    #[test]
    fn test_fnirs_encoding() {
        let mut encoder = CorticalEncoder::new();

        // Single prediction
        let result = encoder.encode_fnirs(1.0, 0.0);
        assert!(result.is_ok());

        // Build up response over time
        let mut max_hbo2 = 0.0f64;
        for i in 0..100 {
            let result = encoder.encode_fnirs(if i < 10 { 1.0 } else { 0.0 }, i as f64 * 100.0);
            if let Ok(pred) = result {
                max_hbo2 = max_hbo2.max(pred.delta_hbo2);
            }
        }

        // Should have seen some response
        assert!(max_hbo2 > 0.0);
    }

    #[test]
    fn test_empty_population_error() {
        let mut encoder = CorticalEncoder::new();

        let state: TactilePopulationState<16> =
            TactilePopulationState::new(BodyRegion::Fingertip(Finger::Index), 0.0);

        let result = encoder.encode_tactile(&state, 0.0);
        assert!(matches!(result, Err(EncoderError::EmptyPopulation { .. })));
    }
}

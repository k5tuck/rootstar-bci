//! SNS Simulation
//!
//! End-to-end simulation of sensory neural processing for testing and validation.
//!
//! # Simulation Pipeline
//!
//! ```text
//! Stimulus → Receptor Population → Spiking Network → Encoder → Predicted EEG
//!                                                       ↓
//!                                              Compare with actual EEG
//!                                                       ↓
//!                                              Decoder → Predicted Activation
//! ```

use std::collections::VecDeque;

use rootstar_bci_core::sns::types::{
    BodyRegion, SensoryModality, TactilePopulationState,
};
use rootstar_bci_core::sns::{
    auditory::{AuditoryPopulationBuilder, AuditoryReceptor},
    gustatory::{TasteBud, TongueRegion},
    tactile::{MeissnerReceptor, MerkelReceptor, PacinianReceptor, RuffiniReceptor, TactileReceptor},
};
use serde::{Deserialize, Serialize};

use super::calibration::{BidirectionalCalibrator, CalibrationSample};
use super::decoder::CorticalDecoder;
use super::encoder::{CorticalEncoder, PredictedEeg, PredictedFnirs};
use super::error::CalibrationResult;

/// Simulation configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Simulation time step (ms)
    pub dt_ms: f64,
    /// Total simulation duration (ms)
    pub duration_ms: f64,
    /// Sample rate for EEG (Hz)
    pub eeg_sample_rate: f64,
    /// Sample rate for fNIRS (Hz)
    pub fnirs_sample_rate: f64,
    /// Enable encoder predictions
    pub enable_encoder: bool,
    /// Enable decoder analysis
    pub enable_decoder: bool,
    /// Enable calibration
    pub enable_calibration: bool,
    /// Noise level for synthetic EEG (µV)
    pub eeg_noise_uv: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            dt_ms: 1.0,
            duration_ms: 10000.0,
            eeg_sample_rate: 250.0,
            fnirs_sample_rate: 10.0,
            enable_encoder: true,
            enable_decoder: true,
            enable_calibration: true,
            eeg_noise_uv: 5.0,
        }
    }
}

/// Simulation state
#[derive(Clone, Debug)]
pub struct SimulationState {
    /// Current simulation time (ms)
    pub time_ms: f64,
    /// Current tactile population state
    pub tactile_state: Option<TactilePopulationState<64>>,
    /// Current auditory population state
    pub auditory_rates: Vec<f64>,
    /// Current gustatory activations
    pub gustatory_activations: [f64; 5],
    /// Latest predicted EEG
    pub predicted_eeg: Option<PredictedEeg>,
    /// Latest predicted fNIRS
    pub predicted_fnirs: Option<PredictedFnirs>,
    /// EEG history for decoder
    pub eeg_history: VecDeque<[f64; 8]>,
    /// fNIRS history for decoder
    pub fnirs_history: VecDeque<(f64, f64)>,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            time_ms: 0.0,
            tactile_state: None,
            auditory_rates: Vec::new(),
            gustatory_activations: [0.0; 5],
            predicted_eeg: None,
            predicted_fnirs: None,
            eeg_history: VecDeque::new(),
            fnirs_history: VecDeque::new(),
        }
    }
}

/// Simple stimulus descriptor for simulation
#[derive(Clone, Debug)]
pub enum SimStimulusKind {
    /// Pressure stimulus
    Pressure { force_n: f32 },
    /// Vibration stimulus
    Vibration { frequency_hz: f32, amplitude: f32 },
    /// Texture stimulus
    Texture { roughness: f32, element_spacing: f32 },
    /// Tone stimulus
    Tone { frequency_hz: f32, amplitude: f32 },
    /// Taste stimulus
    Taste { concentrations: [f32; 5] },
}

/// Stimulus event
#[derive(Clone, Debug)]
pub struct StimulusEvent {
    /// Event onset time (ms)
    pub onset_ms: f64,
    /// Event duration (ms)
    pub duration_ms: f64,
    /// Sensory modality
    pub modality: SensoryModality,
    /// Stimulus kind and parameters
    pub kind: SimStimulusKind,
}

/// SNS simulation
pub struct SnsSimulation {
    /// Configuration
    config: SimulationConfig,
    /// Current state
    state: SimulationState,
    /// Cortical encoder
    encoder: CorticalEncoder,
    /// Cortical decoder
    decoder: CorticalDecoder,
    /// Bidirectional calibrator
    calibrator: BidirectionalCalibrator,
    /// Stimulus event queue
    events: VecDeque<StimulusEvent>,
    /// Tactile receptors
    tactile_receptors: Vec<Box<dyn TactileReceptor + Send>>,
    /// Auditory receptors
    auditory_receptors: Vec<AuditoryReceptor>,
    /// Gustatory receptors
    taste_buds: Vec<TasteBud>,
    /// Random number generator state
    rng_state: u64,
}

impl SnsSimulation {
    /// Create a new simulation
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SimulationConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: SimulationConfig) -> Self {
        Self {
            config,
            state: SimulationState::default(),
            encoder: CorticalEncoder::new(),
            decoder: CorticalDecoder::new(),
            calibrator: BidirectionalCalibrator::new(),
            events: VecDeque::new(),
            tactile_receptors: Vec::new(),
            auditory_receptors: Vec::new(),
            taste_buds: Vec::new(),
            rng_state: 12345,
        }
    }

    /// Initialize tactile receptor population for a body region
    pub fn init_tactile_population(&mut self, region: BodyRegion) {
        self.tactile_receptors.clear();

        // Get receptor density for region
        let density = region.receptor_density();
        let num_receptors = (density as usize).min(64);

        // Create mixed population
        for i in 0..num_receptors {
            // Distribute receptor types based on region
            let receptor: Box<dyn TactileReceptor + Send> = match i % 4 {
                0 => Box::new(MeissnerReceptor::default()),
                1 => Box::new(MerkelReceptor::default()),
                2 => Box::new(PacinianReceptor::default()),
                _ => Box::new(RuffiniReceptor::default()),
            };

            self.tactile_receptors.push(receptor);
        }

        // Initialize state
        self.state.tactile_state = Some(TactilePopulationState::new(region, 0.0));
    }

    /// Initialize auditory receptor population
    pub fn init_auditory_population(&mut self, num_fibers: usize) {
        use rootstar_bci_core::sns::types::Ear;

        self.auditory_receptors.clear();

        // Use builder to get center frequencies
        let builder = AuditoryPopulationBuilder::new()
            .frequency_range(100.0, 8000.0);

        // Get center frequencies and create receptors
        let frequencies = builder.center_frequencies();
        let actual_count = frequencies.len().min(num_fibers);

        for &freq in frequencies.iter().take(actual_count) {
            self.auditory_receptors.push(AuditoryReceptor::at_frequency(freq, Ear::Left));
        }

        self.state.auditory_rates = vec![0.0; actual_count];
    }

    /// Initialize gustatory receptor population
    pub fn init_gustatory_population(&mut self, region: TongueRegion) {
        self.taste_buds.clear();

        // Create taste buds with region-specific sensitivity
        for _ in 0..10 {
            self.taste_buds.push(TasteBud::new(region));
        }

        self.state.gustatory_activations = [0.0; 5];
    }

    /// Add a stimulus event
    pub fn add_event(&mut self, event: StimulusEvent) {
        self.events.push_back(event);
    }

    /// Run one simulation step
    pub fn step(&mut self) -> SimulationStepResult {
        let dt = self.config.dt_ms;
        let time = self.state.time_ms;

        // Process active events
        let active_events: Vec<_> = self
            .events
            .iter()
            .filter(|e| time >= e.onset_ms && time < e.onset_ms + e.duration_ms)
            .cloned()
            .collect();

        // Update receptor populations based on active stimuli
        for event in &active_events {
            match event.modality {
                SensoryModality::Tactile => {
                    self.update_tactile(&event.kind, dt);
                }
                SensoryModality::Auditory => {
                    self.update_auditory(&event.kind, dt);
                }
                SensoryModality::Gustatory => {
                    self.update_gustatory(&event.kind, dt);
                }
                _ => {}
            }
        }

        // Generate encoder predictions
        let mut predicted_eeg = None;
        let mut predicted_fnirs = None;

        if self.config.enable_encoder {
            if let Some(ref tactile_state) = self.state.tactile_state {
                if !tactile_state.firing_rates.is_empty() {
                    if let Ok(eeg) = self.encoder.encode_tactile(tactile_state, time) {
                        predicted_eeg = Some(eeg);
                    }
                }
            }

            // Encode fNIRS based on neural activity
            let neural_activity = self.compute_total_neural_activity();
            if let Ok(fnirs) = self.encoder.encode_fnirs(neural_activity, time) {
                predicted_fnirs = Some(fnirs);
            }
        }

        // Generate synthetic "actual" EEG (predicted + noise)
        let actual_eeg = if let Some(ref pred) = predicted_eeg {
            let mut eeg = pred.channels;
            for ch in &mut eeg {
                *ch += self.gaussian_noise() * self.config.eeg_noise_uv;
            }
            eeg
        } else {
            let mut eeg = [0.0; 8];
            for ch in &mut eeg {
                *ch = self.gaussian_noise() * self.config.eeg_noise_uv;
            }
            eeg
        };

        // Store in history
        self.state.eeg_history.push_back(actual_eeg);
        if self.state.eeg_history.len() > 1000 {
            self.state.eeg_history.pop_front();
        }

        if let Some(ref fnirs) = predicted_fnirs {
            self.state.fnirs_history.push_back((fnirs.delta_hbo2, fnirs.delta_hbr));
            if self.state.fnirs_history.len() > 100 {
                self.state.fnirs_history.pop_front();
            }
        }

        // Decoder analysis
        let decoded_activation = if self.config.enable_decoder && self.state.eeg_history.len() >= 64 {
            let eeg_window: Vec<[f64; 8]> = self.state.eeg_history.iter().rev().take(64).rev().cloned().collect();

            // Simple activation estimate based on C3/C4 amplitude
            let c3_power: f64 = eeg_window.iter().map(|ch| ch[2].powi(2)).sum::<f64>() / 64.0;
            let c4_power: f64 = eeg_window.iter().map(|ch| ch[3].powi(2)).sum::<f64>() / 64.0;

            (c3_power + c4_power).sqrt() / 100.0 // Normalized
        } else {
            0.0
        };

        // Calibration sample
        if self.config.enable_calibration {
            let actual_activation = self.compute_total_neural_activity();

            let sample = CalibrationSample {
                timestamp_ms: time,
                actual_eeg,
                predicted_eeg: predicted_eeg.as_ref().map(|p| p.channels).unwrap_or([0.0; 8]),
                actual_activation,
                decoded_activation,
            };

            let _ = self.calibrator.add_sample(sample);
        }

        // Update state
        self.state.predicted_eeg = predicted_eeg;
        self.state.predicted_fnirs = predicted_fnirs;
        self.state.time_ms += dt;

        // Remove expired events
        self.events.retain(|e| time < e.onset_ms + e.duration_ms);

        SimulationStepResult {
            time_ms: self.state.time_ms,
            actual_eeg,
            predicted_eeg: self.state.predicted_eeg.clone(),
            predicted_fnirs: self.state.predicted_fnirs.clone(),
            decoded_activation,
            neural_activity: self.compute_total_neural_activity(),
        }
    }

    /// Run simulation for specified duration
    pub fn run(&mut self) -> Vec<SimulationStepResult> {
        let mut results = Vec::new();
        let steps = (self.config.duration_ms / self.config.dt_ms) as usize;

        for _ in 0..steps {
            results.push(self.step());
        }

        results
    }

    /// Run calibration after simulation
    pub fn run_calibration(&mut self) -> CalibrationResult<super::calibration::CalibrationMetrics> {
        self.calibrator.compute()
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    /// Get encoder reference
    #[must_use]
    pub fn encoder(&self) -> &CorticalEncoder {
        &self.encoder
    }

    /// Get mutable encoder reference
    pub fn encoder_mut(&mut self) -> &mut CorticalEncoder {
        &mut self.encoder
    }

    /// Get decoder reference
    #[must_use]
    pub fn decoder(&self) -> &CorticalDecoder {
        &self.decoder
    }

    /// Get mutable decoder reference
    pub fn decoder_mut(&mut self) -> &mut CorticalDecoder {
        &mut self.decoder
    }

    /// Get calibrator reference
    #[must_use]
    pub fn calibrator(&self) -> &BidirectionalCalibrator {
        &self.calibrator
    }

    /// Get mutable calibrator reference
    pub fn calibrator_mut(&mut self) -> &mut BidirectionalCalibrator {
        &mut self.calibrator
    }

    /// Reset simulation
    pub fn reset(&mut self) {
        self.state = SimulationState::default();
        self.encoder.reset();
        self.decoder.reset();
        self.calibrator.reset();
        self.events.clear();
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn update_tactile(&mut self, kind: &SimStimulusKind, dt: f64) {
        if let Some(ref mut state) = self.state.tactile_state {
            // Clear previous rates
            state.firing_rates.clear();
            state.timestamp_ms = self.state.time_ms;

            // Get stimulus intensity
            let intensity = match kind {
                SimStimulusKind::Pressure { force_n } => *force_n,
                SimStimulusKind::Vibration { amplitude, .. } => *amplitude,
                SimStimulusKind::Texture { roughness, .. } => *roughness,
                _ => 0.0,
            };

            // Compute firing rates for each receptor
            for receptor in &mut self.tactile_receptors {
                let rate = receptor.compute_rate(intensity, dt as f32);
                let _ = state.firing_rates.push(rate);
            }
        }
    }

    fn update_auditory(&mut self, kind: &SimStimulusKind, dt: f64) {
        let (frequency_hz, amplitude) = match kind {
            SimStimulusKind::Tone { frequency_hz, amplitude } => (*frequency_hz, *amplitude),
            _ => return,
        };

        self.state.auditory_rates.clear();

        for receptor in &mut self.auditory_receptors {
            let rate = receptor.process(frequency_hz, amplitude, dt as f32);
            self.state.auditory_rates.push(rate as f64);
        }
    }

    fn update_gustatory(&mut self, kind: &SimStimulusKind, dt: f64) {
        let concentrations = match kind {
            SimStimulusKind::Taste { concentrations } => concentrations,
            _ => return,
        };

        // Aggregate responses from all taste buds
        self.state.gustatory_activations = [0.0; 5];

        let num_buds = self.taste_buds.len();
        if num_buds == 0 {
            return;
        }

        for bud in &mut self.taste_buds {
            bud.update(concentrations, dt as f32);
            let responses = bud.get_responses();
            for (i, &resp) in responses.iter().enumerate() {
                self.state.gustatory_activations[i] += resp as f64 / num_buds as f64;
            }
        }
    }

    fn compute_total_neural_activity(&self) -> f64 {
        let mut total: f64 = 0.0;
        let mut count = 0;

        // Tactile contribution
        if let Some(ref state) = self.state.tactile_state {
            if !state.firing_rates.is_empty() {
                total += state.mean_rate() as f64;
                count += 1;
            }
        }

        // Auditory contribution
        if !self.state.auditory_rates.is_empty() {
            let mean: f64 = self.state.auditory_rates.iter().sum::<f64>()
                / self.state.auditory_rates.len() as f64;
            total += mean;
            count += 1;
        }

        // Gustatory contribution
        let gust_sum: f64 = self.state.gustatory_activations.iter().sum();
        if gust_sum > 0.0 {
            total += gust_sum / 5.0;
            count += 1;
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    fn gaussian_noise(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.random_f64();
        let u2 = self.random_f64();

        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn random_f64(&mut self) -> f64 {
        // Simple xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }
}

impl Default for SnsSimulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single simulation step
#[derive(Clone, Debug)]
pub struct SimulationStepResult {
    /// Simulation time (ms)
    pub time_ms: f64,
    /// Actual (synthetic) EEG values
    pub actual_eeg: [f64; 8],
    /// Predicted EEG from encoder
    pub predicted_eeg: Option<PredictedEeg>,
    /// Predicted fNIRS from encoder
    pub predicted_fnirs: Option<PredictedFnirs>,
    /// Decoded activation from decoder
    pub decoded_activation: f64,
    /// Total neural activity
    pub neural_activity: f64,
}

/// Simulation builder for common scenarios
pub struct SimulationBuilder {
    config: SimulationConfig,
    tactile_region: Option<BodyRegion>,
    auditory_fibers: Option<usize>,
    gustatory_region: Option<TongueRegion>,
    events: Vec<StimulusEvent>,
}

impl SimulationBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SimulationConfig::default(),
            tactile_region: None,
            auditory_fibers: None,
            gustatory_region: None,
            events: Vec::new(),
        }
    }

    /// Set simulation duration
    #[must_use]
    pub fn duration(mut self, duration_ms: f64) -> Self {
        self.config.duration_ms = duration_ms;
        self
    }

    /// Set time step
    #[must_use]
    pub fn time_step(mut self, dt_ms: f64) -> Self {
        self.config.dt_ms = dt_ms;
        self
    }

    /// Add tactile population
    #[must_use]
    pub fn with_tactile(mut self, region: BodyRegion) -> Self {
        self.tactile_region = Some(region);
        self
    }

    /// Add auditory population
    #[must_use]
    pub fn with_auditory(mut self, num_fibers: usize) -> Self {
        self.auditory_fibers = Some(num_fibers);
        self
    }

    /// Add gustatory population
    #[must_use]
    pub fn with_gustatory(mut self, region: TongueRegion) -> Self {
        self.gustatory_region = Some(region);
        self
    }

    /// Add stimulus event
    #[must_use]
    pub fn with_event(mut self, event: StimulusEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Add pressure stimulus
    #[must_use]
    pub fn with_pressure(self, onset_ms: f64, duration_ms: f64, force_n: f32) -> Self {
        self.with_event(StimulusEvent {
            onset_ms,
            duration_ms,
            modality: SensoryModality::Tactile,
            kind: SimStimulusKind::Pressure { force_n },
        })
    }

    /// Add vibration stimulus
    #[must_use]
    pub fn with_vibration(
        self,
        onset_ms: f64,
        duration_ms: f64,
        frequency_hz: f32,
        amplitude: f32,
    ) -> Self {
        self.with_event(StimulusEvent {
            onset_ms,
            duration_ms,
            modality: SensoryModality::Tactile,
            kind: SimStimulusKind::Vibration { frequency_hz, amplitude },
        })
    }

    /// Add tone stimulus
    #[must_use]
    pub fn with_tone(
        self,
        onset_ms: f64,
        duration_ms: f64,
        frequency_hz: f32,
        amplitude: f32,
    ) -> Self {
        self.with_event(StimulusEvent {
            onset_ms,
            duration_ms,
            modality: SensoryModality::Auditory,
            kind: SimStimulusKind::Tone { frequency_hz, amplitude },
        })
    }

    /// Add taste stimulus
    #[must_use]
    pub fn with_taste(
        self,
        onset_ms: f64,
        duration_ms: f64,
        concentrations: [f32; 5],
    ) -> Self {
        self.with_event(StimulusEvent {
            onset_ms,
            duration_ms,
            modality: SensoryModality::Gustatory,
            kind: SimStimulusKind::Taste { concentrations },
        })
    }

    /// Disable calibration
    #[must_use]
    pub fn without_calibration(mut self) -> Self {
        self.config.enable_calibration = false;
        self
    }

    /// Set noise level
    #[must_use]
    pub fn noise_level(mut self, noise_uv: f64) -> Self {
        self.config.eeg_noise_uv = noise_uv;
        self
    }

    /// Build the simulation
    #[must_use]
    pub fn build(self) -> SnsSimulation {
        let mut sim = SnsSimulation::with_config(self.config);

        if let Some(region) = self.tactile_region {
            sim.init_tactile_population(region);
        }

        if let Some(fibers) = self.auditory_fibers {
            sim.init_auditory_population(fibers);
        }

        if let Some(region) = self.gustatory_region {
            sim.init_gustatory_population(region);
        }

        for event in self.events {
            sim.add_event(event);
        }

        sim
    }
}

impl Default for SimulationBuilder {
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

    #[test]
    fn test_simulation_creation() {
        let sim = SnsSimulation::new();
        assert_eq!(sim.state().time_ms, 0.0);
    }

    #[test]
    fn test_simulation_step() {
        let mut sim = SimulationBuilder::new()
            .duration(100.0)
            .time_step(1.0)
            .with_tactile(BodyRegion::Fingertip(Finger::Index))
            .with_pressure(10.0, 50.0, 1.0)
            .build();

        let result = sim.step();
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_simulation_run() {
        let mut sim = SimulationBuilder::new()
            .duration(100.0)
            .time_step(1.0)
            .with_tactile(BodyRegion::Fingertip(Finger::Index))
            .with_pressure(10.0, 50.0, 1.0)
            .without_calibration()
            .build();

        let results = sim.run();
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_auditory_simulation() {
        let mut sim = SimulationBuilder::new()
            .duration(100.0)
            .with_auditory(32)
            .with_tone(10.0, 50.0, 1000.0, 0.5)
            .build();

        let results = sim.run();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_gustatory_simulation() {
        let mut sim = SimulationBuilder::new()
            .duration(100.0)
            .with_gustatory(TongueRegion::TipAnterior)
            .with_taste(10.0, 50.0, [0.5, 0.0, 0.0, 0.0, 0.0]) // Sweet
            .build();

        let results = sim.run();
        assert!(!results.is_empty());
    }
}

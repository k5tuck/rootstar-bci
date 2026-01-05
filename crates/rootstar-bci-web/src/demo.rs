//! Demo mode for testing the web visualization without hardware.
//!
//! Provides realistic synthetic BCI data for previewing the UI on Vercel
//! or any static hosting without requiring a connected BCI device.

use wasm_bindgen::prelude::*;

/// Sample EEG data representing common brain wave patterns.
/// 8 channels in µV, following 10-20 electrode positions:
/// Fp1, Fp2, C3, C4, P3, P4, O1, O2
pub const DEMO_EEG_BASELINE: [f32; 8] = [
    -5.2, 3.8, -12.4, 8.7, -2.1, 6.3, -8.9, 4.2,
];

/// Demo fNIRS HbO2 values (mM·cm) for 4 optodes
pub const DEMO_FNIRS_HBO2: [f32; 4] = [0.045, 0.052, 0.038, 0.041];

/// Demo fNIRS HbR values (mM·cm) for 4 optodes
pub const DEMO_FNIRS_HBR: [f32; 4] = [-0.012, -0.015, -0.008, -0.011];

/// Demo EMG RMS values (µV) for 8 facial muscles
/// Zyg-L, Zyg-R, Cor-L, Cor-R, Mas-L, Mas-R, Orb-U, Orb-D
pub const DEMO_EMG_RMS: [f32; 8] = [2.5, 2.8, 1.2, 1.4, 8.5, 9.2, 0.8, 0.6];

/// Demo EDA values (µS) for 4 sites
pub const DEMO_EDA_SCL: [f32; 4] = [5.2, 4.8, 5.5, 5.0];
pub const DEMO_EDA_SCR: [f32; 4] = [0.3, 0.1, 0.5, 0.2];

/// EEG frequency bands for demo VR preview
pub const DEMO_BAND_POWER: DemoBandPower = DemoBandPower {
    delta: 0.25,
    theta: 0.18,
    alpha: 0.35,
    beta: 0.15,
    gamma: 0.07,
};

/// Band power values for demo
#[derive(Clone, Copy, Debug)]
pub struct DemoBandPower {
    pub delta: f32, // 0.5-4 Hz
    pub theta: f32, // 4-8 Hz
    pub alpha: f32, // 8-13 Hz
    pub beta: f32,  // 13-30 Hz
    pub gamma: f32, // 30-100 Hz
}

/// Demo scenarios representing different brain states
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoScenario {
    /// Relaxed eyes-closed state (high alpha)
    Relaxed,
    /// Focused attention task (high beta)
    Focused,
    /// Drowsy/sleepy state (high theta)
    Drowsy,
    /// Active thinking/problem solving (high gamma)
    Thinking,
    /// Meditation state (high alpha + theta)
    Meditation,
    /// Motor imagery (mu rhythm suppression)
    MotorImagery,
    /// Emotional response (asymmetric frontal)
    EmotionalResponse,
}

impl DemoScenario {
    /// Get band power modifiers for this scenario
    pub fn band_modifiers(&self) -> [f32; 5] {
        match self {
            Self::Relaxed => [0.8, 0.9, 2.5, 0.5, 0.3],
            Self::Focused => [0.6, 0.7, 0.8, 2.2, 1.5],
            Self::Drowsy => [1.2, 2.0, 0.6, 0.4, 0.3],
            Self::Thinking => [0.7, 0.8, 0.7, 1.5, 2.5],
            Self::Meditation => [0.9, 1.8, 2.0, 0.6, 0.4],
            Self::MotorImagery => [0.8, 1.0, 0.4, 1.8, 1.2],
            Self::EmotionalResponse => [1.0, 1.2, 1.0, 1.3, 0.8],
        }
    }

    /// Get EMG modifiers (facial expression patterns)
    pub fn emg_modifiers(&self) -> [f32; 8] {
        match self {
            Self::Relaxed => [0.3, 0.3, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2],
            Self::Focused => [0.4, 0.4, 1.5, 1.5, 0.8, 0.8, 0.3, 0.3],
            Self::Drowsy => [0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5],
            Self::Thinking => [0.5, 0.5, 1.8, 1.8, 0.6, 0.6, 0.4, 0.4],
            Self::Meditation => [0.2, 0.2, 0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
            Self::MotorImagery => [0.4, 0.4, 0.5, 0.5, 1.2, 1.2, 0.3, 0.3],
            Self::EmotionalResponse => [2.0, 2.0, 0.3, 0.3, 0.5, 0.5, 0.4, 0.4],
        }
    }

    /// Get EDA arousal level (0-1)
    pub fn arousal_level(&self) -> f32 {
        match self {
            Self::Relaxed => 0.2,
            Self::Focused => 0.6,
            Self::Drowsy => 0.1,
            Self::Thinking => 0.5,
            Self::Meditation => 0.15,
            Self::MotorImagery => 0.4,
            Self::EmotionalResponse => 0.85,
        }
    }
}

/// Demo data runner that generates continuous synthetic BCI data
#[wasm_bindgen]
pub struct DemoRunner {
    /// Current simulation time (ms)
    time_ms: f64,
    /// Current scenario
    scenario: DemoScenario,
    /// Whether demo is running
    running: bool,
    /// EEG sample rate (Hz)
    eeg_sample_rate: u32,
    /// fNIRS sample rate (Hz)
    fnirs_sample_rate: u32,
    /// Accumulated time for EEG sampling
    eeg_accum: f64,
    /// Accumulated time for fNIRS sampling
    fnirs_accum: f64,
    /// Alpha wave phase (for continuous oscillation)
    alpha_phase: f64,
    /// Beta wave phase
    beta_phase: f64,
    /// Theta wave phase
    theta_phase: f64,
    /// Noise seed for reproducibility
    noise_seed: u32,
}

#[wasm_bindgen]
impl DemoRunner {
    /// Create a new demo runner
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            time_ms: 0.0,
            scenario: DemoScenario::Relaxed,
            running: false,
            eeg_sample_rate: 250,
            fnirs_sample_rate: 10,
            eeg_accum: 0.0,
            fnirs_accum: 0.0,
            alpha_phase: 0.0,
            beta_phase: 0.0,
            theta_phase: 0.0,
            noise_seed: 12345,
        }
    }

    /// Start the demo
    #[wasm_bindgen]
    pub fn start(&mut self) {
        self.running = true;
        web_sys::console::log_1(&"Demo mode started".into());
    }

    /// Stop the demo
    #[wasm_bindgen]
    pub fn stop(&mut self) {
        self.running = false;
        web_sys::console::log_1(&"Demo mode stopped".into());
    }

    /// Check if demo is running
    #[wasm_bindgen]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Set the demo scenario
    #[wasm_bindgen]
    pub fn set_scenario(&mut self, scenario: DemoScenario) {
        self.scenario = scenario;
        web_sys::console::log_1(&format!("Demo scenario: {:?}", scenario).into());
    }

    /// Set EEG sample rate
    #[wasm_bindgen]
    pub fn set_eeg_sample_rate(&mut self, rate: u32) {
        self.eeg_sample_rate = rate.clamp(100, 1000);
    }

    /// Set fNIRS sample rate
    #[wasm_bindgen]
    pub fn set_fnirs_sample_rate(&mut self, rate: u32) {
        self.fnirs_sample_rate = rate.clamp(1, 100);
    }

    /// Get current simulation time in ms
    #[wasm_bindgen]
    pub fn current_time_ms(&self) -> f64 {
        self.time_ms
    }

    /// Get current scenario name
    #[wasm_bindgen]
    pub fn scenario_name(&self) -> String {
        format!("{:?}", self.scenario)
    }

    /// Advance simulation and return EEG samples generated
    /// Returns number of samples generated during this tick
    #[wasm_bindgen]
    pub fn tick(&mut self, delta_ms: f64) -> u32 {
        if !self.running {
            return 0;
        }

        self.time_ms += delta_ms;
        self.eeg_accum += delta_ms;
        self.fnirs_accum += delta_ms;

        // Update oscillator phases
        let dt = delta_ms / 1000.0;
        self.alpha_phase += 10.0 * std::f64::consts::TAU * dt; // 10 Hz
        self.beta_phase += 20.0 * std::f64::consts::TAU * dt;  // 20 Hz
        self.theta_phase += 6.0 * std::f64::consts::TAU * dt;  // 6 Hz

        // Keep phases bounded
        if self.alpha_phase > std::f64::consts::TAU {
            self.alpha_phase -= std::f64::consts::TAU;
        }
        if self.beta_phase > std::f64::consts::TAU {
            self.beta_phase -= std::f64::consts::TAU;
        }
        if self.theta_phase > std::f64::consts::TAU {
            self.theta_phase -= std::f64::consts::TAU;
        }

        // Calculate samples to generate
        let eeg_period_ms = 1000.0 / self.eeg_sample_rate as f64;
        let samples = (self.eeg_accum / eeg_period_ms).floor() as u32;
        self.eeg_accum -= samples as f64 * eeg_period_ms;

        samples
    }

    /// Generate EEG sample for current time
    /// Returns 8 channel values in µV
    #[wasm_bindgen]
    pub fn generate_eeg(&mut self) -> Vec<f32> {
        let modifiers = self.scenario.band_modifiers();
        let mut channels = Vec::with_capacity(8);

        for (i, &baseline) in DEMO_EEG_BASELINE.iter().enumerate() {
            // Generate oscillatory components
            let alpha = 15.0 * modifiers[2] as f64 * self.alpha_phase.sin();
            let beta = 5.0 * modifiers[3] as f64 * self.beta_phase.sin();
            let theta = 10.0 * modifiers[1] as f64 * self.theta_phase.sin();

            // Add channel-specific phase offset
            let phase_offset = (i as f64 * 0.3).sin() * 0.5;

            // Generate pseudo-random noise
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0 - 1.0) * 3.0;

            let value = baseline + (alpha + beta + theta) as f32 * (1.0 + phase_offset as f32) + noise;
            channels.push(value);
        }

        channels
    }

    /// Generate fNIRS sample (HbO2, HbR for 4 optodes)
    /// Returns 8 values: [HbO2_0, HbO2_1, HbO2_2, HbO2_3, HbR_0, HbR_1, HbR_2, HbR_3]
    #[wasm_bindgen]
    pub fn generate_fnirs(&mut self) -> Vec<f32> {
        let arousal = self.scenario.arousal_level();
        let mut values = Vec::with_capacity(8);

        // HbO2 values
        for &base in &DEMO_FNIRS_HBO2 {
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0 - 1.0) * 0.005;
            let activity_mod = 1.0 + arousal * 0.3;
            values.push(base * activity_mod + noise);
        }

        // HbR values
        for &base in &DEMO_FNIRS_HBR {
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0 - 1.0) * 0.002;
            let activity_mod = 1.0 - arousal * 0.2;
            values.push(base * activity_mod + noise);
        }

        values
    }

    /// Generate EMG RMS values (8 facial muscles)
    #[wasm_bindgen]
    pub fn generate_emg(&mut self) -> Vec<f32> {
        let modifiers = self.scenario.emg_modifiers();
        let mut values = Vec::with_capacity(8);

        for (i, &base) in DEMO_EMG_RMS.iter().enumerate() {
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0) * 0.5;
            values.push(base * modifiers[i] + noise);
        }

        values
    }

    /// Generate EDA values (SCL and SCR for 4 sites)
    /// Returns 8 values: [SCL_0..3, SCR_0..3]
    #[wasm_bindgen]
    pub fn generate_eda(&mut self) -> Vec<f32> {
        let arousal = self.scenario.arousal_level();
        let mut values = Vec::with_capacity(8);

        // SCL values (tonic)
        for &base in &DEMO_EDA_SCL {
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0 - 1.0) * 0.1;
            values.push(base * (1.0 + arousal * 0.5) + noise);
        }

        // SCR values (phasic) - more variable with arousal
        for &base in &DEMO_EDA_SCR {
            self.noise_seed = self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.noise_seed >> 16) as f32 / 32768.0) * arousal;
            values.push(base * (1.0 + arousal * 2.0) + noise);
        }

        values
    }

    /// Generate band power values for VR preview
    #[wasm_bindgen]
    pub fn generate_band_power(&self) -> Vec<f32> {
        let modifiers = self.scenario.band_modifiers();
        vec![
            DEMO_BAND_POWER.delta * modifiers[0],
            DEMO_BAND_POWER.theta * modifiers[1],
            DEMO_BAND_POWER.alpha * modifiers[2],
            DEMO_BAND_POWER.beta * modifiers[3],
            DEMO_BAND_POWER.gamma * modifiers[4],
        ]
    }

    /// Check if fNIRS sample should be generated this tick
    #[wasm_bindgen]
    pub fn should_sample_fnirs(&mut self) -> bool {
        let fnirs_period_ms = 1000.0 / self.fnirs_sample_rate as f64;
        if self.fnirs_accum >= fnirs_period_ms {
            self.fnirs_accum -= fnirs_period_ms;
            true
        } else {
            false
        }
    }

    /// Reset the demo to initial state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.time_ms = 0.0;
        self.eeg_accum = 0.0;
        self.fnirs_accum = 0.0;
        self.alpha_phase = 0.0;
        self.beta_phase = 0.0;
        self.theta_phase = 0.0;
        self.noise_seed = 12345;
    }
}

impl Default for DemoRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-serializable demo fixture data for static preview
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemoFixture {
    /// Fixture name
    pub name: String,
    /// Description
    pub description: String,
    /// Duration in seconds
    pub duration_s: f32,
    /// EEG samples (each inner vec is 8 channels)
    pub eeg_samples: Vec<Vec<f32>>,
    /// fNIRS samples (each inner vec is 8 values: 4 HbO2 + 4 HbR)
    pub fnirs_samples: Vec<Vec<f32>>,
    /// EMG samples (each inner vec is 8 channels)
    pub emg_samples: Vec<Vec<f32>>,
    /// EDA samples (each inner vec is 8 values: 4 SCL + 4 SCR)
    pub eda_samples: Vec<Vec<f32>>,
    /// Band power timeline (each inner vec is 5 bands)
    pub band_power: Vec<Vec<f32>>,
    /// Sample rate for EEG (Hz)
    pub eeg_sample_rate: u32,
    /// Sample rate for fNIRS (Hz)
    pub fnirs_sample_rate: u32,
}

impl DemoFixture {
    /// Generate a demo fixture from a scenario
    pub fn from_scenario(scenario: DemoScenario, duration_s: f32) -> Self {
        let mut runner = DemoRunner::new();
        runner.set_scenario(scenario);
        runner.start();

        let eeg_sample_rate = 250;
        let fnirs_sample_rate = 10;
        let total_eeg_samples = (duration_s * eeg_sample_rate as f32) as usize;
        let total_fnirs_samples = (duration_s * fnirs_sample_rate as f32) as usize;

        let mut eeg_samples = Vec::with_capacity(total_eeg_samples);
        let mut fnirs_samples = Vec::with_capacity(total_fnirs_samples);
        let mut emg_samples = Vec::with_capacity(total_eeg_samples);
        let mut eda_samples = Vec::with_capacity(total_fnirs_samples);
        let mut band_power = Vec::with_capacity(total_fnirs_samples);

        let dt_ms = 1000.0 / eeg_sample_rate as f64;
        let fnirs_interval = eeg_sample_rate / fnirs_sample_rate;

        for i in 0..total_eeg_samples {
            runner.tick(dt_ms);
            eeg_samples.push(runner.generate_eeg());
            emg_samples.push(runner.generate_emg());

            if i % fnirs_interval as usize == 0 {
                fnirs_samples.push(runner.generate_fnirs());
                eda_samples.push(runner.generate_eda());
                band_power.push(runner.generate_band_power());
            }
        }

        Self {
            name: format!("{:?}", scenario),
            description: format!("Demo data for {:?} brain state", scenario),
            duration_s,
            eeg_samples,
            fnirs_samples,
            emg_samples,
            eda_samples,
            band_power,
            eeg_sample_rate,
            fnirs_sample_rate,
        }
    }
}

/// WASM-accessible demo fixture generator
#[wasm_bindgen]
pub fn generate_demo_fixture_json(scenario: DemoScenario, duration_s: f32) -> String {
    let fixture = DemoFixture::from_scenario(scenario, duration_s);
    serde_json::to_string_pretty(&fixture).unwrap_or_else(|_| "{}".to_string())
}

/// Load and parse a demo fixture from JSON
#[wasm_bindgen]
pub fn parse_demo_fixture(json: &str) -> Result<JsValue, JsValue> {
    let fixture: DemoFixture = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;
    serde_wasm_bindgen::to_value(&fixture)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_runner_creation() {
        let runner = DemoRunner::new();
        assert!(!runner.is_running());
        assert_eq!(runner.current_time_ms(), 0.0);
    }

    #[test]
    fn test_demo_runner_generates_samples() {
        let mut runner = DemoRunner::new();
        runner.start();
        let samples = runner.tick(100.0); // 100ms = 25 samples at 250Hz
        assert!(samples > 0);
    }

    #[test]
    fn test_demo_scenario_modifiers() {
        let relaxed = DemoScenario::Relaxed.band_modifiers();
        assert!(relaxed[2] > 1.5); // High alpha in relaxed state

        let focused = DemoScenario::Focused.band_modifiers();
        assert!(focused[3] > 1.5); // High beta in focused state
    }

    #[test]
    fn test_demo_fixture_generation() {
        let fixture = DemoFixture::from_scenario(DemoScenario::Relaxed, 1.0);
        assert_eq!(fixture.eeg_samples.len(), 250); // 1 second at 250 Hz
        assert_eq!(fixture.fnirs_samples.len(), 10); // 1 second at 10 Hz
    }
}

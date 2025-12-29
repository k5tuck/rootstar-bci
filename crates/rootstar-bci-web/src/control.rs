//! Stimulation control panel
//!
//! Provides a web interface for controlling neurostimulation parameters.

use rootstar_bci_core::types::{StimMode, StimParams};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Stimulation control panel
#[wasm_bindgen]
pub struct StimControlPanel {
    /// Current parameters
    params: StimParams,
    /// Whether stimulation is active
    active: bool,
    /// Safety lock (requires explicit unlock)
    locked: bool,
    /// Elapsed time in milliseconds
    elapsed_ms: u32,
    /// Last validation result
    last_error: Option<String>,
}

#[wasm_bindgen]
impl StimControlPanel {
    /// Create a new control panel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            params: StimParams::tdcs(true, 0, 0, 30),
            active: false,
            locked: true,
            elapsed_ms: 0,
            last_error: None,
        }
    }

    /// Unlock the safety lock
    pub fn unlock(&mut self, confirmation: &str) -> bool {
        // Require explicit confirmation
        if confirmation == "I understand the risks" {
            self.locked = false;
            true
        } else {
            false
        }
    }

    /// Lock the panel
    pub fn lock(&mut self) {
        self.locked = true;
        self.active = false;
    }

    /// Check if locked
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Check if stimulation is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Set stimulation mode
    pub fn set_mode(&mut self, mode_str: &str) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change mode while active"));
        }

        let mode = match mode_str.to_lowercase().as_str() {
            "off" => StimMode::Off,
            "tdcs_anodal" | "anodal" => StimMode::TdcsAnodal,
            "tdcs_cathodal" | "cathodal" => StimMode::TdcsCathodal,
            "tacs" => StimMode::Tacs,
            "pbm" => StimMode::Pbm,
            _ => return Err(JsValue::from_str("Unknown stimulation mode")),
        };

        self.params.mode = mode;
        Ok(())
    }

    /// Set amplitude in microamps
    pub fn set_amplitude(&mut self, amplitude_ua: u16) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change amplitude while active"));
        }

        if amplitude_ua > 2000 {
            return Err(JsValue::from_str(
                "Amplitude exceeds safety limit (2000 µA)",
            ));
        }

        self.params.amplitude_ua = amplitude_ua;
        Ok(())
    }

    /// Set frequency for tACS (Hz)
    pub fn set_frequency(&mut self, frequency_hz: u16) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change frequency while active"));
        }

        if frequency_hz > 100 {
            return Err(JsValue::from_str("Frequency exceeds safety limit (100 Hz)"));
        }

        self.params.frequency_hz = frequency_hz;
        Ok(())
    }

    /// Set duration in milliseconds
    pub fn set_duration(&mut self, duration_ms: u32) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change duration while active"));
        }

        // Maximum 30 minutes
        if duration_ms > 30 * 60 * 1000 {
            return Err(JsValue::from_str(
                "Duration exceeds safety limit (30 minutes)",
            ));
        }

        self.params.duration_ms = duration_ms;
        Ok(())
    }

    /// Set ramp time in milliseconds
    pub fn set_ramp(&mut self, ramp_ms: u16) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change ramp while active"));
        }

        if ramp_ms < 10 {
            return Err(JsValue::from_str("Ramp time too short (minimum 10 ms)"));
        }

        self.params.ramp_ms = ramp_ms;
        Ok(())
    }

    /// Validate all parameters
    pub fn validate(&mut self) -> Result<(), JsValue> {
        match self.params.validate() {
            Ok(()) => {
                self.last_error = None;
                Ok(())
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                self.last_error = Some(msg.clone());
                Err(JsValue::from_str(&msg))
            }
        }
    }

    /// Start stimulation
    pub fn start(&mut self) -> Result<(), JsValue> {
        if self.locked {
            return Err(JsValue::from_str("Panel is locked"));
        }

        if self.active {
            return Err(JsValue::from_str("Stimulation already active"));
        }

        // Validate before starting
        self.validate()?;

        self.active = true;
        self.elapsed_ms = 0;
        web_sys::console::log_1(&"Stimulation started".into());

        Ok(())
    }

    /// Stop stimulation
    pub fn stop(&mut self) {
        if self.active {
            self.active = false;
            web_sys::console::log_1(
                &format!("Stimulation stopped after {} ms", self.elapsed_ms).into(),
            );
        }
    }

    /// Emergency stop (immediate)
    pub fn emergency_stop(&mut self) {
        self.active = false;
        self.locked = true;
        web_sys::console::warn_1(&"EMERGENCY STOP activated".into());
    }

    /// Update elapsed time (call from animation frame)
    pub fn tick(&mut self, delta_ms: u32) {
        if self.active {
            self.elapsed_ms += delta_ms;

            // Check if duration exceeded
            if self.elapsed_ms >= self.params.duration_ms {
                self.stop();
            }
        }
    }

    /// Get remaining time in milliseconds
    pub fn remaining_ms(&self) -> u32 {
        if self.active {
            self.params.duration_ms.saturating_sub(self.elapsed_ms)
        } else {
            self.params.duration_ms
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> u32 {
        self.elapsed_ms
    }

    /// Get progress as percentage (0-100)
    pub fn progress(&self) -> f32 {
        if self.params.duration_ms == 0 {
            0.0
        } else {
            (self.elapsed_ms as f32 / self.params.duration_ms as f32 * 100.0).min(100.0)
        }
    }

    /// Get current amplitude (accounting for ramp)
    pub fn current_amplitude(&self) -> u16 {
        if !self.active {
            return 0;
        }

        let ramp_ms = self.params.ramp_ms as u32;
        let amplitude = self.params.amplitude_ua as f32;

        if self.elapsed_ms < ramp_ms {
            // Ramping up
            (amplitude * self.elapsed_ms as f32 / ramp_ms as f32) as u16
        } else if self.elapsed_ms > self.params.duration_ms - ramp_ms {
            // Ramping down
            let remaining = self.params.duration_ms - self.elapsed_ms;
            (amplitude * remaining as f32 / ramp_ms as f32) as u16
        } else {
            // Steady state
            self.params.amplitude_ua
        }
    }

    /// Get parameters as JSON
    pub fn get_params_js(&self) -> Result<JsValue, JsValue> {
        let params_json = StimParamsJson {
            mode: format!("{:?}", self.params.mode),
            amplitude_ua: self.params.amplitude_ua,
            frequency_hz: self.params.frequency_hz,
            duration_ms: self.params.duration_ms,
            ramp_ms: self.params.ramp_ms,
        };

        serde_wasm_bindgen::to_value(&params_json)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Set parameters from JSON
    pub fn set_params_js(&mut self, params_js: JsValue) -> Result<(), JsValue> {
        if self.active {
            return Err(JsValue::from_str("Cannot change parameters while active"));
        }

        let params: StimParamsJson = serde_wasm_bindgen::from_value(params_js)
            .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;

        self.set_mode(&params.mode)?;
        self.set_amplitude(params.amplitude_ua)?;
        self.set_frequency(params.frequency_hz)?;
        self.set_duration(params.duration_ms)?;
        self.set_ramp(params.ramp_ms)?;

        Ok(())
    }

    /// Get last error message
    pub fn get_last_error(&self) -> Option<String> {
        self.last_error.clone()
    }
}

impl Default for StimControlPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-serializable stimulation parameters
#[derive(Serialize, Deserialize)]
struct StimParamsJson {
    mode: String,
    amplitude_ua: u16,
    frequency_hz: u16,
    duration_ms: u32,
    ramp_ms: u16,
}

/// Stimulation preset configurations
#[wasm_bindgen]
pub struct StimPreset {
    name: String,
    description: String,
    params: StimParams,
}

#[wasm_bindgen]
impl StimPreset {
    /// Get preset name
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Get preset description
    pub fn description(&self) -> String {
        self.description.clone()
    }

    /// Standard tDCS anodal preset (1 mA, 20 min)
    pub fn standard_anodal() -> Self {
        Self {
            name: String::from("Standard Anodal"),
            description: String::from("1 mA anodal tDCS for 20 minutes with 30s ramp"),
            params: StimParams::tdcs(true, 1000, 20 * 60 * 1000, 30 * 1000),
        }
    }

    /// Standard tDCS cathodal preset (1 mA, 20 min)
    pub fn standard_cathodal() -> Self {
        Self {
            name: String::from("Standard Cathodal"),
            description: String::from("1 mA cathodal tDCS for 20 minutes with 30s ramp"),
            params: StimParams {
                mode: StimMode::TdcsCathodal,
                amplitude_ua: 1000,
                frequency_hz: 0,
                duration_ms: 20 * 60 * 1000,
                ramp_ms: 30 * 1000,
            },
        }
    }

    /// Alpha tACS preset (10 Hz, 1 mA)
    pub fn alpha_tacs() -> Self {
        Self {
            name: String::from("Alpha tACS"),
            description: String::from("10 Hz tACS at 1 mA for 20 minutes"),
            params: StimParams::tacs(1000, 10, 20 * 60 * 1000, 30 * 1000),
        }
    }

    /// Gamma tACS preset (40 Hz, 1 mA)
    pub fn gamma_tacs() -> Self {
        Self {
            name: String::from("Gamma tACS"),
            description: String::from("40 Hz tACS at 1 mA for 20 minutes"),
            params: StimParams::tacs(1000, 40, 20 * 60 * 1000, 30 * 1000),
        }
    }

    /// Low intensity preset for beginners (0.5 mA, 10 min)
    pub fn low_intensity() -> Self {
        Self {
            name: String::from("Low Intensity"),
            description: String::from("0.5 mA anodal tDCS for 10 minutes"),
            params: StimParams::tdcs(true, 500, 10 * 60 * 1000, 30 * 1000),
        }
    }
}

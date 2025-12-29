//! Neurostimulation Driver
//!
//! Driver for tDCS (transcranial Direct Current Stimulation) and
//! tACS (transcranial Alternating Current Stimulation).
//!
//! # Safety
//!
//! This driver implements software safety limits but REQUIRES hardware
//! current limiting for safe operation. Never rely on software alone.
//!
//! Safety limits:
//! - Maximum current: 2000 µA (2 mA)
//! - Maximum duration: 30 minutes
//! - Minimum ramp time: 10 ms
//! - Maximum tACS frequency: 100 Hz

use rootstar_bci_core::error::StimError;
use rootstar_bci_core::types::{StimMode, StimParams};

/// Stimulation output state
#[derive(Copy, Clone, Debug, Default)]
pub struct StimState {
    /// Current stimulation parameters
    pub params: StimParams,
    /// Whether stimulation is currently active
    pub active: bool,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u32,
    /// Current amplitude (may differ from target during ramp)
    pub current_amplitude_ua: u16,
}

/// Stimulation driver (placeholder for DAC hardware)
///
/// In a real implementation, this would control a DAC (e.g., DAC8564)
/// connected to a voltage-to-current converter (Howland current source).
pub struct StimDriver {
    state: StimState,
    hardware_enabled: bool,
}

impl StimDriver {
    /// Create a new stimulation driver
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: StimState {
                params: StimParams {
                    mode: StimMode::Off,
                    amplitude_ua: 0,
                    frequency_hz: 0,
                    duration_ms: 0,
                    ramp_ms: 0,
                },
                active: false,
                elapsed_ms: 0,
                current_amplitude_ua: 0,
            },
            hardware_enabled: false,
        }
    }

    /// Initialize the stimulation hardware
    pub fn init(&mut self) -> Result<(), StimError> {
        // In real implementation: initialize DAC, verify safety circuit
        self.hardware_enabled = true;
        Ok(())
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> &StimState {
        &self.state
    }

    /// Start stimulation with given parameters
    ///
    /// # Safety
    ///
    /// This function validates parameters but hardware current limiting
    /// is REQUIRED for safe operation.
    pub fn start(&mut self, params: StimParams) -> Result<(), StimError> {
        // Validate parameters
        params.validate()?;

        if !self.hardware_enabled {
            return Err(StimError::DacError);
        }

        self.state = StimState {
            params,
            active: true,
            elapsed_ms: 0,
            current_amplitude_ua: 0,
        };

        Ok(())
    }

    /// Stop stimulation immediately
    pub fn stop(&mut self) {
        self.state.active = false;
        self.state.current_amplitude_ua = 0;
        // In real implementation: set DAC to zero
    }

    /// Update stimulation state (call periodically, e.g., every 1ms)
    ///
    /// Handles ramping and duration timing.
    pub fn update(&mut self, dt_ms: u32) {
        if !self.state.active {
            return;
        }

        self.state.elapsed_ms = self.state.elapsed_ms.saturating_add(dt_ms);

        // Check duration limit
        if self.state.elapsed_ms >= self.state.params.duration_ms {
            self.stop();
            return;
        }

        // Calculate target amplitude with ramping
        let target = self.state.params.amplitude_ua;
        let ramp_ms = u32::from(self.state.params.ramp_ms);

        let amplitude = if self.state.elapsed_ms < ramp_ms {
            // Ramp up
            (u32::from(target) * self.state.elapsed_ms / ramp_ms) as u16
        } else if self.state.elapsed_ms > self.state.params.duration_ms.saturating_sub(ramp_ms) {
            // Ramp down
            let remaining = self.state.params.duration_ms - self.state.elapsed_ms;
            (u32::from(target) * remaining / ramp_ms) as u16
        } else {
            // Full amplitude
            target
        };

        self.state.current_amplitude_ua = amplitude;

        // In real implementation: update DAC value based on mode
        // For tACS, would compute sin wave at frequency_hz
    }

    /// Check if hardware safety circuit has triggered
    pub fn check_safety(&self) -> Result<(), StimError> {
        // In real implementation: read safety comparator output
        // If triggered, return Err(StimError::HardwareSafetyTriggered)
        Ok(())
    }
}

impl Default for StimDriver {
    fn default() -> Self {
        Self::new()
    }
}

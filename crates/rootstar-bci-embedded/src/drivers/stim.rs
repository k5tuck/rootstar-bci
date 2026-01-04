//! Neurostimulation Driver
//!
//! Driver for tDCS (transcranial Direct Current Stimulation) and
//! tACS (transcranial Alternating Current Stimulation).
//!
//! # Hardware
//!
//! Uses TI DAC8564 (16-bit quad DAC) connected via SPI to a Howland
//! current source for voltage-to-current conversion.
//!
//! DAC8564 pinout (typical):
//! - SCLK: SPI clock
//! - DIN: SPI MOSI
//! - SYNC: SPI chip select (active low)
//! - LDAC: Load DAC (active low, tie to GND for immediate update)
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

use embedded_hal::spi::SpiDevice;
use embedded_hal::digital::OutputPin;
use libm::sinf;

use rootstar_bci_core::error::StimError;
use rootstar_bci_core::types::{StimMode, StimParams};

// ============================================================================
// DAC8564 Constants
// ============================================================================

/// DAC8564 command: Write to input register
const DAC_CMD_WRITE_INPUT: u8 = 0x00;

/// DAC8564 command: Update output register (from input)
const DAC_CMD_UPDATE_OUTPUT: u8 = 0x10;

/// DAC8564 command: Write to input and update all outputs
const DAC_CMD_WRITE_UPDATE_ALL: u8 = 0x20;

/// DAC8564 command: Write and update this channel
const DAC_CMD_WRITE_UPDATE: u8 = 0x30;

/// DAC8564 channel A (used for tDCS anode current)
const DAC_CH_A: u8 = 0x00;

/// DAC8564 channel B (used for tDCS cathode current)
const DAC_CH_B: u8 = 0x01;

/// DAC8564 channel C (used for tACS waveform)
const DAC_CH_C: u8 = 0x02;

/// DAC8564 channel D (used for reference/calibration)
const DAC_CH_D: u8 = 0x03;

/// DAC8564 all channels
const DAC_CH_ALL: u8 = 0x0F;

/// DAC resolution (16-bit)
const DAC_RESOLUTION: u32 = 65535;

/// DAC reference voltage (internal 2.5V reference)
const DAC_VREF_MV: u32 = 2500;

/// Howland current source transconductance (µA/mV)
/// Calibrated for the specific hardware design
/// With 10k sense resistor: 1mA = 10mV, so gm = 100 µA/mV
const HOWLAND_TRANSCONDUCTANCE: u32 = 100;

/// Maximum safe DAC code for 2mA limit
const MAX_SAFE_DAC_CODE: u16 = (2000 / HOWLAND_TRANSCONDUCTANCE * DAC_RESOLUTION / DAC_VREF_MV) as u16;

/// Safety check GPIO input (hardware overcurrent comparator output)
/// High = safe, Low = overcurrent triggered

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
    /// Current DAC code being output
    pub dac_code: u16,
    /// Phase accumulator for tACS (in 1/65536 of a cycle)
    pub phase_accumulator: u32,
}

// ============================================================================
// DAC8564 Hardware Driver
// ============================================================================

/// DAC8564 driver using embedded-hal SPI traits.
pub struct Dac8564<SPI> {
    spi: SPI,
}

impl<SPI, E> Dac8564<SPI>
where
    SPI: SpiDevice<u8, Error = E>,
{
    /// Create a new DAC8564 driver.
    #[must_use]
    pub fn new(spi: SPI) -> Self {
        Self { spi }
    }

    /// Write a value to a DAC channel.
    ///
    /// # Arguments
    /// * `channel` - Channel number (0-3 for A-D, 15 for all)
    /// * `value` - 16-bit DAC code (0-65535)
    /// * `update_immediately` - If true, update output immediately
    pub fn write_channel(&mut self, channel: u8, value: u16, update_immediately: bool) -> Result<(), E> {
        // DAC8564 24-bit command format:
        // [23:22] Don't care
        // [21:20] Command: 00=write input, 01=update, 10=write+update all, 11=write+update
        // [19:16] Channel address
        // [15:0]  Data

        let command = if update_immediately {
            DAC_CMD_WRITE_UPDATE
        } else {
            DAC_CMD_WRITE_INPUT
        };

        let cmd_byte = command | (channel & 0x0F);
        let data_high = (value >> 8) as u8;
        let data_low = (value & 0xFF) as u8;

        let buffer = [cmd_byte, data_high, data_low];
        self.spi.write(&buffer)?;

        Ok(())
    }

    /// Write to a channel and update all outputs simultaneously.
    pub fn write_and_update_all(&mut self, channel: u8, value: u16) -> Result<(), E> {
        let cmd_byte = DAC_CMD_WRITE_UPDATE_ALL | (channel & 0x0F);
        let data_high = (value >> 8) as u8;
        let data_low = (value & 0xFF) as u8;

        let buffer = [cmd_byte, data_high, data_low];
        self.spi.write(&buffer)?;

        Ok(())
    }

    /// Set all channels to zero (emergency stop).
    pub fn zero_all(&mut self) -> Result<(), E> {
        self.write_channel(DAC_CH_ALL, 0, true)
    }

    /// Set all channels to midpoint (for bipolar output).
    pub fn set_midpoint(&mut self) -> Result<(), E> {
        self.write_channel(DAC_CH_ALL, 32768, true)
    }
}

// ============================================================================
// Stimulation Driver
// ============================================================================

/// Stimulation driver with DAC8564 hardware support.
///
/// Controls a DAC8564 connected to a Howland current source for
/// precise current output.
pub struct StimDriver<SPI, SAFETY> {
    /// DAC8564 instance
    dac: Option<Dac8564<SPI>>,
    /// Safety GPIO input (hardware overcurrent detection)
    safety_pin: Option<SAFETY>,
    /// Current stimulation state
    state: StimState,
    /// Hardware initialized flag
    hardware_enabled: bool,
    /// Calibration offset (DAC codes)
    calibration_offset: i16,
    /// Calibration gain (256 = 1.0)
    calibration_gain: u16,
}

impl<SPI, SAFETY, E> StimDriver<SPI, SAFETY>
where
    SPI: SpiDevice<u8, Error = E>,
    SAFETY: embedded_hal::digital::InputPin,
{
    /// Create a new stimulation driver with hardware.
    #[must_use]
    pub fn new(spi: SPI, safety_pin: SAFETY) -> Self {
        Self {
            dac: Some(Dac8564::new(spi)),
            safety_pin: Some(safety_pin),
            state: StimState::default(),
            hardware_enabled: false,
            calibration_offset: 0,
            calibration_gain: 256, // 1.0 in fixed-point
        }
    }

    /// Initialize the stimulation hardware.
    ///
    /// Sets all DAC channels to zero and verifies safety circuit.
    pub fn init(&mut self) -> Result<(), StimError> {
        // Zero all DAC channels
        if let Some(ref mut dac) = self.dac {
            dac.zero_all().map_err(|_| StimError::DacError)?;
        }

        // Verify safety circuit is not triggered
        self.check_safety()?;

        self.hardware_enabled = true;
        Ok(())
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> &StimState {
        &self.state
    }

    /// Set calibration parameters.
    ///
    /// # Arguments
    /// * `offset` - Offset in DAC codes to add to output
    /// * `gain` - Gain multiplier (256 = 1.0, 512 = 2.0, etc.)
    pub fn set_calibration(&mut self, offset: i16, gain: u16) {
        self.calibration_offset = offset;
        self.calibration_gain = gain;
    }

    /// Convert current (µA) to DAC code.
    ///
    /// Applies calibration and enforces safety limits.
    fn current_to_dac_code(&self, current_ua: u16) -> u16 {
        // Convert µA to mV using transconductance
        let voltage_mv = u32::from(current_ua) / HOWLAND_TRANSCONDUCTANCE;

        // Convert mV to DAC code
        let raw_code = (voltage_mv * DAC_RESOLUTION / DAC_VREF_MV) as u16;

        // Apply calibration
        let calibrated = (i32::from(raw_code) * i32::from(self.calibration_gain) / 256
            + i32::from(self.calibration_offset)) as u16;

        // Enforce hardware safety limit
        calibrated.min(MAX_SAFE_DAC_CODE)
    }

    /// Start stimulation with given parameters.
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

        // Check safety before starting
        self.check_safety()?;

        self.state = StimState {
            params,
            active: true,
            elapsed_ms: 0,
            current_amplitude_ua: 0,
            dac_code: 0,
            phase_accumulator: 0,
        };

        Ok(())
    }

    /// Stop stimulation immediately (emergency stop).
    pub fn stop(&mut self) -> Result<(), StimError> {
        self.state.active = false;
        self.state.current_amplitude_ua = 0;
        self.state.dac_code = 0;

        // Set DAC to zero immediately
        if let Some(ref mut dac) = self.dac {
            dac.zero_all().map_err(|_| StimError::DacError)?;
        }

        Ok(())
    }

    /// Update stimulation state (call periodically, e.g., every 1ms).
    ///
    /// Handles ramping, duration timing, and waveform generation.
    pub fn update(&mut self, dt_ms: u32) -> Result<(), StimError> {
        if !self.state.active {
            return Ok(());
        }

        // Check safety before updating
        self.check_safety()?;

        self.state.elapsed_ms = self.state.elapsed_ms.saturating_add(dt_ms);

        // Check duration limit
        if self.state.elapsed_ms >= self.state.params.duration_ms {
            return self.stop();
        }

        // Calculate target amplitude with ramping
        let target = self.state.params.amplitude_ua;
        let ramp_ms = u32::from(self.state.params.ramp_ms);

        let envelope = if ramp_ms == 0 {
            target
        } else if self.state.elapsed_ms < ramp_ms {
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

        // Generate output based on mode
        let (dac_code, current_ua) = match self.state.params.mode {
            StimMode::Off => (0, 0),
            StimMode::Tdcs => {
                // DC output
                let code = self.current_to_dac_code(envelope);
                (code, envelope)
            }
            StimMode::Tacs => {
                // AC output (sinusoidal)
                let freq_hz = u32::from(self.state.params.frequency_hz);
                if freq_hz > 0 {
                    // Update phase accumulator
                    // Phase increment = freq * 65536 / 1000 (for 1ms update rate)
                    let phase_inc = freq_hz * 65536 / 1000 * dt_ms;
                    self.state.phase_accumulator = self.state.phase_accumulator.wrapping_add(phase_inc);

                    // Calculate sine value (-1 to 1)
                    let phase_rad = (self.state.phase_accumulator as f32) * core::f32::consts::PI * 2.0 / 65536.0;
                    let sine_val = sinf(phase_rad);

                    // Scale by envelope (output = envelope * sin(phase))
                    // For tACS, we use bipolar output around DAC midpoint
                    let midpoint = 32768u16;
                    let amplitude_code = self.current_to_dac_code(envelope);
                    let offset = (sine_val * amplitude_code as f32) as i32;
                    let code = (midpoint as i32 + offset).clamp(0, 65535) as u16;

                    // Current is absolute value of the sinusoidal output
                    let instantaneous_current = (envelope as f32 * sine_val.abs()) as u16;
                    (code, instantaneous_current)
                } else {
                    (0, 0)
                }
            }
            StimMode::Trns => {
                // Random noise stimulation (pseudo-random)
                // Use a simple LFSR for noise generation
                let noise = self.generate_noise();
                let amplitude_code = self.current_to_dac_code(envelope);
                let midpoint = 32768u16;
                let offset = ((noise as i32 - 128) * amplitude_code as i32 / 128) as i32;
                let code = (midpoint as i32 + offset).clamp(0, 65535) as u16;

                (code, envelope)
            }
        };

        self.state.current_amplitude_ua = current_ua;
        self.state.dac_code = dac_code;

        // Write to DAC
        if let Some(ref mut dac) = self.dac {
            // For tDCS, use channel A for anode
            // For tACS/tRNS, use channel C for waveform
            let channel = match self.state.params.mode {
                StimMode::Tdcs => DAC_CH_A,
                _ => DAC_CH_C,
            };
            dac.write_channel(channel, dac_code, true)
                .map_err(|_| StimError::DacError)?;
        }

        Ok(())
    }

    /// Generate pseudo-random noise for tRNS.
    fn generate_noise(&mut self) -> u8 {
        // Simple LFSR-based noise generator using phase accumulator as state
        let mut lfsr = self.state.phase_accumulator;
        if lfsr == 0 {
            lfsr = 0xACE1u32;
        }
        let bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
        lfsr = (lfsr >> 1) | (bit << 15);
        self.state.phase_accumulator = lfsr;
        (lfsr & 0xFF) as u8
    }

    /// Check if hardware safety circuit has triggered.
    pub fn check_safety(&self) -> Result<(), StimError> {
        if let Some(ref pin) = self.safety_pin {
            // Safety pin is active low (low = overcurrent)
            match pin.is_low() {
                Ok(true) => return Err(StimError::HardwareSafetyTriggered),
                Ok(false) => {} // Safe
                Err(_) => return Err(StimError::HardwareSafetyTriggered), // Assume unsafe on read error
            }
        }
        Ok(())
    }

    /// Get measured current from feedback (if available).
    ///
    /// This would typically read from an ADC connected to the current sense resistor.
    /// Returns estimated current based on DAC output as fallback.
    #[must_use]
    pub fn get_measured_current_ua(&self) -> u16 {
        // In a full implementation, this would read from an ADC
        // For now, estimate from DAC code
        let voltage_mv = (u32::from(self.state.dac_code) * DAC_VREF_MV / DAC_RESOLUTION) as u16;
        voltage_mv * (HOWLAND_TRANSCONDUCTANCE as u16)
    }
}

// ============================================================================
// Software-Only Driver (for testing without hardware)
// ============================================================================

/// Software-only stimulation driver for testing.
///
/// This driver does not require SPI or GPIO hardware.
pub struct StimDriverSoftware {
    state: StimState,
    hardware_enabled: bool,
}

impl StimDriverSoftware {
    /// Create a new software-only stimulation driver.
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
                dac_code: 0,
                phase_accumulator: 0,
            },
            hardware_enabled: false,
        }
    }

    /// Initialize (no-op for software driver).
    pub fn init(&mut self) -> Result<(), StimError> {
        self.hardware_enabled = true;
        Ok(())
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> &StimState {
        &self.state
    }

    /// Start stimulation.
    pub fn start(&mut self, params: StimParams) -> Result<(), StimError> {
        params.validate()?;

        if !self.hardware_enabled {
            return Err(StimError::DacError);
        }

        self.state = StimState {
            params,
            active: true,
            elapsed_ms: 0,
            current_amplitude_ua: 0,
            dac_code: 0,
            phase_accumulator: 0,
        };

        Ok(())
    }

    /// Stop stimulation.
    pub fn stop(&mut self) {
        self.state.active = false;
        self.state.current_amplitude_ua = 0;
        self.state.dac_code = 0;
    }

    /// Update stimulation state.
    pub fn update(&mut self, dt_ms: u32) {
        if !self.state.active {
            return;
        }

        self.state.elapsed_ms = self.state.elapsed_ms.saturating_add(dt_ms);

        if self.state.elapsed_ms >= self.state.params.duration_ms {
            self.stop();
            return;
        }

        let target = self.state.params.amplitude_ua;
        let ramp_ms = u32::from(self.state.params.ramp_ms);

        let amplitude = if ramp_ms == 0 {
            target
        } else if self.state.elapsed_ms < ramp_ms {
            (u32::from(target) * self.state.elapsed_ms / ramp_ms) as u16
        } else if self.state.elapsed_ms > self.state.params.duration_ms.saturating_sub(ramp_ms) {
            let remaining = self.state.params.duration_ms - self.state.elapsed_ms;
            (u32::from(target) * remaining / ramp_ms) as u16
        } else {
            target
        };

        self.state.current_amplitude_ua = amplitude;
    }

    /// Check safety (always safe for software driver).
    pub fn check_safety(&self) -> Result<(), StimError> {
        Ok(())
    }
}

impl Default for StimDriverSoftware {
    fn default() -> Self {
        Self::new()
    }
}

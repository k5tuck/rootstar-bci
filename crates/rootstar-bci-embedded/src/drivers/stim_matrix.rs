//! Stimulation Electrode Matrix Driver
//!
//! Provides multi-channel transcranial stimulation with electrode
//! switching matrix for targeting different brain regions.
//!
//! # Features
//!
//! - 8×8 electrode switching matrix (64 electrodes)
//! - Multi-channel DAC support for independent current sources
//! - Hardware safety interlocks
//! - Current monitoring and feedback
//!
//! # Safety
//!
//! This driver implements multiple safety layers:
//! - Software current limits
//! - Hardware overcurrent detection
//! - Impedance monitoring
//! - Emergency shutdown

use core::marker::PhantomData;

use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::SpiDevice;
use heapless::Vec;

use rootstar_bci_core::error::StimError;
use rootstar_bci_core::fingerprint::{SafetyLimits, SafetyMonitor, SafetyViolation};
use rootstar_bci_core::types::StimParams;

/// Maximum number of stimulation electrodes
pub const MAX_ELECTRODES: usize = 64;

/// Maximum number of independent current sources
pub const MAX_CURRENT_SOURCES: usize = 8;

/// Electrode state in the matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ElectrodeState {
    /// Electrode disconnected (high impedance)
    #[default]
    Disconnected,
    /// Electrode connected as anode (current source)
    Anode,
    /// Electrode connected as cathode (current sink)
    Cathode,
    /// Electrode in impedance measurement mode
    Impedance,
}

/// Matrix switch position
#[derive(Clone, Copy, Debug)]
pub struct SwitchPosition {
    /// Row in the matrix (0-7)
    pub row: u8,
    /// Column in the matrix (0-7)
    pub column: u8,
    /// Electrode state
    pub state: ElectrodeState,
}

impl SwitchPosition {
    /// Create a new switch position
    #[must_use]
    pub const fn new(row: u8, column: u8, state: ElectrodeState) -> Self {
        Self { row, column, state }
    }

    /// Get linear electrode index
    #[must_use]
    pub const fn electrode_index(&self) -> u8 {
        self.row * 8 + self.column
    }

    /// Create from electrode index
    #[must_use]
    pub const fn from_index(index: u8, state: ElectrodeState) -> Self {
        Self {
            row: index / 8,
            column: index % 8,
            state,
        }
    }
}

/// Stimulation matrix configuration
#[derive(Clone, Debug)]
pub struct MatrixConfig {
    /// Number of rows in the matrix
    pub rows: u8,
    /// Number of columns in the matrix
    pub columns: u8,
    /// Number of DAC channels
    pub n_dac_channels: u8,
    /// DAC resolution in bits
    pub dac_resolution: u8,
    /// DAC reference voltage in mV
    pub dac_vref_mv: u16,
    /// Current sense resistor in ohms
    pub sense_resistor_ohm: u16,
    /// Maximum current per channel in µA
    pub max_current_ua: u16,
    /// Enable hardware interlock
    pub hardware_interlock: bool,
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            rows: 8,
            columns: 8,
            n_dac_channels: 4,
            dac_resolution: 16,
            dac_vref_mv: 5000,
            sense_resistor_ohm: 1000,
            max_current_ua: 2000,
            hardware_interlock: true,
        }
    }
}

/// Current measurement result
#[derive(Clone, Copy, Debug)]
pub struct CurrentMeasurement {
    /// Measured current in µA
    pub current_ua: i16,
    /// Measured voltage across sense resistor in mV
    pub voltage_mv: i16,
    /// Electrode index
    pub electrode: u8,
}

/// Impedance measurement result
#[derive(Clone, Copy, Debug)]
pub struct ImpedanceMeasurement {
    /// Measured impedance in kΩ
    pub impedance_kohm: u16,
    /// Electrode index
    pub electrode: u8,
    /// Quality indicator (0-100)
    pub quality: u8,
}

/// Stimulation protocol for matrix
#[derive(Clone, Debug)]
pub struct MatrixStimProtocol {
    /// Anode electrode indices
    pub anodes: Vec<u8, 8>,
    /// Cathode electrode indices
    pub cathodes: Vec<u8, 8>,
    /// Current amplitude in µA
    pub amplitude_ua: u16,
    /// Frequency in Hz (0 for DC)
    pub frequency_hz: u16,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Ramp time in milliseconds
    pub ramp_ms: u16,
}

impl MatrixStimProtocol {
    /// Create a simple bipolar protocol
    #[must_use]
    pub fn bipolar(anode: u8, cathode: u8, amplitude_ua: u16, duration_ms: u32) -> Self {
        let mut anodes = Vec::new();
        let mut cathodes = Vec::new();
        let _ = anodes.push(anode);
        let _ = cathodes.push(cathode);

        Self {
            anodes,
            cathodes,
            amplitude_ua,
            frequency_hz: 0,
            duration_ms,
            ramp_ms: 30_000, // 30s ramp
        }
    }

    /// Create a montage from electrode names
    pub fn from_names(
        anode_names: &[&str],
        cathode_names: &[&str],
        amplitude_ua: u16,
        frequency_hz: u16,
    ) -> Self {
        let mut anodes = Vec::new();
        let mut cathodes = Vec::new();

        // Map electrode names to indices (simplified mapping)
        for name in anode_names {
            if let Some(idx) = name_to_electrode_index(name) {
                let _ = anodes.push(idx);
            }
        }
        for name in cathode_names {
            if let Some(idx) = name_to_electrode_index(name) {
                let _ = cathodes.push(idx);
            }
        }

        Self {
            anodes,
            cathodes,
            amplitude_ua,
            frequency_hz,
            duration_ms: 1200_000, // 20 minutes default
            ramp_ms: 30_000,
        }
    }
}

/// Map electrode name to matrix index
fn name_to_electrode_index(name: &str) -> Option<u8> {
    // Simplified 10-20 mapping for 8×8 matrix
    match name {
        "Fp1" => Some(0),
        "Fp2" => Some(1),
        "F7" => Some(8),
        "F3" => Some(9),
        "Fz" => Some(10),
        "F4" => Some(11),
        "F8" => Some(12),
        "FT7" => Some(16),
        "FC3" => Some(17),
        "FCz" => Some(18),
        "FC4" => Some(19),
        "FT8" => Some(20),
        "T7" => Some(24),
        "C3" => Some(25),
        "Cz" => Some(26),
        "C4" => Some(27),
        "T8" => Some(28),
        "TP7" => Some(32),
        "CP3" => Some(33),
        "CPz" => Some(34),
        "CP4" => Some(35),
        "TP8" => Some(36),
        "P7" => Some(40),
        "P3" => Some(41),
        "Pz" => Some(42),
        "P4" => Some(43),
        "P8" => Some(44),
        "O1" => Some(48),
        "Oz" => Some(49),
        "O2" => Some(50),
        _ => None,
    }
}

/// Stimulation matrix driver
pub struct StimMatrix<SPI, CS, EN, FAULT, ADC, const N_DAC: usize> {
    /// SPI for matrix control and DAC
    spi: SPI,
    /// Chip select for matrix switches
    cs_matrix: CS,
    /// Enable pin for current sources
    enable: EN,
    /// Fault detection input
    fault: FAULT,
    /// ADC for current sensing
    adc: ADC,
    /// Configuration
    config: MatrixConfig,
    /// Current electrode states
    electrode_states: [ElectrodeState; MAX_ELECTRODES],
    /// Safety monitor
    safety: SafetyMonitor,
    /// Active protocol
    active_protocol: Option<MatrixStimProtocol>,
    /// Stimulation start time (us)
    stim_start_us: u64,
    /// Phantom
    _phantom: PhantomData<fn() -> (SPI, CS, EN, FAULT, ADC)>,
}

impl<SPI, CS, EN, FAULT, ADC, E, const N_DAC: usize>
    StimMatrix<SPI, CS, EN, FAULT, ADC, N_DAC>
where
    SPI: SpiDevice<Error = E>,
    CS: OutputPin,
    EN: OutputPin,
    FAULT: InputPin,
    ADC: embedded_hal::spi::SpiDevice,
{
    /// Create a new stimulation matrix driver
    pub fn new(
        spi: SPI,
        cs_matrix: CS,
        enable: EN,
        fault: FAULT,
        adc: ADC,
        config: MatrixConfig,
    ) -> Self {
        Self {
            spi,
            cs_matrix,
            enable,
            fault,
            adc,
            config,
            electrode_states: [ElectrodeState::Disconnected; MAX_ELECTRODES],
            safety: SafetyMonitor::new(SafetyLimits::RESEARCH),
            active_protocol: None,
            stim_start_us: 0,
            _phantom: PhantomData,
        }
    }

    /// Initialize the stimulation matrix
    pub fn init(&mut self) -> Result<(), StimError> {
        // Disable all outputs
        let _ = self.enable.set_low();

        // Reset all switches to disconnected
        self.reset_matrix()?;

        // Verify no faults
        if self.check_fault() {
            return Err(StimError::HardwareSafetyTriggered);
        }

        Ok(())
    }

    /// Reset all matrix switches to disconnected
    fn reset_matrix(&mut self) -> Result<(), StimError> {
        // Send reset command to matrix controller
        let reset_cmd = [0x00u8; 8]; // All switches off
        let _ = self.cs_matrix.set_low();
        let _ = self.spi.write(&reset_cmd);
        let _ = self.cs_matrix.set_high();

        for state in &mut self.electrode_states {
            *state = ElectrodeState::Disconnected;
        }

        Ok(())
    }

    /// Check if hardware fault is detected
    #[inline]
    pub fn check_fault(&mut self) -> bool {
        self.fault.is_low().unwrap_or(true)
    }

    /// Start stimulation with given protocol
    pub fn start_stimulation(
        &mut self,
        protocol: MatrixStimProtocol,
        timestamp_us: u64,
    ) -> Result<(), StimError> {
        // Validate against safety limits
        self.validate_protocol(&protocol)?;

        // Start safety monitor session
        self.safety
            .start_session(timestamp_us)
            .map_err(|_| StimError::HardwareSafetyTriggered)?;

        // Validate with safety monitor
        let duration_min = (protocol.duration_ms / 60000) as u16;
        self.safety
            .validate_protocol(
                protocol.amplitude_ua,
                protocol.frequency_hz,
                duration_min,
                25, // Electrode size
            )
            .map_err(|_| StimError::HardwareSafetyTriggered)?;

        // Configure matrix switches
        self.configure_electrodes(&protocol)?;

        // Store protocol and start time
        self.active_protocol = Some(protocol);
        self.stim_start_us = timestamp_us;

        // Enable output (starts at ramp)
        let _ = self.enable.set_high();

        Ok(())
    }

    /// Validate protocol against safety limits
    fn validate_protocol(&self, protocol: &MatrixStimProtocol) -> Result<(), StimError> {
        if protocol.amplitude_ua > self.config.max_current_ua {
            return Err(StimError::CurrentExceedsLimit {
                requested_ua: protocol.amplitude_ua,
                maximum_ua: self.config.max_current_ua,
            });
        }

        if protocol.anodes.is_empty() || protocol.cathodes.is_empty() {
            return Err(StimError::DacError);
        }

        // Check for electrode conflicts
        for anode in &protocol.anodes {
            if protocol.cathodes.contains(anode) {
                return Err(StimError::DacError);
            }
        }

        Ok(())
    }

    /// Configure electrode switches
    fn configure_electrodes(&mut self, protocol: &MatrixStimProtocol) -> Result<(), StimError> {
        // Build switch configuration
        let mut switch_config = [0u8; 8];

        for &anode in &protocol.anodes {
            let idx = anode as usize;
            if idx < MAX_ELECTRODES {
                self.electrode_states[idx] = ElectrodeState::Anode;
                // Set corresponding bit in switch config
                switch_config[idx / 8] |= 0x80 >> (idx % 8);
            }
        }

        for &cathode in &protocol.cathodes {
            let idx = cathode as usize;
            if idx < MAX_ELECTRODES {
                self.electrode_states[idx] = ElectrodeState::Cathode;
                // Set corresponding bit with cathode flag
                switch_config[idx / 8] |= 0x40 >> (idx % 8);
            }
        }

        // Send to matrix controller
        let _ = self.cs_matrix.set_low();
        let _ = self.spi.write(&switch_config);
        let _ = self.cs_matrix.set_high();

        Ok(())
    }

    /// Update stimulation (call periodically for ramping and safety)
    pub fn update(&mut self, timestamp_us: u64) -> Result<(), StimError> {
        // Check hardware fault
        if self.check_fault() {
            self.emergency_stop();
            return Err(StimError::HardwareSafetyTriggered);
        }

        let protocol = match &self.active_protocol {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        // Calculate elapsed time
        let elapsed_us = timestamp_us.saturating_sub(self.stim_start_us);
        let elapsed_ms = (elapsed_us / 1000) as u32;

        // Check if stimulation complete
        if elapsed_ms >= protocol.duration_ms {
            self.stop_stimulation()?;
            return Ok(());
        }

        // Calculate ramp factor
        let ramp_factor = self.calculate_ramp_factor(elapsed_ms, &protocol);

        // Set DAC output
        let target_current = (protocol.amplitude_ua as f32 * ramp_factor) as u16;
        self.set_current(target_current)?;

        // Read actual current for safety monitoring
        let measured = self.measure_current()?;

        // Update safety monitor
        self.safety
            .monitor(timestamp_us, measured.current_ua as u16, 5) // Assume 5kΩ
            .map_err(|_| StimError::HardwareSafetyTriggered)?;

        Ok(())
    }

    /// Calculate ramp factor (0.0 to 1.0)
    fn calculate_ramp_factor(&self, elapsed_ms: u32, protocol: &MatrixStimProtocol) -> f32 {
        let ramp_ms = protocol.ramp_ms as u32;
        let main_duration = protocol.duration_ms.saturating_sub(ramp_ms * 2);

        if elapsed_ms < ramp_ms {
            // Ramp up
            elapsed_ms as f32 / ramp_ms as f32
        } else if elapsed_ms < ramp_ms + main_duration {
            // Plateau
            1.0
        } else {
            // Ramp down
            let ramp_elapsed = elapsed_ms - ramp_ms - main_duration;
            1.0 - (ramp_elapsed as f32 / ramp_ms as f32)
        }
    }

    /// Set output current
    fn set_current(&mut self, current_ua: u16) -> Result<(), StimError> {
        // Convert current to DAC value
        // I = V / R, V = DAC * Vref / 2^bits
        // DAC = I * R * 2^bits / Vref
        let r = self.config.sense_resistor_ohm as u32;
        let vref = self.config.dac_vref_mv as u32;
        let bits = self.config.dac_resolution as u32;

        let dac_value = (current_ua as u32 * r * (1 << bits)) / (vref * 1000);
        let dac_value = dac_value.min((1 << bits) - 1) as u16;

        // Send to DAC (assuming DAC8564 format)
        let dac_cmd = [
            0x18, // Write to DAC A, update all
            (dac_value >> 8) as u8,
            dac_value as u8,
        ];

        let _ = self.spi.write(&dac_cmd);

        Ok(())
    }

    /// Measure actual output current.
    ///
    /// Reads the current sense ADC to get the actual current flowing through
    /// the stimulation electrodes. Uses the sense resistor to convert voltage
    /// to current.
    fn measure_current(&mut self) -> Result<CurrentMeasurement, StimError> {
        // Read ADC value from current sense resistor
        // ADC command format: [start_bit | single_ended | channel | don't_care]
        // Using MCP3008 or similar 10-bit ADC
        let adc_cmd = [0x01, 0x80, 0x00]; // Start, single-ended, channel 0
        let mut rx_buf = [0u8; 3];

        let _ = self.adc.transfer(&mut rx_buf, &adc_cmd);

        // Extract 10-bit ADC value
        // Response: [x, x, MSB:2bits, LSB:8bits]
        let adc_value = (((rx_buf[1] & 0x03) as u16) << 8) | (rx_buf[2] as u16);

        // Convert ADC value to voltage
        // V = ADC * Vref / 1024 (10-bit ADC)
        let adc_vref_mv = 3300u32; // Typical 3.3V ADC reference
        let voltage_mv = (adc_value as u32 * adc_vref_mv / 1024) as i16;

        // Convert voltage to current using sense resistor
        // I = V / R (in µA = mV * 1000 / Ω)
        let current_ua = (voltage_mv as i32 * 1000 / self.config.sense_resistor_ohm as i32) as i16;

        // Determine active electrode for the measurement
        let active_electrode = self.electrode_states
            .iter()
            .position(|&s| s == ElectrodeState::Anode)
            .unwrap_or(0) as u8;

        Ok(CurrentMeasurement {
            current_ua,
            voltage_mv,
            electrode: active_electrode,
        })
    }

    /// Measure electrode impedance using small test current.
    ///
    /// # Algorithm
    ///
    /// 1. Disconnect all electrodes except the target
    /// 2. Apply small test current (100µA, safe level)
    /// 3. Measure resulting voltage
    /// 4. Compute impedance: Z = V / I
    /// 5. Restore previous electrode configuration
    ///
    /// # Returns
    ///
    /// Impedance measurement including quality indicator.
    /// Quality is based on signal stability and consistency.
    pub fn measure_impedance(&mut self, electrode: u8) -> Result<ImpedanceMeasurement, StimError> {
        let idx = electrode as usize;
        if idx >= MAX_ELECTRODES {
            return Err(StimError::DacError);
        }

        // Store current stimulation state to restore later
        let was_active = self.active_protocol.is_some();
        let previous_states = self.electrode_states;

        // Temporarily disable outputs
        let _ = self.enable.set_low();

        // Configure only the target electrode as anode
        // Use a common ground electrode as cathode
        let common_ground = self.find_ground_electrode(electrode);
        let mut test_config = [0u8; 8];

        // Set anode bit
        test_config[idx / 8] |= 0x80 >> (idx % 8);
        // Set cathode bit for ground electrode
        let gnd_idx = common_ground as usize;
        test_config[gnd_idx / 8] |= 0x40 >> (gnd_idx % 8);

        // Apply test configuration
        let _ = self.cs_matrix.set_low();
        let _ = self.spi.write(&test_config);
        let _ = self.cs_matrix.set_high();

        self.electrode_states[idx] = ElectrodeState::Impedance;
        self.electrode_states[gnd_idx] = ElectrodeState::Cathode;

        // Apply small test current (100µA)
        const TEST_CURRENT_UA: u16 = 100;
        self.set_current(TEST_CURRENT_UA)?;
        let _ = self.enable.set_high();

        // Wait for settling (would use delay in real implementation)
        // For now, take multiple measurements and average
        let mut voltage_sum: i32 = 0;
        let mut min_voltage: i16 = i16::MAX;
        let mut max_voltage: i16 = i16::MIN;
        const N_SAMPLES: usize = 8;

        for _ in 0..N_SAMPLES {
            let measurement = self.measure_current()?;
            let v = measurement.voltage_mv;
            voltage_sum += v as i32;
            min_voltage = min_voltage.min(v);
            max_voltage = max_voltage.max(v);
        }

        // Disable output
        let _ = self.enable.set_low();

        // Calculate average voltage
        let avg_voltage_mv = (voltage_sum / N_SAMPLES as i32) as i16;

        // Calculate impedance: Z = V / I
        // Z (kΩ) = V (mV) / I (µA) * 1000
        let impedance_kohm = if TEST_CURRENT_UA > 0 {
            ((avg_voltage_mv.abs() as u32 * 1000) / TEST_CURRENT_UA as u32) as u16
        } else {
            0
        };

        // Calculate quality based on measurement stability
        // High variance = low quality, low variance = high quality
        let variance = (max_voltage - min_voltage) as u32;
        let quality = if avg_voltage_mv.abs() > 0 {
            let relative_variance = (variance * 100) / avg_voltage_mv.abs() as u32;
            100u8.saturating_sub(relative_variance.min(100) as u8)
        } else {
            0
        };

        // Additional quality checks
        let adjusted_quality = if impedance_kohm < 1 {
            // Very low impedance suggests short circuit
            quality.saturating_sub(50)
        } else if impedance_kohm > 50 {
            // Very high impedance suggests poor contact
            quality.saturating_sub(30)
        } else {
            quality
        };

        // Restore previous configuration
        self.electrode_states = previous_states;
        self.reset_matrix()?;

        // Restore previous stimulation if it was active
        if was_active {
            if let Some(protocol) = self.active_protocol.clone() {
                self.configure_electrodes(&protocol)?;
                let _ = self.enable.set_high();
            }
        }

        Ok(ImpedanceMeasurement {
            impedance_kohm,
            electrode,
            quality: adjusted_quality,
        })
    }

    /// Find a suitable ground electrode for impedance measurement.
    ///
    /// Returns an electrode index that is far from the target electrode
    /// and in a known-good position.
    fn find_ground_electrode(&self, target: u8) -> u8 {
        // Choose electrode on opposite side of matrix
        let target_row = target / 8;
        let target_col = target % 8;

        // Pick electrode in opposite quadrant
        let ground_row = if target_row < 4 { 6 } else { 1 };
        let ground_col = if target_col < 4 { 6 } else { 1 };

        ground_row * 8 + ground_col
    }

    /// Measure impedance for all active electrodes.
    ///
    /// Returns measurements for all electrodes that are currently in use.
    pub fn measure_all_impedances(&mut self) -> Result<Vec<ImpedanceMeasurement, 16>, StimError> {
        let mut measurements = Vec::new();

        // Find all electrodes that need to be measured
        let electrodes_to_measure: Vec<u8, 16> = self.electrode_states
            .iter()
            .enumerate()
            .filter(|(_, &state)| state == ElectrodeState::Anode || state == ElectrodeState::Cathode)
            .map(|(idx, _)| idx as u8)
            .collect();

        for electrode in electrodes_to_measure {
            match self.measure_impedance(electrode) {
                Ok(m) => { let _ = measurements.push(m); }
                Err(e) => return Err(e),
            }
        }

        Ok(measurements)
    }

    /// Check if all electrode impedances are within acceptable range.
    ///
    /// # Arguments
    /// * `min_kohm` - Minimum acceptable impedance (below suggests short)
    /// * `max_kohm` - Maximum acceptable impedance (above suggests poor contact)
    /// * `min_quality` - Minimum quality score (0-100)
    pub fn validate_impedances(
        &mut self,
        min_kohm: u16,
        max_kohm: u16,
        min_quality: u8,
    ) -> Result<bool, StimError> {
        let measurements = self.measure_all_impedances()?;

        for m in measurements {
            if m.impedance_kohm < min_kohm || m.impedance_kohm > max_kohm {
                return Ok(false);
            }
            if m.quality < min_quality {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Stop stimulation gracefully
    pub fn stop_stimulation(&mut self) -> Result<(), StimError> {
        // Ramp down would happen here in real implementation
        let _ = self.enable.set_low();
        self.reset_matrix()?;
        self.active_protocol = None;
        self.safety.end_session();
        Ok(())
    }

    /// Emergency stop - immediate shutdown
    pub fn emergency_stop(&mut self) {
        let _ = self.enable.set_low();
        let _ = self.reset_matrix();
        self.active_protocol = None;
        self.safety.emergency_shutdown();
    }

    /// Get current protocol
    pub fn active_protocol(&self) -> Option<&MatrixStimProtocol> {
        self.active_protocol.as_ref()
    }

    /// Get electrode states
    pub fn electrode_states(&self) -> &[ElectrodeState; MAX_ELECTRODES] {
        &self.electrode_states
    }
}

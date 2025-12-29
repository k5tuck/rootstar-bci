//! Error types for Rootstar BCI Platform
//!
//! This module provides custom error types that work in `no_std` environments.
//! All errors are designed to be informative and include relevant context
//! for debugging without requiring heap allocation.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::types::{EegChannel, FnirsChannel, StimMode};

// ============================================================================
// ADS1299 EEG Driver Errors
// ============================================================================

/// Errors from the ADS1299 EEG ADC driver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ads1299Error<E> {
    /// SPI communication failure
    Spi(E),
    /// Invalid device ID detected (expected 0x3E)
    InvalidDeviceId {
        /// The ID value that was read
        got: u8,
        /// The expected ID value
        expected: u8,
    },
    /// Device not ready after timeout
    NotReady {
        /// Timeout duration in microseconds
        timeout_us: u32,
    },
    /// Register write verification failed
    ConfigurationFailed {
        /// Register address that failed
        register: u8,
        /// Value that was written
        expected: u8,
        /// Value that was read back
        actual: u8,
    },
    /// Data ready signal timeout
    DataReadyTimeout {
        /// Timeout duration in microseconds
        timeout_us: u32,
    },
    /// Channel is saturated (ADC clipping)
    ChannelSaturated {
        /// Which channel is saturated
        channel: u8,
    },
    /// Invalid sample rate configuration
    InvalidSampleRate {
        /// Requested sample rate in Hz
        requested_hz: u16,
    },
    /// Invalid gain configuration
    InvalidGain {
        /// Requested gain value
        requested: u8,
    },
}

impl<E: fmt::Debug> fmt::Display for Ads1299Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spi(e) => write!(f, "SPI communication error: {e:?}"),
            Self::InvalidDeviceId { got, expected } => {
                write!(f, "Invalid device ID: got 0x{got:02X}, expected 0x{expected:02X}")
            }
            Self::NotReady { timeout_us } => {
                write!(f, "Device not ready after {timeout_us}µs timeout")
            }
            Self::ConfigurationFailed { register, expected, actual } => {
                write!(
                    f,
                    "Config failed: register 0x{register:02X} wrote 0x{expected:02X}, read 0x{actual:02X}"
                )
            }
            Self::DataReadyTimeout { timeout_us } => {
                write!(f, "Data ready timeout after {timeout_us}µs")
            }
            Self::ChannelSaturated { channel } => {
                write!(f, "Channel {channel} is saturated (clipping)")
            }
            Self::InvalidSampleRate { requested_hz } => {
                write!(f, "Invalid sample rate: {requested_hz} Hz")
            }
            Self::InvalidGain { requested } => {
                write!(f, "Invalid gain: {requested}")
            }
        }
    }
}

#[cfg(feature = "defmt")]
impl<E: defmt::Format> defmt::Format for Ads1299Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::Spi(e) => defmt::write!(f, "SPI error: {}", e),
            Self::InvalidDeviceId { got, expected } => {
                defmt::write!(f, "Invalid ID: 0x{:02X} (expected 0x{:02X})", got, expected);
            }
            Self::NotReady { timeout_us } => {
                defmt::write!(f, "Not ready after {}us", timeout_us);
            }
            Self::ConfigurationFailed { register, expected, actual } => {
                defmt::write!(f, "Config failed: reg 0x{:02X}", register);
            }
            Self::DataReadyTimeout { timeout_us } => {
                defmt::write!(f, "DRDY timeout: {}us", timeout_us);
            }
            Self::ChannelSaturated { channel } => {
                defmt::write!(f, "Ch{} saturated", channel);
            }
            Self::InvalidSampleRate { requested_hz } => {
                defmt::write!(f, "Invalid rate: {}Hz", requested_hz);
            }
            Self::InvalidGain { requested } => {
                defmt::write!(f, "Invalid gain: {}", requested);
            }
        }
    }
}

// ============================================================================
// fNIRS Driver Errors
// ============================================================================

/// Errors from the fNIRS optical frontend driver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FnirsError<E> {
    /// I2C communication failure (ADS1115 ADC)
    I2c(E),
    /// PWM configuration error (LED driver)
    Pwm,
    /// Calibration failed: baseline intensity too low
    CalibrationFailed {
        /// Measured intensity
        intensity: u16,
        /// Minimum required intensity
        minimum: u16,
    },
    /// Detector saturated (too much light)
    DetectorSaturated {
        /// Which channel is saturated
        channel: FnirsChannel,
        /// Saturated intensity value
        intensity: u16,
    },
    /// Detector dark (no light reaching detector)
    DetectorDark {
        /// Which channel has no signal
        channel: FnirsChannel,
        /// Measured intensity (near zero)
        intensity: u16,
    },
    /// Beer-Lambert computation failed: determinant too small
    SingularMatrix {
        /// Computed determinant value
        determinant: i32,
    },
    /// ADC conversion timeout
    AdcTimeout {
        /// Timeout duration in microseconds
        timeout_us: u32,
    },
}

impl<E: fmt::Debug> fmt::Display for FnirsError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I2c(e) => write!(f, "I2C communication error: {e:?}"),
            Self::Pwm => write!(f, "PWM configuration error"),
            Self::CalibrationFailed { intensity, minimum } => {
                write!(f, "Calibration failed: intensity {intensity} < minimum {minimum}")
            }
            Self::DetectorSaturated { channel, intensity } => {
                write!(
                    f,
                    "Detector saturated on S{}D{}: intensity {intensity}",
                    channel.source, channel.detector
                )
            }
            Self::DetectorDark { channel, intensity } => {
                write!(
                    f,
                    "Detector dark on S{}D{}: intensity {intensity}",
                    channel.source, channel.detector
                )
            }
            Self::SingularMatrix { determinant } => {
                write!(f, "Beer-Lambert failed: singular matrix (det={determinant})")
            }
            Self::AdcTimeout { timeout_us } => {
                write!(f, "ADC conversion timeout after {timeout_us}µs")
            }
        }
    }
}

#[cfg(feature = "defmt")]
impl<E: defmt::Format> defmt::Format for FnirsError<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::I2c(e) => defmt::write!(f, "I2C error: {}", e),
            Self::Pwm => defmt::write!(f, "PWM error"),
            Self::CalibrationFailed { intensity, minimum } => {
                defmt::write!(f, "Cal failed: {} < {}", intensity, minimum);
            }
            Self::DetectorSaturated { channel, intensity } => {
                defmt::write!(f, "Saturated: {}", channel);
            }
            Self::DetectorDark { channel, intensity } => {
                defmt::write!(f, "Dark: {}", channel);
            }
            Self::SingularMatrix { determinant } => {
                defmt::write!(f, "Singular matrix");
            }
            Self::AdcTimeout { timeout_us } => {
                defmt::write!(f, "ADC timeout: {}us", timeout_us);
            }
        }
    }
}

// ============================================================================
// Stimulation Errors
// ============================================================================

/// Errors related to stimulation safety and configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimError {
    /// Current exceeds safety limit
    CurrentExceedsLimit {
        /// Requested current in µA
        requested_ua: u16,
        /// Maximum allowed current in µA
        maximum_ua: u16,
    },
    /// Duration exceeds safety limit
    DurationExceedsLimit {
        /// Requested duration in ms
        requested_ms: u32,
        /// Maximum allowed duration in ms
        maximum_ms: u32,
    },
    /// Ramp time too short (sudden onset)
    RampTooShort {
        /// Requested ramp time in ms
        requested_ms: u16,
        /// Minimum required ramp time in ms
        minimum_ms: u16,
    },
    /// tACS frequency exceeds limit
    FrequencyExceedsLimit {
        /// Requested frequency in Hz
        requested_hz: u16,
        /// Maximum allowed frequency in Hz
        maximum_hz: u16,
    },
    /// Invalid mode for requested operation
    InvalidMode {
        /// The mode that was set
        mode: StimMode,
        /// Description of the issue
        reason: &'static str,
    },
    /// Hardware safety circuit triggered
    HardwareSafetyTriggered,
    /// Electrode impedance too high
    HighImpedance {
        /// Measured impedance in ohms
        impedance_ohms: u32,
        /// Maximum acceptable impedance
        maximum_ohms: u32,
    },
    /// DAC communication error
    DacError,
}

impl fmt::Display for StimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentExceedsLimit { requested_ua, maximum_ua } => {
                write!(f, "Current {requested_ua}µA exceeds limit {maximum_ua}µA")
            }
            Self::DurationExceedsLimit { requested_ms, maximum_ms } => {
                write!(f, "Duration {requested_ms}ms exceeds limit {maximum_ms}ms")
            }
            Self::RampTooShort { requested_ms, minimum_ms } => {
                write!(f, "Ramp {requested_ms}ms too short (minimum {minimum_ms}ms)")
            }
            Self::FrequencyExceedsLimit { requested_hz, maximum_hz } => {
                write!(f, "Frequency {requested_hz}Hz exceeds limit {maximum_hz}Hz")
            }
            Self::InvalidMode { mode, reason } => {
                write!(f, "Invalid mode {mode:?}: {reason}")
            }
            Self::HardwareSafetyTriggered => {
                write!(f, "Hardware safety circuit triggered - stimulation disabled")
            }
            Self::HighImpedance { impedance_ohms, maximum_ohms } => {
                write!(f, "Impedance {impedance_ohms}Ω exceeds limit {maximum_ohms}Ω")
            }
            Self::DacError => write!(f, "DAC communication error"),
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for StimError {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::CurrentExceedsLimit { requested_ua, maximum_ua } => {
                defmt::write!(f, "Current {}uA > {}uA", requested_ua, maximum_ua);
            }
            Self::DurationExceedsLimit { requested_ms, maximum_ms } => {
                defmt::write!(f, "Duration {}ms > {}ms", requested_ms, maximum_ms);
            }
            Self::RampTooShort { requested_ms, minimum_ms } => {
                defmt::write!(f, "Ramp {}ms < {}ms", requested_ms, minimum_ms);
            }
            Self::FrequencyExceedsLimit { requested_hz, maximum_hz } => {
                defmt::write!(f, "Freq {}Hz > {}Hz", requested_hz, maximum_hz);
            }
            Self::InvalidMode { mode, reason } => {
                defmt::write!(f, "Invalid mode: {}", reason);
            }
            Self::HardwareSafetyTriggered => {
                defmt::write!(f, "HW SAFETY TRIGGERED");
            }
            Self::HighImpedance { impedance_ohms, maximum_ohms } => {
                defmt::write!(f, "Impedance {}R > {}R", impedance_ohms, maximum_ohms);
            }
            Self::DacError => defmt::write!(f, "DAC error"),
        }
    }
}

// ============================================================================
// Protocol Errors
// ============================================================================

/// Errors in the communication protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolError {
    /// Invalid sync bytes in packet header
    InvalidSync {
        /// First sync byte received
        got_0: u8,
        /// Second sync byte received
        got_1: u8,
    },
    /// Invalid packet type
    InvalidPacketType {
        /// Unknown packet type value
        packet_type: u8,
    },
    /// Checksum mismatch
    ChecksumMismatch {
        /// Expected checksum
        expected: u8,
        /// Computed checksum
        computed: u8,
    },
    /// Payload length exceeds maximum
    PayloadTooLarge {
        /// Received payload length
        length: u16,
        /// Maximum allowed length
        maximum: u16,
    },
    /// Sequence number indicates packet loss
    SequenceGap {
        /// Expected sequence number
        expected: u32,
        /// Received sequence number
        received: u32,
    },
    /// Buffer overflow (not enough space for data)
    BufferOverflow {
        /// Required size
        required: usize,
        /// Available size
        available: usize,
    },
    /// Incomplete packet (not enough bytes)
    IncompletePacket {
        /// Bytes received
        received: usize,
        /// Bytes expected
        expected: usize,
    },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSync { got_0, got_1 } => {
                write!(f, "Invalid sync: 0x{got_0:02X} 0x{got_1:02X}")
            }
            Self::InvalidPacketType { packet_type } => {
                write!(f, "Invalid packet type: 0x{packet_type:02X}")
            }
            Self::ChecksumMismatch { expected, computed } => {
                write!(f, "Checksum mismatch: expected 0x{expected:02X}, got 0x{computed:02X}")
            }
            Self::PayloadTooLarge { length, maximum } => {
                write!(f, "Payload too large: {length} bytes (max {maximum})")
            }
            Self::SequenceGap { expected, received } => {
                write!(f, "Sequence gap: expected {expected}, got {received}")
            }
            Self::BufferOverflow { required, available } => {
                write!(f, "Buffer overflow: need {required} bytes, have {available}")
            }
            Self::IncompletePacket { received, expected } => {
                write!(f, "Incomplete packet: got {received}/{expected} bytes")
            }
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for ProtocolError {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::InvalidSync { got_0, got_1 } => {
                defmt::write!(f, "Bad sync: {:02X} {:02X}", got_0, got_1);
            }
            Self::InvalidPacketType { packet_type } => {
                defmt::write!(f, "Bad type: {:02X}", packet_type);
            }
            Self::ChecksumMismatch { expected, computed } => {
                defmt::write!(f, "Checksum: {:02X} != {:02X}", expected, computed);
            }
            Self::PayloadTooLarge { length, maximum } => {
                defmt::write!(f, "Payload: {} > {}", length, maximum);
            }
            Self::SequenceGap { expected, received } => {
                defmt::write!(f, "Seq gap: {} -> {}", expected, received);
            }
            Self::BufferOverflow { required, available } => {
                defmt::write!(f, "Overflow: {} > {}", required, available);
            }
            Self::IncompletePacket { received, expected } => {
                defmt::write!(f, "Incomplete: {}/{}", received, expected);
            }
        }
    }
}

// ============================================================================
// Processing Errors (for native tier)
// ============================================================================

/// Errors during signal processing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingError {
    /// Signal saturated on a channel
    Saturation {
        /// Which EEG channel is saturated
        channel: EegChannel,
        /// The saturated value
        value_raw: i32,
        /// Maximum valid value
        threshold: i32,
    },
    /// fNIRS calibration failed
    FnirsCalibrationFailed {
        /// Baseline intensity that was too low
        intensity: u16,
        /// Minimum required intensity
        minimum: u16,
    },
    /// Beer-Lambert computation failed
    BeerLambertFailed {
        /// Determinant that was too small
        determinant_q8: i32,
    },
    /// Temporal alignment failed
    AlignmentFailed {
        /// Alignment window in ms
        window_ms: u32,
    },
    /// Insufficient data for processing
    InsufficientData {
        /// Number of samples available
        available: usize,
        /// Number of samples required
        required: usize,
    },
    /// Filter configuration invalid
    InvalidFilterConfig {
        /// Description of the issue
        reason: &'static str,
    },
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Saturation { channel, value_raw, threshold } => {
                write!(
                    f,
                    "Channel {} saturated: {} exceeds threshold {}",
                    channel.name(), value_raw, threshold
                )
            }
            Self::FnirsCalibrationFailed { intensity, minimum } => {
                write!(f, "fNIRS calibration failed: intensity {intensity} < {minimum}")
            }
            Self::BeerLambertFailed { determinant_q8 } => {
                write!(f, "Beer-Lambert failed: determinant {determinant_q8} too small")
            }
            Self::AlignmentFailed { window_ms } => {
                write!(f, "Temporal alignment failed: no samples within {window_ms}ms")
            }
            Self::InsufficientData { available, required } => {
                write!(f, "Insufficient data: {available}/{required} samples")
            }
            Self::InvalidFilterConfig { reason } => {
                write!(f, "Invalid filter config: {reason}")
            }
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for ProcessingError {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::Saturation { channel, value_raw, threshold } => {
                defmt::write!(f, "{} saturated", channel);
            }
            Self::FnirsCalibrationFailed { intensity, minimum } => {
                defmt::write!(f, "fNIRS cal: {} < {}", intensity, minimum);
            }
            Self::BeerLambertFailed { determinant_q8 } => {
                defmt::write!(f, "Beer-Lambert failed");
            }
            Self::AlignmentFailed { window_ms } => {
                defmt::write!(f, "Align failed: {}ms", window_ms);
            }
            Self::InsufficientData { available, required } => {
                defmt::write!(f, "Data: {}/{}", available, required);
            }
            Self::InvalidFilterConfig { reason } => {
                defmt::write!(f, "Filter: {}", reason);
            }
        }
    }
}

// ============================================================================
// Stimulation Parameter Validation
// ============================================================================

use crate::types::StimParams;

impl StimParams {
    /// Validate stimulation parameters against safety limits.
    ///
    /// # Errors
    ///
    /// Returns a `StimError` if any parameter violates safety constraints.
    pub fn validate(&self) -> Result<(), StimError> {
        // Check current limit
        if self.amplitude_ua > Self::MAX_CURRENT_UA {
            return Err(StimError::CurrentExceedsLimit {
                requested_ua: self.amplitude_ua,
                maximum_ua: Self::MAX_CURRENT_UA,
            });
        }

        // Check duration limit
        if self.duration_ms > Self::MAX_DURATION_MS {
            return Err(StimError::DurationExceedsLimit {
                requested_ms: self.duration_ms,
                maximum_ms: Self::MAX_DURATION_MS,
            });
        }

        // Check ramp time for electrical stimulation
        if self.mode.is_electrical() && self.ramp_ms < Self::MIN_RAMP_MS {
            return Err(StimError::RampTooShort {
                requested_ms: self.ramp_ms,
                minimum_ms: Self::MIN_RAMP_MS,
            });
        }

        // Check tACS frequency
        if self.mode == StimMode::Tacs && self.frequency_hz > Self::MAX_TACS_FREQUENCY_HZ {
            return Err(StimError::FrequencyExceedsLimit {
                requested_hz: self.frequency_hz,
                maximum_hz: Self::MAX_TACS_FREQUENCY_HZ,
            });
        }

        // Check that tACS has non-zero frequency
        if self.mode == StimMode::Tacs && self.frequency_hz == 0 {
            return Err(StimError::InvalidMode {
                mode: self.mode,
                reason: "tACS requires non-zero frequency",
            });
        }

        // Check that DC modes have zero frequency
        if matches!(self.mode, StimMode::TdcsAnodal | StimMode::TdcsCathodal)
            && self.frequency_hz != 0
        {
            return Err(StimError::InvalidMode {
                mode: self.mode,
                reason: "tDCS must have zero frequency",
            });
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stim_params_validation_current() {
        let params = StimParams {
            mode: StimMode::TdcsAnodal,
            amplitude_ua: 2500, // Over limit
            frequency_hz: 0,
            duration_ms: 1000,
            ramp_ms: 100,
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::CurrentExceedsLimit { .. })));
    }

    #[test]
    fn test_stim_params_validation_duration() {
        let params = StimParams {
            mode: StimMode::TdcsAnodal,
            amplitude_ua: 1000,
            frequency_hz: 0,
            duration_ms: 60 * 60 * 1000, // 1 hour, over limit
            ramp_ms: 100,
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::DurationExceedsLimit { .. })));
    }

    #[test]
    fn test_stim_params_validation_ramp() {
        let params = StimParams {
            mode: StimMode::TdcsAnodal,
            amplitude_ua: 1000,
            frequency_hz: 0,
            duration_ms: 1000,
            ramp_ms: 5, // Too short
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::RampTooShort { .. })));
    }

    #[test]
    fn test_stim_params_validation_tacs_frequency() {
        let params = StimParams {
            mode: StimMode::Tacs,
            amplitude_ua: 1000,
            frequency_hz: 150, // Over limit
            duration_ms: 1000,
            ramp_ms: 100,
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::FrequencyExceedsLimit { .. })));
    }

    #[test]
    fn test_stim_params_validation_tacs_zero_frequency() {
        let params = StimParams {
            mode: StimMode::Tacs,
            amplitude_ua: 1000,
            frequency_hz: 0, // Invalid for tACS
            duration_ms: 1000,
            ramp_ms: 100,
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::InvalidMode { .. })));
    }

    #[test]
    fn test_stim_params_validation_tdcs_nonzero_frequency() {
        let params = StimParams {
            mode: StimMode::TdcsAnodal,
            amplitude_ua: 1000,
            frequency_hz: 10, // Invalid for tDCS
            duration_ms: 1000,
            ramp_ms: 100,
        };

        let result = params.validate();
        assert!(matches!(result, Err(StimError::InvalidMode { .. })));
    }

    #[test]
    fn test_stim_params_validation_valid() {
        let params = StimParams {
            mode: StimMode::TdcsAnodal,
            amplitude_ua: 1000,
            frequency_hz: 0,
            duration_ms: 20 * 60 * 1000, // 20 minutes
            ramp_ms: 100,
        };

        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_stim_params_validation_off() {
        let params = StimParams::off();
        assert!(params.validate().is_ok());
    }
}

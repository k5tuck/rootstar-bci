//! SNS Error Types
//!
//! Error types for the Sensory Neural Simulation module, compatible with `no_std`.

use core::fmt;


use super::types::SensoryModality;

/// SNS error types
#[derive(Debug, Clone, PartialEq)]
pub enum SnsError<E> {
    /// Receptor model parameter out of valid range
    InvalidReceptorParameter {
        /// Parameter name
        parameter: &'static str,
        /// Value in Q24.8 raw format
        value_raw: i32,
        /// Minimum allowed value
        min: i32,
        /// Maximum allowed value
        max: i32,
    },

    /// Stimulus intensity exceeds physiological range
    StimulusOverload {
        /// Sensory modality
        modality: SensoryModality,
        /// Intensity in Q24.8 raw format
        intensity: i32,
        /// Maximum safe intensity
        max_safe: i32,
    },

    /// Spiking network configuration invalid
    NetworkTopologyError {
        /// Number of neurons
        neurons: usize,
        /// Number of synapses
        synapses: usize,
        /// Error reason
        reason: &'static str,
    },

    /// Timestamp synchronization failure
    TemporalAlignmentError {
        /// EEG timestamp in microseconds
        eeg_time_us: u64,
        /// SNS timestamp in microseconds
        sns_time_us: u64,
        /// Maximum allowed drift
        max_drift_us: u64,
    },

    /// Population capacity exceeded
    PopulationCapacityExceeded {
        /// Current population size
        current: usize,
        /// Maximum capacity
        capacity: usize,
    },

    /// Receptor not found
    ReceptorNotFound {
        /// Receptor index
        index: usize,
    },

    /// Invalid frequency for auditory processing
    InvalidFrequency {
        /// Frequency in Hz
        frequency_hz: f32,
        /// Minimum frequency
        min_hz: f32,
        /// Maximum frequency
        max_hz: f32,
    },

    /// Invalid concentration for gustatory processing
    InvalidConcentration {
        /// Concentration value
        concentration: f32,
    },

    /// Spike buffer overflow
    SpikeBufferOverflow {
        /// Number of spikes
        spikes: usize,
        /// Buffer capacity
        capacity: usize,
    },

    /// Hardware communication error (wraps driver error)
    HardwareError(E),

    /// Computation error (e.g., division by zero)
    ComputationError {
        /// Operation that failed
        operation: &'static str,
        /// Error description
        reason: &'static str,
    },
}

impl<E: fmt::Debug> fmt::Display for SnsError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReceptorParameter { parameter, value_raw, min, max } => {
                write!(
                    f,
                    "Invalid receptor parameter '{}': value {} not in range [{}, {}]",
                    parameter, value_raw, min, max
                )
            }
            Self::StimulusOverload { modality, intensity, max_safe } => {
                write!(
                    f,
                    "Stimulus overload in {:?}: intensity {} exceeds safe maximum {}",
                    modality, intensity, max_safe
                )
            }
            Self::NetworkTopologyError { neurons, synapses, reason } => {
                write!(
                    f,
                    "Network topology error ({} neurons, {} synapses): {}",
                    neurons, synapses, reason
                )
            }
            Self::TemporalAlignmentError { eeg_time_us, sns_time_us, max_drift_us } => {
                write!(
                    f,
                    "Temporal alignment error: EEG @ {}µs, SNS @ {}µs, max drift {}µs",
                    eeg_time_us, sns_time_us, max_drift_us
                )
            }
            Self::PopulationCapacityExceeded { current, capacity } => {
                write!(
                    f,
                    "Population capacity exceeded: {} receptors, max {}",
                    current, capacity
                )
            }
            Self::ReceptorNotFound { index } => {
                write!(f, "Receptor not found at index {}", index)
            }
            Self::InvalidFrequency { frequency_hz, min_hz, max_hz } => {
                write!(
                    f,
                    "Invalid frequency {}Hz: must be in range [{}, {}]Hz",
                    frequency_hz, min_hz, max_hz
                )
            }
            Self::InvalidConcentration { concentration } => {
                write!(f, "Invalid concentration {}: must be in range [0, 1]", concentration)
            }
            Self::SpikeBufferOverflow { spikes, capacity } => {
                write!(f, "Spike buffer overflow: {} spikes, capacity {}", spikes, capacity)
            }
            Self::HardwareError(e) => {
                write!(f, "Hardware error: {:?}", e)
            }
            Self::ComputationError { operation, reason } => {
                write!(f, "Computation error in '{}': {}", operation, reason)
            }
        }
    }
}

impl<E> SnsError<E> {
    /// Map the hardware error type
    pub fn map_hardware<F, E2>(self, f: F) -> SnsError<E2>
    where
        F: FnOnce(E) -> E2,
    {
        match self {
            Self::InvalidReceptorParameter { parameter, value_raw, min, max } => {
                SnsError::InvalidReceptorParameter { parameter, value_raw, min, max }
            }
            Self::StimulusOverload { modality, intensity, max_safe } => {
                SnsError::StimulusOverload { modality, intensity, max_safe }
            }
            Self::NetworkTopologyError { neurons, synapses, reason } => {
                SnsError::NetworkTopologyError { neurons, synapses, reason }
            }
            Self::TemporalAlignmentError { eeg_time_us, sns_time_us, max_drift_us } => {
                SnsError::TemporalAlignmentError { eeg_time_us, sns_time_us, max_drift_us }
            }
            Self::PopulationCapacityExceeded { current, capacity } => {
                SnsError::PopulationCapacityExceeded { current, capacity }
            }
            Self::ReceptorNotFound { index } => SnsError::ReceptorNotFound { index },
            Self::InvalidFrequency { frequency_hz, min_hz, max_hz } => {
                SnsError::InvalidFrequency { frequency_hz, min_hz, max_hz }
            }
            Self::InvalidConcentration { concentration } => {
                SnsError::InvalidConcentration { concentration }
            }
            Self::SpikeBufferOverflow { spikes, capacity } => {
                SnsError::SpikeBufferOverflow { spikes, capacity }
            }
            Self::HardwareError(e) => SnsError::HardwareError(f(e)),
            Self::ComputationError { operation, reason } => {
                SnsError::ComputationError { operation, reason }
            }
        }
    }
}

/// Validation result type
pub type SnsResult<T, E = ()> = Result<T, SnsError<E>>;

/// Decoder error types (for std environments)
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderErrorKind {
    /// Insufficient EEG data for feature extraction
    InsufficientData {
        /// Number of samples received
        got: usize,
        /// Number of samples needed
        need: usize,
    },

    /// SEP/AEP/GEP component not detected
    ComponentNotFound {
        /// Component name (e.g., "N20", "P1")
        component: &'static str,
        /// Search window start in milliseconds
        start_ms: f64,
        /// Search window end in milliseconds
        end_ms: f64,
    },

    /// fNIRS channel saturated
    FnirsSaturation {
        /// Channel source index
        source: u8,
        /// Channel detector index
        detector: u8,
        /// Measured intensity
        intensity: u16,
        /// Maximum valid intensity
        max: u16,
    },

    /// ML inference failed
    InferenceError {
        /// Error reason
        reason: heapless::String<64>,
    },

    /// Cortical channel mapping undefined
    UndefinedChannelMapping {
        /// Sensory modality
        modality: SensoryModality,
    },

    /// Feature extraction failed
    FeatureExtractionFailed {
        /// Feature name
        feature: &'static str,
        /// Reason
        reason: &'static str,
    },
}

/// Encoder error types (for std environments)
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderErrorKind {
    /// Receptor population empty
    EmptyPopulation {
        /// Sensory modality
        modality: SensoryModality,
    },

    /// HRF convolution failed due to insufficient history
    InsufficientHistory {
        /// Available history in milliseconds
        got_ms: f64,
        /// Required history in milliseconds
        need_ms: f64,
    },

    /// Forward model singular matrix
    SingularForwardModel {
        /// Electrode name
        electrode: &'static str,
    },

    /// Calibration diverged
    CalibrationDiverged {
        /// Current loss
        loss: f32,
        /// Threshold
        threshold: f32,
    },

    /// Invalid encoder state
    InvalidState {
        /// State description
        state: &'static str,
    },
}

/// Render error types (for visualization)
#[derive(Debug, Clone, PartialEq)]
pub enum RenderErrorKind {
    /// Device creation failed
    DeviceCreationFailed {
        /// Reason
        reason: heapless::String<64>,
    },

    /// Mesh not found
    MeshNotFound {
        /// Mesh identifier
        mesh_id: heapless::String<32>,
    },

    /// Unsupported texture format
    UnsupportedTextureFormat {
        /// Format name
        format: &'static str,
    },

    /// Shader compilation failed
    ShaderCompilationFailed {
        /// Shader name
        shader_name: &'static str,
    },

    /// Buffer allocation failed
    BufferAllocationFailed {
        /// Size in bytes
        size: usize,
    },

    /// Pipeline creation failed
    PipelineCreationFailed {
        /// Pipeline name
        pipeline: &'static str,
        /// Reason
        reason: &'static str,
    },
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err: SnsError<()> = SnsError::InvalidReceptorParameter {
            parameter: "tau_adapt",
            value_raw: -100,
            min: 0,
            max: 1000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("tau_adapt"));
        assert!(msg.contains("-100"));
    }

    #[test]
    fn test_error_map() {
        let err: SnsError<i32> = SnsError::HardwareError(42);
        let mapped: SnsError<String> = err.map_hardware(|e| format!("Error code: {}", e));

        match mapped {
            SnsError::HardwareError(s) => assert_eq!(s, "Error code: 42"),
            _ => panic!("Expected HardwareError"),
        }
    }
}

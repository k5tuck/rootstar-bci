//! SNS Error Types for Native Processing
//!
//! Error types for decoders, encoders, and calibration using `thiserror`.

use rootstar_bci_core::sns::SensoryModality;
use rootstar_bci_core::types::{EegChannel, FnirsChannel};
use thiserror::Error;

/// Decoder error types
#[derive(Error, Debug)]
pub enum DecoderError {
    /// Insufficient EEG data for feature extraction
    #[error("Insufficient EEG data: got {got} samples, need {need}")]
    InsufficientData {
        /// Number of samples received
        got: usize,
        /// Number of samples needed
        need: usize,
    },

    /// SEP/AEP/GEP component not detected
    #[error("Component {component} not detected in window [{start_ms}ms, {end_ms}ms]")]
    ComponentNotFound {
        /// Component name (e.g., "N20", "P1")
        component: &'static str,
        /// Search window start in milliseconds
        start_ms: f64,
        /// Search window end in milliseconds
        end_ms: f64,
    },

    /// fNIRS channel saturated
    #[error("fNIRS channel {channel:?} saturated: intensity {intensity} exceeds {max}")]
    FnirsSaturation {
        /// Affected channel
        channel: FnirsChannel,
        /// Measured intensity
        intensity: u16,
        /// Maximum valid intensity
        max: u16,
    },

    /// ML inference failed
    #[error("ML inference failed: {reason}")]
    InferenceError {
        /// Error reason
        reason: String,
    },

    /// Cortical channel mapping undefined
    #[error("Cortical channel mapping undefined for modality {modality:?}")]
    UndefinedChannelMapping {
        /// Sensory modality
        modality: SensoryModality,
    },

    /// Feature extraction failed
    #[error("Feature extraction failed for {feature}: {reason}")]
    FeatureExtractionFailed {
        /// Feature name
        feature: &'static str,
        /// Reason for failure
        reason: String,
    },

    /// Invalid time window
    #[error("Invalid time window: {start_ms}ms to {end_ms}ms")]
    InvalidTimeWindow {
        /// Start time
        start_ms: f64,
        /// End time
        end_ms: f64,
    },

    /// Model not loaded
    #[error("Decoder model not loaded for modality {modality:?}")]
    ModelNotLoaded {
        /// Modality requiring the model
        modality: SensoryModality,
    },
}

/// Encoder error types
#[derive(Error, Debug)]
pub enum EncoderError {
    /// Receptor population empty
    #[error("Receptor population empty for modality {modality:?}")]
    EmptyPopulation {
        /// Sensory modality
        modality: SensoryModality,
    },

    /// HRF convolution failed due to insufficient history
    #[error("HRF convolution failed: insufficient history ({got_ms}ms, need {need_ms}ms)")]
    InsufficientHistory {
        /// Available history in milliseconds
        got_ms: f64,
        /// Required history in milliseconds
        need_ms: f64,
    },

    /// Forward model singular matrix
    #[error("Forward model singular matrix at electrode {electrode:?}")]
    SingularForwardModel {
        /// Affected electrode
        electrode: EegChannel,
    },

    /// Calibration diverged
    #[error("Calibration diverged: loss {loss} exceeds threshold {threshold}")]
    CalibrationDiverged {
        /// Current loss
        loss: f64,
        /// Threshold
        threshold: f64,
    },

    /// Invalid encoder state
    #[error("Invalid encoder state: {state}")]
    InvalidState {
        /// State description
        state: String,
    },

    /// Thalamocortical relay error
    #[error("Thalamocortical relay error: {reason}")]
    ThalamocorticalError {
        /// Error reason
        reason: String,
    },

    /// Computation error
    #[error("Computation error in {operation}: {reason}")]
    ComputationError {
        /// Operation that failed
        operation: &'static str,
        /// Reason
        reason: String,
    },
}

/// Calibration error types
#[derive(Error, Debug)]
pub enum CalibrationError {
    /// Insufficient calibration samples
    #[error("Insufficient calibration samples: got {got}, need {need}")]
    InsufficientSamples {
        /// Number of samples collected
        got: usize,
        /// Number of samples needed
        need: usize,
    },

    /// Calibration failed to converge
    #[error("Calibration failed to converge after {iterations} iterations (loss: {loss})")]
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Final loss value
        loss: f64,
    },

    /// Invalid calibration parameters
    #[error("Invalid calibration parameter {parameter}: {reason}")]
    InvalidParameter {
        /// Parameter name
        parameter: &'static str,
        /// Reason
        reason: String,
    },

    /// Decoder error during calibration
    #[error("Decoder error: {0}")]
    DecoderError(#[from] DecoderError),

    /// Encoder error during calibration
    #[error("Encoder error: {0}")]
    EncoderError(#[from] EncoderError),

    /// Invalid state for calibration operation
    #[error("Invalid calibration state: expected {expected}, got {actual}")]
    InvalidState {
        /// Expected state
        expected: &'static str,
        /// Actual state
        actual: String,
    },
}

/// Result type for decoder operations
pub type DecoderResult<T> = Result<T, DecoderError>;

/// Result type for encoder operations
pub type EncoderResult<T> = Result<T, EncoderError>;

/// Result type for calibration operations
pub type CalibrationResult<T> = Result<T, CalibrationError>;

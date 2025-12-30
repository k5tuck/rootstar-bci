//! Neural Fingerprint module for sensory experience capture and playback.
//!
//! This module provides core types for the Neural Fingerprint Detection &
//! Sensory Simulation System. It enables capturing regional neural activity
//! signatures ("fingerprints") that correspond to specific sensory experiences.
//!
//! # Architecture
//!
//! - [`types`]: Core data structures (NeuralFingerprint, SensoryModality)
//! - [`config`]: Scalable configuration (8 → 128 → 256 channels)
//! - [`safety`]: Stimulation safety limits and monitoring
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::fingerprint::{
//!     NeuralFingerprint, SensoryModality, SystemConfig, ChannelDensity,
//! };
//!
//! // Create a config for 8-channel proof-of-concept
//! let config = SystemConfig::new(ChannelDensity::Basic8);
//!
//! // Or for high-density 128-channel research
//! let config = SystemConfig::new(ChannelDensity::HighDensity128);
//! ```

pub mod config;
pub mod safety;
pub mod types;

// Re-export commonly used types
pub use config::{ChannelDensity, FnirsConfig, SystemConfig};
pub use safety::{SafetyLimits, SafetyMonitor, SafetyViolation};
pub use types::{
    ElectrodePosition, FingerprintId, FingerprintMetadata, FrequencyBand, NeuralFingerprint,
    QualityMetrics, SensoryModality,
};

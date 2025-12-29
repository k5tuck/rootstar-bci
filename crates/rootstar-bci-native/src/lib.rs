//! Rootstar BCI Native - Host signal processing and ML inference
//!
//! This crate provides host-side processing for BCI data:
//! - Signal processing (filtering, FFT, band power)
//! - fNIRS processing (Beer-Lambert Law)
//! - EEG + fNIRS data fusion
//! - ML inference for intent decoding
//! - Bridge to LSL/Brainflow ecosystems
//!
//! # Modules
//!
//! - [`bridge`]: Communication bridges (USB, LSL, Brainflow)
//! - [`processing`]: Signal processing pipelines
//! - [`ml`]: Machine learning inference

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod bridge;
pub mod ml;
pub mod processing;
pub mod sns;

/// Native visualization module (requires `viz` feature)
#[cfg(feature = "viz")]
pub mod viz;

// Re-export key types
pub use processing::fnirs::FnirsProcessor;
pub use processing::fusion::{AlignedSample, TemporalAligner};

// Re-export SNS types
pub use sns::{
    BidirectionalCalibrator, CalibrationMetrics, CorticalDecoder, CorticalEncoder,
    SnsSimulation, SimulationConfig,
};

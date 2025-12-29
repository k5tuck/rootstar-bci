//! Sensory Neural Simulation Processing
//!
//! This module provides host-side SNS processing:
//! - Cortical decoders (EEG/fNIRS → receptor activation)
//! - Cortical encoders (receptor activation → predicted EEG)
//! - Bidirectional calibration loop
//! - Cortical mapping utilities
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SNS Processing Pipeline                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  DECODE PATH                      ENCODE PATH                   │
//! │  ═══════════                      ═══════════                   │
//! │                                                                 │
//! │  EEG/fNIRS ──► Feature      Receptor ──► Thalamocortical       │
//! │               Extraction    Activation     Model                │
//! │                  │              │             │                  │
//! │                  ▼              ▼             ▼                  │
//! │              ML Model      Population    Predicted              │
//! │              Inference     Simulation    EEG/fNIRS              │
//! │                  │              │             │                  │
//! │                  ▼              ▼             ▼                  │
//! │              Receptor     ◄─────────────────►                   │
//! │              Prediction    Calibration Loop                     │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod calibration;
pub mod cortical_map;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod features;
pub mod hrf;
pub mod simulation;

// Re-export key types
pub use calibration::{BidirectionalCalibrator, CalibrationMetrics, CalibrationSample};
pub use cortical_map::{
    AuditoryCortexChannels, CorticalChannelMap, GustatoryCortexChannels, TactileCortexChannels,
};
pub use decoder::{
    AuditoryPrediction, CorticalDecoder, DecoderConfig, GustatoryPrediction, TactilePrediction,
};
pub use encoder::{CorticalEncoder, EncoderConfig, PredictedEeg, PredictedFnirs};
pub use error::{DecoderError, EncoderError};
pub use hrf::HemodynamicResponseFunction;
pub use simulation::{SnsSimulation, SimulationConfig};

//! Neural Fingerprint processing for native platform.
//!
//! This module provides host-side processing for neural fingerprint
//! extraction, fusion, and stimulation control.
//!
//! # Modules
//!
//! - [`extractor`]: Feature extraction from EEG and fNIRS signals
//! - [`fusion`]: Multimodal fusion for fingerprint generation
//! - [`stimulation`]: Stimulation protocol controller
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::fingerprint::{
//!     EegFeatureExtractor, FnirsFeatureExtractor, FingerprintFusion,
//! };
//!
//! // Create extractors
//! let eeg_extractor = EegFeatureExtractor::new(8, 500.0);
//! let fnirs_extractor = FnirsFeatureExtractor::new(4, 25.0);
//!
//! // Extract features
//! let eeg_features = eeg_extractor.extract(&eeg_data);
//! let fnirs_features = fnirs_extractor.extract(&fnirs_data);
//!
//! // Fuse into fingerprint
//! let mut fusion = FingerprintFusion::new();
//! let fingerprint = fusion.generate(eeg_features, fnirs_features, metadata);
//! ```

pub mod extractor;
pub mod fusion;
pub mod stimulation;

// Re-export key types
pub use extractor::{
    EegFeatureExtractor, EegFeatures, FnirsFeatureExtractor, FnirsFeatures,
};
pub use fusion::{FingerprintFusion, FusionConfig};
pub use stimulation::{StimulationController, StimulationSession};

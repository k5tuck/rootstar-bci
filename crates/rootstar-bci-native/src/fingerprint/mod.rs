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
//! - [`database`]: Shared fingerprint storage (requires `database` feature)
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
//!
//! # Shared Database Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::fingerprint::database::{FingerprintDatabase, StoredFingerprint};
//!
//! // Open shared database
//! let db = FingerprintDatabase::open("fingerprints.db")?;
//!
//! // Store a fingerprint (accessible from any device)
//! let stored = StoredFingerprint::new(fingerprint, "Apple Taste".to_string())
//!     .with_device("device-001".to_string());
//! db.store(&stored)?;
//!
//! // Find similar patterns
//! let matches = db.find_similar(&current_fingerprint, 0.8, 10)?;
//! ```

pub mod extractor;
pub mod fusion;
pub mod stimulation;

/// Shared fingerprint database (requires `database` feature)
#[cfg(feature = "database")]
pub mod database;

// Re-export key types
pub use extractor::{
    EegFeatureExtractor, EegFeatures, FnirsFeatureExtractor, FnirsFeatures,
};
pub use fusion::{FingerprintFusion, FusionConfig};
pub use stimulation::{StimulationController, StimulationSession};

// Re-export database types when feature enabled
#[cfg(feature = "database")]
pub use database::{
    DatabaseError, DatabaseStats, DbResult, FingerprintDatabase, FingerprintQuery,
    SimilarityMatch, StoredFingerprint,
};

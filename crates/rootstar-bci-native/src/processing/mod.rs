//! Signal processing pipelines
//!
//! This module provides signal processing for BCI data:
//! - [`filters`]: Digital filtering (IIR/FIR)
//! - [`fft`]: Spectral analysis and band power
//! - [`fnirs`]: fNIRS Beer-Lambert processing
//! - [`fusion`]: EEG + fNIRS data fusion
//! - [`hyperscanning`]: Cross-device sync and inter-brain coherence

pub mod fft;
pub mod filters;
pub mod fnirs;
pub mod fusion;
pub mod hyperscanning;

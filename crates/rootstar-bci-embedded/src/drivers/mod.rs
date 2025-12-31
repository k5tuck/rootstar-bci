//! Hardware drivers for BCI acquisition
//!
//! This module contains drivers for the various hardware components:
//! - [`ads1299`]: TI ADS1299 24-bit EEG ADC (single chip)
//! - [`ads1299_array`]: High-density EEG with daisy-chained ADS1299s (8-256 channels)
//! - [`fnirs`]: fNIRS optical frontend (ADS1115 + LED drivers)
//! - [`fnirs_array`]: High-density fNIRS with multiplexed optode arrays
//! - [`stim`]: Neurostimulation output (tDCS/tACS)
//! - [`stim_matrix`]: Electrode switching matrix for multi-site stimulation
//! - [`emg`]: Facial EMG for emotional valence detection
//! - [`eda`]: Electrodermal activity for autonomic arousal

pub mod ads1299;
pub mod ads1299_array;
pub mod eda;
pub mod emg;
pub mod fnirs;
pub mod fnirs_array;
pub mod stim;
pub mod stim_matrix;

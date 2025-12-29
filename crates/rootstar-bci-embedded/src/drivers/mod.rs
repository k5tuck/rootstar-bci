//! Hardware drivers for BCI acquisition
//!
//! This module contains drivers for the various hardware components:
//! - [`ads1299`]: TI ADS1299 24-bit EEG ADC
//! - [`fnirs`]: fNIRS optical frontend (ADS1115 + LED drivers)
//! - [`stim`]: Neurostimulation output (tDCS/tACS)

pub mod ads1299;
pub mod fnirs;
pub mod stim;

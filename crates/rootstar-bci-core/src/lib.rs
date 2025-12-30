//! Rootstar BCI Core - `no_std` compatible types and utilities
//!
//! This crate provides the foundational types, math utilities, and protocol
//! definitions for the Rootstar BCI Platform. It is designed to work in
//! `no_std` environments (embedded devices) as well as `std` environments.
//!
//! # Modules
//!
//! - [`types`]: Core data types (fixed-point, channels, samples, stimulation)
//! - [`error`]: Error types for drivers, processing, and protocol
//! - [`math`]: Fixed-point math, filters, and signal processing utilities
//! - [`protocol`]: Wire protocol for device communication
//!
//! # Features
//!
//! - `std`: Enable standard library support
//! - `defmt`: Enable `defmt` formatting for embedded logging
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::types::{EegSample, Fixed24_8, EegChannel};
//!
//! // Create a sample with fixed-point values
//! let mut sample = EegSample::new(1000, 1);
//! sample.set_channel(EegChannel::C3, Fixed24_8::from_f32(15.5));
//!
//! // Access channel data
//! let c3_value = sample.channel(EegChannel::C3);
//! assert!((c3_value.to_f32() - 15.5).abs() < 0.1);
//! ```

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod fingerprint;
pub mod math;
pub mod protocol;
pub mod sns;
pub mod types;

// Re-export commonly used types at crate root
pub use error::{Ads1299Error, FnirsError, ProcessingError, ProtocolError, StimError};
pub use math::{BeerLambertSolver, BiquadFilter, IirFilter, MovingAverage};
pub use protocol::{DeviceConfig, PacketHeader, PacketType};
pub use types::{
    EegBand, EegChannel, EegSample, EdaDecomposed, EdaSample, EdaSite, EmgChannel, EmgSample,
    Fixed24_8, FnirsChannel, FnirsSample, HemodynamicSample, StimMode, StimParams, Wavelength,
};

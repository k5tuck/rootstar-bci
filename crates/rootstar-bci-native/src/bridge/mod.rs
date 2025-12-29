//! Communication bridges for BCI data streaming
//!
//! This module provides bridges to external systems:
//! - [`usb`]: Direct USB communication with ESP-EEG (requires `usb` feature)
//! - [`lsl`]: Lab Streaming Layer integration (placeholder)

#[cfg(feature = "usb")]
pub mod usb;

// LSL and Brainflow would require external dependencies
// pub mod lsl;
// pub mod brainflow;

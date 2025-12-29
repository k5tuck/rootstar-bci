//! Rootstar BCI Embedded - ESP32 drivers and acquisition
//!
//! This crate provides hardware drivers for the ESP-EEG platform:
//! - ADS1299 EEG ADC driver (SPI)
//! - ADS1115 fNIRS ADC driver (I2C)
//! - LED driver for fNIRS sources (PWM)
//! - Stimulation output driver (DAC)
//! - Coordinated acquisition scheduler
//!
//! # Hardware Requirements
//!
//! - ESP32-WROOM-DA (as used in ESP-EEG)
//! - TI ADS1299 for EEG acquisition
//! - TI ADS1115 for fNIRS acquisition
//! - NIR LEDs at 760nm and 850nm
//! - OPT101 photodiode detectors
//!
//! # GPIO Assignments
//!
//! ```text
//! SPI (ADS1299):  MOSI=23, MISO=19, SCLK=18, CS=5, DRDY=4
//! I2C (fNIRS):    SDA=21, SCL=22
//! PWM (LEDs):     GPIO 25, 26, 27
//! DAC (Stim):     GPIO 32, 33
//! Status LED:     GPIO 17
//! ```

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod drivers;

// Re-export driver types
pub use drivers::ads1299::{Ads1299, Gain, SampleRate};
pub use drivers::fnirs::FnirsDriver;

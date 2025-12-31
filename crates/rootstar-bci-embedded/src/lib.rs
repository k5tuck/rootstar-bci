//! Rootstar BCI Embedded - ESP32 drivers and acquisition
//!
//! This crate provides hardware drivers for the ESP-EEG platform:
//! - ADS1299 EEG ADC driver (SPI)
//! - ADS1115 fNIRS ADC driver (I2C)
//! - LED driver for fNIRS sources (PWM)
//! - Stimulation output driver (DAC)
//! - Coordinated acquisition scheduler
//!
//! # Neural Fingerprint System (Phase 2)
//!
//! High-density acquisition and stimulation for neural fingerprinting:
//! - Daisy-chained ADS1299 arrays (8-256 EEG channels)
//! - Multiplexed fNIRS optode arrays (4-128 channels)
//! - Electrode switching matrix (8×8 = 64 electrodes)
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
//! Matrix:         I2C via MCP23017 expanders
//! Status LED:     GPIO 17
//! ```

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod ble;
pub mod drivers;

// Re-export driver types (basic single-chip)
pub use drivers::ads1299::{Ads1299, Gain, SampleRate};
pub use drivers::fnirs::FnirsDriver;

// Re-export high-density array drivers (Phase 2)
pub use drivers::ads1299_array::{Ads1299Array, ArrayConfig, HdEegSample};
pub use drivers::fnirs_array::{FnirsArray, FnirsArrayConfig, HdFnirsSample};
pub use drivers::stim_matrix::{MatrixStimProtocol, StimMatrix};

// Re-export EMG driver (facial muscle activity)
pub use drivers::emg::{EmgDriver, EmgFeatures, EmgGain, EmgSampleRate, EmgValenceCalculator};

// Re-export EDA driver (skin conductance)
pub use drivers::eda::{EdaArrayConfig, EdaDriver, EdaError, EdaProcessor, EdaSampleRate};

// Re-export BLE service definitions
pub use ble::{
    BleCommand, DeviceState, StatusPacket,
    BCI_SERVICE_UUID, EEG_DATA_CHAR_UUID, FNIRS_DATA_CHAR_UUID,
    COMMAND_CHAR_UUID, STATUS_CHAR_UUID, EMG_DATA_CHAR_UUID, EDA_DATA_CHAR_UUID,
    pack_eeg_samples, unpack_eeg_samples, pack_fnirs_samples,
};

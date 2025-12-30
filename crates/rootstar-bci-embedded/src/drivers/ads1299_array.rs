//! Multi-ADS1299 Array Driver for High-Density EEG
//!
//! Supports daisy-chained ADS1299 chips for 32-256 channel configurations.
//!
//! # Configurations
//!
//! - 4 chips (32 channels): Standard research
//! - 8 chips (64 channels): Clinical
//! - 16 chips (128 channels): High-density
//! - 32 chips (256 channels): Ultra high-density
//!
//! # Daisy Chain Operation
//!
//! ADS1299 chips share SCLK, MOSI, and START signals. Each chip has an
//! independent CS pin and outputs data to the next chip's DOUT→DIN.
//! Data is clocked out in reverse order (last chip first).

use core::marker::PhantomData;

use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::SpiDevice;
use heapless::Vec;

use rootstar_bci_core::error::Ads1299Error;
use rootstar_bci_core::fingerprint::ChannelDensity;
use rootstar_bci_core::types::Fixed24_8;

use super::ads1299::{Gain, SampleRate};

/// Maximum number of daisy-chained ADS1299 chips
pub const MAX_CHIPS: usize = 32;

/// Channels per ADS1299 chip
pub const CHANNELS_PER_CHIP: usize = 8;

/// Maximum total channels (32 chips × 8 channels)
pub const MAX_CHANNELS: usize = MAX_CHIPS * CHANNELS_PER_CHIP;

/// High-density EEG sample with variable channel count
#[derive(Clone, Debug)]
pub struct HdEegSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Channel values in microvolts (Q24.8 fixed-point)
    pub channels: Vec<Fixed24_8, MAX_CHANNELS>,
    /// Sequence number for packet ordering/loss detection
    pub sequence: u32,
    /// Number of active channels
    pub n_channels: u16,
}

impl HdEegSample {
    /// Create a new sample with zero values
    #[must_use]
    pub fn new(timestamp_us: u64, sequence: u32, n_channels: u16) -> Self {
        let mut channels = Vec::new();
        for _ in 0..n_channels {
            let _ = channels.push(Fixed24_8::ZERO);
        }
        Self { timestamp_us, channels, sequence, n_channels }
    }

    /// Get the value for a specific channel
    #[inline]
    pub fn channel(&self, ch: usize) -> Option<Fixed24_8> {
        self.channels.get(ch).copied()
    }

    /// Set the value for a specific channel
    #[inline]
    pub fn set_channel(&mut self, ch: usize, value: Fixed24_8) -> bool {
        if ch < self.channels.len() {
            self.channels[ch] = value;
            true
        } else {
            false
        }
    }
}

/// Configuration for ADS1299 array
#[derive(Clone, Debug)]
pub struct ArrayConfig {
    /// Number of chips in the array
    pub n_chips: u8,
    /// Sample rate for all chips
    pub sample_rate: SampleRate,
    /// Gain for all channels
    pub gain: Gain,
    /// Enable daisy chain mode
    pub daisy_chain: bool,
    /// Master chip index (0-based)
    pub master_chip: u8,
}

impl ArrayConfig {
    /// Create configuration from channel density
    #[must_use]
    pub fn from_density(density: ChannelDensity) -> Self {
        Self {
            n_chips: density.ads1299_count(),
            sample_rate: SampleRate::Sps500,
            gain: Gain::X24,
            daisy_chain: density.ads1299_count() > 1,
            master_chip: 0,
        }
    }

    /// Total number of channels
    #[must_use]
    pub const fn n_channels(&self) -> u16 {
        self.n_chips as u16 * CHANNELS_PER_CHIP as u16
    }

    /// Basic 8-channel configuration
    pub const BASIC_8: Self = Self {
        n_chips: 1,
        sample_rate: SampleRate::Sps500,
        gain: Gain::X24,
        daisy_chain: false,
        master_chip: 0,
    };

    /// Standard 32-channel configuration
    pub const STANDARD_32: Self = Self {
        n_chips: 4,
        sample_rate: SampleRate::Sps500,
        gain: Gain::X24,
        daisy_chain: true,
        master_chip: 0,
    };

    /// High-density 128-channel configuration
    pub const HIGH_DENSITY_128: Self = Self {
        n_chips: 16,
        sample_rate: SampleRate::Sps500,
        gain: Gain::X24,
        daisy_chain: true,
        master_chip: 0,
    };
}

impl Default for ArrayConfig {
    fn default() -> Self {
        Self::BASIC_8
    }
}

/// Chip status in the array
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChipStatus {
    /// Chip not initialized
    Uninitialized,
    /// Chip initialized and ready
    Ready,
    /// Chip initialization failed
    InitFailed,
    /// Chip not responding
    NotResponding,
}

/// Multi-chip ADS1299 array driver
///
/// Manages a daisy-chained array of ADS1299 chips for high-density EEG.
pub struct Ads1299Array<SPI, CS, DRDY, RST, const N: usize> {
    spi: SPI,
    /// Chip select pins (one per chip)
    cs_pins: [CS; N],
    /// DRDY pins (can be shared or individual)
    drdy: DRDY,
    /// Reset pin (shared across all chips)
    reset: RST,
    /// Configuration
    config: ArrayConfig,
    /// Status of each chip
    chip_status: [ChipStatus; N],
    /// Sequence counter
    sequence: u32,
    /// Marker for SPI error type
    _spi: PhantomData<fn() -> SPI>,
}

impl<SPI, CS, DRDY, RST, E, const N: usize> Ads1299Array<SPI, CS, DRDY, RST, N>
where
    SPI: SpiDevice<Error = E>,
    CS: OutputPin,
    DRDY: InputPin,
    RST: OutputPin,
{
    /// ADS1299 commands
    const CMD_WAKEUP: u8 = 0x02;
    const CMD_STANDBY: u8 = 0x04;
    const CMD_RESET: u8 = 0x06;
    const CMD_START: u8 = 0x08;
    const CMD_STOP: u8 = 0x0A;
    const CMD_RDATAC: u8 = 0x10;
    const CMD_SDATAC: u8 = 0x11;
    const CMD_RREG: u8 = 0x20;
    const CMD_WREG: u8 = 0x40;

    /// ADS1299 registers
    const REG_ID: u8 = 0x00;
    const REG_CONFIG1: u8 = 0x01;
    const REG_CONFIG2: u8 = 0x02;
    const REG_CONFIG3: u8 = 0x03;
    const REG_CH1SET: u8 = 0x05;
    const REG_BIAS_SENSP: u8 = 0x0D;
    const REG_BIAS_SENSN: u8 = 0x0E;
    const REG_MISC1: u8 = 0x15;

    /// Create a new ADS1299 array driver
    pub fn new(spi: SPI, cs_pins: [CS; N], drdy: DRDY, reset: RST, config: ArrayConfig) -> Self {
        Self {
            spi,
            cs_pins,
            drdy,
            reset,
            config,
            chip_status: [ChipStatus::Uninitialized; N],
            sequence: 0,
            _spi: PhantomData,
        }
    }

    /// Initialize all chips in the array
    pub fn init(&mut self) -> Result<u8, Ads1299Error<E>> {
        // Hardware reset (shared reset line)
        let _ = self.reset.set_low();
        cortex_m::asm::delay(100_000); // ~1ms
        let _ = self.reset.set_high();
        cortex_m::asm::delay(500_000); // ~5ms power-up

        let mut initialized = 0u8;

        // Initialize each chip
        for chip in 0..N.min(self.config.n_chips as usize) {
            match self.init_chip(chip) {
                Ok(()) => {
                    self.chip_status[chip] = ChipStatus::Ready;
                    initialized += 1;
                }
                Err(_) => {
                    self.chip_status[chip] = ChipStatus::InitFailed;
                }
            }
        }

        if initialized == 0 {
            return Err(Ads1299Error::InvalidDeviceId { got: 0, expected: 0x3E });
        }

        Ok(initialized)
    }

    /// Initialize a single chip
    fn init_chip(&mut self, chip: usize) -> Result<(), Ads1299Error<E>> {
        if chip >= N {
            return Err(Ads1299Error::InvalidDeviceId { got: chip as u8, expected: 0 });
        }

        // Stop continuous read mode
        self.send_command_to_chip(chip, Self::CMD_SDATAC)?;

        // Verify device ID
        let id = self.read_register_from_chip(chip, Self::REG_ID)?;
        if (id & 0x1F) != 0x1E {
            return Err(Ads1299Error::InvalidDeviceId { got: id, expected: 0x3E });
        }

        // Configure for EEG
        self.configure_chip_for_eeg(chip)?;

        Ok(())
    }

    /// Configure a chip for EEG acquisition
    fn configure_chip_for_eeg(&mut self, chip: usize) -> Result<(), Ads1299Error<E>> {
        // CONFIG1: Sample rate, daisy chain
        let config1 = if self.config.daisy_chain && chip > 0 {
            0xD0 | (self.config.sample_rate as u8) // Daisy chain enabled
        } else {
            0x90 | (self.config.sample_rate as u8) // Daisy chain disabled (master)
        };
        self.write_register_to_chip(chip, Self::REG_CONFIG1, config1)?;

        // CONFIG2: Internal test signal disabled
        self.write_register_to_chip(chip, Self::REG_CONFIG2, 0xC0)?;

        // CONFIG3: Internal reference, bias enabled
        self.write_register_to_chip(chip, Self::REG_CONFIG3, 0xEC)?;

        // Configure all channels with gain
        let ch_config = (self.config.gain as u8) << 4;
        for ch_reg in Self::REG_CH1SET..Self::REG_CH1SET + 8 {
            self.write_register_to_chip(chip, ch_reg, ch_config)?;
        }

        // BIAS routing
        self.write_register_to_chip(chip, Self::REG_BIAS_SENSP, 0xFF)?;
        self.write_register_to_chip(chip, Self::REG_BIAS_SENSN, 0xFF)?;

        // MISC1: SRB1 as common reference
        self.write_register_to_chip(chip, Self::REG_MISC1, 0x20)?;

        Ok(())
    }

    /// Start synchronized acquisition on all chips
    pub fn start_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        // Send START to all chips simultaneously
        for chip in 0..N.min(self.config.n_chips as usize) {
            if self.chip_status[chip] == ChipStatus::Ready {
                self.send_command_to_chip(chip, Self::CMD_START)?;
            }
        }

        // Enable continuous data mode
        for chip in 0..N.min(self.config.n_chips as usize) {
            if self.chip_status[chip] == ChipStatus::Ready {
                self.send_command_to_chip(chip, Self::CMD_RDATAC)?;
            }
        }

        Ok(())
    }

    /// Stop acquisition on all chips
    pub fn stop_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        for chip in 0..N.min(self.config.n_chips as usize) {
            if self.chip_status[chip] == ChipStatus::Ready {
                let _ = self.send_command_to_chip(chip, Self::CMD_SDATAC);
                let _ = self.send_command_to_chip(chip, Self::CMD_STOP);
            }
        }
        Ok(())
    }

    /// Check if new data is ready
    #[inline]
    pub fn data_ready(&mut self) -> bool {
        self.drdy.is_low().unwrap_or(false)
    }

    /// Read a synchronized sample from all chips
    pub fn read_sample(&mut self, timestamp_us: u64) -> Result<HdEegSample, Ads1299Error<E>> {
        let n_active = self.config.n_chips as usize;
        let n_channels = n_active * CHANNELS_PER_CHIP;

        let mut sample = HdEegSample::new(timestamp_us, self.sequence, n_channels as u16);
        self.sequence = self.sequence.wrapping_add(1);

        // Read from each chip in daisy chain order (last to first)
        for chip in (0..n_active.min(N)).rev() {
            if self.chip_status[chip] != ChipStatus::Ready {
                continue;
            }

            // Read 27 bytes: 3 status + 8 × 3 channel data
            let mut buffer = [0u8; 27];

            let _ = self.cs_pins[chip].set_low();
            self.spi.transfer_in_place(&mut buffer).map_err(Ads1299Error::Spi)?;
            let _ = self.cs_pins[chip].set_high();

            // Parse channel data
            let channel_offset = chip * CHANNELS_PER_CHIP;
            for ch in 0..CHANNELS_PER_CHIP {
                let offset = 3 + ch * 3;
                let raw = ((buffer[offset] as i32) << 24)
                    | ((buffer[offset + 1] as i32) << 16)
                    | ((buffer[offset + 2] as i32) << 8);
                let raw = raw >> 8; // Sign extend

                let value = Fixed24_8::from_ads1299_raw(raw, self.config.gain.multiplier());
                sample.set_channel(channel_offset + ch, value);
            }
        }

        Ok(sample)
    }

    /// Get status of all chips
    pub fn chip_statuses(&self) -> &[ChipStatus; N] {
        &self.chip_status
    }

    /// Get number of ready chips
    pub fn ready_chip_count(&self) -> u8 {
        self.chip_status
            .iter()
            .filter(|&&s| s == ChipStatus::Ready)
            .count() as u8
    }

    fn send_command_to_chip(&mut self, chip: usize, cmd: u8) -> Result<(), Ads1299Error<E>> {
        if chip >= N {
            return Ok(());
        }
        let _ = self.cs_pins[chip].set_low();
        self.spi.write(&[cmd]).map_err(Ads1299Error::Spi)?;
        let _ = self.cs_pins[chip].set_high();
        Ok(())
    }

    fn read_register_from_chip(&mut self, chip: usize, addr: u8) -> Result<u8, Ads1299Error<E>> {
        if chip >= N {
            return Err(Ads1299Error::InvalidDeviceId { got: chip as u8, expected: 0 });
        }
        let mut buffer = [Self::CMD_RREG | addr, 0x00, 0x00];
        let _ = self.cs_pins[chip].set_low();
        self.spi.transfer_in_place(&mut buffer).map_err(Ads1299Error::Spi)?;
        let _ = self.cs_pins[chip].set_high();
        Ok(buffer[2])
    }

    fn write_register_to_chip(&mut self, chip: usize, addr: u8, value: u8) -> Result<(), Ads1299Error<E>> {
        if chip >= N {
            return Ok(());
        }
        let buffer = [Self::CMD_WREG | addr, 0x00, value];
        let _ = self.cs_pins[chip].set_low();
        self.spi.write(&buffer).map_err(Ads1299Error::Spi)?;
        let _ = self.cs_pins[chip].set_high();
        Ok(())
    }
}

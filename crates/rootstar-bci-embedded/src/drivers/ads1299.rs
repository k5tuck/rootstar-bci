//! ADS1299 EEG ADC Driver
//!
//! Driver for the Texas Instruments ADS1299, a 24-bit, 8-channel ADC
//! designed for biopotential measurements (EEG, ECG, EMG).
//!
//! # Features
//!
//! - 24-bit resolution per channel
//! - Programmable gain (1-24x)
//! - Sample rates: 250, 500, 1000, 2000 SPS
//! - Built-in bias drive (DRL) for common-mode rejection
//!
//! # Example
//!
//! ```ignore
//! let mut ads = Ads1299::new(spi, cs, drdy, reset);
//! ads.init()?;
//! ads.start_acquisition()?;
//!
//! loop {
//!     if ads.data_ready() {
//!         let sample = ads.read_sample(timestamp, sequence)?;
//!         // Process sample...
//!     }
//! }
//! ```

use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::SpiDevice;

use rootstar_bci_core::error::Ads1299Error;
use rootstar_bci_core::types::{EegSample, Fixed24_8};

/// ADS1299 register addresses
#[allow(dead_code)]
mod regs {
    pub const ID: u8 = 0x00;
    pub const CONFIG1: u8 = 0x01;
    pub const CONFIG2: u8 = 0x02;
    pub const CONFIG3: u8 = 0x03;
    pub const LOFF: u8 = 0x04;
    pub const CH1SET: u8 = 0x05;
    pub const CH2SET: u8 = 0x06;
    pub const CH3SET: u8 = 0x07;
    pub const CH4SET: u8 = 0x08;
    pub const CH5SET: u8 = 0x09;
    pub const CH6SET: u8 = 0x0A;
    pub const CH7SET: u8 = 0x0B;
    pub const CH8SET: u8 = 0x0C;
    pub const BIAS_SENSP: u8 = 0x0D;
    pub const BIAS_SENSN: u8 = 0x0E;
    pub const LOFF_SENSP: u8 = 0x0F;
    pub const LOFF_SENSN: u8 = 0x10;
    pub const LOFF_FLIP: u8 = 0x11;
    pub const LOFF_STATP: u8 = 0x12;
    pub const LOFF_STATN: u8 = 0x13;
    pub const GPIO: u8 = 0x14;
    pub const MISC1: u8 = 0x15;
    pub const MISC2: u8 = 0x16;
    pub const CONFIG4: u8 = 0x17;
}

/// ADS1299 commands
mod cmd {
    pub const WAKEUP: u8 = 0x02;
    pub const STANDBY: u8 = 0x04;
    pub const RESET: u8 = 0x06;
    pub const START: u8 = 0x08;
    pub const STOP: u8 = 0x0A;
    pub const RDATAC: u8 = 0x10;
    pub const SDATAC: u8 = 0x11;
    pub const RDATA: u8 = 0x12;
    pub const RREG: u8 = 0x20;
    pub const WREG: u8 = 0x40;
}

/// Sample rate configuration
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SampleRate {
    /// 250 Hz (default)
    Sps250 = 0x06,
    /// 500 Hz
    Sps500 = 0x05,
    /// 1000 Hz
    Sps1000 = 0x04,
    /// 2000 Hz
    Sps2000 = 0x03,
}

impl SampleRate {
    /// Get sample rate in Hz
    #[must_use]
    pub const fn hz(self) -> u16 {
        match self {
            Self::Sps250 => 250,
            Self::Sps500 => 500,
            Self::Sps1000 => 1000,
            Self::Sps2000 => 2000,
        }
    }
}

/// Programmable gain setting
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Gain {
    /// 1x gain
    X1 = 0x00,
    /// 2x gain
    X2 = 0x01,
    /// 4x gain
    X4 = 0x02,
    /// 6x gain
    X6 = 0x03,
    /// 8x gain
    X8 = 0x04,
    /// 12x gain
    X12 = 0x05,
    /// 24x gain (default for EEG)
    X24 = 0x06,
}

impl Gain {
    /// Get the gain multiplier
    #[must_use]
    pub const fn multiplier(self) -> u8 {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X6 => 6,
            Self::X8 => 8,
            Self::X12 => 12,
            Self::X24 => 24,
        }
    }
}

/// ADS1299 driver
pub struct Ads1299<SPI, CS, DRDY, RST> {
    spi: SPI,
    cs: CS,
    drdy: DRDY,
    reset: RST,
    gain: Gain,
    sample_rate: SampleRate,
}

impl<SPI, CS, DRDY, RST, E> Ads1299<SPI, CS, DRDY, RST>
where
    SPI: SpiDevice<Error = E>,
    CS: OutputPin,
    DRDY: InputPin,
    RST: OutputPin,
{
    /// Create a new ADS1299 driver
    #[must_use]
    pub fn new(spi: SPI, cs: CS, drdy: DRDY, reset: RST) -> Self {
        Self {
            spi,
            cs,
            drdy,
            reset,
            gain: Gain::X24,
            sample_rate: SampleRate::Sps250,
        }
    }

    /// Set the programmable gain
    pub fn set_gain(&mut self, gain: Gain) {
        self.gain = gain;
    }

    /// Set the sample rate
    pub fn set_sample_rate(&mut self, rate: SampleRate) {
        self.sample_rate = rate;
    }

    /// Get current gain setting
    #[must_use]
    pub fn gain(&self) -> Gain {
        self.gain
    }

    /// Get current sample rate setting
    #[must_use]
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Initialize the ADS1299
    ///
    /// Performs hardware reset, verifies device ID, and configures for EEG acquisition.
    pub fn init(&mut self) -> Result<(), Ads1299Error<E>> {
        // Hardware reset
        let _ = self.reset.set_low();
        cortex_m::asm::delay(100_000); // ~1ms at 100MHz
        let _ = self.reset.set_high();
        cortex_m::asm::delay(500_000); // ~5ms power-up delay

        // Stop continuous read mode
        self.send_command(cmd::SDATAC)?;

        // Verify device ID (should be 0x3E for ADS1299)
        let id = self.read_register(regs::ID)?;
        if (id & 0x1F) != 0x1E {
            return Err(Ads1299Error::InvalidDeviceId { got: id, expected: 0x3E });
        }

        // Configure for EEG acquisition
        self.configure_for_eeg()?;

        Ok(())
    }

    /// Configure registers for EEG acquisition with active bias (DRL)
    fn configure_for_eeg(&mut self) -> Result<(), Ads1299Error<E>> {
        // CONFIG1: Sample rate, daisy chain disabled
        self.write_register(regs::CONFIG1, 0x90 | (self.sample_rate as u8))?;

        // CONFIG2: Internal test signal disabled, reference buffer enabled
        self.write_register(regs::CONFIG2, 0xC0)?;

        // CONFIG3: Internal reference, bias enabled with bias buffer
        self.write_register(regs::CONFIG3, 0xEC)?;

        // Configure all channels for normal operation with gain
        let ch_config = (self.gain as u8) << 4;
        for ch_reg in regs::CH1SET..=regs::CH8SET {
            self.write_register(ch_reg, ch_config)?;
        }

        // BIAS_SENSP/N: Route all channels to bias derivation
        self.write_register(regs::BIAS_SENSP, 0xFF)?;
        self.write_register(regs::BIAS_SENSN, 0xFF)?;

        // MISC1: SRB1 as common reference
        self.write_register(regs::MISC1, 0x20)?;

        Ok(())
    }

    /// Start continuous data acquisition
    pub fn start_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        self.send_command(cmd::START)?;
        self.send_command(cmd::RDATAC)?;
        Ok(())
    }

    /// Stop data acquisition
    pub fn stop_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        self.send_command(cmd::SDATAC)?;
        self.send_command(cmd::STOP)?;
        Ok(())
    }

    /// Check if new data is ready (DRDY pin low)
    #[inline]
    pub fn data_ready(&mut self) -> bool {
        self.drdy.is_low().unwrap_or(false)
    }

    /// Read a single sample (call when `data_ready()` returns true)
    pub fn read_sample(&mut self, timestamp_us: u64, sequence: u32) -> Result<EegSample, Ads1299Error<E>> {
        // Data format: 3 bytes status + 8 channels × 3 bytes = 27 bytes
        let mut buffer = [0u8; 27];

        let _ = self.cs.set_low();
        self.spi.transfer_in_place(&mut buffer).map_err(Ads1299Error::Spi)?;
        let _ = self.cs.set_high();

        // Parse channel data (24-bit signed, two's complement)
        let mut channels = [Fixed24_8::ZERO; 8];
        for (i, ch) in channels.iter_mut().enumerate() {
            let offset = 3 + i * 3;
            let raw = ((buffer[offset] as i32) << 24)
                | ((buffer[offset + 1] as i32) << 16)
                | ((buffer[offset + 2] as i32) << 8);
            // Sign-extend from 24-bit to 32-bit
            let raw = raw >> 8;

            *ch = Fixed24_8::from_ads1299_raw(raw, self.gain.multiplier());
        }

        Ok(EegSample { timestamp_us, channels, sequence })
    }

    fn send_command(&mut self, cmd: u8) -> Result<(), Ads1299Error<E>> {
        let _ = self.cs.set_low();
        self.spi.write(&[cmd]).map_err(Ads1299Error::Spi)?;
        let _ = self.cs.set_high();
        Ok(())
    }

    fn read_register(&mut self, addr: u8) -> Result<u8, Ads1299Error<E>> {
        let mut buffer = [cmd::RREG | addr, 0x00, 0x00];
        let _ = self.cs.set_low();
        self.spi.transfer_in_place(&mut buffer).map_err(Ads1299Error::Spi)?;
        let _ = self.cs.set_high();
        Ok(buffer[2])
    }

    fn write_register(&mut self, addr: u8, value: u8) -> Result<(), Ads1299Error<E>> {
        let buffer = [cmd::WREG | addr, 0x00, value];
        let _ = self.cs.set_low();
        self.spi.write(&buffer).map_err(Ads1299Error::Spi)?;
        let _ = self.cs.set_high();
        Ok(())
    }
}

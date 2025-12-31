//! High-Density fNIRS Array Driver
//!
//! Supports multi-source, multi-detector fNIRS configurations for
//! diffuse optical tomography (DOT) and high-resolution hemodynamic imaging.
//!
//! # Features
//!
//! - Up to 48 LED sources (dual wavelength: 760nm + 850nm)
//! - Up to 48 photodiode detectors
//! - Multiplexed LED control for sequential illumination
//! - Multiple ADC support (ADS1115 or dedicated fNIRS ADC)
//! - Configurable source-detector geometry

use core::marker::PhantomData;

use embedded_hal::digital::OutputPin;
use embedded_hal::i2c::I2c;
use heapless::Vec;

use rootstar_bci_core::error::FnirsError;
use rootstar_bci_core::types::{FnirsChannel, FnirsSample, Wavelength};

/// Maximum number of LED sources
pub const MAX_SOURCES: usize = 48;

/// Maximum number of photodiode detectors
pub const MAX_DETECTORS: usize = 48;

/// Maximum number of measurement channels
pub const MAX_FNIRS_CHANNELS: usize = 128;

/// LED control mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedControlMode {
    /// Direct GPIO control (up to 8 LEDs)
    DirectGpio,
    /// I2C multiplexer (PCA9685 or similar)
    I2cMultiplexer,
    /// SPI shift register (74HC595 chain)
    SpiShiftRegister,
}

/// fNIRS channel type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelType {
    /// Long channel for cortical measurement (30-40mm)
    LongChannel,
    /// Short channel for superficial regression (10-15mm)
    ShortChannel,
}

/// fNIRS measurement channel definition
#[derive(Clone, Copy, Debug)]
pub struct MeasurementChannel {
    /// Source LED index
    pub source: u8,
    /// Detector photodiode index
    pub detector: u8,
    /// Source-detector separation in mm
    pub separation_mm: u8,
    /// Channel type
    pub channel_type: ChannelType,
    /// ADC channel for this measurement
    pub adc_channel: u8,
    /// ADC device index (for multi-ADC systems)
    pub adc_device: u8,
}

impl MeasurementChannel {
    /// Create a new long-channel measurement
    #[must_use]
    pub const fn long(source: u8, detector: u8, adc_device: u8, adc_channel: u8) -> Self {
        Self {
            source,
            detector,
            separation_mm: 30,
            channel_type: ChannelType::LongChannel,
            adc_channel,
            adc_device,
        }
    }

    /// Create a new short-channel measurement
    #[must_use]
    pub const fn short(source: u8, detector: u8, adc_device: u8, adc_channel: u8) -> Self {
        Self {
            source,
            detector,
            separation_mm: 15,
            channel_type: ChannelType::ShortChannel,
            adc_channel,
            adc_device,
        }
    }

    /// Convert to core FnirsChannel type
    #[must_use]
    pub const fn to_fnirs_channel(&self) -> FnirsChannel {
        FnirsChannel::new(self.source, self.detector, self.separation_mm)
    }
}

/// Configuration for high-density fNIRS array
#[derive(Clone, Debug)]
pub struct FnirsArrayConfig {
    /// Number of LED sources
    pub n_sources: u8,
    /// Number of detectors
    pub n_detectors: u8,
    /// LED control mode
    pub led_mode: LedControlMode,
    /// Number of ADC devices
    pub n_adcs: u8,
    /// Channels per ADC
    pub channels_per_adc: u8,
    /// Sample rate in Hz
    pub sample_rate_hz: u8,
    /// LED illumination duration in microseconds
    pub led_on_time_us: u16,
    /// Dark period between illuminations in microseconds
    pub dark_period_us: u16,
    /// Measurement channels
    pub channels: Vec<MeasurementChannel, MAX_FNIRS_CHANNELS>,
}

impl FnirsArrayConfig {
    /// Basic 4-channel configuration (2×2)
    pub fn basic() -> Self {
        let mut channels = Vec::new();
        let _ = channels.push(MeasurementChannel::long(0, 0, 0, 0));
        let _ = channels.push(MeasurementChannel::long(0, 1, 0, 1));
        let _ = channels.push(MeasurementChannel::long(1, 0, 0, 2));
        let _ = channels.push(MeasurementChannel::long(1, 1, 0, 3));

        Self {
            n_sources: 2,
            n_detectors: 2,
            led_mode: LedControlMode::DirectGpio,
            n_adcs: 1,
            channels_per_adc: 4,
            sample_rate_hz: 10,
            led_on_time_us: 1000,
            dark_period_us: 500,
            channels,
        }
    }

    /// Standard 16-channel configuration (4×4)
    pub fn standard() -> Self {
        let mut channels = Vec::new();

        // Create 16 long channels
        for src in 0..4 {
            for det in 0..4 {
                let _ = channels.push(MeasurementChannel::long(
                    src,
                    det,
                    (src / 2) as u8, // Distribute across 2 ADCs
                    (det + (src % 2) * 4) as u8,
                ));
            }
        }

        Self {
            n_sources: 4,
            n_detectors: 4,
            led_mode: LedControlMode::I2cMultiplexer,
            n_adcs: 2,
            channels_per_adc: 8,
            sample_rate_hz: 25,
            led_on_time_us: 500,
            dark_period_us: 200,
            channels,
        }
    }

    /// High-density 64-channel DOT configuration
    pub fn high_density() -> Self {
        let mut channels = Vec::new();

        // Create channels for DOT with overlapping geometry
        for src in 0..32 {
            for det in 0..2 {
                // Each source illuminates 2 nearest detectors
                let det_idx = (src / 4 * 2 + det) % 16;
                let _ = channels.push(MeasurementChannel::long(
                    src as u8,
                    det_idx as u8,
                    (src / 8) as u8,
                    (src % 8) as u8,
                ));
            }
        }

        // Add short channels for superficial regression
        for i in 0..16 {
            let _ = channels.push(MeasurementChannel::short(
                (i * 2) as u8,
                (i * 2) as u8,
                4, // Dedicated short-channel ADC
                i as u8,
            ));
        }

        Self {
            n_sources: 32,
            n_detectors: 32,
            led_mode: LedControlMode::SpiShiftRegister,
            n_adcs: 5, // 4 for long + 1 for short channels
            channels_per_adc: 16,
            sample_rate_hz: 25,
            led_on_time_us: 300,
            dark_period_us: 100,
            channels,
        }
    }

    /// Total number of measurement channels
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }
}

impl Default for FnirsArrayConfig {
    fn default() -> Self {
        Self::basic()
    }
}

/// High-density fNIRS sample
#[derive(Clone, Debug)]
pub struct HdFnirsSample {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Sequence number
    pub sequence: u32,
    /// Intensity values at 760nm per channel
    pub intensity_760: Vec<u16, MAX_FNIRS_CHANNELS>,
    /// Intensity values at 850nm per channel
    pub intensity_850: Vec<u16, MAX_FNIRS_CHANNELS>,
    /// Number of channels
    pub n_channels: u16,
}

impl HdFnirsSample {
    /// Create a new sample
    #[must_use]
    pub fn new(timestamp_us: u64, sequence: u32, n_channels: u16) -> Self {
        let mut intensity_760 = Vec::new();
        let mut intensity_850 = Vec::new();
        for _ in 0..n_channels {
            let _ = intensity_760.push(0);
            let _ = intensity_850.push(0);
        }
        Self {
            timestamp_us,
            sequence,
            intensity_760,
            intensity_850,
            n_channels,
        }
    }

    /// Get intensity for a channel and wavelength
    pub fn get_intensity(&self, channel: usize, wavelength: Wavelength) -> Option<u16> {
        match wavelength {
            Wavelength::Nm760 => self.intensity_760.get(channel).copied(),
            Wavelength::Nm850 => self.intensity_850.get(channel).copied(),
        }
    }

    /// Convert to standard FnirsSample for a specific channel
    pub fn to_fnirs_sample(&self, channel: &MeasurementChannel, ch_idx: usize) -> Option<FnirsSample> {
        let i760 = self.intensity_760.get(ch_idx)?;
        let i850 = self.intensity_850.get(ch_idx)?;

        Some(FnirsSample::new(
            self.timestamp_us,
            channel.to_fnirs_channel(),
            *i760,
            *i850,
            self.sequence,
        ))
    }
}

/// LED controller abstraction
pub trait LedController {
    /// Error type
    type Error;

    /// Turn on a specific LED source
    fn set_source(&mut self, source: u8, wavelength: Wavelength, intensity: u8) -> Result<(), Self::Error>;

    /// Turn off all LEDs
    fn all_off(&mut self) -> Result<(), Self::Error>;
}

/// ADC abstraction for fNIRS
pub trait FnirsAdc {
    /// Error type
    type Error;

    /// Read a single channel
    fn read_channel(&mut self, channel: u8) -> Result<u16, Self::Error>;

    /// Start conversion on all channels
    fn start_conversion(&mut self) -> Result<(), Self::Error>;

    /// Check if conversion is complete
    fn conversion_ready(&mut self) -> bool;
}

/// High-density fNIRS array driver
///
/// Coordinates LED multiplexing and multi-channel ADC acquisition.
pub struct FnirsArray<LED, ADC, const N_LED: usize, const N_ADC: usize> {
    /// LED controller
    led_controller: LED,
    /// ADC devices
    adcs: [ADC; N_ADC],
    /// Configuration
    config: FnirsArrayConfig,
    /// Sequence counter
    sequence: u32,
    /// Current source being illuminated
    current_source: u8,
    /// Current wavelength
    current_wavelength: Wavelength,
    /// Marker
    _phantom: PhantomData<fn() -> (LED, ADC)>,
}

impl<LED, ADC, ELED, EADC, const N_LED: usize, const N_ADC: usize>
    FnirsArray<LED, ADC, N_LED, N_ADC>
where
    LED: LedController<Error = ELED>,
    ADC: FnirsAdc<Error = EADC>,
{
    /// Create a new fNIRS array driver
    pub fn new(led_controller: LED, adcs: [ADC; N_ADC], config: FnirsArrayConfig) -> Self {
        Self {
            led_controller,
            adcs,
            config,
            sequence: 0,
            current_source: 0,
            current_wavelength: Wavelength::Nm760,
            _phantom: PhantomData,
        }
    }

    /// Initialize the fNIRS array
    pub fn init(&mut self) -> Result<(), FnirsError<EADC>> {
        // Turn off all LEDs
        let _ = self.led_controller.all_off();

        // Initialize ADCs would happen here
        // (specific to ADC implementation)

        Ok(())
    }

    /// Acquire a single sample (full multiplexing cycle)
    ///
    /// This performs a complete measurement cycle:
    /// 1. For each source:
    ///    a. Illuminate with 760nm
    ///    b. Read all detectors
    ///    c. Illuminate with 850nm
    ///    d. Read all detectors
    pub fn acquire_sample(&mut self, timestamp_us: u64) -> Result<HdFnirsSample, FnirsError<EADC>> {
        let n_channels = self.config.n_channels();
        let mut sample = HdFnirsSample::new(timestamp_us, self.sequence, n_channels as u16);
        self.sequence = self.sequence.wrapping_add(1);

        // Iterate through all sources
        for source in 0..self.config.n_sources {
            // Measure at 760nm
            self.measure_wavelength(source, Wavelength::Nm760, &mut sample)?;

            // Measure at 850nm
            self.measure_wavelength(source, Wavelength::Nm850, &mut sample)?;
        }

        // Dark measurement (LEDs off)
        let _ = self.led_controller.all_off();

        Ok(sample)
    }

    /// Measure at a specific wavelength
    fn measure_wavelength(
        &mut self,
        source: u8,
        wavelength: Wavelength,
        sample: &mut HdFnirsSample,
    ) -> Result<(), FnirsError<EADC>> {
        // Turn on LED
        let _ = self.led_controller.set_source(source, wavelength, 255);

        // Wait for LED to stabilize (would use timer in real impl)
        // cortex_m::asm::delay(self.config.led_on_time_us as u32 * 100);

        // Read all detectors for channels with this source
        for (ch_idx, channel) in self.config.channels.iter().enumerate() {
            if channel.source != source {
                continue;
            }

            let adc_idx = channel.adc_device as usize;
            if adc_idx >= N_ADC {
                continue;
            }

            // Read ADC
            let intensity = self.adcs[adc_idx]
                .read_channel(channel.adc_channel)
                .map_err(FnirsError::I2c)?;

            // Store in appropriate wavelength array
            match wavelength {
                Wavelength::Nm760 => {
                    if ch_idx < sample.intensity_760.len() {
                        sample.intensity_760[ch_idx] = intensity;
                    }
                }
                Wavelength::Nm850 => {
                    if ch_idx < sample.intensity_850.len() {
                        sample.intensity_850[ch_idx] = intensity;
                    }
                }
            }
        }

        // Turn off LED
        let _ = self.led_controller.all_off();

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &FnirsArrayConfig {
        &self.config
    }
}

/// Simple GPIO-based LED controller for basic configurations
pub struct GpioLedController<LED760, LED850> {
    leds_760: [LED760; 4],
    leds_850: [LED850; 4],
}

impl<LED760, LED850, E> LedController for GpioLedController<LED760, LED850>
where
    LED760: OutputPin<Error = E>,
    LED850: OutputPin<Error = E>,
{
    type Error = E;

    fn set_source(&mut self, source: u8, wavelength: Wavelength, _intensity: u8) -> Result<(), E> {
        // Turn off all first
        for led in &mut self.leds_760 {
            let _ = led.set_low();
        }
        for led in &mut self.leds_850 {
            let _ = led.set_low();
        }

        // Turn on selected source
        let idx = (source as usize) % 4;
        match wavelength {
            Wavelength::Nm760 => self.leds_760[idx].set_high(),
            Wavelength::Nm850 => self.leds_850[idx].set_high(),
        }
    }

    fn all_off(&mut self) -> Result<(), E> {
        for led in &mut self.leds_760 {
            let _ = led.set_low();
        }
        for led in &mut self.leds_850 {
            let _ = led.set_low();
        }
        Ok(())
    }
}

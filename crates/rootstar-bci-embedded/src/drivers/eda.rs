//! EDA (Electrodermal Activity) Driver
//!
//! Driver for skin conductance measurement using the TI ADS1115 ADC.
//! EDA measures the electrical conductivity of skin, which varies with
//! sweat gland activity controlled by the sympathetic nervous system.
//!
//! # Signal Components
//!
//! - **SCL (Skin Conductance Level)**: Tonic, slow-varying baseline
//! - **SCR (Skin Conductance Response)**: Phasic, event-related peaks
//!
//! # Hardware Setup
//!
//! - ADS1115 16-bit ADC (I2C)
//! - Ag/AgCl electrodes on palmar or thenar sites
//! - Constant voltage circuit (typically 0.5V DC)
//!
//! # Example
//!
//! ```ignore
//! let mut eda = EdaDriver::new(i2c, 0x48);
//! eda.init()?;
//!
//! loop {
//!     let sample = eda.sample(timestamp, sequence)?;
//!     // Process EDA sample...
//! }
//! ```

use embedded_hal::i2c::I2c;

use rootstar_bci_core::types::{EdaSample, EdaSite, Fixed24_8};

/// EDA-specific error type
#[derive(Debug)]
pub enum EdaError<E> {
    /// I2C communication error
    I2c(E),
    /// ADC value out of expected range
    ValueOutOfRange { value: u16, site: u8 },
    /// Calibration failed
    CalibrationFailed,
}

/// ADS1115 register addresses
mod ads1115 {
    pub const CONVERSION: u8 = 0x00;
    pub const CONFIG: u8 = 0x01;
    pub const LO_THRESH: u8 = 0x02;
    pub const HI_THRESH: u8 = 0x03;
}

/// ADS1115 configuration bits
mod config {
    /// Start single conversion
    pub const OS_SINGLE: u16 = 0x8000;
    /// Input multiplexer (AINp = AIN0, AINn = GND)
    pub const MUX_AIN0: u16 = 0x4000;
    pub const MUX_AIN1: u16 = 0x5000;
    pub const MUX_AIN2: u16 = 0x6000;
    pub const MUX_AIN3: u16 = 0x7000;
    /// Programmable gain: ±4.096V
    pub const PGA_4V: u16 = 0x0200;
    /// Single-shot mode
    pub const MODE_SINGLE: u16 = 0x0100;
    /// Data rate: 128 SPS
    pub const DR_128: u16 = 0x0080;
    /// Comparator disable
    pub const COMP_DISABLE: u16 = 0x0003;
}

/// EDA sample rate
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EdaSampleRate {
    /// 8 samples per second
    Sps8,
    /// 16 samples per second
    Sps16,
    /// 32 samples per second
    Sps32,
    /// 64 samples per second
    Sps64,
    /// 128 samples per second
    Sps128,
}

impl EdaSampleRate {
    /// Get the ADS1115 data rate configuration bits
    fn config_bits(self) -> u16 {
        match self {
            Self::Sps8 => 0x0000,
            Self::Sps16 => 0x0020,
            Self::Sps32 => 0x0040,
            Self::Sps64 => 0x0060,
            Self::Sps128 => 0x0080,
        }
    }

    /// Get rate in Hz
    pub fn hz(self) -> u8 {
        match self {
            Self::Sps8 => 8,
            Self::Sps16 => 16,
            Self::Sps32 => 32,
            Self::Sps64 => 64,
            Self::Sps128 => 128,
        }
    }
}

impl Default for EdaSampleRate {
    fn default() -> Self {
        Self::Sps32
    }
}

/// EDA driver using ADS1115
pub struct EdaDriver<I2C> {
    i2c: I2C,
    addr: u8,
    sample_rate: EdaSampleRate,
    /// Calibration offset per site (µS)
    calibration_offset: [f32; 4],
    /// Excitation voltage (V)
    excitation_voltage: f32,
    /// Series resistor value (ohms)
    series_resistance: f32,
}

impl<I2C, E> EdaDriver<I2C>
where
    I2C: I2c<Error = E>,
{
    /// Create a new EDA driver
    ///
    /// # Arguments
    ///
    /// * `i2c` - I2C bus
    /// * `addr` - ADS1115 I2C address (typically 0x48-0x4B)
    #[must_use]
    pub fn new(i2c: I2C, addr: u8) -> Self {
        Self {
            i2c,
            addr,
            sample_rate: EdaSampleRate::default(),
            calibration_offset: [0.0; 4],
            // Default excitation circuit: 0.5V through 10kΩ series resistor
            excitation_voltage: 0.5,
            series_resistance: 10_000.0,
        }
    }

    /// Set the sample rate
    pub fn set_sample_rate(&mut self, rate: EdaSampleRate) {
        self.sample_rate = rate;
    }

    /// Configure the excitation circuit parameters
    ///
    /// # Arguments
    ///
    /// * `voltage` - Excitation voltage in volts (typically 0.5V)
    /// * `resistance` - Series resistance in ohms (typically 10kΩ)
    pub fn set_excitation(&mut self, voltage: f32, resistance: f32) {
        self.excitation_voltage = voltage;
        self.series_resistance = resistance;
    }

    /// Initialize the EDA system
    pub fn init(&mut self) -> Result<(), EdaError<E>> {
        // Verify ADS1115 by reading config register
        let mut buf = [0u8; 2];
        self.i2c
            .write_read(self.addr, &[ads1115::CONFIG], &mut buf)
            .map_err(EdaError::I2c)?;

        // Configure for single-shot mode, ±4.096V range
        let base_config = config::PGA_4V | config::MODE_SINGLE | config::COMP_DISABLE;
        self.write_config(base_config | config::MUX_AIN0)?;

        Ok(())
    }

    /// Calibrate by measuring baseline at each site
    ///
    /// Should be called when electrodes are attached and subject is relaxed.
    pub fn calibrate(&mut self, samples: u8) -> Result<(), EdaError<E>> {
        let mut sums = [0.0f32; 4];

        for _ in 0..samples {
            for site_idx in 0u8..4 {
                let raw = self.read_channel(site_idx)?;
                let conductance = self.raw_to_conductance(raw);
                sums[site_idx as usize] += conductance;
            }
            cortex_m::asm::delay(100_000); // ~10ms between samples
        }

        // Store baseline as calibration offset
        for (i, sum) in sums.iter().enumerate() {
            self.calibration_offset[i] = sum / samples as f32;
        }

        Ok(())
    }

    /// Take a complete EDA sample (all 4 sites)
    pub fn sample(
        &mut self,
        timestamp_us: u64,
        sequence: u32,
    ) -> Result<EdaSample, EdaError<E>> {
        let mut sample = EdaSample::new(timestamp_us, sequence);

        for (idx, site) in EdaSite::ALL.iter().enumerate() {
            let raw = self.read_channel(idx as u8)?;
            let conductance = self.raw_to_conductance(raw);
            sample.set_site(*site, Fixed24_8::from_f32(conductance));
        }

        Ok(sample)
    }

    /// Get calibration offset for a site (baseline SCL)
    #[must_use]
    pub fn baseline(&self, site: EdaSite) -> f32 {
        self.calibration_offset[site.index()]
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn write_config(&mut self, config: u16) -> Result<(), EdaError<E>> {
        let bytes = [(config >> 8) as u8, config as u8];
        self.i2c
            .write(self.addr, &[ads1115::CONFIG, bytes[0], bytes[1]])
            .map_err(EdaError::I2c)
    }

    fn read_channel(&mut self, channel: u8) -> Result<u16, EdaError<E>> {
        // Select channel and start conversion
        let mux = match channel {
            0 => config::MUX_AIN0,
            1 => config::MUX_AIN1,
            2 => config::MUX_AIN2,
            _ => config::MUX_AIN3,
        };

        let config_val = config::OS_SINGLE
            | mux
            | config::PGA_4V
            | config::MODE_SINGLE
            | self.sample_rate.config_bits()
            | config::COMP_DISABLE;

        self.write_config(config_val)?;

        // Wait for conversion (based on sample rate)
        let delay_cycles = match self.sample_rate {
            EdaSampleRate::Sps8 => 2_000_000,
            EdaSampleRate::Sps16 => 1_000_000,
            EdaSampleRate::Sps32 => 500_000,
            EdaSampleRate::Sps64 => 250_000,
            EdaSampleRate::Sps128 => 125_000,
        };
        cortex_m::asm::delay(delay_cycles);

        // Read conversion result
        let mut buf = [0u8; 2];
        self.i2c
            .write_read(self.addr, &[ads1115::CONVERSION], &mut buf)
            .map_err(EdaError::I2c)?;

        Ok(((buf[0] as u16) << 8) | (buf[1] as u16))
    }

    /// Convert raw ADC value to skin conductance in µS
    fn raw_to_conductance(&self, raw: u16) -> f32 {
        // ADS1115 at ±4.096V: LSB = 4.096 / 32768 = 0.000125V
        let voltage = (raw as i16 as f32) * 0.000125;

        // Voltage divider: V_measured = V_excitation * R_skin / (R_series + R_skin)
        // Solving for R_skin: R_skin = V_measured * R_series / (V_excitation - V_measured)
        let v_diff = self.excitation_voltage - voltage;
        if v_diff.abs() < 0.001 {
            return 0.0; // Avoid division by zero
        }

        let resistance = voltage * self.series_resistance / v_diff;

        // Conductance = 1 / Resistance, convert to µS
        if resistance > 0.0 {
            1_000_000.0 / resistance
        } else {
            0.0
        }
    }
}

/// EDA signal processor for tonic/phasic decomposition
pub struct EdaProcessor {
    /// Lowpass filter coefficient for SCL extraction
    scl_alpha: f32,
    /// Current SCL estimate per site
    scl: [f32; 4],
    /// SCR detection threshold (µS)
    scr_threshold: f32,
    /// SCR peak tracker
    scr_peak: [f32; 4],
    /// SCR onset timestamp
    scr_onset_us: [u64; 4],
    /// Last sample timestamp
    last_timestamp_us: u64,
}

impl EdaProcessor {
    /// Create a new EDA processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            scl_alpha: 0.01, // Very slow lowpass for tonic level
            scl: [0.0; 4],
            scr_threshold: 0.02, // 0.02 µS threshold for SCR
            scr_peak: [0.0; 4],
            scr_onset_us: [0; 4],
            last_timestamp_us: 0,
        }
    }

    /// Set SCL lowpass filter coefficient
    ///
    /// Lower values = slower response (more averaging)
    /// Typical range: 0.001 - 0.05
    pub fn set_scl_filter(&mut self, alpha: f32) {
        self.scl_alpha = alpha.clamp(0.001, 0.1);
    }

    /// Set SCR detection threshold in µS
    pub fn set_scr_threshold(&mut self, threshold: f32) {
        self.scr_threshold = threshold;
    }

    /// Process a sample and extract SCL/SCR
    ///
    /// Returns (site, scl, scr, is_scr_peak)
    pub fn process(&mut self, sample: &EdaSample) -> [(f32, f32, bool); 4] {
        let mut results = [(0.0, 0.0, false); 4];

        for (i, site) in EdaSite::ALL.iter().enumerate() {
            let conductance = sample.site(*site).to_f32();

            // Update SCL (tonic level) with lowpass filter
            self.scl[i] = self.scl_alpha * conductance + (1.0 - self.scl_alpha) * self.scl[i];

            // SCR = Raw - SCL (phasic component)
            let scr = conductance - self.scl[i];

            // Detect SCR peaks
            let is_peak = if scr > self.scr_threshold {
                if scr > self.scr_peak[i] {
                    self.scr_peak[i] = scr;
                    if self.scr_onset_us[i] == 0 {
                        self.scr_onset_us[i] = sample.timestamp_us;
                    }
                    false // Still rising
                } else {
                    // Peak detected (scr decreased from max)
                    let is_first_peak = self.scr_peak[i] > self.scr_threshold;
                    self.scr_peak[i] = 0.0;
                    self.scr_onset_us[i] = 0;
                    is_first_peak
                }
            } else {
                self.scr_peak[i] = 0.0;
                self.scr_onset_us[i] = 0;
                false
            };

            results[i] = (self.scl[i], scr, is_peak);
        }

        self.last_timestamp_us = sample.timestamp_us;
        results
    }

    /// Get current SCL for all sites
    #[must_use]
    pub fn scl(&self) -> [f32; 4] {
        self.scl
    }

    /// Calculate overall arousal from SCL
    ///
    /// Returns value from 0.0 (low) to 1.0 (high)
    #[must_use]
    pub fn arousal(&self) -> f32 {
        let mean_scl: f32 = self.scl.iter().sum::<f32>() / 4.0;
        // Normalize: typical range 2-20 µS
        ((mean_scl - 2.0) / 18.0).clamp(0.0, 1.0)
    }
}

impl Default for EdaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// High-density EDA configuration for multiple ADS1115 ADCs
pub struct EdaArrayConfig {
    /// I2C addresses for each ADC
    pub adc_addresses: [u8; 4],
    /// Number of active sites
    pub active_sites: u8,
    /// Sample rate
    pub sample_rate: EdaSampleRate,
}

impl Default for EdaArrayConfig {
    fn default() -> Self {
        Self {
            adc_addresses: [0x48, 0x49, 0x4A, 0x4B],
            active_sites: 4,
            sample_rate: EdaSampleRate::Sps32,
        }
    }
}

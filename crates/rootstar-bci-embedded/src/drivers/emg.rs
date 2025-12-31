//! EMG (Electromyography) Driver
//!
//! Driver for facial EMG using the TI ADS1299. EMG signals are captured
//! from facial muscles to detect emotional responses and motor activity
//! during sensory experiences.
//!
//! # Muscle Targets
//!
//! - **Zygomaticus major**: Smile muscle (positive valence)
//! - **Corrugator supercilii**: Frown muscle (negative valence)
//! - **Masseter**: Jaw muscle (chewing, taste)
//! - **Orbicularis oris**: Lip muscle (taste, speech)
//!
//! # Configuration
//!
//! EMG requires different settings than EEG:
//! - Lower gain (4-12x vs 24x for EEG) due to higher amplitude
//! - Higher sample rate (500-1000 Hz) to capture fast muscle transients
//! - Bandpass filtering (20-500 Hz) in post-processing
//!
//! # Example
//!
//! ```ignore
//! let mut emg = EmgDriver::new(spi, cs, drdy, reset);
//! emg.init()?;
//! emg.start_acquisition()?;
//!
//! loop {
//!     if emg.data_ready() {
//!         let sample = emg.read_sample(timestamp, sequence)?;
//!         // Process EMG sample...
//!     }
//! }
//! ```

use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::SpiDevice;

use rootstar_bci_core::error::Ads1299Error;
use rootstar_bci_core::types::{EmgChannel, EmgSample, Fixed24_8};

/// ADS1299 register addresses (same as EEG driver)
mod regs {
    pub const ID: u8 = 0x00;
    pub const CONFIG1: u8 = 0x01;
    pub const CONFIG2: u8 = 0x02;
    pub const CONFIG3: u8 = 0x03;
    pub const CH1SET: u8 = 0x05;
    pub const BIAS_SENSP: u8 = 0x0D;
    pub const BIAS_SENSN: u8 = 0x0E;
}

/// ADS1299 commands
mod cmd {
    pub const WAKEUP: u8 = 0x02;
    pub const RESET: u8 = 0x06;
    pub const START: u8 = 0x08;
    pub const STOP: u8 = 0x0A;
    pub const RDATAC: u8 = 0x10;
    pub const SDATAC: u8 = 0x11;
    pub const RDATA: u8 = 0x12;
    pub const RREG: u8 = 0x20;
    pub const WREG: u8 = 0x40;
}

/// EMG sample rate configuration
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EmgSampleRate {
    /// 500 Hz (standard EMG)
    Sps500 = 0x05,
    /// 1000 Hz (high resolution)
    Sps1000 = 0x04,
    /// 2000 Hz (research grade)
    Sps2000 = 0x03,
}

impl EmgSampleRate {
    /// Get sample rate in Hz
    #[must_use]
    pub const fn hz(self) -> u16 {
        match self {
            Self::Sps500 => 500,
            Self::Sps1000 => 1000,
            Self::Sps2000 => 2000,
        }
    }
}

impl Default for EmgSampleRate {
    fn default() -> Self {
        Self::Sps500
    }
}

/// EMG gain setting (typically lower than EEG)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EmgGain {
    /// 4x gain (for strong muscle signals)
    X4 = 0x02,
    /// 6x gain
    X6 = 0x03,
    /// 8x gain (default for facial EMG)
    X8 = 0x04,
    /// 12x gain (for weak signals)
    X12 = 0x05,
}

impl EmgGain {
    /// Get the gain multiplier
    #[must_use]
    pub const fn multiplier(self) -> u8 {
        match self {
            Self::X4 => 4,
            Self::X6 => 6,
            Self::X8 => 8,
            Self::X12 => 12,
        }
    }
}

impl Default for EmgGain {
    fn default() -> Self {
        Self::X8
    }
}

/// EMG driver using ADS1299
pub struct EmgDriver<SPI, CS, DRDY, RST> {
    spi: SPI,
    cs: CS,
    drdy: DRDY,
    reset: RST,
    gain: EmgGain,
    sample_rate: EmgSampleRate,
}

impl<SPI, CS, DRDY, RST, E> EmgDriver<SPI, CS, DRDY, RST>
where
    SPI: SpiDevice<Error = E>,
    CS: OutputPin,
    DRDY: InputPin,
    RST: OutputPin,
{
    /// Create a new EMG driver
    #[must_use]
    pub fn new(spi: SPI, cs: CS, drdy: DRDY, reset: RST) -> Self {
        Self {
            spi,
            cs,
            drdy,
            reset,
            gain: EmgGain::default(),
            sample_rate: EmgSampleRate::default(),
        }
    }

    /// Set the gain for all channels
    pub fn set_gain(&mut self, gain: EmgGain) {
        self.gain = gain;
    }

    /// Set the sample rate
    pub fn set_sample_rate(&mut self, rate: EmgSampleRate) {
        self.sample_rate = rate;
    }

    /// Initialize the ADS1299 for EMG acquisition
    pub fn init(&mut self) -> Result<(), Ads1299Error<E>> {
        // Hardware reset
        self.reset.set_low().ok();
        cortex_m::asm::delay(100_000);
        self.reset.set_high().ok();
        cortex_m::asm::delay(1_000_000);

        // Stop continuous data mode
        self.send_command(cmd::SDATAC)?;
        cortex_m::asm::delay(10_000);

        // Verify device ID
        let id = self.read_register(regs::ID)?;
        if id & 0x1F != 0x1E {
            return Err(Ads1299Error::InvalidDeviceId {
                expected: 0x3E,
                got: id,
            });
        }

        // CONFIG1: Daisy-chain disabled, clock output disabled, sample rate
        self.write_register(regs::CONFIG1, 0x90 | (self.sample_rate as u8))?;

        // CONFIG2: Test signals disabled, internal reference
        self.write_register(regs::CONFIG2, 0xC0)?;

        // CONFIG3: Internal reference buffer enabled, bias measurement enabled
        self.write_register(regs::CONFIG3, 0xEC)?;

        // Configure all 8 channels for EMG with appropriate gain
        let ch_config = (self.gain as u8) << 4;
        for ch in 0..8 {
            self.write_register(regs::CH1SET + ch, ch_config)?;
        }

        // Enable bias for all channels (for common-mode rejection)
        self.write_register(regs::BIAS_SENSP, 0xFF)?;
        self.write_register(regs::BIAS_SENSN, 0xFF)?;

        Ok(())
    }

    /// Start continuous data acquisition
    pub fn start_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        self.send_command(cmd::RDATAC)?;
        cortex_m::asm::delay(10_000);
        self.send_command(cmd::START)?;
        Ok(())
    }

    /// Stop data acquisition
    pub fn stop_acquisition(&mut self) -> Result<(), Ads1299Error<E>> {
        self.send_command(cmd::STOP)?;
        self.send_command(cmd::SDATAC)?;
        Ok(())
    }

    /// Check if new data is ready
    #[must_use]
    pub fn data_ready(&mut self) -> bool {
        self.drdy.is_low().unwrap_or(false)
    }

    /// Read a complete EMG sample (all 8 channels)
    pub fn read_sample(
        &mut self,
        timestamp_us: u64,
        sequence: u32,
    ) -> Result<EmgSample, Ads1299Error<E>> {
        // Read 27 bytes: 3 status + 8×3 channel data
        let mut buffer = [0u8; 27];
        self.cs.set_low().ok();
        self.spi.read(&mut buffer).map_err(Ads1299Error::Spi)?;
        self.cs.set_high().ok();

        let mut sample = EmgSample::new(timestamp_us, sequence);

        // Parse 24-bit samples from each channel
        for ch in 0..8 {
            let offset = 3 + ch * 3;
            let raw = Self::parse_24bit(&buffer[offset..offset + 3]);
            let uv = Fixed24_8::from_ads1299_raw(raw, self.gain.multiplier());
            sample.set_channel(EmgChannel::from_index(ch).unwrap_or(EmgChannel::ZygomaticusL), uv);
        }

        Ok(sample)
    }

    /// Get current sample rate in Hz
    #[must_use]
    pub fn sample_rate_hz(&self) -> u16 {
        self.sample_rate.hz()
    }

    /// Get current gain setting
    #[must_use]
    pub fn gain(&self) -> EmgGain {
        self.gain
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn send_command(&mut self, cmd: u8) -> Result<(), Ads1299Error<E>> {
        self.cs.set_low().ok();
        self.spi.write(&[cmd]).map_err(Ads1299Error::Spi)?;
        self.cs.set_high().ok();
        Ok(())
    }

    fn read_register(&mut self, reg: u8) -> Result<u8, Ads1299Error<E>> {
        let cmd = [cmd::RREG | reg, 0x00];
        let mut response = [0u8; 1];
        self.cs.set_low().ok();
        self.spi.write(&cmd).map_err(Ads1299Error::Spi)?;
        self.spi.read(&mut response).map_err(Ads1299Error::Spi)?;
        self.cs.set_high().ok();
        Ok(response[0])
    }

    fn write_register(&mut self, reg: u8, value: u8) -> Result<(), Ads1299Error<E>> {
        let cmd = [cmd::WREG | reg, 0x00, value];
        self.cs.set_low().ok();
        self.spi.write(&cmd).map_err(Ads1299Error::Spi)?;
        self.cs.set_high().ok();
        Ok(())
    }

    fn parse_24bit(bytes: &[u8]) -> i32 {
        let raw = ((bytes[0] as i32) << 16) | ((bytes[1] as i32) << 8) | (bytes[2] as i32);
        // Sign-extend from 24-bit to 32-bit
        if raw & 0x800000 != 0 {
            raw | !0xFFFFFF
        } else {
            raw
        }
    }
}

/// Processed EMG features for a single channel
#[derive(Copy, Clone, Debug)]
pub struct EmgFeatures {
    /// RMS amplitude in µV
    pub rms_amplitude: f32,
    /// Mean frequency in Hz (from power spectrum)
    pub mean_frequency: f32,
    /// Median frequency in Hz
    pub median_frequency: f32,
    /// Integrated EMG (area under rectified signal)
    pub iemg: f32,
    /// Zero-crossing rate (activity indicator)
    pub zero_crossings: u32,
}

impl EmgFeatures {
    /// Create empty features
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            rms_amplitude: 0.0,
            mean_frequency: 0.0,
            median_frequency: 0.0,
            iemg: 0.0,
            zero_crossings: 0,
        }
    }

    /// Calculate features from a buffer of samples
    ///
    /// # Arguments
    ///
    /// * `samples` - Buffer of EMG samples (µV)
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_buffer(samples: &[f32], sample_rate: f32) -> Self {
        if samples.is_empty() {
            return Self::zero();
        }

        let n = samples.len() as f32;

        // RMS amplitude
        let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
        let rms_amplitude = libm::sqrtf(sum_sq / n);

        // Integrated EMG (sum of absolute values)
        let iemg: f32 = samples.iter().map(|&x| libm::fabsf(x)).sum();

        // Zero crossings
        let mut zero_crossings = 0u32;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        // Estimate mean frequency from zero-crossing rate
        // Mean frequency ≈ ZCR × sample_rate / (2 × N)
        let duration_s = n / sample_rate;
        let mean_frequency = (zero_crossings as f32) / (2.0 * duration_s);

        Self {
            rms_amplitude,
            mean_frequency,
            median_frequency: mean_frequency, // Simplified; full FFT needed for accurate value
            iemg,
            zero_crossings,
        }
    }
}

/// EMG-based emotional valence calculation
pub struct EmgValenceCalculator {
    /// Baseline RMS for each channel
    baseline: [f32; 8],
    /// Smoothing factor (0-1)
    alpha: f32,
    /// Current smoothed values
    smoothed: [f32; 8],
}

impl EmgValenceCalculator {
    /// Create a new valence calculator
    #[must_use]
    pub fn new() -> Self {
        Self {
            baseline: [0.0; 8],
            alpha: 0.1,
            smoothed: [0.0; 8],
        }
    }

    /// Set baseline from a calibration period
    pub fn set_baseline(&mut self, baseline: [f32; 8]) {
        self.baseline = baseline;
        self.smoothed = baseline;
    }

    /// Update with new sample and return valence/arousal
    ///
    /// Returns (valence, arousal) where:
    /// - valence: -1.0 (negative) to 1.0 (positive)
    /// - arousal: 0.0 (calm) to 1.0 (excited)
    pub fn update(&mut self, sample: &EmgSample) -> (f32, f32) {
        // Update smoothed values
        for (i, ch) in EmgChannel::ALL.iter().enumerate() {
            let raw = sample.channel(*ch).to_f32();
            self.smoothed[i] = self.alpha * raw + (1.0 - self.alpha) * self.smoothed[i];
        }

        // Calculate valence from smile vs frown ratio
        // Zygomaticus (smile) = channels 0, 1
        // Corrugator (frown) = channels 2, 3
        let smile_activation = (self.smoothed[0].abs() + self.smoothed[1].abs()) / 2.0;
        let frown_activation = (self.smoothed[2].abs() + self.smoothed[3].abs()) / 2.0;

        let smile_baseline = (self.baseline[0] + self.baseline[1]) / 2.0;
        let frown_baseline = (self.baseline[2] + self.baseline[3]) / 2.0;

        let smile_delta = smile_activation - smile_baseline;
        let frown_delta = frown_activation - frown_baseline;

        // Valence: positive for smile, negative for frown
        let valence_raw = smile_delta - frown_delta;
        let valence = (valence_raw / 50.0).clamp(-1.0, 1.0); // Scale to [-1, 1]

        // Arousal: overall facial muscle activation
        let total_activation: f32 = self.smoothed.iter().map(|&x| x.abs()).sum();
        let baseline_total: f32 = self.baseline.iter().sum();
        let arousal_raw = (total_activation - baseline_total) / baseline_total.max(1.0);
        let arousal = arousal_raw.clamp(0.0, 1.0);

        (valence, arousal)
    }
}

impl Default for EmgValenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

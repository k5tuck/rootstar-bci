//! fNIRS Optical Frontend Driver
//!
//! Driver for the fNIRS optical frontend consisting of:
//! - NIR LEDs at 760nm and 850nm (PWM controlled)
//! - OPT101 photodiode detectors
//! - ADS1115 16-bit ADC (I2C)
//!
//! Uses time-division multiplexing to sample both wavelengths sequentially.

use embedded_hal::i2c::I2c;
use embedded_hal::pwm::SetDutyCycle;

use rootstar_bci_core::error::FnirsError;
use rootstar_bci_core::types::{FnirsChannel, FnirsSample, Wavelength};

/// ADS1115 register addresses
mod ads1115_regs {
    pub const CONVERSION: u8 = 0x00;
    pub const CONFIG: u8 = 0x01;
}

/// fNIRS optical frontend driver
pub struct FnirsDriver<I2C, PWM760, PWM850> {
    adc: I2C,
    led_760: PWM760,
    led_850: PWM850,
    adc_addr: u8,
    current_channel: FnirsChannel,
    baseline_760: u16,
    baseline_850: u16,
}

impl<I2C, PWM760, PWM850, E> FnirsDriver<I2C, PWM760, PWM850>
where
    I2C: I2c<Error = E>,
    PWM760: SetDutyCycle,
    PWM850: SetDutyCycle,
{
    /// Create a new fNIRS driver
    #[must_use]
    pub fn new(adc: I2C, led_760: PWM760, led_850: PWM850) -> Self {
        Self {
            adc,
            led_760,
            led_850,
            adc_addr: 0x48, // Default ADS1115 address
            current_channel: FnirsChannel::new(0, 0, 30),
            baseline_760: 0,
            baseline_850: 0,
        }
    }

    /// Set the I2C address for the ADS1115
    pub fn set_adc_address(&mut self, addr: u8) {
        self.adc_addr = addr;
    }

    /// Set the current channel being measured
    pub fn set_channel(&mut self, channel: FnirsChannel) {
        self.current_channel = channel;
    }

    /// Initialize the fNIRS system
    pub fn init(&mut self) -> Result<(), FnirsError<E>> {
        // Turn off LEDs initially
        self.set_led_intensity(Wavelength::Nm760, 0)?;
        self.set_led_intensity(Wavelength::Nm850, 0)?;

        // Configure ADS1115 for single-shot, 860 SPS, ±4.096V range
        let config: u16 = 0x8583;
        let config_bytes = [(config >> 8) as u8, config as u8];
        self.adc.write(self.adc_addr, &[ads1115_regs::CONFIG, config_bytes[0], config_bytes[1]])
            .map_err(FnirsError::I2c)?;

        Ok(())
    }

    /// Calibrate by measuring baseline intensities
    pub fn calibrate(&mut self) -> Result<(), FnirsError<E>> {
        // Measure with LEDs off (ambient)
        self.set_led_intensity(Wavelength::Nm760, 0)?;
        self.set_led_intensity(Wavelength::Nm850, 0)?;
        cortex_m::asm::delay(10_000_000);

        let ambient = self.read_adc_average(16)?;

        // Measure 760nm baseline
        self.set_led_intensity(Wavelength::Nm760, 128)?;
        cortex_m::asm::delay(1_000_000);
        let raw_760 = self.read_adc_average(16)?;
        self.baseline_760 = raw_760.saturating_sub(ambient);

        // Measure 850nm baseline
        self.set_led_intensity(Wavelength::Nm760, 0)?;
        self.set_led_intensity(Wavelength::Nm850, 128)?;
        cortex_m::asm::delay(1_000_000);
        let raw_850 = self.read_adc_average(16)?;
        self.baseline_850 = raw_850.saturating_sub(ambient);

        self.set_led_intensity(Wavelength::Nm850, 0)?;

        // Verify baselines are valid
        if self.baseline_760 < 100 || self.baseline_850 < 100 {
            return Err(FnirsError::CalibrationFailed {
                intensity: self.baseline_760.min(self.baseline_850),
                minimum: 100,
            });
        }

        Ok(())
    }

    /// Get baseline intensity for a wavelength
    #[must_use]
    pub fn baseline(&self, wavelength: Wavelength) -> u16 {
        match wavelength {
            Wavelength::Nm760 => self.baseline_760,
            Wavelength::Nm850 => self.baseline_850,
        }
    }

    /// Take a complete fNIRS sample (both wavelengths)
    ///
    /// Uses time-division multiplexing: LED760 → read → LED850 → read
    pub fn sample(&mut self, timestamp_us: u64, sequence: u32) -> Result<FnirsSample, FnirsError<E>> {
        // Sample at 760nm
        self.set_led_intensity(Wavelength::Nm760, 200)?;
        cortex_m::asm::delay(50_000); // 500µs LED stabilization
        let intensity_760 = self.read_adc()?;
        self.set_led_intensity(Wavelength::Nm760, 0)?;

        // Brief dark period
        cortex_m::asm::delay(10_000);

        // Sample at 850nm
        self.set_led_intensity(Wavelength::Nm850, 200)?;
        cortex_m::asm::delay(50_000);
        let intensity_850 = self.read_adc()?;
        self.set_led_intensity(Wavelength::Nm850, 0)?;

        Ok(FnirsSample {
            timestamp_us,
            channel: self.current_channel,
            intensity_760,
            intensity_850,
            sequence,
        })
    }

    fn set_led_intensity(&mut self, wavelength: Wavelength, duty: u8) -> Result<(), FnirsError<E>> {
        let duty_percent = (u16::from(duty) * 100 / 255) as u8;
        match wavelength {
            Wavelength::Nm760 => {
                self.led_760.set_duty_cycle_percent(duty_percent).map_err(|_| FnirsError::Pwm)?;
            }
            Wavelength::Nm850 => {
                self.led_850.set_duty_cycle_percent(duty_percent).map_err(|_| FnirsError::Pwm)?;
            }
        }
        Ok(())
    }

    fn read_adc(&mut self) -> Result<u16, FnirsError<E>> {
        // Start single-shot conversion
        let config: u16 = 0xC583;
        let config_bytes = [(config >> 8) as u8, config as u8];
        self.adc.write(self.adc_addr, &[ads1115_regs::CONFIG, config_bytes[0], config_bytes[1]])
            .map_err(FnirsError::I2c)?;

        // Wait for conversion (~1.2ms for 860 SPS)
        cortex_m::asm::delay(150_000);

        // Read result
        let mut buffer = [0u8; 2];
        self.adc.write_read(self.adc_addr, &[ads1115_regs::CONVERSION], &mut buffer)
            .map_err(FnirsError::I2c)?;

        Ok(u16::from_be_bytes(buffer))
    }

    fn read_adc_average(&mut self, count: u8) -> Result<u16, FnirsError<E>> {
        let mut sum: u32 = 0;
        for _ in 0..count {
            sum += u32::from(self.read_adc()?);
        }
        Ok((sum / u32::from(count)) as u16)
    }
}

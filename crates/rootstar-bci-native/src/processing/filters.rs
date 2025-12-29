//! Digital filters for EEG processing
//!
//! Provides floating-point IIR filters optimized for host processing.

use rootstar_bci_core::types::EegBand;

/// Butterworth IIR filter coefficients (second-order section)
#[derive(Clone, Debug)]
pub struct BiquadCoeffs {
    /// Numerator coefficients [b0, b1, b2]
    pub b: [f64; 3],
    /// Denominator coefficients [a0=1, a1, a2]
    pub a: [f64; 3],
}

/// Second-order biquad filter section
#[derive(Clone, Debug)]
pub struct Biquad {
    coeffs: BiquadCoeffs,
    /// State: [z1, z2]
    state: [f64; 2],
}

impl Biquad {
    /// Create a new biquad section with given coefficients
    #[must_use]
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self { coeffs, state: [0.0, 0.0] }
    }

    /// Create a second-order Butterworth lowpass filter
    #[must_use]
    pub fn lowpass(sample_rate: f64, cutoff: f64) -> Self {
        let omega = std::f64::consts::PI * cutoff / sample_rate;
        let k = omega.tan();
        let k2 = k * k;
        let sqrt2 = std::f64::consts::SQRT_2;

        let norm = 1.0 / (1.0 + sqrt2 * k + k2);

        let coeffs = BiquadCoeffs {
            b: [k2 * norm, 2.0 * k2 * norm, k2 * norm],
            a: [1.0, 2.0 * (k2 - 1.0) * norm, (1.0 - sqrt2 * k + k2) * norm],
        };

        Self::new(coeffs)
    }

    /// Create a second-order Butterworth highpass filter
    #[must_use]
    pub fn highpass(sample_rate: f64, cutoff: f64) -> Self {
        let omega = std::f64::consts::PI * cutoff / sample_rate;
        let k = omega.tan();
        let k2 = k * k;
        let sqrt2 = std::f64::consts::SQRT_2;

        let norm = 1.0 / (1.0 + sqrt2 * k + k2);

        let coeffs = BiquadCoeffs {
            b: [norm, -2.0 * norm, norm],
            a: [1.0, 2.0 * (k2 - 1.0) * norm, (1.0 - sqrt2 * k + k2) * norm],
        };

        Self::new(coeffs)
    }

    /// Create a notch filter for power line interference
    #[must_use]
    pub fn notch(sample_rate: f64, notch_freq: f64, q: f64) -> Self {
        let omega = 2.0 * std::f64::consts::PI * notch_freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q);

        let norm = 1.0 / (1.0 + alpha);

        let coeffs = BiquadCoeffs {
            b: [norm, -2.0 * cos_omega * norm, norm],
            a: [1.0, -2.0 * cos_omega * norm, (1.0 - alpha) * norm],
        };

        Self::new(coeffs)
    }

    /// Process a single sample
    pub fn filter(&mut self, input: f64) -> f64 {
        let output = self.coeffs.b[0] * input
            + self.coeffs.b[1] * self.state[0]
            + self.coeffs.b[2] * self.state[1]
            - self.coeffs.a[1] * self.state[0]
            - self.coeffs.a[2] * self.state[1];

        self.state[1] = self.state[0];
        self.state[0] = output;

        output
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = [0.0, 0.0];
    }
}

/// Bandpass filter for EEG frequency bands
#[derive(Clone, Debug)]
pub struct BandpassFilter {
    lowpass: Biquad,
    highpass: Biquad,
}

impl BandpassFilter {
    /// Create a bandpass filter for a frequency range
    #[must_use]
    pub fn new(sample_rate: f64, low_cutoff: f64, high_cutoff: f64) -> Self {
        Self {
            lowpass: Biquad::lowpass(sample_rate, high_cutoff),
            highpass: Biquad::highpass(sample_rate, low_cutoff),
        }
    }

    /// Create a bandpass filter for a standard EEG band
    #[must_use]
    pub fn for_band(sample_rate: f64, band: EegBand) -> Self {
        let (low, high) = band.range_hz();
        Self::new(sample_rate, f64::from(low), f64::from(high))
    }

    /// Process a single sample
    pub fn filter(&mut self, input: f64) -> f64 {
        let hp_out = self.highpass.filter(input);
        self.lowpass.filter(hp_out)
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.lowpass.reset();
        self.highpass.reset();
    }
}

/// Multi-channel EEG filter bank
pub struct FilterBank {
    /// Highpass filter for DC removal (0.5 Hz)
    dc_removal: [Biquad; 8],
    /// Notch filters for power line (50/60 Hz)
    notch: [Biquad; 8],
}

impl FilterBank {
    /// Create a new filter bank
    #[must_use]
    pub fn new(sample_rate: f64, notch_freq: f64) -> Self {
        Self {
            dc_removal: core::array::from_fn(|_| Biquad::highpass(sample_rate, 0.5)),
            notch: core::array::from_fn(|_| Biquad::notch(sample_rate, notch_freq, 30.0)),
        }
    }

    /// Process all 8 channels
    pub fn filter(&mut self, channels: &mut [f64; 8]) {
        for (i, ch) in channels.iter_mut().enumerate() {
            *ch = self.dc_removal[i].filter(*ch);
            *ch = self.notch[i].filter(*ch);
        }
    }

    /// Reset all filters
    pub fn reset(&mut self) {
        for f in &mut self.dc_removal {
            f.reset();
        }
        for f in &mut self.notch {
            f.reset();
        }
    }
}

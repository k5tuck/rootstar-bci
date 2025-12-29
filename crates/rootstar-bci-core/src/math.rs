//! Math utilities for BCI signal processing (`no_std` compatible)
//!
//! This module provides:
//! - Fixed-point digital filters (IIR/FIR)
//! - Unit conversions for ADC data
//! - Statistical functions
//! - Modified Beer-Lambert Law computations (fixed-point)

use serde::{Deserialize, Serialize};

use crate::types::Fixed24_8;

// ============================================================================
// Physical Constants (Q24.8 fixed-point where applicable)
// ============================================================================

/// Physical constants for BCI calculations
pub mod constants {
    #[allow(unused_imports)]
    use super::Fixed24_8;

    /// ADS1299 LSB in µV at Gain=1 (4.5V / 2^23 * 1e6 ≈ 0.536 µV)
    pub const ADS1299_LSB_UV_GAIN1: f32 = 0.536;

    /// ADS1299 LSB in Q24.8 at Gain=1 (0.536 * 256 ≈ 137)
    pub const ADS1299_LSB_UV_GAIN1_Q8: i32 = 137;

    /// Default Differential Pathlength Factor for 760nm (adult head)
    pub const DPF_760NM_ADULT: f32 = 5.13;

    /// Default Differential Pathlength Factor for 850nm (adult head)
    pub const DPF_850NM_ADULT: f32 = 4.99;

    /// Hemodynamic delay (neural activity to blood flow response) in ms
    pub const HEMODYNAMIC_DELAY_MS: u32 = 5000;

    /// Extinction coefficient for HbO₂ at 760nm (cm⁻¹·M⁻¹)
    pub const E_HBO2_760: f64 = 1486.0;

    /// Extinction coefficient for HbR at 760nm (cm⁻¹·M⁻¹)
    pub const E_HBR_760: f64 = 3843.0;

    /// Extinction coefficient for HbO₂ at 850nm (cm⁻¹·M⁻¹)
    pub const E_HBO2_850: f64 = 2526.0;

    /// Extinction coefficient for HbR at 850nm (cm⁻¹·M⁻¹)
    pub const E_HBR_850: f64 = 1798.0;

    /// Maximum valid EEG amplitude (saturation threshold) in µV
    pub const EEG_SATURATION_THRESHOLD_UV: f32 = 3200.0;

    /// Minimum valid fNIRS intensity (dark threshold)
    pub const FNIRS_DARK_THRESHOLD: u16 = 100;

    /// Maximum valid fNIRS intensity (saturation threshold)
    pub const FNIRS_SATURATION_THRESHOLD: u16 = 65000;
}

// ============================================================================
// First-Order IIR Filter (Fixed-Point)
// ============================================================================

/// First-order IIR low-pass filter using fixed-point arithmetic.
///
/// Implements: y[n] = α * x[n] + (1-α) * y[n-1]
///
/// Suitable for smoothing hemodynamic signals in `no_std` environments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IirFilter {
    /// Filter coefficient α in Q24.8 (0 < α < 1)
    alpha_q8: i32,
    /// Previous output in Q24.8
    state_q8: i32,
}

impl IirFilter {
    /// Create a new filter with given time constant.
    ///
    /// # Arguments
    ///
    /// * `sample_rate_hz` - Sampling frequency in Hz
    /// * `time_constant_s` - Filter time constant in seconds (τ)
    ///
    /// The cutoff frequency fc ≈ 1/(2πτ)
    #[must_use]
    pub fn new(sample_rate_hz: f32, time_constant_s: f32) -> Self {
        // α = dt / (τ + dt), where dt = 1/fs
        let dt = 1.0 / sample_rate_hz;
        let alpha = dt / (time_constant_s + dt);

        // Convert to Q24.8
        let alpha_q8 = (alpha * 256.0) as i32;

        Self {
            alpha_q8: alpha_q8.clamp(1, 255),
            state_q8: 0,
        }
    }

    /// Create a filter with direct alpha coefficient (Q24.8).
    ///
    /// Useful for embedded systems where the coefficient is pre-computed.
    #[must_use]
    pub const fn from_alpha_q8(alpha_q8: i32) -> Self {
        Self {
            alpha_q8,
            state_q8: 0,
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn filter(&mut self, input: Fixed24_8) -> Fixed24_8 {
        // y = α*x + (1-α)*y_prev
        // In Q24.8: y_q8 = (α_q8 * x_q8 + (256-α_q8) * y_prev_q8) >> 8
        let x_q8 = input.to_raw() as i64;
        let y_prev = self.state_q8 as i64;
        let alpha = self.alpha_q8 as i64;

        let y_new = ((alpha * x_q8) + ((256 - alpha) * y_prev)) >> 8;
        self.state_q8 = y_new as i32;

        Fixed24_8::from_raw(self.state_q8)
    }

    /// Reset filter state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.state_q8 = 0;
    }

    /// Set filter state to a specific value.
    #[inline]
    pub fn set_state(&mut self, value: Fixed24_8) {
        self.state_q8 = value.to_raw();
    }

    /// Get current filter state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> Fixed24_8 {
        Fixed24_8::from_raw(self.state_q8)
    }
}

// ============================================================================
// Second-Order IIR Filter (Biquad, Fixed-Point)
// ============================================================================

/// Second-order IIR (biquad) filter using fixed-point arithmetic.
///
/// Direct Form I implementation:
/// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
///
/// Coefficients are in Q8.24 format for higher precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiquadFilter {
    // Coefficients (Q8.24 fixed-point)
    b0: i32,
    b1: i32,
    b2: i32,
    a1: i32,
    a2: i32,

    // State variables (Q24.8)
    x1: i32,
    x2: i32,
    y1: i32,
    y2: i32,
}

impl BiquadFilter {
    /// Create a biquad filter with Q8.24 coefficients.
    ///
    /// Coefficients should be computed externally (e.g., with scipy.signal)
    /// and converted to Q8.24: coef_q24 = (int)(coef_f32 * 16777216)
    #[must_use]
    pub const fn new(b0: i32, b1: i32, b2: i32, a1: i32, a2: i32) -> Self {
        Self {
            b0, b1, b2, a1, a2,
            x1: 0, x2: 0, y1: 0, y2: 0,
        }
    }

    /// Create a simple second-order Butterworth low-pass filter.
    ///
    /// Approximation suitable for embedded use.
    ///
    /// # Arguments
    ///
    /// * `sample_rate_hz` - Sampling frequency
    /// * `cutoff_hz` - Cutoff frequency
    #[must_use]
    pub fn lowpass_butterworth(sample_rate_hz: f32, cutoff_hz: f32) -> Self {
        // Pre-warp the cutoff frequency
        let omega = core::f32::consts::PI * cutoff_hz / sample_rate_hz;
        let omega_tan = fast_tan(omega);
        let k = omega_tan;
        let k2 = k * k;
        let sqrt2 = 1.41421356;

        // Butterworth coefficients
        let norm = 1.0 / (1.0 + sqrt2 * k + k2);

        let b0 = k2 * norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - sqrt2 * k + k2) * norm;

        // Convert to Q8.24
        const SCALE: f32 = 16777216.0; // 2^24
        Self::new(
            (b0 * SCALE) as i32,
            (b1 * SCALE) as i32,
            (b2 * SCALE) as i32,
            (a1 * SCALE) as i32,
            (a2 * SCALE) as i32,
        )
    }

    /// Create a second-order Butterworth high-pass filter.
    #[must_use]
    pub fn highpass_butterworth(sample_rate_hz: f32, cutoff_hz: f32) -> Self {
        let omega = core::f32::consts::PI * cutoff_hz / sample_rate_hz;
        let omega_tan = fast_tan(omega);
        let k = omega_tan;
        let k2 = k * k;
        let sqrt2 = 1.41421356;

        let norm = 1.0 / (1.0 + sqrt2 * k + k2);

        let b0 = norm;
        let b1 = -2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - sqrt2 * k + k2) * norm;

        const SCALE: f32 = 16777216.0;
        Self::new(
            (b0 * SCALE) as i32,
            (b1 * SCALE) as i32,
            (b2 * SCALE) as i32,
            (a1 * SCALE) as i32,
            (a2 * SCALE) as i32,
        )
    }

    /// Create a notch filter for power line interference (50/60 Hz).
    #[must_use]
    pub fn notch(sample_rate_hz: f32, notch_hz: f32, q_factor: f32) -> Self {
        let omega = 2.0 * core::f32::consts::PI * notch_hz / sample_rate_hz;
        let cos_omega = fast_cos(omega);
        let sin_omega = fast_sin(omega);
        let alpha = sin_omega / (2.0 * q_factor);

        let norm = 1.0 / (1.0 + alpha);

        let b0 = norm;
        let b1 = -2.0 * cos_omega * norm;
        let b2 = norm;
        let a1 = -2.0 * cos_omega * norm;
        let a2 = (1.0 - alpha) * norm;

        const SCALE: f32 = 16777216.0;
        Self::new(
            (b0 * SCALE) as i32,
            (b1 * SCALE) as i32,
            (b2 * SCALE) as i32,
            (a1 * SCALE) as i32,
            (a2 * SCALE) as i32,
        )
    }

    /// Process a single sample.
    #[inline]
    pub fn filter(&mut self, input: Fixed24_8) -> Fixed24_8 {
        let x0 = input.to_raw() as i64;

        // y = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        // Coefficients are Q8.24, inputs are Q24.8
        // Result is Q32.32, shift right by 24 to get Q24.8
        let y = (self.b0 as i64 * x0
            + self.b1 as i64 * self.x1 as i64
            + self.b2 as i64 * self.x2 as i64
            - self.a1 as i64 * self.y1 as i64
            - self.a2 as i64 * self.y2 as i64) >> 24;

        // Update state
        self.x2 = self.x1;
        self.x1 = x0 as i32;
        self.y2 = self.y1;
        self.y1 = y as i32;

        Fixed24_8::from_raw(y as i32)
    }

    /// Reset filter state to zero.
    pub fn reset(&mut self) {
        self.x1 = 0;
        self.x2 = 0;
        self.y1 = 0;
        self.y2 = 0;
    }
}

// ============================================================================
// Moving Average Filter
// ============================================================================

/// Simple moving average filter with fixed window size.
///
/// Uses a circular buffer for O(1) updates.
#[derive(Clone, Debug)]
pub struct MovingAverage<const N: usize> {
    buffer: [i32; N],
    index: usize,
    sum: i64,
    filled: bool,
}

impl<const N: usize> MovingAverage<N> {
    /// Create a new moving average filter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: [0; N],
            index: 0,
            sum: 0,
            filled: false,
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn filter(&mut self, input: Fixed24_8) -> Fixed24_8 {
        let x = input.to_raw();

        // Remove oldest value from sum
        self.sum -= self.buffer[self.index] as i64;

        // Add new value
        self.buffer[self.index] = x;
        self.sum += x as i64;

        // Update index
        self.index = (self.index + 1) % N;
        if self.index == 0 {
            self.filled = true;
        }

        // Compute average
        let count = if self.filled { N } else { self.index.max(1) };
        Fixed24_8::from_raw((self.sum / count as i64) as i32)
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.buffer = [0; N];
        self.index = 0;
        self.sum = 0;
        self.filled = false;
    }
}

impl<const N: usize> Default for MovingAverage<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Modified Beer-Lambert Law (Fixed-Point)
// ============================================================================

/// Modified Beer-Lambert Law solver for fNIRS (fixed-point version).
///
/// Solves for hemoglobin concentration changes from optical density changes
/// at two wavelengths.
#[derive(Clone, Debug)]
pub struct BeerLambertSolver {
    /// Precomputed matrix elements for 2x2 inverse (Q16.16)
    inv_a11_q16: i32,
    inv_a12_q16: i32,
    inv_a21_q16: i32,
    inv_a22_q16: i32,
}

impl BeerLambertSolver {
    /// Create a new solver for given optical geometry.
    ///
    /// # Arguments
    ///
    /// * `distance_cm` - Source-detector distance in cm
    /// * `dpf_760` - Differential pathlength factor at 760nm
    /// * `dpf_850` - Differential pathlength factor at 850nm
    #[must_use]
    pub fn new(distance_cm: f32, dpf_760: f32, dpf_850: f32) -> Self {
        // Path lengths
        let path_760 = distance_cm * dpf_760;
        let path_850 = distance_cm * dpf_850;

        // Matrix elements: A = [ε_HbO2_λ * L_λ, ε_HbR_λ * L_λ] for each wavelength
        let a11 = constants::E_HBO2_760 as f32 * path_760;
        let a12 = constants::E_HBR_760 as f32 * path_760;
        let a21 = constants::E_HBO2_850 as f32 * path_850;
        let a22 = constants::E_HBR_850 as f32 * path_850;

        // Determinant
        let det = a11 * a22 - a12 * a21;

        // Inverse matrix elements (scaled by 1e6 to convert M to µM)
        let scale = 1e6 / det;
        let inv_a11 = a22 * scale;
        let inv_a12 = -a12 * scale;
        let inv_a21 = -a21 * scale;
        let inv_a22 = a11 * scale;

        // Convert to Q16.16
        const Q16_SCALE: f32 = 65536.0;
        Self {
            inv_a11_q16: (inv_a11 * Q16_SCALE) as i32,
            inv_a12_q16: (inv_a12 * Q16_SCALE) as i32,
            inv_a21_q16: (inv_a21 * Q16_SCALE) as i32,
            inv_a22_q16: (inv_a22 * Q16_SCALE) as i32,
        }
    }

    /// Create solver with default adult DPF values.
    #[must_use]
    pub fn with_adult_dpf(distance_cm: f32) -> Self {
        Self::new(distance_cm, constants::DPF_760NM_ADULT, constants::DPF_850NM_ADULT)
    }

    /// Compute hemoglobin concentration changes from optical density changes.
    ///
    /// # Arguments
    ///
    /// * `delta_od_760` - Change in optical density at 760nm (Q24.8)
    /// * `delta_od_850` - Change in optical density at 850nm (Q24.8)
    ///
    /// # Returns
    ///
    /// (Δ[HbO₂], Δ[HbR]) in µM as Q24.8 fixed-point
    #[inline]
    pub fn solve(&self, delta_od_760: Fixed24_8, delta_od_850: Fixed24_8) -> (Fixed24_8, Fixed24_8) {
        let od_760 = delta_od_760.to_raw() as i64;
        let od_850 = delta_od_850.to_raw() as i64;

        // [Δ[HbO₂]]   [inv_a11  inv_a12] [ΔOD_760]
        // [Δ[HbR] ] = [inv_a21  inv_a22] [ΔOD_850]

        // Result is Q24.8 * Q16.16 >> 16 = Q24.8
        let delta_hbo2 = ((self.inv_a11_q16 as i64 * od_760
            + self.inv_a12_q16 as i64 * od_850) >> 16) as i32;
        let delta_hbr = ((self.inv_a21_q16 as i64 * od_760
            + self.inv_a22_q16 as i64 * od_850) >> 16) as i32;

        (Fixed24_8::from_raw(delta_hbo2), Fixed24_8::from_raw(delta_hbr))
    }
}

// ============================================================================
// Optical Density Calculation
// ============================================================================

/// Calculate change in optical density from intensity measurements.
///
/// ΔOD = -log₁₀(I / I₀)
///
/// Uses a fixed-point approximation of log₁₀.
///
/// # Arguments
///
/// * `intensity` - Current intensity (ADC counts)
/// * `baseline` - Baseline intensity (ADC counts)
///
/// # Returns
///
/// ΔOD in Q24.8 fixed-point
#[must_use]
pub fn compute_delta_od(intensity: u16, baseline: u16) -> Fixed24_8 {
    if baseline == 0 || intensity == 0 {
        return Fixed24_8::ZERO;
    }

    // ΔOD = -log₁₀(I/I₀) = log₁₀(I₀) - log₁₀(I)
    // Using natural log approximation: log₁₀(x) = ln(x) / ln(10) ≈ ln(x) * 0.4343

    let ratio = (baseline as f32) / (intensity as f32);
    let delta_od = fast_log(ratio) * 0.4343;

    Fixed24_8::from_f32(delta_od)
}

// ============================================================================
// Fast Math Functions (no libm dependency for core operations)
// ============================================================================

/// Fast approximation of natural logarithm.
///
/// Uses a polynomial approximation accurate to ~0.1% for 0.5 < x < 2.
/// For other ranges, uses the identity ln(x) = ln(x/2^n) + n*ln(2).
#[inline]
#[must_use]
pub fn fast_log(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }

    // Extract mantissa and exponent
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007FFFFF) | 0x3F800000);

    // Polynomial approximation for ln(mantissa) where 1 <= mantissa < 2
    // ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
    // Using a simpler rational approximation:
    let m = mantissa - 1.0;
    let ln_mantissa = m * (2.0 + m * 0.333333) / (2.0 + m * 1.333333);

    // Combine: ln(x) = ln(mantissa) + exponent * ln(2)
    ln_mantissa + (exponent as f32) * 0.693147
}

/// Fast approximation of tangent.
#[inline]
#[must_use]
pub fn fast_tan(x: f32) -> f32 {
    fast_sin(x) / fast_cos(x)
}

/// Fast approximation of sine using Bhaskara I's approximation.
#[inline]
#[must_use]
pub fn fast_sin(mut x: f32) -> f32 {
    const PI: f32 = core::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;

    // Normalize to [-π, π]
    x = x % TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }

    // Bhaskara I approximation
    let x2 = x * x;
    16.0 * x * (PI - x.abs()) / (5.0 * PI * PI - 4.0 * x2)
}

/// Fast approximation of cosine.
#[inline]
#[must_use]
pub fn fast_cos(x: f32) -> f32 {
    fast_sin(x + core::f32::consts::FRAC_PI_2)
}

/// Fast integer square root (for fixed-point).
#[inline]
#[must_use]
pub const fn isqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }

    let mut x = n;
    let mut y = (x + 1) >> 1;

    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }

    x
}

// ============================================================================
// Statistics
// ============================================================================

/// Compute mean of fixed-point values.
#[must_use]
pub fn mean(values: &[Fixed24_8]) -> Fixed24_8 {
    if values.is_empty() {
        return Fixed24_8::ZERO;
    }

    let sum: i64 = values.iter().map(|v| v.to_raw() as i64).sum();
    Fixed24_8::from_raw((sum / values.len() as i64) as i32)
}

/// Compute variance of fixed-point values.
#[must_use]
pub fn variance(values: &[Fixed24_8]) -> Fixed24_8 {
    if values.len() < 2 {
        return Fixed24_8::ZERO;
    }

    let m = mean(values);
    let m_raw = m.to_raw() as i64;

    let sum_sq: i64 = values.iter()
        .map(|v| {
            let diff = v.to_raw() as i64 - m_raw;
            diff * diff
        })
        .sum();

    // Divide by (n-1) for sample variance, shift right by 8 for Q24.8 * Q24.8
    let var = (sum_sq / (values.len() as i64 - 1)) >> 8;
    Fixed24_8::from_raw(var as i32)
}

/// Compute standard deviation of fixed-point values.
#[must_use]
pub fn std_dev(values: &[Fixed24_8]) -> Fixed24_8 {
    let var = variance(values);
    // Approximate sqrt using integer square root
    let var_raw = var.to_raw().unsigned_abs();
    let std_raw = isqrt(var_raw);
    Fixed24_8::from_raw(std_raw as i32)
}

/// Find minimum value.
#[must_use]
pub fn min(values: &[Fixed24_8]) -> Option<Fixed24_8> {
    values.iter().min().copied()
}

/// Find maximum value.
#[must_use]
pub fn max(values: &[Fixed24_8]) -> Option<Fixed24_8> {
    values.iter().max().copied()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iir_filter_smoothing() {
        let mut filter = IirFilter::new(100.0, 0.1); // 100 Hz, 0.1s time constant

        // Step response
        let mut output = Fixed24_8::ZERO;
        for _ in 0..100 {
            output = filter.filter(Fixed24_8::ONE);
        }

        // Should approach 1.0 after many samples
        assert!(output.to_f32() > 0.9);
    }

    #[test]
    fn test_moving_average() {
        let mut filter = MovingAverage::<4>::new();

        // Fill with 1, 2, 3, 4
        filter.filter(Fixed24_8::from_int(1));
        filter.filter(Fixed24_8::from_int(2));
        filter.filter(Fixed24_8::from_int(3));
        let avg = filter.filter(Fixed24_8::from_int(4));

        // Average should be 2.5
        assert!((avg.to_f32() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_beer_lambert_solver() {
        let solver = BeerLambertSolver::with_adult_dpf(3.0); // 3cm separation

        // Test with typical small OD changes
        let od_760 = Fixed24_8::from_f32(0.01);
        let od_850 = Fixed24_8::from_f32(0.008);

        let (delta_hbo2, delta_hbr) = solver.solve(od_760, od_850);

        // Results should be in reasonable µM range
        assert!(delta_hbo2.to_f32().abs() < 100.0);
        assert!(delta_hbr.to_f32().abs() < 100.0);
    }

    #[test]
    fn test_fast_log() {
        // Test a few known values
        assert!((fast_log(1.0) - 0.0).abs() < 0.01);
        assert!((fast_log(core::f32::consts::E) - 1.0).abs() < 0.1);
        assert!((fast_log(10.0) - 2.303).abs() < 0.1);
    }

    #[test]
    fn test_fast_sin_cos() {
        // Test at known values
        assert!((fast_sin(0.0) - 0.0).abs() < 0.01);
        assert!((fast_cos(0.0) - 1.0).abs() < 0.01);
        assert!((fast_sin(core::f32::consts::FRAC_PI_2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(99), 9); // Floor
    }

    #[test]
    fn test_statistics() {
        let values = [
            Fixed24_8::from_int(1),
            Fixed24_8::from_int(2),
            Fixed24_8::from_int(3),
            Fixed24_8::from_int(4),
            Fixed24_8::from_int(5),
        ];

        let m = mean(&values);
        assert!((m.to_f32() - 3.0).abs() < 0.01);

        let v = variance(&values);
        assert!((v.to_f32() - 2.5).abs() < 0.1);
    }
}

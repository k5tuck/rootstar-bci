//! Core types for Rootstar BCI Platform
//!
//! This module provides fundamental types used across all tiers of the BCI system:
//! - Fixed-point math for `no_std` environments
//! - EEG channel identifiers following the 10-20 system
//! - fNIRS channel definitions (source-detector pairs)
//! - Sample types for EEG, fNIRS, and hemodynamic data
//! - Stimulation parameters for tDCS/tACS/PBM

use core::ops::{Add, Div, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

// ============================================================================
// Fixed-Point Math (Q24.8 format)
// ============================================================================

/// Fixed-point number in Q24.8 format for `no_std` math.
///
/// This format provides 24 bits of integer and 8 bits of fraction,
/// allowing values from approximately -8,388,608 to +8,388,607 with
/// a precision of 1/256 ≈ 0.00390625.
///
/// # Example
///
/// ```
/// use rootstar_bci_core::types::Fixed24_8;
///
/// let a = Fixed24_8::from_f32(1.5);
/// let b = Fixed24_8::from_f32(2.25);
/// let c = a + b;
/// assert!((c.to_f32() - 3.75).abs() < 0.01);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Fixed24_8(i32);

impl Fixed24_8 {
    /// Zero value (0.0)
    pub const ZERO: Self = Self(0);

    /// One value (1.0)
    pub const ONE: Self = Self(256);

    /// Maximum representable value
    pub const MAX: Self = Self(i32::MAX);

    /// Minimum representable value
    pub const MIN: Self = Self(i32::MIN);

    /// Fractional bits (8)
    pub const FRAC_BITS: u32 = 8;

    /// Scale factor (256)
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;

    /// Create from raw underlying representation
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    /// Get the raw underlying representation
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Convert from f32 to fixed-point
    #[inline]
    #[must_use]
    pub fn from_f32(f: f32) -> Self {
        Self((f * Self::SCALE as f32) as i32)
    }

    /// Convert from fixed-point to f32
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    /// Convert from i32 integer to fixed-point
    #[inline]
    #[must_use]
    pub const fn from_int(i: i32) -> Self {
        Self(i << Self::FRAC_BITS)
    }

    /// Convert from fixed-point to i32 (truncates fractional part)
    #[inline]
    #[must_use]
    pub const fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    /// Absolute value
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Saturating addition
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating multiplication
    #[inline]
    #[must_use]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let product = (self.0 as i64).saturating_mul(rhs.0 as i64);
        let shifted = product >> Self::FRAC_BITS;
        Self(shifted.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// Convert from ADS1299 24-bit signed raw value to microvolts.
    ///
    /// The ADS1299 has an LSB of 4.5V / 2^23 / gain ≈ 0.536 µV at gain=1.
    /// At gain=24 (typical for EEG), LSB ≈ 0.0223 µV.
    ///
    /// # Arguments
    ///
    /// * `raw` - 24-bit signed ADC value (sign-extended to i32)
    /// * `gain` - Programmable gain (1, 2, 4, 6, 8, 12, or 24)
    #[inline]
    #[must_use]
    pub fn from_ads1299_raw(raw: i32, gain: u8) -> Self {
        // LSB in µV at gain=1: 4.5V / 2^23 * 1e6 ≈ 0.536 µV
        // In Q24.8: 0.536 * 256 ≈ 137
        const LSB_UV_Q8: i32 = 137;

        // Scale: raw * 0.536 / gain, in Q24.8 format
        // To avoid overflow: (raw * LSB_UV_Q8) / (gain * 256)
        Self((raw as i64 * LSB_UV_Q8 as i64 / (gain as i64 * 256)) as i32)
    }
}

impl Add for Fixed24_8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl Sub for Fixed24_8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for Fixed24_8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // (a * b) >> 8 to maintain Q24.8 format
        // Use i64 to prevent overflow
        Self(((self.0 as i64 * rhs.0 as i64) >> Self::FRAC_BITS) as i32)
    }
}

impl Div for Fixed24_8 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        // (a << 8) / b to maintain Q24.8 format
        // Use i64 to prevent overflow
        Self((((self.0 as i64) << Self::FRAC_BITS) / rhs.0 as i64) as i32)
    }
}

impl Neg for Fixed24_8 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for Fixed24_8 {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}Q8", self.0);
    }
}

// ============================================================================
// EEG Channel Types
// ============================================================================

/// EEG channel identifier following the 10-20 system.
///
/// The ESP-EEG (ADS1299) supports 8 differential channels. This enum
/// maps to a typical 8-channel montage covering frontal, central,
/// parietal, and occipital regions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EegChannel {
    /// Frontal-polar left (prefrontal cortex)
    Fp1 = 0,
    /// Frontal-polar right (prefrontal cortex)
    Fp2 = 1,
    /// Central left (motor cortex, right hand)
    C3 = 2,
    /// Central right (motor cortex, left hand)
    C4 = 3,
    /// Parietal left (somatosensory)
    P3 = 4,
    /// Parietal right (somatosensory)
    P4 = 5,
    /// Occipital left (visual cortex)
    O1 = 6,
    /// Occipital right (visual cortex)
    O2 = 7,
}

impl EegChannel {
    /// All channels in order
    pub const ALL: [Self; 8] = [
        Self::Fp1, Self::Fp2, Self::C3, Self::C4,
        Self::P3, Self::P4, Self::O1, Self::O2,
    ];

    /// Number of channels
    pub const COUNT: usize = 8;

    /// Get the array index for this channel
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Get channel from index (returns None if out of range)
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Fp1),
            1 => Some(Self::Fp2),
            2 => Some(Self::C3),
            3 => Some(Self::C4),
            4 => Some(Self::P3),
            5 => Some(Self::P4),
            6 => Some(Self::O1),
            7 => Some(Self::O2),
            _ => None,
        }
    }

    /// Get the 10-20 system name for this channel
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp1 => "Fp1",
            Self::Fp2 => "Fp2",
            Self::C3 => "C3",
            Self::C4 => "C4",
            Self::P3 => "P3",
            Self::P4 => "P4",
            Self::O1 => "O1",
            Self::O2 => "O2",
        }
    }

    /// Check if this channel is on the left hemisphere
    #[inline]
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, Self::Fp1 | Self::C3 | Self::P3 | Self::O1)
    }

    /// Check if this channel is on the right hemisphere
    #[inline]
    #[must_use]
    pub const fn is_right(self) -> bool {
        matches!(self, Self::Fp2 | Self::C4 | Self::P4 | Self::O2)
    }

    /// Get the contralateral (opposite hemisphere) channel
    #[inline]
    #[must_use]
    pub const fn contralateral(self) -> Self {
        match self {
            Self::Fp1 => Self::Fp2,
            Self::Fp2 => Self::Fp1,
            Self::C3 => Self::C4,
            Self::C4 => Self::C3,
            Self::P3 => Self::P4,
            Self::P4 => Self::P3,
            Self::O1 => Self::O2,
            Self::O2 => Self::O1,
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EegChannel {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

// ============================================================================
// fNIRS Channel Types
// ============================================================================

/// NIR wavelength for fNIRS measurements.
///
/// Dual-wavelength fNIRS uses two wavelengths in the optical window (700-900nm)
/// to distinguish between oxygenated and deoxygenated hemoglobin.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum Wavelength {
    /// 760nm: HbR (deoxygenated hemoglobin) has higher absorption
    Nm760 = 760,
    /// 850nm: HbO₂ (oxygenated hemoglobin) has higher absorption
    Nm850 = 850,
}

impl Wavelength {
    /// Get the wavelength in nanometers
    #[inline]
    #[must_use]
    pub const fn nm(self) -> u16 {
        self as u16
    }

    /// Get the typical Differential Pathlength Factor (DPF) for adult head
    ///
    /// DPF accounts for photon scattering in tissue. Values from Duncan et al. (1996).
    #[inline]
    #[must_use]
    pub const fn adult_dpf(self) -> f32 {
        match self {
            Self::Nm760 => 5.13,
            Self::Nm850 => 4.99,
        }
    }

    /// Get the extinction coefficient for HbO₂ at this wavelength (cm⁻¹·M⁻¹)
    #[inline]
    #[must_use]
    pub const fn extinction_hbo2(self) -> f64 {
        match self {
            Self::Nm760 => 1486.0,
            Self::Nm850 => 2526.0,
        }
    }

    /// Get the extinction coefficient for HbR at this wavelength (cm⁻¹·M⁻¹)
    #[inline]
    #[must_use]
    pub const fn extinction_hbr(self) -> f64 {
        match self {
            Self::Nm760 => 3843.0,
            Self::Nm850 => 1798.0,
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for Wavelength {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}nm", self.nm());
    }
}

/// fNIRS channel definition (source-detector pair).
///
/// Each fNIRS channel is defined by an LED source, a photodiode detector,
/// and the separation distance between them. The penetration depth is
/// approximately half the source-detector separation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FnirsChannel {
    /// LED source index (0-3 typical)
    pub source: u8,
    /// Photodiode detector index (0-3 typical)
    pub detector: u8,
    /// Source-detector separation in millimeters (typically 25-40mm)
    pub separation_mm: u8,
}

impl FnirsChannel {
    /// Create a new fNIRS channel
    #[inline]
    #[must_use]
    pub const fn new(source: u8, detector: u8, separation_mm: u8) -> Self {
        Self { source, detector, separation_mm }
    }

    /// Approximate cortical penetration depth in millimeters.
    ///
    /// The "banana-shaped" photon path means penetration is roughly
    /// half the source-detector separation.
    #[inline]
    #[must_use]
    pub const fn depth_mm(self) -> u8 {
        self.separation_mm / 2
    }

    /// Get the source-detector distance in centimeters
    #[inline]
    #[must_use]
    pub fn distance_cm(self) -> f32 {
        self.separation_mm as f32 / 10.0
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for FnirsChannel {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "S{}D{}@{}mm", self.source, self.detector, self.separation_mm);
    }
}

// ============================================================================
// Sample Types
// ============================================================================

/// Single EEG sample containing all 8 channels.
///
/// Samples are timestamped in microseconds and include a sequence number
/// for detecting dropped packets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EegSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Channel values in microvolts (Q24.8 fixed-point)
    pub channels: [Fixed24_8; 8],
    /// Sequence number for packet ordering/loss detection
    pub sequence: u32,
}

impl EegSample {
    /// Create a new sample with zero values
    #[inline]
    #[must_use]
    pub const fn new(timestamp_us: u64, sequence: u32) -> Self {
        Self {
            timestamp_us,
            channels: [Fixed24_8::ZERO; 8],
            sequence,
        }
    }

    /// Get the value for a specific channel
    #[inline]
    #[must_use]
    pub fn channel(&self, ch: EegChannel) -> Fixed24_8 {
        self.channels[ch.index()]
    }

    /// Set the value for a specific channel
    #[inline]
    pub fn set_channel(&mut self, ch: EegChannel, value: Fixed24_8) {
        self.channels[ch.index()] = value;
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EegSample {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "EEG[{}]@{}us", self.sequence, self.timestamp_us);
    }
}

/// Single fNIRS raw intensity sample (both wavelengths).
///
/// Raw ADC values must be processed through the Modified Beer-Lambert Law
/// to obtain hemoglobin concentration changes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnirsSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Source-detector channel
    pub channel: FnirsChannel,
    /// Raw ADC intensity at 760nm
    pub intensity_760: u16,
    /// Raw ADC intensity at 850nm
    pub intensity_850: u16,
    /// Sequence number for packet ordering/loss detection
    pub sequence: u32,
}

impl FnirsSample {
    /// Create a new sample
    #[inline]
    #[must_use]
    pub const fn new(
        timestamp_us: u64,
        channel: FnirsChannel,
        intensity_760: u16,
        intensity_850: u16,
        sequence: u32,
    ) -> Self {
        Self { timestamp_us, channel, intensity_760, intensity_850, sequence }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for FnirsSample {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f, "fNIRS[{}]@{}us: 760nm={}, 850nm={}",
            self.sequence, self.timestamp_us, self.intensity_760, self.intensity_850
        );
    }
}

/// Computed hemodynamic values from fNIRS data.
///
/// These values represent changes from baseline in hemoglobin concentrations,
/// computed using the Modified Beer-Lambert Law.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HemodynamicSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Source-detector channel
    pub channel: FnirsChannel,
    /// Change in oxygenated hemoglobin concentration (µM)
    pub delta_hbo2: Fixed24_8,
    /// Change in deoxygenated hemoglobin concentration (µM)
    pub delta_hbr: Fixed24_8,
    /// Change in total hemoglobin (HbO₂ + HbR) (µM)
    pub delta_hbt: Fixed24_8,
}

impl HemodynamicSample {
    /// Create a new sample
    #[inline]
    #[must_use]
    pub const fn new(
        timestamp_us: u64,
        channel: FnirsChannel,
        delta_hbo2: Fixed24_8,
        delta_hbr: Fixed24_8,
    ) -> Self {
        Self {
            timestamp_us,
            channel,
            delta_hbo2,
            delta_hbr,
            delta_hbt: Fixed24_8::from_raw(delta_hbo2.to_raw().saturating_add(delta_hbr.to_raw())),
        }
    }

    /// Calculate the oxygenation index (HbO₂ / (HbO₂ + |HbR|))
    ///
    /// Returns a value between 0.0 and 1.0 representing tissue oxygenation.
    #[inline]
    #[must_use]
    pub fn oxygenation_index(&self) -> f32 {
        let hbo2 = self.delta_hbo2.to_f32();
        let hbr = self.delta_hbr.to_f32().abs();
        let total = hbo2 + hbr;
        if total > 0.001 {
            hbo2 / total
        } else {
            0.5 // Baseline when no change detected
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for HemodynamicSample {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f, "Hemo@{}us: HbO2={}, HbR={}",
            self.timestamp_us, self.delta_hbo2, self.delta_hbr
        );
    }
}

// ============================================================================
// Stimulation Types
// ============================================================================

/// Stimulation mode for tDCS/tACS/PBM.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum StimMode {
    /// Stimulation disabled
    #[default]
    Off,
    /// tDCS anodal (positive at target electrode)
    TdcsAnodal,
    /// tDCS cathodal (negative at target electrode)
    TdcsCathodal,
    /// tACS (alternating current at specified frequency)
    Tacs,
    /// Photobiomodulation (NIR light stimulation)
    Pbm,
}

impl StimMode {
    /// Check if this mode involves electrical stimulation
    #[inline]
    #[must_use]
    pub const fn is_electrical(self) -> bool {
        matches!(self, Self::TdcsAnodal | Self::TdcsCathodal | Self::Tacs)
    }

    /// Check if this mode involves optical stimulation
    #[inline]
    #[must_use]
    pub const fn is_optical(self) -> bool {
        matches!(self, Self::Pbm)
    }

    /// Check if stimulation is active
    #[inline]
    #[must_use]
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for StimMode {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::Off => defmt::write!(f, "Off"),
            Self::TdcsAnodal => defmt::write!(f, "tDCS+"),
            Self::TdcsCathodal => defmt::write!(f, "tDCS-"),
            Self::Tacs => defmt::write!(f, "tACS"),
            Self::Pbm => defmt::write!(f, "PBM"),
        }
    }
}

/// Stimulation parameters.
///
/// # Safety Limits
///
/// - Maximum current: 2000 µA (2 mA) - HARDWARE ENFORCED
/// - Maximum duration: 30 minutes
/// - Minimum ramp time: 10 ms
/// - Maximum tACS frequency: 100 Hz
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StimParams {
    /// Stimulation mode
    pub mode: StimMode,
    /// Current amplitude in microamps (0-2000)
    pub amplitude_ua: u16,
    /// Frequency in Hz for tACS (0 for DC modes)
    pub frequency_hz: u16,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Ramp up/down time in milliseconds
    pub ramp_ms: u16,
}

impl StimParams {
    /// Maximum allowed current (2 mA)
    pub const MAX_CURRENT_UA: u16 = 2000;

    /// Maximum allowed duration (30 minutes)
    pub const MAX_DURATION_MS: u32 = 30 * 60 * 1000;

    /// Minimum required ramp time (10 ms)
    pub const MIN_RAMP_MS: u16 = 10;

    /// Maximum tACS frequency (100 Hz)
    pub const MAX_TACS_FREQUENCY_HZ: u16 = 100;

    /// Create new stimulation parameters (disabled)
    #[inline]
    #[must_use]
    pub const fn off() -> Self {
        Self {
            mode: StimMode::Off,
            amplitude_ua: 0,
            frequency_hz: 0,
            duration_ms: 0,
            ramp_ms: 0,
        }
    }

    /// Create tDCS parameters
    #[inline]
    #[must_use]
    pub const fn tdcs(anodal: bool, amplitude_ua: u16, duration_ms: u32, ramp_ms: u16) -> Self {
        Self {
            mode: if anodal { StimMode::TdcsAnodal } else { StimMode::TdcsCathodal },
            amplitude_ua,
            frequency_hz: 0,
            duration_ms,
            ramp_ms,
        }
    }

    /// Create tACS parameters
    #[inline]
    #[must_use]
    pub const fn tacs(amplitude_ua: u16, frequency_hz: u16, duration_ms: u32, ramp_ms: u16) -> Self {
        Self {
            mode: StimMode::Tacs,
            amplitude_ua,
            frequency_hz,
            duration_ms,
            ramp_ms,
        }
    }

    /// Create PBM parameters
    #[inline]
    #[must_use]
    pub const fn pbm(intensity: u16, duration_ms: u32) -> Self {
        Self {
            mode: StimMode::Pbm,
            amplitude_ua: intensity, // Reused for LED intensity
            frequency_hz: 0,
            duration_ms,
            ramp_ms: 0,
        }
    }
}

impl Default for StimParams {
    fn default() -> Self {
        Self::off()
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for StimParams {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f, "Stim({}, {}uA, {}Hz, {}ms, ramp={}ms)",
            self.mode, self.amplitude_ua, self.frequency_hz, self.duration_ms, self.ramp_ms
        );
    }
}

// ============================================================================
// EMG (Electromyography) Types
// ============================================================================

/// EMG channel identifier for facial muscle measurements.
///
/// EMG channels target specific facial muscles relevant for sensory
/// experience detection (taste, smell, emotion).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EmgChannel {
    /// Zygomaticus major (smile muscle) - left
    ZygomaticusL = 0,
    /// Zygomaticus major (smile muscle) - right
    ZygomaticusR = 1,
    /// Corrugator supercilii (frown muscle) - left
    CorrugatorL = 2,
    /// Corrugator supercilii (frown muscle) - right
    CorrugatorR = 3,
    /// Masseter (jaw muscle) - left
    MasseterL = 4,
    /// Masseter (jaw muscle) - right
    MasseterR = 5,
    /// Orbicularis oris (lip muscle) - upper
    OrbicularisU = 6,
    /// Orbicularis oris (lip muscle) - lower
    OrbicularisD = 7,
}

impl EmgChannel {
    /// All channels in order
    pub const ALL: [Self; 8] = [
        Self::ZygomaticusL, Self::ZygomaticusR,
        Self::CorrugatorL, Self::CorrugatorR,
        Self::MasseterL, Self::MasseterR,
        Self::OrbicularisU, Self::OrbicularisD,
    ];

    /// Number of channels
    pub const COUNT: usize = 8;

    /// Get the array index for this channel
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Get channel from index (returns None if out of range)
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::ZygomaticusL),
            1 => Some(Self::ZygomaticusR),
            2 => Some(Self::CorrugatorL),
            3 => Some(Self::CorrugatorR),
            4 => Some(Self::MasseterL),
            5 => Some(Self::MasseterR),
            6 => Some(Self::OrbicularisU),
            7 => Some(Self::OrbicularisD),
            _ => None,
        }
    }

    /// Get the muscle name for this channel
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ZygomaticusL => "Zyg-L",
            Self::ZygomaticusR => "Zyg-R",
            Self::CorrugatorL => "Cor-L",
            Self::CorrugatorR => "Cor-R",
            Self::MasseterL => "Mas-L",
            Self::MasseterR => "Mas-R",
            Self::OrbicularisU => "Orb-U",
            Self::OrbicularisD => "Orb-D",
        }
    }

    /// Get full muscle name
    #[inline]
    #[must_use]
    pub const fn full_name(self) -> &'static str {
        match self {
            Self::ZygomaticusL => "Zygomaticus Major Left",
            Self::ZygomaticusR => "Zygomaticus Major Right",
            Self::CorrugatorL => "Corrugator Supercilii Left",
            Self::CorrugatorR => "Corrugator Supercilii Right",
            Self::MasseterL => "Masseter Left",
            Self::MasseterR => "Masseter Right",
            Self::OrbicularisU => "Orbicularis Oris Upper",
            Self::OrbicularisD => "Orbicularis Oris Lower",
        }
    }

    /// Check if this channel is on the left side
    #[inline]
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, Self::ZygomaticusL | Self::CorrugatorL | Self::MasseterL)
    }

    /// Check if this channel is on the right side
    #[inline]
    #[must_use]
    pub const fn is_right(self) -> bool {
        matches!(self, Self::ZygomaticusR | Self::CorrugatorR | Self::MasseterR)
    }

    /// Get the associated emotional valence for this muscle
    /// Positive = smile/pleasure, Negative = frown/displeasure
    #[inline]
    #[must_use]
    pub const fn valence(self) -> i8 {
        match self {
            Self::ZygomaticusL | Self::ZygomaticusR => 1,  // Smile
            Self::CorrugatorL | Self::CorrugatorR => -1,   // Frown
            Self::MasseterL | Self::MasseterR => 0,        // Neutral (chewing)
            Self::OrbicularisU | Self::OrbicularisD => 0,  // Neutral (lip movement)
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EmgChannel {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

/// Single EMG sample containing all channels.
///
/// EMG measures electrical activity from facial muscles at high sample rates
/// (500-1000 Hz). Values represent muscle activation in microvolts.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmgSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Channel values in microvolts (Q24.8 fixed-point)
    pub channels: [Fixed24_8; 8],
    /// Sequence number for packet ordering/loss detection
    pub sequence: u32,
}

impl EmgSample {
    /// Create a new sample with zero values
    #[inline]
    #[must_use]
    pub const fn new(timestamp_us: u64, sequence: u32) -> Self {
        Self {
            timestamp_us,
            channels: [Fixed24_8::ZERO; 8],
            sequence,
        }
    }

    /// Get the value for a specific channel
    #[inline]
    #[must_use]
    pub fn channel(&self, ch: EmgChannel) -> Fixed24_8 {
        self.channels[ch.index()]
    }

    /// Set the value for a specific channel
    #[inline]
    pub fn set_channel(&mut self, ch: EmgChannel, value: Fixed24_8) {
        self.channels[ch.index()] = value;
    }

    /// Compute RMS (root mean square) activation across all channels
    #[inline]
    #[must_use]
    pub fn rms_activation(&self) -> Fixed24_8 {
        let mut sum_sq = 0i64;
        for &val in &self.channels {
            let v = val.to_raw() as i64;
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / 8;
        let rms = libm::sqrtf(mean_sq as f32 / 65536.0); // Adjust for Q24.8
        Fixed24_8::from_f32(rms)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EmgSample {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "EMG[{}]@{}us", self.sequence, self.timestamp_us);
    }
}

// ============================================================================
// EDA (Electrodermal Activity) Types
// ============================================================================

/// EDA measurement site.
///
/// Electrodermal activity is typically measured at sites with high
/// sweat gland density.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdaSite {
    /// Palmar (palm) - highest sweat gland density
    PalmarL = 0,
    /// Palmar (palm) - right hand
    PalmarR = 1,
    /// Thenar (thumb base)
    ThenarL = 2,
    /// Thenar (thumb base) - right hand
    ThenarR = 3,
}

impl EdaSite {
    /// All sites in order
    pub const ALL: [Self; 4] = [
        Self::PalmarL, Self::PalmarR,
        Self::ThenarL, Self::ThenarR,
    ];

    /// Number of sites
    pub const COUNT: usize = 4;

    /// Get the array index for this site
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Get site name
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PalmarL => "Palm-L",
            Self::PalmarR => "Palm-R",
            Self::ThenarL => "Then-L",
            Self::ThenarR => "Then-R",
        }
    }

    /// Get site from index
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::PalmarL),
            1 => Some(Self::PalmarR),
            2 => Some(Self::ThenarL),
            3 => Some(Self::ThenarR),
            _ => None,
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EdaSite {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

/// Single EDA sample with skin conductance measurements.
///
/// EDA measures skin conductance in microsiemens (µS). The signal has two
/// components:
/// - Tonic: Slow baseline level (SCL - Skin Conductance Level)
/// - Phasic: Fast responses to stimuli (SCR - Skin Conductance Response)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdaSample {
    /// Timestamp in microseconds since device boot
    pub timestamp_us: u64,
    /// Raw skin conductance values per site in micro-siemens (µS, Q24.8)
    pub conductance: [Fixed24_8; 4],
    /// Sequence number for packet ordering/loss detection
    pub sequence: u32,
}

impl EdaSample {
    /// Create a new sample with zero values
    #[inline]
    #[must_use]
    pub const fn new(timestamp_us: u64, sequence: u32) -> Self {
        Self {
            timestamp_us,
            conductance: [Fixed24_8::ZERO; 4],
            sequence,
        }
    }

    /// Get the value for a specific site
    #[inline]
    #[must_use]
    pub fn site(&self, site: EdaSite) -> Fixed24_8 {
        self.conductance[site.index()]
    }

    /// Set the value for a specific site
    #[inline]
    pub fn set_site(&mut self, site: EdaSite, value: Fixed24_8) {
        self.conductance[site.index()] = value;
    }

    /// Get mean conductance across all sites
    #[inline]
    #[must_use]
    pub fn mean_conductance(&self) -> Fixed24_8 {
        let sum: i32 = self.conductance.iter().map(|c| c.to_raw()).sum();
        Fixed24_8::from_raw(sum / 4)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EdaSample {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "EDA[{}]@{}us", self.sequence, self.timestamp_us);
    }
}

/// Processed EDA with tonic/phasic decomposition.
///
/// The skin conductance signal is decomposed into:
/// - SCL (Skin Conductance Level): Tonic, slow-varying baseline
/// - SCR (Skin Conductance Response): Phasic, event-related peaks
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdaDecomposed {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Measurement site
    pub site: EdaSite,
    /// Tonic level (SCL) in µS (Q24.8)
    pub scl: Fixed24_8,
    /// Phasic response (SCR) in µS (Q24.8)
    pub scr: Fixed24_8,
    /// SCR amplitude (peak height) if response detected
    pub scr_amplitude: Fixed24_8,
    /// SCR rise time in milliseconds (0 if no response)
    pub scr_rise_time_ms: u16,
}

impl EdaDecomposed {
    /// Create a new decomposed EDA sample
    #[inline]
    #[must_use]
    pub const fn new(timestamp_us: u64, site: EdaSite, scl: Fixed24_8, scr: Fixed24_8) -> Self {
        Self {
            timestamp_us,
            site,
            scl,
            scr,
            scr_amplitude: Fixed24_8::ZERO,
            scr_rise_time_ms: 0,
        }
    }

    /// Check if a significant SCR (skin conductance response) is present
    ///
    /// A typical threshold is 0.01-0.05 µS for a significant response.
    #[inline]
    #[must_use]
    pub fn has_response(&self) -> bool {
        self.scr_amplitude.to_f32() >= 0.01
    }

    /// Calculate arousal level (0.0 to 1.0) based on SCL
    ///
    /// Typical resting SCL is 2-20 µS, with arousal increasing the level.
    #[inline]
    #[must_use]
    pub fn arousal_level(&self) -> f32 {
        let scl = self.scl.to_f32();
        // Normalize: 2 µS = 0.0, 20 µS = 1.0
        ((scl - 2.0) / 18.0).clamp(0.0, 1.0)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EdaDecomposed {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f, "EDA@{}us: SCL={}, SCR={}",
            self.timestamp_us, self.scl, self.scr
        );
    }
}

// ============================================================================
// EEG Frequency Bands
// ============================================================================

/// Standard EEG frequency band definitions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EegBand {
    /// Delta: 0.5-4 Hz (deep sleep)
    Delta,
    /// Theta: 4-8 Hz (drowsiness, memory)
    Theta,
    /// Alpha: 8-13 Hz (relaxed, eyes closed)
    Alpha,
    /// Beta: 13-30 Hz (active thinking)
    Beta,
    /// Gamma: 30-100 Hz (cognitive processing)
    Gamma,
}

impl EegBand {
    /// Get the frequency range for this band (low, high) in Hz
    #[inline]
    #[must_use]
    pub const fn range_hz(self) -> (f32, f32) {
        match self {
            Self::Delta => (0.5, 4.0),
            Self::Theta => (4.0, 8.0),
            Self::Alpha => (8.0, 13.0),
            Self::Beta => (13.0, 30.0),
            Self::Gamma => (30.0, 100.0),
        }
    }

    /// Get the band name
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Delta => "Delta",
            Self::Theta => "Theta",
            Self::Alpha => "Alpha",
            Self::Beta => "Beta",
            Self::Gamma => "Gamma",
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for EegBand {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed24_8_conversion() {
        let f = Fixed24_8::from_f32(1.5);
        assert!((f.to_f32() - 1.5).abs() < 0.01);

        let f = Fixed24_8::from_f32(-2.25);
        assert!((f.to_f32() - (-2.25)).abs() < 0.01);
    }

    #[test]
    fn test_fixed24_8_arithmetic() {
        let a = Fixed24_8::from_f32(1.5);
        let b = Fixed24_8::from_f32(2.25);

        let sum = a + b;
        assert!((sum.to_f32() - 3.75).abs() < 0.01);

        let diff = b - a;
        assert!((diff.to_f32() - 0.75).abs() < 0.01);

        let prod = a * b;
        assert!((prod.to_f32() - 3.375).abs() < 0.02);

        let quot = b / a;
        assert!((quot.to_f32() - 1.5).abs() < 0.02);
    }

    #[test]
    fn test_fixed24_8_from_int() {
        let i = Fixed24_8::from_int(42);
        assert_eq!(i.to_int(), 42);
        assert!((i.to_f32() - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_ads1299_conversion() {
        // Full scale positive at gain 24
        let raw = 0x7FFFFF_i32; // Max 24-bit positive
        let uv = Fixed24_8::from_ads1299_raw(raw, 24);
        // Expected: 8388607 * 0.536 / 24 ≈ 187.5 µV
        assert!((uv.to_f32() - 187.5).abs() < 1.0);

        // Typical EEG signal (~10 µV)
        let raw = 44739; // ~10 µV at gain 24
        let uv = Fixed24_8::from_ads1299_raw(raw, 24);
        assert!((uv.to_f32() - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_eeg_channel_symmetry() {
        for ch in EegChannel::ALL {
            let contra = ch.contralateral();
            assert_eq!(contra.contralateral(), ch);
            assert_ne!(ch.is_left(), contra.is_left());
        }
    }

    #[test]
    fn test_fnirs_channel_depth() {
        let ch = FnirsChannel::new(0, 0, 30);
        assert_eq!(ch.depth_mm(), 15);
        assert!((ch.distance_cm() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_hemodynamic_oxygenation_index() {
        let ch = FnirsChannel::new(0, 0, 30);

        // Fully oxygenated (HbO2 positive, HbR zero)
        let sample = HemodynamicSample::new(0, ch, Fixed24_8::from_f32(10.0), Fixed24_8::ZERO);
        assert!((sample.oxygenation_index() - 1.0).abs() < 0.01);

        // Fully deoxygenated (HbO2 zero, HbR negative)
        let sample = HemodynamicSample::new(0, ch, Fixed24_8::ZERO, Fixed24_8::from_f32(-10.0));
        assert!((sample.oxygenation_index() - 0.0).abs() < 0.01);

        // Balanced
        let sample = HemodynamicSample::new(0, ch, Fixed24_8::from_f32(5.0), Fixed24_8::from_f32(-5.0));
        assert!((sample.oxygenation_index() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stim_mode_classification() {
        assert!(StimMode::TdcsAnodal.is_electrical());
        assert!(StimMode::TdcsCathodal.is_electrical());
        assert!(StimMode::Tacs.is_electrical());
        assert!(!StimMode::Pbm.is_electrical());
        assert!(!StimMode::Off.is_electrical());

        assert!(StimMode::Pbm.is_optical());
        assert!(!StimMode::TdcsAnodal.is_optical());

        assert!(!StimMode::Off.is_active());
        assert!(StimMode::TdcsAnodal.is_active());
    }

    #[test]
    fn test_eeg_band_ranges() {
        let (delta_low, delta_high) = EegBand::Delta.range_hz();
        let (theta_low, _) = EegBand::Theta.range_hz();

        assert!((delta_low - 0.5).abs() < 0.001);
        assert!((delta_high - theta_low).abs() < 0.001);
    }
}

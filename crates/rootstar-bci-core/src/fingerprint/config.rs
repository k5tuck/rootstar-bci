//! Configuration types for scalable Neural Fingerprint system.
//!
//! Supports progressive hardware configurations from 8-channel proof-of-concept
//! to 256-channel high-density research systems.

use serde::{Deserialize, Serialize};

// ============================================================================
// Channel Density Configurations
// ============================================================================

/// Hardware channel density configuration.
///
/// Defines the number of EEG and fNIRS channels based on hardware setup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelDensity {
    /// Basic 8-channel proof-of-concept (1× ADS1299)
    Basic8,
    /// Standard 32-channel research (4× ADS1299)
    Standard32,
    /// Medium 64-channel clinical (8× ADS1299)
    Medium64,
    /// High-density 128-channel (16× ADS1299)
    HighDensity128,
    /// Ultra high-density 256-channel (32× ADS1299)
    UltraHighDensity256,
    /// Custom configuration
    Custom {
        /// Number of EEG channels
        eeg_channels: u16,
        /// Number of fNIRS source-detector pairs
        fnirs_channels: u16,
    },
}

impl ChannelDensity {
    /// Get the number of EEG channels for this density.
    #[inline]
    #[must_use]
    pub const fn eeg_channels(self) -> u16 {
        match self {
            Self::Basic8 => 8,
            Self::Standard32 => 32,
            Self::Medium64 => 64,
            Self::HighDensity128 => 128,
            Self::UltraHighDensity256 => 256,
            Self::Custom { eeg_channels, .. } => eeg_channels,
        }
    }

    /// Get the number of fNIRS channels for this density.
    #[inline]
    #[must_use]
    pub const fn fnirs_channels(self) -> u16 {
        match self {
            Self::Basic8 => 4,
            Self::Standard32 => 16,
            Self::Medium64 => 32,
            Self::HighDensity128 => 64,
            Self::UltraHighDensity256 => 128,
            Self::Custom { fnirs_channels, .. } => fnirs_channels,
        }
    }

    /// Get the number of ADS1299 chips required.
    #[inline]
    #[must_use]
    pub const fn ads1299_count(self) -> u8 {
        match self {
            Self::Basic8 => 1,
            Self::Standard32 => 4,
            Self::Medium64 => 8,
            Self::HighDensity128 => 16,
            Self::UltraHighDensity256 => 32,
            Self::Custom { eeg_channels, .. } => ((eeg_channels + 7) / 8) as u8,
        }
    }

    /// Get the recommended inter-electrode distance in mm.
    #[inline]
    #[must_use]
    pub const fn electrode_spacing_mm(self) -> u8 {
        match self {
            Self::Basic8 => 60,    // Wide spacing for basic montage
            Self::Standard32 => 35,
            Self::Medium64 => 25,
            Self::HighDensity128 => 15,
            Self::UltraHighDensity256 => 10,
            Self::Custom { .. } => 20, // Default for custom
        }
    }

    /// Get configuration name.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Basic8 => "8-channel",
            Self::Standard32 => "32-channel",
            Self::Medium64 => "64-channel",
            Self::HighDensity128 => "128-channel HD",
            Self::UltraHighDensity256 => "256-channel UHD",
            Self::Custom { .. } => "custom",
        }
    }
}

impl Default for ChannelDensity {
    fn default() -> Self {
        Self::Basic8
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for ChannelDensity {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.name());
    }
}

// ============================================================================
// fNIRS Configuration
// ============================================================================

/// fNIRS optode array configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnirsConfig {
    /// Number of LED light sources
    pub num_sources: u8,
    /// Number of photodiode detectors
    pub num_detectors: u8,
    /// Standard source-detector distance in mm
    pub long_channel_mm: u8,
    /// Short-channel distance in mm (for superficial regression)
    pub short_channel_mm: u8,
    /// Number of short-channel pairs
    pub num_short_channels: u8,
    /// Sampling rate in Hz
    pub sample_rate_hz: u8,
}

impl FnirsConfig {
    /// Basic 4-channel fNIRS (2 sources × 2 detectors).
    pub const BASIC: Self = Self {
        num_sources: 2,
        num_detectors: 2,
        long_channel_mm: 30,
        short_channel_mm: 15,
        num_short_channels: 2,
        sample_rate_hz: 10,
    };

    /// Standard 16-channel fNIRS (4 sources × 4 detectors).
    pub const STANDARD: Self = Self {
        num_sources: 4,
        num_detectors: 4,
        long_channel_mm: 30,
        short_channel_mm: 15,
        num_short_channels: 4,
        sample_rate_hz: 25,
    };

    /// High-density 64-channel fNIRS for DOT.
    pub const HIGH_DENSITY: Self = Self {
        num_sources: 32,
        num_detectors: 32,
        long_channel_mm: 30,
        short_channel_mm: 15,
        num_short_channels: 16,
        sample_rate_hz: 25,
    };

    /// Ultra high-density 128-channel fNIRS.
    pub const ULTRA_HIGH_DENSITY: Self = Self {
        num_sources: 48,
        num_detectors: 48,
        long_channel_mm: 30,
        short_channel_mm: 15,
        num_short_channels: 32,
        sample_rate_hz: 25,
    };

    /// Get total number of measurement channels.
    #[inline]
    #[must_use]
    pub const fn total_channels(&self) -> u16 {
        // Each source can illuminate multiple detectors
        // Simplified: num_sources × num_detectors / 2 (non-overlapping)
        (self.num_sources as u16 * self.num_detectors as u16 / 2)
            + self.num_short_channels as u16
    }

    /// Get estimated cortical penetration depth in mm.
    #[inline]
    #[must_use]
    pub const fn penetration_depth_mm(&self) -> u8 {
        self.long_channel_mm / 2
    }

    /// Get configuration for channel density.
    #[inline]
    #[must_use]
    pub const fn for_density(density: ChannelDensity) -> Self {
        match density {
            ChannelDensity::Basic8 => Self::BASIC,
            ChannelDensity::Standard32 => Self::STANDARD,
            ChannelDensity::Medium64 => Self::STANDARD,
            ChannelDensity::HighDensity128 => Self::HIGH_DENSITY,
            ChannelDensity::UltraHighDensity256 => Self::ULTRA_HIGH_DENSITY,
            ChannelDensity::Custom { .. } => Self::STANDARD,
        }
    }
}

impl Default for FnirsConfig {
    fn default() -> Self {
        Self::BASIC
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for FnirsConfig {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "fNIRS({}S×{}D, {}Hz)",
            self.num_sources,
            self.num_detectors,
            self.sample_rate_hz
        );
    }
}

// ============================================================================
// System Configuration
// ============================================================================

/// Complete system configuration for Neural Fingerprint platform.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Channel density configuration
    pub density: ChannelDensity,
    /// fNIRS configuration
    pub fnirs: FnirsConfig,
    /// EEG sampling rate in Hz
    pub eeg_sample_rate_hz: u16,
    /// Enable stimulation subsystem
    pub stimulation_enabled: bool,
    /// Number of stimulation channels
    pub stim_channels: u8,
    /// Master clock frequency in Hz
    pub master_clock_hz: u32,
    /// fNIRS trigger divisor (master_clock / divisor = fNIRS rate)
    pub fnirs_trigger_divisor: u32,
    /// Maximum timestamp drift before resync (microseconds)
    pub max_drift_us: u32,
}

impl SystemConfig {
    /// Create configuration for given channel density.
    #[must_use]
    pub const fn new(density: ChannelDensity) -> Self {
        Self {
            density,
            fnirs: FnirsConfig::for_density(density),
            eeg_sample_rate_hz: 500,
            stimulation_enabled: true,
            stim_channels: 8,
            master_clock_hz: 2_048_000,
            fnirs_trigger_divisor: 81920, // 2.048 MHz / 81920 = 25 Hz
            max_drift_us: 100,
        }
    }

    /// Get total data rate in bits per second.
    #[must_use]
    pub const fn data_rate_bps(&self) -> u32 {
        let eeg_channels = self.density.eeg_channels() as u32;
        let fnirs_channels = self.fnirs.total_channels() as u32;
        let eeg_rate = self.eeg_sample_rate_hz as u32;
        let fnirs_rate = self.fnirs.sample_rate_hz as u32;

        // EEG: channels × 24-bit × sample_rate
        let eeg_bps = eeg_channels * 24 * eeg_rate;

        // fNIRS: channels × 2 wavelengths × 16-bit × sample_rate
        let fnirs_bps = fnirs_channels * 2 * 16 * fnirs_rate;

        eeg_bps + fnirs_bps
    }

    /// Check if configuration requires USB 2.0 (vs USB 1.1).
    #[must_use]
    pub const fn requires_usb2(&self) -> bool {
        // USB 1.1 Full Speed: 12 Mbps (practical ~8 Mbps)
        self.data_rate_bps() > 8_000_000
    }

    /// Get recommended buffer size in samples.
    #[must_use]
    pub const fn buffer_size_samples(&self) -> usize {
        // 100ms buffer
        (self.eeg_sample_rate_hz as usize) / 10
    }

    /// Basic 8-channel configuration.
    pub const BASIC_8: Self = Self::new(ChannelDensity::Basic8);

    /// Standard 32-channel configuration.
    pub const STANDARD_32: Self = Self::new(ChannelDensity::Standard32);

    /// High-density 128-channel configuration.
    pub const HIGH_DENSITY_128: Self = Self::new(ChannelDensity::HighDensity128);

    /// Ultra high-density 256-channel configuration.
    pub const ULTRA_HIGH_DENSITY_256: Self = Self::new(ChannelDensity::UltraHighDensity256);
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self::BASIC_8
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SystemConfig {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "Config({}, {}ch EEG @ {}Hz, {} fNIRS)",
            self.density.name(),
            self.density.eeg_channels(),
            self.eeg_sample_rate_hz,
            self.fnirs.total_channels()
        );
    }
}

// ============================================================================
// Synchronization Protocol
// ============================================================================

/// Time synchronization between EEG and fNIRS subsystems.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncProtocol {
    /// Master clock source frequency (EEG ADC clock)
    pub master_clock_hz: u32,
    /// fNIRS sampling trigger divisor
    pub fnirs_trigger_divisor: u32,
    /// Timestamp resolution in microseconds
    pub timestamp_resolution_us: u32,
    /// Maximum allowed clock drift before resync
    pub max_drift_us: u32,
}

impl SyncProtocol {
    /// Calculate fNIRS sample rate.
    #[inline]
    #[must_use]
    pub const fn fnirs_sample_rate(&self) -> u32 {
        self.master_clock_hz / self.fnirs_trigger_divisor
    }

    /// Check if drift exceeds threshold.
    #[inline]
    #[must_use]
    pub const fn needs_resync(&self, current_drift_us: u32) -> bool {
        current_drift_us > self.max_drift_us
    }
}

impl Default for SyncProtocol {
    fn default() -> Self {
        Self {
            master_clock_hz: 2_048_000,     // 2.048 MHz master clock
            fnirs_trigger_divisor: 81920,   // Results in 25 Hz fNIRS
            timestamp_resolution_us: 1,
            max_drift_us: 100,
        }
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SyncProtocol {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "Sync(master={}Hz, fNIRS={}Hz)",
            self.master_clock_hz,
            self.fnirs_sample_rate()
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_density_values() {
        assert_eq!(ChannelDensity::Basic8.eeg_channels(), 8);
        assert_eq!(ChannelDensity::Basic8.fnirs_channels(), 4);
        assert_eq!(ChannelDensity::Basic8.ads1299_count(), 1);

        assert_eq!(ChannelDensity::HighDensity128.eeg_channels(), 128);
        assert_eq!(ChannelDensity::HighDensity128.ads1299_count(), 16);
    }

    #[test]
    fn test_custom_density() {
        let custom = ChannelDensity::Custom {
            eeg_channels: 96,
            fnirs_channels: 48,
        };
        assert_eq!(custom.eeg_channels(), 96);
        assert_eq!(custom.fnirs_channels(), 48);
        assert_eq!(custom.ads1299_count(), 12); // (96 + 7) / 8 = 12
    }

    #[test]
    fn test_fnirs_config() {
        let config = FnirsConfig::BASIC;
        assert_eq!(config.num_sources, 2);
        assert_eq!(config.penetration_depth_mm(), 15);
    }

    #[test]
    fn test_system_config_data_rate() {
        let basic = SystemConfig::BASIC_8;
        // 8 ch × 24 bit × 500 Hz = 96,000 bps EEG
        // ~4 ch × 2 × 16 bit × 10 Hz = 1,280 bps fNIRS
        let rate = basic.data_rate_bps();
        assert!(rate > 90_000);
        assert!(rate < 200_000);
        assert!(!basic.requires_usb2());

        let hd = SystemConfig::HIGH_DENSITY_128;
        let rate = hd.data_rate_bps();
        // 128 × 24 × 500 = 1,536,000 bps EEG
        assert!(rate > 1_000_000);
    }

    #[test]
    fn test_sync_protocol() {
        let sync = SyncProtocol::default();
        assert_eq!(sync.fnirs_sample_rate(), 25);
        assert!(!sync.needs_resync(50));
        assert!(sync.needs_resync(150));
    }
}

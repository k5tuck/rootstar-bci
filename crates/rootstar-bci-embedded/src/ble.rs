//! BLE Peripheral for BCI Data Streaming
//!
//! Production-ready BLE GATT server for ESP32 BCI devices.
//! Provides real-time streaming of EEG, fNIRS, EMG, and EDA data.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    BLE Peripheral                       │
//! ├─────────────────────────────────────────────────────────┤
//! │  Generic Access Service (0x1800)                        │
//! │  ├── Device Name (0x2A00)                               │
//! │  └── Appearance (0x2A01)                                │
//! ├─────────────────────────────────────────────────────────┤
//! │  Device Information Service (0x180A)                    │
//! │  ├── Manufacturer Name (0x2A29)                         │
//! │  ├── Model Number (0x2A24)                              │
//! │  ├── Serial Number (0x2A25)                             │
//! │  ├── Firmware Revision (0x2A26)                         │
//! │  └── Hardware Revision (0x2A27)                         │
//! ├─────────────────────────────────────────────────────────┤
//! │  Battery Service (0x180F)                               │
//! │  └── Battery Level (0x2A19) [Read, Notify]              │
//! ├─────────────────────────────────────────────────────────┤
//! │  BCI Data Service (Custom)                              │
//! │  ├── EEG Data (Notify)         - 8ch × 24-bit @ 250Hz   │
//! │  ├── fNIRS Data (Notify)       - 4ch HbO/HbR @ 25Hz     │
//! │  ├── EMG Data (Notify)         - Envelope + features    │
//! │  ├── EDA Data (Notify)         - SCL + SCR              │
//! │  ├── Impedance Data (Notify)   - Per-channel impedance  │
//! │  ├── Command (Write)           - Control commands       │
//! │  ├── Status (Read, Notify)     - Device state           │
//! │  └── Config (Read, Write)      - Configuration          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Data Rates
//!
//! With BLE 5.0 and 2M PHY:
//! - EEG: 8ch × 3 bytes × 250 Hz = 6000 bytes/sec (fits in ~30 packets/sec)
//! - fNIRS: 16 bytes × 25 Hz = 400 bytes/sec
//! - EMG: 8 bytes × 100 Hz = 800 bytes/sec
//! - EDA: 5 bytes × 10 Hz = 50 bytes/sec
//! - Total: ~7.3 KB/sec (well within BLE 5.0 capacity)

use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU8, Ordering};
use heapless::{String, Vec};

// ============================================================================
// Service and Characteristic UUIDs
// ============================================================================

/// BCI Data Service UUID (custom 128-bit)
pub const BCI_SERVICE_UUID: u128 = 0x12340001_1234_5678_9abc_def012345678;

/// EEG Data Characteristic - 8 channels × 24-bit samples
pub const EEG_DATA_CHAR_UUID: u128 = 0x12340002_1234_5678_9abc_def012345678;

/// fNIRS Data Characteristic - HbO/HbR concentrations
pub const FNIRS_DATA_CHAR_UUID: u128 = 0x12340003_1234_5678_9abc_def012345678;

/// Command Characteristic - Control interface
pub const COMMAND_CHAR_UUID: u128 = 0x12340004_1234_5678_9abc_def012345678;

/// Status Characteristic - Device state
pub const STATUS_CHAR_UUID: u128 = 0x12340005_1234_5678_9abc_def012345678;

/// EMG Data Characteristic - Muscle activity
pub const EMG_DATA_CHAR_UUID: u128 = 0x12340006_1234_5678_9abc_def012345678;

/// EDA Data Characteristic - Skin conductance
pub const EDA_DATA_CHAR_UUID: u128 = 0x12340007_1234_5678_9abc_def012345678;

/// Impedance Data Characteristic - Electrode impedance
pub const IMPEDANCE_CHAR_UUID: u128 = 0x12340008_1234_5678_9abc_def012345678;

/// Configuration Characteristic - Device settings
pub const CONFIG_CHAR_UUID: u128 = 0x12340009_1234_5678_9abc_def012345678;

/// Stimulation Control Characteristic - tDCS/tACS control
pub const STIM_CHAR_UUID: u128 = 0x1234000A_1234_5678_9abc_def012345678;

/// Sync Marker Characteristic - For hyperscanning
pub const SYNC_CHAR_UUID: u128 = 0x1234000B_1234_5678_9abc_def012345678;

// Standard service UUIDs (16-bit)
/// Generic Access Service
pub const GAP_SERVICE_UUID: u16 = 0x1800;
/// Device Information Service
pub const DIS_SERVICE_UUID: u16 = 0x180A;
/// Battery Service
pub const BATTERY_SERVICE_UUID: u16 = 0x180F;

// ============================================================================
// Packet Sizes and Constraints
// ============================================================================

/// Maximum EEG packet size (8 channels × 3 bytes + 2 byte header)
pub const EEG_PACKET_SIZE: usize = 26;

/// Maximum fNIRS packet size (4ch HbO + 4ch HbR × 2 bytes + 2 byte header)
pub const FNIRS_PACKET_SIZE: usize = 18;

/// EMG packet size (envelope + 4 band powers + valence)
pub const EMG_PACKET_SIZE: usize = 12;

/// EDA packet size (SCL + SCR amplitude + rise time + recovery)
pub const EDA_PACKET_SIZE: usize = 10;

/// Impedance packet size (8 channels × 2 bytes + flags)
pub const IMPEDANCE_PACKET_SIZE: usize = 18;

/// Status packet size
pub const STATUS_PACKET_SIZE: usize = 16;

/// Configuration packet size
pub const CONFIG_PACKET_SIZE: usize = 32;

/// Maximum BLE MTU (negotiated, default 23, max 517)
pub const DEFAULT_MTU: usize = 23;
/// Preferred MTU for high-throughput data
pub const PREFERRED_MTU: usize = 247;

/// Maximum notification queue depth
pub const NOTIFY_QUEUE_DEPTH: usize = 16;

// ============================================================================
// Command Protocol
// ============================================================================

/// BLE command identifiers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Command {
    // Acquisition control (0x01-0x0F)
    /// Start data acquisition
    StartAcquisition = 0x01,
    /// Stop data acquisition
    StopAcquisition = 0x02,
    /// Set sample rate: [rate_h, rate_l]
    SetSampleRate = 0x03,
    /// Set channel gain: [channel, gain_code]
    SetGain = 0x04,
    /// Set channel mask: [mask_h, mask_l]
    SetChannelMask = 0x05,
    /// Trigger single sample
    TriggerSample = 0x06,

    // Calibration (0x10-0x1F)
    /// Start impedance measurement
    StartImpedance = 0x10,
    /// Stop impedance measurement
    StopImpedance = 0x11,
    /// Start signal calibration
    StartCalibration = 0x12,
    /// Stop calibration
    StopCalibration = 0x13,
    /// Set reference channel: [channel]
    SetReference = 0x14,

    // Stimulation (0x20-0x2F)
    /// Start stimulation: [protocol_id, params...]
    StartStimulation = 0x20,
    /// Stop stimulation
    StopStimulation = 0x21,
    /// Set stimulation amplitude: [amp_h, amp_l] (microamps)
    SetStimAmplitude = 0x22,
    /// Set stimulation frequency: [freq_h, freq_l] (0.01 Hz units)
    SetStimFrequency = 0x23,
    /// Set stimulation duration: [dur_h, dur_l] (seconds)
    SetStimDuration = 0x24,
    /// Set electrode montage: [anode_mask, cathode_mask]
    SetMontage = 0x25,

    // Synchronization (0x30-0x3F)
    /// Send sync pulse (hyperscanning)
    SendSyncPulse = 0x30,
    /// Set device ID: [id_bytes...]
    SetDeviceId = 0x31,
    /// Request timestamp sync
    RequestTimeSync = 0x32,
    /// Set group ID (for multi-device): [group_id]
    SetGroupId = 0x33,

    // System (0xF0-0xFF)
    /// Request status update
    RequestStatus = 0xF0,
    /// Request device info
    RequestDeviceInfo = 0xF1,
    /// Set device name: [name_bytes...]
    SetDeviceName = 0xF2,
    /// Save configuration to flash
    SaveConfig = 0xF3,
    /// Reset to factory defaults
    FactoryReset = 0xF4,
    /// Enter DFU bootloader
    EnterBootloader = 0xFE,
    /// Software reset
    SoftReset = 0xFF,
}

impl Command {
    /// Parse command from byte
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::StartAcquisition),
            0x02 => Some(Self::StopAcquisition),
            0x03 => Some(Self::SetSampleRate),
            0x04 => Some(Self::SetGain),
            0x05 => Some(Self::SetChannelMask),
            0x06 => Some(Self::TriggerSample),
            0x10 => Some(Self::StartImpedance),
            0x11 => Some(Self::StopImpedance),
            0x12 => Some(Self::StartCalibration),
            0x13 => Some(Self::StopCalibration),
            0x14 => Some(Self::SetReference),
            0x20 => Some(Self::StartStimulation),
            0x21 => Some(Self::StopStimulation),
            0x22 => Some(Self::SetStimAmplitude),
            0x23 => Some(Self::SetStimFrequency),
            0x24 => Some(Self::SetStimDuration),
            0x25 => Some(Self::SetMontage),
            0x30 => Some(Self::SendSyncPulse),
            0x31 => Some(Self::SetDeviceId),
            0x32 => Some(Self::RequestTimeSync),
            0x33 => Some(Self::SetGroupId),
            0xF0 => Some(Self::RequestStatus),
            0xF1 => Some(Self::RequestDeviceInfo),
            0xF2 => Some(Self::SetDeviceName),
            0xF3 => Some(Self::SaveConfig),
            0xF4 => Some(Self::FactoryReset),
            0xFE => Some(Self::EnterBootloader),
            0xFF => Some(Self::SoftReset),
            _ => None,
        }
    }

    /// Get expected parameter length
    #[must_use]
    pub const fn param_length(&self) -> usize {
        match self {
            Self::StartAcquisition | Self::StopAcquisition => 0,
            Self::SetSampleRate => 2,
            Self::SetGain => 2,
            Self::SetChannelMask => 2,
            Self::TriggerSample => 0,
            Self::StartImpedance | Self::StopImpedance => 0,
            Self::StartCalibration | Self::StopCalibration => 0,
            Self::SetReference => 1,
            Self::StartStimulation => 8, // protocol + params
            Self::StopStimulation => 0,
            Self::SetStimAmplitude => 2,
            Self::SetStimFrequency => 2,
            Self::SetStimDuration => 2,
            Self::SetMontage => 2,
            Self::SendSyncPulse => 0,
            Self::SetDeviceId => 4,
            Self::RequestTimeSync => 0,
            Self::SetGroupId => 1,
            Self::RequestStatus | Self::RequestDeviceInfo => 0,
            Self::SetDeviceName => 0, // Variable length
            Self::SaveConfig | Self::FactoryReset => 0,
            Self::EnterBootloader | Self::SoftReset => 0,
        }
    }
}

/// Command response codes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CommandResponse {
    /// Command executed successfully
    Success = 0x00,
    /// Unknown command
    UnknownCommand = 0x01,
    /// Invalid parameters
    InvalidParams = 0x02,
    /// Command not allowed in current state
    NotAllowed = 0x03,
    /// Hardware error during execution
    HardwareError = 0x04,
    /// Resource busy
    Busy = 0x05,
    /// Safety limit exceeded
    SafetyLimit = 0x06,
    /// Not implemented
    NotImplemented = 0x07,
}

// ============================================================================
// Device State Machine
// ============================================================================

/// Device operating state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DeviceState {
    /// Initial state, not connected
    #[default]
    Disconnected = 0x00,
    /// Connected but idle
    Idle = 0x01,
    /// Acquiring EEG/fNIRS data
    Acquiring = 0x02,
    /// Running impedance check
    ImpedanceCheck = 0x03,
    /// Calibrating
    Calibrating = 0x04,
    /// Stimulating (tDCS/tACS)
    Stimulating = 0x05,
    /// Acquiring + Stimulating (closed-loop)
    ClosedLoop = 0x06,
    /// Error state
    Error = 0xFF,
}

impl DeviceState {
    /// Parse from byte
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0x00 => Self::Disconnected,
            0x01 => Self::Idle,
            0x02 => Self::Acquiring,
            0x03 => Self::ImpedanceCheck,
            0x04 => Self::Calibrating,
            0x05 => Self::Stimulating,
            0x06 => Self::ClosedLoop,
            _ => Self::Error,
        }
    }

    /// Check if acquisition is active
    #[must_use]
    pub const fn is_acquiring(&self) -> bool {
        matches!(self, Self::Acquiring | Self::ClosedLoop)
    }

    /// Check if stimulation is active
    #[must_use]
    pub const fn is_stimulating(&self) -> bool {
        matches!(self, Self::Stimulating | Self::ClosedLoop)
    }
}

// ============================================================================
// Status Flags
// ============================================================================

/// Status flag bits
pub mod status_flags {
    /// Battery level low (<20%)
    pub const BATTERY_LOW: u16 = 1 << 0;
    /// Battery critical (<5%)
    pub const BATTERY_CRITICAL: u16 = 1 << 1;
    /// Charging
    pub const CHARGING: u16 = 1 << 2;
    /// Electrode contact issue (any channel)
    pub const ELECTRODE_OFF: u16 = 1 << 3;
    /// Signal saturation detected
    pub const SIGNAL_SATURATED: u16 = 1 << 4;
    /// Stimulation active
    pub const STIM_ACTIVE: u16 = 1 << 5;
    /// Impedance out of range
    pub const IMPEDANCE_HIGH: u16 = 1 << 6;
    /// Recording to SD card
    pub const RECORDING: u16 = 1 << 7;
    /// External sync received
    pub const SYNC_RECEIVED: u16 = 1 << 8;
    /// Hardware self-test failed
    pub const SELF_TEST_FAIL: u16 = 1 << 9;
    /// Temperature warning
    pub const TEMP_WARNING: u16 = 1 << 10;
    /// DFU mode available
    pub const DFU_AVAILABLE: u16 = 1 << 11;
    /// Configuration unsaved
    pub const CONFIG_DIRTY: u16 = 1 << 12;
}

// ============================================================================
// Data Packets
// ============================================================================

/// EEG data packet (sent at sample rate, e.g., 250 Hz)
#[derive(Clone, Copy, Debug)]
pub struct EegPacket {
    /// Sequence number (wraps at 65535)
    pub sequence: u16,
    /// 8 channels of 24-bit signed samples
    pub channels: [i32; 8],
}

impl EegPacket {
    /// Packet size in bytes
    pub const SIZE: usize = EEG_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; EEG_PACKET_SIZE] {
        let mut buf = [0u8; EEG_PACKET_SIZE];
        buf[0] = (self.sequence >> 8) as u8;
        buf[1] = (self.sequence & 0xFF) as u8;

        for (i, &sample) in self.channels.iter().enumerate() {
            let offset = 2 + i * 3;
            buf[offset] = ((sample >> 16) & 0xFF) as u8;
            buf[offset + 1] = ((sample >> 8) & 0xFF) as u8;
            buf[offset + 2] = (sample & 0xFF) as u8;
        }
        buf
    }

    /// Deserialize from bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; EEG_PACKET_SIZE]) -> Self {
        let sequence = ((buf[0] as u16) << 8) | (buf[1] as u16);
        let mut channels = [0i32; 8];

        for i in 0..8 {
            let offset = 2 + i * 3;
            let raw = ((buf[offset] as i32) << 16)
                | ((buf[offset + 1] as i32) << 8)
                | (buf[offset + 2] as i32);
            // Sign-extend from 24-bit
            channels[i] = if raw & 0x800000 != 0 {
                raw | (0xFF << 24)
            } else {
                raw
            };
        }

        Self { sequence, channels }
    }
}

/// fNIRS data packet (sent at ~25 Hz)
#[derive(Clone, Copy, Debug)]
pub struct FnirsPacket {
    /// Sequence number
    pub sequence: u16,
    /// HbO concentration (4 channels, micromolar × 100)
    pub hbo: [i16; 4],
    /// HbR concentration (4 channels, micromolar × 100)
    pub hbr: [i16; 4],
}

impl FnirsPacket {
    /// Packet size in bytes
    pub const SIZE: usize = FNIRS_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; FNIRS_PACKET_SIZE] {
        let mut buf = [0u8; FNIRS_PACKET_SIZE];
        buf[0] = (self.sequence >> 8) as u8;
        buf[1] = (self.sequence & 0xFF) as u8;

        for (i, &val) in self.hbo.iter().enumerate() {
            let offset = 2 + i * 2;
            buf[offset] = (val >> 8) as u8;
            buf[offset + 1] = (val & 0xFF) as u8;
        }

        for (i, &val) in self.hbr.iter().enumerate() {
            let offset = 10 + i * 2;
            buf[offset] = (val >> 8) as u8;
            buf[offset + 1] = (val & 0xFF) as u8;
        }

        buf
    }

    /// Deserialize from bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; FNIRS_PACKET_SIZE]) -> Self {
        let sequence = ((buf[0] as u16) << 8) | (buf[1] as u16);
        let mut hbo = [0i16; 4];
        let mut hbr = [0i16; 4];

        for i in 0..4 {
            let offset = 2 + i * 2;
            hbo[i] = ((buf[offset] as i16) << 8) | (buf[offset + 1] as i16);
        }

        for i in 0..4 {
            let offset = 10 + i * 2;
            hbr[i] = ((buf[offset] as i16) << 8) | (buf[offset + 1] as i16);
        }

        Self { sequence, hbo, hbr }
    }

    /// Convert HbO/HbR from i16 to micromolar
    #[must_use]
    pub fn hbo_micromolar(&self, channel: usize) -> f32 {
        self.hbo.get(channel).map(|&v| v as f32 / 100.0).unwrap_or(0.0)
    }

    /// Convert HbO/HbR from i16 to micromolar
    #[must_use]
    pub fn hbr_micromolar(&self, channel: usize) -> f32 {
        self.hbr.get(channel).map(|&v| v as f32 / 100.0).unwrap_or(0.0)
    }
}

/// EMG data packet (sent at ~100 Hz)
#[derive(Clone, Copy, Debug)]
pub struct EmgPacket {
    /// Sequence number
    pub sequence: u16,
    /// RMS envelope (microvolts)
    pub envelope_uv: u16,
    /// Mean frequency (Hz × 10)
    pub mean_freq_hz10: u16,
    /// Median frequency (Hz × 10)
    pub median_freq_hz10: u16,
    /// Band powers (delta, theta, alpha, beta) as percentages
    pub band_powers: [u8; 4],
    /// Valence estimate (-100 to +100)
    pub valence: i8,
    /// Reserved
    pub _reserved: u8,
}

impl EmgPacket {
    /// Packet size in bytes
    pub const SIZE: usize = EMG_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; EMG_PACKET_SIZE] {
        [
            (self.sequence >> 8) as u8,
            (self.sequence & 0xFF) as u8,
            (self.envelope_uv >> 8) as u8,
            (self.envelope_uv & 0xFF) as u8,
            (self.mean_freq_hz10 >> 8) as u8,
            (self.mean_freq_hz10 & 0xFF) as u8,
            (self.median_freq_hz10 >> 8) as u8,
            (self.median_freq_hz10 & 0xFF) as u8,
            self.band_powers[0],
            self.band_powers[1],
            self.valence as u8,
            self._reserved,
        ]
    }
}

/// EDA data packet (sent at ~10 Hz)
#[derive(Clone, Copy, Debug)]
pub struct EdaPacket {
    /// Sequence number
    pub sequence: u16,
    /// Skin conductance level (microsiemens × 100)
    pub scl_us100: u16,
    /// SCR amplitude (microsiemens × 100)
    pub scr_amplitude_us100: u16,
    /// SCR rise time (milliseconds)
    pub scr_rise_time_ms: u16,
    /// SCR recovery half-time (milliseconds)
    pub scr_recovery_ms: u16,
}

impl EdaPacket {
    /// Packet size in bytes
    pub const SIZE: usize = EDA_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; EDA_PACKET_SIZE] {
        [
            (self.sequence >> 8) as u8,
            (self.sequence & 0xFF) as u8,
            (self.scl_us100 >> 8) as u8,
            (self.scl_us100 & 0xFF) as u8,
            (self.scr_amplitude_us100 >> 8) as u8,
            (self.scr_amplitude_us100 & 0xFF) as u8,
            (self.scr_rise_time_ms >> 8) as u8,
            (self.scr_rise_time_ms & 0xFF) as u8,
            (self.scr_recovery_ms >> 8) as u8,
            (self.scr_recovery_ms & 0xFF) as u8,
        ]
    }
}

/// Impedance measurement packet
#[derive(Clone, Copy, Debug)]
pub struct ImpedancePacket {
    /// Sequence number
    pub sequence: u16,
    /// Per-channel impedance (kOhm, 0xFFFF = open/unmeasured)
    pub impedance_kohm: [u16; 8],
}

impl ImpedancePacket {
    /// Packet size in bytes
    pub const SIZE: usize = IMPEDANCE_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; IMPEDANCE_PACKET_SIZE] {
        let mut buf = [0u8; IMPEDANCE_PACKET_SIZE];
        buf[0] = (self.sequence >> 8) as u8;
        buf[1] = (self.sequence & 0xFF) as u8;

        for (i, &imp) in self.impedance_kohm.iter().enumerate() {
            let offset = 2 + i * 2;
            buf[offset] = (imp >> 8) as u8;
            buf[offset + 1] = (imp & 0xFF) as u8;
        }
        buf
    }
}

/// Device status packet
#[derive(Clone, Copy, Debug, Default)]
pub struct StatusPacket {
    /// Current device state
    pub state: DeviceState,
    /// Status flags
    pub flags: u16,
    /// Battery percentage (0-100)
    pub battery_percent: u8,
    /// Current sample rate (Hz)
    pub sample_rate_hz: u16,
    /// Active channel mask
    pub channel_mask: u8,
    /// Current gain setting
    pub gain: u8,
    /// Stimulation amplitude (if active, microamps)
    pub stim_amplitude_ua: u16,
    /// Uptime in seconds
    pub uptime_sec: u32,
    /// Packet sequence
    pub sequence: u16,
}

impl StatusPacket {
    /// Packet size in bytes
    pub const SIZE: usize = STATUS_PACKET_SIZE;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; STATUS_PACKET_SIZE] {
        [
            self.state as u8,
            (self.flags >> 8) as u8,
            (self.flags & 0xFF) as u8,
            self.battery_percent,
            (self.sample_rate_hz >> 8) as u8,
            (self.sample_rate_hz & 0xFF) as u8,
            self.channel_mask,
            self.gain,
            (self.stim_amplitude_ua >> 8) as u8,
            (self.stim_amplitude_ua & 0xFF) as u8,
            ((self.uptime_sec >> 24) & 0xFF) as u8,
            ((self.uptime_sec >> 16) & 0xFF) as u8,
            ((self.uptime_sec >> 8) & 0xFF) as u8,
            (self.uptime_sec & 0xFF) as u8,
            (self.sequence >> 8) as u8,
            (self.sequence & 0xFF) as u8,
        ]
    }

    /// Deserialize from bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; STATUS_PACKET_SIZE]) -> Self {
        Self {
            state: DeviceState::from_u8(buf[0]),
            flags: ((buf[1] as u16) << 8) | (buf[2] as u16),
            battery_percent: buf[3],
            sample_rate_hz: ((buf[4] as u16) << 8) | (buf[5] as u16),
            channel_mask: buf[6],
            gain: buf[7],
            stim_amplitude_ua: ((buf[8] as u16) << 8) | (buf[9] as u16),
            uptime_sec: ((buf[10] as u32) << 24)
                | ((buf[11] as u32) << 16)
                | ((buf[12] as u32) << 8)
                | (buf[13] as u32),
            sequence: ((buf[14] as u16) << 8) | (buf[15] as u16),
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Device configuration (stored in flash)
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// Device name (max 20 chars)
    pub name: String<20>,
    /// Device ID (4 bytes)
    pub device_id: [u8; 4],
    /// Group ID (for hyperscanning)
    pub group_id: u8,
    /// Default sample rate (Hz)
    pub sample_rate_hz: u16,
    /// Default gain setting
    pub default_gain: u8,
    /// Active channel mask
    pub channel_mask: u8,
    /// Enable auto-reconnect
    pub auto_reconnect: bool,
    /// Enable low-latency mode
    pub low_latency: bool,
    /// Notification interval (ms, 0 = every sample)
    pub notify_interval_ms: u8,
    /// Stimulation safety limit (microamps)
    pub stim_limit_ua: u16,
    /// Impedance threshold (kOhm)
    pub impedance_threshold_kohm: u16,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            name: String::try_from("RootstarBCI").unwrap_or_default(),
            device_id: [0x00, 0x00, 0x00, 0x01],
            group_id: 0,
            sample_rate_hz: 250,
            default_gain: 24, // 24x gain
            channel_mask: 0xFF, // All 8 channels
            auto_reconnect: true,
            low_latency: false,
            notify_interval_ms: 0,
            stim_limit_ua: 2000, // 2mA safety limit
            impedance_threshold_kohm: 50,
        }
    }
}

impl DeviceConfig {
    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; CONFIG_PACKET_SIZE] {
        let mut buf = [0u8; CONFIG_PACKET_SIZE];

        // Name (bytes 0-19)
        let name_bytes = self.name.as_bytes();
        let len = name_bytes.len().min(20);
        buf[..len].copy_from_slice(&name_bytes[..len]);

        // Device ID (bytes 20-23)
        buf[20..24].copy_from_slice(&self.device_id);

        // Other settings
        buf[24] = self.group_id;
        buf[25] = (self.sample_rate_hz >> 8) as u8;
        buf[26] = (self.sample_rate_hz & 0xFF) as u8;
        buf[27] = self.default_gain;
        buf[28] = self.channel_mask;

        let mut flags = 0u8;
        if self.auto_reconnect { flags |= 0x01; }
        if self.low_latency { flags |= 0x02; }
        buf[29] = flags;

        buf[30] = self.notify_interval_ms;
        // Byte 31 reserved

        buf
    }
}

// ============================================================================
// BLE Peripheral State
// ============================================================================

/// Shared state for BLE peripheral (lock-free atomics for ISR safety)
pub struct BlePeripheralState {
    /// Current device state
    state: AtomicU8,
    /// Status flags
    flags: AtomicU16,
    /// Battery percentage
    battery: AtomicU8,
    /// Connection handle (0 = disconnected)
    conn_handle: AtomicU16,
    /// Negotiated MTU
    mtu: AtomicU16,
    /// EEG notification enabled
    eeg_notify_enabled: AtomicBool,
    /// fNIRS notification enabled
    fnirs_notify_enabled: AtomicBool,
    /// EMG notification enabled
    emg_notify_enabled: AtomicBool,
    /// EDA notification enabled
    eda_notify_enabled: AtomicBool,
    /// Status notification enabled
    status_notify_enabled: AtomicBool,
    /// Sequence counters
    eeg_sequence: AtomicU16,
    fnirs_sequence: AtomicU16,
    emg_sequence: AtomicU16,
    eda_sequence: AtomicU16,
    /// Uptime counter (seconds)
    uptime: AtomicU32,
    /// Packets sent
    packets_sent: AtomicU32,
    /// Packets dropped (queue full)
    packets_dropped: AtomicU32,
}

impl BlePeripheralState {
    /// Create new peripheral state
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(DeviceState::Disconnected as u8),
            flags: AtomicU16::new(0),
            battery: AtomicU8::new(100),
            conn_handle: AtomicU16::new(0),
            mtu: AtomicU16::new(DEFAULT_MTU as u16),
            eeg_notify_enabled: AtomicBool::new(false),
            fnirs_notify_enabled: AtomicBool::new(false),
            emg_notify_enabled: AtomicBool::new(false),
            eda_notify_enabled: AtomicBool::new(false),
            status_notify_enabled: AtomicBool::new(false),
            eeg_sequence: AtomicU16::new(0),
            fnirs_sequence: AtomicU16::new(0),
            emg_sequence: AtomicU16::new(0),
            eda_sequence: AtomicU16::new(0),
            uptime: AtomicU32::new(0),
            packets_sent: AtomicU32::new(0),
            packets_dropped: AtomicU32::new(0),
        }
    }

    /// Get current device state
    #[must_use]
    pub fn device_state(&self) -> DeviceState {
        DeviceState::from_u8(self.state.load(Ordering::Relaxed))
    }

    /// Set device state
    pub fn set_device_state(&self, state: DeviceState) {
        self.state.store(state as u8, Ordering::Relaxed);
    }

    /// Check if connected
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.conn_handle.load(Ordering::Relaxed) != 0
    }

    /// Set connection handle (0 = disconnected)
    pub fn set_connection(&self, handle: u16) {
        self.conn_handle.store(handle, Ordering::Relaxed);
        if handle == 0 {
            self.set_device_state(DeviceState::Disconnected);
            // Disable all notifications
            self.eeg_notify_enabled.store(false, Ordering::Relaxed);
            self.fnirs_notify_enabled.store(false, Ordering::Relaxed);
            self.emg_notify_enabled.store(false, Ordering::Relaxed);
            self.eda_notify_enabled.store(false, Ordering::Relaxed);
            self.status_notify_enabled.store(false, Ordering::Relaxed);
        } else {
            self.set_device_state(DeviceState::Idle);
        }
    }

    /// Get negotiated MTU
    #[must_use]
    pub fn mtu(&self) -> usize {
        self.mtu.load(Ordering::Relaxed) as usize
    }

    /// Set negotiated MTU
    pub fn set_mtu(&self, mtu: u16) {
        self.mtu.store(mtu, Ordering::Relaxed);
    }

    /// Check if EEG notifications enabled
    #[must_use]
    pub fn eeg_notify_enabled(&self) -> bool {
        self.eeg_notify_enabled.load(Ordering::Relaxed)
    }

    /// Enable/disable EEG notifications
    pub fn set_eeg_notify(&self, enabled: bool) {
        self.eeg_notify_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if fNIRS notifications enabled
    #[must_use]
    pub fn fnirs_notify_enabled(&self) -> bool {
        self.fnirs_notify_enabled.load(Ordering::Relaxed)
    }

    /// Enable/disable fNIRS notifications
    pub fn set_fnirs_notify(&self, enabled: bool) {
        self.fnirs_notify_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get and increment EEG sequence number
    pub fn next_eeg_sequence(&self) -> u16 {
        self.eeg_sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Get and increment fNIRS sequence number
    pub fn next_fnirs_sequence(&self) -> u16 {
        self.fnirs_sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Increment uptime (call once per second)
    pub fn tick_uptime(&self) {
        self.uptime.fetch_add(1, Ordering::Relaxed);
    }

    /// Get uptime in seconds
    #[must_use]
    pub fn uptime_sec(&self) -> u32 {
        self.uptime.load(Ordering::Relaxed)
    }

    /// Record packet sent
    pub fn record_packet_sent(&self) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record packet dropped
    pub fn record_packet_dropped(&self) {
        self.packets_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Set battery percentage
    pub fn set_battery(&self, percent: u8) {
        self.battery.store(percent.min(100), Ordering::Relaxed);
    }

    /// Get battery percentage
    #[must_use]
    pub fn battery(&self) -> u8 {
        self.battery.load(Ordering::Relaxed)
    }

    /// Set status flag
    pub fn set_flag(&self, flag: u16) {
        self.flags.fetch_or(flag, Ordering::Relaxed);
    }

    /// Clear status flag
    pub fn clear_flag(&self, flag: u16) {
        self.flags.fetch_and(!flag, Ordering::Relaxed);
    }

    /// Get all status flags
    #[must_use]
    pub fn flags(&self) -> u16 {
        self.flags.load(Ordering::Relaxed)
    }

    /// Build status packet
    #[must_use]
    pub fn build_status(&self, config: &DeviceConfig) -> StatusPacket {
        StatusPacket {
            state: self.device_state(),
            flags: self.flags(),
            battery_percent: self.battery(),
            sample_rate_hz: config.sample_rate_hz,
            channel_mask: config.channel_mask,
            gain: config.default_gain,
            stim_amplitude_ua: 0, // TODO: from stim controller
            uptime_sec: self.uptime_sec(),
            sequence: 0,
        }
    }
}

impl Default for BlePeripheralState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Pack 8 channels of 24-bit EEG data
#[must_use]
pub fn pack_eeg_samples(samples: &[i32; 8]) -> [u8; 24] {
    let mut buf = [0u8; 24];
    for (i, &sample) in samples.iter().enumerate() {
        let offset = i * 3;
        buf[offset] = ((sample >> 16) & 0xFF) as u8;
        buf[offset + 1] = ((sample >> 8) & 0xFF) as u8;
        buf[offset + 2] = (sample & 0xFF) as u8;
    }
    buf
}

/// Unpack 8 channels of 24-bit EEG data
#[must_use]
pub fn unpack_eeg_samples(buf: &[u8; 24]) -> [i32; 8] {
    let mut samples = [0i32; 8];
    for i in 0..8 {
        let offset = i * 3;
        let raw = ((buf[offset] as i32) << 16)
            | ((buf[offset + 1] as i32) << 8)
            | (buf[offset + 2] as i32);
        samples[i] = if raw & 0x800000 != 0 {
            raw | (0xFF << 24)
        } else {
            raw
        };
    }
    samples
}

/// Pack fNIRS HbO/HbR values (f32 micromolar to i16)
#[must_use]
pub fn pack_fnirs_samples(hbo: &[f32; 4], hbr: &[f32; 4]) -> [u8; 16] {
    let mut buf = [0u8; 16];

    for (i, &val) in hbo.iter().enumerate() {
        let fixed = (val * 100.0).clamp(-32768.0, 32767.0) as i16;
        buf[i * 2] = (fixed >> 8) as u8;
        buf[i * 2 + 1] = (fixed & 0xFF) as u8;
    }

    for (i, &val) in hbr.iter().enumerate() {
        let fixed = (val * 100.0).clamp(-32768.0, 32767.0) as i16;
        buf[8 + i * 2] = (fixed >> 8) as u8;
        buf[8 + i * 2 + 1] = (fixed & 0xFF) as u8;
    }

    buf
}

/// Unpack fNIRS values to f32 micromolar
#[must_use]
pub fn unpack_fnirs_samples(buf: &[u8; 16]) -> ([f32; 4], [f32; 4]) {
    let mut hbo = [0.0f32; 4];
    let mut hbr = [0.0f32; 4];

    for i in 0..4 {
        let fixed = ((buf[i * 2] as i16) << 8) | (buf[i * 2 + 1] as i16);
        hbo[i] = fixed as f32 / 100.0;
    }

    for i in 0..4 {
        let fixed = ((buf[8 + i * 2] as i16) << 8) | (buf[8 + i * 2 + 1] as i16);
        hbr[i] = fixed as f32 / 100.0;
    }

    (hbo, hbr)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_packet_roundtrip() {
        let original = EegPacket {
            sequence: 12345,
            channels: [100, -100, 8388607, -8388608, 0, 1, -1, 1000000],
        };

        let bytes = original.to_bytes();
        let restored = EegPacket::from_bytes(&bytes);

        assert_eq!(restored.sequence, original.sequence);
        for i in 0..8 {
            let expected = original.channels[i].clamp(-8388608, 8388607);
            assert_eq!(restored.channels[i], expected, "Channel {} mismatch", i);
        }
    }

    #[test]
    fn test_fnirs_packet_roundtrip() {
        let original = FnirsPacket {
            sequence: 9999,
            hbo: [100, -50, 200, -100],
            hbr: [-30, 60, -90, 120],
        };

        let bytes = original.to_bytes();
        let restored = FnirsPacket::from_bytes(&bytes);

        assert_eq!(restored.sequence, original.sequence);
        assert_eq!(restored.hbo, original.hbo);
        assert_eq!(restored.hbr, original.hbr);
    }

    #[test]
    fn test_status_packet_roundtrip() {
        let original = StatusPacket {
            state: DeviceState::Acquiring,
            flags: status_flags::RECORDING | status_flags::STIM_ACTIVE,
            battery_percent: 85,
            sample_rate_hz: 500,
            channel_mask: 0xFF,
            gain: 24,
            stim_amplitude_ua: 1500,
            uptime_sec: 3600,
            sequence: 42,
        };

        let bytes = original.to_bytes();
        let restored = StatusPacket::from_bytes(&bytes);

        assert_eq!(restored.state, original.state);
        assert_eq!(restored.flags, original.flags);
        assert_eq!(restored.battery_percent, original.battery_percent);
        assert_eq!(restored.uptime_sec, original.uptime_sec);
    }

    #[test]
    fn test_command_parsing() {
        assert_eq!(Command::from_u8(0x01), Some(Command::StartAcquisition));
        assert_eq!(Command::from_u8(0x20), Some(Command::StartStimulation));
        assert_eq!(Command::from_u8(0xFF), Some(Command::SoftReset));
        assert_eq!(Command::from_u8(0x99), None);
    }

    #[test]
    fn test_device_state() {
        let state = BlePeripheralState::new();
        assert_eq!(state.device_state(), DeviceState::Disconnected);

        state.set_connection(1);
        assert_eq!(state.device_state(), DeviceState::Idle);
        assert!(state.is_connected());

        state.set_device_state(DeviceState::Acquiring);
        assert!(state.device_state().is_acquiring());

        state.set_connection(0);
        assert!(!state.is_connected());
        assert_eq!(state.device_state(), DeviceState::Disconnected);
    }

    #[test]
    fn test_pack_unpack_eeg() {
        let samples = [100, -100, 8388607, -8388608, 0, 1, -1, 1000000];
        let packed = pack_eeg_samples(&samples);
        let unpacked = unpack_eeg_samples(&packed);

        for i in 0..8 {
            let expected = samples[i].clamp(-8388608, 8388607);
            assert_eq!(unpacked[i], expected);
        }
    }

    #[test]
    fn test_pack_unpack_fnirs() {
        let hbo = [1.5, -0.5, 2.0, -1.0];
        let hbr = [-0.3, 0.6, -0.9, 1.2];

        let packed = pack_fnirs_samples(&hbo, &hbr);
        let (hbo2, hbr2) = unpack_fnirs_samples(&packed);

        for i in 0..4 {
            assert!((hbo2[i] - hbo[i]).abs() < 0.011);
            assert!((hbr2[i] - hbr[i]).abs() < 0.011);
        }
    }
}

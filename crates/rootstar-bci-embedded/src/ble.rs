//! BLE service definitions for BCI data streaming
//!
//! This module defines the GATT service structure for BCI data.
//! It can be used with any embedded BLE stack (esp32-nimble, etc.).
//!
//! # BCI Service Structure
//!
//! ```text
//! BCI Data Service (UUID: 12340001-1234-5678-9abc-def012345678)
//! ├── EEG Data Characteristic (12340002-...) [Notify]
//! │   └── 8×3 bytes = 24 bytes per sample (8ch × 24-bit)
//! ├── fNIRS Data Characteristic (12340003-...) [Notify]
//! │   └── 8×2 bytes = 16 bytes per sample (4ch HbO + 4ch HbR)
//! ├── Command Characteristic (12340004-...) [Write]
//! │   └── 1-16 bytes command format
//! ├── Status Characteristic (12340005-...) [Read, Notify]
//! │   └── Device status, battery, signal quality
//! ├── EMG Data Characteristic (12340006-...) [Notify]
//! │   └── EMG envelope and frequency features
//! └── EDA Data Characteristic (12340007-...) [Notify]
//!     └── Skin conductance level and response
//! ```
//!
//! # Example Usage (with esp32-nimble)
//!
//! ```ignore
//! use rootstar_bci_embedded::ble::{BCI_SERVICE_UUID, EEG_DATA_CHAR_UUID};
//!
//! let bci_service = server.create_service(BCI_SERVICE_UUID);
//! let eeg_char = bci_service.create_characteristic(
//!     EEG_DATA_CHAR_UUID,
//!     NimbleProperties::NOTIFY,
//! );
//! ```

/// BCI service UUID (custom service)
/// UUID: 12340001-1234-5678-9abc-def012345678
pub const BCI_SERVICE_UUID: u128 = 0x12340001_1234_5678_9abc_def012345678;

/// EEG data characteristic UUID (notify)
/// Sends raw 24-bit EEG samples for all channels.
/// Format: [ch0_msb, ch0_mid, ch0_lsb, ch1_msb, ...] (24 bytes for 8 channels)
pub const EEG_DATA_CHAR_UUID: u128 = 0x12340002_1234_5678_9abc_def012345678;

/// fNIRS data characteristic UUID (notify)
/// Sends hemoglobin concentration values.
/// Format: [hbo0_h, hbo0_l, hbr0_h, hbr0_l, ...] (16 bytes for 4 channels)
pub const FNIRS_DATA_CHAR_UUID: u128 = 0x12340003_1234_5678_9abc_def012345678;

/// Command characteristic UUID (write)
/// Receives commands from the host.
/// Format: [command_id, ...params]
pub const COMMAND_CHAR_UUID: u128 = 0x12340004_1234_5678_9abc_def012345678;

/// Status characteristic UUID (read/notify)
/// Reports device status.
/// Format: [state, battery_pct, signal_quality, flags]
pub const STATUS_CHAR_UUID: u128 = 0x12340005_1234_5678_9abc_def012345678;

/// EMG data characteristic UUID (notify)
/// Sends EMG envelope and features.
/// Format: [envelope_h, envelope_l, mean_freq_h, mean_freq_l, valence]
pub const EMG_DATA_CHAR_UUID: u128 = 0x12340006_1234_5678_9abc_def012345678;

/// EDA data characteristic UUID (notify)
/// Sends skin conductance data.
/// Format: [scl_h, scl_l, scr_amplitude_h, scr_amplitude_l, scr_rise_time]
pub const EDA_DATA_CHAR_UUID: u128 = 0x12340007_1234_5678_9abc_def012345678;

/// BLE command IDs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BleCommand {
    /// Start data acquisition (0x01)
    StartAcquisition = 0x01,
    /// Stop data acquisition (0x02)
    StopAcquisition = 0x02,
    /// Set sample rate (0x03) - params: [rate_hz_h, rate_hz_l]
    SetSampleRate = 0x03,
    /// Set gain (0x04) - params: [gain_code]
    SetGain = 0x04,
    /// Run impedance check (0x05)
    RunImpedanceCheck = 0x05,
    /// Enter calibration mode (0x06)
    StartCalibration = 0x06,
    /// Exit calibration mode (0x07)
    StopCalibration = 0x07,
    /// Request status update (0x08)
    RequestStatus = 0x08,
    /// Enable specific channels (0x09) - params: [channel_mask]
    SetChannelMask = 0x09,
    /// Set device name (0x0A) - params: [name_bytes...]
    SetDeviceName = 0x0A,
    /// Start stimulation (0x10) - params: [protocol_id, ...]
    StartStimulation = 0x10,
    /// Stop stimulation (0x11)
    StopStimulation = 0x11,
    /// Set stimulation parameters (0x12) - params: [amp_h, amp_l, freq_h, freq_l]
    SetStimParams = 0x12,
    /// Enter bootloader (0xFF)
    EnterBootloader = 0xFF,
}

impl BleCommand {
    /// Try to parse a command from a byte
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::StartAcquisition),
            0x02 => Some(Self::StopAcquisition),
            0x03 => Some(Self::SetSampleRate),
            0x04 => Some(Self::SetGain),
            0x05 => Some(Self::RunImpedanceCheck),
            0x06 => Some(Self::StartCalibration),
            0x07 => Some(Self::StopCalibration),
            0x08 => Some(Self::RequestStatus),
            0x09 => Some(Self::SetChannelMask),
            0x0A => Some(Self::SetDeviceName),
            0x10 => Some(Self::StartStimulation),
            0x11 => Some(Self::StopStimulation),
            0x12 => Some(Self::SetStimParams),
            0xFF => Some(Self::EnterBootloader),
            _ => None,
        }
    }
}

/// Device state for status characteristic
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DeviceState {
    /// Device is idle, not acquiring
    #[default]
    Idle = 0x00,
    /// Device is acquiring data
    Acquiring = 0x01,
    /// Device is calibrating
    Calibrating = 0x02,
    /// Device is running impedance check
    ImpedanceCheck = 0x03,
    /// Device is stimulating
    Stimulating = 0x04,
    /// Device error state
    Error = 0xFF,
}

/// Status flags
pub mod status_flags {
    /// Battery low warning
    pub const BATTERY_LOW: u8 = 0x01;
    /// Electrode contact issue
    pub const ELECTRODE_OFF: u8 = 0x02;
    /// Signal saturation detected
    pub const SIGNAL_SATURATED: u8 = 0x04;
    /// Stimulation active
    pub const STIM_ACTIVE: u8 = 0x08;
    /// Recording active
    pub const RECORDING: u8 = 0x10;
    /// Sync pulse received
    pub const SYNC_PULSE: u8 = 0x20;
}

/// Status packet format
#[derive(Clone, Copy, Debug, Default)]
pub struct StatusPacket {
    /// Current device state
    pub state: DeviceState,
    /// Battery percentage (0-100)
    pub battery_percent: u8,
    /// Signal quality (0-100, average across channels)
    pub signal_quality: u8,
    /// Status flags (bitfield)
    pub flags: u8,
    /// Current sample rate in Hz
    pub sample_rate_hz: u16,
    /// Sequence number
    pub sequence: u16,
}

impl StatusPacket {
    /// Serialize to bytes
    pub const fn to_bytes(&self) -> [u8; 8] {
        [
            self.state as u8,
            self.battery_percent,
            self.signal_quality,
            self.flags,
            (self.sample_rate_hz >> 8) as u8,
            (self.sample_rate_hz & 0xFF) as u8,
            (self.sequence >> 8) as u8,
            (self.sequence & 0xFF) as u8,
        ]
    }

    /// Deserialize from bytes
    pub const fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            state: match bytes[0] {
                0x00 => DeviceState::Idle,
                0x01 => DeviceState::Acquiring,
                0x02 => DeviceState::Calibrating,
                0x03 => DeviceState::ImpedanceCheck,
                0x04 => DeviceState::Stimulating,
                _ => DeviceState::Error,
            },
            battery_percent: bytes[1],
            signal_quality: bytes[2],
            flags: bytes[3],
            sample_rate_hz: ((bytes[4] as u16) << 8) | (bytes[5] as u16),
            sequence: ((bytes[6] as u16) << 8) | (bytes[7] as u16),
        }
    }
}

/// Maximum EEG packet size (8 channels × 3 bytes per sample)
pub const MAX_EEG_PACKET_SIZE: usize = 24;

/// Maximum fNIRS packet size (4 channels HbO + 4 channels HbR × 2 bytes each)
pub const MAX_FNIRS_PACKET_SIZE: usize = 16;

/// Maximum command packet size
pub const MAX_COMMAND_SIZE: usize = 20;

/// BLE MTU for data packets (default conservative value)
pub const BLE_DATA_MTU: usize = 20;

/// Pack EEG samples into a BLE packet
///
/// Takes 8 24-bit samples and packs them into 24 bytes.
pub fn pack_eeg_samples(samples: &[i32; 8]) -> [u8; MAX_EEG_PACKET_SIZE] {
    let mut packet = [0u8; MAX_EEG_PACKET_SIZE];
    for (i, &sample) in samples.iter().enumerate() {
        let offset = i * 3;
        // Pack as 24-bit big-endian, sign-extended
        packet[offset] = ((sample >> 16) & 0xFF) as u8;
        packet[offset + 1] = ((sample >> 8) & 0xFF) as u8;
        packet[offset + 2] = (sample & 0xFF) as u8;
    }
    packet
}

/// Unpack EEG samples from a BLE packet
pub fn unpack_eeg_samples(packet: &[u8; MAX_EEG_PACKET_SIZE]) -> [i32; 8] {
    let mut samples = [0i32; 8];
    for i in 0..8 {
        let offset = i * 3;
        let raw = ((packet[offset] as i32) << 16)
            | ((packet[offset + 1] as i32) << 8)
            | (packet[offset + 2] as i32);
        // Sign-extend from 24-bit to 32-bit
        samples[i] = if raw & 0x800000 != 0 {
            raw | 0xFF000000u32 as i32
        } else {
            raw
        };
    }
    samples
}

/// Pack fNIRS samples into a BLE packet
///
/// Takes 4 HbO and 4 HbR values and packs them as 16-bit fixed point.
pub fn pack_fnirs_samples(hbo: &[f32; 4], hbr: &[f32; 4]) -> [u8; MAX_FNIRS_PACKET_SIZE] {
    let mut packet = [0u8; MAX_FNIRS_PACKET_SIZE];

    // Pack HbO values (4 × 2 bytes = 8 bytes)
    for (i, &value) in hbo.iter().enumerate() {
        let fixed = (value * 1000.0) as i16; // micro-molar units
        packet[i * 2] = (fixed >> 8) as u8;
        packet[i * 2 + 1] = (fixed & 0xFF) as u8;
    }

    // Pack HbR values (4 × 2 bytes = 8 bytes)
    for (i, &value) in hbr.iter().enumerate() {
        let fixed = (value * 1000.0) as i16;
        let offset = 8 + i * 2;
        packet[offset] = (fixed >> 8) as u8;
        packet[offset + 1] = (fixed & 0xFF) as u8;
    }

    packet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_pack_unpack() {
        let samples = [100, -100, 8388607, -8388608, 0, 1, -1, 1000000];
        let packed = pack_eeg_samples(&samples);
        let unpacked = unpack_eeg_samples(&packed);

        // Verify values within 24-bit range are preserved
        for i in 0..8 {
            let expected = samples[i].clamp(-8388608, 8388607);
            assert_eq!(unpacked[i], expected, "Sample {} mismatch", i);
        }
    }

    #[test]
    fn test_status_packet() {
        let status = StatusPacket {
            state: DeviceState::Acquiring,
            battery_percent: 85,
            signal_quality: 95,
            flags: status_flags::RECORDING | status_flags::STIM_ACTIVE,
            sample_rate_hz: 500,
            sequence: 1234,
        };

        let bytes = status.to_bytes();
        let restored = StatusPacket::from_bytes(&bytes);

        assert_eq!(restored.state, status.state);
        assert_eq!(restored.battery_percent, status.battery_percent);
        assert_eq!(restored.signal_quality, status.signal_quality);
        assert_eq!(restored.flags, status.flags);
        assert_eq!(restored.sample_rate_hz, status.sample_rate_hz);
        assert_eq!(restored.sequence, status.sequence);
    }

    #[test]
    fn test_command_parsing() {
        assert_eq!(BleCommand::from_u8(0x01), Some(BleCommand::StartAcquisition));
        assert_eq!(BleCommand::from_u8(0x02), Some(BleCommand::StopAcquisition));
        assert_eq!(BleCommand::from_u8(0xFF), Some(BleCommand::EnterBootloader));
        assert_eq!(BleCommand::from_u8(0x99), None);
    }
}

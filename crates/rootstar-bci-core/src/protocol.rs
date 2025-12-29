//! Wire protocol for BCI device communication
//!
//! This module defines the packet format for communication between the
//! ESP32 embedded device and host software. The protocol is designed to be:
//! - Compact for low-latency streaming
//! - Self-synchronizing with sync bytes
//! - Error-detecting with XOR checksum
//! - Extensible via packet type field

use serde::{Deserialize, Serialize};

use crate::error::ProtocolError;
use crate::types::{EegSample, FnirsSample, Fixed24_8, StimParams, FnirsChannel};

// ============================================================================
// Packet Types
// ============================================================================

/// Packet type identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// EEG sample data (8 channels)
    EegData = 0x01,
    /// fNIRS raw intensity data
    FnirsData = 0x02,
    /// Computed hemodynamic data
    HemodynamicData = 0x03,
    /// Electrode impedance measurement
    Impedance = 0x04,
    /// Device status/heartbeat
    Status = 0x05,

    /// Stimulation command (host → device)
    StimCommand = 0x10,
    /// Stimulation status (device → host)
    StimStatus = 0x11,

    /// Configuration request (host → device)
    ConfigRequest = 0x20,
    /// Configuration response (device → host)
    ConfigResponse = 0x21,
    /// Configuration set (host → device)
    ConfigSet = 0x22,
    /// Configuration acknowledgment (device → host)
    ConfigAck = 0x23,

    /// Start acquisition (host → device)
    StartAcquisition = 0x30,
    /// Stop acquisition (host → device)
    StopAcquisition = 0x31,

    /// Error packet
    Error = 0xFF,
}

impl PacketType {
    /// Try to convert a byte to a packet type.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(Self::EegData),
            0x02 => Some(Self::FnirsData),
            0x03 => Some(Self::HemodynamicData),
            0x04 => Some(Self::Impedance),
            0x05 => Some(Self::Status),
            0x10 => Some(Self::StimCommand),
            0x11 => Some(Self::StimStatus),
            0x20 => Some(Self::ConfigRequest),
            0x21 => Some(Self::ConfigResponse),
            0x22 => Some(Self::ConfigSet),
            0x23 => Some(Self::ConfigAck),
            0x30 => Some(Self::StartAcquisition),
            0x31 => Some(Self::StopAcquisition),
            0xFF => Some(Self::Error),
            _ => None,
        }
    }

    /// Check if this packet type carries data from device to host.
    #[must_use]
    pub const fn is_device_to_host(self) -> bool {
        matches!(
            self,
            Self::EegData
                | Self::FnirsData
                | Self::HemodynamicData
                | Self::Impedance
                | Self::Status
                | Self::StimStatus
                | Self::ConfigResponse
                | Self::ConfigAck
                | Self::Error
        )
    }

    /// Check if this packet type is a command from host to device.
    #[must_use]
    pub const fn is_host_to_device(self) -> bool {
        matches!(
            self,
            Self::StimCommand
                | Self::ConfigRequest
                | Self::ConfigSet
                | Self::StartAcquisition
                | Self::StopAcquisition
        )
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for PacketType {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::EegData => defmt::write!(f, "EEG"),
            Self::FnirsData => defmt::write!(f, "fNIRS"),
            Self::HemodynamicData => defmt::write!(f, "Hemo"),
            Self::Impedance => defmt::write!(f, "Imp"),
            Self::Status => defmt::write!(f, "Status"),
            Self::StimCommand => defmt::write!(f, "StimCmd"),
            Self::StimStatus => defmt::write!(f, "StimStat"),
            Self::ConfigRequest => defmt::write!(f, "CfgReq"),
            Self::ConfigResponse => defmt::write!(f, "CfgResp"),
            Self::ConfigSet => defmt::write!(f, "CfgSet"),
            Self::ConfigAck => defmt::write!(f, "CfgAck"),
            Self::StartAcquisition => defmt::write!(f, "Start"),
            Self::StopAcquisition => defmt::write!(f, "Stop"),
            Self::Error => defmt::write!(f, "Error"),
        }
    }
}

// ============================================================================
// Packet Header
// ============================================================================

/// Packet header structure.
///
/// All packets begin with this header:
/// - 2 sync bytes (0xA5, 0x5A) for frame alignment
/// - 1 byte packet type
/// - 2 bytes sequence number (little-endian)
/// - 2 bytes payload length (little-endian)
/// - 1 byte header checksum (XOR of type + seq + len)
///
/// Total header size: 8 bytes
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PacketHeader {
    /// Packet type
    pub packet_type: PacketType,
    /// Sequence number (wraps at 65535)
    pub sequence: u16,
    /// Payload length in bytes
    pub payload_len: u16,
}

impl PacketHeader {
    /// Sync byte 0
    pub const SYNC_0: u8 = 0xA5;
    /// Sync byte 1
    pub const SYNC_1: u8 = 0x5A;
    /// Header size in bytes
    pub const SIZE: usize = 8;
    /// Maximum payload size
    pub const MAX_PAYLOAD: u16 = 1024;

    /// Create a new header.
    #[must_use]
    pub const fn new(packet_type: PacketType, sequence: u16, payload_len: u16) -> Self {
        Self { packet_type, sequence, payload_len }
    }

    /// Compute the header checksum.
    #[must_use]
    pub const fn checksum(&self) -> u8 {
        let seq_lo = self.sequence as u8;
        let seq_hi = (self.sequence >> 8) as u8;
        let len_lo = self.payload_len as u8;
        let len_hi = (self.payload_len >> 8) as u8;

        (self.packet_type as u8) ^ seq_lo ^ seq_hi ^ len_lo ^ len_hi
    }

    /// Serialize header to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        [
            Self::SYNC_0,
            Self::SYNC_1,
            self.packet_type as u8,
            self.sequence as u8,
            (self.sequence >> 8) as u8,
            self.payload_len as u8,
            (self.payload_len >> 8) as u8,
            self.checksum(),
        ]
    }

    /// Parse header from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if sync bytes are wrong, packet type is invalid,
    /// or checksum doesn't match.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.len() < Self::SIZE {
            return Err(ProtocolError::IncompletePacket {
                received: bytes.len(),
                expected: Self::SIZE,
            });
        }

        // Check sync bytes
        if bytes[0] != Self::SYNC_0 || bytes[1] != Self::SYNC_1 {
            return Err(ProtocolError::InvalidSync {
                got_0: bytes[0],
                got_1: bytes[1],
            });
        }

        // Parse packet type
        let packet_type = PacketType::from_byte(bytes[2])
            .ok_or(ProtocolError::InvalidPacketType { packet_type: bytes[2] })?;

        // Parse sequence and length (little-endian)
        let sequence = u16::from_le_bytes([bytes[3], bytes[4]]);
        let payload_len = u16::from_le_bytes([bytes[5], bytes[6]]);

        // Validate payload length
        if payload_len > Self::MAX_PAYLOAD {
            return Err(ProtocolError::PayloadTooLarge {
                length: payload_len,
                maximum: Self::MAX_PAYLOAD,
            });
        }

        // Verify checksum
        let header = Self { packet_type, sequence, payload_len };
        let expected_checksum = header.checksum();
        if bytes[7] != expected_checksum {
            return Err(ProtocolError::ChecksumMismatch {
                expected: expected_checksum,
                computed: bytes[7],
            });
        }

        Ok(header)
    }
}

// ============================================================================
// Payload Serialization
// ============================================================================

/// Compute XOR checksum for payload.
#[must_use]
pub fn payload_checksum(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |acc, &b| acc ^ b)
}

/// Serialize an EEG sample to bytes.
///
/// Format:
/// - 8 bytes: timestamp (little-endian u64)
/// - 4 bytes: sequence (little-endian u32)
/// - 32 bytes: 8 channels × 4 bytes each (Q24.8 i32, little-endian)
/// - 1 byte: payload checksum
///
/// Total: 45 bytes
pub fn serialize_eeg_sample(sample: &EegSample, buffer: &mut [u8]) -> Result<usize, ProtocolError> {
    const SIZE: usize = 45;

    if buffer.len() < SIZE {
        return Err(ProtocolError::BufferOverflow {
            required: SIZE,
            available: buffer.len(),
        });
    }

    let mut offset = 0;

    // Timestamp
    buffer[offset..offset + 8].copy_from_slice(&sample.timestamp_us.to_le_bytes());
    offset += 8;

    // Sequence
    buffer[offset..offset + 4].copy_from_slice(&sample.sequence.to_le_bytes());
    offset += 4;

    // Channels
    for ch in &sample.channels {
        buffer[offset..offset + 4].copy_from_slice(&ch.to_raw().to_le_bytes());
        offset += 4;
    }

    // Checksum
    buffer[offset] = payload_checksum(&buffer[..offset]);

    Ok(SIZE)
}

/// Deserialize an EEG sample from bytes.
pub fn deserialize_eeg_sample(data: &[u8]) -> Result<EegSample, ProtocolError> {
    const SIZE: usize = 45;

    if data.len() < SIZE {
        return Err(ProtocolError::IncompletePacket {
            received: data.len(),
            expected: SIZE,
        });
    }

    // Verify checksum
    let expected_checksum = payload_checksum(&data[..SIZE - 1]);
    if data[SIZE - 1] != expected_checksum {
        return Err(ProtocolError::ChecksumMismatch {
            expected: expected_checksum,
            computed: data[SIZE - 1],
        });
    }

    let mut offset = 0;

    // Timestamp
    let timestamp_us = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // Sequence
    let sequence = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Channels
    let mut channels = [Fixed24_8::ZERO; 8];
    for ch in &mut channels {
        let raw = i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        *ch = Fixed24_8::from_raw(raw);
        offset += 4;
    }

    Ok(EegSample { timestamp_us, channels, sequence })
}

/// Serialize an fNIRS sample to bytes.
///
/// Format:
/// - 8 bytes: timestamp
/// - 4 bytes: sequence
/// - 3 bytes: channel (source, detector, separation)
/// - 2 bytes: intensity_760
/// - 2 bytes: intensity_850
/// - 1 byte: checksum
///
/// Total: 20 bytes
pub fn serialize_fnirs_sample(sample: &FnirsSample, buffer: &mut [u8]) -> Result<usize, ProtocolError> {
    const SIZE: usize = 20;

    if buffer.len() < SIZE {
        return Err(ProtocolError::BufferOverflow {
            required: SIZE,
            available: buffer.len(),
        });
    }

    let mut offset = 0;

    // Timestamp
    buffer[offset..offset + 8].copy_from_slice(&sample.timestamp_us.to_le_bytes());
    offset += 8;

    // Sequence
    buffer[offset..offset + 4].copy_from_slice(&sample.sequence.to_le_bytes());
    offset += 4;

    // Channel
    buffer[offset] = sample.channel.source;
    buffer[offset + 1] = sample.channel.detector;
    buffer[offset + 2] = sample.channel.separation_mm;
    offset += 3;

    // Intensities
    buffer[offset..offset + 2].copy_from_slice(&sample.intensity_760.to_le_bytes());
    offset += 2;
    buffer[offset..offset + 2].copy_from_slice(&sample.intensity_850.to_le_bytes());
    offset += 2;

    // Checksum
    buffer[offset] = payload_checksum(&buffer[..offset]);

    Ok(SIZE)
}

/// Deserialize an fNIRS sample from bytes.
pub fn deserialize_fnirs_sample(data: &[u8]) -> Result<FnirsSample, ProtocolError> {
    const SIZE: usize = 20;

    if data.len() < SIZE {
        return Err(ProtocolError::IncompletePacket {
            received: data.len(),
            expected: SIZE,
        });
    }

    // Verify checksum
    let expected_checksum = payload_checksum(&data[..SIZE - 1]);
    if data[SIZE - 1] != expected_checksum {
        return Err(ProtocolError::ChecksumMismatch {
            expected: expected_checksum,
            computed: data[SIZE - 1],
        });
    }

    let mut offset = 0;

    let timestamp_us = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    offset += 8;

    let sequence = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let channel = FnirsChannel::new(data[offset], data[offset + 1], data[offset + 2]);
    offset += 3;

    let intensity_760 = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let intensity_850 = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());

    Ok(FnirsSample { timestamp_us, channel, intensity_760, intensity_850, sequence })
}

/// Serialize stimulation parameters to bytes.
///
/// Format:
/// - 1 byte: mode
/// - 2 bytes: amplitude_ua
/// - 2 bytes: frequency_hz
/// - 4 bytes: duration_ms
/// - 2 bytes: ramp_ms
/// - 1 byte: checksum
///
/// Total: 12 bytes
pub fn serialize_stim_params(params: &StimParams, buffer: &mut [u8]) -> Result<usize, ProtocolError> {
    const SIZE: usize = 12;

    if buffer.len() < SIZE {
        return Err(ProtocolError::BufferOverflow {
            required: SIZE,
            available: buffer.len(),
        });
    }

    let mut offset = 0;

    // Mode
    buffer[offset] = params.mode as u8;
    offset += 1;

    // Amplitude
    buffer[offset..offset + 2].copy_from_slice(&params.amplitude_ua.to_le_bytes());
    offset += 2;

    // Frequency
    buffer[offset..offset + 2].copy_from_slice(&params.frequency_hz.to_le_bytes());
    offset += 2;

    // Duration
    buffer[offset..offset + 4].copy_from_slice(&params.duration_ms.to_le_bytes());
    offset += 4;

    // Ramp
    buffer[offset..offset + 2].copy_from_slice(&params.ramp_ms.to_le_bytes());
    offset += 2;

    // Checksum
    buffer[offset] = payload_checksum(&buffer[..offset]);

    Ok(SIZE)
}

/// Deserialize stimulation parameters from bytes.
pub fn deserialize_stim_params(data: &[u8]) -> Result<StimParams, ProtocolError> {
    use crate::types::StimMode;

    const SIZE: usize = 12;

    if data.len() < SIZE {
        return Err(ProtocolError::IncompletePacket {
            received: data.len(),
            expected: SIZE,
        });
    }

    // Verify checksum
    let expected_checksum = payload_checksum(&data[..SIZE - 1]);
    if data[SIZE - 1] != expected_checksum {
        return Err(ProtocolError::ChecksumMismatch {
            expected: expected_checksum,
            computed: data[SIZE - 1],
        });
    }

    let mut offset = 0;

    let mode = match data[offset] {
        0 => StimMode::Off,
        1 => StimMode::TdcsAnodal,
        2 => StimMode::TdcsCathodal,
        3 => StimMode::Tacs,
        4 => StimMode::Pbm,
        _ => StimMode::Off,
    };
    offset += 1;

    let amplitude_ua = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let frequency_hz = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let duration_ms = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let ramp_ms = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());

    Ok(StimParams { mode, amplitude_ua, frequency_hz, duration_ms, ramp_ms })
}

// ============================================================================
// Configuration Parameters
// ============================================================================

/// Device configuration parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// EEG sample rate in Hz (250, 500, 1000, or 2000)
    pub eeg_sample_rate_hz: u16,
    /// EEG gain (1, 2, 4, 6, 8, 12, or 24)
    pub eeg_gain: u8,
    /// fNIRS sample rate in Hz (typically 10-100)
    pub fnirs_sample_rate_hz: u8,
    /// fNIRS LED intensity (0-255)
    pub fnirs_led_intensity: u8,
    /// Enable active bias (DRL)
    pub enable_drl: bool,
    /// Enable 50/60 Hz notch filter
    pub enable_notch: bool,
    /// Notch frequency (50 or 60 Hz)
    pub notch_freq_hz: u8,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            eeg_sample_rate_hz: 250,
            eeg_gain: 24,
            fnirs_sample_rate_hz: 10,
            fnirs_led_intensity: 200,
            enable_drl: true,
            enable_notch: true,
            notch_freq_hz: 60,
        }
    }
}

impl DeviceConfig {
    /// Serialize to bytes (8 bytes + checksum).
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut bytes = [0u8; 9];

        bytes[0] = self.eeg_sample_rate_hz as u8;
        bytes[1] = (self.eeg_sample_rate_hz >> 8) as u8;
        bytes[2] = self.eeg_gain;
        bytes[3] = self.fnirs_sample_rate_hz;
        bytes[4] = self.fnirs_led_intensity;
        bytes[5] = u8::from(self.enable_drl) | (u8::from(self.enable_notch) << 1);
        bytes[6] = self.notch_freq_hz;
        bytes[7] = 0; // Reserved

        bytes[8] = payload_checksum(&bytes[..8]);

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ProtocolError> {
        if data.len() < 9 {
            return Err(ProtocolError::IncompletePacket {
                received: data.len(),
                expected: 9,
            });
        }

        let expected_checksum = payload_checksum(&data[..8]);
        if data[8] != expected_checksum {
            return Err(ProtocolError::ChecksumMismatch {
                expected: expected_checksum,
                computed: data[8],
            });
        }

        Ok(Self {
            eeg_sample_rate_hz: u16::from_le_bytes([data[0], data[1]]),
            eeg_gain: data[2],
            fnirs_sample_rate_hz: data[3],
            fnirs_led_intensity: data[4],
            enable_drl: data[5] & 0x01 != 0,
            enable_notch: data[5] & 0x02 != 0,
            notch_freq_hz: data[6],
        })
    }
}

// ============================================================================
// Packet Builder/Parser
// ============================================================================

/// Complete packet with header and payload.
#[derive(Clone, Debug)]
pub struct Packet<'a> {
    /// Packet header
    pub header: PacketHeader,
    /// Payload data (does not include checksum)
    pub payload: &'a [u8],
}

impl<'a> Packet<'a> {
    /// Create a new packet.
    #[must_use]
    pub const fn new(header: PacketHeader, payload: &'a [u8]) -> Self {
        Self { header, payload }
    }

    /// Total packet size including header and payload.
    #[must_use]
    pub fn total_size(&self) -> usize {
        PacketHeader::SIZE + self.payload.len()
    }

    /// Serialize packet to a buffer.
    ///
    /// Returns the number of bytes written.
    pub fn serialize(&self, buffer: &mut [u8]) -> Result<usize, ProtocolError> {
        let total = self.total_size();

        if buffer.len() < total {
            return Err(ProtocolError::BufferOverflow {
                required: total,
                available: buffer.len(),
            });
        }

        // Write header
        buffer[..PacketHeader::SIZE].copy_from_slice(&self.header.to_bytes());

        // Write payload
        buffer[PacketHeader::SIZE..total].copy_from_slice(self.payload);

        Ok(total)
    }
}

/// Scan a buffer for the start of a valid packet.
///
/// Returns the offset of the sync bytes if found, or None if not found.
#[must_use]
pub fn find_sync(buffer: &[u8]) -> Option<usize> {
    for i in 0..buffer.len().saturating_sub(1) {
        if buffer[i] == PacketHeader::SYNC_0 && buffer[i + 1] == PacketHeader::SYNC_1 {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader::new(PacketType::EegData, 1234, 45);
        let bytes = header.to_bytes();
        let parsed = PacketHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.packet_type, PacketType::EegData);
        assert_eq!(parsed.sequence, 1234);
        assert_eq!(parsed.payload_len, 45);
    }

    #[test]
    fn test_packet_header_invalid_sync() {
        let bytes = [0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = PacketHeader::from_bytes(&bytes);

        assert!(matches!(result, Err(ProtocolError::InvalidSync { .. })));
    }

    #[test]
    fn test_eeg_sample_roundtrip() {
        let sample = EegSample {
            timestamp_us: 123456789,
            channels: [
                Fixed24_8::from_f32(1.0),
                Fixed24_8::from_f32(2.0),
                Fixed24_8::from_f32(3.0),
                Fixed24_8::from_f32(4.0),
                Fixed24_8::from_f32(5.0),
                Fixed24_8::from_f32(6.0),
                Fixed24_8::from_f32(7.0),
                Fixed24_8::from_f32(8.0),
            ],
            sequence: 42,
        };

        let mut buffer = [0u8; 64];
        let size = serialize_eeg_sample(&sample, &mut buffer).unwrap();
        let parsed = deserialize_eeg_sample(&buffer[..size]).unwrap();

        assert_eq!(parsed.timestamp_us, sample.timestamp_us);
        assert_eq!(parsed.sequence, sample.sequence);
        for i in 0..8 {
            assert_eq!(parsed.channels[i].to_raw(), sample.channels[i].to_raw());
        }
    }

    #[test]
    fn test_fnirs_sample_roundtrip() {
        let sample = FnirsSample {
            timestamp_us: 987654321,
            channel: FnirsChannel::new(1, 2, 30),
            intensity_760: 45000,
            intensity_850: 42000,
            sequence: 99,
        };

        let mut buffer = [0u8; 32];
        let size = serialize_fnirs_sample(&sample, &mut buffer).unwrap();
        let parsed = deserialize_fnirs_sample(&buffer[..size]).unwrap();

        assert_eq!(parsed.timestamp_us, sample.timestamp_us);
        assert_eq!(parsed.channel.source, sample.channel.source);
        assert_eq!(parsed.channel.detector, sample.channel.detector);
        assert_eq!(parsed.intensity_760, sample.intensity_760);
        assert_eq!(parsed.intensity_850, sample.intensity_850);
        assert_eq!(parsed.sequence, sample.sequence);
    }

    #[test]
    fn test_stim_params_roundtrip() {
        use crate::types::StimMode;

        let params = StimParams {
            mode: StimMode::Tacs,
            amplitude_ua: 1500,
            frequency_hz: 40,
            duration_ms: 1200000,
            ramp_ms: 100,
        };

        let mut buffer = [0u8; 16];
        let size = serialize_stim_params(&params, &mut buffer).unwrap();
        let parsed = deserialize_stim_params(&buffer[..size]).unwrap();

        assert_eq!(parsed.mode, params.mode);
        assert_eq!(parsed.amplitude_ua, params.amplitude_ua);
        assert_eq!(parsed.frequency_hz, params.frequency_hz);
        assert_eq!(parsed.duration_ms, params.duration_ms);
        assert_eq!(parsed.ramp_ms, params.ramp_ms);
    }

    #[test]
    fn test_device_config_roundtrip() {
        let config = DeviceConfig {
            eeg_sample_rate_hz: 500,
            eeg_gain: 12,
            fnirs_sample_rate_hz: 20,
            fnirs_led_intensity: 180,
            enable_drl: true,
            enable_notch: true,
            notch_freq_hz: 50,
        };

        let bytes = config.to_bytes();
        let parsed = DeviceConfig::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.eeg_sample_rate_hz, config.eeg_sample_rate_hz);
        assert_eq!(parsed.eeg_gain, config.eeg_gain);
        assert_eq!(parsed.fnirs_sample_rate_hz, config.fnirs_sample_rate_hz);
        assert_eq!(parsed.enable_drl, config.enable_drl);
        assert_eq!(parsed.enable_notch, config.enable_notch);
        assert_eq!(parsed.notch_freq_hz, config.notch_freq_hz);
    }

    #[test]
    fn test_find_sync() {
        let buffer = [0x00, 0x01, 0xA5, 0x5A, 0x01, 0x00, 0x00];
        assert_eq!(find_sync(&buffer), Some(2));

        let no_sync = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(find_sync(&no_sync), None);
    }

    #[test]
    fn test_payload_checksum() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = payload_checksum(&data);
        assert_eq!(checksum, 0x01 ^ 0x02 ^ 0x03 ^ 0x04);
    }
}

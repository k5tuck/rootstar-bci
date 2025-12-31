//! BrainFlow-compatible data format
//!
//! BrainFlow is a library for obtaining, parsing, and analyzing EEG, EMG,
//! ECG and other biosignal data. This module provides compatibility with
//! the BrainFlow data format, enabling integration with:
//!
//! - OpenBCI boards
//! - BrainFlow-based applications
//! - BCI research pipelines
//!
//! # Data Format
//!
//! BrainFlow uses a 2D array format where:
//! - Each column is a data sample
//! - Each row is a channel/data type
//!
//! The row layout varies by board type, but typically includes:
//! - Package number
//! - EEG channels (1-N)
//! - Accelerometer (X, Y, Z)
//! - Gyroscope (X, Y, Z)
//! - Analog channels
//! - Timestamp
//! - Marker channel
//!
//! # Board Compatibility
//!
//! Rootstar BCI presents itself as a synthetic board with custom
//! channel layout supporting EEG, fNIRS, EMG, and EDA.

use std::collections::HashMap;

/// BrainFlow board identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BoardId {
    /// Playback file board
    PlaybackFile = -3,
    /// Streaming board
    Streaming = -2,
    /// Synthetic board
    Synthetic = -1,
    /// Cyton board (OpenBCI 8-channel)
    Cyton = 0,
    /// Ganglion board (OpenBCI 4-channel)
    Ganglion = 1,
    /// Cyton Daisy (OpenBCI 16-channel)
    CytonDaisy = 2,
    /// Cyton WiFi
    CytonWifi = 5,
    /// Ganglion WiFi
    GanglionWifi = 4,
    /// BrainBit
    BrainBit = 7,
    /// Notion 1
    Notion1 = 13,
    /// Notion 2
    Notion2 = 14,
    /// Muse S
    MuseS = 21,
    /// Muse 2
    Muse2 = 22,
    /// Rootstar BCI (custom ID)
    RootstarBci = 100,
}

impl BoardId {
    /// Get board name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::PlaybackFile => "Playback File",
            Self::Streaming => "Streaming Board",
            Self::Synthetic => "Synthetic Board",
            Self::Cyton => "OpenBCI Cyton",
            Self::Ganglion => "OpenBCI Ganglion",
            Self::CytonDaisy => "OpenBCI Cyton+Daisy",
            Self::CytonWifi => "OpenBCI Cyton WiFi",
            Self::GanglionWifi => "OpenBCI Ganglion WiFi",
            Self::BrainBit => "BrainBit",
            Self::Notion1 => "Neurosity Notion 1",
            Self::Notion2 => "Neurosity Notion 2",
            Self::MuseS => "Muse S",
            Self::Muse2 => "Muse 2",
            Self::RootstarBci => "Rootstar BCI",
        }
    }

    /// Get number of EEG channels
    #[must_use]
    pub fn eeg_channels(&self) -> usize {
        match self {
            Self::PlaybackFile | Self::Streaming | Self::Synthetic => 8,
            Self::Cyton | Self::CytonWifi => 8,
            Self::Ganglion | Self::GanglionWifi => 4,
            Self::CytonDaisy => 16,
            Self::BrainBit => 4,
            Self::Notion1 | Self::Notion2 => 8,
            Self::MuseS | Self::Muse2 => 4,
            Self::RootstarBci => 8,
        }
    }

    /// Get sampling rate
    #[must_use]
    pub fn sampling_rate(&self) -> f64 {
        match self {
            Self::PlaybackFile | Self::Streaming | Self::Synthetic => 250.0,
            Self::Cyton | Self::CytonWifi | Self::CytonDaisy => 250.0,
            Self::Ganglion | Self::GanglionWifi => 200.0,
            Self::BrainBit => 250.0,
            Self::Notion1 | Self::Notion2 => 256.0,
            Self::MuseS | Self::Muse2 => 256.0,
            Self::RootstarBci => 250.0,
        }
    }
}

/// Channel types in BrainFlow format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Package/sequence number
    PackageNum,
    /// EEG channel
    Eeg,
    /// EMG channel
    Emg,
    /// ECG channel
    Ecg,
    /// EOG channel
    Eog,
    /// EDA/GSR channel
    Eda,
    /// PPG channel
    Ppg,
    /// Accelerometer
    Accel,
    /// Gyroscope
    Gyro,
    /// Temperature
    Temperature,
    /// Battery level
    Battery,
    /// Timestamp
    Timestamp,
    /// Marker/trigger
    Marker,
    /// Analog input
    Analog,
    /// Other/unknown
    Other,
}

/// BrainFlow data row description
#[derive(Debug, Clone)]
pub struct ChannelDesc {
    /// Channel type
    pub channel_type: ChannelType,
    /// Row index in data array
    pub row_index: usize,
    /// Channel name
    pub name: String,
    /// Unit
    pub unit: String,
}

/// BrainFlow packet format descriptor
#[derive(Debug, Clone)]
pub struct BrainFlowFormat {
    /// Board ID
    pub board_id: BoardId,
    /// Number of rows (channels)
    pub num_rows: usize,
    /// Channel descriptions
    pub channels: Vec<ChannelDesc>,
    /// EEG channel indices
    pub eeg_channels: Vec<usize>,
    /// fNIRS channels (Rootstar extension)
    pub fnirs_channels: Vec<usize>,
    /// EMG channel indices
    pub emg_channels: Vec<usize>,
    /// EDA channel indices
    pub eda_channels: Vec<usize>,
    /// Accelerometer channel indices
    pub accel_channels: Vec<usize>,
    /// Gyroscope channel indices
    pub gyro_channels: Vec<usize>,
    /// Timestamp channel index
    pub timestamp_channel: usize,
    /// Marker channel index
    pub marker_channel: usize,
    /// Package number channel index
    pub package_num_channel: usize,
}

impl BrainFlowFormat {
    /// Create format for Rootstar BCI
    ///
    /// Row layout:
    /// 0: Package number
    /// 1-8: EEG channels (8)
    /// 9-16: fNIRS HbO (4) + HbR (4)
    /// 17: EMG envelope
    /// 18: EMG valence
    /// 19: EDA SCL
    /// 20: EDA SCR
    /// 21-23: Accelerometer (X, Y, Z)
    /// 24-26: Gyroscope (X, Y, Z)
    /// 27: Battery level
    /// 28: Timestamp
    /// 29: Marker
    #[must_use]
    pub fn rootstar_bci() -> Self {
        let mut channels = Vec::new();

        // Package number
        channels.push(ChannelDesc {
            channel_type: ChannelType::PackageNum,
            row_index: 0,
            name: "Package".to_string(),
            unit: "".to_string(),
        });

        // EEG channels 1-8
        let eeg_labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"];
        for (i, label) in eeg_labels.iter().enumerate() {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Eeg,
                row_index: 1 + i,
                name: (*label).to_string(),
                unit: "uV".to_string(),
            });
        }

        // fNIRS HbO channels (Rootstar extension)
        for i in 0..4 {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Other,
                row_index: 9 + i,
                name: format!("HbO_{}", i + 1),
                unit: "uM".to_string(),
            });
        }

        // fNIRS HbR channels
        for i in 0..4 {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Other,
                row_index: 13 + i,
                name: format!("HbR_{}", i + 1),
                unit: "uM".to_string(),
            });
        }

        // EMG
        channels.push(ChannelDesc {
            channel_type: ChannelType::Emg,
            row_index: 17,
            name: "EMG_Envelope".to_string(),
            unit: "uV".to_string(),
        });
        channels.push(ChannelDesc {
            channel_type: ChannelType::Emg,
            row_index: 18,
            name: "EMG_Valence".to_string(),
            unit: "".to_string(),
        });

        // EDA
        channels.push(ChannelDesc {
            channel_type: ChannelType::Eda,
            row_index: 19,
            name: "EDA_SCL".to_string(),
            unit: "uS".to_string(),
        });
        channels.push(ChannelDesc {
            channel_type: ChannelType::Eda,
            row_index: 20,
            name: "EDA_SCR".to_string(),
            unit: "uS".to_string(),
        });

        // Accelerometer
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Accel,
                row_index: 21 + i,
                name: format!("Accel_{}", axis),
                unit: "g".to_string(),
            });
        }

        // Gyroscope
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Gyro,
                row_index: 24 + i,
                name: format!("Gyro_{}", axis),
                unit: "dps".to_string(),
            });
        }

        // Battery
        channels.push(ChannelDesc {
            channel_type: ChannelType::Battery,
            row_index: 27,
            name: "Battery".to_string(),
            unit: "%".to_string(),
        });

        // Timestamp
        channels.push(ChannelDesc {
            channel_type: ChannelType::Timestamp,
            row_index: 28,
            name: "Timestamp".to_string(),
            unit: "s".to_string(),
        });

        // Marker
        channels.push(ChannelDesc {
            channel_type: ChannelType::Marker,
            row_index: 29,
            name: "Marker".to_string(),
            unit: "".to_string(),
        });

        Self {
            board_id: BoardId::RootstarBci,
            num_rows: 30,
            channels,
            eeg_channels: (1..=8).collect(),
            fnirs_channels: (9..=16).collect(),
            emg_channels: vec![17, 18],
            eda_channels: vec![19, 20],
            accel_channels: vec![21, 22, 23],
            gyro_channels: vec![24, 25, 26],
            timestamp_channel: 28,
            marker_channel: 29,
            package_num_channel: 0,
        }
    }

    /// Create format for OpenBCI Cyton (for compatibility testing)
    #[must_use]
    pub fn cyton() -> Self {
        let mut channels = Vec::new();

        // Package number
        channels.push(ChannelDesc {
            channel_type: ChannelType::PackageNum,
            row_index: 0,
            name: "Package".to_string(),
            unit: "".to_string(),
        });

        // EEG channels 1-8
        for i in 0..8 {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Eeg,
                row_index: 1 + i,
                name: format!("EEG_{}", i + 1),
                unit: "uV".to_string(),
            });
        }

        // Accelerometer
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Accel,
                row_index: 9 + i,
                name: format!("Accel_{}", axis),
                unit: "g".to_string(),
            });
        }

        // Analog channels
        for i in 0..3 {
            channels.push(ChannelDesc {
                channel_type: ChannelType::Analog,
                row_index: 12 + i,
                name: format!("Analog_{}", i + 1),
                unit: "mV".to_string(),
            });
        }

        // Timestamp
        channels.push(ChannelDesc {
            channel_type: ChannelType::Timestamp,
            row_index: 15,
            name: "Timestamp".to_string(),
            unit: "s".to_string(),
        });

        // Marker
        channels.push(ChannelDesc {
            channel_type: ChannelType::Marker,
            row_index: 16,
            name: "Marker".to_string(),
            unit: "".to_string(),
        });

        Self {
            board_id: BoardId::Cyton,
            num_rows: 17,
            channels,
            eeg_channels: (1..=8).collect(),
            fnirs_channels: vec![],
            emg_channels: vec![],
            eda_channels: vec![],
            accel_channels: vec![9, 10, 11],
            gyro_channels: vec![],
            timestamp_channel: 15,
            marker_channel: 16,
            package_num_channel: 0,
        }
    }

    /// Get EEG data from packet
    pub fn get_eeg(&self, packet: &BrainFlowPacket) -> Vec<f64> {
        self.eeg_channels
            .iter()
            .filter_map(|&idx| packet.data.get(idx).copied())
            .collect()
    }

    /// Get fNIRS data (HbO, HbR)
    pub fn get_fnirs(&self, packet: &BrainFlowPacket) -> (Vec<f64>, Vec<f64>) {
        let hbo: Vec<f64> = self.fnirs_channels[..4]
            .iter()
            .filter_map(|&idx| packet.data.get(idx).copied())
            .collect();
        let hbr: Vec<f64> = self.fnirs_channels[4..]
            .iter()
            .filter_map(|&idx| packet.data.get(idx).copied())
            .collect();
        (hbo, hbr)
    }

    /// Get timestamp from packet
    #[must_use]
    pub fn get_timestamp(&self, packet: &BrainFlowPacket) -> f64 {
        packet.data.get(self.timestamp_channel).copied().unwrap_or(0.0)
    }

    /// Get marker from packet
    #[must_use]
    pub fn get_marker(&self, packet: &BrainFlowPacket) -> i32 {
        packet.data.get(self.marker_channel).map(|&v| v as i32).unwrap_or(0)
    }
}

/// A single data packet in BrainFlow format
#[derive(Debug, Clone)]
pub struct BrainFlowPacket {
    /// Row data (channel values)
    pub data: Vec<f64>,
}

impl BrainFlowPacket {
    /// Create a new packet with specified number of rows
    #[must_use]
    pub fn new(num_rows: usize) -> Self {
        Self {
            data: vec![0.0; num_rows],
        }
    }

    /// Create a packet for Rootstar BCI
    #[must_use]
    pub fn rootstar() -> Self {
        Self::new(30)
    }

    /// Set package number
    pub fn set_package_num(&mut self, num: u32) {
        if !self.data.is_empty() {
            self.data[0] = num as f64;
        }
    }

    /// Set EEG channels (expects 8 values)
    pub fn set_eeg(&mut self, samples: &[f64]) {
        for (i, &sample) in samples.iter().take(8).enumerate() {
            if let Some(cell) = self.data.get_mut(1 + i) {
                *cell = sample;
            }
        }
    }

    /// Set fNIRS HbO channels (expects 4 values)
    pub fn set_fnirs_hbo(&mut self, samples: &[f64]) {
        for (i, &sample) in samples.iter().take(4).enumerate() {
            if let Some(cell) = self.data.get_mut(9 + i) {
                *cell = sample;
            }
        }
    }

    /// Set fNIRS HbR channels (expects 4 values)
    pub fn set_fnirs_hbr(&mut self, samples: &[f64]) {
        for (i, &sample) in samples.iter().take(4).enumerate() {
            if let Some(cell) = self.data.get_mut(13 + i) {
                *cell = sample;
            }
        }
    }

    /// Set EMG data
    pub fn set_emg(&mut self, envelope: f64, valence: f64) {
        if let Some(cell) = self.data.get_mut(17) {
            *cell = envelope;
        }
        if let Some(cell) = self.data.get_mut(18) {
            *cell = valence;
        }
    }

    /// Set EDA data
    pub fn set_eda(&mut self, scl: f64, scr: f64) {
        if let Some(cell) = self.data.get_mut(19) {
            *cell = scl;
        }
        if let Some(cell) = self.data.get_mut(20) {
            *cell = scr;
        }
    }

    /// Set accelerometer data
    pub fn set_accel(&mut self, x: f64, y: f64, z: f64) {
        if let Some(cell) = self.data.get_mut(21) {
            *cell = x;
        }
        if let Some(cell) = self.data.get_mut(22) {
            *cell = y;
        }
        if let Some(cell) = self.data.get_mut(23) {
            *cell = z;
        }
    }

    /// Set gyroscope data
    pub fn set_gyro(&mut self, x: f64, y: f64, z: f64) {
        if let Some(cell) = self.data.get_mut(24) {
            *cell = x;
        }
        if let Some(cell) = self.data.get_mut(25) {
            *cell = y;
        }
        if let Some(cell) = self.data.get_mut(26) {
            *cell = z;
        }
    }

    /// Set battery level (0-100)
    pub fn set_battery(&mut self, percent: f64) {
        if let Some(cell) = self.data.get_mut(27) {
            *cell = percent.clamp(0.0, 100.0);
        }
    }

    /// Set timestamp
    pub fn set_timestamp(&mut self, timestamp: f64) {
        if let Some(cell) = self.data.get_mut(28) {
            *cell = timestamp;
        }
    }

    /// Set marker
    pub fn set_marker(&mut self, marker: i32) {
        if let Some(cell) = self.data.get_mut(29) {
            *cell = marker as f64;
        }
    }

    /// Convert to CSV row
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        self.data
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse from CSV row
    pub fn from_csv_row(row: &str) -> Option<Self> {
        let data: Vec<f64> = row
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if data.is_empty() {
            None
        } else {
            Some(Self { data })
        }
    }
}

/// BrainFlow-compatible data buffer
pub struct BrainFlowBuffer {
    format: BrainFlowFormat,
    data: Vec<BrainFlowPacket>,
    capacity: usize,
    write_index: usize,
    sequence: u32,
}

impl BrainFlowBuffer {
    /// Create a new buffer
    #[must_use]
    pub fn new(format: BrainFlowFormat, capacity: usize) -> Self {
        Self {
            format,
            data: Vec::with_capacity(capacity),
            capacity,
            write_index: 0,
            sequence: 0,
        }
    }

    /// Add a packet to the buffer
    pub fn push(&mut self, mut packet: BrainFlowPacket) {
        packet.set_package_num(self.sequence);
        self.sequence = self.sequence.wrapping_add(1);

        if self.data.len() < self.capacity {
            self.data.push(packet);
        } else {
            self.data[self.write_index] = packet;
            self.write_index = (self.write_index + 1) % self.capacity;
        }
    }

    /// Get recent data as 2D array (BrainFlow format)
    #[must_use]
    pub fn get_board_data(&self, num_samples: usize) -> Vec<Vec<f64>> {
        let samples = num_samples.min(self.data.len());
        let start = if self.data.len() >= self.capacity {
            (self.write_index + self.capacity - samples) % self.capacity
        } else {
            self.data.len().saturating_sub(samples)
        };

        // Transpose to BrainFlow format (rows = channels, cols = samples)
        let mut result = vec![vec![0.0; samples]; self.format.num_rows];

        for (col, i) in (0..samples).enumerate() {
            let idx = (start + i) % self.data.len();
            if let Some(packet) = self.data.get(idx) {
                for (row, &val) in packet.data.iter().enumerate() {
                    if row < self.format.num_rows {
                        result[row][col] = val;
                    }
                }
            }
        }

        result
    }

    /// Get format descriptor
    #[must_use]
    pub fn format(&self) -> &BrainFlowFormat {
        &self.format
    }

    /// Get number of samples in buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.write_index = 0;
    }

    /// Export to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        let headers: Vec<_> = self.format.channels.iter().map(|c| c.name.as_str()).collect();
        csv.push_str(&headers.join(","));
        csv.push('\n');

        // Data rows
        for packet in &self.data {
            csv.push_str(&packet.to_csv_row());
            csv.push('\n');
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rootstar_format() {
        let format = BrainFlowFormat::rootstar_bci();
        assert_eq!(format.board_id, BoardId::RootstarBci);
        assert_eq!(format.num_rows, 30);
        assert_eq!(format.eeg_channels.len(), 8);
        assert_eq!(format.fnirs_channels.len(), 8);
    }

    #[test]
    fn test_packet_creation() {
        let mut packet = BrainFlowPacket::rootstar();
        packet.set_eeg(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        packet.set_timestamp(123.456);

        assert_eq!(packet.data[1], 1.0);
        assert_eq!(packet.data[8], 8.0);
        assert_eq!(packet.data[28], 123.456);
    }

    #[test]
    fn test_buffer() {
        let format = BrainFlowFormat::rootstar_bci();
        let mut buffer = BrainFlowBuffer::new(format, 100);

        for i in 0..10 {
            let mut packet = BrainFlowPacket::rootstar();
            packet.set_eeg(&[i as f64; 8]);
            buffer.push(packet);
        }

        assert_eq!(buffer.len(), 10);

        let data = buffer.get_board_data(5);
        assert_eq!(data.len(), 30);
        assert_eq!(data[0].len(), 5);
    }

    #[test]
    fn test_csv_roundtrip() {
        let mut packet = BrainFlowPacket::rootstar();
        packet.set_eeg(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let csv = packet.to_csv_row();
        let restored = BrainFlowPacket::from_csv_row(&csv).unwrap();

        assert_eq!(restored.data[1], 1.0);
        assert_eq!(restored.data[8], 8.0);
    }
}

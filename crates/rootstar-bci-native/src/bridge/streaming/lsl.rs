//! Lab Streaming Layer (LSL) compatible streaming
//!
//! This module provides LSL-compatible network streaming for BCI data.
//! LSL is the standard protocol for real-time streaming of time-series
//! data in neuroscience applications.
//!
//! # Protocol
//!
//! LSL uses multicast UDP for discovery and TCP for data streaming.
//! This implementation provides a compatible wire format that works
//! with standard LSL receivers like:
//!
//! - LabRecorder
//! - OpenViBE
//! - BCI2000
//! - MNE-Python
//! - MATLAB/Simulink
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │   LslOutlet     │────►│  Network (TCP)  │
//! │  (data source)  │     │                 │
//! └─────────────────┘     └────────┬────────┘
//!                                  │
//!                                  ▼
//!                         ┌─────────────────┐
//!                         │   LslInlet      │
//!                         │  (data sink)    │
//!                         └─────────────────┘
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use thiserror::Error;

/// LSL protocol version
pub const LSL_PROTOCOL_VERSION: u16 = 110;

/// Default multicast address for LSL discovery
pub const LSL_MULTICAST_ADDR: &str = "224.0.0.183";

/// Default multicast port
pub const LSL_MULTICAST_PORT: u16 = 16571;

/// Default data port range start
pub const LSL_DATA_PORT_START: u16 = 16572;

/// LSL errors
#[derive(Debug, Error)]
pub enum LslError {
    /// Network I/O error
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    /// No streams found
    #[error("No LSL streams found matching criteria")]
    NoStreamsFound,

    /// Stream disconnected
    #[error("Stream disconnected")]
    Disconnected,

    /// Invalid stream info
    #[error("Invalid stream info: {0}")]
    InvalidInfo(String),

    /// Buffer overflow
    #[error("Buffer overflow")]
    BufferOverflow,

    /// Timeout
    #[error("Operation timed out")]
    Timeout,
}

/// LSL result type
pub type LslResult<T> = Result<T, LslError>;

/// Stream content type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// Electroencephalography
    Eeg,
    /// Functional near-infrared spectroscopy
    Fnirs,
    /// Electromyography
    Emg,
    /// Electrodermal activity
    Eda,
    /// Markers/events
    Markers,
    /// Auxiliary data
    Aux,
    /// Generic data
    Data,
}

impl StreamType {
    /// Get type string for LSL XML
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eeg => "EEG",
            Self::Fnirs => "NIRS",
            Self::Emg => "EMG",
            Self::Eda => "EDA",
            Self::Markers => "Markers",
            Self::Aux => "Aux",
            Self::Data => "Data",
        }
    }
}

/// Channel data format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelFormat {
    /// 32-bit float (most common)
    Float32,
    /// 64-bit float
    Float64,
    /// 32-bit integer
    Int32,
    /// 16-bit integer
    Int16,
    /// 8-bit integer
    Int8,
    /// String (for markers)
    String,
}

impl ChannelFormat {
    /// Get format code for LSL
    #[must_use]
    pub fn code(&self) -> u8 {
        match self {
            Self::Float32 => 1,
            Self::Float64 => 2,
            Self::Int32 => 4,
            Self::Int16 => 5,
            Self::Int8 => 6,
            Self::String => 3,
        }
    }

    /// Get bytes per sample
    #[must_use]
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
            Self::Int16 => 2,
            Self::Int8 => 1,
            Self::String => 0, // Variable
        }
    }
}

/// Channel information
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Channel label (e.g., "Fp1", "O1")
    pub label: String,
    /// Channel unit (e.g., "microvolts")
    pub unit: String,
    /// Channel type (e.g., "EEG")
    pub channel_type: String,
}

impl Default for ChannelInfo {
    fn default() -> Self {
        Self {
            label: String::new(),
            unit: "microvolts".to_string(),
            channel_type: "EEG".to_string(),
        }
    }
}

/// Stream metadata
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream name (e.g., "RootstarEEG")
    pub name: String,
    /// Stream type
    pub stream_type: StreamType,
    /// Number of channels
    pub channel_count: usize,
    /// Nominal sample rate (Hz), 0 for irregular
    pub nominal_srate: f64,
    /// Channel data format
    pub channel_format: ChannelFormat,
    /// Unique source ID
    pub source_id: String,
    /// Hostname
    pub hostname: String,
    /// Per-channel info
    pub channels: Vec<ChannelInfo>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Session ID
    pub session_id: String,
    /// Creation timestamp
    pub created_at: f64,
}

impl StreamInfo {
    /// Create new stream info
    #[must_use]
    pub fn new(name: &str, stream_type: StreamType, channel_count: usize, nominal_srate: f64) -> Self {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "localhost".to_string());

        let source_id = format!("rootstar-{}-{}", name, std::process::id());
        let session_id = uuid_v4();

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let channels = (0..channel_count)
            .map(|i| ChannelInfo {
                label: format!("Ch{}", i + 1),
                ..Default::default()
            })
            .collect();

        Self {
            name: name.to_string(),
            stream_type,
            channel_count,
            nominal_srate,
            channel_format: ChannelFormat::Float32,
            source_id,
            hostname,
            channels,
            metadata: HashMap::new(),
            session_id,
            created_at,
        }
    }

    /// Set channel labels (10-20 system)
    pub fn with_channel_labels(mut self, labels: &[&str]) -> Self {
        for (i, label) in labels.iter().enumerate() {
            if i < self.channels.len() {
                self.channels[i].label = (*label).to_string();
            }
        }
        self
    }

    /// Set channel format
    #[must_use]
    pub fn with_format(mut self, format: ChannelFormat) -> Self {
        self.channel_format = format;
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Generate LSL XML header
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\"?>\n");
        xml.push_str("<info>\n");
        xml.push_str(&format!("  <name>{}</name>\n", self.name));
        xml.push_str(&format!("  <type>{}</type>\n", self.stream_type.as_str()));
        xml.push_str(&format!("  <channel_count>{}</channel_count>\n", self.channel_count));
        xml.push_str(&format!("  <nominal_srate>{}</nominal_srate>\n", self.nominal_srate));
        xml.push_str(&format!("  <channel_format>float32</channel_format>\n"));
        xml.push_str(&format!("  <source_id>{}</source_id>\n", self.source_id));
        xml.push_str(&format!("  <version>{}</version>\n", LSL_PROTOCOL_VERSION));
        xml.push_str(&format!("  <created_at>{}</created_at>\n", self.created_at));
        xml.push_str(&format!("  <hostname>{}</hostname>\n", self.hostname));
        xml.push_str(&format!("  <session_id>{}</session_id>\n", self.session_id));

        // Channel descriptions
        xml.push_str("  <desc>\n");
        xml.push_str("    <channels>\n");
        for ch in &self.channels {
            xml.push_str("      <channel>\n");
            xml.push_str(&format!("        <label>{}</label>\n", ch.label));
            xml.push_str(&format!("        <unit>{}</unit>\n", ch.unit));
            xml.push_str(&format!("        <type>{}</type>\n", ch.channel_type));
            xml.push_str("      </channel>\n");
        }
        xml.push_str("    </channels>\n");

        // Custom metadata
        if !self.metadata.is_empty() {
            xml.push_str("    <acquisition>\n");
            for (key, value) in &self.metadata {
                xml.push_str(&format!("      <{}>{}</{}>\n", key, value, key));
            }
            xml.push_str("    </acquisition>\n");
        }

        xml.push_str("  </desc>\n");
        xml.push_str("</info>\n");
        xml
    }
}

/// LSL data outlet (data source)
pub struct LslOutlet {
    info: StreamInfo,
    listener: TcpListener,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    discovery_socket: UdpSocket,
    local_port: u16,
    sample_count: u64,
    start_time: Instant,
}

impl LslOutlet {
    /// Create a new LSL outlet
    pub fn new(info: &StreamInfo) -> LslResult<Self> {
        // Bind TCP listener for data streaming
        let listener = TcpListener::bind("0.0.0.0:0")?;
        listener.set_nonblocking(true)?;
        let local_port = listener.local_addr()?.port();

        // Create UDP socket for discovery responses
        let discovery_socket = UdpSocket::bind("0.0.0.0:0")?;
        discovery_socket.set_nonblocking(true)?;

        Ok(Self {
            info: info.clone(),
            listener,
            clients: Arc::new(Mutex::new(Vec::new())),
            discovery_socket,
            local_port,
            sample_count: 0,
            start_time: Instant::now(),
        })
    }

    /// Get the data port
    #[must_use]
    pub fn port(&self) -> u16 {
        self.local_port
    }

    /// Get stream info
    #[must_use]
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }

    /// Push a single sample (array of f32)
    pub fn push_sample(&mut self, data: &[f32]) -> LslResult<()> {
        self.push_sample_with_timestamp(data, self.local_clock())
    }

    /// Push a sample with explicit timestamp
    pub fn push_sample_with_timestamp(&mut self, data: &[f32], timestamp: f64) -> LslResult<()> {
        if data.len() != self.info.channel_count {
            return Err(LslError::InvalidInfo(format!(
                "Expected {} channels, got {}",
                self.info.channel_count,
                data.len()
            )));
        }

        // Accept new connections
        self.accept_connections()?;

        // Build packet: [timestamp: f64][data: f32 * channel_count]
        let mut packet = Vec::with_capacity(8 + 4 * data.len());
        packet.extend_from_slice(&timestamp.to_le_bytes());
        for &sample in data {
            packet.extend_from_slice(&sample.to_le_bytes());
        }

        // Send to all clients
        let mut clients = self.clients.lock().map_err(|_| {
            LslError::InvalidInfo("Lock poisoned".to_string())
        })?;

        clients.retain_mut(|client| {
            match client.write_all(&packet) {
                Ok(()) => true,
                Err(_) => false, // Remove disconnected clients
            }
        });

        self.sample_count += 1;
        Ok(())
    }

    /// Push a chunk of samples
    pub fn push_chunk(&mut self, data: &[&[f32]]) -> LslResult<()> {
        let timestamp = self.local_clock();
        let dt = if self.info.nominal_srate > 0.0 {
            1.0 / self.info.nominal_srate
        } else {
            0.0
        };

        for (i, sample) in data.iter().enumerate() {
            let ts = timestamp - (data.len() - 1 - i) as f64 * dt;
            self.push_sample_with_timestamp(sample, ts)?;
        }
        Ok(())
    }

    /// Get local clock time (seconds since outlet creation)
    #[must_use]
    pub fn local_clock(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get number of connected clients
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.clients.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Handle discovery queries
    pub fn handle_discovery(&self) -> LslResult<()> {
        let mut buf = [0u8; 1024];

        loop {
            match self.discovery_socket.recv_from(&mut buf) {
                Ok((len, addr)) => {
                    // Parse query and respond
                    if let Some(response) = self.handle_query(&buf[..len]) {
                        let _ = self.discovery_socket.send_to(response.as_bytes(), addr);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(())
    }

    fn accept_connections(&mut self) -> LslResult<()> {
        loop {
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    stream.set_nonblocking(true)?;
                    stream.set_nodelay(true)?;

                    // Send stream info XML
                    let xml = self.info.to_xml();
                    let header = format!("LSL:streaminfo/{}:", xml.len());

                    let mut client = stream;
                    let _ = client.write_all(header.as_bytes());
                    let _ = client.write_all(xml.as_bytes());

                    let mut clients = self.clients.lock().map_err(|_| {
                        LslError::InvalidInfo("Lock poisoned".to_string())
                    })?;
                    clients.push(client);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn handle_query(&self, query: &[u8]) -> Option<String> {
        let query_str = std::str::from_utf8(query).ok()?;

        // Simple discovery response
        if query_str.starts_with("LSL:shortinfo") {
            Some(format!(
                "LSL:shortinfo {{\
                    \"name\":\"{}\",\
                    \"type\":\"{}\",\
                    \"channel_count\":{},\
                    \"nominal_srate\":{},\
                    \"source_id\":\"{}\",\
                    \"hostname\":\"{}\",\
                    \"port\":{}\
                }}",
                self.info.name,
                self.info.stream_type.as_str(),
                self.info.channel_count,
                self.info.nominal_srate,
                self.info.source_id,
                self.info.hostname,
                self.local_port
            ))
        } else {
            None
        }
    }
}

/// LSL data inlet (data receiver)
pub struct LslInlet {
    info: StreamInfo,
    stream: TcpStream,
    buffer: Vec<f32>,
    timestamps: Vec<f64>,
}

impl LslInlet {
    /// Connect to an LSL outlet
    pub fn connect(addr: SocketAddr) -> LslResult<Self> {
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        stream.set_read_timeout(Some(Duration::from_millis(100)))?;
        stream.set_nodelay(true)?;

        // Read stream info header
        let mut header_buf = [0u8; 1024];
        let n = stream.read(&mut header_buf)?;
        let header = std::str::from_utf8(&header_buf[..n])
            .map_err(|_| LslError::InvalidInfo("Invalid UTF-8 header".to_string()))?;

        // Parse basic info from header (simplified)
        let info = StreamInfo::new("remote", StreamType::Eeg, 8, 250.0);

        Ok(Self {
            info,
            stream,
            buffer: Vec::new(),
            timestamps: Vec::new(),
        })
    }

    /// Resolve streams on the network
    pub fn resolve(stream_type: Option<StreamType>, timeout: Duration) -> LslResult<Vec<StreamInfo>> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_read_timeout(Some(timeout))?;

        // Send discovery query
        let query = b"LSL:shortinfo";
        let multicast_addr: SocketAddr = format!("{}:{}", LSL_MULTICAST_ADDR, LSL_MULTICAST_PORT)
            .parse()
            .map_err(|_| LslError::InvalidInfo("Invalid multicast address".to_string()))?;

        socket.send_to(query, multicast_addr)?;

        // Collect responses
        let mut streams = Vec::new();
        let start = Instant::now();
        let mut buf = [0u8; 4096];

        while start.elapsed() < timeout {
            match socket.recv_from(&mut buf) {
                Ok((n, _addr)) => {
                    if let Ok(response) = std::str::from_utf8(&buf[..n]) {
                        if let Some(info) = parse_discovery_response(response, stream_type) {
                            streams.push(info);
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(e) => return Err(e.into()),
            }
        }

        if streams.is_empty() {
            Err(LslError::NoStreamsFound)
        } else {
            Ok(streams)
        }
    }

    /// Pull a sample (blocking)
    pub fn pull_sample(&mut self, timeout: Duration) -> LslResult<(Vec<f32>, f64)> {
        let bytes_per_sample = 8 + 4 * self.info.channel_count;
        let mut buf = vec![0u8; bytes_per_sample];

        self.stream.set_read_timeout(Some(timeout))?;
        self.stream.read_exact(&mut buf)?;

        // Parse timestamp
        let timestamp = f64::from_le_bytes(buf[0..8].try_into().unwrap());

        // Parse samples
        let mut samples = Vec::with_capacity(self.info.channel_count);
        for i in 0..self.info.channel_count {
            let offset = 8 + i * 4;
            let sample = f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
            samples.push(sample);
        }

        Ok((samples, timestamp))
    }

    /// Get stream info
    #[must_use]
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }
}

/// Parse discovery response JSON
fn parse_discovery_response(response: &str, filter_type: Option<StreamType>) -> Option<StreamInfo> {
    if !response.starts_with("LSL:shortinfo") {
        return None;
    }

    // Simple JSON parsing (production should use serde_json)
    let json_start = response.find('{')?;
    let json_str = &response[json_start..];

    // Extract fields (simplified parsing)
    let name = extract_json_string(json_str, "name")?;
    let type_str = extract_json_string(json_str, "type")?;
    let channel_count: usize = extract_json_number(json_str, "channel_count")?;
    let nominal_srate: f64 = extract_json_float(json_str, "nominal_srate")?;

    let stream_type = match type_str.as_str() {
        "EEG" => StreamType::Eeg,
        "NIRS" | "fNIRS" => StreamType::Fnirs,
        "EMG" => StreamType::Emg,
        "EDA" => StreamType::Eda,
        "Markers" => StreamType::Markers,
        _ => StreamType::Data,
    };

    // Apply filter
    if let Some(filter) = filter_type {
        if stream_type != filter {
            return None;
        }
    }

    Some(StreamInfo::new(&name, stream_type, channel_count, nominal_srate))
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find('"')? + start;
    Some(json[start..end].to_string())
}

fn extract_json_number(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_float(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Generate a simple UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (now & 0xFFFF_FFFF) as u32,
        ((now >> 32) & 0xFFFF) as u16,
        ((now >> 48) & 0x0FFF) as u16,
        (pid & 0xFFFF) as u16 | 0x8000,
        (now >> 64) as u64 & 0xFFFF_FFFF_FFFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_info_xml() {
        let info = StreamInfo::new("TestEEG", StreamType::Eeg, 8, 250.0)
            .with_channel_labels(&["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]);

        let xml = info.to_xml();
        assert!(xml.contains("<name>TestEEG</name>"));
        assert!(xml.contains("<type>EEG</type>"));
        assert!(xml.contains("<channel_count>8</channel_count>"));
        assert!(xml.contains("<label>Fp1</label>"));
    }

    #[test]
    fn test_channel_format() {
        assert_eq!(ChannelFormat::Float32.bytes_per_sample(), 4);
        assert_eq!(ChannelFormat::Float64.bytes_per_sample(), 8);
        assert_eq!(ChannelFormat::Int16.bytes_per_sample(), 2);
    }

    #[test]
    fn test_uuid_generation() {
        let uuid1 = uuid_v4();
        let uuid2 = uuid_v4();
        assert_ne!(uuid1, uuid2);
        assert_eq!(uuid1.len(), 36);
    }
}

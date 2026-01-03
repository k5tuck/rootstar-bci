//! Multi-device connection manager.
//!
//! This module provides a unified interface for managing multiple BCI device
//! connections, supporting both USB and Bluetooth Low Energy (BLE) transports.
//!
//! # Features
//!
//! - Device discovery (USB ports and BLE scanning)
//! - Connection pooling with automatic reconnection
//! - Packet routing based on device ID
//! - Per-device state tracking
//! - Event-driven architecture for multi-device data streams
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::bridge::{DeviceManager, DeviceEvent};
//!
//! // Create manager
//! let mut manager = DeviceManager::new();
//!
//! // Discover devices
//! let usb_devices = manager.discover_usb()?;
//! let ble_devices = manager.discover_ble().await?;
//!
//! // Connect to a device
//! manager.connect(&device_info)?;
//!
//! // Receive data from all devices
//! while let Some(event) = manager.next_event().await {
//!     match event {
//!         DeviceEvent::Data { device_id, packet } => { ... }
//!         DeviceEvent::Connected { device_id } => { ... }
//!         DeviceEvent::Disconnected { device_id, reason } => { ... }
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use rootstar_bci_core::protocol::{DeviceId, PacketHeaderV2, PacketType, ProtocolVersion};

/// Command to send to a device
#[derive(Debug)]
pub struct DeviceCommand {
    /// Packet type
    pub packet_type: PacketType,
    /// Payload data
    pub payload: Vec<u8>,
    /// Response channel for acknowledgement
    pub response_tx: oneshot::Sender<Result<(), String>>,
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during device management.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// Device not found
    #[error("Device not found: {0}")]
    NotFound(String),

    /// Device already connected
    #[error("Device already connected: {0}")]
    AlreadyConnected(DeviceId),

    /// Connection failed
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Device disconnected unexpectedly
    #[error("Device disconnected: {0}")]
    Disconnected(DeviceId),

    /// Communication error
    #[error("Communication error: {0}")]
    Communication(String),

    /// USB error
    #[cfg(feature = "usb")]
    #[error("USB error: {0}")]
    Usb(#[from] serialport::Error),

    /// BLE error
    #[cfg(feature = "ble")]
    #[error("BLE error: {0}")]
    Ble(String),

    /// Too many devices connected
    #[error("Maximum device limit reached ({0})")]
    TooManyDevices(usize),

    /// Device busy
    #[error("Device is busy")]
    Busy,
}

/// Result type for device operations.
pub type DeviceResult<T> = Result<T, DeviceError>;

// ============================================================================
// Device Information
// ============================================================================

/// Type of device connection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// USB/Serial connection
    Usb {
        /// Serial port path (e.g., "/dev/ttyUSB0" or "COM3")
        port: String,
        /// Baud rate
        baud_rate: u32,
    },
    /// Bluetooth Low Energy connection
    Ble {
        /// BLE device address
        address: String,
        /// Signal strength (RSSI) in dBm
        rssi: i16,
    },
}

/// Information about a discovered or connected device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub device_id: DeviceId,

    /// User-assigned friendly name
    pub name: String,

    /// Connection type and parameters
    pub connection_type: ConnectionType,

    /// Firmware version (if known)
    pub firmware_version: Option<String>,

    /// Hardware revision (if known)
    pub hardware_revision: Option<String>,

    /// Device capabilities
    pub capabilities: DeviceCapabilities,

    /// Protocol version supported
    pub protocol_version: ProtocolVersion,
}

impl DeviceInfo {
    /// Create a new device info for USB connection.
    #[must_use]
    pub fn new_usb(port: String, baud_rate: u32) -> Self {
        // Generate device ID from port hash
        let port_hash = Self::hash_string(&port);
        let device_id = DeviceId::from_u32(port_hash);

        Self {
            device_id,
            name: format!("BCI Device ({})", port),
            connection_type: ConnectionType::Usb { port, baud_rate },
            firmware_version: None,
            hardware_revision: None,
            capabilities: DeviceCapabilities::default(),
            protocol_version: ProtocolVersion::V2,
        }
    }

    /// Create a new device info for BLE connection.
    #[must_use]
    pub fn new_ble(address: String, rssi: i16) -> Self {
        // Generate device ID from address
        let addr_hash = Self::hash_string(&address);
        let device_id = DeviceId::from_u32(addr_hash);

        Self {
            device_id,
            name: format!("BCI Device (BLE:{})", &address[..8.min(address.len())]),
            connection_type: ConnectionType::Ble { address, rssi },
            firmware_version: None,
            hardware_revision: None,
            capabilities: DeviceCapabilities::default(),
            protocol_version: ProtocolVersion::V2,
        }
    }

    /// Simple string hash for generating device IDs.
    fn hash_string(s: &str) -> u32 {
        let mut hash: u32 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u32::from(byte));
        }
        hash
    }

    /// Check if this is a USB device.
    #[must_use]
    pub fn is_usb(&self) -> bool {
        matches!(self.connection_type, ConnectionType::Usb { .. })
    }

    /// Check if this is a BLE device.
    #[must_use]
    pub fn is_ble(&self) -> bool {
        matches!(self.connection_type, ConnectionType::Ble { .. })
    }
}

/// Device capabilities flags.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Number of EEG channels
    pub eeg_channels: u8,
    /// Number of fNIRS channels
    pub fnirs_channels: u8,
    /// Supports stimulation
    pub has_stimulation: bool,
    /// Supports impedance measurement
    pub has_impedance: bool,
    /// Supports EMG
    pub has_emg: bool,
    /// Supports EDA
    pub has_eda: bool,
    /// Maximum sample rate (Hz)
    pub max_sample_rate_hz: u16,
}

// ============================================================================
// Device State
// ============================================================================

/// Connection state of a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    /// Discovered but not connected
    Discovered,
    /// Currently connecting
    Connecting,
    /// Connected and ready
    Connected,
    /// Disconnecting
    Disconnecting,
    /// Disconnected (with optional reason)
    Disconnected { reason: Option<String> },
    /// Error state
    Error { message: String },
}

/// Runtime state for a connected device.
#[derive(Clone, Debug)]
pub struct DeviceState {
    /// Device information
    pub info: DeviceInfo,

    /// Current connection state
    pub connection_state: ConnectionState,

    /// When the device was connected
    pub connected_at: Option<Instant>,

    /// Last data received timestamp
    pub last_data_at: Option<Instant>,

    /// Packets received count
    pub packets_received: u64,

    /// Packets sent count
    pub packets_sent: u64,

    /// Packet errors count
    pub packet_errors: u64,

    /// Signal quality (0-100%)
    pub signal_quality: u8,

    /// User-assigned color for UI
    pub color: [u8; 3],

    /// Device is muted (data ignored)
    pub muted: bool,
}

impl DeviceState {
    /// Create a new device state from device info.
    #[must_use]
    pub fn new(info: DeviceInfo) -> Self {
        Self {
            info,
            connection_state: ConnectionState::Discovered,
            connected_at: None,
            last_data_at: None,
            packets_received: 0,
            packets_sent: 0,
            packet_errors: 0,
            signal_quality: 0,
            color: [0x00, 0x7A, 0xCC], // Default blue
            muted: false,
        }
    }

    /// Check if device is connected.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        matches!(self.connection_state, ConnectionState::Connected)
    }

    /// Get connection duration.
    #[must_use]
    pub fn connection_duration(&self) -> Option<Duration> {
        self.connected_at.map(|t| t.elapsed())
    }

    /// Get time since last data.
    #[must_use]
    pub fn time_since_data(&self) -> Option<Duration> {
        self.last_data_at.map(|t| t.elapsed())
    }
}

// ============================================================================
// Device Events
// ============================================================================

/// Events emitted by the device manager.
#[derive(Clone, Debug)]
pub enum DeviceEvent {
    /// Device was discovered during scan
    Discovered {
        device_id: DeviceId,
        info: DeviceInfo,
    },

    /// Device connection established
    Connected {
        device_id: DeviceId,
    },

    /// Device disconnected
    Disconnected {
        device_id: DeviceId,
        reason: Option<String>,
    },

    /// Data packet received from device
    Data {
        device_id: DeviceId,
        packet_type: PacketType,
        payload: Vec<u8>,
    },

    /// Error occurred for a device
    Error {
        device_id: DeviceId,
        message: String,
    },

    /// Device signal quality changed
    SignalQuality {
        device_id: DeviceId,
        quality: u8,
    },
}

// ============================================================================
// Device Manager
// ============================================================================

/// Maximum number of simultaneous device connections.
const MAX_DEVICES: usize = 16;

/// Configuration for the device manager.
#[derive(Clone, Debug)]
pub struct DeviceManagerConfig {
    /// Maximum number of devices
    pub max_devices: usize,

    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,

    /// Reconnect delay
    pub reconnect_delay: Duration,

    /// USB baud rate
    pub usb_baud_rate: u32,

    /// BLE scan duration
    pub ble_scan_duration: Duration,

    /// Data timeout (device considered disconnected if no data)
    pub data_timeout: Duration,
}

impl Default for DeviceManagerConfig {
    fn default() -> Self {
        Self {
            max_devices: MAX_DEVICES,
            auto_reconnect: true,
            reconnect_delay: Duration::from_secs(2),
            usb_baud_rate: 921_600,
            ble_scan_duration: Duration::from_secs(5),
            data_timeout: Duration::from_secs(5),
        }
    }
}

/// Multi-device connection manager.
///
/// Manages connections to multiple BCI devices simultaneously, providing
/// a unified interface for device discovery, connection, and data routing.
pub struct DeviceManager {
    /// Configuration
    config: DeviceManagerConfig,

    /// Connected devices by ID
    devices: Arc<RwLock<HashMap<DeviceId, DeviceState>>>,

    /// Command senders per device
    command_senders: Arc<RwLock<HashMap<DeviceId, mpsc::Sender<DeviceCommand>>>>,

    /// Event sender
    event_tx: mpsc::Sender<DeviceEvent>,

    /// Event receiver
    event_rx: mpsc::Receiver<DeviceEvent>,

    /// Next device color index (for auto-assignment)
    next_color_index: usize,
}

/// Predefined colors for devices.
const DEVICE_COLORS: [[u8; 3]; 8] = [
    [0x00, 0x7A, 0xCC], // Blue
    [0x00, 0xCC, 0x66], // Green
    [0xCC, 0x66, 0x00], // Orange
    [0xCC, 0x00, 0x66], // Pink
    [0x66, 0x00, 0xCC], // Purple
    [0x00, 0xCC, 0xCC], // Cyan
    [0xCC, 0xCC, 0x00], // Yellow
    [0xCC, 0x00, 0x00], // Red
];

impl DeviceManager {
    /// Create a new device manager with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DeviceManagerConfig::default())
    }

    /// Create a new device manager with custom configuration.
    #[must_use]
    pub fn with_config(config: DeviceManagerConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(256);

        Self {
            config,
            devices: Arc::new(RwLock::new(HashMap::new())),
            command_senders: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx,
            next_color_index: 0,
        }
    }

    /// Get the next event from any connected device.
    pub async fn next_event(&mut self) -> Option<DeviceEvent> {
        self.event_rx.recv().await
    }

    /// Get event sender for spawning device tasks.
    #[must_use]
    pub fn event_sender(&self) -> mpsc::Sender<DeviceEvent> {
        self.event_tx.clone()
    }

    /// Discover available USB devices.
    #[cfg(feature = "usb")]
    pub fn discover_usb(&self) -> DeviceResult<Vec<DeviceInfo>> {
        let ports = serialport::available_ports()?;
        let mut devices = Vec::new();

        for port in ports {
            let info = DeviceInfo::new_usb(port.port_name, self.config.usb_baud_rate);
            devices.push(info);
        }

        Ok(devices)
    }

    /// Discover available USB devices (stub when feature disabled).
    #[cfg(not(feature = "usb"))]
    pub fn discover_usb(&self) -> DeviceResult<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Discover available BLE devices.
    #[cfg(feature = "ble")]
    pub async fn discover_ble(&self) -> DeviceResult<Vec<DeviceInfo>> {
        use btleplug::api::{Central, Manager as _, ScanFilter};
        use btleplug::platform::Manager;

        let manager = Manager::new()
            .await
            .map_err(|e| DeviceError::Ble(e.to_string()))?;

        let adapters = manager
            .adapters()
            .await
            .map_err(|e| DeviceError::Ble(e.to_string()))?;

        let adapter = adapters
            .into_iter()
            .next()
            .ok_or_else(|| DeviceError::Ble("No Bluetooth adapter found".to_string()))?;

        // Start scanning
        adapter
            .start_scan(ScanFilter::default())
            .await
            .map_err(|e| DeviceError::Ble(e.to_string()))?;

        // Wait for scan duration
        tokio::time::sleep(self.config.ble_scan_duration).await;

        // Stop scanning
        adapter
            .stop_scan()
            .await
            .map_err(|e| DeviceError::Ble(e.to_string()))?;

        // Get discovered peripherals
        let peripherals = adapter
            .peripherals()
            .await
            .map_err(|e| DeviceError::Ble(e.to_string()))?;

        let mut devices = Vec::new();

        for peripheral in peripherals {
            let properties = peripheral
                .properties()
                .await
                .map_err(|e| DeviceError::Ble(e.to_string()))?;

            if let Some(props) = properties {
                // Filter for Rootstar BCI devices by name prefix
                if let Some(name) = &props.local_name {
                    if name.starts_with("Rootstar") || name.starts_with("BCI") {
                        let address = peripheral.id().to_string();
                        let rssi = props.rssi.unwrap_or(0);
                        let mut info = DeviceInfo::new_ble(address, rssi);
                        info.name = name.clone();
                        devices.push(info);
                    }
                }
            }
        }

        Ok(devices)
    }

    /// Discover available BLE devices (stub when feature disabled).
    #[cfg(not(feature = "ble"))]
    pub async fn discover_ble(&self) -> DeviceResult<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Connect to a device.
    pub async fn connect(&mut self, info: &DeviceInfo) -> DeviceResult<DeviceId> {
        // Check device limit
        let devices = self.devices.read().map_err(|_| DeviceError::Busy)?;
        let connected_count = devices.values().filter(|d| d.is_connected()).count();
        if connected_count >= self.config.max_devices {
            return Err(DeviceError::TooManyDevices(self.config.max_devices));
        }

        // Check if already connected
        if devices.get(&info.device_id).map(|d| d.is_connected()).unwrap_or(false) {
            return Err(DeviceError::AlreadyConnected(info.device_id));
        }
        drop(devices);

        // Create device state
        let mut state = DeviceState::new(info.clone());
        state.connection_state = ConnectionState::Connecting;
        state.color = DEVICE_COLORS[self.next_color_index % DEVICE_COLORS.len()];
        self.next_color_index += 1;

        // Store state
        {
            let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;
            devices.insert(info.device_id, state);
        }

        // Attempt connection based on type
        let result = match &info.connection_type {
            ConnectionType::Usb { port, baud_rate } => {
                self.connect_usb(info.device_id, port, *baud_rate).await
            }
            ConnectionType::Ble { address, .. } => {
                self.connect_ble(info.device_id, address).await
            }
        };

        match result {
            Ok(()) => {
                // Update state to connected
                let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;
                if let Some(state) = devices.get_mut(&info.device_id) {
                    state.connection_state = ConnectionState::Connected;
                    state.connected_at = Some(Instant::now());
                }

                // Send connected event
                let _ = self.event_tx.send(DeviceEvent::Connected {
                    device_id: info.device_id,
                }).await;

                Ok(info.device_id)
            }
            Err(e) => {
                // Update state to error
                let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;
                if let Some(state) = devices.get_mut(&info.device_id) {
                    state.connection_state = ConnectionState::Error {
                        message: e.to_string(),
                    };
                }
                Err(e)
            }
        }
    }

    /// Connect to USB device.
    #[cfg(feature = "usb")]
    async fn connect_usb(&self, device_id: DeviceId, port: &str, baud_rate: u32) -> DeviceResult<()> {
        use crate::bridge::usb::UsbBridge;

        // Open USB connection
        let bridge = UsbBridge::open(port, baud_rate)
            .map_err(|e| DeviceError::ConnectionFailed(e.to_string()))?;

        // Create command channel
        let (cmd_tx, cmd_rx) = mpsc::channel::<DeviceCommand>(32);

        // Store command sender
        {
            let mut senders = self.command_senders.write().map_err(|_| DeviceError::Busy)?;
            senders.insert(device_id, cmd_tx);
        }

        // Spawn reader/writer task
        let event_tx = self.event_tx.clone();
        let devices = Arc::clone(&self.devices);
        let command_senders = Arc::clone(&self.command_senders);

        tokio::spawn(async move {
            Self::usb_device_task(device_id, bridge, cmd_rx, event_tx, devices, command_senders).await;
        });

        Ok(())
    }

    /// USB reader task stub when feature disabled.
    #[cfg(not(feature = "usb"))]
    async fn connect_usb(&self, _device_id: DeviceId, _port: &str, _baud_rate: u32) -> DeviceResult<()> {
        Err(DeviceError::ConnectionFailed("USB support not enabled".to_string()))
    }

    /// USB device task - handles both reading and writing.
    #[cfg(feature = "usb")]
    async fn usb_device_task(
        device_id: DeviceId,
        mut bridge: crate::bridge::usb::UsbBridge,
        mut cmd_rx: mpsc::Receiver<DeviceCommand>,
        event_tx: mpsc::Sender<DeviceEvent>,
        devices: Arc<RwLock<HashMap<DeviceId, DeviceState>>>,
        command_senders: Arc<RwLock<HashMap<DeviceId, mpsc::Sender<DeviceCommand>>>>,
    ) {
        loop {
            // Check for commands first (non-blocking)
            while let Ok(cmd) = cmd_rx.try_recv() {
                let result = bridge.send_packet(cmd.packet_type, &cmd.payload);
                let response = match result {
                    Ok(()) => {
                        // Update packets sent count
                        if let Ok(mut devices) = devices.write() {
                            if let Some(state) = devices.get_mut(&device_id) {
                                state.packets_sent += 1;
                            }
                        }
                        Ok(())
                    }
                    Err(e) => Err(e.to_string()),
                };
                let _ = cmd.response_tx.send(response);
            }

            // Try to read a packet
            match bridge.read_packet() {
                Ok(Some((header, payload))) => {
                    // Update device state
                    if let Ok(mut devices) = devices.write() {
                        if let Some(state) = devices.get_mut(&device_id) {
                            state.last_data_at = Some(Instant::now());
                            state.packets_received += 1;
                        }
                    }

                    // Send data event
                    let _ = event_tx.send(DeviceEvent::Data {
                        device_id,
                        packet_type: header.packet_type,
                        payload,
                    }).await;
                }
                Ok(None) => {
                    // No data available, yield
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err(e) => {
                    // Update error count
                    if let Ok(mut devices) = devices.write() {
                        if let Some(state) = devices.get_mut(&device_id) {
                            state.packet_errors += 1;
                        }
                    }

                    // Check if fatal error
                    if e.to_string().contains("disconnected") {
                        let _ = event_tx.send(DeviceEvent::Disconnected {
                            device_id,
                            reason: Some(e.to_string()),
                        }).await;
                        break;
                    }
                }
            }
        }

        // Clean up command sender on disconnect
        if let Ok(mut senders) = command_senders.write() {
            senders.remove(&device_id);
        }
    }

    /// Connect to BLE device.
    #[cfg(feature = "ble")]
    async fn connect_ble(&self, _device_id: DeviceId, _address: &str) -> DeviceResult<()> {
        // BLE connection implementation
        // This would use btleplug to connect and set up GATT characteristics
        Err(DeviceError::Ble("BLE connection not yet implemented".to_string()))
    }

    /// BLE connection stub when feature disabled.
    #[cfg(not(feature = "ble"))]
    async fn connect_ble(&self, _device_id: DeviceId, _address: &str) -> DeviceResult<()> {
        Err(DeviceError::ConnectionFailed("BLE support not enabled".to_string()))
    }

    /// Disconnect from a device.
    pub async fn disconnect(&mut self, device_id: DeviceId) -> DeviceResult<()> {
        let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;

        if let Some(state) = devices.get_mut(&device_id) {
            state.connection_state = ConnectionState::Disconnected { reason: None };

            let _ = self.event_tx.send(DeviceEvent::Disconnected {
                device_id,
                reason: None,
            }).await;

            Ok(())
        } else {
            Err(DeviceError::NotFound(device_id.to_string()))
        }
    }

    /// Disconnect all devices.
    pub async fn disconnect_all(&mut self) {
        let device_ids: Vec<DeviceId> = {
            let devices = self.devices.read().unwrap();
            devices.keys().copied().collect()
        };

        for device_id in device_ids {
            let _ = self.disconnect(device_id).await;
        }
    }

    /// Get all device states.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceState> {
        self.devices.read()
            .map(|devices| devices.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get connected device count.
    #[must_use]
    pub fn connected_count(&self) -> usize {
        self.devices.read()
            .map(|devices| devices.values().filter(|d| d.is_connected()).count())
            .unwrap_or(0)
    }

    /// Get a specific device state.
    #[must_use]
    pub fn get_device(&self, device_id: DeviceId) -> Option<DeviceState> {
        self.devices.read()
            .ok()
            .and_then(|devices| devices.get(&device_id).cloned())
    }

    /// Update device name.
    pub fn set_device_name(&mut self, device_id: DeviceId, name: String) -> DeviceResult<()> {
        let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;

        if let Some(state) = devices.get_mut(&device_id) {
            state.info.name = name;
            Ok(())
        } else {
            Err(DeviceError::NotFound(device_id.to_string()))
        }
    }

    /// Update device color.
    pub fn set_device_color(&mut self, device_id: DeviceId, color: [u8; 3]) -> DeviceResult<()> {
        let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;

        if let Some(state) = devices.get_mut(&device_id) {
            state.color = color;
            Ok(())
        } else {
            Err(DeviceError::NotFound(device_id.to_string()))
        }
    }

    /// Mute/unmute a device (ignore its data).
    pub fn set_device_muted(&mut self, device_id: DeviceId, muted: bool) -> DeviceResult<()> {
        let mut devices = self.devices.write().map_err(|_| DeviceError::Busy)?;

        if let Some(state) = devices.get_mut(&device_id) {
            state.muted = muted;
            Ok(())
        } else {
            Err(DeviceError::NotFound(device_id.to_string()))
        }
    }

    /// Send a command to a specific device.
    pub async fn send_command(
        &self,
        device_id: DeviceId,
        packet_type: PacketType,
        payload: &[u8],
    ) -> DeviceResult<()> {
        // Check device is connected
        {
            let devices = self.devices.read().map_err(|_| DeviceError::Busy)?;

            let state = devices.get(&device_id)
                .ok_or_else(|| DeviceError::NotFound(device_id.to_string()))?;

            if !state.is_connected() {
                return Err(DeviceError::Disconnected(device_id));
            }
        }

        // Get command sender for this device
        let cmd_tx = {
            let senders = self.command_senders.read().map_err(|_| DeviceError::Busy)?;
            senders.get(&device_id).cloned()
        };

        let cmd_tx = cmd_tx.ok_or_else(|| DeviceError::Disconnected(device_id))?;

        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();

        // Build command
        let command = DeviceCommand {
            packet_type,
            payload: payload.to_vec(),
            response_tx,
        };

        // Send command to device task
        cmd_tx.send(command).await
            .map_err(|_| DeviceError::Disconnected(device_id))?;

        // Wait for response
        let result = response_rx.await
            .map_err(|_| DeviceError::Communication("Command response channel closed".to_string()))?;

        result.map_err(|e| DeviceError::Communication(e))
    }

    /// Broadcast a command to all connected devices.
    pub async fn broadcast_command(
        &self,
        packet_type: PacketType,
        payload: &[u8],
    ) -> DeviceResult<usize> {
        // Collect connected device IDs first to avoid holding lock during async operations
        let connected_ids: Vec<DeviceId> = {
            let devices = self.devices.read().map_err(|_| DeviceError::Busy)?;
            devices
                .iter()
                .filter(|(_, state)| state.is_connected())
                .map(|(id, _)| *id)
                .collect()
        };

        let mut sent_count = 0;
        for device_id in connected_ids {
            if self.send_command(device_id, packet_type, payload).await.is_ok() {
                sent_count += 1;
            }
        }

        Ok(sent_count)
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_usb() {
        let info = DeviceInfo::new_usb("/dev/ttyUSB0".to_string(), 921600);
        assert!(info.is_usb());
        assert!(!info.is_ble());
        assert!(!info.device_id.is_null());
    }

    #[test]
    fn test_device_info_ble() {
        let info = DeviceInfo::new_ble("A4:C1:38:00:11:22".to_string(), -45);
        assert!(!info.is_usb());
        assert!(info.is_ble());
        assert!(!info.device_id.is_null());
    }

    #[test]
    fn test_device_state() {
        let info = DeviceInfo::new_usb("/dev/ttyUSB0".to_string(), 921600);
        let state = DeviceState::new(info);

        assert!(!state.is_connected());
        assert!(state.connection_duration().is_none());
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert_eq!(manager.connected_count(), 0);
        assert!(manager.devices().is_empty());
    }

    #[tokio::test]
    async fn test_device_manager_config() {
        let config = DeviceManagerConfig {
            max_devices: 4,
            auto_reconnect: false,
            ..Default::default()
        };

        let manager = DeviceManager::with_config(config);
        assert_eq!(manager.connected_count(), 0);
    }
}

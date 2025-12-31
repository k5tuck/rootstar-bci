//! BLE bridge for wireless BCI communication
//!
//! Handles Bluetooth Low Energy communication with ESP32 BCI devices.
//!
//! # Service UUIDs
//!
//! The BCI device exposes the following BLE services:
//! - `00001800-0000-1000-8000-00805f9b34fb` - Generic Access (GAP)
//! - `0000180a-0000-1000-8000-00805f9b34fb` - Device Information
//! - `6e400001-b5a3-f393-e0a9-e50e24dcca9e` - Nordic UART Service (NUS)
//! - `12340001-1234-5678-9abc-def012345678` - BCI Data Service (custom)
//!
//! # Characteristics
//!
//! BCI Data Service:
//! - `12340002-...` - EEG Data (notify)
//! - `12340003-...` - fNIRS Data (notify)
//! - `12340004-...` - Command (write)
//! - `12340005-...` - Status (read/notify)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use btleplug::api::{
    Central, Characteristic, Manager as _, Peripheral as _, ScanFilter, WriteType,
};
use btleplug::platform::{Adapter, Manager, Peripheral};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::StreamExt;
use uuid::Uuid;

use rootstar_bci_core::protocol::{DeviceId, PacketType};

/// BCI service UUID (custom service)
pub const BCI_SERVICE_UUID: Uuid = Uuid::from_u128(0x12340001_1234_5678_9abc_def012345678);

/// EEG data characteristic UUID (notify)
pub const EEG_DATA_CHAR_UUID: Uuid = Uuid::from_u128(0x12340002_1234_5678_9abc_def012345678);

/// fNIRS data characteristic UUID (notify)
pub const FNIRS_DATA_CHAR_UUID: Uuid = Uuid::from_u128(0x12340003_1234_5678_9abc_def012345678);

/// Command characteristic UUID (write)
pub const COMMAND_CHAR_UUID: Uuid = Uuid::from_u128(0x12340004_1234_5678_9abc_def012345678);

/// Status characteristic UUID (read/notify)
pub const STATUS_CHAR_UUID: Uuid = Uuid::from_u128(0x12340005_1234_5678_9abc_def012345678);

/// EMG data characteristic UUID (notify)
pub const EMG_DATA_CHAR_UUID: Uuid = Uuid::from_u128(0x12340006_1234_5678_9abc_def012345678);

/// EDA data characteristic UUID (notify)
pub const EDA_DATA_CHAR_UUID: Uuid = Uuid::from_u128(0x12340007_1234_5678_9abc_def012345678);

/// BLE scan result
#[derive(Clone, Debug)]
pub struct BleDevice {
    /// Unique device identifier
    pub device_id: DeviceId,
    /// Device address (MAC address string)
    pub address: String,
    /// Device name (from advertisement)
    pub name: Option<String>,
    /// Signal strength (RSSI in dBm)
    pub rssi: i16,
    /// Whether device has BCI service
    pub has_bci_service: bool,
}

/// Events from BLE bridge
#[derive(Clone, Debug)]
pub enum BleEvent {
    /// Device discovered during scan
    DeviceDiscovered(BleDevice),
    /// Connected to device
    Connected { device_id: DeviceId },
    /// Disconnected from device
    Disconnected {
        device_id: DeviceId,
        reason: Option<String>,
    },
    /// Data received from device
    Data {
        device_id: DeviceId,
        packet_type: PacketType,
        payload: Vec<u8>,
    },
    /// Error occurred
    Error { device_id: DeviceId, message: String },
}

/// Active BLE connection
struct BleConnection {
    peripheral: Peripheral,
    device_id: DeviceId,
    eeg_char: Option<Characteristic>,
    fnirs_char: Option<Characteristic>,
    command_char: Option<Characteristic>,
    status_char: Option<Characteristic>,
    emg_char: Option<Characteristic>,
    eda_char: Option<Characteristic>,
    connected_at: Instant,
    sequence: u16,
}

/// BLE bridge for BCI devices
pub struct BleBridge {
    /// BLE adapter
    adapter: Adapter,
    /// Active connections
    connections: Arc<RwLock<HashMap<DeviceId, BleConnection>>>,
    /// Event sender
    event_tx: mpsc::Sender<BleEvent>,
    /// Scan duration
    scan_duration: Duration,
}

impl BleBridge {
    /// Create a new BLE bridge
    ///
    /// # Errors
    ///
    /// Returns error if Bluetooth is not available
    pub async fn new() -> Result<(Self, mpsc::Receiver<BleEvent>), anyhow::Error> {
        let manager = Manager::new().await?;
        let adapters = manager.adapters().await?;

        let adapter = adapters
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No Bluetooth adapter found"))?;

        let (event_tx, event_rx) = mpsc::channel(256);

        Ok((
            Self {
                adapter,
                connections: Arc::new(RwLock::new(HashMap::new())),
                event_tx,
                scan_duration: Duration::from_secs(5),
            },
            event_rx,
        ))
    }

    /// Set scan duration
    pub fn set_scan_duration(&mut self, duration: Duration) {
        self.scan_duration = duration;
    }

    /// Scan for BCI devices
    ///
    /// Returns a list of discovered devices with BCI service.
    pub async fn scan(&self) -> Result<Vec<BleDevice>, anyhow::Error> {
        tracing::info!("Starting BLE scan for BCI devices...");

        // Start scanning with filter for BCI service
        let filter = ScanFilter {
            services: vec![BCI_SERVICE_UUID],
        };
        self.adapter.start_scan(filter).await?;

        // Wait for scan duration
        tokio::time::sleep(self.scan_duration).await;

        // Stop scanning
        self.adapter.stop_scan().await?;

        // Collect discovered devices
        let peripherals = self.adapter.peripherals().await?;
        let mut devices = Vec::new();

        for peripheral in peripherals {
            if let Some(properties) = peripheral.properties().await? {
                let address = peripheral.address().to_string();
                let has_bci = properties
                    .services
                    .iter()
                    .any(|uuid| *uuid == BCI_SERVICE_UUID);

                // Generate device ID from address
                let device_id = DeviceId::from_u32(hash_address(&address));

                let device = BleDevice {
                    device_id,
                    address,
                    name: properties.local_name.clone(),
                    rssi: properties.rssi.unwrap_or(0),
                    has_bci_service: has_bci,
                };

                // Emit discovery event
                let _ = self.event_tx.send(BleEvent::DeviceDiscovered(device.clone())).await;

                if has_bci {
                    devices.push(device);
                }
            }
        }

        tracing::info!("Scan complete: found {} BCI devices", devices.len());
        Ok(devices)
    }

    /// Connect to a BCI device
    pub async fn connect(&self, device: &BleDevice) -> Result<(), anyhow::Error> {
        tracing::info!("Connecting to BLE device: {:?}", device.address);

        // Find peripheral by address
        let peripherals = self.adapter.peripherals().await?;
        let peripheral = peripherals
            .into_iter()
            .find(|p| p.address().to_string() == device.address)
            .ok_or_else(|| anyhow::anyhow!("Device not found: {}", device.address))?;

        // Connect
        peripheral.connect().await?;
        tracing::info!("Connected to {}", device.address);

        // Discover services
        peripheral.discover_services().await?;

        // Find characteristics
        let mut eeg_char = None;
        let mut fnirs_char = None;
        let mut command_char = None;
        let mut status_char = None;
        let mut emg_char = None;
        let mut eda_char = None;

        for service in peripheral.services() {
            for char in &service.characteristics {
                match char.uuid {
                    uuid if uuid == EEG_DATA_CHAR_UUID => eeg_char = Some(char.clone()),
                    uuid if uuid == FNIRS_DATA_CHAR_UUID => fnirs_char = Some(char.clone()),
                    uuid if uuid == COMMAND_CHAR_UUID => command_char = Some(char.clone()),
                    uuid if uuid == STATUS_CHAR_UUID => status_char = Some(char.clone()),
                    uuid if uuid == EMG_DATA_CHAR_UUID => emg_char = Some(char.clone()),
                    uuid if uuid == EDA_DATA_CHAR_UUID => eda_char = Some(char.clone()),
                    _ => {}
                }
            }
        }

        // Subscribe to notifications
        if let Some(ref char) = eeg_char {
            peripheral.subscribe(char).await?;
            tracing::debug!("Subscribed to EEG notifications");
        }
        if let Some(ref char) = fnirs_char {
            peripheral.subscribe(char).await?;
            tracing::debug!("Subscribed to fNIRS notifications");
        }
        if let Some(ref char) = status_char {
            peripheral.subscribe(char).await?;
            tracing::debug!("Subscribed to status notifications");
        }
        if let Some(ref char) = emg_char {
            peripheral.subscribe(char).await?;
            tracing::debug!("Subscribed to EMG notifications");
        }
        if let Some(ref char) = eda_char {
            peripheral.subscribe(char).await?;
            tracing::debug!("Subscribed to EDA notifications");
        }

        // Store connection
        let connection = BleConnection {
            peripheral: peripheral.clone(),
            device_id: device.device_id.clone(),
            eeg_char,
            fnirs_char,
            command_char,
            status_char,
            emg_char,
            eda_char,
            connected_at: Instant::now(),
            sequence: 0,
        };

        {
            let mut connections = self.connections.write().await;
            connections.insert(device.device_id.clone(), connection);
        }

        // Start notification handler
        self.spawn_notification_handler(device.device_id.clone(), peripheral)
            .await;

        // Emit connected event
        let _ = self
            .event_tx
            .send(BleEvent::Connected {
                device_id: device.device_id.clone(),
            })
            .await;

        Ok(())
    }

    /// Disconnect from a device
    pub async fn disconnect(&self, device_id: &DeviceId) -> Result<(), anyhow::Error> {
        let connection = {
            let mut connections = self.connections.write().await;
            connections.remove(device_id)
        };

        if let Some(conn) = connection {
            conn.peripheral.disconnect().await?;
            tracing::info!("Disconnected from {:?}", device_id);

            let _ = self
                .event_tx
                .send(BleEvent::Disconnected {
                    device_id: device_id.clone(),
                    reason: None,
                })
                .await;
        }

        Ok(())
    }

    /// Send a command to a device
    pub async fn send_command(
        &self,
        device_id: &DeviceId,
        command: &[u8],
    ) -> Result<(), anyhow::Error> {
        let connections = self.connections.read().await;
        let conn = connections
            .get(device_id)
            .ok_or_else(|| anyhow::anyhow!("Device not connected: {:?}", device_id))?;

        let char = conn
            .command_char
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Command characteristic not found"))?;

        conn.peripheral
            .write(char, command, WriteType::WithResponse)
            .await?;

        Ok(())
    }

    /// Start acquisition on a device
    pub async fn start_acquisition(&self, device_id: &DeviceId) -> Result<(), anyhow::Error> {
        // Command format: [0x01] = start acquisition
        self.send_command(device_id, &[0x01]).await
    }

    /// Stop acquisition on a device
    pub async fn stop_acquisition(&self, device_id: &DeviceId) -> Result<(), anyhow::Error> {
        // Command format: [0x02] = stop acquisition
        self.send_command(device_id, &[0x02]).await
    }

    /// Get connected device IDs
    pub async fn connected_devices(&self) -> Vec<DeviceId> {
        let connections = self.connections.read().await;
        connections.keys().cloned().collect()
    }

    /// Check if a device is connected
    pub async fn is_connected(&self, device_id: &DeviceId) -> bool {
        let connections = self.connections.read().await;
        connections.contains_key(device_id)
    }

    /// Spawn notification handler for a peripheral
    async fn spawn_notification_handler(&self, device_id: DeviceId, peripheral: Peripheral) {
        let event_tx = self.event_tx.clone();
        let connections = Arc::clone(&self.connections);

        tokio::spawn(async move {
            let mut stream = match peripheral.notifications().await {
                Ok(s) => s,
                Err(e) => {
                    let _ = event_tx
                        .send(BleEvent::Error {
                            device_id: device_id.clone(),
                            message: format!("Failed to get notification stream: {}", e),
                        })
                        .await;
                    return;
                }
            };

            while let Some(notification) = stream.next().await {
                // Determine packet type from characteristic UUID
                let packet_type = match notification.uuid {
                    uuid if uuid == EEG_DATA_CHAR_UUID => PacketType::EegData,
                    uuid if uuid == FNIRS_DATA_CHAR_UUID => PacketType::FnirsData,
                    uuid if uuid == STATUS_CHAR_UUID => PacketType::Status,
                    uuid if uuid == EMG_DATA_CHAR_UUID => PacketType::EmgData,
                    uuid if uuid == EDA_DATA_CHAR_UUID => PacketType::EdaData,
                    _ => continue,
                };

                let _ = event_tx
                    .send(BleEvent::Data {
                        device_id: device_id.clone(),
                        packet_type,
                        payload: notification.value,
                    })
                    .await;

                // Update sequence counter
                {
                    let mut conns = connections.write().await;
                    if let Some(conn) = conns.get_mut(&device_id) {
                        conn.sequence = conn.sequence.wrapping_add(1);
                    }
                }
            }

            // Stream ended, device disconnected
            {
                let mut conns = connections.write().await;
                conns.remove(&device_id);
            }

            let _ = event_tx
                .send(BleEvent::Disconnected {
                    device_id,
                    reason: Some("Notification stream ended".to_string()),
                })
                .await;
        });
    }
}

/// Hash a BLE address to a u32 for device ID
fn hash_address(address: &str) -> u32 {
    let mut hash: u32 = 5381;
    for byte in address.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(u32::from(byte));
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_address() {
        let addr1 = "AA:BB:CC:DD:EE:FF";
        let addr2 = "11:22:33:44:55:66";

        let hash1 = hash_address(addr1);
        let hash2 = hash_address(addr2);

        assert_ne!(hash1, hash2);
        assert_eq!(hash1, hash_address(addr1)); // Deterministic
    }

    #[test]
    fn test_service_uuids() {
        // Verify UUIDs are valid
        assert!(!BCI_SERVICE_UUID.is_nil());
        assert!(!EEG_DATA_CHAR_UUID.is_nil());
        assert!(!COMMAND_CHAR_UUID.is_nil());
    }
}

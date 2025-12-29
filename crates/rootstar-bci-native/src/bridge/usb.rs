//! USB bridge for ESP-EEG communication
//!
//! Handles serial communication with the ESP32 device over USB.

use std::io::{Read, Write};
use std::time::Duration;

use rootstar_bci_core::error::ProtocolError;
use rootstar_bci_core::protocol::{find_sync, PacketHeader, PacketType};
use rootstar_bci_core::types::EegSample;

/// USB connection to ESP-EEG device
pub struct UsbBridge {
    port: Box<dyn serialport::SerialPort>,
    read_buffer: Vec<u8>,
    sequence_tracker: u32,
}

impl UsbBridge {
    /// Open a USB connection to the ESP-EEG device
    ///
    /// # Arguments
    ///
    /// * `port_name` - Serial port name (e.g., "/dev/ttyUSB0" or "COM3")
    /// * `baud_rate` - Baud rate (typically 115200 or 921600)
    ///
    /// # Errors
    ///
    /// Returns error if port cannot be opened
    pub fn open(port_name: &str, baud_rate: u32) -> Result<Self, anyhow::Error> {
        let port = serialport::new(port_name, baud_rate)
            .timeout(Duration::from_millis(100))
            .open()?;

        Ok(Self {
            port,
            read_buffer: Vec::with_capacity(1024),
            sequence_tracker: 0,
        })
    }

    /// List available serial ports
    #[must_use]
    pub fn list_ports() -> Vec<String> {
        serialport::available_ports()
            .map(|ports| ports.into_iter().map(|p| p.port_name).collect())
            .unwrap_or_default()
    }

    /// Read and parse packets from the device
    ///
    /// Returns parsed samples or None if no complete packet available.
    pub fn read_packet(&mut self) -> Result<Option<(PacketType, Vec<u8>)>, anyhow::Error> {
        // Read available data
        let mut temp = [0u8; 256];
        match self.port.read(&mut temp) {
            Ok(n) if n > 0 => {
                self.read_buffer.extend_from_slice(&temp[..n]);
            }
            Ok(_) => return Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        // Look for sync bytes
        let Some(sync_pos) = find_sync(&self.read_buffer) else {
            // Discard bytes before any potential sync
            if self.read_buffer.len() > 256 {
                self.read_buffer.drain(..self.read_buffer.len() - 2);
            }
            return Ok(None);
        };

        // Discard any bytes before sync
        if sync_pos > 0 {
            self.read_buffer.drain(..sync_pos);
        }

        // Check if we have a complete header
        if self.read_buffer.len() < PacketHeader::SIZE {
            return Ok(None);
        }

        // Parse header
        let header = match PacketHeader::from_bytes(&self.read_buffer) {
            Ok(h) => h,
            Err(_) => {
                // Invalid header, skip sync bytes and try again
                self.read_buffer.drain(..2);
                return Ok(None);
            }
        };

        // Check if we have complete payload
        let total_size = PacketHeader::SIZE + header.payload_len as usize;
        if self.read_buffer.len() < total_size {
            return Ok(None);
        }

        // Extract payload
        let payload = self.read_buffer[PacketHeader::SIZE..total_size].to_vec();
        self.read_buffer.drain(..total_size);

        // Track sequence for gap detection
        if header.sequence != (self.sequence_tracker.wrapping_add(1) as u16)
            && self.sequence_tracker != 0
        {
            tracing::warn!(
                "Sequence gap: expected {}, got {}",
                self.sequence_tracker + 1,
                header.sequence
            );
        }
        self.sequence_tracker = u32::from(header.sequence);

        Ok(Some((header.packet_type, payload)))
    }

    /// Send a packet to the device
    pub fn send_packet(&mut self, packet_type: PacketType, payload: &[u8]) -> Result<(), anyhow::Error> {
        let header = PacketHeader::new(packet_type, 0, payload.len() as u16);
        let header_bytes = header.to_bytes();

        self.port.write_all(&header_bytes)?;
        self.port.write_all(payload)?;
        self.port.flush()?;

        Ok(())
    }

    /// Send start acquisition command
    pub fn start_acquisition(&mut self) -> Result<(), anyhow::Error> {
        self.send_packet(PacketType::StartAcquisition, &[])
    }

    /// Send stop acquisition command
    pub fn stop_acquisition(&mut self) -> Result<(), anyhow::Error> {
        self.send_packet(PacketType::StopAcquisition, &[])
    }
}

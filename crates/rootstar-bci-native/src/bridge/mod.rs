//! Communication bridges for BCI data streaming
//!
//! This module provides bridges to external systems:
//! - [`usb`]: Direct USB communication with ESP-EEG (requires `usb` feature)
//! - [`ble`]: Bluetooth Low Energy communication (requires `ble` feature)
//! - [`device_manager`]: Multi-device connection manager
//! - [`lsl`]: Lab Streaming Layer integration (placeholder)
//!
//! # Multi-Device Support
//!
//! The [`DeviceManager`] provides a unified interface for managing multiple
//! BCI device connections simultaneously. It supports both USB and BLE
//! transports, with automatic device discovery and reconnection.
//!
//! ```rust,ignore
//! use rootstar_bci_native::bridge::{DeviceManager, DeviceEvent};
//!
//! let mut manager = DeviceManager::new();
//!
//! // Discover and connect devices
//! for device in manager.discover_usb()? {
//!     manager.connect(&device).await?;
//! }
//!
//! // Receive data from all devices
//! while let Some(event) = manager.next_event().await {
//!     match event {
//!         DeviceEvent::Data { device_id, packet_type, payload } => { ... }
//!         _ => {}
//!     }
//! }
//! ```

#[cfg(feature = "usb")]
pub mod usb;

#[cfg(feature = "ble")]
pub mod ble;

pub mod device_manager;
pub mod device_context;
pub mod streaming;

// Re-export key types
pub use device_manager::{
    ConnectionState, ConnectionType, DeviceCapabilities, DeviceCommand, DeviceError, DeviceEvent,
    DeviceInfo, DeviceManager, DeviceManagerConfig, DeviceResult, DeviceState,
};

pub use device_context::{
    BufferConfig, BufferStats, ContextError, ContextResult, DataBuffers,
    DeviceContext, MultiDeviceContextManager, StimulationState,
};

#[cfg(feature = "database")]
pub use device_context::FingerprintRegistry;

#[cfg(feature = "usb")]
pub use usb::UsbBridge;

#[cfg(feature = "ble")]
pub use ble::{
    BleBridge, BleDevice, BleEvent,
    BCI_SERVICE_UUID, EEG_DATA_CHAR_UUID, FNIRS_DATA_CHAR_UUID,
    COMMAND_CHAR_UUID, STATUS_CHAR_UUID, EMG_DATA_CHAR_UUID, EDA_DATA_CHAR_UUID,
};

// Streaming protocols (LSL, OSC, BrainFlow)
pub use streaming::{
    LslOutlet, LslInlet, StreamInfo, StreamType, ChannelFormat,
    LslError, LslResult,
    BrainFlowFormat, BrainFlowPacket, BoardId,
};

#[cfg(feature = "osc")]
pub use streaming::{OscSender, OscReceiver, OscMessage, OscError, OscResult};

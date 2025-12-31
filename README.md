# Rootstar BCI Platform

A modular Brain-Computer Interface (BCI) platform built in Rust for multi-modal neural sensing and neurostimulation.

## Overview

Rootstar BCI is an embedded-first platform that combines:
- **EEG** (Electroencephalography) - 8-channel electrical brain signal acquisition
- **fNIRS** (functional Near-Infrared Spectroscopy) - Hemodynamic/oxygenation monitoring
- **EMG** (Electromyography) - Facial muscle activity sensing
- **EDA** (Electrodermal Activity) - Skin conductance monitoring
- **Neurostimulation** - tDCS/tACS/Photobiomodulation support
- **Multi-device** - Bluetooth LE connectivity and hyperscanning

## Architecture

The platform uses a tiered architecture:

| Tier | Crate | Description |
|------|-------|-------------|
| 0 | `rootstar-bci-core` | `no_std` compatible types, math, and protocols |
| 1 | `rootstar-bci-embedded` | ESP32 firmware and hardware drivers |
| 2 | `rootstar-bci-native` | Host-side processing, ML inference, visualization |
| 3 | `rootstar-bci-web` | WASM-based web interface and 3D visualization |

## Getting Started

### Prerequisites

- **Rust 1.85+** (Edition 2024)
- **Linux**: `sudo apt-get install libudev-dev pkg-config libdbus-1-dev`
- **macOS**: `brew install libusb`
- **Windows**: Visual C++ build tools

### Building

```bash
# Build entire workspace
cargo build

# Build with USB communication support
cargo build -p rootstar-bci-native --features usb

# Build with Bluetooth LE support
cargo build -p rootstar-bci-native --features ble

# Build with native visualization
cargo build -p rootstar-bci-native --features viz

# Build with external streaming (LSL, OSC, BrainFlow)
cargo build -p rootstar-bci-native --features streaming

# Build for embedded (ESP32)
cargo build -p rootstar-bci-embedded --profile embedded

# Build for web (WASM)
cargo build -p rootstar-bci-web --target wasm32-unknown-unknown

# Release build
cargo build --release
```

### Running

```bash
# Run tests
cargo test

# Run native application with visualization
cargo run -p rootstar-bci-native --features viz

# Run with USB device connection
cargo run -p rootstar-bci-native --features usb

# Run with all features
cargo run -p rootstar-bci-native --features "usb,ble,viz,streaming,database"
```

### ESP32 Firmware Flashing

```bash
# Install ESP toolchain
rustup target add xtensa-esp32-espidf
cargo install espup espflash

# Flash firmware
espflash flash --monitor target/xtensa-esp32-espidf/embedded/rootstar-bci-embedded
```

## BLE Command Protocol

The BCI device exposes a Bluetooth Low Energy GATT server with the following services and commands:

### GATT Services

| Service | UUID | Description |
|---------|------|-------------|
| BCI Data Service | `12340001-1234-5678-9abc-def012345678` | Main data streaming and control |
| Device Information | `0x180A` | Standard device info |
| Battery Service | `0x180F` | Battery level reporting |

### BLE Characteristics

| Characteristic | UUID | Properties | Description |
|---------------|------|------------|-------------|
| EEG Data | `12340002-...` | Notify | 8ch × 24-bit @ 250Hz |
| fNIRS Data | `12340003-...` | Notify | 4ch HbO/HbR @ 25Hz |
| Command | `12340004-...` | Write | Control commands |
| Status | `12340005-...` | Read, Notify | Device state |
| EMG Data | `12340006-...` | Notify | Envelope + features |
| EDA Data | `12340007-...` | Notify | SCL + SCR |
| Impedance | `12340008-...` | Notify | Per-channel impedance |
| Config | `12340009-...` | Read, Write | Device settings |
| Stimulation | `1234000A-...` | Write | tDCS/tACS control |
| Sync | `1234000B-...` | Notify | Hyperscanning sync |

### Command Reference

#### Acquisition Control (0x01-0x0F)

| Command | Code | Parameters | Description |
|---------|------|------------|-------------|
| `StartAcquisition` | `0x01` | None | Begin data streaming |
| `StopAcquisition` | `0x02` | None | Stop data streaming |
| `SetSampleRate` | `0x03` | `[rate_h, rate_l]` | Set sample rate (Hz) |
| `SetGain` | `0x04` | `[channel, gain_code]` | Set channel gain |
| `SetChannelMask` | `0x05` | `[mask_h, mask_l]` | Enable/disable channels |
| `TriggerSample` | `0x06` | None | Single sample capture |

#### Calibration (0x10-0x1F)

| Command | Code | Parameters | Description |
|---------|------|------------|-------------|
| `StartImpedance` | `0x10` | None | Begin impedance check |
| `StopImpedance` | `0x11` | None | End impedance check |
| `StartCalibration` | `0x12` | None | Begin signal calibration |
| `StopCalibration` | `0x13` | None | End calibration |
| `SetReference` | `0x14` | `[channel]` | Set reference channel |

#### Stimulation (0x20-0x2F)

| Command | Code | Parameters | Description |
|---------|------|------------|-------------|
| `StartStimulation` | `0x20` | `[protocol_id, params...]` | Begin stimulation |
| `StopStimulation` | `0x21` | None | Stop stimulation |
| `SetStimAmplitude` | `0x22` | `[amp_h, amp_l]` | Set amplitude (μA) |
| `SetStimFrequency` | `0x23` | `[freq_h, freq_l]` | Set frequency (0.01 Hz units) |
| `SetStimDuration` | `0x24` | `[dur_h, dur_l]` | Set duration (seconds) |
| `SetMontage` | `0x25` | `[anode_mask, cathode_mask]` | Set electrode montage |

#### Synchronization (0x30-0x3F)

| Command | Code | Parameters | Description |
|---------|------|------------|-------------|
| `SendSyncPulse` | `0x30` | None | Hyperscanning sync pulse |
| `SetDeviceId` | `0x31` | `[id_bytes...]` | Set device ID |
| `RequestTimeSync` | `0x32` | None | Request timestamp sync |
| `SetGroupId` | `0x33` | `[group_id]` | Set hyperscanning group |

#### System (0xF0-0xFF)

| Command | Code | Parameters | Description |
|---------|------|------------|-------------|
| `RequestStatus` | `0xF0` | None | Get device status |
| `RequestDeviceInfo` | `0xF1` | None | Get device info |
| `SetDeviceName` | `0xF2` | `[name_bytes...]` | Set device name |
| `SaveConfig` | `0xF3` | None | Save config to flash |
| `FactoryReset` | `0xF4` | None | Reset to defaults |
| `EnterBootloader` | `0xFE` | None | Enter DFU mode |
| `SoftReset` | `0xFF` | None | Software reset |

## External Application Integration

Rootstar BCI supports streaming to external applications similar to OpenBCI:

### Lab Streaming Layer (LSL)

Stream data to neuroscience applications (LabRecorder, OpenViBE, BCI2000, MNE-Python):

```rust
use rootstar_bci_native::bridge::streaming::{LslOutlet, StreamInfo, StreamType};

// Create an EEG stream
let info = StreamInfo::new("RootstarEEG", StreamType::Eeg, 8, 250.0)
    .with_channel_labels(&["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]);

let mut outlet = LslOutlet::new(&info)?;

// Push samples as they arrive
outlet.push_sample(&eeg_data)?;

// Or push with explicit timestamp
outlet.push_sample_with_timestamp(&eeg_data, timestamp)?;
```

### Open Sound Control (OSC)

Stream to audio/music applications (Max/MSP, Pure Data, SuperCollider, TouchDesigner):

```rust
use rootstar_bci_native::bridge::streaming::{OscSender, OscMessage};

let sender = OscSender::new("127.0.0.1:9000")?
    .with_prefix("/bci");

// Send EEG band powers
sender.send_band_powers("alpha", &[0.5, 0.6, 0.4, 0.7])?;
sender.send_band_powers("beta", &[0.3, 0.4, 0.3, 0.5])?;

// Send fNIRS data
sender.send_fnirs(&hbo, &hbr)?;

// Send EMG/EDA metrics
sender.send_emg(envelope, valence)?;
sender.send_eda(scl, scr)?;

// Send cognitive metrics
sender.send_attention(0.8)?;
sender.send_relaxation(0.6)?;

// Send event markers
sender.send_marker("stimulus_onset")?;
```

**Default OSC Address Space:**
```
/bci/eeg/raw          - Raw EEG samples (8 floats)
/bci/eeg/alpha        - Alpha band power per channel
/bci/eeg/beta         - Beta band power per channel
/bci/eeg/theta        - Theta band power per channel
/bci/eeg/gamma        - Gamma band power per channel
/bci/fnirs/hbo        - HbO concentration (4 floats)
/bci/fnirs/hbr        - HbR concentration (4 floats)
/bci/emg/envelope     - EMG envelope value
/bci/emg/valence      - Facial expression valence
/bci/eda/scl          - Skin conductance level
/bci/eda/scr          - SCR (phasic) response
/bci/markers/event    - Event markers (string)
/bci/attention        - Attention metric (0-1)
/bci/relaxation       - Relaxation metric (0-1)
```

### BrainFlow Format

Export data in BrainFlow-compatible format for OpenBCI ecosystem:

```rust
use rootstar_bci_native::bridge::streaming::{BrainFlowFormat, BrainFlowPacket, BrainFlowBuffer};

// Create Rootstar format (30 channels)
let format = BrainFlowFormat::rootstar_bci();

// Build packets
let mut packet = BrainFlowPacket::rootstar();
packet.set_eeg(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
packet.set_fnirs_hbo(&[0.5, 0.6, 0.4, 0.3]);
packet.set_fnirs_hbr(&[-0.2, -0.1, -0.3, -0.2]);
packet.set_emg(150.0, 0.5);  // envelope, valence
packet.set_eda(2.5, 0.8);    // SCL, SCR
packet.set_timestamp(1234.567);

// Buffer for recording
let mut buffer = BrainFlowBuffer::new(format, 10000);
buffer.push(packet);

// Export to CSV
let csv = buffer.to_csv();

// Get as 2D array (BrainFlow format)
let data = buffer.get_board_data(1000);  // Last 1000 samples
```

**Rootstar BrainFlow Channel Layout (30 rows):**
```
0:      Package number
1-8:    EEG channels (Fp1, Fp2, F3, F4, C3, C4, O1, O2)
9-12:   fNIRS HbO (4 channels)
13-16:  fNIRS HbR (4 channels)
17:     EMG envelope
18:     EMG valence
19:     EDA SCL
20:     EDA SCR
21-23:  Accelerometer (X, Y, Z)
24-26:  Gyroscope (X, Y, Z)
27:     Battery level
28:     Timestamp
29:     Marker
```

## Multi-Device & Hyperscanning

Support for multiple simultaneous devices with clock synchronization:

```rust
use rootstar_bci_native::bridge::DeviceManager;
use rootstar_bci_native::processing::HyperscanningSession;

// Discover and connect multiple devices
let manager = DeviceManager::new();
manager.scan_ble().await?;

// Connect to discovered devices
for device in manager.discovered_devices() {
    manager.connect(device.id).await?;
}

// Create hyperscanning session
let mut session = HyperscanningSession::with_defaults(2);  // 2 participants
session.set_reference_device(device1_id)?;

// Add data with timestamps for synchronization
session.add_sample(device1_id, eeg_sample1, timestamp1)?;
session.add_sample(device2_id, eeg_sample2, timestamp2)?;

// Calculate inter-brain coherence
let coherence = session.compute_coherence(FrequencyBand::Alpha)?;
```

## Native Visualization

The visualization module provides real-time signal display with electrode status:

```rust
use rootstar_bci_native::viz::{
    MultiDeviceDashboard, ViewMode, ElectrodeStatusPanel
};

// Create dashboard
let mut dashboard = MultiDeviceDashboard::new();
dashboard.add_device(device_id, device_info);

// Set view mode
dashboard.set_view_mode(ViewMode::Grid);  // Single, SideBySide, Overlay, Grid

// Update electrode impedance
if let Some(device) = dashboard.device_mut(&device_id) {
    device.update_impedances(&[15.0, 12.0, 18.0, 14.0, 20.0, 16.0, 22.0, 19.0]);
}

// Render (in egui context)
dashboard.ui(ctx);
```

**Electrode Status Panel Features:**
- 10-20 system head diagram
- Real-time connection status (green/yellow/red)
- Impedance values per electrode
- Signal quality indicators
- Interactive electrode selection

## Hardware Requirements

For the complete hardware shopping list with part numbers, suppliers, and assembly instructions, see **[HARDWARE.md](HARDWARE.md)**.

### Quick Overview

**Estimated Total Cost:** $250-600 (depending on electrode quality and optional components)

| Category | Components | Price Range |
|----------|------------|-------------|
| **Core Electronics** | ESP32-WROOM-DA, ADS1299 (EEG), ADS1115 (fNIRS) | $55-85 |
| **fNIRS Optics** | 760nm/850nm NIR LEDs, OPT101 photodetectors | $25-50 |
| **EEG Electrodes** | Ag/AgCl electrodes (8+), conductive gel | $75-195 |
| **Headset Option A** | Pre-made electrode cap (optional) | $50-200 |
| **Headset Option B** | 3D printed DIY headset (optional) | ~$100 |
| **Neurostimulation** | DAC8564, Howland current source (optional) | $45-85 |
| **Passive Components** | Capacitors, resistors, wiring, breadboard | $40-50 |

See [HARDWARE.md](HARDWARE.md) for 3D printed headset designs including adjustable band and ergonomic shell form factors.

### GPIO Pin Assignments (ESP32)

```
SPI (ADS1299 EEG):
  MOSI  = GPIO 23
  MISO  = GPIO 19
  SCLK  = GPIO 18
  CS    = GPIO 5
  DRDY  = GPIO 4 (data ready interrupt)

I2C (ADS1115 fNIRS):
  SDA   = GPIO 21
  SCL   = GPIO 22

PWM (NIR LEDs):
  760nm = GPIO 25, 26
  850nm = GPIO 27

DAC (Stimulation):
  OUT   = GPIO 32, 33

Status LED:
  LED   = GPIO 17
```

### EEG Channel Mapping (10-20 System)

| Channel | Position | Brain Region | Function |
|---------|----------|--------------|----------|
| 0 | Fp1 | Left Prefrontal | Executive function, attention |
| 1 | Fp2 | Right Prefrontal | Executive function, attention |
| 2 | F3 | Left Frontal | Motor planning |
| 3 | F4 | Right Frontal | Motor planning |
| 4 | C3 | Left Central | Motor control (right body) |
| 5 | C4 | Right Central | Motor control (left body) |
| 6 | O1 | Left Occipital | Visual processing |
| 7 | O2 | Right Occipital | Visual processing |

## Safety Considerations

**IMPORTANT: Neurostimulation Safety Limits**

The platform enforces these hardware-level safety limits:
- Maximum current: **2 mA (2000 μA)**
- Maximum duration: **30 minutes**
- Maximum tACS frequency: **100 Hz**
- Minimum ramp time: **10 ms**

> Software limits alone are NOT sufficient. Always implement hardware current limiting via a Howland current source circuit with physical current clamping.

## Project Structure

```
rootstar-bci/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── rootstar-bci-core/       # Tier 0: no_std core types
│   │   └── src/
│   │       ├── types.rs         # Fixed24.8, samples, channels
│   │       ├── error.rs         # Error handling
│   │       ├── math.rs          # Filters, Beer-Lambert
│   │       ├── protocol.rs      # Serial protocol
│   │       └── sns/             # Spiking Neural Stochastic
│   │
│   ├── rootstar-bci-embedded/   # Tier 1: ESP32 firmware
│   │   └── src/
│   │       ├── ble.rs           # BLE GATT server & protocol
│   │       └── drivers/
│   │           ├── ads1299.rs   # EEG ADC driver
│   │           ├── fnirs.rs     # fNIRS frontend
│   │           ├── emg.rs       # EMG driver
│   │           ├── eda.rs       # EDA driver
│   │           └── stim.rs      # Neurostimulation
│   │
│   ├── rootstar-bci-native/     # Tier 2: Host processing
│   │   └── src/
│   │       ├── bridge/          # Device communication
│   │       │   ├── usb.rs       # USB/Serial
│   │       │   ├── ble.rs       # Bluetooth LE
│   │       │   ├── device_manager.rs
│   │       │   └── streaming/   # External protocols
│   │       │       ├── lsl.rs   # Lab Streaming Layer
│   │       │       ├── osc.rs   # Open Sound Control
│   │       │       └── brainflow.rs
│   │       ├── processing/      # Signal processing
│   │       │   ├── filters.rs
│   │       │   ├── fft.rs
│   │       │   ├── fusion.rs
│   │       │   └── hyperscanning.rs
│   │       ├── ml/              # Feature extraction
│   │       ├── sns/             # SNS encoding/decoding
│   │       └── viz/             # Native visualization
│   │           ├── dashboard.rs
│   │           └── electrode_status.rs
│   │
│   └── rootstar-bci-web/        # Tier 3: WASM web interface
│       └── src/
│           └── sns_viz/         # 3D visualization
```

## Technical Specifications

### EEG (ADS1299)
- Resolution: 24-bit
- Channels: 8
- Sample rates: 250, 500, 1000, 2000 Hz
- Programmable gain: 1x, 2x, 4x, 6x, 8x, 12x, 24x
- LSB: 0.0223 μV @ 24x gain
- Interface: SPI

### fNIRS (ADS1115)
- Resolution: 16-bit
- Sample rate: 860 SPS
- Range: +/- 4.096V
- Wavelengths: 760nm (HbR), 850nm (HbO2)
- Interface: I2C

### BLE (ESP32)
- BLE 5.0 with 2M PHY
- Data throughput: ~7.3 KB/sec (all modalities)
- MTU: 247 bytes (negotiated)

### Communication Protocol
- Format: Postcard (compact binary)
- Checksum: CRC-based
- Baud rates: 115200, 921600 bps

## Features

- [x] `no_std` compatible core for embedded systems
- [x] Fixed-point math (Q24.8) for embedded constraints
- [x] Multi-modal sensing (EEG + fNIRS + EMG + EDA)
- [x] Real-time signal processing
- [x] Beer-Lambert hemodynamic calculation
- [x] IIR/Biquad digital filtering
- [x] FFT spectral analysis
- [x] EEG + fNIRS data fusion
- [x] USB/Serial device communication
- [x] Bluetooth LE connectivity
- [x] Multi-device support & hyperscanning
- [x] Native GPU visualization (wgpu/egui)
- [x] Electrode status visualization
- [x] WASM web visualization
- [x] Lab Streaming Layer (LSL) integration
- [x] OSC streaming for audio apps
- [x] BrainFlow-compatible format
- [ ] ONNX ML inference
- [ ] Full neurostimulation implementation

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

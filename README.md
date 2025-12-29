# Rootstar BCI Platform

A modular Brain-Computer Interface (BCI) platform built in Rust for multi-modal neural sensing and neurostimulation.

## Overview

Rootstar BCI is an embedded-first platform that combines:
- **EEG** (Electroencephalography) - 8-channel electrical brain signal acquisition
- **fNIRS** (functional Near-Infrared Spectroscopy) - Hemodynamic/oxygenation monitoring
- **Neurostimulation** - tDCS/tACS/Photobiomodulation support

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
- **Linux**: `sudo apt-get install libudev-dev pkg-config`
- **macOS**: `brew install libusb`
- **Windows**: Visual C++ build tools

### Building

```bash
# Build entire workspace
cargo build

# Build with USB communication support
cargo build -p rootstar-bci-native --features usb

# Build with native visualization
cargo build -p rootstar-bci-native --features viz

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
```

### ESP32 Firmware Flashing

```bash
# Install ESP toolchain
rustup target add xtensa-esp32-espidf
cargo install espup espflash

# Flash firmware
espflash flash --monitor target/xtensa-esp32-espidf/embedded/rootstar-bci-embedded
```

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
| 2 | C3 | Left Central | Motor control (right body) |
| 3 | C4 | Right Central | Motor control (left body) |
| 4 | P3 | Left Parietal | Sensory integration |
| 5 | P4 | Right Parietal | Sensory integration |
| 6 | O1 | Left Occipital | Visual processing |
| 7 | O2 | Right Occipital | Visual processing |

## Safety Considerations

**IMPORTANT: Neurostimulation Safety Limits**

The platform enforces these hardware-level safety limits:
- Maximum current: **2 mA (2000 uA)**
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
│   │   └── src/drivers/
│   │       ├── ads1299.rs       # EEG ADC driver
│   │       ├── fnirs.rs         # fNIRS frontend
│   │       └── stim.rs          # Neurostimulation
│   │
│   ├── rootstar-bci-native/     # Tier 2: Host processing
│   │   └── src/
│   │       ├── bridge/          # USB/Serial communication
│   │       ├── processing/      # Filters, FFT, fusion
│   │       ├── ml/              # Feature extraction
│   │       ├── sns/             # SNS encoding/decoding
│   │       └── viz/             # Native visualization
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
- LSB: 0.0223 uV @ 24x gain
- Interface: SPI

### fNIRS (ADS1115)
- Resolution: 16-bit
- Sample rate: 860 SPS
- Range: +/- 4.096V
- Wavelengths: 760nm (HbR), 850nm (HbO2)
- Interface: I2C

### Communication Protocol
- Format: Postcard (compact binary)
- Checksum: CRC-based
- Baud rates: 115200, 921600 bps

## Features

- [x] `no_std` compatible core for embedded systems
- [x] Fixed-point math (Q24.8) for embedded constraints
- [x] Multi-modal sensing (EEG + fNIRS)
- [x] Real-time signal processing
- [x] Beer-Lambert hemodynamic calculation
- [x] IIR/Biquad digital filtering
- [x] FFT spectral analysis
- [x] EEG + fNIRS data fusion
- [x] USB/Serial device communication
- [x] Native GPU visualization (wgpu/egui)
- [x] WASM web visualization
- [ ] ONNX ML inference (placeholder)
- [ ] Full neurostimulation implementation

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

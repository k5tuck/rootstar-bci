# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

Rootstar BCI is a Brain-Computer Interface platform written in Rust. It uses a tiered architecture for multi-modal neural sensing (EEG + fNIRS) and neurostimulation (tDCS/tACS).

## Build Commands

```bash
# Build entire workspace
cargo build

# Run tests
cargo test

# Build specific crates with features
cargo build -p rootstar-bci-native --features usb    # USB communication
cargo build -p rootstar-bci-native --features viz    # Visualization
cargo build -p rootstar-bci-embedded --profile embedded  # ESP32 firmware
cargo build -p rootstar-bci-web --target wasm32-unknown-unknown  # WASM

# Release build
cargo build --release
```

## Architecture

The codebase is organized into 4 tiers:

- **Tier 0 (`rootstar-bci-core`)**: `no_std` compatible. Contains core types (Fixed24.8, EegSample, FnirsSample), error handling, math (filters, Beer-Lambert), and serial protocol.

- **Tier 1 (`rootstar-bci-embedded`)**: ESP32 firmware. Hardware drivers for ADS1299 (EEG), ADS1115 (fNIRS ADC), NIR LEDs, and neurostimulation.

- **Tier 2 (`rootstar-bci-native`)**: Host-side processing. USB/Serial bridge, signal processing (FFT, filters), EEG+fNIRS fusion, ML features, and native visualization.

- **Tier 3 (`rootstar-bci-web`)**: WASM web interface with Leptos and 3D visualization.

## Key Directories

```
crates/
├── rootstar-bci-core/src/
│   ├── types.rs      # Core data types, Fixed24.8, samples
│   ├── error.rs      # Error types for all subsystems
│   ├── math.rs       # IIR/Biquad filters, Beer-Lambert solver
│   ├── protocol.rs   # Serial packet format and CRC
│   └── sns/          # Spiking Neural Stochastic encoding
│
├── rootstar-bci-embedded/src/drivers/
│   ├── ads1299.rs    # EEG ADC driver (SPI)
│   ├── fnirs.rs      # fNIRS optical frontend (I2C + PWM)
│   └── stim.rs       # Neurostimulation driver (DAC)
│
├── rootstar-bci-native/src/
│   ├── bridge/usb.rs # Serial/USB communication
│   ├── processing/   # Filters, FFT, fusion
│   ├── ml/           # Feature extraction
│   ├── sns/          # Host-side SNS processing
│   └── viz/          # Native GUI (egui/wgpu)
│
└── rootstar-bci-web/src/
    └── sns_viz/      # Web 3D visualization
```

## Hardware Context

- **EEG**: TI ADS1299 (8-channel, 24-bit, SPI)
- **fNIRS**: TI ADS1115 + 760nm/850nm LEDs + OPT101 photodetectors
- **MCU**: ESP32-WROOM-DA
- **Stimulation**: DAC8564 (16-bit quad DAC) + Howland current source

GPIO assignments: SPI on 23/19/18/5/4, I2C on 21/22, PWM on 25-27, DAC on 32-33.

## Code Conventions

- Edition 2024, MSRV 1.85
- `no_std` in core crate (use `heapless` collections, `libm` math)
- Fixed-point math using `Fixed24_8` (Q24.8 format)
- Async-first: Embassy on embedded, Tokio on native
- Strict lints: pedantic clippy, no `todo!()` or `unimplemented!()`
- Safety limits for neurostimulation are enforced in code (max 2mA, 30min, 100Hz)

## Common Tasks

### Adding a new EEG processing algorithm
1. Add filter/algorithm to `crates/rootstar-bci-native/src/processing/`
2. Use `rustfft` for spectral analysis
3. Integrate with fusion pipeline in `fusion.rs`

### Modifying hardware drivers
1. Edit drivers in `crates/rootstar-bci-embedded/src/drivers/`
2. Use `embedded-hal` traits for portability
3. Test with `cargo build -p rootstar-bci-embedded`

### Adding new packet types
1. Add variant to `PacketType` enum in `crates/rootstar-bci-core/src/protocol.rs`
2. Implement serialization with `postcard`
3. Update both embedded and native sides

### Running with hardware
```bash
# With USB device connected
cargo run -p rootstar-bci-native --features usb

# Monitor serial output
espflash monitor
```

## Dependencies to Know

- `heapless`: No-allocation collections for `no_std`
- `postcard`: Compact binary serialization
- `embedded-hal`: Hardware abstraction traits
- `esp-hal`: ESP32 HAL
- `embassy-*`: Async embedded runtime
- `rustfft`: FFT for spectral analysis
- `egui`/`wgpu`: Native visualization
- `leptos`: Web framework for WASM

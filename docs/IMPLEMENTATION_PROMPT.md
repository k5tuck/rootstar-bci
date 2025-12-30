# Neural Fingerprint System Implementation Prompt

Use this prompt in a new Claude Code chat to complete the implementation.

---

## Prompt

```
Implement the Neural Fingerprint Detection & Sensory Simulation System as specified in docs/NEURAL_FINGERPRINT_SYSTEM.md

The implementation should be done in two phases:

## Phase 1: Core Software Architecture (8-channel proof-of-concept)

Implement the following components that work with current hardware (8-ch EEG, basic fNIRS):

### 1. Rust Core Types (crates/rootstar-bci-core/src/fingerprint/)
- mod.rs - Module exports
- types.rs - NeuralFingerprint, SensoryModality, FrequencyBand structs
- config.rs - SystemConfig with scalable channel counts (8 → 128 → 256)
- safety.rs - SafetyLimits, SafetyMonitor for stimulation

### 2. Rust Native Processing (crates/rootstar-bci-native/src/fingerprint/)
- mod.rs - Module exports
- extractor.rs - EEGFeatureExtractor, FnirsProcessor
- fusion.rs - MultimodalFusion for EEG+fNIRS alignment
- stimulation.rs - StimulationProtocol, StimulationController

### 3. Python ML & Database (python/neural_fingerprint/)
- __init__.py
- models.py - NeuralFingerprintEncoder, ContrastiveFingerprintLoss, InverseStimulationModel (PyTorch)
- database.py - FingerprintRecord SQLAlchemy model, FingerprintIndex with FAISS
- controller.py - RealTimeFeedbackController, BidirectionalSession
- vr_bridge.py - VRSensoryBridge WebSocket server
- collection.py - DataCollectionManager, StimulusSession

### 4. Python Package Setup
- python/requirements.txt - torch, numpy, scipy, faiss-cpu, sqlalchemy, websockets
- python/setup.py - Package configuration

## Phase 2: Hardware Expansion Support

Extend the implementation to support high-density configurations:

### 1. Multi-ADS1299 Support (crates/rootstar-bci-embedded/src/drivers/)
- ads1299_array.rs - Driver for 16-32 daisy-chained ADS1299 chips (128-256 channels)
- Configurable SPI bus management
- Synchronized sampling across all chips

### 2. High-Density fNIRS (crates/rootstar-bci-embedded/src/drivers/)
- fnirs_array.rs - Driver for 32+ source/detector pairs
- Multiplexed LED control
- Multi-channel ADC support (multiple ADS1115 or dedicated fNIRS ADC)

### 3. Stimulation Matrix (crates/rootstar-bci-embedded/src/drivers/)
- stim_matrix.rs - 8x8 electrode switching matrix
- Multi-channel DAC support
- Hardware safety interlocks

### 4. Scalable Configuration System
- Runtime configuration for different hardware densities
- Automatic detection of connected hardware
- Graceful degradation if hardware unavailable

## Requirements

- Full implementation, no placeholders or TODOs
- All code must compile and pass cargo test / pytest
- Maintain compatibility with existing rootstar-bci architecture
- Use the tier system: core (no_std) → embedded → native → web
- Include proper error handling and safety checks
- Document public APIs

## Key Design Principles

1. **Scalability**: Config-driven channel counts (8 → 128 → 256)
2. **Safety First**: Hardware + software safety limits for stimulation
3. **Modularity**: Each component independently testable
4. **Real-time**: Async architecture for feedback loops
5. **Interoperability**: Clean Python bindings for ML stack

Commit each major component as you complete it. Push to the current branch when done.
```

---

## File Structure After Implementation

```
rootstar-bci/
├── docs/
│   ├── NEURAL_FINGERPRINT_SYSTEM.md    # Design document
│   └── IMPLEMENTATION_PROMPT.md        # This file
│
├── crates/
│   ├── rootstar-bci-core/src/
│   │   ├── fingerprint/
│   │   │   ├── mod.rs
│   │   │   ├── types.rs
│   │   │   ├── config.rs
│   │   │   └── safety.rs
│   │   └── lib.rs (updated)
│   │
│   ├── rootstar-bci-embedded/src/
│   │   ├── drivers/
│   │   │   ├── ads1299_array.rs    # Phase 2
│   │   │   ├── fnirs_array.rs      # Phase 2
│   │   │   └── stim_matrix.rs      # Phase 2
│   │   └── lib.rs (updated)
│   │
│   └── rootstar-bci-native/src/
│       ├── fingerprint/
│       │   ├── mod.rs
│       │   ├── extractor.rs
│       │   ├── fusion.rs
│       │   └── stimulation.rs
│       └── lib.rs (updated)
│
└── python/
    ├── neural_fingerprint/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── database.py
    │   ├── controller.py
    │   ├── vr_bridge.py
    │   └── collection.py
    ├── requirements.txt
    └── setup.py
```

---

## Verification Checklist

After implementation, verify:

- [ ] `cargo build` succeeds for all crates
- [ ] `cargo test` passes
- [ ] `pip install -e python/` succeeds
- [ ] `pytest python/` passes
- [ ] Safety limits are enforced in stimulation code
- [ ] Configuration supports 8, 128, and 256 channel modes
- [ ] Python can import and use all modules
- [ ] WebSocket VR bridge starts and accepts connections

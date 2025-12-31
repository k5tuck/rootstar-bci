# Neural Fingerprint System - Complete Architecture Analysis

## System Overview

After hardware completion, the Neural Fingerprint System enables **bidirectional sensory experiences**: capturing the neural signature of a real stimulus (e.g., tasting chocolate) and later reproducing that experience through transcranial stimulation.

---

## 1. Hardware Layer (Tier 1 - Embedded)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ESP32-WROOM-DA                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Ads1299Array    │  │   FnirsArray     │  │  StimMatrix   │  │
│  │  (8-256 ch EEG)  │  │  (4-128 ch)      │  │  (64 elec)    │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           │ SPI                 │ I2C/PWM            │ SPI      │
│  ┌────────▼─────────────────────▼──────────────────▼────────┐  │
│  │              Master Clock (2.048 MHz)                     │  │
│  │     fNIRS trigger = clock ÷ 81920 = 25 Hz                │  │
│  │     EEG sample rate = 500 Hz                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────┘
                                     │ USB
```

### Key Hardware Components

| Component | File | Purpose |
|-----------|------|---------|
| `Ads1299Array` | `ads1299_array.rs:165` | Daisy-chains 1-32 ADS1299 chips for 8-256 EEG channels at 500 Hz |
| `FnirsArray` | `fnirs_array.rs:318` | Multiplexes LED sources (760nm/850nm) and reads from multiple ADCs at 25 Hz |
| `StimMatrix` | `stim_matrix.rs:252` | Controls 8×8 electrode switching matrix for tDCS/tACS delivery |

### GPIO Pin Assignments

```
SPI (ADS1299):  MOSI=23, MISO=19, SCLK=18, CS=5, DRDY=4
I2C (fNIRS):    SDA=21, SCL=22
PWM (LEDs):     GPIO 25, 26, 27
DAC (Stim):     GPIO 32, 33
Matrix:         I2C via MCP23017 expanders
Status LED:     GPIO 17
```

---

## 2. Acquisition Flow

### Boot Sequence

```
1. HARDWARE BOOT
   └─> Ads1299Array::init() validates each chip (ID = 0x3E)
   └─> FnirsArray::init() configures LED drivers and ADCs
   └─> StimMatrix::init() resets matrix to safe state
```

### Synchronized Acquisition

```
┌──────────────────────────────────────────────────────────────┐
│ Master Clock (2.048 MHz) drives both subsystems:             │
│  • EEG: DRDY interrupt every 2ms (500 Hz)                    │
│  • fNIRS: Trigger every 40ms (25 Hz)                         │
│  • Timestamps: Microsecond resolution for alignment          │
└──────────────────────────────────────────────────────────────┘
```

### Data Packet Structures

```rust
// High-density EEG sample
HdEegSample {
    timestamp_us: u64,        // Absolute timestamp
    channels: Vec<Fixed24_8>, // 8-256 channels × 24-bit resolution
    sequence: u32             // For loss detection
}

// High-density fNIRS sample
HdFnirsSample {
    timestamp_us: u64,
    intensity_760: Vec<u16>,  // Per-channel at 760nm
    intensity_850: Vec<u16>,  // Per-channel at 850nm
}
```

---

## 3. Signal Processing Pipeline (Tier 2 - Native)

```
┌─────────────────────────────────────────────────────────────────┐
│                   HOST COMPUTER (Rust Native)                   │
│                                                                 │
│  Raw Data ─────────────────────────────────────────────────────►│
│  ┌─────────────────┐    ┌─────────────────┐   ┌──────────────┐ │
│  │EegFeatureExtractor│  │FnirsFeatureExtractor│ │TemporalAligner│ │
│  │ (extractor.rs:81) │  │ (extractor.rs:383) │ │ (fusion.rs:67)│ │
│  │                   │  │                     │ │               │ │
│  │ • Band power (FFT)│  │ • HbO/HbR via MBLL │ │ • Align EEG  │ │
│  │ • Coherence matrix│  │ • Activation maps   │ │   to fNIRS   │ │
│  │ • Phase sync      │  │ • NV coupling lag   │ │   (±5s)      │ │
│  │ • Sample entropy  │  │ • Regional peaks    │ │               │ │
│  └─────────┬─────────┘  └─────────┬───────────┘ └───────┬──────┘ │
│            └──────────────────┬────────────────────────────┘      │
│                               ▼                                   │
│                    ┌──────────────────────┐                       │
│                    │  FingerprintFusion   │                       │
│                    │  (fusion.rs:152)     │                       │
│                    │                      │                       │
│                    │  Combines EEG+fNIRS  │                       │
│                    │  into NeuralFingerprint                      │
│                    └──────────┬───────────┘                       │
│                               ▼                                   │
│                    ┌──────────────────────┐                       │
│                    │  NeuralFingerprint   │                       │
│                    │  (types.rs:472)      │                       │
│                    │                      │                       │
│                    │  • eeg_band_power[2560]                      │
│                    │  • eeg_topography[256]                       │
│                    │  • fnirs_hbo_activation[128]                 │
│                    │  • nv_coupling_lag_ms[128]                   │
│                    └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Extraction Details

#### EEG Features (`extractor.rs:81-379`)

| Feature | Method | Output |
|---------|--------|--------|
| Band Power | FFT with Hann window | Power in Delta, Theta, Alpha, Beta, Gamma + sensory-specific bands |
| Coherence | Cross-spectral density | 2D coherence matrix between channel pairs |
| Phase Sync | Phase Locking Value (PLV) | Synchronization between brain regions |
| Entropy | Sample entropy | Complexity measure per channel |

#### fNIRS Features (`extractor.rs:383-543`)

| Feature | Method | Output |
|---------|--------|--------|
| HbO/HbR | Modified Beer-Lambert Law | Hemoglobin concentration changes |
| Activation Maps | Spatial analysis of HbO | Regional activation patterns |
| Neurovascular Coupling | Cross-correlation EEG↔fNIRS | Lag typically 4-6 seconds |

---

## 4. Machine Learning Layer (Python)

### Neural Fingerprint Encoder

```
┌─────────────────────────────────────────────────────────────┐
│  NeuralFingerprintEncoder (models.py:22)                    │
│                                                              │
│  Input:                                                      │
│   • EEG spectral: (batch, channels, frequency_bins)         │
│   • EEG connectivity: (batch, pairs, 1)                     │
│   • fNIRS: (batch, channels, 2) [HbO, HbR]                 │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  SpectralBranch  │  │ConnectivityBranch│  │fNIRS Branch│ │
│  │  Conv1d layers   │  │  Linear layers   │  │  Linear    │ │
│  └────────┬─────────┘  └────────┬─────────┘  └─────┬──────┘ │
│           └─────────────────────┼──────────────────┘        │
│                                 ▼                            │
│                       ┌─────────────────┐                   │
│                       │    Attention    │                   │
│                       │   Fusion Head   │                   │
│                       └────────┬────────┘                   │
│                                ▼                            │
│                   256-dimensional embedding                 │
└─────────────────────────────────────────────────────────────┘
```

### Training Objective

```python
# ContrastiveFingerprintLoss (models.py:137)
# NT-Xent loss: Same stimulus → embeddings close
# Temperature τ = 0.07 for sharp similarity distribution

loss = -log(exp(sim(z_i, z_j) / τ) / Σ exp(sim(z_i, z_k) / τ))
```

### Storage and Retrieval

```
┌─────────────────────────────────────────────────────────────┐
│  FingerprintDatabase (database.py:168)                      │
│                                                              │
│  Storage: SQLite + FAISS                                    │
│   • Metadata in SQLite (stimulus label, subject, timestamp) │
│   • Embeddings in FAISS IndexFlatIP (cosine similarity)     │
│                                                              │
│  search(query_embedding, k=5) → nearest fingerprints        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Stimulation Feedback Loop

```
CLOSED-LOOP STIMULATION (stimulation.rs + controller.py)

Target Fingerprint                Current Brain State
      │                                    │
      ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    similarity = cosine(target.embedding, current.embedding) │
│                                                             │
│    if similarity < 0.9:                                     │
│        error = target - current                             │
│        delta_params = InverseStimulationModel(error)        │
│        adjust_stimulation(delta_params)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  StimMatrix Driver (stim_matrix.rs)                         │
│                                                             │
│  • Select electrodes via switching matrix                   │
│  • Set current (0-2000 µA) with 30s ramp                   │
│  • Apply frequency (0-100 Hz for tACS)                     │
│  • Monitor via SafetyMonitor (safety.rs:103)               │
└─────────────────────────────────────────────────────────────┘
```

### Safety Limits (RESEARCH Preset)

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| `max_current_ua` | 2000 µA | Standard tDCS safety limit |
| `max_duration_min` | 30 min | Prevent tissue heating |
| `min_ramp_s` | 30 s | Avoid phosphenes |
| `max_frequency_hz` | 100 Hz | Within tACS research range |
| `max_charge_density_uc_cm2` | 40 µC/cm² | Prevent lesions |

---

## 6. VR Integration

```
┌─────────────────────────────────────────────────────────────┐
│  VRSensoryBridge (vr_bridge.py:54)                          │
│                                                             │
│  WebSocket Server (ws://localhost:8765)                     │
│                                                             │
│  Messages:                                                  │
│   VR → BCI: {"type": "trigger_sensory",                    │
│              "modality": "gustatory",                       │
│              "params": {"stimulus": "chocolate"}}           │
│                                                             │
│   BCI → VR: {"type": "fingerprint_match",                  │
│              "similarity": 0.92,                            │
│              "stimulus_label": "sweet"}                     │
│                                                             │
│  Flow:                                                      │
│  1. VR environment triggers "taste chocolate"              │
│  2. System finds matching fingerprint in database          │
│  3. Starts stimulation session with that target            │
│  4. Reports similarity progress back to VR                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. End-to-End Workflows

### Recording a New Sensory Experience

```
1. Subject tastes chocolate
2. DataCollectionManager runs collection session:
   - 10s baseline recording
   - 5s stimulus presentation
   - Neural data captured (EEG + fNIRS)
   - 30s washout period
   - Repeat 10× for averaging
3. EegFeatureExtractor + FnirsFeatureExtractor compute features
4. FingerprintFusion creates NeuralFingerprint
5. NeuralFingerprintEncoder produces 256-dim embedding
6. FingerprintDatabase stores with label "chocolate"
```

### Reproducing the Experience

```
1. Query database for "chocolate" fingerprint
2. Create StimulationSession with target fingerprint
3. Configure StimMatrix electrodes (e.g., F3 anode, Fp2 cathode)
4. Start stimulation with 30s ramp-up
5. Every 100ms:
   - Acquire current neural state
   - Compute similarity to target
   - Adjust stimulation parameters
   - Check safety limits
6. Continue until similarity > 0.9 or max iterations
7. Ramp down over 30s
```

---

## 8. Hardware Scaling

| Configuration | EEG Channels | ADS1299 Chips | fNIRS Channels | Data Rate |
|--------------|-------------|---------------|----------------|-----------|
| Basic8       | 8           | 1             | 4              | ~100 kbps |
| Standard32   | 32          | 4             | 16             | ~400 kbps |
| Medium64     | 64          | 8             | 32             | ~800 kbps |
| HighDensity128| 128        | 16            | 64             | ~1.6 Mbps |
| UltraHD256   | 256         | 32            | 128            | ~3.2 Mbps |

The system scales from proof-of-concept (8 channels) to research-grade (256 channels) by simply adding more ADS1299 chips in daisy chain configuration.

---

## 9. File Reference

### Core Types (Tier 0)
- `crates/rootstar-bci-core/src/fingerprint/types.rs` - NeuralFingerprint, SensoryModality, FrequencyBand
- `crates/rootstar-bci-core/src/fingerprint/config.rs` - ChannelDensity, SystemConfig
- `crates/rootstar-bci-core/src/fingerprint/safety.rs` - SafetyMonitor, SafetyLimits

### Embedded Drivers (Tier 1)
- `crates/rootstar-bci-embedded/src/drivers/ads1299_array.rs` - High-density EEG
- `crates/rootstar-bci-embedded/src/drivers/fnirs_array.rs` - High-density fNIRS
- `crates/rootstar-bci-embedded/src/drivers/stim_matrix.rs` - Electrode switching

### Native Processing (Tier 2)
- `crates/rootstar-bci-native/src/fingerprint/extractor.rs` - Feature extraction
- `crates/rootstar-bci-native/src/fingerprint/fusion.rs` - Multimodal fusion
- `crates/rootstar-bci-native/src/fingerprint/stimulation.rs` - Closed-loop control

### Python ML Stack
- `python/neural_fingerprint/models.py` - PyTorch encoder and loss
- `python/neural_fingerprint/database.py` - SQLite + FAISS storage
- `python/neural_fingerprint/controller.py` - Real-time feedback
- `python/neural_fingerprint/vr_bridge.py` - WebSocket VR integration
- `python/neural_fingerprint/collection.py` - Data collection protocols

---

## 10. Future Extensions

1. **Multi-subject transfer learning** - Train encoder on population, fine-tune per subject
2. **Temporal fingerprints** - Capture time-varying patterns, not just static signatures
3. **Olfactory hardware** - Integrate scent delivery for smell reproduction
4. **Haptic feedback** - Add vibrotactile arrays for touch simulation
5. **Real-time artifact rejection** - ML-based artifact detection during acquisition

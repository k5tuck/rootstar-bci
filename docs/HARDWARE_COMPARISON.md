# Hardware Comparison: OpenBCI Galea vs Rootstar-BCI

## Executive Summary

This document compares OpenBCI's Galea multi-modal headset with the proposed Rootstar-BCI Neural Fingerprint System. While Galea is a commercial product focused on VR integration, Rootstar-BCI is designed as a research platform for sensory experience capture and reproduction.

---

## 1. Modality Comparison

| Feature | OpenBCI Galea | Rootstar-BCI (Proposed) |
|---------|---------------|-------------------------|
| **EEG Channels** | 16 dry electrodes | 8-256 channels (ADS1299 array) |
| **fNIRS** | Not included | 4-128 channels (dual wavelength) |
| **EMG** | 4 facial channels | 8 facial channels |
| **EDA (GSR)** | 2 channels | 4 channels |
| **PPG/Heart Rate** | Yes (2 sensors) | Not currently included |
| **Eye Tracking** | 7 channels (EOG) | Not included |
| **Accelerometer** | 3-axis IMU | Not included |

### Key Differences

**Galea Strengths:**
- Integrated VR headset compatibility (Quest Pro, Varjo)
- Dry electrode system (no gel required)
- PPG for heart rate variability
- Eye tracking integration
- Commercial polish and support

**Rootstar-BCI Strengths:**
- High-density EEG (up to 256 channels vs 16)
- fNIRS for hemodynamic imaging (missing in Galea)
- Transcranial stimulation capability (tDCS/tACS)
- Neural fingerprint storage and playback
- Open research platform

---

## 2. EEG Specifications

### OpenBCI Galea EEG
```
Chip:           ADS1299 (single)
Channels:       16 differential
Resolution:     24-bit
Sample Rate:    250 Hz
Electrode Type: Dry (AgCl coated)
Coverage:       Frontal, temporal, parietal
Noise:          <1 µVrms
```

### Rootstar-BCI EEG
```
Chip:           ADS1299 (1-32 daisy-chained)
Channels:       8, 32, 64, 128, or 256
Resolution:     24-bit
Sample Rate:    250, 500, 1000, 2000 Hz
Electrode Type: Wet gel or dry (configurable)
Coverage:       Full 10-20 system (high-density)
Noise:          <1 µVrms per chip
Data Rate:      100 kbps - 3.2 Mbps
```

**Analysis:** Rootstar-BCI offers significantly higher channel density (up to 16x more channels), enabling source localization and high-resolution brain mapping. Galea prioritizes convenience with dry electrodes.

---

## 3. fNIRS Comparison

### OpenBCI Galea
- **fNIRS:** Not included
- No hemodynamic imaging capability

### Rootstar-BCI
```
Wavelengths:    760nm (HbR) + 850nm (HbO2)
Channels:       4-128 optode pairs
Sample Rate:    25 Hz
Penetration:    ~15-20mm cortical depth
ADC:            ADS1115 (16-bit)
Processing:     Modified Beer-Lambert Law
Output:         ΔHbO, ΔHbR, oxygenation index
```

**Analysis:** fNIRS is critical for measuring hemodynamic responses that correlate with neural activity on a slower timescale (seconds). This enables neurovascular coupling analysis for the Neural Fingerprint System.

---

## 4. EMG Specifications

### OpenBCI Galea EMG
```
Channels:       4
Target Muscles: Frontalis, temporalis, masseter, zygomatic
Resolution:     24-bit (shared with EEG ADC)
Sample Rate:    250 Hz
Purpose:        Facial expression detection
```

### Rootstar-BCI EMG
```
Channels:       8
Target Muscles:
  - Zygomaticus major (L/R) - smile
  - Corrugator supercilii (L/R) - frown
  - Masseter (L/R) - jaw/chewing
  - Orbicularis oris (U/D) - lips
Resolution:     24-bit (ADS1299)
Sample Rate:    500-2000 Hz
Purpose:        Emotional valence + gustatory activity
Features:       RMS, mean frequency, valence score
```

**Analysis:** Rootstar-BCI provides more channels with higher temporal resolution, specifically targeting muscles relevant to taste and emotional responses.

---

## 5. EDA (Electrodermal Activity) Specifications

### OpenBCI Galea EDA
```
Channels:       2
Sites:          Fingers (through hand controllers)
Resolution:     16-bit
Sample Rate:    Unknown (likely 10-50 Hz)
Output:         Raw conductance
```

### Rootstar-BCI EDA
```
Channels:       4
Sites:          Palm (L/R), Thenar (L/R)
Resolution:     16-bit (ADS1115)
Sample Rate:    8-128 Hz
Output:
  - Raw conductance (µS)
  - SCL (tonic level)
  - SCR (phasic responses)
  - Arousal score
Processing:     Tonic/phasic decomposition
```

**Analysis:** Rootstar-BCI offers more measurement sites and on-device processing for tonic/phasic decomposition, providing better arousal measurement.

---

## 6. Unique Rootstar-BCI Capabilities

### Neural Stimulation (Not Available in Galea)
```
Technology:     tDCS/tACS via electrode matrix
Electrodes:     8×8 switching matrix (64 sites)
Current Range:  0-2000 µA
Frequency:      0-100 Hz (tACS)
Safety:
  - Hardware current limiting
  - 30s ramp up/down
  - 40 µC/cm² charge density limit
  - Real-time impedance monitoring
```

### Neural Fingerprint System (Not Available in Galea)
```
Purpose:        Capture and reproduce sensory experiences
Capture:
  - EEG band power analysis
  - fNIRS hemodynamic patterns
  - EMG valence/arousal
  - EDA autonomic response

Storage:
  - 256-dimensional embeddings
  - FAISS similarity search
  - SQLite metadata

Reproduction:
  - Closed-loop stimulation
  - Real-time feedback
  - Similarity convergence
```

---

## 7. Hardware Architecture

### OpenBCI Galea
```
┌─────────────────────────────────────────┐
│            VR Headset (Host)            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │   EEG   │  │   EMG   │  │   EDA   │ │
│  │ 16 ch   │  │  4 ch   │  │  2 ch   │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       └───────────┬┴───────────┘       │
│              ADS1299 + MCU             │
│                   │                     │
│              Bluetooth/USB              │
└─────────────────────────────────────────┘
```

### Rootstar-BCI
```
┌─────────────────────────────────────────────────────────────┐
│                     ESP32-WROOM-DA                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ ADS1299    │ │   fNIRS    │ │   EMG      │ │   EDA    │ │
│  │ Array      │ │   Array    │ │ (ADS1299)  │ │(ADS1115) │ │
│  │ 8-256 ch   │ │ 4-128 ch   │ │   8 ch     │ │  4 ch    │ │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └────┬─────┘ │
│        │SPI           │I2C           │SPI          │I2C    │
│  ┌─────┴──────────────┴──────────────┴─────────────┴─────┐ │
│  │                  Master Clock (2.048 MHz)              │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                 │
│  ┌────────────────────────┴───────────────────────────────┐ │
│  │         Stimulation Matrix (8×8, 64 electrodes)        │ │
│  │                DAC8564 + Current Sources               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │ USB
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Host Computer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Rust Native  │  │ Python ML    │  │ Web Visualization │ │
│  │ Processing   │  │ Fingerprint  │  │ (WASM + WebGL)    │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Use Case Comparison

| Use Case | Galea | Rootstar-BCI |
|----------|-------|--------------|
| VR Gaming | Excellent | Limited |
| Meditation/Wellness | Good | Excellent |
| Neuroscience Research | Limited | Excellent |
| Source Localization | No (16 ch) | Yes (64+ ch) |
| Hemodynamic Imaging | No | Yes (fNIRS) |
| Sensory Experience Capture | No | Yes |
| Experience Reproduction | No | Yes (stimulation) |
| BCI Development | Limited | Excellent |
| Consumer Accessibility | High | Low (research) |

---

## 9. Price Point Comparison

| System | Estimated Price | Notes |
|--------|-----------------|-------|
| OpenBCI Galea | ~$5,000-8,000 | Commercial product |
| Rootstar-BCI (8-ch) | ~$500-800 | DIY/research |
| Rootstar-BCI (64-ch) | ~$2,000-3,000 | Research grade |
| Rootstar-BCI (256-ch) | ~$8,000-12,000 | High-density |

---

## 10. Recommendations

### Choose OpenBCI Galea if:
- VR integration is primary requirement
- Dry electrodes preferred (convenience)
- Consumer/wellness application
- Commercial support required
- Eye tracking needed

### Choose Rootstar-BCI if:
- High-density EEG required (>16 channels)
- fNIRS hemodynamic imaging needed
- Neural stimulation required
- Neural fingerprint capture/playback
- Open-source/customizable platform
- Research/academic use

---

## 11. Future Roadmap

### Potential Rootstar-BCI Additions
1. **PPG Integration** - Add pulse oximetry via MAX30102
2. **Eye Tracking** - EOG channels or camera-based
3. **Olfactory Hardware** - Scent delivery for smell simulation
4. **Haptic Feedback** - Vibrotactile arrays for touch
5. **VR Integration** - Direct headset connection

### Integration Opportunity
Both systems could potentially be combined:
- Galea for dry EEG + eye tracking + VR
- Rootstar-BCI for fNIRS + stimulation + fingerprinting

---

## Appendix: Research Institutions Doing Similar Work

| Institution | Focus Area |
|-------------|------------|
| Kernel (Flow) | High-density fNIRS headset |
| Neuralink | Invasive neural interfaces |
| Synchron | Endovascular BCI |
| OpenBCI | Consumer EEG/EMG |
| Precision Neuroscience | Thin-film electrodes |
| University of Geneva | Orbitofrontal stimulation for smell |
| Meta Reality Labs | EMG wristband |

**Key Research:** University of Geneva demonstrated that stimulating the orbitofrontal cortex can induce specific olfactory percepts (lemon, coffee smells) - directly relevant to Neural Fingerprint System goals.

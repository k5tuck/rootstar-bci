# Neural Fingerprint Detection & Sensory Simulation System

## Pattern-Based Approach for Non-Invasive Brain-Computer Interface

### Document Version: 1.0

### Target Platform: Cerelog ESP-EEG / OpenBCI Integration

-----

## 1. System Overview

This document describes a pattern-based neural fingerprint detection system for capturing and reproducing sensory experiences (taste, smell, touch, sound) through non-invasive brain-computer interface technology. Rather than mapping individual neurons, this approach captures regional neural activity signatures ("fingerprints") that correspond to specific sensory experiences.

### 1.1 Core Principle

The brain generates reproducible patterns of electrical and hemodynamic activity when processing sensory information. By capturing these patterns during actual sensory experiences and replaying them through targeted stimulation, we can theoretically recreate the subjective experience without the original sensory input.

### 1.2 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL FINGERPRINT SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   CAPTURE    │───▶│   PROCESS    │───▶│    STORE     │       │
│  │   MODULE     │    │   MODULE     │    │   MODULE     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  HD-EEG +    │    │  ML Pattern  │    │  Fingerprint │       │
│  │  fNIRS Array │    │  Extraction  │    │  Database    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   RETRIEVE   │◀───│   DECODE     │◀───│  STIMULATE   │       │
│  │   MODULE     │    │   MODULE     │    │   MODULE     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Pattern     │    │  Inverse ML  │    │  tDCS/tACS   │       │
│  │  Matching    │    │  Transform   │    │  Delivery    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

-----

## 2. Hardware Configuration

### 2.1 High-Density EEG Array Specifications

For capturing neural fingerprints with sufficient spatial resolution, you need a minimum of **128 electrodes**, with **256 electrodes** recommended for gustatory and olfactory cortex targeting.

#### 2.1.1 Electrode Density Requirements

|Sensory Target       |Minimum Electrodes|Recommended|Inter-Electrode Distance|
|---------------------|------------------|-----------|------------------------|
|Gustatory (Taste)    |64                |128        |10-15mm                 |
|Olfactory (Smell)    |64                |128        |10-15mm                 |
|Somatosensory (Touch)|128               |256        |8-12mm                  |
|Auditory (Sound)     |64                |128        |10-15mm                 |
|Visual (Sight)       |128               |256        |8-12mm                  |

#### 2.1.2 Electrode Placement Map for Gustatory/Olfactory Focus

```
                    FRONTAL VIEW - ELECTRODE PLACEMENT

                         Nasion (Reference Point)
                              ▼
                    ┌─────────●─────────┐
                   /    Fp1 ● ● Fp2     \
                  /   AF7●  ●AFz●  ●AF8   \
                 /  F7● F3● Fz ●F4 ●F8    \
                │  FT7● FC3●FCz●FC4●FT8    │
                │   T7● C3 ●Cz ●C4 ●T8     │
                │  TP7● CP3●CPz●CP4●TP8    │
                 \  P7● P3● Pz ●P4 ●P8    /
                  \   PO7● POz ●PO8      /
                   \    O1 ● ● O2       /
                    └─────────●─────────┘
                            Inion

    GUSTATORY CORTEX FOCUS ZONE (Frontal-Insular Region):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Primary electrodes: F7, F8, FT7, FT8, T7, T8
    Secondary electrodes: F3, F4, FC3, FC4, C3, C4

    OLFACTORY CORTEX FOCUS ZONE (Orbitofrontal Region):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Primary electrodes: Fp1, Fp2, AF7, AF8, F7, F8
    Secondary electrodes: AFz, Fz, F3, F4
```

#### 2.1.3 High-Density Cluster Configuration

For enhanced resolution over target regions, create **dense electrode clusters**:

```
    GUSTATORY CLUSTER (Left Hemisphere)        GUSTATORY CLUSTER (Right Hemisphere)

         ●───●───●                                    ●───●───●
        /│\  │  /│\                                  /│\  │  /│\
       ● │ ● ● ● │ ●                                ● │ ● ● ● │ ●
        \│/  │  \│/                                  \│/  │  \│/
         ●───●───●                                    ●───●───●
        /│\  │  /│\                                  /│\  │  /│\
       ● │ ● ● ● │ ●                                ● │ ● ● ● │ ●
        \│/  │  \│/                                  \│/  │  \│/
         ●───●───●                                    ●───●───●

    Cluster Center: F7/FT7 region                Cluster Center: F8/FT8 region
    Electrode Spacing: 8mm                       Electrode Spacing: 8mm
    Total per cluster: 25 electrodes             Total per cluster: 25 electrodes
```

### 2.2 fNIRS Array Configuration

#### 2.2.1 Optode Specifications

|Parameter               |Specification                          |
|------------------------|---------------------------------------|
|Light Sources (LEDs)    |760nm and 850nm dual-wavelength        |
|Detectors               |Silicon photodiodes with 10^12 V/A gain|
|Source-Detector Distance|30mm (standard), 15mm (short-channel)  |
|Sampling Rate           |Minimum 10 Hz, recommended 25 Hz       |
|Optode Density          |32 sources, 32 detectors minimum       |

#### 2.2.2 fNIRS Optode Placement for Deep Cortical Access

```
    fNIRS MONTAGE - GUSTATORY/OLFACTORY TARGETING

    ◆ = Light Source (LED)
    ○ = Detector (Photodiode)
    ━ = Measurement Channel (30mm)
    ─ = Short-Channel (15mm)

                      FRONTAL REGION

              ◆━━━○───◆━━━○───◆━━━○
              │   │   │   │   │   │
              ○───◆━━━○───◆━━━○───◆
              │   │   │   │   │   │
              ◆━━━○───◆━━━○───◆━━━○
              │   │   │   │   │   │
              ○───◆━━━○───◆━━━○───◆

    LEFT TEMPORAL                      RIGHT TEMPORAL
    (Gustatory L)                      (Gustatory R)

    ◆━━━○───◆                          ◆───○━━━◆
    │   │   │                          │   │   │
    ○───◆━━━○                          ○━━━◆───○
    │   │   │                          │   │   │
    ◆━━━○───◆                          ◆───○━━━◆
```

#### 2.2.3 Diffuse Optical Tomography (DOT) Enhancement

For 3D reconstruction of neural activity patterns, implement overlapping measurement channels:

```python
# DOT Channel Configuration
DOT_CONFIG = {
    "source_positions": 48,  # Total LED sources
    "detector_positions": 48,  # Total photodiodes
    "short_channels": 16,  # For superficial signal regression
    "long_channels": 128,  # Primary measurement channels
    "overlap_factor": 3,  # Each brain voxel sampled by 3+ channels
    "spatial_resolution": "~10mm isotropic",
    "depth_penetration": "20-30mm from scalp"
}
```

### 2.3 Combined EEG-fNIRS Integration with Cerelog

#### 2.3.1 Hardware Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    CERELOG INTEGRATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐      ┌─────────────────┐               │
│  │   ESP32-S3      │      │   ADS1299       │               │
│  │   Main MCU      │◀────▶│   8-Ch EEG AFE  │ x 16 chips    │
│  │                 │      │   (128 channels)│               │
│  └────────┬────────┘      └─────────────────┘               │
│           │                                                  │
│           │              ┌─────────────────┐                │
│           │              │   fNIRS Module  │                │
│           └─────────────▶│   LED Driver +  │                │
│                          │   TIA Array     │                │
│                          └─────────────────┘                │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   DATA PIPELINE                          ││
│  │                                                          ││
│  │  EEG: 128ch × 24-bit × 500 Hz = 1.536 Mbps              ││
│  │  fNIRS: 128ch × 16-bit × 25 Hz = 51.2 kbps              ││
│  │  Total: ~1.6 Mbps → USB 2.0 or WiFi streaming           ││
│  │                                                          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 Synchronization Protocol

```rust
/// Time synchronization between EEG and fNIRS subsystems
pub struct SyncProtocol {
    /// Master clock source (EEG ADC clock)
    master_clock_hz: u32,

    /// fNIRS sampling trigger derived from master
    fnirs_trigger_divisor: u32,

    /// Timestamp resolution in microseconds
    timestamp_resolution_us: u32,

    /// Maximum allowed clock drift before resync
    max_drift_us: u32,
}

impl Default for SyncProtocol {
    fn default() -> Self {
        Self {
            master_clock_hz: 2_048_000,  // 2.048 MHz master clock
            fnirs_trigger_divisor: 81920, // Results in 25 Hz fNIRS
            timestamp_resolution_us: 1,
            max_drift_us: 100,
        }
    }
}
```

-----

## 3. Signal Acquisition Pipeline

### 3.1 EEG Signal Conditioning

```
┌─────────────────────────────────────────────────────────────────┐
│                    EEG SIGNAL CHAIN                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Scalp ──▶ Electrode ──▶ Preamp ──▶ Bandpass ──▶ ADC ──▶ DSP   │
│            (Ag/AgCl)     (×1000)    (0.1-100Hz)  (24-bit)       │
│                                                                  │
│  Noise Floor Target: < 1 µVrms                                  │
│  Input Impedance: > 1 GΩ                                        │
│  CMRR: > 110 dB                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.1 Frequency Band Extraction

```python
# Neural frequency bands for pattern extraction
FREQUENCY_BANDS = {
    "delta": (0.5, 4),      # Deep sleep, unconscious processing
    "theta": (4, 8),        # Memory, emotional processing
    "alpha": (8, 13),       # Relaxed awareness, inhibition
    "beta": (13, 30),       # Active thinking, motor planning
    "gamma": (30, 100),     # Sensory binding, perception
    "high_gamma": (70, 150) # Fine sensory discrimination
}

# Critical for gustatory/olfactory processing
SENSORY_FOCUS_BANDS = {
    "gustatory_theta": (4, 7),      # Taste memory retrieval
    "gustatory_gamma": (35, 45),    # Taste discrimination
    "olfactory_theta": (4, 8),      # Odor identification
    "olfactory_gamma": (40, 80),    # Odor-object binding
}
```

### 3.2 fNIRS Signal Processing

#### 3.2.1 Hemodynamic Response Extraction

```python
import numpy as np
from scipy.signal import butter, filtfilt

class fNIRSProcessor:
    """
    Processes raw fNIRS intensity signals to extract
    oxygenated (HbO) and deoxygenated (HbR) hemoglobin
    concentration changes.
    """

    def __init__(self, sampling_rate: float = 25.0):
        self.fs = sampling_rate

        # Extinction coefficients (cm^-1 / molar)
        self.extinction = {
            760: {"HbO": 1486.5865, "HbR": 3843.707},
            850: {"HbO": 2526.391, "HbR": 1798.643}
        }

        # Differential pathlength factor
        self.dpf = {760: 6.26, 850: 5.86}

    def intensity_to_od(self, intensity: np.ndarray,
                        baseline: np.ndarray) -> np.ndarray:
        """Convert intensity to optical density change."""
        return -np.log10(intensity / baseline)

    def od_to_concentration(self, od_760: np.ndarray,
                            od_850: np.ndarray,
                            source_detector_dist: float) -> dict:
        """
        Apply Modified Beer-Lambert Law to compute
        hemoglobin concentration changes.
        """
        # Build extinction coefficient matrix
        E = np.array([
            [self.extinction[760]["HbO"], self.extinction[760]["HbR"]],
            [self.extinction[850]["HbO"], self.extinction[850]["HbR"]]
        ])

        # Apply DPF correction
        L = np.array([
            [self.dpf[760] * source_detector_dist],
            [self.dpf[850] * source_detector_dist]
        ])

        # Solve for concentration changes
        E_inv = np.linalg.pinv(E * L)

        od_stack = np.vstack([od_760, od_850])
        concentrations = E_inv @ od_stack

        return {
            "HbO": concentrations[0],  # Oxygenated hemoglobin
            "HbR": concentrations[1],  # Deoxygenated hemoglobin
            "HbT": concentrations[0] + concentrations[1]  # Total
        }

    def bandpass_filter(self, signal: np.ndarray,
                        low: float = 0.01,
                        high: float = 0.5) -> np.ndarray:
        """
        Bandpass filter to isolate hemodynamic response
        and remove physiological noise.
        """
        nyq = self.fs / 2
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, signal, axis=-1)
```

### 3.3 Multimodal Fusion

#### 3.3.1 EEG-fNIRS Temporal Alignment

```python
class MultimodalFusion:
    """
    Fuses EEG and fNIRS signals for comprehensive
    neural pattern extraction.
    """

    def __init__(self, eeg_fs: float = 500.0, fnirs_fs: float = 25.0):
        self.eeg_fs = eeg_fs
        self.fnirs_fs = fnirs_fs
        self.upsample_factor = int(eeg_fs / fnirs_fs)

    def align_signals(self, eeg_data: np.ndarray,
                      fnirs_data: np.ndarray,
                      timestamps_eeg: np.ndarray,
                      timestamps_fnirs: np.ndarray) -> dict:
        """
        Temporally align EEG and fNIRS based on timestamps.
        Upsample fNIRS to match EEG sampling rate.
        """
        from scipy.interpolate import interp1d

        # Interpolate fNIRS to EEG timebase
        fnirs_interp = interp1d(
            timestamps_fnirs, fnirs_data,
            axis=-1, kind='cubic', fill_value='extrapolate'
        )
        fnirs_aligned = fnirs_interp(timestamps_eeg)

        return {
            "eeg": eeg_data,
            "fnirs": fnirs_aligned,
            "timestamps": timestamps_eeg
        }

    def compute_neurovascular_coupling(self,
                                       eeg_power: np.ndarray,
                                       hbo: np.ndarray,
                                       lag_range_s: tuple = (-5, 15)) -> np.ndarray:
        """
        Compute cross-correlation between EEG band power
        and HbO to characterize neurovascular coupling.

        The hemodynamic response typically lags neural
        activity by 4-6 seconds.
        """
        from scipy.signal import correlate

        lag_samples = (
            int(lag_range_s[0] * self.eeg_fs),
            int(lag_range_s[1] * self.eeg_fs)
        )

        correlation = correlate(hbo, eeg_power, mode='full')
        lags = np.arange(-len(eeg_power)+1, len(hbo))

        # Extract relevant lag range
        mask = (lags >= lag_samples[0]) & (lags <= lag_samples[1])

        return {
            "correlation": correlation[mask],
            "lags_seconds": lags[mask] / self.eeg_fs,
            "peak_lag": lags[mask][np.argmax(np.abs(correlation[mask]))] / self.eeg_fs
        }
```

-----

## 4. Neural Fingerprint Extraction

### 4.1 Feature Engineering

#### 4.1.1 EEG Feature Matrix

```python
class EEGFeatureExtractor:
    """
    Extracts comprehensive feature set from multi-channel EEG
    for neural fingerprint generation.
    """

    def __init__(self, n_channels: int = 128, fs: float = 500.0):
        self.n_channels = n_channels
        self.fs = fs

    def extract_features(self, epoch: np.ndarray) -> dict:
        """
        Extract features from a single EEG epoch.

        Args:
            epoch: Shape (n_channels, n_samples)

        Returns:
            Feature dictionary for fingerprint encoding
        """
        features = {}

        # 1. Spectral Power Features
        features["band_power"] = self._compute_band_power(epoch)

        # 2. Connectivity Features
        features["coherence"] = self._compute_coherence(epoch)
        features["phase_sync"] = self._compute_phase_sync(epoch)

        # 3. Spatial Features
        features["topography"] = self._compute_topography(epoch)

        # 4. Temporal Features
        features["erp_components"] = self._compute_erp(epoch)

        # 5. Complexity Features
        features["entropy"] = self._compute_entropy(epoch)

        return features

    def _compute_band_power(self, epoch: np.ndarray) -> np.ndarray:
        """Compute power spectral density in canonical bands."""
        from scipy.signal import welch

        band_powers = np.zeros((self.n_channels, len(FREQUENCY_BANDS)))

        for ch in range(self.n_channels):
            freqs, psd = welch(epoch[ch], fs=self.fs, nperseg=256)

            for i, (band_name, (low, high)) in enumerate(FREQUENCY_BANDS.items()):
                mask = (freqs >= low) & (freqs <= high)
                band_powers[ch, i] = np.trapz(psd[mask], freqs[mask])

        return band_powers

    def _compute_coherence(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute inter-channel coherence matrix.
        Critical for understanding functional connectivity
        in sensory processing networks.
        """
        from scipy.signal import coherence

        n_bands = len(SENSORY_FOCUS_BANDS)
        coh_matrix = np.zeros((n_bands, self.n_channels, self.n_channels))

        for band_idx, (band_name, (low, high)) in enumerate(SENSORY_FOCUS_BANDS.items()):
            for i in range(self.n_channels):
                for j in range(i+1, self.n_channels):
                    freqs, coh = coherence(epoch[i], epoch[j],
                                          fs=self.fs, nperseg=128)
                    mask = (freqs >= low) & (freqs <= high)
                    coh_matrix[band_idx, i, j] = np.mean(coh[mask])
                    coh_matrix[band_idx, j, i] = coh_matrix[band_idx, i, j]

        return coh_matrix

    def _compute_phase_sync(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute phase locking value (PLV) between channel pairs.
        Important for tracking synchronized neural assemblies
        during sensory perception.
        """
        from scipy.signal import hilbert

        # Filter to gamma band for sensory binding
        analytic = hilbert(epoch, axis=-1)
        phase = np.angle(analytic)

        plv_matrix = np.zeros((self.n_channels, self.n_channels))

        for i in range(self.n_channels):
            for j in range(i+1, self.n_channels):
                phase_diff = phase[i] - phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv

        return plv_matrix

    def _compute_topography(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute spatial distribution of activity.
        Returns normalized activation pattern across scalp.
        """
        rms = np.sqrt(np.mean(epoch**2, axis=-1))
        return rms / np.max(rms)

    def _compute_erp(self, epoch: np.ndarray) -> dict:
        """
        Extract event-related potential components.
        For gustatory: Focus on 150-250ms (taste recognition)
        For olfactory: Focus on 200-400ms (odor identification)
        """
        # Define component windows (in samples at 500 Hz)
        components = {
            "N1": (50, 100),      # 100-200ms
            "P2": (100, 175),     # 200-350ms
            "N2": (150, 225),     # 300-450ms
            "P3": (200, 350),     # 400-700ms
        }

        erp_features = {}
        for comp_name, (start, end) in components.items():
            erp_features[comp_name] = {
                "amplitude": np.mean(epoch[:, start:end], axis=-1),
                "latency": start + np.argmax(np.abs(epoch[:, start:end]), axis=-1)
            }

        return erp_features

    def _compute_entropy(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute sample entropy for each channel.
        Higher entropy = more complex, less predictable activity.
        """
        from scipy.stats import entropy as scipy_entropy

        entropies = np.zeros(self.n_channels)

        for ch in range(self.n_channels):
            # Compute histogram-based entropy
            hist, _ = np.histogram(epoch[ch], bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropies[ch] = scipy_entropy(hist)

        return entropies
```

### 4.2 Neural Fingerprint Encoding

#### 4.2.1 Fingerprint Data Structure

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class NeuralFingerprint:
    """
    Complete neural signature for a specific sensory experience.
    """
    # Metadata
    fingerprint_id: str
    sensory_modality: str  # "gustatory", "olfactory", "tactile", "auditory"
    stimulus_label: str    # e.g., "apple_taste", "rose_smell"
    timestamp: datetime
    subject_id: str

    # EEG Features
    eeg_band_power: np.ndarray          # Shape: (n_channels, n_bands)
    eeg_coherence: np.ndarray           # Shape: (n_bands, n_channels, n_channels)
    eeg_phase_sync: np.ndarray          # Shape: (n_channels, n_channels)
    eeg_topography: np.ndarray          # Shape: (n_channels,)
    eeg_erp_components: Dict            # ERP amplitude and latency
    eeg_entropy: np.ndarray             # Shape: (n_channels,)

    # fNIRS Features
    fnirs_hbo_pattern: np.ndarray       # Shape: (n_optodes, n_timepoints)
    fnirs_hbr_pattern: np.ndarray       # Shape: (n_optodes, n_timepoints)
    fnirs_spatial_activation: np.ndarray # Shape: (n_optodes,)

    # Derived Features
    neurovascular_coupling: Dict        # Cross-modal correlation
    temporal_dynamics: np.ndarray       # Time-frequency representation

    # Quality Metrics
    signal_quality: float               # 0-1 quality score
    confidence: float                   # 0-1 confidence in fingerprint

    # Raw Data Reference (for reprocessing)
    raw_data_path: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """
        Flatten fingerprint to single feature vector for ML.
        """
        components = [
            self.eeg_band_power.flatten(),
            self.eeg_coherence.flatten(),
            self.eeg_phase_sync.flatten(),
            self.eeg_topography,
            self.eeg_entropy,
            self.fnirs_spatial_activation,
        ]
        return np.concatenate(components)

    def similarity(self, other: 'NeuralFingerprint') -> float:
        """
        Compute cosine similarity between fingerprints.
        """
        v1 = self.to_vector()
        v2 = other.to_vector()

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### 4.3 Machine Learning Pattern Extraction

#### 4.3.1 Fingerprint Classification Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralFingerprintEncoder(nn.Module):
    """
    Deep neural network for encoding neural patterns into
    compact fingerprint embeddings.

    Architecture: Multi-branch network processing EEG and fNIRS
    separately before fusion.
    """

    def __init__(self,
                 n_eeg_channels: int = 128,
                 n_fnirs_channels: int = 64,
                 n_bands: int = 6,
                 embedding_dim: int = 256):
        super().__init__()

        self.n_eeg_channels = n_eeg_channels
        self.n_fnirs_channels = n_fnirs_channels

        # EEG Branch - Processes spectral and spatial features
        self.eeg_spectral = nn.Sequential(
            nn.Linear(n_eeg_channels * n_bands, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # EEG Connectivity Branch - Processes coherence matrices
        self.eeg_connectivity = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
        )

        # fNIRS Branch - Processes hemodynamic patterns
        self.fnirs_encoder = nn.Sequential(
            nn.Linear(n_fnirs_channels * 2, 256),  # HbO + HbR
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Fusion Network
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        )

        # Classification Head (for training)
        self.classifier = nn.Linear(embedding_dim, 100)  # 100 sensory classes

    def forward(self, eeg_power: torch.Tensor,
                eeg_coherence: torch.Tensor,
                fnirs_activation: torch.Tensor) -> dict:
        """
        Forward pass through fingerprint encoder.

        Args:
            eeg_power: (batch, n_channels, n_bands)
            eeg_coherence: (batch, n_bands, n_channels, n_channels)
            fnirs_activation: (batch, n_channels, 2) - HbO and HbR

        Returns:
            Dictionary with embedding and class logits
        """
        batch_size = eeg_power.size(0)

        # Process EEG spectral features
        eeg_flat = eeg_power.view(batch_size, -1)
        eeg_spectral_feat = self.eeg_spectral(eeg_flat)

        # Process EEG connectivity
        eeg_conn_feat = self.eeg_connectivity(eeg_coherence)

        # Process fNIRS
        fnirs_flat = fnirs_activation.view(batch_size, -1)
        fnirs_feat = self.fnirs_encoder(fnirs_flat)

        # Fuse modalities
        combined = torch.cat([eeg_spectral_feat, eeg_conn_feat, fnirs_feat], dim=-1)
        embedding = self.fusion(combined)

        # Normalize embedding for cosine similarity
        embedding_norm = F.normalize(embedding, p=2, dim=-1)

        # Classification (for supervised training)
        logits = self.classifier(embedding)

        return {
            "embedding": embedding_norm,
            "logits": logits,
            "eeg_features": eeg_spectral_feat,
            "fnirs_features": fnirs_feat,
        }
```

#### 4.3.2 Contrastive Learning for Fingerprint Similarity

```python
class ContrastiveFingerprintLoss(nn.Module):
    """
    Contrastive loss for learning fingerprint embeddings.

    Same sensory experience → embeddings close together
    Different experiences → embeddings far apart
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss.

        Args:
            embeddings: (batch, embedding_dim) normalized vectors
            labels: (batch,) sensory class labels

        Returns:
            Scalar loss value
        """
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive pair mask (same label)
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)  # Exclude self

        # Compute log softmax
        log_prob = F.log_softmax(sim_matrix, dim=-1)

        # Compute loss over positive pairs
        positive_log_prob = (positive_mask * log_prob).sum(dim=-1)
        num_positives = positive_mask.sum(dim=-1)

        # Average over positives (avoid div by zero)
        loss = -positive_log_prob / (num_positives + 1e-8)

        return loss.mean()
```

-----

## 5. Fingerprint Database & Retrieval

### 5.1 Database Schema

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

class FingerprintRecord(Base):
    """
    Database schema for storing neural fingerprints.
    """
    __tablename__ = 'neural_fingerprints'

    id = Column(Integer, primary_key=True)
    fingerprint_id = Column(String(64), unique=True, index=True)

    # Sensory Classification
    modality = Column(String(32), index=True)  # gustatory, olfactory, etc.
    stimulus_label = Column(String(128), index=True)
    stimulus_category = Column(String(64))  # fruit, spice, floral, etc.

    # Subject Info
    subject_id = Column(String(64), index=True)
    session_id = Column(String(64))
    timestamp = Column(DateTime, index=True)

    # Embedding Vector (for similarity search)
    embedding = Column(ARRAY(Float))
    embedding_dim = Column(Integer)

    # Compressed Feature Data
    features_blob = Column(LargeBinary)  # Compressed numpy arrays

    # Quality Metrics
    signal_quality = Column(Float)
    confidence = Column(Float)
    validation_score = Column(Float)  # Cross-validation accuracy

    # Hardware Config
    eeg_channels = Column(Integer)
    fnirs_channels = Column(Integer)
    hardware_version = Column(String(32))


class StimulusLibrary(Base):
    """
    Reference library of standard sensory stimuli.
    """
    __tablename__ = 'stimulus_library'

    id = Column(Integer, primary_key=True)
    stimulus_label = Column(String(128), unique=True)
    modality = Column(String(32))
    category = Column(String(64))

    # Semantic descriptors
    descriptors = Column(ARRAY(String))  # ["sweet", "crisp", "fresh"]
    intensity_level = Column(Float)  # 0-1 normalized

    # Reference fingerprint (average across subjects)
    reference_embedding = Column(ARRAY(Float))

    # VR Integration
    vr_asset_id = Column(String(64))  # Link to VR object
    stimulation_protocol_id = Column(String(64))
```

### 5.2 Similarity Search

```python
import faiss
import numpy as np
from typing import List, Tuple

class FingerprintIndex:
    """
    Fast similarity search over neural fingerprint database
    using FAISS vector index.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

        # Use Inner Product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Mapping from index position to fingerprint ID
        self.id_map: List[str] = []

    def add_fingerprint(self, fingerprint_id: str,
                        embedding: np.ndarray) -> None:
        """Add a fingerprint to the index."""
        embedding = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)

        self.index.add(embedding)
        self.id_map.append(fingerprint_id)

    def search(self, query_embedding: np.ndarray,
               k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k most similar fingerprints to query.

        Returns:
            List of (fingerprint_id, similarity_score) tuples
        """
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.id_map[idx], float(score)))

        return results

    def find_matching_experience(self,
                                 current_fingerprint: np.ndarray,
                                 threshold: float = 0.85) -> Optional[str]:
        """
        Find best matching stored experience for playback.

        Returns fingerprint_id if similarity > threshold, else None.
        """
        results = self.search(current_fingerprint, k=1)

        if results and results[0][1] >= threshold:
            return results[0][0]
        return None
```

-----

## 6. Stimulation System for Sensory Playback

### 6.1 Transcranial Stimulation Hardware

#### 6.1.1 tDCS/tACS Module Design

```
┌─────────────────────────────────────────────────────────────────┐
│              TRANSCRANIAL STIMULATION MODULE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   DAC       │───▶│  Voltage-   │───▶│  Current    │          │
│  │  16-bit     │    │  Controlled │    │  Monitor    │          │
│  │  ±10V       │    │  Current    │    │  Feedback   │          │
│  └─────────────┘    │  Source     │    └─────────────┘          │
│                     │  ±2mA max   │          │                   │
│  ┌─────────────┐    └──────┬──────┘          │                   │
│  │  Waveform   │           │                 │                   │
│  │  Generator  │           ▼                 ▼                   │
│  │  (ESP32)    │    ┌─────────────┐    ┌─────────────┐          │
│  └─────────────┘    │  Electrode  │    │  Safety     │          │
│        │            │  Switch     │    │  Interlock  │          │
│        │            │  Matrix     │    │  System     │          │
│        │            │  (8x8)      │    └─────────────┘          │
│        │            └─────────────┘                              │
│        │                  │                                      │
│        ▼                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    ELECTRODE ARRAY                           ││
│  │   Up to 64 stimulation electrodes selectable via matrix     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.1.2 Stimulation Parameters

```python
@dataclass
class StimulationProtocol:
    """
    Defines stimulation parameters for sensory playback.
    """
    # Current parameters
    current_amplitude_ua: float  # 100-2000 µA
    waveform: str  # "DC", "AC", "pulsed"
    frequency_hz: Optional[float]  # For AC/pulsed (0.1-100 Hz)
    duty_cycle: Optional[float]  # For pulsed (0-1)

    # Duration
    ramp_up_s: float = 30.0  # Gradual onset
    stimulation_duration_s: float = 300.0  # Main stimulation
    ramp_down_s: float = 30.0  # Gradual offset

    # Electrode configuration
    anode_electrodes: List[str]  # e.g., ["F7", "FT7"]
    cathode_electrodes: List[str]  # e.g., ["Fp2", "F8"]

    # Safety limits
    max_current_density_ua_cm2: float = 25.0
    max_charge_density_uc_cm2: float = 40.0

    def validate(self) -> bool:
        """Check protocol is within safety limits."""
        if self.current_amplitude_ua > 2000:
            return False
        if self.frequency_hz and self.frequency_hz > 100:
            return False
        return True


# Predefined protocols for sensory modalities
GUSTATORY_PROTOCOL = StimulationProtocol(
    current_amplitude_ua=1000,
    waveform="AC",
    frequency_hz=40.0,  # Gamma entrainment
    anode_electrodes=["F7", "FT7", "T7"],
    cathode_electrodes=["F8", "FT8", "T8"],
    stimulation_duration_s=10.0,
)

OLFACTORY_PROTOCOL = StimulationProtocol(
    current_amplitude_ua=800,
    waveform="pulsed",
    frequency_hz=6.0,  # Theta rhythm (olfactory)
    duty_cycle=0.5,
    anode_electrodes=["Fp1", "Fp2", "AF7", "AF8"],
    cathode_electrodes=["Cz", "Pz"],
    stimulation_duration_s=5.0,
)
```

### 6.2 Fingerprint-to-Stimulation Mapping

#### 6.2.1 Inverse Model

```python
class FingerprintToStimulation:
    """
    Maps neural fingerprint patterns to stimulation parameters
    that will recreate similar brain states.
    """

    def __init__(self, model_path: str):
        self.model = self._load_inverse_model(model_path)

        # Electrode positions for field modeling
        self.electrode_positions = self._load_electrode_positions()

    def _load_inverse_model(self, path: str) -> nn.Module:
        """Load trained inverse mapping neural network."""
        model = InverseStimulationModel()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def compute_stimulation_params(self,
                                   target_fingerprint: NeuralFingerprint,
                                   current_brain_state: np.ndarray) -> StimulationProtocol:
        """
        Compute optimal stimulation to drive brain toward target state.

        Uses forward model of brain response to stimulation combined
        with optimization to find minimal stimulation that achieves
        target pattern.
        """
        # Convert fingerprint to target activation pattern
        target_activation = self._fingerprint_to_activation(target_fingerprint)

        # Compute difference from current state
        delta_activation = target_activation - current_brain_state

        # Run inverse model
        with torch.no_grad():
            stim_params = self.model(
                torch.tensor(delta_activation).float().unsqueeze(0)
            )

        # Convert to protocol
        return self._params_to_protocol(stim_params.numpy())

    def _fingerprint_to_activation(self, fp: NeuralFingerprint) -> np.ndarray:
        """Convert fingerprint to target cortical activation map."""
        # Use topography and band power to create spatial target
        activation = np.zeros(128)  # One value per electrode

        # Weight by gamma power (sensory processing)
        gamma_idx = list(FREQUENCY_BANDS.keys()).index("gamma")
        activation = fp.eeg_band_power[:, gamma_idx] * fp.eeg_topography

        return activation / np.max(activation)

    def _params_to_protocol(self, params: np.ndarray) -> StimulationProtocol:
        """Convert model output to stimulation protocol."""
        # params: [amplitude, frequency, phase, electrode_weights...]

        amplitude = np.clip(params[0] * 2000, 100, 2000)  # µA
        frequency = np.clip(params[1] * 100, 0.1, 100)  # Hz

        # Select electrodes based on weights
        electrode_weights = params[2:2+128]
        anode_mask = electrode_weights > 0.5
        cathode_mask = electrode_weights < -0.5

        electrode_names = list(ELECTRODE_POSITIONS.keys())
        anodes = [electrode_names[i] for i in np.where(anode_mask)[0][:4]]
        cathodes = [electrode_names[i] for i in np.where(cathode_mask)[0][:4]]

        return StimulationProtocol(
            current_amplitude_ua=amplitude,
            waveform="AC",
            frequency_hz=frequency,
            anode_electrodes=anodes or ["Cz"],
            cathode_electrodes=cathodes or ["Fp1", "Fp2"],
        )


class InverseStimulationModel(nn.Module):
    """
    Neural network that learns mapping from desired brain state
    change to required stimulation parameters.

    Trained on paired data of (stimulation, brain_response).
    """

    def __init__(self, n_electrodes: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_electrodes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Output heads
        self.amplitude_head = nn.Linear(128, 1)
        self.frequency_head = nn.Linear(128, 1)
        self.electrode_head = nn.Linear(128, n_electrodes)

    def forward(self, target_delta: torch.Tensor) -> torch.Tensor:
        features = self.encoder(target_delta)

        amplitude = torch.sigmoid(self.amplitude_head(features))
        frequency = torch.sigmoid(self.frequency_head(features))
        electrodes = torch.tanh(self.electrode_head(features))

        return torch.cat([amplitude, frequency, electrodes], dim=-1)
```

-----

## 7. Bidirectional Feedback Loop

### 7.1 Real-Time Processing Pipeline

```python
import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum

class SystemState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    STIMULATING = "stimulating"
    FEEDBACK = "feedback"

@dataclass
class BidirectionalSession:
    """
    Manages real-time bidirectional neural interface session.
    """
    session_id: str
    target_experience: str  # e.g., "apple_taste"

    # State tracking
    state: SystemState = SystemState.IDLE
    current_fingerprint: Optional[NeuralFingerprint] = None
    target_fingerprint: Optional[NeuralFingerprint] = None

    # Feedback metrics
    similarity_history: deque = field(default_factory=lambda: deque(maxlen=100))
    stimulation_history: List[StimulationProtocol] = field(default_factory=list)


class RealTimeFeedbackController:
    """
    Controls bidirectional feedback loop for sensory simulation.

    Loop:
    1. Acquire current brain state (EEG + fNIRS)
    2. Compare to target fingerprint
    3. Compute corrective stimulation
    4. Apply stimulation
    5. Measure response
    6. Repeat until target achieved or timeout
    """

    def __init__(self,
                 acquisition_system,
                 stimulation_system,
                 fingerprint_db: FingerprintIndex,
                 inverse_model: FingerprintToStimulation):
        self.acquisition = acquisition_system
        self.stimulation = stimulation_system
        self.fingerprint_db = fingerprint_db
        self.inverse_model = inverse_model

        self.session: Optional[BidirectionalSession] = None
        self.running = False

        # Control parameters
        self.update_rate_hz = 10.0
        self.similarity_threshold = 0.90
        self.max_iterations = 100

    async def start_session(self, target_experience: str) -> str:
        """Initialize new sensory simulation session."""
        session_id = self._generate_session_id()

        # Load target fingerprint
        target_fp = self._load_target_fingerprint(target_experience)
        if target_fp is None:
            raise ValueError(f"No fingerprint found for: {target_experience}")

        self.session = BidirectionalSession(
            session_id=session_id,
            target_experience=target_experience,
            target_fingerprint=target_fp,
        )

        self.running = True
        asyncio.create_task(self._feedback_loop())

        return session_id

    async def _feedback_loop(self):
        """Main feedback control loop."""
        iteration = 0

        while self.running and iteration < self.max_iterations:
            self.session.state = SystemState.RECORDING

            # 1. Acquire current brain state
            eeg_data, fnirs_data = await self.acquisition.get_current_data(
                duration_s=0.1  # 100ms window
            )

            self.session.state = SystemState.PROCESSING

            # 2. Extract current fingerprint
            current_fp = self._extract_fingerprint(eeg_data, fnirs_data)
            self.session.current_fingerprint = current_fp

            # 3. Compute similarity to target
            similarity = current_fp.similarity(self.session.target_fingerprint)
            self.session.similarity_history.append(similarity)

            # 4. Check if target achieved
            if similarity >= self.similarity_threshold:
                print(f"Target achieved! Similarity: {similarity:.3f}")
                break

            # 5. Compute corrective stimulation
            current_state = current_fp.to_vector()
            protocol = self.inverse_model.compute_stimulation_params(
                self.session.target_fingerprint,
                current_state
            )

            self.session.state = SystemState.STIMULATING

            # 6. Apply stimulation
            await self.stimulation.apply_protocol(protocol)
            self.session.stimulation_history.append(protocol)

            self.session.state = SystemState.FEEDBACK

            # 7. Wait for next iteration
            await asyncio.sleep(1.0 / self.update_rate_hz)
            iteration += 1

        self.session.state = SystemState.IDLE
        self.running = False

    def _extract_fingerprint(self, eeg_data: np.ndarray,
                            fnirs_data: np.ndarray) -> NeuralFingerprint:
        """Extract fingerprint from current data window."""
        # Use feature extractor from Section 4
        extractor = EEGFeatureExtractor()
        features = extractor.extract_features(eeg_data)

        # Create fingerprint object
        return NeuralFingerprint(
            fingerprint_id=f"live_{time.time()}",
            sensory_modality=self.session.target_fingerprint.sensory_modality,
            stimulus_label="live_recording",
            timestamp=datetime.now(),
            subject_id=self.session.session_id,
            eeg_band_power=features["band_power"],
            eeg_coherence=features["coherence"],
            eeg_phase_sync=features["phase_sync"],
            eeg_topography=features["topography"],
            eeg_erp_components=features["erp_components"],
            eeg_entropy=features["entropy"],
            fnirs_hbo_pattern=fnirs_data[:, 0, :],
            fnirs_hbr_pattern=fnirs_data[:, 1, :],
            fnirs_spatial_activation=np.mean(fnirs_data[:, 0, :], axis=-1),
            neurovascular_coupling={},
            temporal_dynamics=np.array([]),
            signal_quality=0.0,
            confidence=0.0,
        )

    async def stop_session(self):
        """Gracefully stop current session."""
        self.running = False
        await self.stimulation.ramp_down()
```

### 7.2 VR Integration Interface

```python
from typing import Callable
import websockets
import json

class VRSensoryBridge:
    """
    WebSocket bridge between VR environment and neural interface.

    Receives sensory event triggers from VR and initiates
    corresponding neural stimulation sequences.
    """

    def __init__(self,
                 feedback_controller: RealTimeFeedbackController,
                 host: str = "localhost",
                 port: int = 8765):
        self.controller = feedback_controller
        self.host = host
        self.port = port

        self.active_connections = set()

    async def start_server(self):
        """Start WebSocket server for VR connection."""
        async with websockets.serve(self._handle_connection, self.host, self.port):
            print(f"VR Sensory Bridge listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def _handle_connection(self, websocket):
        """Handle incoming VR connection."""
        self.active_connections.add(websocket)

        try:
            async for message in websocket:
                event = json.loads(message)
                await self._process_vr_event(event, websocket)
        finally:
            self.active_connections.remove(websocket)

    async def _process_vr_event(self, event: dict, websocket):
        """
        Process sensory event from VR environment.

        Event format:
        {
            "type": "sensory_trigger",
            "modality": "gustatory",
            "stimulus": "apple_taste",
            "intensity": 0.8,
            "duration_s": 5.0,
            "position": {"x": 0, "y": 0, "z": 0}  # VR coordinates
        }
        """
        event_type = event.get("type")

        if event_type == "sensory_trigger":
            # Start neural stimulation for requested experience
            stimulus = event["stimulus"]
            intensity = event.get("intensity", 1.0)

            session_id = await self.controller.start_session(stimulus)

            # Send confirmation back to VR
            await websocket.send(json.dumps({
                "type": "stimulation_started",
                "session_id": session_id,
                "stimulus": stimulus,
            }))

        elif event_type == "sensory_stop":
            # Stop current stimulation
            await self.controller.stop_session()

            await websocket.send(json.dumps({
                "type": "stimulation_stopped",
            }))

        elif event_type == "feedback_request":
            # Send current similarity feedback to VR
            if self.controller.session:
                history = list(self.controller.session.similarity_history)
                await websocket.send(json.dumps({
                    "type": "feedback_data",
                    "similarity": history[-1] if history else 0,
                    "history": history[-10:],  # Last 10 values
                }))

    async def broadcast_status(self, status: dict):
        """Broadcast status update to all connected VR clients."""
        if self.active_connections:
            message = json.dumps(status)
            await asyncio.gather(
                *[ws.send(message) for ws in self.active_connections]
            )
```

-----

## 8. Training Data Collection Protocol

### 8.1 Sensory Stimulus Presentation

```python
@dataclass
class StimulusSession:
    """
    Protocol for collecting neural fingerprints from real stimuli.
    """
    session_id: str
    subject_id: str
    modality: str
    stimuli: List[str]

    # Timing parameters
    baseline_duration_s: float = 10.0
    stimulus_duration_s: float = 5.0
    washout_duration_s: float = 30.0
    repetitions_per_stimulus: int = 10

    # Randomization
    randomize_order: bool = True
    include_catch_trials: bool = True


class DataCollectionManager:
    """
    Manages collection of training data for fingerprint library.
    """

    def __init__(self, acquisition_system, database):
        self.acquisition = acquisition_system
        self.database = database

    async def run_collection_session(self, session: StimulusSession):
        """
        Run full data collection session.

        For each stimulus:
        1. Record baseline
        2. Present stimulus
        3. Record neural response
        4. Store fingerprint
        5. Washout period
        """
        stimulus_order = session.stimuli.copy()
        if session.randomize_order:
            np.random.shuffle(stimulus_order)

        all_fingerprints = []

        for rep in range(session.repetitions_per_stimulus):
            for stimulus in stimulus_order:
                print(f"Trial {rep+1}/{session.repetitions_per_stimulus}: {stimulus}")

                # 1. Baseline recording
                print("  Recording baseline...")
                baseline_data = await self.acquisition.record(
                    duration_s=session.baseline_duration_s
                )

                # 2. Present stimulus (hardware-specific)
                print(f"  Presenting: {stimulus}")
                await self._present_stimulus(stimulus, session.modality)

                # 3. Record response
                print("  Recording response...")
                response_data = await self.acquisition.record(
                    duration_s=session.stimulus_duration_s
                )

                # 4. Extract and store fingerprint
                fingerprint = self._extract_fingerprint(
                    response_data,
                    baseline_data,
                    stimulus,
                    session
                )
                all_fingerprints.append(fingerprint)

                # 5. Washout
                print(f"  Washout ({session.washout_duration_s}s)...")
                await asyncio.sleep(session.washout_duration_s)

        # Compute average fingerprint per stimulus
        averaged = self._average_fingerprints_by_stimulus(all_fingerprints)

        # Store in database
        for fp in averaged:
            self.database.store_fingerprint(fp)

        return averaged

    async def _present_stimulus(self, stimulus: str, modality: str):
        """
        Present sensory stimulus to subject.

        For gustatory: Dispense taste solution
        For olfactory: Release odorant
        """
        if modality == "gustatory":
            # Interface with taste delivery system
            await self.taste_dispenser.deliver(stimulus)
        elif modality == "olfactory":
            # Interface with olfactometer
            await self.olfactometer.present(stimulus)

    def _average_fingerprints_by_stimulus(self,
                                          fingerprints: List[NeuralFingerprint]
                                          ) -> List[NeuralFingerprint]:
        """Average multiple trials of same stimulus."""
        from collections import defaultdict

        grouped = defaultdict(list)
        for fp in fingerprints:
            grouped[fp.stimulus_label].append(fp)

        averaged = []
        for label, fps in grouped.items():
            # Average all numeric features
            avg_fp = NeuralFingerprint(
                fingerprint_id=f"avg_{label}_{time.time()}",
                sensory_modality=fps[0].sensory_modality,
                stimulus_label=label,
                timestamp=datetime.now(),
                subject_id=fps[0].subject_id,
                eeg_band_power=np.mean([f.eeg_band_power for f in fps], axis=0),
                eeg_coherence=np.mean([f.eeg_coherence for f in fps], axis=0),
                eeg_phase_sync=np.mean([f.eeg_phase_sync for f in fps], axis=0),
                eeg_topography=np.mean([f.eeg_topography for f in fps], axis=0),
                eeg_erp_components=fps[0].eeg_erp_components,  # Use first
                eeg_entropy=np.mean([f.eeg_entropy for f in fps], axis=0),
                fnirs_hbo_pattern=np.mean([f.fnirs_hbo_pattern for f in fps], axis=0),
                fnirs_hbr_pattern=np.mean([f.fnirs_hbr_pattern for f in fps], axis=0),
                fnirs_spatial_activation=np.mean(
                    [f.fnirs_spatial_activation for f in fps], axis=0
                ),
                neurovascular_coupling={},
                temporal_dynamics=np.array([]),
                signal_quality=np.mean([f.signal_quality for f in fps]),
                confidence=1.0 / np.std([f.similarity(fps[0]) for f in fps]),
            )
            averaged.append(avg_fp)

        return averaged
```

-----

## 9. Safety Considerations

### 9.1 Stimulation Safety Limits

```python
@dataclass
class SafetyLimits:
    """
    Hard safety limits for transcranial stimulation.
    Based on established safety guidelines.
    """
    # Current limits
    MAX_CURRENT_UA: int = 2000
    MAX_CURRENT_DENSITY_UA_CM2: float = 25.0

    # Charge limits
    MAX_CHARGE_PER_PHASE_UC: float = 60.0
    MAX_CHARGE_DENSITY_UC_CM2: float = 40.0

    # Duration limits
    MAX_SESSION_DURATION_MIN: int = 40
    MIN_INTER_SESSION_HOURS: int = 24

    # Electrode limits
    MIN_ELECTRODE_SIZE_CM2: float = 9.0  # 3x3 cm minimum

    # Frequency limits for tACS
    MAX_FREQUENCY_HZ: float = 100.0

    # Impedance monitoring
    MAX_IMPEDANCE_KOHM: float = 10.0
    MIN_IMPEDANCE_KOHM: float = 0.1


class SafetyMonitor:
    """
    Real-time safety monitoring during stimulation.
    """

    def __init__(self, limits: SafetyLimits = SafetyLimits()):
        self.limits = limits
        self.session_start_time: Optional[datetime] = None
        self.total_charge_delivered_uc: float = 0.0

    def check_protocol(self, protocol: StimulationProtocol) -> Tuple[bool, str]:
        """Validate protocol before execution."""

        # Check current amplitude
        if protocol.current_amplitude_ua > self.limits.MAX_CURRENT_UA:
            return False, f"Current exceeds limit: {protocol.current_amplitude_ua} µA"

        # Check frequency
        if protocol.frequency_hz and protocol.frequency_hz > self.limits.MAX_FREQUENCY_HZ:
            return False, f"Frequency exceeds limit: {protocol.frequency_hz} Hz"

        # Check session duration
        if self.session_start_time:
            elapsed = (datetime.now() - self.session_start_time).seconds / 60
            if elapsed + protocol.stimulation_duration_s/60 > self.limits.MAX_SESSION_DURATION_MIN:
                return False, "Would exceed maximum session duration"

        return True, "Protocol approved"

    def monitor_during_stimulation(self,
                                   current_ua: float,
                                   impedance_kohm: float) -> Optional[str]:
        """
        Real-time monitoring during active stimulation.
        Returns error message if limits exceeded, None if OK.
        """
        if current_ua > self.limits.MAX_CURRENT_UA:
            return "EMERGENCY: Current limit exceeded"

        if impedance_kohm > self.limits.MAX_IMPEDANCE_KOHM:
            return "WARNING: High impedance detected"

        if impedance_kohm < self.limits.MIN_IMPEDANCE_KOHM:
            return "WARNING: Possible electrode short"

        return None

    def emergency_shutdown(self):
        """Immediately cease all stimulation."""
        # Send shutdown signal to hardware
        pass
```

-----

## 10. Integration with Cerelog/OpenBCI

### 10.1 Hardware Abstraction Layer

```rust
//! Cerelog integration module for neural fingerprint system
//!
//! Provides unified interface for EEG + fNIRS acquisition
//! and transcranial stimulation control.

use crate::hal::{EegAdc, FnirsController, StimulationDriver};

/// Main interface to Cerelog hardware
pub struct CerelogInterface {
    eeg: EegAdc,
    fnirs: FnirsController,
    stim: StimulationDriver,

    config: CerelogConfig,
}

#[derive(Clone)]
pub struct CerelogConfig {
    pub eeg_channels: u8,
    pub eeg_sample_rate: u32,
    pub fnirs_sources: u8,
    pub fnirs_detectors: u8,
    pub fnirs_sample_rate: u32,
    pub stim_channels: u8,
}

impl Default for CerelogConfig {
    fn default() -> Self {
        Self {
            eeg_channels: 128,
            eeg_sample_rate: 500,
            fnirs_sources: 32,
            fnirs_detectors: 32,
            fnirs_sample_rate: 25,
            stim_channels: 8,
        }
    }
}

impl CerelogInterface {
    pub fn new(config: CerelogConfig) -> Result<Self, CerelogError> {
        Ok(Self {
            eeg: EegAdc::init(config.eeg_channels, config.eeg_sample_rate)?,
            fnirs: FnirsController::init(config.fnirs_sources, config.fnirs_detectors)?,
            stim: StimulationDriver::init(config.stim_channels)?,
            config,
        })
    }

    /// Start synchronized acquisition
    pub fn start_acquisition(&mut self) -> Result<(), CerelogError> {
        self.eeg.start()?;
        self.fnirs.start()?;
        Ok(())
    }

    /// Get latest data from all modalities
    pub fn get_data(&mut self) -> AcquisitionData {
        AcquisitionData {
            eeg: self.eeg.read_buffer(),
            fnirs: self.fnirs.read_buffer(),
            timestamp_us: self.eeg.get_timestamp(),
        }
    }

    /// Apply stimulation protocol
    pub fn apply_stimulation(&mut self, protocol: &StimProtocol) -> Result<(), CerelogError> {
        // Validate safety
        if !protocol.validate() {
            return Err(CerelogError::SafetyViolation);
        }

        self.stim.configure(protocol)?;
        self.stim.start()?;

        Ok(())
    }
}

pub struct AcquisitionData {
    pub eeg: Vec<Vec<f32>>,      // [channel][sample]
    pub fnirs: FnirsData,
    pub timestamp_us: u64,
}

pub struct FnirsData {
    pub intensity_760: Vec<Vec<f32>>,  // [channel][sample]
    pub intensity_850: Vec<Vec<f32>>,
}
```

### 10.2 Python Bindings

```python
# cerelog_bindings.py - Python interface to Rust Cerelog library

import ctypes
from pathlib import Path
import numpy as np

class CerelogPython:
    """
    Python bindings for Cerelog hardware interface.
    Wraps the Rust library via ctypes FFI.
    """

    def __init__(self, lib_path: str = "./libcerelog.so"):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()

        # Initialize hardware
        self.handle = self.lib.cerelog_init()
        if not self.handle:
            raise RuntimeError("Failed to initialize Cerelog")

    def _setup_function_signatures(self):
        """Define C function signatures."""
        self.lib.cerelog_init.restype = ctypes.c_void_p
        self.lib.cerelog_start.argtypes = [ctypes.c_void_p]
        self.lib.cerelog_get_eeg.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        self.lib.cerelog_get_fnirs.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

    def start(self):
        """Start data acquisition."""
        self.lib.cerelog_start(self.handle)

    def get_eeg_data(self, n_samples: int = 500) -> np.ndarray:
        """Get latest EEG samples."""
        buffer = (ctypes.c_float * (128 * n_samples))()
        self.lib.cerelog_get_eeg(self.handle, buffer, n_samples)
        return np.array(buffer).reshape(128, n_samples)

    def get_fnirs_data(self, n_samples: int = 25) -> np.ndarray:
        """Get latest fNIRS samples."""
        buffer = (ctypes.c_float * (64 * 2 * n_samples))()
        self.lib.cerelog_get_fnirs(self.handle, buffer, n_samples)
        return np.array(buffer).reshape(64, 2, n_samples)

    def apply_stimulation(self, protocol: dict):
        """Apply stimulation protocol."""
        # Convert protocol to C struct and call
        pass

    def close(self):
        """Cleanup hardware."""
        self.lib.cerelog_close(self.handle)
```

-----

## 11. Quick Start Guide

### 11.1 Installation

```bash
# Clone repository
git clone https://github.com/your-org/neural-fingerprint-system.git
cd neural-fingerprint-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Cerelog Rust library
cd cerelog-driver
cargo build --release
cp target/release/libcerelog.so ../

# Initialize database
python scripts/init_database.py
```

### 11.2 Basic Usage

```python
from neural_fingerprint import (
    CerelogPython,
    FingerprintExtractor,
    StimulationController,
    VRSensoryBridge
)

# Initialize hardware
cerelog = CerelogPython()
cerelog.start()

# Create fingerprint extractor
extractor = FingerprintExtractor(
    eeg_channels=128,
    fnirs_channels=64
)

# Load fingerprint database
from neural_fingerprint.database import FingerprintDatabase
db = FingerprintDatabase("fingerprints.db")

# Record a new fingerprint
print("Present stimulus now...")
eeg_data = cerelog.get_eeg_data(n_samples=2500)  # 5 seconds
fnirs_data = cerelog.get_fnirs_data(n_samples=125)

fingerprint = extractor.extract(eeg_data, fnirs_data)
fingerprint.stimulus_label = "apple_taste"
db.store(fingerprint)

# Playback a stored fingerprint
target = db.load("apple_taste")
controller = StimulationController(cerelog)
controller.play_fingerprint(target)
```

-----

## 12. Future Enhancements

1. **Higher Resolution fNIRS**: Upgrade to time-domain fNIRS for improved depth resolution
2. **Closed-Loop Optimization**: Implement reinforcement learning for stimulation parameter tuning
3. **Multi-Subject Fingerprints**: Transfer learning across subjects for universal fingerprints
4. **Haptic Integration**: Add vibrotactile feedback for touch simulation
5. **Emotion Mapping**: Extend to emotional state fingerprints

-----

## References

1. Nitsche, M. A., & Paulus, W. (2000). Excitability changes induced in the human motor cortex by weak transcranial direct current stimulation. *The Journal of physiology*, 527(3), 633-639.
2. Scholkmann, F., et al. (2014). A review on continuous wave functional near-infrared spectroscopy and imaging instrumentation and methodology. *Neuroimage*, 85, 6-27.
3. Herrmann, C. S., et al. (2013). Transcranial alternating current stimulation: a review of the underlying mechanisms and modulation of cognitive processes. *Frontiers in human neuroscience*, 7, 279.
4. Makeig, S., et al. (2004). Mining event-related brain dynamics. *Trends in cognitive sciences*, 8(5), 204-210.

-----

*Document generated for BEAST Project - Bidirectional Electromagnetic Adaptive Sensory Technology*

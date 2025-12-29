//! Cortical Channel Mapping
//!
//! Maps sensory modalities to EEG electrodes and fNIRS optodes.
//!
//! # Somatotopic Organization
//!
//! - S1 (Primary Somatosensory): C3/C4 for contralateral body
//! - A1 (Primary Auditory): T7/T8 temporal electrodes
//! - Gustatory cortex: Insula/frontal operculum (deep, fNIRS better)

use rootstar_bci_core::sns::types::{BodyRegion, Ear};
use rootstar_bci_core::types::{EegChannel, FnirsChannel};
use serde::{Deserialize, Serialize};

/// Cortical channel mapping for all modalities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorticalChannelMap {
    /// Tactile (somatosensory) channels
    pub tactile: TactileCortexChannels,
    /// Auditory channels
    pub auditory: AuditoryCortexChannels,
    /// Gustatory channels
    pub gustatory: GustatoryCortexChannels,
}

impl CorticalChannelMap {
    /// Create default mapping based on standard 10-20 electrode positions
    #[must_use]
    pub fn default_8ch() -> Self {
        Self {
            tactile: TactileCortexChannels::default_8ch(),
            auditory: AuditoryCortexChannels::default_8ch(),
            gustatory: GustatoryCortexChannels::default_8ch(),
        }
    }
}

impl Default for CorticalChannelMap {
    fn default() -> Self {
        Self::default_8ch()
    }
}

/// Somatosensory cortex channel assignments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TactileCortexChannels {
    /// C3 electrode (right body representation)
    pub c3: EegChannel,
    /// C4 electrode (left body representation)
    pub c4: EegChannel,
    /// Cz electrode (midline - feet, trunk)
    pub cz: Option<EegChannel>,
    /// fNIRS optodes over S1 (if available)
    pub fnirs_s1: Vec<FnirsChannel>,
    /// Reference channels for common average
    pub reference_channels: Vec<EegChannel>,
}

impl TactileCortexChannels {
    /// Create for 8-channel setup (no explicit Cz, use C3/C4)
    #[must_use]
    pub fn default_8ch() -> Self {
        Self {
            c3: EegChannel::C3,
            c4: EegChannel::C4,
            cz: None, // Not available in standard 8ch montage
            fnirs_s1: vec![
                FnirsChannel::new(0, 0, 30), // Left S1
                FnirsChannel::new(1, 1, 30), // Right S1
            ],
            reference_channels: vec![EegChannel::P3, EegChannel::P4],
        }
    }

    /// Get the primary electrode for a body region
    #[must_use]
    pub fn electrode_for_region(&self, region: BodyRegion) -> EegChannel {
        use rootstar_bci_core::sns::types::{Finger, Hand, Side};

        match region {
            // Right side body → Left hemisphere (C3)
            BodyRegion::Fingertip(Finger::Thumb | Finger::Index | Finger::Middle | Finger::Ring | Finger::Little) => {
                // Default to right hand → C3
                self.c3
            }
            BodyRegion::Palm(Hand::Right) => self.c3,
            BodyRegion::Palm(Hand::Left) => self.c4,
            BodyRegion::Forearm(Side::Right) | BodyRegion::UpperArm(Side::Right) => self.c3,
            BodyRegion::Forearm(Side::Left) | BodyRegion::UpperArm(Side::Left) => self.c4,
            BodyRegion::Face => self.c3, // Face is bilateral, use C3 as default
            BodyRegion::Trunk => self.cz.unwrap_or(self.c3),
            BodyRegion::FootSole(Side::Right) => self.c3,
            BodyRegion::FootSole(Side::Left) => self.c4,
        }
    }

    /// Get the contralateral electrode pair for bilateral comparison
    #[must_use]
    pub fn bilateral_pair(&self) -> (EegChannel, EegChannel) {
        (self.c3, self.c4)
    }
}

impl Default for TactileCortexChannels {
    fn default() -> Self {
        Self::default_8ch()
    }
}

/// Auditory cortex channel assignments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditoryCortexChannels {
    /// T7 electrode (left auditory cortex) - may need remapping
    pub t7: Option<EegChannel>,
    /// T8 electrode (right auditory cortex) - may need remapping
    pub t8: Option<EegChannel>,
    /// Cz electrode (vertex - AEP)
    pub cz: Option<EegChannel>,
    /// Fz electrode (frontal - for MMN)
    pub fz: Option<EegChannel>,
    /// Fallback channels if T7/T8 not available
    pub fallback_left: EegChannel,
    pub fallback_right: EegChannel,
    /// fNIRS optodes over temporal cortex
    pub fnirs_a1: Vec<FnirsChannel>,
}

impl AuditoryCortexChannels {
    /// Create for 8-channel setup (approximate with available channels)
    #[must_use]
    pub fn default_8ch() -> Self {
        // Standard 8ch doesn't have T7/T8, use approximations
        Self {
            t7: None,
            t8: None,
            cz: None, // Approximate with P3/P4
            fz: None, // Use Fp1/Fp2 as frontal approximation
            fallback_left: EegChannel::P3,  // Posterior left
            fallback_right: EegChannel::P4, // Posterior right
            fnirs_a1: vec![
                FnirsChannel::new(2, 2, 30), // Left temporal
                FnirsChannel::new(3, 3, 30), // Right temporal
            ],
        }
    }

    /// Get the primary electrodes for auditory processing
    #[must_use]
    pub fn primary_electrodes(&self) -> Vec<EegChannel> {
        let mut electrodes = Vec::new();
        if let Some(t7) = self.t7 {
            electrodes.push(t7);
        } else {
            electrodes.push(self.fallback_left);
        }
        if let Some(t8) = self.t8 {
            electrodes.push(t8);
        } else {
            electrodes.push(self.fallback_right);
        }
        electrodes
    }

    /// Get electrode for specific ear
    #[must_use]
    pub fn electrode_for_ear(&self, ear: Ear) -> EegChannel {
        match ear {
            // Contralateral processing
            Ear::Left => self.t8.unwrap_or(self.fallback_right),
            Ear::Right => self.t7.unwrap_or(self.fallback_left),
        }
    }
}

impl Default for AuditoryCortexChannels {
    fn default() -> Self {
        Self::default_8ch()
    }
}

/// Gustatory cortex channel assignments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GustatoryCortexChannels {
    /// Frontal electrodes (nearest to insula surface projection)
    pub frontal_left: EegChannel,
    pub frontal_right: EegChannel,
    /// fNIRS optodes over frontal operculum (best for gustatory)
    pub fnirs_gustatory: Vec<FnirsChannel>,
    /// Theta band is particularly relevant for gustatory processing
    pub theta_band_hz: (f32, f32),
}

impl GustatoryCortexChannels {
    /// Create for 8-channel setup
    #[must_use]
    pub fn default_8ch() -> Self {
        Self {
            frontal_left: EegChannel::Fp1,
            frontal_right: EegChannel::Fp2,
            fnirs_gustatory: vec![
                FnirsChannel::new(0, 0, 25), // Frontal left, shorter distance for better insula access
                FnirsChannel::new(1, 1, 25), // Frontal right
            ],
            theta_band_hz: (4.0, 8.0),
        }
    }

    /// Get primary electrodes for gustatory processing
    #[must_use]
    pub fn primary_electrodes(&self) -> Vec<EegChannel> {
        vec![self.frontal_left, self.frontal_right]
    }
}

impl Default for GustatoryCortexChannels {
    fn default() -> Self {
        Self::default_8ch()
    }
}

/// SEP (Somatosensory Evoked Potential) components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SepComponents {
    /// N20 component (20ms, initial cortical response)
    pub n20_latency_ms: f64,
    pub n20_amplitude_uv: f64,
    /// P25 component (25ms, secondary response)
    pub p25_latency_ms: f64,
    pub p25_amplitude_uv: f64,
    /// N30 component (30ms, later processing)
    pub n30_latency_ms: f64,
    pub n30_amplitude_uv: f64,
    /// Mu rhythm suppression (8-12 Hz)
    pub mu_suppression: f64,
    /// Beta rebound (15-25 Hz)
    pub beta_rebound: f64,
}

impl Default for SepComponents {
    fn default() -> Self {
        Self {
            n20_latency_ms: 20.0,
            n20_amplitude_uv: 0.0,
            p25_latency_ms: 25.0,
            p25_amplitude_uv: 0.0,
            n30_latency_ms: 30.0,
            n30_amplitude_uv: 0.0,
            mu_suppression: 0.0,
            beta_rebound: 0.0,
        }
    }
}

/// AEP (Auditory Evoked Potential) components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AepComponents {
    /// N1 component (100ms, auditory attention)
    pub n1_latency_ms: f64,
    pub n1_amplitude_uv: f64,
    /// P2 component (200ms)
    pub p2_latency_ms: f64,
    pub p2_amplitude_uv: f64,
    /// N2 component (200-300ms)
    pub n2_latency_ms: f64,
    pub n2_amplitude_uv: f64,
    /// 40 Hz ASSR (Auditory Steady-State Response)
    pub assr_power: f64,
    pub assr_phase: f64,
    /// Gamma band power (auditory binding)
    pub gamma_power: f64,
}

impl Default for AepComponents {
    fn default() -> Self {
        Self {
            n1_latency_ms: 100.0,
            n1_amplitude_uv: 0.0,
            p2_latency_ms: 200.0,
            p2_amplitude_uv: 0.0,
            n2_latency_ms: 250.0,
            n2_amplitude_uv: 0.0,
            assr_power: 0.0,
            assr_phase: 0.0,
            gamma_power: 0.0,
        }
    }
}

/// GEP (Gustatory Evoked Potential) components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepComponents {
    /// P1 component (130ms, highly variable)
    pub p1_latency_ms: f64,
    pub p1_amplitude_uv: f64,
    /// N1 component (180ms)
    pub n1_latency_ms: f64,
    pub n1_amplitude_uv: f64,
    /// Frontal theta power (associated with taste processing)
    pub frontal_theta: f64,
}

impl Default for GepComponents {
    fn default() -> Self {
        Self {
            p1_latency_ms: 130.0,
            p1_amplitude_uv: 0.0,
            n1_latency_ms: 180.0,
            n1_amplitude_uv: 0.0,
            frontal_theta: 0.0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortical_channel_map_default() {
        let map = CorticalChannelMap::default();

        assert_eq!(map.tactile.c3, EegChannel::C3);
        assert_eq!(map.tactile.c4, EegChannel::C4);
    }

    #[test]
    fn test_electrode_for_region() {
        use rootstar_bci_core::sns::types::{Finger, Hand};

        let channels = TactileCortexChannels::default_8ch();

        // Right hand → C3 (contralateral)
        let electrode = channels.electrode_for_region(BodyRegion::Palm(Hand::Right));
        assert_eq!(electrode, EegChannel::C3);

        // Left hand → C4 (contralateral)
        let electrode = channels.electrode_for_region(BodyRegion::Palm(Hand::Left));
        assert_eq!(electrode, EegChannel::C4);
    }

    #[test]
    fn test_auditory_electrodes() {
        let channels = AuditoryCortexChannels::default_8ch();
        let electrodes = channels.primary_electrodes();

        assert!(!electrodes.is_empty());
    }
}

//! Visual sensory receptor simulation
//!
//! This module models the visual pathway from photoreceptors to cortical processing:
//!
//! # Visual Pathway
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                          Visual Processing Pipeline                      │
//! │                                                                         │
//! │  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────┐ │
//! │  │ Photoreceptor│    │ Retinal Ganglion│    │ Primary Visual Cortex  │ │
//! │  │              │───▶│     Cells       │───▶│        (V1)            │ │
//! │  │ Rods/Cones   │    │ ON/OFF center   │    │ Retinotopic mapping    │ │
//! │  └──────────────┘    └─────────────────┘    └────────────────────────┘ │
//! │         │                    │                        │                │
//! │         ▼                    ▼                        ▼                │
//! │  Light → Neural       Contrast/Motion          Orientation/Spatial    │
//! │  Transduction         Detection                 Frequency             │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Retinotopic Mapping
//!
//! The visual field maps to V1 cortex with:
//! - Cortical magnification: fovea has more cortical area per degree
//! - Log-polar transformation: eccentricity maps logarithmically
//!
//! # Phosphene Model
//!
//! For BCI visual prosthetics, electrical stimulation of V1 produces phosphenes
//! (perceived flashes of light). The model includes:
//! - Phosphene size vs. stimulation current
//! - Retinotopic location from electrode position
//! - Temporal dynamics (onset, persistence, afterimages)

use serde::{Deserialize, Serialize};

use super::types::{Position2D, Position3D, ReceptiveField, ReceptorType, SpatialExtent};

// ============================================================================
// Photoreceptor Types
// ============================================================================

/// Cone type (wavelength sensitivity)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConeType {
    /// Short wavelength (blue), peak ~420nm
    S,
    /// Medium wavelength (green), peak ~530nm
    M,
    /// Long wavelength (red), peak ~560nm
    L,
}

impl ConeType {
    /// Get the peak sensitivity wavelength in nanometers
    #[inline]
    #[must_use]
    pub const fn peak_wavelength_nm(self) -> f32 {
        match self {
            Self::S => 420.0,
            Self::M => 530.0,
            Self::L => 560.0,
        }
    }

    /// Get the spectral sensitivity at a given wavelength (normalized 0-1)
    #[must_use]
    pub fn spectral_sensitivity(self, wavelength_nm: f32) -> f32 {
        let peak = self.peak_wavelength_nm();
        // Gaussian approximation of cone spectral sensitivity
        let sigma = match self {
            Self::S => 30.0,  // Blue cones are narrower
            Self::M => 40.0,
            Self::L => 40.0,
        };
        let diff = wavelength_nm - peak;
        libm::expf(-(diff * diff) / (2.0 * sigma * sigma))
    }

    /// Get the receptor type for this cone
    #[inline]
    #[must_use]
    pub const fn receptor_type(self) -> ReceptorType {
        match self {
            Self::S => ReceptorType::SCone,
            Self::M => ReceptorType::MCone,
            Self::L => ReceptorType::LCone,
        }
    }
}

// ============================================================================
// Photoreceptor Models
// ============================================================================

/// Rod photoreceptor model (scotopic vision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RodPhotoreceptor {
    /// Adaptation state (0 = dark adapted, 1 = fully bleached)
    pub adaptation: f32,
    /// Current response (normalized)
    pub response: f32,
    /// Position in retinal coordinates (degrees from fovea)
    pub eccentricity_deg: f32,
    /// Recovery time constant (ms)
    pub tau_recovery_ms: f32,
}

impl RodPhotoreceptor {
    /// Create a new rod at given eccentricity
    #[inline]
    #[must_use]
    pub fn new(eccentricity_deg: f32) -> Self {
        Self {
            adaptation: 0.0,
            response: 0.0,
            eccentricity_deg,
            tau_recovery_ms: 30000.0, // ~30 seconds for dark adaptation
        }
    }

    /// Compute response to luminance (scotopic cd/m²)
    ///
    /// Rods saturate at ~1-10 cd/m² and are most sensitive at ~10⁻⁶ cd/m²
    #[must_use]
    pub fn compute_response(&mut self, luminance: f32, dt_ms: f32) -> f32 {
        // Weber-Fechner log response with saturation
        let log_lum = libm::log10f(luminance.max(1e-7));

        // Sensitivity range: -6 to 1 log units
        let normalized = ((log_lum + 6.0) / 7.0).clamp(0.0, 1.0);

        // Apply adaptation (reduces sensitivity in bright light)
        let adapted_response = normalized * (1.0 - self.adaptation);

        // Update adaptation state
        if luminance > 0.1 {
            // Light adaptation (fast)
            self.adaptation += (1.0 - self.adaptation) * (dt_ms / 1000.0);
        } else {
            // Dark adaptation (slow)
            self.adaptation -= self.adaptation * (dt_ms / self.tau_recovery_ms);
        }

        self.response = adapted_response;
        self.response
    }

    /// Check if rod is functional (not in fovea)
    #[inline]
    #[must_use]
    pub fn is_functional(&self) -> bool {
        // No rods in foveola (central ~0.35°)
        self.eccentricity_deg > 0.35
    }
}

impl Default for RodPhotoreceptor {
    fn default() -> Self {
        Self::new(10.0) // Peripheral rod
    }
}

/// Cone photoreceptor model (photopic vision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConePhotoreceptor {
    /// Cone type (S, M, or L)
    pub cone_type: ConeType,
    /// Adaptation state (0 = dark adapted, 1 = fully adapted)
    pub adaptation: f32,
    /// Current response (normalized)
    pub response: f32,
    /// Position (degrees from fovea center)
    pub eccentricity_deg: f32,
}

impl ConePhotoreceptor {
    /// Create a new cone at given position
    #[inline]
    #[must_use]
    pub fn new(cone_type: ConeType, eccentricity_deg: f32) -> Self {
        Self {
            cone_type,
            adaptation: 0.5,
            response: 0.0,
            eccentricity_deg,
        }
    }

    /// Compute response to spectral light
    #[must_use]
    pub fn compute_response(&mut self, luminance: f32, wavelength_nm: f32, dt_ms: f32) -> f32 {
        // Spectral sensitivity
        let sensitivity = self.cone_type.spectral_sensitivity(wavelength_nm);

        // Log response (Weber-Fechner)
        let effective_lum = luminance * sensitivity;
        let log_lum = libm::log10f(effective_lum.max(1e-3));

        // Cones work from ~1 to 10^6 cd/m²
        let normalized = ((log_lum + 3.0) / 9.0).clamp(0.0, 1.0);

        // Fast adaptation for cones
        let target_adaptation = libm::sqrtf(normalized);
        self.adaptation += (target_adaptation - self.adaptation) * (dt_ms / 100.0);

        // Contrast-normalized response
        self.response = (normalized - self.adaptation * 0.5).clamp(0.0, 1.0);
        self.response
    }

    /// Get the density at this eccentricity (cones per mm²)
    #[must_use]
    pub fn density_at_eccentricity(&self) -> f32 {
        if self.eccentricity_deg < 1.0 {
            // Peak foveal density ~200,000/mm²
            200_000.0
        } else {
            // Falls off rapidly outside fovea
            200_000.0 / (1.0 + self.eccentricity_deg * self.eccentricity_deg)
        }
    }
}

impl Default for ConePhotoreceptor {
    fn default() -> Self {
        Self::new(ConeType::M, 0.0) // Foveal M-cone
    }
}

// ============================================================================
// Retinal Ganglion Cells
// ============================================================================

/// Ganglion cell type
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GanglionCellType {
    /// ON-center: excited by light in center, inhibited by surround
    OnCenter,
    /// OFF-center: inhibited by light in center, excited by surround
    OffCenter,
    /// Intrinsically photosensitive (ipRGC): melanopsin, circadian
    Intrinsic,
}

/// Retinal ganglion cell model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetinalGanglionCell {
    /// Cell type
    pub cell_type: GanglionCellType,
    /// Position in visual field (degrees)
    pub position: Position2D,
    /// Receptive field center radius (degrees)
    pub center_radius_deg: f32,
    /// Surround radius (degrees)
    pub surround_radius_deg: f32,
    /// Current firing rate (spikes/s)
    pub firing_rate: f32,
    /// Baseline firing rate
    pub baseline_rate: f32,
}

impl RetinalGanglionCell {
    /// Create a new ganglion cell
    #[inline]
    #[must_use]
    pub fn new(cell_type: GanglionCellType, position: Position2D, eccentricity_deg: f32) -> Self {
        // Receptive field size increases with eccentricity
        let center_radius = 0.05 + 0.02 * eccentricity_deg;

        Self {
            cell_type,
            position,
            center_radius_deg: center_radius,
            surround_radius_deg: center_radius * 3.0,
            firing_rate: 10.0,
            baseline_rate: 10.0,
        }
    }

    /// Compute response to center and surround luminance
    #[must_use]
    pub fn compute_response(&mut self, center_lum: f32, surround_lum: f32) -> f32 {
        let center_weight = 1.0;
        let surround_weight = 0.6;

        let response = match self.cell_type {
            GanglionCellType::OnCenter => {
                center_lum * center_weight - surround_lum * surround_weight
            }
            GanglionCellType::OffCenter => {
                surround_lum * surround_weight - center_lum * center_weight
            }
            GanglionCellType::Intrinsic => {
                // Slow, sustained response to blue light
                center_lum * 0.3
            }
        };

        // Convert to firing rate (baseline + modulation)
        self.firing_rate = (self.baseline_rate + response * 100.0).clamp(0.0, 200.0);
        self.firing_rate
    }
}

impl Default for RetinalGanglionCell {
    fn default() -> Self {
        Self::new(GanglionCellType::OnCenter, Position2D::new(0.0, 0.0), 5.0)
    }
}

// ============================================================================
// Retinotopic Mapping
// ============================================================================

/// Retinotopic mapping from visual field to V1 cortex
#[derive(Clone, Debug)]
pub struct RetinotopicMap {
    /// Cortical magnification factor at fovea (mm/degree)
    pub m0: f32,
    /// Eccentricity scaling factor
    pub e2: f32,
}

impl RetinotopicMap {
    /// Create a standard human retinotopic map
    #[inline]
    #[must_use]
    pub fn human() -> Self {
        Self {
            m0: 7.99,  // mm/degree at fovea
            e2: 0.75,  // degrees (half-magnitude eccentricity)
        }
    }

    /// Map visual field position to V1 cortical position
    ///
    /// Uses the log-polar transformation: w = k * ln(z + a)
    /// where z = x + iy in visual field, w = u + iv in cortex
    #[must_use]
    pub fn visual_to_cortical(&self, visual_pos: Position2D) -> Position2D {
        let x = visual_pos.x;
        let y = visual_pos.y;

        // Eccentricity in visual field
        let ecc = libm::sqrtf(x * x + y * y);

        // Polar angle
        let theta = libm::atan2f(y, x);

        // Cortical magnification factor at this eccentricity (unused but kept for documentation)
        let _m = self.m0 / (1.0 + ecc / self.e2);

        // Log-polar mapping
        let cortical_r = self.m0 * self.e2 * libm::logf(1.0 + ecc / self.e2);

        // Convert back to Cartesian (cortical coordinates in mm)
        Position2D {
            x: cortical_r * libm::cosf(theta),
            y: cortical_r * libm::sinf(theta),
        }
    }

    /// Map V1 cortical position back to visual field
    #[must_use]
    pub fn cortical_to_visual(&self, cortical_pos: Position2D) -> Position2D {
        let u = cortical_pos.x;
        let v = cortical_pos.y;

        // Cortical distance from foveal representation
        let cortical_r = libm::sqrtf(u * u + v * v);

        // Polar angle (preserved)
        let theta = libm::atan2f(v, u);

        // Inverse log mapping
        let ecc = self.e2 * (libm::expf(cortical_r / (self.m0 * self.e2)) - 1.0);

        Position2D {
            x: ecc * libm::cosf(theta),
            y: ecc * libm::sinf(theta),
        }
    }

    /// Get cortical magnification factor at given eccentricity (mm/degree)
    #[must_use]
    pub fn magnification(&self, eccentricity_deg: f32) -> f32 {
        self.m0 / (1.0 + eccentricity_deg / self.e2)
    }
}

impl Default for RetinotopicMap {
    fn default() -> Self {
        Self::human()
    }
}

// ============================================================================
// Phosphene Model (for V1 stimulation)
// ============================================================================

/// Phosphene appearance from cortical stimulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phosphene {
    /// Visual field position (degrees from fixation)
    pub position: Position2D,
    /// Apparent size (degrees)
    pub size_deg: f32,
    /// Brightness (0-1)
    pub brightness: f32,
    /// Color (hue, 0=white, wavelength in nm for colored)
    pub color_nm: Option<f32>,
    /// Onset time (ms since stimulation start)
    pub onset_ms: f32,
    /// Duration (ms)
    pub duration_ms: f32,
}

/// Model for predicting phosphenes from electrical stimulation
#[derive(Clone, Debug)]
pub struct PhospheneModel {
    /// Retinotopic map for location prediction
    pub retinotopic_map: RetinotopicMap,
    /// Current threshold for phosphene perception (μA)
    pub threshold_ua: f32,
    /// Size scaling factor (deg/mm² of cortical activation)
    pub size_factor: f32,
}

impl PhospheneModel {
    /// Create a new phosphene model
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            retinotopic_map: RetinotopicMap::human(),
            threshold_ua: 100.0,  // ~100μA typical threshold
            size_factor: 0.5,
        }
    }

    /// Predict phosphene from electrode stimulation
    ///
    /// # Arguments
    /// * `electrode_pos` - Position on V1 cortex (mm from foveal representation)
    /// * `current_ua` - Stimulation current in microamperes
    /// * `pulse_duration_ms` - Duration of stimulation pulse
    #[must_use]
    pub fn predict_phosphene(
        &self,
        electrode_pos: Position2D,
        current_ua: f32,
        pulse_duration_ms: f32,
    ) -> Option<Phosphene> {
        // Below threshold, no phosphene
        if current_ua < self.threshold_ua {
            return None;
        }

        // Map electrode position to visual field
        let visual_pos = self.retinotopic_map.cortical_to_visual(electrode_pos);

        // Size increases with current (suprathreshold)
        let suprathreshold = (current_ua / self.threshold_ua) - 1.0;
        let size = self.size_factor * (1.0 + libm::sqrtf(suprathreshold));

        // Brightness depends on current
        let brightness = (suprathreshold / 5.0).clamp(0.0, 1.0);

        // Onset delay depends on current
        let onset = 50.0 / (1.0 + suprathreshold);

        Some(Phosphene {
            position: visual_pos,
            size_deg: size,
            brightness,
            color_nm: None, // White phosphenes typical
            onset_ms: onset,
            duration_ms: pulse_duration_ms + 100.0, // Persists after stimulation
        })
    }

    /// Get the electrode density needed for given visual resolution
    #[must_use]
    pub fn electrodes_for_resolution(&self, target_resolution_deg: f32, fov_deg: f32) -> u32 {
        // Approximate number of electrodes for uniform coverage
        let area = fov_deg * fov_deg;
        let electrode_spacing = target_resolution_deg * 2.0; // Nyquist
        let n = (area / (electrode_spacing * electrode_spacing)) as u32;
        n.max(1)
    }
}

impl Default for PhospheneModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Visual Cortex Representation
// ============================================================================

/// Represents a region of visual cortex (V1)
#[derive(Clone, Debug)]
pub struct VisualCortexRegion {
    /// Center position in cortical coordinates (mm)
    pub center: Position2D,
    /// Receptive field in visual space
    pub receptive_field: ReceptiveField,
    /// Preferred orientation (radians, 0-π)
    pub orientation_preference: f32,
    /// Preferred spatial frequency (cycles/degree)
    pub spatial_freq_preference: f32,
    /// Ocular dominance (-1 = left eye, +1 = right eye)
    pub ocular_dominance: f32,
    /// Current activation level
    pub activation: f32,
}

impl VisualCortexRegion {
    /// Create a new V1 region
    #[must_use]
    pub fn new(center: Position2D, visual_field_center: Position3D) -> Self {
        Self {
            center,
            receptive_field: ReceptiveField::center_surround(visual_field_center, 0.5, 0.3),
            orientation_preference: 0.0,
            spatial_freq_preference: 3.0, // cycles/degree
            ocular_dominance: 0.0,
            activation: 0.0,
        }
    }

    /// Compute response to oriented grating stimulus
    #[must_use]
    pub fn orientation_response(&self, stimulus_orientation: f32, stimulus_sf: f32) -> f32 {
        // Orientation tuning (von Mises)
        let orientation_diff = stimulus_orientation - self.orientation_preference;
        let orientation_term = libm::expf(2.0 * libm::cosf(2.0 * orientation_diff));

        // Spatial frequency tuning (log-Gaussian)
        let sf_ratio = libm::logf(stimulus_sf / self.spatial_freq_preference);
        let sf_term = libm::expf(-sf_ratio * sf_ratio / 0.5);

        orientation_term * sf_term / libm::expf(2.0)
    }
}

// ============================================================================
// Unified Visual Receptor Interface
// ============================================================================

/// Unified visual receptor that can represent any visual cell type
#[derive(Clone, Debug)]
pub enum VisualReceptor {
    /// Rod photoreceptor
    Rod(RodPhotoreceptor),
    /// Cone photoreceptor
    Cone(ConePhotoreceptor),
    /// Retinal ganglion cell
    Ganglion(RetinalGanglionCell),
}

impl VisualReceptor {
    /// Get the receptor type
    #[must_use]
    pub fn receptor_type(&self) -> ReceptorType {
        match self {
            Self::Rod(_) => ReceptorType::RodPhotoreceptor,
            Self::Cone(c) => c.cone_type.receptor_type(),
            Self::Ganglion(g) => match g.cell_type {
                GanglionCellType::OnCenter => ReceptorType::GanglionOnCenter,
                GanglionCellType::OffCenter => ReceptorType::GanglionOffCenter,
                GanglionCellType::Intrinsic => ReceptorType::IpRGC,
            },
        }
    }

    /// Get the spatial extent for visual receptors
    #[inline]
    #[must_use]
    pub fn spatial_extent() -> SpatialExtent {
        SpatialExtent::retina()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_spectral_sensitivity() {
        let s_cone = ConeType::S;
        let m_cone = ConeType::M;
        let l_cone = ConeType::L;

        // Each cone should be most sensitive at its peak
        assert!(s_cone.spectral_sensitivity(420.0) > s_cone.spectral_sensitivity(550.0));
        assert!(m_cone.spectral_sensitivity(530.0) > m_cone.spectral_sensitivity(420.0));
        assert!(l_cone.spectral_sensitivity(560.0) > l_cone.spectral_sensitivity(420.0));
    }

    #[test]
    fn test_rod_response() {
        let mut rod = RodPhotoreceptor::new(10.0);

        // Dark adapted rod should respond to dim light
        let response = rod.compute_response(0.001, 10.0);
        assert!(response > 0.0);

        // Should saturate in bright light
        let bright_response = rod.compute_response(10.0, 10.0);
        assert!(bright_response < response || rod.adaptation > 0.5);
    }

    #[test]
    fn test_retinotopic_mapping() {
        let map = RetinotopicMap::human();

        // Fovea should map to near origin
        let fovea = map.visual_to_cortical(Position2D::new(0.0, 0.0));
        assert!(fovea.x.abs() < 0.1);
        assert!(fovea.y.abs() < 0.1);

        // Peripheral should map further
        let peripheral = map.visual_to_cortical(Position2D::new(10.0, 0.0));
        assert!(peripheral.x > fovea.x);

        // Round-trip should preserve approximate position
        let roundtrip = map.cortical_to_visual(peripheral);
        assert!((roundtrip.x - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_phosphene_model() {
        let model = PhospheneModel::new();

        // Below threshold, no phosphene
        let none = model.predict_phosphene(Position2D::new(5.0, 0.0), 50.0, 1.0);
        assert!(none.is_none());

        // Above threshold, phosphene generated
        let phosphene = model.predict_phosphene(Position2D::new(5.0, 0.0), 200.0, 1.0);
        assert!(phosphene.is_some());
        let p = phosphene.unwrap();
        assert!(p.brightness > 0.0);
        assert!(p.size_deg > 0.0);
    }

    #[test]
    fn test_ganglion_cell_response() {
        let mut on_cell = RetinalGanglionCell::new(
            GanglionCellType::OnCenter,
            Position2D::new(5.0, 0.0),
            5.0,
        );

        // ON-center should increase firing with center light
        let response = on_cell.compute_response(1.0, 0.0);
        assert!(response > on_cell.baseline_rate);

        // OFF-center should decrease
        let mut off_cell = RetinalGanglionCell::new(
            GanglionCellType::OffCenter,
            Position2D::new(5.0, 0.0),
            5.0,
        );
        let off_response = off_cell.compute_response(1.0, 0.0);
        assert!(off_response < off_cell.baseline_rate);
    }
}

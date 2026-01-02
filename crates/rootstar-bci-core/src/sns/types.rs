//! Core SNS types for sensory receptor simulation
//!
//! This module defines the fundamental types used across all sensory modalities:
//! - Sensory modality identification
//! - Receptor type classification
//! - Spatial representations (positions, receptive fields)
//! - Stimulus descriptors
//! - Population structures

use core::f32::consts::PI;

use serde::{Deserialize, Serialize};

// ============================================================================
// Sensory Modality and Receptor Types
// ============================================================================

/// Sensory modality classification
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    /// Tactile/somatosensory (touch, pressure, vibration)
    Tactile,
    /// Auditory (hearing)
    Auditory,
    /// Gustatory (taste)
    Gustatory,
    /// Visual (sight)
    Visual,
    /// Olfactory (smell)
    Olfactory,
    /// Proprioceptive (body position)
    Proprioceptive,
    /// Nociceptive (pain)
    Nociceptive,
    /// Thermoreceptive (temperature)
    Thermoreceptive,
}

impl SensoryModality {
    /// Get the primary cortical region for this modality
    #[inline]
    #[must_use]
    pub const fn primary_cortex(self) -> &'static str {
        match self {
            Self::Tactile => "S1 (Primary Somatosensory)",
            Self::Auditory => "A1 (Primary Auditory)",
            Self::Gustatory => "Insula/Frontal Operculum",
            Self::Visual => "V1 (Primary Visual Cortex)",
            Self::Olfactory => "Piriform Cortex/Olfactory Bulb",
            Self::Proprioceptive => "S1 (Area 3a)",
            Self::Nociceptive => "S1/Insula/ACC",
            Self::Thermoreceptive => "Insula",
        }
    }

    /// Get the typical neural pathway delay in milliseconds
    #[inline]
    #[must_use]
    pub const fn pathway_delay_ms(self) -> u32 {
        match self {
            Self::Tactile => 20,       // SEP N20 latency
            Self::Auditory => 10,      // ABR wave V
            Self::Gustatory => 150,    // GEP P1 latency
            Self::Visual => 50,        // VEP P100 / 2 (approx)
            Self::Olfactory => 100,    // Olfactory ERP latency
            Self::Proprioceptive => 25,
            Self::Nociceptive => 200,  // C-fiber delay
            Self::Thermoreceptive => 150,
        }
    }
}

/// Receptor type classification covering all modalities
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReceptorType {
    // === Tactile Mechanoreceptors ===
    /// Meissner corpuscle: rapid adaptation (RA-I), light touch, flutter
    Meissner,
    /// Merkel disc: slow adaptation (SA-I), pressure, edges
    Merkel,
    /// Pacinian corpuscle: very rapid adaptation (RA-II), vibration 40-400Hz
    Pacinian,
    /// Ruffini ending: slow adaptation (SA-II), skin stretch
    Ruffini,
    /// Free nerve ending: nociception, temperature
    FreeNerveEnding,

    // === Auditory Hair Cells ===
    /// Inner hair cell: primary transduction to auditory nerve
    InnerHairCell,
    /// Outer hair cell: cochlear amplification
    OuterHairCell,

    // === Gustatory Receptors ===
    /// Type II taste receptor cell (sweet, umami, bitter via GPCRs)
    TypeIITasteCell,
    /// Type III taste receptor cell (sour via OTOP1)
    TypeIIITasteCell,
    /// ENaC-expressing cell (salty via epithelial sodium channel)
    ENaCTasteCell,

    // === Papillae Types ===
    /// Fungiform papilla: scattered on anterior 2/3 of tongue
    FungiformPapilla,
    /// Foliate papilla: lateral edges of tongue
    FoliatePapilla,
    /// Circumvallate papilla: V-shaped row at posterior tongue
    CircumvallatePapilla,

    // === Visual Photoreceptors ===
    /// Rod photoreceptor: scotopic (dim light) vision, high sensitivity
    RodPhotoreceptor,
    /// S-cone (short wavelength): blue-sensitive, ~420nm peak
    SCone,
    /// M-cone (medium wavelength): green-sensitive, ~530nm peak
    MCone,
    /// L-cone (long wavelength): red-sensitive, ~560nm peak
    LCone,
    /// Retinal ganglion cell (ON-center): responds to light onset
    GanglionOnCenter,
    /// Retinal ganglion cell (OFF-center): responds to light offset
    GanglionOffCenter,
    /// Intrinsically photosensitive RGC: circadian/pupil reflex
    IpRGC,

    // === Olfactory Receptors ===
    /// Olfactory receptor neuron: ~400 types, one receptor type per neuron
    OlfactoryReceptorNeuron,
    /// Mitral cell: principal output neuron of olfactory bulb
    MitralCell,
    /// Tufted cell: secondary output neuron, faster responses
    TuftedCell,
    /// Granule cell: inhibitory interneuron in olfactory bulb
    OlfactoryGranuleCell,
}

impl ReceptorType {
    /// Get the sensory modality for this receptor type
    #[inline]
    #[must_use]
    pub const fn modality(self) -> SensoryModality {
        match self {
            Self::Meissner | Self::Merkel | Self::Pacinian | Self::Ruffini => SensoryModality::Tactile,
            Self::FreeNerveEnding => SensoryModality::Nociceptive,
            Self::InnerHairCell | Self::OuterHairCell => SensoryModality::Auditory,
            Self::TypeIITasteCell | Self::TypeIIITasteCell | Self::ENaCTasteCell |
            Self::FungiformPapilla | Self::FoliatePapilla | Self::CircumvallatePapilla => {
                SensoryModality::Gustatory
            }
            Self::RodPhotoreceptor | Self::SCone | Self::MCone | Self::LCone |
            Self::GanglionOnCenter | Self::GanglionOffCenter | Self::IpRGC => {
                SensoryModality::Visual
            }
            Self::OlfactoryReceptorNeuron | Self::MitralCell | Self::TuftedCell |
            Self::OlfactoryGranuleCell => {
                SensoryModality::Olfactory
            }
        }
    }

    /// Get the typical adaptation rate
    #[inline]
    #[must_use]
    pub const fn adaptation_rate(self) -> AdaptationRate {
        match self {
            Self::Meissner => AdaptationRate::Rapid,
            Self::Merkel => AdaptationRate::Slow,
            Self::Pacinian => AdaptationRate::VeryRapid,
            Self::Ruffini => AdaptationRate::Slow,
            Self::FreeNerveEnding => AdaptationRate::Variable,
            Self::InnerHairCell => AdaptationRate::Rapid,
            Self::OuterHairCell => AdaptationRate::VeryRapid,
            Self::TypeIITasteCell | Self::TypeIIITasteCell | Self::ENaCTasteCell => AdaptationRate::Slow,
            Self::FungiformPapilla | Self::FoliatePapilla | Self::CircumvallatePapilla => AdaptationRate::Slow,
            // Visual: rods are slow, cones are rapid, ganglion cells vary
            Self::RodPhotoreceptor => AdaptationRate::Slow,
            Self::SCone | Self::MCone | Self::LCone => AdaptationRate::Rapid,
            Self::GanglionOnCenter | Self::GanglionOffCenter => AdaptationRate::Rapid,
            Self::IpRGC => AdaptationRate::Slow, // Slow for circadian
            // Olfactory: ORNs adapt, mitral/tufted are rapid, granule slow
            Self::OlfactoryReceptorNeuron => AdaptationRate::Slow,
            Self::MitralCell | Self::TuftedCell => AdaptationRate::Rapid,
            Self::OlfactoryGranuleCell => AdaptationRate::Slow,
        }
    }

    /// Get the typical receptive field size in millimeters
    /// For visual: degrees of visual angle; for olfactory: conceptual spread
    #[inline]
    #[must_use]
    pub fn receptive_field_mm(self) -> f32 {
        match self {
            Self::Meissner => 4.0,    // 3-5mm
            Self::Merkel => 2.5,      // 2-3mm
            Self::Pacinian => 15.0,   // >10mm
            Self::Ruffini => 20.0,    // Large
            Self::FreeNerveEnding => 1.0,
            Self::InnerHairCell => 0.1,  // Frequency-specific
            Self::OuterHairCell => 0.1,
            Self::TypeIITasteCell | Self::TypeIIITasteCell | Self::ENaCTasteCell => 0.5,
            Self::FungiformPapilla => 1.0,
            Self::FoliatePapilla => 2.0,
            Self::CircumvallatePapilla => 3.0,
            // Visual: foveal cones are tiny, peripheral rods larger, ganglion cells larger still
            Self::SCone | Self::MCone | Self::LCone => 0.005, // ~5 microns foveal
            Self::RodPhotoreceptor => 0.002,  // ~2 microns
            Self::GanglionOnCenter | Self::GanglionOffCenter => 0.1, // ~0.1° center
            Self::IpRGC => 1.0,  // Large receptive fields
            // Olfactory: not spatial in traditional sense
            Self::OlfactoryReceptorNeuron => 0.01, // Single glomerulus
            Self::MitralCell => 0.05,
            Self::TuftedCell => 0.03,
            Self::OlfactoryGranuleCell => 0.1,
        }
    }
}

/// Adaptation rate classification
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationRate {
    /// Very rapid adaptation (RA-II): responds only to onset/offset
    VeryRapid,
    /// Rapid adaptation (RA-I): responds to changing stimuli
    Rapid,
    /// Slow adaptation (SA): sustained response during stimulus
    Slow,
    /// Variable adaptation depending on stimulus
    Variable,
}

impl AdaptationRate {
    /// Get the typical time constant in milliseconds
    #[inline]
    #[must_use]
    pub const fn tau_ms(self) -> f32 {
        match self {
            Self::VeryRapid => 5.0,
            Self::Rapid => 50.0,
            Self::Slow => 500.0,
            Self::Variable => 100.0,
        }
    }
}

// ============================================================================
// Body Region and Spatial Types
// ============================================================================

/// Body region for tactile mapping
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyRegion {
    /// Fingertip (highest receptor density)
    Fingertip(Finger),
    /// Palm
    Palm(Hand),
    /// Forearm
    Forearm(Side),
    /// Upper arm
    UpperArm(Side),
    /// Face/lips
    Face,
    /// Trunk/torso
    Trunk,
    /// Foot sole
    FootSole(Side),
}

/// Finger identification
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Finger {
    /// Thumb
    Thumb,
    /// Index finger
    Index,
    /// Middle finger
    Middle,
    /// Ring finger
    Ring,
    /// Little finger
    Little,
}

/// Hand side
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Hand {
    /// Left hand
    Left,
    /// Right hand
    Right,
}

/// Body side
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Left side
    Left,
    /// Right side
    Right,
}

/// Ear identification
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ear {
    /// Left ear
    Left,
    /// Right ear
    Right,
}

impl BodyRegion {
    /// Get the receptor density per square centimeter
    #[inline]
    #[must_use]
    pub const fn receptor_density(self) -> u16 {
        match self {
            Self::Fingertip(_) => 240,
            Self::Palm(_) => 60,
            Self::Face => 100,
            Self::Forearm(_) => 15,
            Self::UpperArm(_) => 10,
            Self::Trunk => 5,
            Self::FootSole(_) => 40,
        }
    }

    /// Get the two-point discrimination threshold in millimeters
    #[inline]
    #[must_use]
    pub const fn two_point_threshold_mm(self) -> u8 {
        match self {
            Self::Fingertip(_) => 2,
            Self::Palm(_) => 10,
            Self::Face => 5,
            Self::Forearm(_) => 40,
            Self::UpperArm(_) => 45,
            Self::Trunk => 40,
            Self::FootSole(_) => 20,
        }
    }

    /// Get the typical dimensions in millimeters (width, height)
    #[inline]
    #[must_use]
    pub const fn dimensions_mm(self) -> (u16, u16) {
        match self {
            Self::Fingertip(_) => (15, 20),
            Self::Palm(_) => (80, 100),
            Self::Face => (150, 200),
            Self::Forearm(_) => (60, 250),
            Self::UpperArm(_) => (80, 300),
            Self::Trunk => (300, 500),
            Self::FootSole(_) => (80, 200),
        }
    }

    /// Get the contralateral cortical electrode (C3 for right, C4 for left)
    #[inline]
    #[must_use]
    pub const fn cortical_electrode(self) -> &'static str {
        match self {
            Self::Fingertip(finger) => match finger {
                Finger::Thumb | Finger::Index | Finger::Middle | Finger::Ring | Finger::Little => "C3/C4",
            },
            Self::Palm(hand) => match hand {
                Hand::Left => "C4",
                Hand::Right => "C3",
            },
            Self::Forearm(side) | Self::UpperArm(side) => match side {
                Side::Left => "C4",
                Side::Right => "C3",
            },
            Self::Face => "C3/C4",
            Self::Trunk => "Cz",
            Self::FootSole(_) => "Cz",
        }
    }
}

/// 2D position (used for UV coordinates on meshes)
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Position2D {
    /// X coordinate (normalized 0-1 or in mm)
    pub x: f32,
    /// Y coordinate (normalized 0-1 or in mm)
    pub y: f32,
}

impl Position2D {
    /// Create a new 2D position
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Create from UV coordinates (0-1 range)
    #[inline]
    #[must_use]
    pub const fn from_uv(u: f32, v: f32) -> Self {
        Self { x: u, y: v }
    }

    /// Euclidean distance to another point
    #[inline]
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        libm::sqrtf(dx * dx + dy * dy)
    }
}

/// 3D position in tissue model
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    /// X coordinate in millimeters
    pub x: f32,
    /// Y coordinate in millimeters
    pub y: f32,
    /// Z coordinate in millimeters (depth)
    pub z: f32,
}

impl Position3D {
    /// Create a new 3D position
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point
    #[inline]
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        libm::sqrtf(dx * dx + dy * dy + dz * dz)
    }

    /// Convert to 2D by projecting onto XY plane
    #[inline]
    #[must_use]
    pub const fn to_2d(self) -> Position2D {
        Position2D { x: self.x, y: self.y }
    }
}

// ============================================================================
// Receptive Field
// ============================================================================

/// Receptive field definition with center-surround organization
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceptiveField {
    /// Center position
    pub center: Position3D,
    /// Gaussian width (sigma) in millimeters
    pub sigma: f32,
    /// Surround inhibition strength (0 = no surround, 1 = full surround)
    pub surround_inhibition: f32,
    /// Surround sigma relative to center (typically 2-3x center sigma)
    pub surround_sigma_ratio: f32,
}

impl ReceptiveField {
    /// Create a simple Gaussian receptive field
    #[inline]
    #[must_use]
    pub fn gaussian(center: Position3D, sigma: f32) -> Self {
        Self {
            center,
            sigma,
            surround_inhibition: 0.0,
            surround_sigma_ratio: 2.0,
        }
    }

    /// Create a center-surround receptive field (Mexican hat)
    #[inline]
    #[must_use]
    pub fn center_surround(center: Position3D, center_sigma: f32, surround_strength: f32) -> Self {
        Self {
            center,
            sigma: center_sigma,
            surround_inhibition: surround_strength,
            surround_sigma_ratio: 2.5,
        }
    }

    /// Compute response weight for a stimulus at given position
    #[inline]
    #[must_use]
    pub fn response_weight(&self, position: Position3D) -> f32 {
        let dist = self.center.distance(position);

        // Center (excitatory) Gaussian
        let center_response = libm::expf(-dist * dist / (2.0 * self.sigma * self.sigma));

        if self.surround_inhibition > 0.0 {
            // Surround (inhibitory) Gaussian
            let surround_sigma = self.sigma * self.surround_sigma_ratio;
            let surround_response = libm::expf(-dist * dist / (2.0 * surround_sigma * surround_sigma));

            // Mexican hat: center - surround
            (center_response - self.surround_inhibition * surround_response).max(0.0)
        } else {
            center_response
        }
    }
}

impl Default for ReceptiveField {
    fn default() -> Self {
        Self::gaussian(Position3D::default(), 2.0)
    }
}

// ============================================================================
// Stimulus Types
// ============================================================================

/// Type of stimulus being applied
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StimulusType {
    /// Mechanical stimulus (pressure, force in mN)
    Mechanical {
        /// Force in millinewtons
        force_mn: f32,
        /// Indentation depth in micrometers
        indentation_um: f32,
    },
    /// Vibratory stimulus
    Vibration {
        /// Amplitude in micrometers
        amplitude_um: f32,
        /// Frequency in Hz
        frequency_hz: f32,
        /// Phase in radians
        phase_rad: f32,
    },
    /// Acoustic stimulus
    Acoustic {
        /// Sound pressure level in dB SPL
        level_db_spl: f32,
        /// Frequency in Hz
        frequency_hz: f32,
        /// Phase in radians
        phase_rad: f32,
    },
    /// Chemical/taste stimulus
    Chemical {
        /// Concentration (arbitrary units, 0-1 normalized)
        concentration: f32,
        /// Taste quality
        quality: TasteQualitySimple,
    },
    /// Thermal stimulus
    Thermal {
        /// Temperature in Celsius
        temperature_c: f32,
        /// Rate of change in °C/s
        rate_c_per_s: f32,
    },
    /// Visual stimulus (light)
    Visual {
        /// Luminance in candelas per square meter
        luminance_cd_m2: f32,
        /// Wavelength in nanometers (for spectral, 0 for broadband)
        wavelength_nm: f32,
        /// Contrast (Michelson contrast, 0-1)
        contrast: f32,
        /// Spatial frequency in cycles per degree (for gratings)
        spatial_freq_cpd: f32,
    },
    /// Olfactory stimulus (odor)
    Olfactory {
        /// Concentration (arbitrary units, 0-1 normalized)
        concentration: f32,
        /// Odorant class/identity
        odorant: OdorantClass,
    },
}

/// Simplified odorant classification for stimulus
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OdorantClass {
    /// Floral scents (rose, jasmine)
    Floral,
    /// Fruity scents (citrus, berry)
    Fruity,
    /// Woody scents (cedar, pine)
    Woody,
    /// Minty scents (menthol, peppermint)
    Minty,
    /// Sweet scents (vanilla, caramel)
    Sweet,
    /// Pungent scents (ammonia, vinegar)
    Pungent,
    /// Decayed/putrid scents
    Putrid,
    /// Musky scents
    Musky,
}

/// Simplified taste quality for stimulus
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TasteQualitySimple {
    /// Sweet (sugars, artificial sweeteners)
    Sweet,
    /// Salty (NaCl)
    Salty,
    /// Sour (acids)
    Sour,
    /// Bitter (alkaloids)
    Bitter,
    /// Umami (glutamate)
    Umami,
}

/// Descriptor for a stimulus applied to a receptor population
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StimulusDescriptor {
    /// Stimulus type and parameters
    pub stimulus_type: StimulusType,
    /// Location of stimulus application
    pub location: Position3D,
    /// Spatial extent (radius) in millimeters
    pub radius_mm: f32,
    /// Onset time in milliseconds
    pub onset_ms: f64,
    /// Duration in milliseconds (None = continuous)
    pub duration_ms: Option<f64>,
}

impl StimulusDescriptor {
    /// Create a point stimulus
    #[inline]
    #[must_use]
    pub fn point(stimulus_type: StimulusType, location: Position3D, onset_ms: f64) -> Self {
        Self {
            stimulus_type,
            location,
            radius_mm: 1.0,
            onset_ms,
            duration_ms: None,
        }
    }

    /// Create a brief pulse stimulus
    #[inline]
    #[must_use]
    pub fn pulse(stimulus_type: StimulusType, location: Position3D, onset_ms: f64, duration_ms: f64) -> Self {
        Self {
            stimulus_type,
            location,
            radius_mm: 1.0,
            onset_ms,
            duration_ms: Some(duration_ms),
        }
    }

    /// Check if stimulus is active at given time
    #[inline]
    #[must_use]
    pub fn is_active(&self, time_ms: f64) -> bool {
        if time_ms < self.onset_ms {
            return false;
        }
        match self.duration_ms {
            Some(duration) => time_ms < self.onset_ms + duration,
            None => true,
        }
    }
}

// ============================================================================
// Stimulus Field
// ============================================================================

/// Spatial field of stimulus values
#[derive(Clone, Debug)]
pub struct StimulusField {
    /// Active stimuli in the field
    stimuli: heapless::Vec<StimulusDescriptor, 16>,
    /// Current simulation time in milliseconds
    current_time_ms: f64,
}

impl StimulusField {
    /// Create an empty stimulus field
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stimuli: heapless::Vec::new(),
            current_time_ms: 0.0,
        }
    }

    /// Add a stimulus to the field
    #[inline]
    pub fn add_stimulus(&mut self, stimulus: StimulusDescriptor) -> Result<(), StimulusDescriptor> {
        self.stimuli.push(stimulus)
    }

    /// Remove expired stimuli
    pub fn cleanup(&mut self) {
        self.stimuli.retain(|s| s.is_active(self.current_time_ms) || s.onset_ms > self.current_time_ms);
    }

    /// Update the current time
    #[inline]
    pub fn set_time(&mut self, time_ms: f64) {
        self.current_time_ms = time_ms;
    }

    /// Sample the stimulus field at a given position with receptive field weighting
    pub fn sample_at(&self, position: Position3D, receptive_field: &ReceptiveField) -> f32 {
        let mut total_stimulus = 0.0f32;

        for stimulus in &self.stimuli {
            if !stimulus.is_active(self.current_time_ms) {
                continue;
            }

            // Distance from stimulus center to sampling position
            let dist = position.distance(stimulus.location);

            // Stimulus spatial falloff (Gaussian)
            let stim_weight = if dist <= stimulus.radius_mm {
                1.0
            } else {
                let d = dist - stimulus.radius_mm;
                let r = stimulus.radius_mm;
                libm::expf(-(d * d) / (2.0 * r * r))
            };

            // Receptive field weighting
            let rf_weight = receptive_field.response_weight(stimulus.location);

            // Extract intensity based on stimulus type
            let intensity = match &stimulus.stimulus_type {
                StimulusType::Mechanical { force_mn, .. } => *force_mn / 100.0, // Normalize
                StimulusType::Vibration { amplitude_um, frequency_hz, phase_rad } => {
                    let t = self.current_time_ms / 1000.0;
                    *amplitude_um * libm::sinf(2.0 * PI * frequency_hz * t as f32 + phase_rad)
                }
                StimulusType::Acoustic { level_db_spl, .. } => {
                    // Convert dB SPL to linear amplitude (normalized)
                    libm::powf(10.0, (*level_db_spl - 60.0) / 20.0)
                }
                StimulusType::Chemical { concentration, .. } => *concentration,
                StimulusType::Thermal { temperature_c, .. } => {
                    // Normalize around 32°C (skin temperature)
                    (*temperature_c - 32.0) / 10.0
                }
                StimulusType::Visual { luminance_cd_m2, contrast, .. } => {
                    // Log-scale luminance (Weber-Fechner) with contrast modulation
                    let log_lum = libm::log10f(luminance_cd_m2.max(0.001) / 100.0);
                    (log_lum + 2.0) * contrast // Normalized around 100 cd/m²
                }
                StimulusType::Olfactory { concentration, .. } => {
                    // Olfactory response follows Hill equation (sigmoidal)
                    *concentration
                }
            };

            total_stimulus += intensity * stim_weight * rf_weight;
        }

        total_stimulus
    }
}

impl Default for StimulusField {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Spatial Extent
// ============================================================================

/// Spatial extent of a receptor population
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialExtent {
    /// Width in millimeters (for skin) or Hz (for cochlea)
    pub width: f32,
    /// Height in millimeters (for skin) or amplitude range (for cochlea)
    pub height: f32,
    /// Depth in millimeters (tissue depth)
    pub depth: f32,
}

impl SpatialExtent {
    /// Create a new spatial extent
    #[inline]
    #[must_use]
    pub const fn new(width: f32, height: f32, depth: f32) -> Self {
        Self { width, height, depth }
    }

    /// Create extent for a skin patch
    #[inline]
    #[must_use]
    pub const fn skin_patch(width_mm: f32, height_mm: f32) -> Self {
        Self {
            width: width_mm,
            height: height_mm,
            depth: 4.0, // Typical dermis depth
        }
    }

    /// Create extent for the cochlea (frequency range)
    #[inline]
    #[must_use]
    pub const fn cochlea() -> Self {
        Self {
            width: 35.0,      // Length in mm
            height: 20000.0,  // Frequency range in Hz
            depth: 0.5,       // Organ of Corti depth
        }
    }

    /// Create extent for the tongue
    #[inline]
    #[must_use]
    pub const fn tongue() -> Self {
        Self {
            width: 50.0,   // Width in mm
            height: 100.0, // Length in mm
            depth: 3.0,    // Papillae depth
        }
    }

    /// Create extent for the retina (visual field in degrees)
    #[inline]
    #[must_use]
    pub const fn retina() -> Self {
        Self {
            width: 180.0,  // Horizontal field of view in degrees
            height: 120.0, // Vertical field of view in degrees
            depth: 0.3,    // Retinal thickness in mm
        }
    }

    /// Create extent for the fovea (high-acuity central vision)
    #[inline]
    #[must_use]
    pub const fn fovea() -> Self {
        Self {
            width: 5.0,    // Central 5 degrees
            height: 5.0,   // Central 5 degrees
            depth: 0.2,    // Foveal pit depth
        }
    }

    /// Create extent for the olfactory epithelium
    #[inline]
    #[must_use]
    pub const fn olfactory_epithelium() -> Self {
        Self {
            width: 10.0,   // Width in mm
            height: 50.0,  // Total area ~5 cm² per side
            depth: 0.1,    // Epithelial depth
        }
    }
}

impl Default for SpatialExtent {
    fn default() -> Self {
        Self::skin_patch(20.0, 20.0)
    }
}

// ============================================================================
// Spatial Receptor
// ============================================================================

/// A receptor with spatial position and receptive field
#[derive(Clone, Debug)]
pub struct SpatialReceptor<R> {
    /// The receptor model
    pub model: R,
    /// Position in tissue
    pub position: Position3D,
    /// Receptive field
    pub receptive_field: ReceptiveField,
    /// Receptor type
    pub receptor_type: ReceptorType,
}

impl<R> SpatialReceptor<R> {
    /// Create a new spatial receptor
    #[inline]
    #[must_use]
    pub fn new(model: R, position: Position3D, receptor_type: ReceptorType) -> Self {
        let sigma = receptor_type.receptive_field_mm();
        Self {
            model,
            position,
            receptive_field: ReceptiveField::gaussian(position, sigma),
            receptor_type,
        }
    }

    /// Create with custom receptive field
    #[inline]
    #[must_use]
    pub fn with_receptive_field(model: R, position: Position3D, receptor_type: ReceptorType, receptive_field: ReceptiveField) -> Self {
        Self {
            model,
            position,
            receptive_field,
            receptor_type,
        }
    }
}

// ============================================================================
// Receptor Population
// ============================================================================

/// A population of spatially distributed receptors
#[derive(Clone, Debug)]
pub struct ReceptorPopulation<R, const N: usize> {
    /// Array of receptors
    pub receptors: heapless::Vec<SpatialReceptor<R>, N>,
    /// Spatial extent of the population
    pub spatial_extent: SpatialExtent,
    /// Sensory modality
    pub modality: SensoryModality,
}

impl<R, const N: usize> ReceptorPopulation<R, N> {
    /// Create an empty population
    #[inline]
    #[must_use]
    pub fn new(spatial_extent: SpatialExtent, modality: SensoryModality) -> Self {
        Self {
            receptors: heapless::Vec::new(),
            spatial_extent,
            modality,
        }
    }

    /// Add a receptor to the population
    #[inline]
    pub fn add(&mut self, receptor: SpatialReceptor<R>) -> Result<(), SpatialReceptor<R>> {
        self.receptors.push(receptor)
    }

    /// Get the number of receptors
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.receptors.len()
    }

    /// Check if population is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.receptors.is_empty()
    }
}

// ============================================================================
// Timestamped Activation
// ============================================================================

/// Receptor activation with timestamp
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimestampedActivation {
    /// Timestamp in milliseconds
    pub timestamp_ms: f64,
    /// Activation level (firing rate or normalized)
    pub activation: f32,
    /// Receptor index
    pub receptor_index: u16,
}

impl TimestampedActivation {
    /// Create a new timestamped activation
    #[inline]
    #[must_use]
    pub const fn new(timestamp_ms: f64, activation: f32, receptor_index: u16) -> Self {
        Self { timestamp_ms, activation, receptor_index }
    }
}

// ============================================================================
// Population State
// ============================================================================

/// State of a tactile receptor population
#[derive(Clone, Debug)]
pub struct TactilePopulationState<const N: usize> {
    /// Firing rates for each receptor
    pub firing_rates: heapless::Vec<f32, N>,
    /// Timestamp in milliseconds
    pub timestamp_ms: f64,
    /// Body region
    pub region: BodyRegion,
}

impl<const N: usize> TactilePopulationState<N> {
    /// Create a new population state
    #[inline]
    #[must_use]
    pub fn new(region: BodyRegion, timestamp_ms: f64) -> Self {
        Self {
            firing_rates: heapless::Vec::new(),
            timestamp_ms,
            region,
        }
    }

    /// Get the mean firing rate
    #[inline]
    #[must_use]
    pub fn mean_rate(&self) -> f32 {
        if self.firing_rates.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.firing_rates.iter().sum();
        sum / self.firing_rates.len() as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance(p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_receptive_field_gaussian() {
        let rf = ReceptiveField::gaussian(Position3D::new(0.0, 0.0, 0.0), 2.0);

        // At center, weight should be 1.0
        let w_center = rf.response_weight(Position3D::new(0.0, 0.0, 0.0));
        assert!((w_center - 1.0).abs() < 0.001);

        // At 2*sigma, weight should be ~0.135
        let w_2sigma = rf.response_weight(Position3D::new(4.0, 0.0, 0.0));
        assert!((w_2sigma - 0.135).abs() < 0.02);
    }

    #[test]
    fn test_receptor_type_modality() {
        assert_eq!(ReceptorType::Meissner.modality(), SensoryModality::Tactile);
        assert_eq!(ReceptorType::InnerHairCell.modality(), SensoryModality::Auditory);
        assert_eq!(ReceptorType::TypeIITasteCell.modality(), SensoryModality::Gustatory);
    }

    #[test]
    fn test_body_region_density() {
        assert!(BodyRegion::Fingertip(Finger::Index).receptor_density() > BodyRegion::Trunk.receptor_density());
    }

    #[test]
    fn test_stimulus_field_sampling() {
        let mut field = StimulusField::new();
        field.set_time(0.0);

        let stimulus = StimulusDescriptor::point(
            StimulusType::Mechanical { force_mn: 100.0, indentation_um: 500.0 },
            Position3D::new(0.0, 0.0, 0.0),
            0.0,
        );
        field.add_stimulus(stimulus).ok();

        let rf = ReceptiveField::gaussian(Position3D::new(0.0, 0.0, 0.0), 2.0);
        let value = field.sample_at(Position3D::new(0.0, 0.0, 0.0), &rf);

        assert!(value > 0.0);
    }
}

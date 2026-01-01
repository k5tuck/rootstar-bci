//! Olfactory sensory receptor simulation
//!
//! This module models the olfactory pathway from receptor neurons to the olfactory bulb:
//!
//! # Olfactory Pathway
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Olfactory Processing Pipeline                     │
//! │                                                                         │
//! │  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────┐ │
//! │  │  Olfactory   │    │   Glomerulus    │    │    Piriform Cortex     │ │
//! │  │  Receptor    │───▶│ (Olfactory Bulb)│───▶│                        │ │
//! │  │  Neurons     │    │ Mitral/Tufted   │    │  Pattern Recognition   │ │
//! │  └──────────────┘    └─────────────────┘    └────────────────────────┘ │
//! │         │                    │                        │                │
//! │         ▼                    ▼                        ▼                │
//! │  Odorant Binding      Lateral Inhibition        Odor Identity         │
//! │  (~400 OR types)      Contrast Enhancement                            │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Combinatorial Coding**: Each odorant activates multiple receptor types
//! - **Glomerular Convergence**: ~1000 ORNs → 1 glomerulus (same receptor type)
//! - **Lateral Inhibition**: Granule cells sharpen odor representations
//! - **Adaptation**: Rapid adaptation to sustained odors

use serde::{Deserialize, Serialize};

use super::types::{OdorantClass, Position2D, ReceptorType, SpatialExtent};

// ============================================================================
// Odorant Representation
// ============================================================================

/// Odorant descriptor for receptor activation patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Odorant {
    /// Odorant class
    pub class: OdorantClass,
    /// Molecular weight (Daltons, affects volatility)
    pub molecular_weight: f32,
    /// Hydrophobicity (log P, affects receptor binding)
    pub log_p: f32,
    /// Vapor pressure (affects concentration in air)
    pub vapor_pressure: f32,
    /// Activation pattern across receptor types (simplified)
    pub receptor_affinities: OdorantAffinities,
}

/// Simplified receptor affinity profile for an odorant
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OdorantAffinities {
    /// Affinity for fruity/ester receptors (0-1)
    pub fruity: f32,
    /// Affinity for floral/terpene receptors
    pub floral: f32,
    /// Affinity for woody/sesquiterpene receptors
    pub woody: f32,
    /// Affinity for minty/menthol receptors
    pub minty: f32,
    /// Affinity for sweet/vanilla receptors
    pub sweet: f32,
    /// Affinity for pungent/trigeminal receptors
    pub pungent: f32,
    /// Affinity for musky/macrocyclic receptors
    pub musky: f32,
    /// Affinity for sulfurous/thiol receptors
    pub sulfurous: f32,
}

impl OdorantAffinities {
    /// Create affinities from odorant class
    #[must_use]
    pub fn from_class(class: OdorantClass) -> Self {
        match class {
            OdorantClass::Fruity => Self { fruity: 0.9, sweet: 0.3, floral: 0.2, ..Default::default() },
            OdorantClass::Floral => Self { floral: 0.9, sweet: 0.2, fruity: 0.2, ..Default::default() },
            OdorantClass::Woody => Self { woody: 0.9, musky: 0.3, ..Default::default() },
            OdorantClass::Minty => Self { minty: 0.9, pungent: 0.2, ..Default::default() },
            OdorantClass::Sweet => Self { sweet: 0.9, fruity: 0.3, floral: 0.2, ..Default::default() },
            OdorantClass::Pungent => Self { pungent: 0.9, sulfurous: 0.3, ..Default::default() },
            OdorantClass::Putrid => Self { sulfurous: 0.9, pungent: 0.5, ..Default::default() },
            OdorantClass::Musky => Self { musky: 0.9, woody: 0.3, sweet: 0.2, ..Default::default() },
        }
    }

    /// Get affinity vector for distance calculations
    #[must_use]
    pub fn as_vector(&self) -> [f32; 8] {
        [
            self.fruity, self.floral, self.woody, self.minty,
            self.sweet, self.pungent, self.musky, self.sulfurous,
        ]
    }

    /// Compute similarity to another affinity profile
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        let a = self.as_vector();
        let b = other.as_vector();

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = libm::sqrtf(a.iter().map(|x| x * x).sum::<f32>());
        let mag_b: f32 = libm::sqrtf(b.iter().map(|x| x * x).sum::<f32>());

        if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }
}

impl Odorant {
    /// Create a typical odorant from class
    #[must_use]
    pub fn from_class(class: OdorantClass) -> Self {
        let (mw, log_p, vp) = match class {
            OdorantClass::Fruity => (130.0, 2.0, 5.0),   // Ethyl butyrate
            OdorantClass::Floral => (154.0, 3.5, 0.5),   // Linalool
            OdorantClass::Woody => (204.0, 4.5, 0.01),   // Cedrene
            OdorantClass::Minty => (156.0, 3.0, 0.5),    // Menthol
            OdorantClass::Sweet => (152.0, 1.5, 0.1),    // Vanillin
            OdorantClass::Pungent => (60.0, 0.5, 100.0), // Acetic acid
            OdorantClass::Putrid => (62.0, 0.8, 50.0),   // Hydrogen sulfide
            OdorantClass::Musky => (256.0, 5.0, 0.001),  // Muscone
        };

        Self {
            class,
            molecular_weight: mw,
            log_p,
            vapor_pressure: vp,
            receptor_affinities: OdorantAffinities::from_class(class),
        }
    }
}

// ============================================================================
// Olfactory Receptor Neuron
// ============================================================================

/// Olfactory receptor type (one of ~400 in humans)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OlfactoryReceptorId(pub u16);

impl OlfactoryReceptorId {
    /// Get the receptor class (0-7 for simplified model)
    #[inline]
    #[must_use]
    pub fn class_index(self) -> usize {
        (self.0 % 8) as usize
    }
}

/// Olfactory receptor neuron (ORN)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OlfactoryReceptorNeuron {
    /// Receptor type expressed by this neuron
    pub receptor_id: OlfactoryReceptorId,
    /// Position in olfactory epithelium
    pub position: Position2D,
    /// Tuning profile (which odorant classes this receptor responds to)
    pub tuning: OdorantAffinities,
    /// Current adaptation state (0 = fresh, 1 = fully adapted)
    pub adaptation: f32,
    /// Current firing rate (Hz)
    pub firing_rate: f32,
    /// Maximum firing rate (Hz)
    pub max_rate: f32,
    /// Adaptation time constant (ms)
    pub tau_adaptation_ms: f32,
}

impl OlfactoryReceptorNeuron {
    /// Create a new ORN with given receptor type
    #[must_use]
    pub fn new(receptor_id: OlfactoryReceptorId, position: Position2D) -> Self {
        // Create tuning based on receptor class
        let class_idx = receptor_id.class_index();
        let mut tuning = OdorantAffinities::default();

        // Set primary tuning based on receptor class
        match class_idx {
            0 => tuning.fruity = 1.0,
            1 => tuning.floral = 1.0,
            2 => tuning.woody = 1.0,
            3 => tuning.minty = 1.0,
            4 => tuning.sweet = 1.0,
            5 => tuning.pungent = 1.0,
            6 => tuning.musky = 1.0,
            7 => tuning.sulfurous = 1.0,
            _ => {}
        }

        Self {
            receptor_id,
            position,
            tuning,
            adaptation: 0.0,
            firing_rate: 0.0,
            max_rate: 100.0,
            tau_adaptation_ms: 500.0,
        }
    }

    /// Compute response to an odorant at given concentration
    #[must_use]
    pub fn compute_response(&mut self, odorant: &Odorant, concentration: f32, dt_ms: f32) -> f32 {
        // Compute binding affinity
        let affinity = self.tuning.similarity(&odorant.receptor_affinities);

        // Hill equation for dose-response
        let k_d = 0.1; // Half-maximal concentration
        let n = 1.5;   // Hill coefficient
        let occupancy = libm::powf(concentration, n) / (libm::powf(k_d, n) + libm::powf(concentration, n));

        // Effective activation
        let activation = affinity * occupancy;

        // Apply adaptation
        let adapted_activation = activation * (1.0 - self.adaptation);

        // Update adaptation state
        if activation > 0.01 {
            self.adaptation += (activation - self.adaptation) * (dt_ms / self.tau_adaptation_ms);
        } else {
            // Recovery when no stimulus
            self.adaptation -= self.adaptation * (dt_ms / (self.tau_adaptation_ms * 2.0));
        }
        self.adaptation = self.adaptation.clamp(0.0, 0.95);

        // Convert to firing rate
        self.firing_rate = self.max_rate * adapted_activation;
        self.firing_rate
    }

    /// Reset adaptation state
    pub fn reset_adaptation(&mut self) {
        self.adaptation = 0.0;
    }
}

impl Default for OlfactoryReceptorNeuron {
    fn default() -> Self {
        Self::new(OlfactoryReceptorId(0), Position2D::new(0.0, 0.0))
    }
}

// ============================================================================
// Olfactory Bulb Cells
// ============================================================================

/// Glomerulus - convergence point for ORNs of same type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Glomerulus {
    /// Receptor type that converges here
    pub receptor_id: OlfactoryReceptorId,
    /// Position in olfactory bulb
    pub position: Position2D,
    /// Number of converging ORNs
    pub n_inputs: u16,
    /// Aggregate input activity
    pub input_activity: f32,
}

impl Glomerulus {
    /// Create a new glomerulus
    #[inline]
    #[must_use]
    pub fn new(receptor_id: OlfactoryReceptorId, position: Position2D) -> Self {
        Self {
            receptor_id,
            position,
            n_inputs: 1000, // ~1000 ORNs per glomerulus
            input_activity: 0.0,
        }
    }

    /// Update with aggregated ORN input
    pub fn update(&mut self, total_orn_rate: f32) {
        self.input_activity = total_orn_rate / self.n_inputs as f32;
    }
}

/// Mitral cell - principal output neuron of olfactory bulb
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MitralCell {
    /// Associated glomerulus
    pub glomerulus_id: u16,
    /// Current firing rate (Hz)
    pub firing_rate: f32,
    /// Baseline firing rate
    pub baseline_rate: f32,
    /// Lateral inhibition received
    pub inhibition: f32,
}

impl MitralCell {
    /// Create a new mitral cell
    #[inline]
    #[must_use]
    pub fn new(glomerulus_id: u16) -> Self {
        Self {
            glomerulus_id,
            firing_rate: 0.0,
            baseline_rate: 5.0,
            inhibition: 0.0,
        }
    }

    /// Compute firing rate from glomerular input
    #[must_use]
    pub fn compute_rate(&mut self, glomerular_input: f32, lateral_inhibition: f32) -> f32 {
        self.inhibition = lateral_inhibition;

        // Excitation minus inhibition with threshold
        let net_input = glomerular_input - lateral_inhibition * 0.5;
        self.firing_rate = (self.baseline_rate + net_input * 50.0).clamp(0.0, 150.0);
        self.firing_rate
    }
}

impl Default for MitralCell {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Tufted cell - secondary output neuron (faster, less lateral inhibition)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuftedCell {
    /// Associated glomerulus
    pub glomerulus_id: u16,
    /// Current firing rate (Hz)
    pub firing_rate: f32,
    /// Response latency (ms)
    pub latency_ms: f32,
}

impl TuftedCell {
    /// Create a new tufted cell
    #[inline]
    #[must_use]
    pub fn new(glomerulus_id: u16) -> Self {
        Self {
            glomerulus_id,
            firing_rate: 0.0,
            latency_ms: 10.0, // Faster than mitral cells
        }
    }

    /// Compute firing rate (less affected by inhibition than mitral)
    #[must_use]
    pub fn compute_rate(&mut self, glomerular_input: f32) -> f32 {
        self.firing_rate = (glomerular_input * 80.0).clamp(0.0, 200.0);
        self.firing_rate
    }
}

impl Default for TuftedCell {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Granule cell - inhibitory interneuron
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GranuleCell {
    /// Activation level
    pub activation: f32,
    /// Connected mitral cells (indices)
    pub connections: heapless::Vec<u16, 8>,
}

impl GranuleCell {
    /// Create a new granule cell
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            activation: 0.0,
            connections: heapless::Vec::new(),
        }
    }

    /// Compute inhibition output based on mitral cell activity
    #[must_use]
    pub fn compute_inhibition(&mut self, mitral_activities: &[f32]) -> f32 {
        // Sum input from connected mitral cells
        let total_input: f32 = self.connections
            .iter()
            .filter_map(|&idx| mitral_activities.get(idx as usize))
            .sum();

        // Sigmoidal activation
        self.activation = 1.0 / (1.0 + libm::expf(-0.1 * (total_input - 50.0)));

        // Return inhibitory output
        self.activation * 20.0 // Max 20 Hz inhibition
    }
}

impl Default for GranuleCell {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Olfactory Bulb Model
// ============================================================================

/// Complete olfactory bulb model
#[derive(Clone, Debug)]
pub struct OlfactoryBulb<const N_GLOM: usize, const N_MITRAL: usize, const N_GRANULE: usize> {
    /// Glomeruli
    pub glomeruli: heapless::Vec<Glomerulus, N_GLOM>,
    /// Mitral cells
    pub mitral_cells: heapless::Vec<MitralCell, N_MITRAL>,
    /// Granule cells
    pub granule_cells: heapless::Vec<GranuleCell, N_GRANULE>,
    /// Current odor representation (pattern across glomeruli)
    pub odor_pattern: heapless::Vec<f32, N_GLOM>,
}

impl<const N_GLOM: usize, const N_MITRAL: usize, const N_GRANULE: usize>
    OlfactoryBulb<N_GLOM, N_MITRAL, N_GRANULE>
{
    /// Create a new olfactory bulb
    #[must_use]
    pub fn new() -> Self {
        Self {
            glomeruli: heapless::Vec::new(),
            mitral_cells: heapless::Vec::new(),
            granule_cells: heapless::Vec::new(),
            odor_pattern: heapless::Vec::new(),
        }
    }

    /// Initialize with standard topology
    pub fn initialize(&mut self) {
        // Create glomeruli (one per receptor type, simplified)
        for i in 0..N_GLOM.min(8) {
            let pos = Position2D::new(
                (i as f32 / 8.0) * 10.0,
                libm::sinf(i as f32 * 0.5) * 5.0,
            );
            let _ = self.glomeruli.push(Glomerulus::new(OlfactoryReceptorId(i as u16), pos));
            let _ = self.odor_pattern.push(0.0);
        }

        // Create mitral cells (multiple per glomerulus)
        for i in 0..N_MITRAL.min(N_GLOM * 5) {
            let glom_id = (i % N_GLOM) as u16;
            let _ = self.mitral_cells.push(MitralCell::new(glom_id));
        }

        // Create granule cells
        for _ in 0..N_GRANULE.min(20) {
            let _ = self.granule_cells.push(GranuleCell::new());
        }
    }

    /// Process odorant input and return output pattern
    pub fn process(&mut self, orn_rates: &[f32]) -> &[f32] {
        // Aggregate ORN input to glomeruli
        for (i, glom) in self.glomeruli.iter_mut().enumerate() {
            if let Some(&rate) = orn_rates.get(i) {
                glom.update(rate * glom.n_inputs as f32);
            }
        }

        // Compute granule cell inhibition
        let mitral_rates: heapless::Vec<f32, N_MITRAL> = self.mitral_cells
            .iter()
            .map(|m| m.firing_rate)
            .collect();

        let mut total_inhibition = 0.0f32;
        for gc in &mut self.granule_cells {
            total_inhibition += gc.compute_inhibition(&mitral_rates);
        }
        let avg_inhibition = total_inhibition / self.granule_cells.len().max(1) as f32;

        // Update mitral cells with lateral inhibition
        for mc in &mut self.mitral_cells {
            let glom_input = self.glomeruli
                .get(mc.glomerulus_id as usize)
                .map(|g| g.input_activity)
                .unwrap_or(0.0);
            mc.compute_rate(glom_input, avg_inhibition);
        }

        // Update output pattern
        for (i, glom) in self.glomeruli.iter().enumerate() {
            if let Some(p) = self.odor_pattern.get_mut(i) {
                *p = glom.input_activity;
            }
        }

        &self.odor_pattern
    }
}

impl<const N_GLOM: usize, const N_MITRAL: usize, const N_GRANULE: usize> Default
    for OlfactoryBulb<N_GLOM, N_MITRAL, N_GRANULE>
{
    fn default() -> Self {
        let mut bulb = Self::new();
        bulb.initialize();
        bulb
    }
}

// ============================================================================
// Unified Olfactory Receptor Interface
// ============================================================================

/// Unified olfactory receptor that can represent any olfactory cell type
#[derive(Clone, Debug)]
pub enum OlfactoryReceptor {
    /// Olfactory receptor neuron
    Orn(OlfactoryReceptorNeuron),
    /// Mitral cell
    Mitral(MitralCell),
    /// Tufted cell
    Tufted(TuftedCell),
    /// Granule cell
    Granule(GranuleCell),
}

impl OlfactoryReceptor {
    /// Get the receptor type
    #[must_use]
    pub fn receptor_type(&self) -> ReceptorType {
        match self {
            Self::Orn(_) => ReceptorType::OlfactoryReceptorNeuron,
            Self::Mitral(_) => ReceptorType::MitralCell,
            Self::Tufted(_) => ReceptorType::TuftedCell,
            Self::Granule(_) => ReceptorType::OlfactoryGranuleCell,
        }
    }

    /// Get the spatial extent for olfactory receptors
    #[inline]
    #[must_use]
    pub fn spatial_extent() -> SpatialExtent {
        SpatialExtent::olfactory_epithelium()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_odorant_affinities() {
        let fruity = OdorantAffinities::from_class(OdorantClass::Fruity);
        let floral = OdorantAffinities::from_class(OdorantClass::Floral);

        // Same class should have high self-similarity
        assert!(fruity.similarity(&fruity) > 0.9);

        // Different classes should have lower similarity
        let cross_sim = fruity.similarity(&floral);
        assert!(cross_sim < fruity.similarity(&fruity));
    }

    #[test]
    fn test_orn_response() {
        let mut orn = OlfactoryReceptorNeuron::new(
            OlfactoryReceptorId(0), // Fruity-tuned
            Position2D::new(0.0, 0.0),
        );

        let fruity_odorant = Odorant::from_class(OdorantClass::Fruity);

        // Should respond to matching odorant
        let response = orn.compute_response(&fruity_odorant, 0.5, 10.0);
        assert!(response > 0.0);

        // Should adapt with sustained exposure
        for _ in 0..100 {
            orn.compute_response(&fruity_odorant, 0.5, 10.0);
        }
        assert!(orn.adaptation > 0.1);
    }

    #[test]
    fn test_mitral_cell() {
        let mut mc = MitralCell::new(0);

        // Should increase firing with input
        let rate = mc.compute_rate(1.0, 0.0);
        assert!(rate > mc.baseline_rate);

        // Should decrease with inhibition
        let inhibited_rate = mc.compute_rate(1.0, 1.0);
        assert!(inhibited_rate < rate);
    }

    #[test]
    fn test_olfactory_bulb() {
        let mut bulb: OlfactoryBulb<8, 16, 8> = OlfactoryBulb::default();

        // Process some input
        let orn_rates = [10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pattern = bulb.process(&orn_rates);

        // Should produce non-zero pattern
        assert!(pattern.iter().any(|&p| p > 0.0));
    }
}

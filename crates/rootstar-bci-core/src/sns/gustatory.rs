//! Gustatory Receptor Models
//!
//! Phenomenological models for taste receptor cells:
//! - **Sweet**: T1R2+T1R3 GPCRs (sugars, artificial sweeteners)
//! - **Umami**: T1R1+T1R3 GPCRs (glutamate, nucleotides)
//! - **Bitter**: T2R family (25+ GPCRs) (alkaloids, toxins)
//! - **Sour**: OTOP1 proton channel (acids)
//! - **Salty**: ENaC sodium channel (NaCl)
//!
//! # Taste Bud Organization
//!
//! Taste buds contain 50-100 taste receptor cells (TRCs) of three types:
//! - Type I: Glial-like support cells
//! - Type II: GPCR-expressing cells (sweet, umami, bitter)
//! - Type III: Sour-sensing cells
//! - ENaC cells: Salt-sensing (subset of Type I)
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::gustatory::{TasteReceptorCell, TasteQuality};
//!
//! let mut cell = TasteReceptorCell::new(TasteQuality::Sweet);
//! let response = cell.compute_response(0.5, 10.0);
//! ```

use serde::{Deserialize, Serialize};

use super::types::ReceptorType;

// ============================================================================
// Taste Quality
// ============================================================================

/// Primary taste quality
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TasteQuality {
    /// Sweet: sugars, saccharin, aspartame, stevia
    Sweet,
    /// Salty: sodium chloride, potassium chloride
    Salty,
    /// Sour: acids (citric, acetic, HCl)
    Sour,
    /// Bitter: quinine, caffeine, denatonium
    Bitter,
    /// Umami: monosodium glutamate, nucleotides (IMP, GMP)
    Umami,
}

impl TasteQuality {
    /// All taste qualities
    pub const ALL: [Self; 5] = [Self::Sweet, Self::Salty, Self::Sour, Self::Bitter, Self::Umami];

    /// Get the receptor type associated with this quality
    #[inline]
    #[must_use]
    pub const fn receptor_type(self) -> ReceptorType {
        match self {
            Self::Sweet | Self::Bitter | Self::Umami => ReceptorType::TypeIITasteCell,
            Self::Sour => ReceptorType::TypeIIITasteCell,
            Self::Salty => ReceptorType::ENaCTasteCell,
        }
    }

    /// Get the transduction mechanism
    #[inline]
    #[must_use]
    pub const fn mechanism(self) -> &'static str {
        match self {
            Self::Sweet => "T1R2+T1R3 GPCR → cAMP/PLCβ2",
            Self::Umami => "T1R1+T1R3 GPCR → cAMP/PLCβ2",
            Self::Bitter => "T2R family GPCR → PLCβ2/TRPM5",
            Self::Sour => "OTOP1 proton channel",
            Self::Salty => "ENaC epithelial Na+ channel",
        }
    }

    /// Get the detection threshold (normalized, 0-1)
    #[inline]
    #[must_use]
    pub const fn detection_threshold(self) -> f32 {
        match self {
            Self::Sweet => 0.01,   // Relatively high threshold
            Self::Salty => 0.005,  // Moderate threshold
            Self::Sour => 0.002,   // Low threshold (sensitive to acid)
            Self::Bitter => 0.0001, // Very low (evolutionary protection)
            Self::Umami => 0.002,  // Low threshold
        }
    }

    /// Get the typical hedonic value (pleasantness: -1 to +1)
    #[inline]
    #[must_use]
    pub const fn hedonic_default(self) -> f32 {
        match self {
            Self::Sweet => 0.8,   // Generally pleasant
            Self::Salty => 0.3,   // Pleasant at low concentrations
            Self::Sour => -0.2,   // Mildly unpleasant
            Self::Bitter => -0.8, // Generally unpleasant
            Self::Umami => 0.5,   // Pleasant, savory
        }
    }

    /// Check if this quality is typically aversive
    #[inline]
    #[must_use]
    pub const fn is_aversive(self) -> bool {
        matches!(self, Self::Bitter | Self::Sour)
    }
}

// ============================================================================
// Taste Sensitivity Profile
// ============================================================================

/// Sensitivity profile for all five taste qualities
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TasteSensitivity {
    /// Sweet sensitivity (0-1)
    pub sweet: f32,
    /// Salty sensitivity (0-1)
    pub salty: f32,
    /// Sour sensitivity (0-1)
    pub sour: f32,
    /// Bitter sensitivity (0-1)
    pub bitter: f32,
    /// Umami sensitivity (0-1)
    pub umami: f32,
}

impl TasteSensitivity {
    /// Create a uniform sensitivity profile
    #[inline]
    #[must_use]
    pub const fn uniform(level: f32) -> Self {
        Self {
            sweet: level,
            salty: level,
            sour: level,
            bitter: level,
            umami: level,
        }
    }

    /// Create a profile sensitive to all qualities
    #[inline]
    #[must_use]
    pub const fn all() -> Self {
        Self::uniform(1.0)
    }

    /// Get sensitivity for a specific quality
    #[inline]
    #[must_use]
    pub fn for_quality(&self, quality: TasteQuality) -> f32 {
        match quality {
            TasteQuality::Sweet => self.sweet,
            TasteQuality::Salty => self.salty,
            TasteQuality::Sour => self.sour,
            TasteQuality::Bitter => self.bitter,
            TasteQuality::Umami => self.umami,
        }
    }

    /// Set sensitivity for a specific quality
    pub fn set_for_quality(&mut self, quality: TasteQuality, value: f32) {
        match quality {
            TasteQuality::Sweet => self.sweet = value,
            TasteQuality::Salty => self.salty = value,
            TasteQuality::Sour => self.sour = value,
            TasteQuality::Bitter => self.bitter = value,
            TasteQuality::Umami => self.umami = value,
        }
    }
}

impl Default for TasteSensitivity {
    fn default() -> Self {
        Self::all()
    }
}

// ============================================================================
// Taste Receptor Cell
// ============================================================================

/// Taste receptor cell model
///
/// Models the response of a single TRC to tastant stimulation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TasteReceptorCell {
    /// Primary taste quality
    pub quality: TasteQuality,
    /// Sensitivity (0-1)
    pub sensitivity: f32,
    /// Half-activation concentration (EC50)
    pub ec50: f32,
    /// Hill coefficient (cooperativity)
    pub hill_coefficient: f32,
    /// Maximum response (arbitrary units)
    pub max_response: f32,
    /// Adaptation time constant (ms)
    pub tau_adapt: f32,
    /// Current adaptation state
    adaptation_state: f32,
    /// Current intracellular calcium level
    calcium_level: f32,
}

impl TasteReceptorCell {
    /// Create a new TRC for a specific taste quality
    #[must_use]
    pub fn new(quality: TasteQuality) -> Self {
        let (ec50, hill) = match quality {
            TasteQuality::Sweet => (0.1, 1.5),
            TasteQuality::Salty => (0.05, 1.0),
            TasteQuality::Sour => (0.02, 1.2),
            TasteQuality::Bitter => (0.001, 2.0), // High cooperativity
            TasteQuality::Umami => (0.03, 1.5),
        };

        Self {
            quality,
            sensitivity: 1.0,
            ec50,
            hill_coefficient: hill,
            max_response: 100.0,
            tau_adapt: 2000.0, // Slow adaptation
            adaptation_state: 0.0,
            calcium_level: 0.0,
        }
    }

    /// Create with custom sensitivity
    #[must_use]
    pub fn with_sensitivity(quality: TasteQuality, sensitivity: f32) -> Self {
        let mut cell = Self::new(quality);
        cell.sensitivity = sensitivity.clamp(0.0, 1.0);
        cell
    }

    /// Compute response to tastant concentration
    ///
    /// # Arguments
    /// * `concentration` - Normalized tastant concentration (0-1)
    /// * `dt_ms` - Time step in milliseconds
    ///
    /// # Returns
    /// Response level (0 to max_response)
    pub fn compute_response(&mut self, concentration: f32, dt_ms: f32) -> f32 {
        // Apply sensitivity
        let effective_conc = concentration * self.sensitivity;

        // Hill equation: R = R_max × [C]^n / (EC50^n + [C]^n)
        let c_n = libm::powf(effective_conc.max(0.0), self.hill_coefficient);
        let ec50_n = libm::powf(self.ec50, self.hill_coefficient);
        let steady_state = if c_n + ec50_n > 1e-10 {
            self.max_response * (c_n / (ec50_n + c_n))
        } else {
            0.0
        };

        // Slow adaptation
        let alpha = 1.0 - libm::expf(-dt_ms / self.tau_adapt);
        self.adaptation_state += alpha * (steady_state - self.adaptation_state);

        // Calcium dynamics (simplified)
        let calcium_target = self.adaptation_state / self.max_response;
        let calcium_tau = 100.0; // Faster than overall adaptation
        let calcium_alpha = 1.0 - libm::expf(-dt_ms / calcium_tau);
        self.calcium_level += calcium_alpha * (calcium_target - self.calcium_level);

        self.adaptation_state.max(0.0)
    }

    /// Get current intracellular calcium level
    #[inline]
    #[must_use]
    pub fn calcium(&self) -> f32 {
        self.calcium_level
    }

    /// Check if the cell is responding (above threshold)
    #[inline]
    #[must_use]
    pub fn is_responding(&self) -> bool {
        self.calcium_level > 0.1
    }

    /// Reset cell state
    pub fn reset(&mut self) {
        self.adaptation_state = 0.0;
        self.calcium_level = 0.0;
    }
}

impl Default for TasteReceptorCell {
    fn default() -> Self {
        Self::new(TasteQuality::Sweet)
    }
}

// ============================================================================
// Taste Bud
// ============================================================================

/// Taste bud model containing multiple TRCs
///
/// A typical taste bud contains 50-100 cells of different types.
#[derive(Clone, Debug)]
pub struct TasteBud {
    /// Sweet-responsive cells
    pub sweet_cells: heapless::Vec<TasteReceptorCell, 8>,
    /// Salty-responsive cells
    pub salty_cells: heapless::Vec<TasteReceptorCell, 8>,
    /// Sour-responsive cells
    pub sour_cells: heapless::Vec<TasteReceptorCell, 8>,
    /// Bitter-responsive cells
    pub bitter_cells: heapless::Vec<TasteReceptorCell, 8>,
    /// Umami-responsive cells
    pub umami_cells: heapless::Vec<TasteReceptorCell, 8>,
    /// Position on tongue (UV coordinates)
    pub position: (f32, f32),
    /// Papilla type
    pub papilla_type: PapillaType,
    /// Last computed responses [sweet, salty, sour, bitter, umami]
    last_responses: [f32; 5],
}

/// Type of papilla containing the taste bud
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PapillaType {
    /// Fungiform: scattered on anterior 2/3
    Fungiform,
    /// Foliate: lateral edges
    Foliate,
    /// Circumvallate: V-shaped row at posterior
    Circumvallate,
}

impl TasteBud {
    /// Create a new taste bud at given position
    #[must_use]
    pub fn at_position(position: (f32, f32), papilla_type: PapillaType) -> Self {
        let mut bud = Self {
            sweet_cells: heapless::Vec::new(),
            salty_cells: heapless::Vec::new(),
            sour_cells: heapless::Vec::new(),
            bitter_cells: heapless::Vec::new(),
            umami_cells: heapless::Vec::new(),
            position,
            papilla_type,
            last_responses: [0.0; 5],
        };

        // Populate with typical distribution
        bud.populate_default();
        bud
    }

    /// Create a taste bud for a tongue region
    #[must_use]
    pub fn new(region: TongueRegion) -> Self {
        let (papilla_type, position) = match region {
            TongueRegion::TipAnterior => (PapillaType::Fungiform, (0.5, 0.9)),
            TongueRegion::Anterior => (PapillaType::Fungiform, (0.5, 0.7)),
            TongueRegion::Lateral => (PapillaType::Foliate, (0.2, 0.5)),
            TongueRegion::Posterior => (PapillaType::Circumvallate, (0.5, 0.2)),
            TongueRegion::SoftPalate => (PapillaType::Circumvallate, (0.5, 0.1)),
        };
        Self::at_position(position, papilla_type)
    }

    /// Populate with default cell distribution
    fn populate_default(&mut self) {
        // Typical distribution based on papilla type
        let (n_sweet, n_salty, n_sour, n_bitter, n_umami) = match self.papilla_type {
            PapillaType::Fungiform => (3, 2, 2, 2, 2),
            PapillaType::Foliate => (2, 2, 3, 3, 2),
            PapillaType::Circumvallate => (2, 1, 2, 4, 2),
        };

        for _ in 0..n_sweet {
            let _ = self.sweet_cells.push(TasteReceptorCell::new(TasteQuality::Sweet));
        }
        for _ in 0..n_salty {
            let _ = self.salty_cells.push(TasteReceptorCell::new(TasteQuality::Salty));
        }
        for _ in 0..n_sour {
            let _ = self.sour_cells.push(TasteReceptorCell::new(TasteQuality::Sour));
        }
        for _ in 0..n_bitter {
            let _ = self.bitter_cells.push(TasteReceptorCell::new(TasteQuality::Bitter));
        }
        for _ in 0..n_umami {
            let _ = self.umami_cells.push(TasteReceptorCell::new(TasteQuality::Umami));
        }
    }

    /// Compute response to a tastant stimulus
    ///
    /// # Arguments
    /// * `quality` - Taste quality of the stimulus
    /// * `concentration` - Normalized concentration (0-1)
    /// * `dt_ms` - Time step in milliseconds
    ///
    /// # Returns
    /// Mean response of cells sensitive to this quality
    pub fn compute_response(&mut self, quality: TasteQuality, concentration: f32, dt_ms: f32) -> f32 {
        let cells = match quality {
            TasteQuality::Sweet => &mut self.sweet_cells,
            TasteQuality::Salty => &mut self.salty_cells,
            TasteQuality::Sour => &mut self.sour_cells,
            TasteQuality::Bitter => &mut self.bitter_cells,
            TasteQuality::Umami => &mut self.umami_cells,
        };

        if cells.is_empty() {
            return 0.0;
        }

        let sum: f32 = cells.iter_mut()
            .map(|c| c.compute_response(concentration, dt_ms))
            .sum();

        sum / cells.len() as f32
    }

    /// Compute responses to all qualities
    pub fn compute_all_responses(&mut self, concentrations: &TasteSensitivity, dt_ms: f32) -> TasteSensitivity {
        TasteSensitivity {
            sweet: self.compute_response(TasteQuality::Sweet, concentrations.sweet, dt_ms),
            salty: self.compute_response(TasteQuality::Salty, concentrations.salty, dt_ms),
            sour: self.compute_response(TasteQuality::Sour, concentrations.sour, dt_ms),
            bitter: self.compute_response(TasteQuality::Bitter, concentrations.bitter, dt_ms),
            umami: self.compute_response(TasteQuality::Umami, concentrations.umami, dt_ms),
        }
    }

    /// Get total cell count
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.sweet_cells.len() +
            self.salty_cells.len() +
            self.sour_cells.len() +
            self.bitter_cells.len() +
            self.umami_cells.len()
    }

    /// Reset all cells
    pub fn reset(&mut self) {
        for cell in &mut self.sweet_cells { cell.reset(); }
        for cell in &mut self.salty_cells { cell.reset(); }
        for cell in &mut self.sour_cells { cell.reset(); }
        for cell in &mut self.bitter_cells { cell.reset(); }
        for cell in &mut self.umami_cells { cell.reset(); }
        self.last_responses = [0.0; 5];
    }

    /// Update responses from concentration array [sweet, salty, sour, bitter, umami]
    pub fn update(&mut self, concentrations: &[f32; 5], dt_ms: f32) {
        self.last_responses[0] = self.compute_response(TasteQuality::Sweet, concentrations[0], dt_ms);
        self.last_responses[1] = self.compute_response(TasteQuality::Salty, concentrations[1], dt_ms);
        self.last_responses[2] = self.compute_response(TasteQuality::Sour, concentrations[2], dt_ms);
        self.last_responses[3] = self.compute_response(TasteQuality::Bitter, concentrations[3], dt_ms);
        self.last_responses[4] = self.compute_response(TasteQuality::Umami, concentrations[4], dt_ms);
    }

    /// Get last computed responses [sweet, salty, sour, bitter, umami]
    #[must_use]
    pub fn get_responses(&self) -> &[f32; 5] {
        &self.last_responses
    }
}

// ============================================================================
// Gustatory Receptor (Combined)
// ============================================================================

/// Combined gustatory receptor model
pub type GustatoryReceptor = TasteBud;

// ============================================================================
// Tongue Region
// ============================================================================

/// Regions of the tongue with different taste sensitivities
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TongueRegion {
    /// Tip of tongue (very anterior)
    TipAnterior,
    /// Front/middle of tongue
    Anterior,
    /// Sides of tongue (lateral)
    Lateral,
    /// Back of tongue (posterior)
    Posterior,
    /// Soft palate (roof of mouth)
    SoftPalate,
}

impl TongueRegion {
    /// Get the sensitivity profile for this region
    ///
    /// Note: The "tongue map" is largely a myth, but there are slight variations.
    #[must_use]
    pub const fn sensitivity_profile(self) -> TasteSensitivity {
        match self {
            Self::TipAnterior => TasteSensitivity {
                sweet: 1.0,
                salty: 0.9,
                sour: 0.7,
                bitter: 0.6,
                umami: 0.8,
            },
            Self::Anterior => TasteSensitivity {
                sweet: 0.9,
                salty: 0.8,
                sour: 0.7,
                bitter: 0.7,
                umami: 0.8,
            },
            Self::Lateral => TasteSensitivity {
                sweet: 0.8,
                salty: 0.9,
                sour: 1.0, // Slightly more sour-sensitive
                bitter: 0.8,
                umami: 0.7,
            },
            Self::Posterior => TasteSensitivity {
                sweet: 0.7,
                salty: 0.7,
                sour: 0.8,
                bitter: 1.0, // More bitter-sensitive (protective)
                umami: 0.9,
            },
            Self::SoftPalate => TasteSensitivity {
                sweet: 0.6,
                salty: 0.5,
                sour: 0.6,
                bitter: 0.9,
                umami: 0.7,
            },
        }
    }

    /// Get typical papilla count for this region
    #[must_use]
    pub const fn papilla_count(self) -> u16 {
        match self {
            Self::TipAnterior => 50,
            Self::Anterior => 80,
            Self::Lateral => 40,
            Self::Posterior => 12, // Circumvallate are few but large
            Self::SoftPalate => 20,
        }
    }

    /// Get dominant papilla type
    #[must_use]
    pub const fn dominant_papilla(self) -> PapillaType {
        match self {
            Self::TipAnterior | Self::Anterior => PapillaType::Fungiform,
            Self::Lateral => PapillaType::Foliate,
            Self::Posterior | Self::SoftPalate => PapillaType::Circumvallate,
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
    fn test_taste_cell_response() {
        let mut cell = TasteReceptorCell::new(TasteQuality::Sweet);

        // Zero concentration
        let r0 = cell.compute_response(0.0, 10.0);
        assert!(r0 < 0.1);

        cell.reset();

        // High concentration
        let r_high = cell.compute_response(1.0, 10.0);
        assert!(r_high > 50.0);
    }

    #[test]
    fn test_taste_cell_adaptation() {
        let mut cell = TasteReceptorCell::new(TasteQuality::Bitter);

        // Initial response
        let r1 = cell.compute_response(0.5, 10.0);

        // Continued exposure (many time steps)
        let mut r_final = 0.0;
        for _ in 0..100 {
            r_final = cell.compute_response(0.5, 100.0);
        }

        // Adaptation should cause slight decrease (but still responsive)
        assert!(r_final <= r1 * 1.1); // Allow for some variation
    }

    #[test]
    fn test_taste_bud() {
        let mut bud = TasteBud::new((0.5, 0.5), PapillaType::Fungiform);

        assert!(bud.cell_count() > 5);
        assert!(bud.cell_count() < 50);

        // Test response
        let response = bud.compute_response(TasteQuality::Sweet, 0.5, 10.0);
        assert!(response > 0.0);
    }

    #[test]
    fn test_taste_quality_threshold() {
        // Bitter should have lowest threshold (protective)
        assert!(TasteQuality::Bitter.detection_threshold() < TasteQuality::Sweet.detection_threshold());
    }

    #[test]
    fn test_tongue_region() {
        let posterior = TongueRegion::Posterior.sensitivity_profile();
        let tip = TongueRegion::Tip.sensitivity_profile();

        // Posterior more bitter-sensitive, tip more sweet-sensitive
        assert!(posterior.bitter >= tip.bitter);
        assert!(tip.sweet >= posterior.sweet);
    }
}

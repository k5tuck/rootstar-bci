//! Body region definitions and mapping.
//!
//! Maps UV coordinates to anatomical body regions for neural stimulation.

use serde::{Deserialize, Serialize};

/// Body region identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BodyRegionId {
    // Head regions
    /// Forehead
    HeadForehead = 0,
    /// Left cheek
    HeadLeftCheek = 1,
    /// Right cheek
    HeadRightCheek = 2,

    // Left arm regions
    /// Left upper arm
    ArmLeftUpper = 10,
    /// Left forearm
    ArmLeftForearm = 11,
    /// Left hand
    ArmLeftHand = 12,

    // Right arm regions
    /// Right upper arm
    ArmRightUpper = 20,
    /// Right forearm
    ArmRightForearm = 21,
    /// Right hand
    ArmRightHand = 22,

    // Torso regions
    /// Chest
    TorsoChest = 30,
    /// Abdomen
    TorsoAbdomen = 31,
    /// Upper back
    TorsoUpperBack = 32,
    /// Lower back
    TorsoLowerBack = 33,

    // Left leg regions
    /// Left thigh
    LegLeftThigh = 40,
    /// Left calf
    LegLeftCalf = 41,
    /// Left foot
    LegLeftFoot = 42,

    // Right leg regions
    /// Right thigh
    LegRightThigh = 50,
    /// Right calf
    LegRightCalf = 51,
    /// Right foot
    LegRightFoot = 52,

    /// Unknown region
    Unknown = 255,
}

impl BodyRegionId {
    /// Get the string identifier for database lookup.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::HeadForehead => "HEAD_01",
            Self::HeadLeftCheek => "HEAD_02",
            Self::HeadRightCheek => "HEAD_03",
            Self::ArmLeftUpper => "ARM_L_01",
            Self::ArmLeftForearm => "ARM_L_02",
            Self::ArmLeftHand => "ARM_L_03",
            Self::ArmRightUpper => "ARM_R_01",
            Self::ArmRightForearm => "ARM_R_02",
            Self::ArmRightHand => "ARM_R_03",
            Self::TorsoChest => "TORSO_01",
            Self::TorsoAbdomen => "TORSO_02",
            Self::TorsoUpperBack => "TORSO_03",
            Self::TorsoLowerBack => "TORSO_04",
            Self::LegLeftThigh => "LEG_L_01",
            Self::LegLeftCalf => "LEG_L_02",
            Self::LegLeftFoot => "LEG_L_03",
            Self::LegRightThigh => "LEG_R_01",
            Self::LegRightCalf => "LEG_R_02",
            Self::LegRightFoot => "LEG_R_03",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::HeadForehead => "Forehead",
            Self::HeadLeftCheek => "Left Cheek",
            Self::HeadRightCheek => "Right Cheek",
            Self::ArmLeftUpper => "Left Upper Arm",
            Self::ArmLeftForearm => "Left Forearm",
            Self::ArmLeftHand => "Left Hand",
            Self::ArmRightUpper => "Right Upper Arm",
            Self::ArmRightForearm => "Right Forearm",
            Self::ArmRightHand => "Right Hand",
            Self::TorsoChest => "Chest",
            Self::TorsoAbdomen => "Abdomen",
            Self::TorsoUpperBack => "Upper Back",
            Self::TorsoLowerBack => "Lower Back",
            Self::LegLeftThigh => "Left Thigh",
            Self::LegLeftCalf => "Left Calf",
            Self::LegLeftFoot => "Left Foot",
            Self::LegRightThigh => "Right Thigh",
            Self::LegRightCalf => "Right Calf",
            Self::LegRightFoot => "Right Foot",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if this is a limb region (arm or leg).
    #[must_use]
    pub const fn is_limb(&self) -> bool {
        matches!(
            self,
            Self::ArmLeftUpper
                | Self::ArmLeftForearm
                | Self::ArmLeftHand
                | Self::ArmRightUpper
                | Self::ArmRightForearm
                | Self::ArmRightHand
                | Self::LegLeftThigh
                | Self::LegLeftCalf
                | Self::LegLeftFoot
                | Self::LegRightThigh
                | Self::LegRightCalf
                | Self::LegRightFoot
        )
    }

    /// Check if this is a left-side region.
    #[must_use]
    pub const fn is_left(&self) -> bool {
        matches!(
            self,
            Self::HeadLeftCheek
                | Self::ArmLeftUpper
                | Self::ArmLeftForearm
                | Self::ArmLeftHand
                | Self::LegLeftThigh
                | Self::LegLeftCalf
                | Self::LegLeftFoot
        )
    }
}

impl Default for BodyRegionId {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Sensitivity level for neural stimulation.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SensitivityLevel(f32);

impl SensitivityLevel {
    /// Create a new sensitivity level.
    #[must_use]
    pub fn new(multiplier: f32) -> Self {
        Self(multiplier.max(0.1))
    }

    /// Standard sensitivity (1.0).
    #[must_use]
    pub const fn standard() -> Self {
        Self(1.0)
    }

    /// High sensitivity (for hands, face).
    #[must_use]
    pub const fn high() -> Self {
        Self(2.5)
    }

    /// Low sensitivity (for back, legs).
    #[must_use]
    pub const fn low() -> Self {
        Self(0.8)
    }

    /// Get the multiplier value.
    #[must_use]
    pub fn multiplier(&self) -> f32 {
        self.0
    }
}

impl Default for SensitivityLevel {
    fn default() -> Self {
        Self::standard()
    }
}

/// A body region with coordinate bounds and sensitivity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyRegion {
    /// Region identifier.
    pub id: BodyRegionId,
    /// UV coordinate bounds (min_x, max_x, min_y, max_y).
    pub uv_bounds: (f32, f32, f32, f32),
    /// Approximate vertex count in this region.
    pub vertex_count: u32,
    /// Sensitivity multiplier for stimulation.
    pub sensitivity: SensitivityLevel,
}

impl BodyRegion {
    /// Create a new body region.
    #[must_use]
    pub fn new(
        id: BodyRegionId,
        uv_min_x: f32,
        uv_max_x: f32,
        uv_min_y: f32,
        uv_max_y: f32,
        vertex_count: u32,
        sensitivity: SensitivityLevel,
    ) -> Self {
        Self {
            id,
            uv_bounds: (uv_min_x, uv_max_x, uv_min_y, uv_max_y),
            vertex_count,
            sensitivity,
        }
    }

    /// Check if a UV coordinate is within this region.
    #[must_use]
    pub fn contains_uv(&self, uv_x: f32, uv_y: f32) -> bool {
        uv_x >= self.uv_bounds.0
            && uv_x <= self.uv_bounds.1
            && uv_y >= self.uv_bounds.2
            && uv_y <= self.uv_bounds.3
    }

    /// Calculate coverage ratio for intensity calculation.
    ///
    /// Returns the fraction of affected vertices in the region.
    #[must_use]
    pub fn coverage_ratio(&self, affected_count: u32) -> f32 {
        if self.vertex_count == 0 {
            0.0
        } else {
            (affected_count as f32 / self.vertex_count as f32).min(1.0)
        }
    }
}

/// Map of all body regions for fast UV lookup.
#[derive(Debug)]
pub struct BodyRegionMap {
    regions: Vec<BodyRegion>,
}

impl BodyRegionMap {
    /// Create a new empty region map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Create the standard body region map from the VR Neural Mapping spec.
    #[must_use]
    pub fn standard() -> Self {
        let regions = vec![
            // Head regions
            BodyRegion::new(
                BodyRegionId::HeadForehead,
                0.45,
                0.55,
                0.90,
                0.95,
                200,
                SensitivityLevel::high(),
            ),
            BodyRegion::new(
                BodyRegionId::HeadLeftCheek,
                0.40,
                0.45,
                0.85,
                0.90,
                150,
                SensitivityLevel::high(),
            ),
            BodyRegion::new(
                BodyRegionId::HeadRightCheek,
                0.55,
                0.60,
                0.85,
                0.90,
                150,
                SensitivityLevel::high(),
            ),
            // Left arm regions
            BodyRegion::new(
                BodyRegionId::ArmLeftUpper,
                0.15,
                0.25,
                0.60,
                0.75,
                400,
                SensitivityLevel::standard(),
            ),
            BodyRegion::new(
                BodyRegionId::ArmLeftForearm,
                0.10,
                0.20,
                0.45,
                0.60,
                350,
                SensitivityLevel::new(1.2),
            ),
            BodyRegion::new(
                BodyRegionId::ArmLeftHand,
                0.05,
                0.15,
                0.35,
                0.45,
                500,
                SensitivityLevel::high(),
            ),
            // Right arm regions
            BodyRegion::new(
                BodyRegionId::ArmRightUpper,
                0.75,
                0.85,
                0.60,
                0.75,
                400,
                SensitivityLevel::standard(),
            ),
            BodyRegion::new(
                BodyRegionId::ArmRightForearm,
                0.80,
                0.90,
                0.45,
                0.60,
                350,
                SensitivityLevel::new(1.2),
            ),
            BodyRegion::new(
                BodyRegionId::ArmRightHand,
                0.85,
                0.95,
                0.35,
                0.45,
                500,
                SensitivityLevel::high(),
            ),
            // Torso regions
            BodyRegion::new(
                BodyRegionId::TorsoChest,
                0.35,
                0.65,
                0.55,
                0.75,
                800,
                SensitivityLevel::standard(),
            ),
            BodyRegion::new(
                BodyRegionId::TorsoAbdomen,
                0.35,
                0.65,
                0.40,
                0.55,
                600,
                SensitivityLevel::standard(),
            ),
            BodyRegion::new(
                BodyRegionId::TorsoUpperBack,
                0.35,
                0.65,
                0.55,
                0.75,
                700,
                SensitivityLevel::low(),
            ),
            BodyRegion::new(
                BodyRegionId::TorsoLowerBack,
                0.35,
                0.65,
                0.40,
                0.55,
                500,
                SensitivityLevel::low(),
            ),
            // Left leg regions
            BodyRegion::new(
                BodyRegionId::LegLeftThigh,
                0.30,
                0.45,
                0.20,
                0.40,
                500,
                SensitivityLevel::low(),
            ),
            BodyRegion::new(
                BodyRegionId::LegLeftCalf,
                0.30,
                0.45,
                0.05,
                0.20,
                400,
                SensitivityLevel::low(),
            ),
            BodyRegion::new(
                BodyRegionId::LegLeftFoot,
                0.30,
                0.40,
                0.00,
                0.05,
                300,
                SensitivityLevel::high(),
            ),
            // Right leg regions
            BodyRegion::new(
                BodyRegionId::LegRightThigh,
                0.55,
                0.70,
                0.20,
                0.40,
                500,
                SensitivityLevel::low(),
            ),
            BodyRegion::new(
                BodyRegionId::LegRightCalf,
                0.55,
                0.70,
                0.05,
                0.20,
                400,
                SensitivityLevel::low(),
            ),
            BodyRegion::new(
                BodyRegionId::LegRightFoot,
                0.60,
                0.70,
                0.00,
                0.05,
                300,
                SensitivityLevel::high(),
            ),
        ];

        Self { regions }
    }

    /// Find the region containing a UV coordinate.
    #[must_use]
    pub fn find_region(&self, uv_x: f32, uv_y: f32) -> Option<&BodyRegion> {
        self.regions.iter().find(|r| r.contains_uv(uv_x, uv_y))
    }

    /// Find the region ID containing a UV coordinate.
    #[must_use]
    pub fn find_region_id(&self, uv_x: f32, uv_y: f32) -> BodyRegionId {
        self.find_region(uv_x, uv_y)
            .map(|r| r.id)
            .unwrap_or(BodyRegionId::Unknown)
    }

    /// Get a region by ID.
    #[must_use]
    pub fn get_region(&self, id: BodyRegionId) -> Option<&BodyRegion> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Get all regions.
    #[must_use]
    pub fn all_regions(&self) -> &[BodyRegion] {
        &self.regions
    }

    /// Get total vertex count across all regions.
    #[must_use]
    pub fn total_vertex_count(&self) -> u32 {
        self.regions.iter().map(|r| r.vertex_count).sum()
    }
}

impl Default for BodyRegionMap {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_lookup() {
        let map = BodyRegionMap::standard();

        // Test left hand (high sensitivity)
        let region = map.find_region(0.10, 0.40);
        assert!(region.is_some());
        assert_eq!(region.unwrap().id, BodyRegionId::ArmLeftHand);

        // Test chest
        let region = map.find_region(0.50, 0.65);
        assert!(region.is_some());
        assert_eq!(region.unwrap().id, BodyRegionId::TorsoChest);

        // Test forehead
        let region = map.find_region(0.50, 0.92);
        assert!(region.is_some());
        assert_eq!(region.unwrap().id, BodyRegionId::HeadForehead);
    }

    #[test]
    fn test_region_id_string() {
        assert_eq!(BodyRegionId::ArmLeftHand.as_str(), "ARM_L_03");
        assert_eq!(BodyRegionId::TorsoChest.as_str(), "TORSO_01");
    }

    #[test]
    fn test_sensitivity_levels() {
        let map = BodyRegionMap::standard();

        let hand = map.get_region(BodyRegionId::ArmLeftHand).unwrap();
        let back = map.get_region(BodyRegionId::TorsoLowerBack).unwrap();

        assert!(hand.sensitivity.multiplier() > back.sensitivity.multiplier());
    }

    #[test]
    fn test_coverage_ratio() {
        let region = BodyRegion::new(
            BodyRegionId::ArmLeftHand,
            0.05,
            0.15,
            0.35,
            0.45,
            500,
            SensitivityLevel::high(),
        );

        assert!((region.coverage_ratio(250) - 0.5).abs() < 0.01);
        assert!((region.coverage_ratio(500) - 1.0).abs() < 0.01);
        assert!((region.coverage_ratio(600) - 1.0).abs() < 0.01); // Clamped
    }
}

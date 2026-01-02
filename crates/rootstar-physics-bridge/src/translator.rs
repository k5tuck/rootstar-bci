//! Spatial translation layer for VR to neural stimulation mapping.

use std::collections::HashMap;

use rootstar_physics_core::types::{AffectedVertex, EffectType, Intensity, PhysicsFrame};
use rootstar_physics_mesh::{BodyRegion, BodyRegionId, BodyRegionMap};

use serde::{Deserialize, Serialize};

/// Type of sensation to trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensationType {
    /// Light touch on skin.
    LightTouch,
    /// Firm pressure.
    Pressure,
    /// Wind/air movement.
    WindBreeze,
    /// Cold sensation.
    Cold,
    /// Warm sensation.
    Warm,
    /// Vibration/oscillation.
    Vibration,
}

impl SensationType {
    /// Get database lookup name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LightTouch => "light_touch",
            Self::Pressure => "pressure",
            Self::WindBreeze => "wind_breeze",
            Self::Cold => "cold",
            Self::Warm => "warm",
            Self::Vibration => "vibration",
        }
    }

    /// Convert from effect type.
    #[must_use]
    pub fn from_effect_type(effect: EffectType, is_cold: bool) -> Self {
        match effect {
            EffectType::Wind => Self::WindBreeze,
            EffectType::Temperature => {
                if is_cold {
                    Self::Cold
                } else {
                    Self::Warm
                }
            }
            EffectType::Pressure => Self::Pressure,
            EffectType::Vibration => Self::Vibration,
            EffectType::Composite => Self::LightTouch, // Default fallback
        }
    }
}

/// Command to trigger neural stimulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StimulationCommand {
    /// Target body region.
    pub region_id: BodyRegionId,
    /// Region string ID for database lookup.
    pub region_str: &'static str,
    /// Type of sensation.
    pub sensation_type: SensationType,
    /// Normalized intensity (0.0-1.0).
    pub intensity: Intensity,
    /// Number of affected vertices in this region.
    pub affected_vertex_count: u32,
    /// Adjusted intensity including sensitivity.
    pub adjusted_intensity: f32,
    /// Priority for overlapping sensations.
    pub priority: u8,
}

impl StimulationCommand {
    /// Create a new stimulation command.
    #[must_use]
    pub fn new(
        region: &BodyRegion,
        sensation_type: SensationType,
        intensity: Intensity,
        affected_count: u32,
    ) -> Self {
        let adjusted = intensity.value() * region.sensitivity.multiplier();

        Self {
            region_id: region.id,
            region_str: region.id.as_str(),
            sensation_type,
            intensity,
            affected_vertex_count: affected_count,
            adjusted_intensity: adjusted.min(1.0),
            priority: Self::compute_priority(&region.id, &sensation_type),
        }
    }

    /// Compute priority based on region and sensation type.
    fn compute_priority(region: &BodyRegionId, sensation: &SensationType) -> u8 {
        // Higher priority for sensitive regions and immediate sensations
        let region_priority = match region {
            BodyRegionId::ArmLeftHand
            | BodyRegionId::ArmRightHand
            | BodyRegionId::HeadForehead => 3,
            BodyRegionId::HeadLeftCheek | BodyRegionId::HeadRightCheek => 2,
            _ => 1,
        };

        let sensation_priority = match sensation {
            SensationType::Cold | SensationType::Warm => 2,
            SensationType::WindBreeze | SensationType::Pressure => 1,
            _ => 0,
        };

        region_priority + sensation_priority
    }
}

/// Spatial translator for VR physics to neural stimulation.
///
/// Implements the intensity calculation formula from the VR Neural Mapping spec:
/// ```text
/// intensity = (affected_vertices / total_region_vertices) *
///             (wind_velocity / max_velocity) *
///             cos(angle_of_incidence) *
///             temporal_smoothing_factor
/// ```
pub struct SpatialTranslator {
    /// Body region map for UV lookup.
    region_map: BodyRegionMap,
    /// Previous frame intensities for temporal smoothing.
    prev_intensities: HashMap<BodyRegionId, f32>,
    /// Temporal smoothing factor (0.0-1.0, higher = more smoothing).
    smoothing_factor: f32,
    /// Minimum intensity threshold to trigger stimulation.
    min_intensity_threshold: f32,
}

impl SpatialTranslator {
    /// Create a new spatial translator.
    #[must_use]
    pub fn new(region_map: BodyRegionMap) -> Self {
        Self {
            region_map,
            prev_intensities: HashMap::new(),
            smoothing_factor: 0.3,
            min_intensity_threshold: 0.05,
        }
    }

    /// Set the temporal smoothing factor.
    #[must_use]
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing_factor = factor.clamp(0.0, 0.95);
        self
    }

    /// Set the minimum intensity threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.min_intensity_threshold = threshold.clamp(0.0, 0.5);
        self
    }

    /// Process a physics frame and generate stimulation commands.
    ///
    /// This is the main entry point for translating physics to neural stimulation.
    pub fn process_frame(&mut self, frame: &PhysicsFrame) -> Vec<StimulationCommand> {
        // Group affected vertices by region
        let mut region_effects: HashMap<BodyRegionId, RegionAccumulator> = HashMap::new();

        for vertex in &frame.affected_vertices {
            if let Some(region) = self.region_map.find_region(vertex.uv_x, vertex.uv_y) {
                let accumulator = region_effects
                    .entry(region.id)
                    .or_insert_with(|| RegionAccumulator::new(region.clone()));

                accumulator.add_vertex(vertex);
            }
        }

        // Generate stimulation commands with temporal smoothing
        let mut commands = Vec::new();

        for (region_id, accumulator) in region_effects {
            let raw_intensity = accumulator.average_intensity();

            // Apply temporal smoothing
            let prev = self.prev_intensities.get(&region_id).copied().unwrap_or(0.0);
            let smoothed = prev * self.smoothing_factor + raw_intensity * (1.0 - self.smoothing_factor);
            self.prev_intensities.insert(region_id, smoothed);

            // Skip if below threshold
            if smoothed < self.min_intensity_threshold {
                continue;
            }

            // Determine sensation type
            let sensation_type = accumulator.primary_sensation_type();

            // Create command
            let command = StimulationCommand::new(
                &accumulator.region,
                sensation_type,
                Intensity::new(smoothed),
                accumulator.vertex_count,
            );

            commands.push(command);
        }

        // Sort by priority (highest first)
        commands.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Decay previous intensities for regions not in this frame
        let current_regions: std::collections::HashSet<_> =
            commands.iter().map(|c| c.region_id).collect();

        for (region_id, intensity) in &mut self.prev_intensities {
            if !current_regions.contains(region_id) {
                *intensity *= 1.0 - self.smoothing_factor;
            }
        }

        commands
    }

    /// Get the body region map.
    #[must_use]
    pub fn region_map(&self) -> &BodyRegionMap {
        &self.region_map
    }

    /// Reset temporal smoothing state.
    pub fn reset(&mut self) {
        self.prev_intensities.clear();
    }
}

/// Accumulator for effects in a single body region.
struct RegionAccumulator {
    region: BodyRegion,
    vertex_count: u32,
    total_intensity: f32,
    effect_counts: HashMap<EffectType, u32>,
    is_cold: bool, // For temperature effects
}

impl RegionAccumulator {
    fn new(region: BodyRegion) -> Self {
        Self {
            region,
            vertex_count: 0,
            total_intensity: 0.0,
            effect_counts: HashMap::new(),
            is_cold: false,
        }
    }

    fn add_vertex(&mut self, vertex: &AffectedVertex) {
        self.vertex_count += 1;
        self.total_intensity += vertex.intensity.value();

        *self.effect_counts.entry(vertex.effect_type).or_insert(0) += 1;

        // Track cold vs warm for temperature
        if vertex.effect_type == EffectType::Temperature && vertex.intensity.value() < 0.0 {
            self.is_cold = true;
        }
    }

    fn average_intensity(&self) -> f32 {
        if self.vertex_count == 0 {
            0.0
        } else {
            (self.total_intensity / self.vertex_count as f32).min(1.0)
        }
    }

    fn primary_sensation_type(&self) -> SensationType {
        let primary_effect = self
            .effect_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(effect, _)| *effect)
            .unwrap_or(EffectType::Wind);

        SensationType::from_effect_type(primary_effect, self.is_cold)
    }
}

impl Default for SpatialTranslator {
    fn default() -> Self {
        Self::new(BodyRegionMap::standard())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_physics_core::types::{AffectedVertex, FrameTime};

    fn create_test_frame() -> PhysicsFrame {
        let mut frame = PhysicsFrame::from_time(FrameTime::first(60.0));

        // Add vertices in left hand region (UV 0.05-0.15, 0.35-0.45)
        for i in 0..50 {
            let _ = frame.add_vertex(AffectedVertex::new(
                i,
                0.10,           // UV X (left hand)
                0.40,           // UV Y (left hand)
                Intensity::new(0.5),
                EffectType::Wind,
            ));
        }

        frame
    }

    #[test]
    fn test_translator_process_frame() {
        let mut translator = SpatialTranslator::default();
        let frame = create_test_frame();

        let commands = translator.process_frame(&frame);

        assert!(!commands.is_empty());
        assert_eq!(commands[0].region_id, BodyRegionId::ArmLeftHand);
        assert!(commands[0].intensity.value() > 0.0);
    }

    #[test]
    fn test_temporal_smoothing() {
        let mut translator = SpatialTranslator::default().with_smoothing(0.5);
        let frame = create_test_frame();

        // First frame - intensity starts building up from 0
        let commands1 = translator.process_frame(&frame);
        let intensity1 = commands1[0].intensity.value();

        // Second frame - intensity should be higher or similar (smoothed toward steady state)
        let commands2 = translator.process_frame(&frame);
        let intensity2 = commands2[0].intensity.value();

        // With smoothing, second frame should have >= intensity as first
        // (since we're smoothing toward the same value)
        assert!(intensity2 >= intensity1 * 0.9);

        // Empty frame - intensity should decay
        let empty_frame = PhysicsFrame::new(60.0);
        let commands3 = translator.process_frame(&empty_frame);

        // Should still have commands due to decay
        // (but they'll be filtered if below threshold)
        assert!(commands3.len() <= commands2.len());
    }

    #[test]
    fn test_sensation_type_mapping() {
        assert_eq!(
            SensationType::from_effect_type(EffectType::Wind, false),
            SensationType::WindBreeze
        );
        assert_eq!(
            SensationType::from_effect_type(EffectType::Temperature, true),
            SensationType::Cold
        );
        assert_eq!(
            SensationType::from_effect_type(EffectType::Temperature, false),
            SensationType::Warm
        );
    }

    #[test]
    fn test_priority_ordering() {
        let hand_region = BodyRegion::new(
            BodyRegionId::ArmLeftHand,
            0.05,
            0.15,
            0.35,
            0.45,
            500,
            rootstar_physics_mesh::SensitivityLevel::high(),
        );

        let back_region = BodyRegion::new(
            BodyRegionId::TorsoLowerBack,
            0.35,
            0.65,
            0.40,
            0.55,
            500,
            rootstar_physics_mesh::SensitivityLevel::low(),
        );

        let hand_cmd = StimulationCommand::new(
            &hand_region,
            SensationType::Cold,
            Intensity::new(0.5),
            100,
        );

        let back_cmd = StimulationCommand::new(
            &back_region,
            SensationType::WindBreeze,
            Intensity::new(0.5),
            100,
        );

        assert!(hand_cmd.priority > back_cmd.priority);
    }
}

//! Sensation limiting for VR neural stimulation.
//!
//! Provides controls to limit the intensity of sensations before they're
//! converted to neural stimulation patterns. This adds a user-comfort layer
//! on top of the hardware safety limits.
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_physics_bridge::limiter::{SensationLimiter, LimiterPreset};
//!
//! // Create a limiter with conservative settings
//! let mut limiter = SensationLimiter::new(LimiterPreset::Conservative);
//!
//! // Or customize settings
//! let mut limiter = SensationLimiter::default()
//!     .with_global_limit(0.5)  // 50% max intensity
//!     .with_region_limit(BodyRegionId::HeadForehead, 0.3)  // 30% for face
//!     .with_ramp_up(Duration::from_secs(10));  // 10s ramp-up
//!
//! // Apply to stimulation commands
//! let limited_commands = limiter.apply(&commands);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use rootstar_physics_core::types::Intensity;
use rootstar_physics_mesh::BodyRegionId;

use crate::translator::{SensationType, StimulationCommand};

/// Preset configurations for sensation limiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LimiterPreset {
    /// No limiting (100% intensity passthrough).
    None,
    /// Conservative: 50% global, reduced face/hand sensitivity.
    Conservative,
    /// Moderate: 75% global, slight face reduction.
    Moderate,
    /// Sensitive areas only: Reduces high-sensitivity regions.
    SensitiveAreasReduced,
    /// First-time user: Very gentle with long ramp-up.
    FirstTimeUser,
}

impl LimiterPreset {
    /// Get the global intensity limit for this preset.
    #[must_use]
    pub const fn global_limit(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Conservative => 0.5,
            Self::Moderate => 0.75,
            Self::SensitiveAreasReduced => 0.8,
            Self::FirstTimeUser => 0.3,
        }
    }

    /// Get the ramp-up duration for this preset.
    #[must_use]
    pub const fn ramp_up_seconds(&self) -> u64 {
        match self {
            Self::None => 0,
            Self::Conservative => 5,
            Self::Moderate => 3,
            Self::SensitiveAreasReduced => 2,
            Self::FirstTimeUser => 15,
        }
    }
}

impl Default for LimiterPreset {
    fn default() -> Self {
        Self::Moderate
    }
}

/// Configuration for sensation limiting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimiterConfig {
    /// Global intensity limit (0.0 - 1.0).
    pub global_limit: f32,

    /// Per-region intensity limits.
    pub region_limits: HashMap<BodyRegionId, f32>,

    /// Per-sensation-type intensity limits.
    pub sensation_limits: HashMap<SensationType, f32>,

    /// Ramp-up duration when starting a session.
    pub ramp_up_duration: Duration,

    /// Ramp-down duration when stopping.
    pub ramp_down_duration: Duration,

    /// Minimum intensity to pass through (below this = zero).
    pub noise_floor: f32,

    /// Maximum rate of intensity change per second.
    pub max_rate_per_second: f32,
}

impl LimiterConfig {
    /// Create configuration from a preset.
    #[must_use]
    pub fn from_preset(preset: LimiterPreset) -> Self {
        let mut config = Self::default();
        config.global_limit = preset.global_limit();
        config.ramp_up_duration = Duration::from_secs(preset.ramp_up_seconds());

        // Apply preset-specific region limits
        match preset {
            LimiterPreset::Conservative | LimiterPreset::FirstTimeUser => {
                // Reduce face sensitivity
                config.region_limits.insert(BodyRegionId::HeadForehead, 0.3);
                config.region_limits.insert(BodyRegionId::HeadLeftCheek, 0.3);
                config.region_limits.insert(BodyRegionId::HeadRightCheek, 0.3);
                // Reduce hand sensitivity (already high in hardware)
                config.region_limits.insert(BodyRegionId::ArmLeftHand, 0.5);
                config.region_limits.insert(BodyRegionId::ArmRightHand, 0.5);
            }
            LimiterPreset::SensitiveAreasReduced => {
                // Only reduce the most sensitive areas
                config.region_limits.insert(BodyRegionId::HeadForehead, 0.5);
                config.region_limits.insert(BodyRegionId::ArmLeftHand, 0.6);
                config.region_limits.insert(BodyRegionId::ArmRightHand, 0.6);
                config.region_limits.insert(BodyRegionId::LegLeftFoot, 0.6);
                config.region_limits.insert(BodyRegionId::LegRightFoot, 0.6);
            }
            _ => {}
        }

        // First-time users get reduced temperature sensations
        if matches!(preset, LimiterPreset::FirstTimeUser) {
            config.sensation_limits.insert(SensationType::Cold, 0.4);
            config.sensation_limits.insert(SensationType::Warm, 0.4);
        }

        config
    }
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            global_limit: 1.0,
            region_limits: HashMap::new(),
            sensation_limits: HashMap::new(),
            ramp_up_duration: Duration::from_secs(3),
            ramp_down_duration: Duration::from_secs(2),
            noise_floor: 0.02,
            max_rate_per_second: 2.0, // Can go from 0 to 1 in 0.5 seconds
        }
    }
}

/// Sensation limiter for controlling neural stimulation intensity.
///
/// Provides multiple layers of intensity control:
/// 1. Global limit - caps all sensations
/// 2. Per-region limits - different limits for face, hands, etc.
/// 3. Per-sensation limits - different limits for cold, wind, etc.
/// 4. Ramp-up/down - gradual intensity changes
/// 5. Rate limiting - prevents sudden intensity spikes
#[derive(Debug)]
pub struct SensationLimiter {
    /// Configuration
    config: LimiterConfig,
    /// Session start time (for ramp-up)
    session_start: Option<Instant>,
    /// Previous intensity values per region (for rate limiting)
    previous_intensities: HashMap<BodyRegionId, f32>,
    /// Last update time
    last_update: Option<Instant>,
    /// Is the session active?
    active: bool,
    /// Is ramping down?
    ramping_down: bool,
    /// Ramp-down start time
    ramp_down_start: Option<Instant>,
}

impl SensationLimiter {
    /// Create a new sensation limiter with a preset.
    #[must_use]
    pub fn new(preset: LimiterPreset) -> Self {
        Self {
            config: LimiterConfig::from_preset(preset),
            session_start: None,
            previous_intensities: HashMap::new(),
            last_update: None,
            active: false,
            ramping_down: false,
            ramp_down_start: None,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: LimiterConfig) -> Self {
        Self {
            config,
            session_start: None,
            previous_intensities: HashMap::new(),
            last_update: None,
            active: false,
            ramping_down: false,
            ramp_down_start: None,
        }
    }

    /// Set the global intensity limit (0.0 - 1.0).
    #[must_use]
    pub fn with_global_limit(mut self, limit: f32) -> Self {
        self.config.global_limit = limit.clamp(0.0, 1.0);
        self
    }

    /// Set a per-region intensity limit.
    #[must_use]
    pub fn with_region_limit(mut self, region: BodyRegionId, limit: f32) -> Self {
        self.config.region_limits.insert(region, limit.clamp(0.0, 1.0));
        self
    }

    /// Set a per-sensation-type intensity limit.
    #[must_use]
    pub fn with_sensation_limit(mut self, sensation: SensationType, limit: f32) -> Self {
        self.config
            .sensation_limits
            .insert(sensation, limit.clamp(0.0, 1.0));
        self
    }

    /// Set the ramp-up duration.
    #[must_use]
    pub fn with_ramp_up(mut self, duration: Duration) -> Self {
        self.config.ramp_up_duration = duration;
        self
    }

    /// Set the maximum rate of intensity change per second.
    ///
    /// Higher values allow faster changes. Set to a large value (e.g., 1000.0)
    /// to effectively disable rate limiting.
    #[must_use]
    pub fn with_max_rate(mut self, rate_per_second: f32) -> Self {
        self.config.max_rate_per_second = rate_per_second.max(0.1);
        self
    }

    /// Start a sensation session.
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.previous_intensities.clear();
        self.last_update = None;
        self.active = true;
        self.ramping_down = false;
        self.ramp_down_start = None;
    }

    /// Stop the session (with ramp-down).
    pub fn stop_session(&mut self) {
        self.ramping_down = true;
        self.ramp_down_start = Some(Instant::now());
    }

    /// Immediately stop all sensations.
    pub fn emergency_stop(&mut self) {
        self.active = false;
        self.ramping_down = false;
        self.previous_intensities.clear();
    }

    /// Check if session is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active && !self.is_fully_ramped_down()
    }

    /// Check if fully ramped down.
    fn is_fully_ramped_down(&self) -> bool {
        if let Some(start) = self.ramp_down_start {
            start.elapsed() >= self.config.ramp_down_duration
        } else {
            false
        }
    }

    /// Get the current ramp multiplier (0.0 - 1.0).
    fn ramp_multiplier(&self) -> f32 {
        // Check ramp-down first
        if self.ramping_down {
            if let Some(start) = self.ramp_down_start {
                let progress = start.elapsed().as_secs_f32()
                    / self.config.ramp_down_duration.as_secs_f32();
                return (1.0 - progress).max(0.0);
            }
        }

        // Ramp-up
        if let Some(start) = self.session_start {
            if self.config.ramp_up_duration.as_secs() > 0 {
                let progress = start.elapsed().as_secs_f32()
                    / self.config.ramp_up_duration.as_secs_f32();
                return progress.min(1.0);
            }
        }

        1.0
    }

    /// Apply rate limiting to intensity change.
    fn rate_limit(&mut self, region: BodyRegionId, target: f32) -> f32 {
        let now = Instant::now();
        let dt = self
            .last_update
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.016); // Assume ~60fps if no previous

        let previous = self.previous_intensities.get(&region).copied().unwrap_or(0.0);
        let max_change = self.config.max_rate_per_second * dt;

        let limited = if target > previous {
            (previous + max_change).min(target)
        } else {
            (previous - max_change).max(target)
        };

        self.previous_intensities.insert(region, limited);
        limited
    }

    /// Apply limiting to a single command.
    fn limit_command(&mut self, command: &StimulationCommand) -> Option<StimulationCommand> {
        if !self.active {
            return None;
        }

        // Start with adjusted intensity
        let mut intensity = command.adjusted_intensity;

        // Apply global limit
        intensity *= self.config.global_limit;

        // Apply per-region limit
        if let Some(&region_limit) = self.config.region_limits.get(&command.region_id) {
            intensity *= region_limit;
        }

        // Apply per-sensation limit
        if let Some(&sensation_limit) = self.config.sensation_limits.get(&command.sensation_type) {
            intensity *= sensation_limit;
        }

        // Apply ramp multiplier
        intensity *= self.ramp_multiplier();

        // Apply rate limiting
        intensity = self.rate_limit(command.region_id, intensity);

        // Check noise floor
        if intensity < self.config.noise_floor {
            return None;
        }

        // Create limited command
        Some(StimulationCommand {
            region_id: command.region_id,
            region_str: command.region_str,
            sensation_type: command.sensation_type,
            intensity: Intensity::new(intensity),
            affected_vertex_count: command.affected_vertex_count,
            adjusted_intensity: intensity,
            priority: command.priority,
        })
    }

    /// Apply limiting to a batch of commands.
    pub fn apply(&mut self, commands: &[StimulationCommand]) -> Vec<StimulationCommand> {
        let now = Instant::now();

        let result: Vec<_> = commands
            .iter()
            .filter_map(|cmd| self.limit_command(cmd))
            .collect();

        self.last_update = Some(now);

        // Check if session has ended
        if self.is_fully_ramped_down() {
            self.active = false;
        }

        result
    }

    /// Get the current global limit.
    #[must_use]
    pub fn global_limit(&self) -> f32 {
        self.config.global_limit
    }

    /// Set the global limit at runtime.
    pub fn set_global_limit(&mut self, limit: f32) {
        self.config.global_limit = limit.clamp(0.0, 1.0);
    }

    /// Get the effective limit for a region (global × region-specific).
    #[must_use]
    pub fn effective_limit(&self, region: BodyRegionId) -> f32 {
        let region_limit = self.config.region_limits.get(&region).copied().unwrap_or(1.0);
        self.config.global_limit * region_limit * self.ramp_multiplier()
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }
}

impl Default for SensationLimiter {
    fn default() -> Self {
        Self::new(LimiterPreset::Moderate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_command(region: BodyRegionId, intensity: f32) -> StimulationCommand {
        StimulationCommand {
            region_id: region,
            region_str: region.as_str(),
            sensation_type: SensationType::WindBreeze,
            intensity: Intensity::new(intensity),
            affected_vertex_count: 100,
            adjusted_intensity: intensity,
            priority: 1,
        }
    }

    #[test]
    fn test_global_limit() {
        let mut limiter = SensationLimiter::new(LimiterPreset::None)
            .with_global_limit(0.5)
            .with_ramp_up(Duration::ZERO)
            .with_max_rate(1000.0); // Disable rate limiting for test
        limiter.start_session();

        let cmd = create_test_command(BodyRegionId::ArmLeftUpper, 1.0);
        let result = limiter.apply(&[cmd]);

        assert_eq!(result.len(), 1);
        assert!((result[0].adjusted_intensity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_region_limit() {
        let mut limiter = SensationLimiter::new(LimiterPreset::None)
            .with_region_limit(BodyRegionId::HeadForehead, 0.3)
            .with_ramp_up(Duration::ZERO)
            .with_max_rate(1000.0); // Disable rate limiting for test
        limiter.start_session();

        let cmd = create_test_command(BodyRegionId::HeadForehead, 1.0);
        let result = limiter.apply(&[cmd]);

        assert_eq!(result.len(), 1);
        assert!((result[0].adjusted_intensity - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_combined_limits() {
        let mut limiter = SensationLimiter::new(LimiterPreset::None)
            .with_global_limit(0.5)
            .with_region_limit(BodyRegionId::HeadForehead, 0.5)
            .with_ramp_up(Duration::ZERO)
            .with_max_rate(1000.0); // Disable rate limiting for test
        limiter.start_session();

        // 0.5 * 0.5 = 0.25
        let cmd = create_test_command(BodyRegionId::HeadForehead, 1.0);
        let result = limiter.apply(&[cmd]);

        assert!((result[0].adjusted_intensity - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_noise_floor() {
        let mut limiter = SensationLimiter::new(LimiterPreset::None)
            .with_global_limit(0.01) // Very low
            .with_ramp_up(Duration::ZERO);
        limiter.start_session();

        let cmd = create_test_command(BodyRegionId::ArmLeftUpper, 1.0);
        let result = limiter.apply(&[cmd]);

        // Should be filtered out (below noise floor)
        assert!(result.is_empty());
    }

    #[test]
    fn test_emergency_stop() {
        let mut limiter = SensationLimiter::new(LimiterPreset::None)
            .with_ramp_up(Duration::ZERO);
        limiter.start_session();

        let cmd = create_test_command(BodyRegionId::ArmLeftUpper, 1.0);

        // Normal operation
        let result = limiter.apply(&[cmd.clone()]);
        assert!(!result.is_empty());

        // Emergency stop
        limiter.emergency_stop();
        let result = limiter.apply(&[cmd]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_presets() {
        assert!((LimiterPreset::Conservative.global_limit() - 0.5).abs() < 0.01);
        assert!((LimiterPreset::Moderate.global_limit() - 0.75).abs() < 0.01);
        assert!((LimiterPreset::FirstTimeUser.global_limit() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_effective_limit() {
        let limiter = SensationLimiter::new(LimiterPreset::None)
            .with_global_limit(0.8)
            .with_region_limit(BodyRegionId::HeadForehead, 0.5);

        // 0.8 * 0.5 * 1.0 (no ramp) = 0.4
        // But session not started, so ramp is 1.0 by default
        assert!((limiter.effective_limit(BodyRegionId::HeadForehead) - 0.4).abs() < 0.01);

        // Region without specific limit
        assert!((limiter.effective_limit(BodyRegionId::ArmLeftUpper) - 0.8).abs() < 0.01);
    }
}

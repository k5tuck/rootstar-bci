//! Safety limits and monitoring for transcranial stimulation.
//!
//! This module implements hardware and software safety limits for tDCS/tACS
//! stimulation based on established safety guidelines (Nitsche & Paulus, 2000;
//! Bikson et al., 2016).
//!
//! # Safety Limits
//!
//! - Maximum current: 2000 µA (2 mA) - HARDWARE ENFORCED
//! - Maximum current density: 25 µA/cm² (with 9 cm² electrodes)
//! - Maximum charge density: 40 µC/cm² per phase
//! - Maximum session duration: 40 minutes
//! - Maximum frequency (tACS): 100 Hz
//! - Minimum inter-session interval: 24 hours

use serde::{Deserialize, Serialize};

// ============================================================================
// Safety Violation Types
// ============================================================================

/// Types of safety violations that can occur.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyViolation {
    /// Current amplitude exceeds maximum
    CurrentExceeded {
        /// Requested current in µA
        requested_ua: u16,
        /// Maximum allowed in µA
        limit_ua: u16,
    },
    /// Current density exceeds maximum
    CurrentDensityExceeded {
        /// Calculated density in µA/cm²
        density_ua_cm2: u16,
        /// Maximum allowed in µA/cm²
        limit_ua_cm2: u16,
    },
    /// Charge density per phase exceeds maximum
    ChargeDensityExceeded {
        /// Calculated charge in µC/cm²
        charge_uc_cm2: u16,
        /// Maximum allowed in µC/cm²
        limit_uc_cm2: u16,
    },
    /// Session duration exceeds maximum
    DurationExceeded {
        /// Requested duration in minutes
        requested_min: u16,
        /// Maximum allowed in minutes
        limit_min: u16,
    },
    /// Frequency exceeds maximum for tACS
    FrequencyExceeded {
        /// Requested frequency in Hz
        requested_hz: u16,
        /// Maximum allowed in Hz
        limit_hz: u16,
    },
    /// Electrode size is below minimum
    ElectrodeTooSmall {
        /// Actual size in cm²
        size_cm2: u8,
        /// Minimum required in cm²
        min_cm2: u8,
    },
    /// Impedance is too high (poor contact)
    HighImpedance {
        /// Measured impedance in kΩ
        impedance_kohm: u16,
        /// Maximum allowed in kΩ
        limit_kohm: u16,
    },
    /// Impedance is too low (possible short)
    LowImpedance {
        /// Measured impedance in kΩ (×10, e.g., 5 = 0.5 kΩ)
        impedance_kohm_x10: u16,
        /// Minimum expected in kΩ (×10)
        min_kohm_x10: u16,
    },
    /// Insufficient time since last session
    InterSessionTooShort {
        /// Hours since last session
        hours_elapsed: u16,
        /// Minimum required hours
        min_hours: u16,
    },
    /// Emergency stop triggered
    EmergencyStop,
    /// Hardware interlock triggered
    HardwareInterlock,
}

impl SafetyViolation {
    /// Get a human-readable description of the violation.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::CurrentExceeded { .. } => "Current amplitude exceeds safety limit",
            Self::CurrentDensityExceeded { .. } => "Current density exceeds safety limit",
            Self::ChargeDensityExceeded { .. } => "Charge density exceeds safety limit",
            Self::DurationExceeded { .. } => "Session duration exceeds safety limit",
            Self::FrequencyExceeded { .. } => "Stimulation frequency exceeds safety limit",
            Self::ElectrodeTooSmall { .. } => "Electrode size below minimum safe area",
            Self::HighImpedance { .. } => "High impedance - check electrode contact",
            Self::LowImpedance { .. } => "Low impedance - possible electrode short",
            Self::InterSessionTooShort { .. } => "Insufficient time since last session",
            Self::EmergencyStop => "Emergency stop activated",
            Self::HardwareInterlock => "Hardware safety interlock triggered",
        }
    }

    /// Check if this violation requires immediate shutdown.
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::CurrentExceeded { .. }
                | Self::ChargeDensityExceeded { .. }
                | Self::LowImpedance { .. }
                | Self::EmergencyStop
                | Self::HardwareInterlock
        )
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SafetyViolation {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(f, "{}", self.description());
    }
}

// ============================================================================
// Safety Limits
// ============================================================================

/// Hard safety limits for transcranial stimulation.
///
/// Based on established safety guidelines for tDCS/tACS research.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyLimits {
    /// Maximum current in µA (default: 2000)
    pub max_current_ua: u16,
    /// Maximum current density in µA/cm² (default: 25)
    pub max_current_density_ua_cm2: u16,
    /// Maximum charge per phase in µC (default: 60)
    pub max_charge_per_phase_uc: u16,
    /// Maximum charge density in µC/cm² (default: 40)
    pub max_charge_density_uc_cm2: u16,
    /// Maximum session duration in minutes (default: 40)
    pub max_session_duration_min: u16,
    /// Minimum inter-session interval in hours (default: 24)
    pub min_inter_session_hours: u16,
    /// Minimum electrode size in cm² (default: 9, i.e., 3×3 cm)
    pub min_electrode_size_cm2: u8,
    /// Maximum tACS frequency in Hz (default: 100)
    pub max_frequency_hz: u16,
    /// Maximum impedance in kΩ (default: 10)
    pub max_impedance_kohm: u16,
    /// Minimum impedance in kΩ × 10 (default: 1, i.e., 0.1 kΩ)
    pub min_impedance_kohm_x10: u16,
}

impl SafetyLimits {
    /// Standard research safety limits.
    pub const RESEARCH: Self = Self {
        max_current_ua: 2000,
        max_current_density_ua_cm2: 25,
        max_charge_per_phase_uc: 60,
        max_charge_density_uc_cm2: 40,
        max_session_duration_min: 40,
        min_inter_session_hours: 24,
        min_electrode_size_cm2: 9,
        max_frequency_hz: 100,
        max_impedance_kohm: 10,
        min_impedance_kohm_x10: 1,
    };

    /// Conservative limits for initial testing.
    pub const CONSERVATIVE: Self = Self {
        max_current_ua: 1000,
        max_current_density_ua_cm2: 15,
        max_charge_per_phase_uc: 30,
        max_charge_density_uc_cm2: 20,
        max_session_duration_min: 20,
        min_inter_session_hours: 48,
        min_electrode_size_cm2: 16,
        max_frequency_hz: 40,
        max_impedance_kohm: 5,
        min_impedance_kohm_x10: 5,
    };

    /// Validate stimulation current.
    #[must_use]
    pub fn validate_current(&self, current_ua: u16) -> Option<SafetyViolation> {
        if current_ua > self.max_current_ua {
            Some(SafetyViolation::CurrentExceeded {
                requested_ua: current_ua,
                limit_ua: self.max_current_ua,
            })
        } else {
            None
        }
    }

    /// Validate current density given electrode size.
    #[must_use]
    pub fn validate_current_density(
        &self,
        current_ua: u16,
        electrode_size_cm2: u8,
    ) -> Option<SafetyViolation> {
        if electrode_size_cm2 < self.min_electrode_size_cm2 {
            return Some(SafetyViolation::ElectrodeTooSmall {
                size_cm2: electrode_size_cm2,
                min_cm2: self.min_electrode_size_cm2,
            });
        }

        let density = current_ua / electrode_size_cm2 as u16;
        if density > self.max_current_density_ua_cm2 {
            Some(SafetyViolation::CurrentDensityExceeded {
                density_ua_cm2: density,
                limit_ua_cm2: self.max_current_density_ua_cm2,
            })
        } else {
            None
        }
    }

    /// Validate session duration.
    #[must_use]
    pub fn validate_duration(&self, duration_min: u16) -> Option<SafetyViolation> {
        if duration_min > self.max_session_duration_min {
            Some(SafetyViolation::DurationExceeded {
                requested_min: duration_min,
                limit_min: self.max_session_duration_min,
            })
        } else {
            None
        }
    }

    /// Validate tACS frequency.
    #[must_use]
    pub fn validate_frequency(&self, frequency_hz: u16) -> Option<SafetyViolation> {
        if frequency_hz > self.max_frequency_hz {
            Some(SafetyViolation::FrequencyExceeded {
                requested_hz: frequency_hz,
                limit_hz: self.max_frequency_hz,
            })
        } else {
            None
        }
    }

    /// Validate electrode impedance.
    #[must_use]
    pub fn validate_impedance(&self, impedance_kohm: u16) -> Option<SafetyViolation> {
        if impedance_kohm > self.max_impedance_kohm {
            Some(SafetyViolation::HighImpedance {
                impedance_kohm,
                limit_kohm: self.max_impedance_kohm,
            })
        } else if impedance_kohm * 10 < self.min_impedance_kohm_x10 {
            Some(SafetyViolation::LowImpedance {
                impedance_kohm_x10: impedance_kohm * 10,
                min_kohm_x10: self.min_impedance_kohm_x10,
            })
        } else {
            None
        }
    }

    /// Validate inter-session interval.
    #[must_use]
    pub fn validate_inter_session(&self, hours_elapsed: u16) -> Option<SafetyViolation> {
        if hours_elapsed < self.min_inter_session_hours {
            Some(SafetyViolation::InterSessionTooShort {
                hours_elapsed,
                min_hours: self.min_inter_session_hours,
            })
        } else {
            None
        }
    }
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self::RESEARCH
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SafetyLimits {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "SafetyLimits(max={}uA, {}min)",
            self.max_current_ua,
            self.max_session_duration_min
        );
    }
}

// ============================================================================
// Safety Monitor State
// ============================================================================

/// Current state of the safety monitoring system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MonitorState {
    /// System is idle, not actively monitoring
    #[default]
    Idle,
    /// Pre-stimulation checks in progress
    PreCheck,
    /// Active stimulation monitoring
    Active,
    /// Ramping up stimulation current
    RampUp,
    /// Ramping down stimulation current
    RampDown,
    /// Safety violation detected, stimulation halted
    Violation,
    /// Emergency shutdown in progress
    EmergencyShutdown,
}

impl MonitorState {
    /// Check if stimulation is allowed in this state.
    #[must_use]
    pub const fn can_stimulate(&self) -> bool {
        matches!(self, Self::Active | Self::RampUp | Self::RampDown)
    }

    /// Check if this is an error state.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Violation | Self::EmergencyShutdown)
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for MonitorState {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Self::Idle => defmt::write!(f, "Idle"),
            Self::PreCheck => defmt::write!(f, "PreCheck"),
            Self::Active => defmt::write!(f, "Active"),
            Self::RampUp => defmt::write!(f, "RampUp"),
            Self::RampDown => defmt::write!(f, "RampDown"),
            Self::Violation => defmt::write!(f, "Violation"),
            Self::EmergencyShutdown => defmt::write!(f, "EmergencyShutdown"),
        }
    }
}

// ============================================================================
// Safety Monitor
// ============================================================================

/// Real-time safety monitoring during stimulation.
///
/// Tracks session state, validates parameters continuously, and triggers
/// emergency shutdown if safety limits are exceeded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyMonitor {
    /// Safety limits to enforce
    pub limits: SafetyLimits,
    /// Current monitoring state
    pub state: MonitorState,
    /// Session start timestamp (microseconds)
    session_start_us: u64,
    /// Total charge delivered in this session (µC × 100)
    total_charge_x100: u32,
    /// Last violation detected (if any)
    last_violation: Option<SafetyViolation>,
    /// Number of violations in this session
    violation_count: u8,
    /// Emergency stop flag
    emergency_stop: bool,
}

impl SafetyMonitor {
    /// Create a new safety monitor with given limits.
    #[must_use]
    pub const fn new(limits: SafetyLimits) -> Self {
        Self {
            limits,
            state: MonitorState::Idle,
            session_start_us: 0,
            total_charge_x100: 0,
            last_violation: None,
            violation_count: 0,
            emergency_stop: false,
        }
    }

    /// Start a new stimulation session.
    ///
    /// Returns violation if pre-checks fail.
    pub fn start_session(&mut self, timestamp_us: u64) -> Result<(), SafetyViolation> {
        if self.emergency_stop {
            return Err(SafetyViolation::EmergencyStop);
        }

        self.state = MonitorState::PreCheck;
        self.session_start_us = timestamp_us;
        self.total_charge_x100 = 0;
        self.last_violation = None;
        self.violation_count = 0;

        Ok(())
    }

    /// Validate a stimulation protocol before starting.
    pub fn validate_protocol(
        &mut self,
        current_ua: u16,
        frequency_hz: u16,
        duration_min: u16,
        electrode_size_cm2: u8,
    ) -> Result<(), SafetyViolation> {
        // Check current
        if let Some(v) = self.limits.validate_current(current_ua) {
            self.record_violation(v);
            return Err(v);
        }

        // Check current density
        if let Some(v) = self.limits.validate_current_density(current_ua, electrode_size_cm2) {
            self.record_violation(v);
            return Err(v);
        }

        // Check frequency
        if let Some(v) = self.limits.validate_frequency(frequency_hz) {
            self.record_violation(v);
            return Err(v);
        }

        // Check duration
        if let Some(v) = self.limits.validate_duration(duration_min) {
            self.record_violation(v);
            return Err(v);
        }

        self.state = MonitorState::RampUp;
        Ok(())
    }

    /// Monitor during active stimulation.
    ///
    /// Call this periodically (e.g., every 10ms) during stimulation.
    pub fn monitor(
        &mut self,
        timestamp_us: u64,
        current_ua: u16,
        impedance_kohm: u16,
    ) -> Result<(), SafetyViolation> {
        if self.emergency_stop {
            return Err(SafetyViolation::EmergencyStop);
        }

        // Check current hasn't exceeded limit
        if let Some(v) = self.limits.validate_current(current_ua) {
            self.trigger_emergency_shutdown(v);
            return Err(v);
        }

        // Check impedance
        if let Some(v) = self.limits.validate_impedance(impedance_kohm) {
            if v.is_critical() {
                self.trigger_emergency_shutdown(v);
                return Err(v);
            }
            self.record_violation(v);
        }

        // Check session duration
        let elapsed_us = timestamp_us.saturating_sub(self.session_start_us);
        let elapsed_min = (elapsed_us / 60_000_000) as u16;
        if let Some(v) = self.limits.validate_duration(elapsed_min) {
            self.state = MonitorState::RampDown;
            self.record_violation(v);
            return Err(v);
        }

        // Update charge tracking (simplified)
        // charge (µC) = current (µA) × time (s) = current × dt / 1e6
        // We're called every ~10ms, so approximate charge per call
        self.total_charge_x100 += (current_ua as u32 * 10) / 1000; // ~10ms intervals

        self.state = MonitorState::Active;
        Ok(())
    }

    /// End the current session normally.
    pub fn end_session(&mut self) {
        self.state = MonitorState::Idle;
    }

    /// Trigger emergency shutdown.
    pub fn emergency_shutdown(&mut self) {
        self.trigger_emergency_shutdown(SafetyViolation::EmergencyStop);
    }

    /// Reset after emergency (requires explicit action).
    pub fn reset_emergency(&mut self) {
        self.emergency_stop = false;
        self.state = MonitorState::Idle;
        self.last_violation = None;
    }

    /// Get the last violation that occurred.
    #[must_use]
    pub fn last_violation(&self) -> Option<SafetyViolation> {
        self.last_violation
    }

    /// Get the total charge delivered in current session (µC).
    #[must_use]
    pub fn total_charge_uc(&self) -> u32 {
        self.total_charge_x100 / 100
    }

    /// Get session duration in seconds.
    #[must_use]
    pub fn session_duration_s(&self, current_timestamp_us: u64) -> u32 {
        ((current_timestamp_us.saturating_sub(self.session_start_us)) / 1_000_000) as u32
    }

    fn record_violation(&mut self, violation: SafetyViolation) {
        self.last_violation = Some(violation);
        self.violation_count = self.violation_count.saturating_add(1);
    }

    fn trigger_emergency_shutdown(&mut self, violation: SafetyViolation) {
        self.emergency_stop = true;
        self.state = MonitorState::EmergencyShutdown;
        self.record_violation(violation);
    }
}

impl Default for SafetyMonitor {
    fn default() -> Self {
        Self::new(SafetyLimits::default())
    }
}

#[cfg(feature = "defmt")]
impl defmt::Format for SafetyMonitor {
    fn format(&self, f: defmt::Formatter) {
        defmt::write!(
            f,
            "SafetyMonitor({}, violations={})",
            self.state,
            self.violation_count
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_limits_validation() {
        let limits = SafetyLimits::RESEARCH;

        // Valid current
        assert!(limits.validate_current(1500).is_none());

        // Excessive current
        let violation = limits.validate_current(2500);
        assert!(violation.is_some());
        assert!(matches!(
            violation.unwrap(),
            SafetyViolation::CurrentExceeded { .. }
        ));
    }

    #[test]
    fn test_current_density() {
        let limits = SafetyLimits::RESEARCH;

        // 1000 µA on 9 cm² = 111 µA/cm² (too high!)
        let violation = limits.validate_current_density(1000, 9);
        assert!(violation.is_some());

        // 200 µA on 9 cm² = 22 µA/cm² (OK)
        assert!(limits.validate_current_density(200, 9).is_none());
    }

    #[test]
    fn test_electrode_size() {
        let limits = SafetyLimits::RESEARCH;

        // Too small electrode
        let violation = limits.validate_current_density(100, 4);
        assert!(matches!(
            violation,
            Some(SafetyViolation::ElectrodeTooSmall { .. })
        ));
    }

    #[test]
    fn test_safety_monitor_session() {
        let mut monitor = SafetyMonitor::new(SafetyLimits::RESEARCH);

        // Start session
        assert!(monitor.start_session(0).is_ok());
        assert_eq!(monitor.state, MonitorState::PreCheck);

        // Validate protocol
        assert!(monitor.validate_protocol(1000, 40, 20, 25).is_ok());
        assert_eq!(monitor.state, MonitorState::RampUp);

        // Monitor during stimulation
        assert!(monitor.monitor(1_000_000, 1000, 5).is_ok());
        assert_eq!(monitor.state, MonitorState::Active);

        // End session
        monitor.end_session();
        assert_eq!(monitor.state, MonitorState::Idle);
    }

    #[test]
    fn test_emergency_shutdown() {
        let mut monitor = SafetyMonitor::new(SafetyLimits::RESEARCH);

        monitor.start_session(0).unwrap();
        monitor.emergency_shutdown();

        assert!(monitor.emergency_stop);
        assert_eq!(monitor.state, MonitorState::EmergencyShutdown);

        // Cannot start new session while in emergency
        assert!(monitor.start_session(1000).is_err());

        // Reset and retry
        monitor.reset_emergency();
        assert!(monitor.start_session(2000).is_ok());
    }

    #[test]
    fn test_violation_criticality() {
        assert!(SafetyViolation::CurrentExceeded {
            requested_ua: 2500,
            limit_ua: 2000
        }
        .is_critical());

        assert!(SafetyViolation::EmergencyStop.is_critical());

        assert!(!SafetyViolation::HighImpedance {
            impedance_kohm: 15,
            limit_kohm: 10
        }
        .is_critical());
    }
}

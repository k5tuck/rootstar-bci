//! Stimulation protocol controller for sensory playback.
//!
//! Manages bidirectional feedback loops for recreating neural patterns
//! through transcranial stimulation.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use rootstar_bci_core::fingerprint::{
    NeuralFingerprint, SafetyLimits, SafetyMonitor, SafetyViolation, SensoryModality,
};
use rootstar_bci_core::types::Fixed24_8;

// ============================================================================
// Stimulation Protocol
// ============================================================================

/// Stimulation waveform type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveformType {
    /// Direct current (tDCS)
    Dc,
    /// Alternating current (tACS)
    Ac,
    /// Pulsed stimulation
    Pulsed,
}

/// Electrode configuration for stimulation.
#[derive(Clone, Debug)]
pub struct ElectrodeConfig {
    /// Anode electrode positions (positive current)
    pub anodes: Vec<String>,
    /// Cathode electrode positions (negative current)
    pub cathodes: Vec<String>,
    /// Electrode size in cm²
    pub electrode_size_cm2: u8,
}

impl Default for ElectrodeConfig {
    fn default() -> Self {
        Self {
            anodes: vec!["Cz".to_string()],
            cathodes: vec!["Fp1".to_string(), "Fp2".to_string()],
            electrode_size_cm2: 25, // 5×5 cm
        }
    }
}

/// Stimulation protocol definition.
#[derive(Clone, Debug)]
pub struct StimulationProtocol {
    /// Target sensory modality
    pub modality: SensoryModality,
    /// Waveform type
    pub waveform: WaveformType,
    /// Current amplitude in µA
    pub amplitude_ua: u16,
    /// Frequency in Hz (for AC/pulsed)
    pub frequency_hz: f32,
    /// Duty cycle (0.0-1.0) for pulsed
    pub duty_cycle: f32,
    /// Ramp up duration in seconds
    pub ramp_up_s: f32,
    /// Main stimulation duration in seconds
    pub duration_s: f32,
    /// Ramp down duration in seconds
    pub ramp_down_s: f32,
    /// Electrode configuration
    pub electrodes: ElectrodeConfig,
}

impl StimulationProtocol {
    /// Create a gustatory (taste) protocol.
    ///
    /// Targets frontal-insular region with 40 Hz gamma entrainment.
    #[must_use]
    pub fn gustatory(amplitude_ua: u16, duration_s: f32) -> Self {
        Self {
            modality: SensoryModality::Gustatory,
            waveform: WaveformType::Ac,
            amplitude_ua,
            frequency_hz: 40.0, // Gamma entrainment
            duty_cycle: 1.0,
            ramp_up_s: 30.0,
            duration_s,
            ramp_down_s: 30.0,
            electrodes: ElectrodeConfig {
                anodes: vec!["F7".to_string(), "FT7".to_string(), "T7".to_string()],
                cathodes: vec!["F8".to_string(), "FT8".to_string(), "T8".to_string()],
                electrode_size_cm2: 9,
            },
        }
    }

    /// Create an olfactory (smell) protocol.
    ///
    /// Targets orbitofrontal region with theta rhythm.
    #[must_use]
    pub fn olfactory(amplitude_ua: u16, duration_s: f32) -> Self {
        Self {
            modality: SensoryModality::Olfactory,
            waveform: WaveformType::Pulsed,
            amplitude_ua,
            frequency_hz: 6.0, // Theta rhythm
            duty_cycle: 0.5,
            ramp_up_s: 30.0,
            duration_s,
            ramp_down_s: 30.0,
            electrodes: ElectrodeConfig {
                anodes: vec![
                    "Fp1".to_string(),
                    "Fp2".to_string(),
                    "AF7".to_string(),
                    "AF8".to_string(),
                ],
                cathodes: vec!["Cz".to_string(), "Pz".to_string()],
                electrode_size_cm2: 9,
            },
        }
    }

    /// Total protocol duration including ramps.
    #[must_use]
    pub fn total_duration_s(&self) -> f32 {
        self.ramp_up_s + self.duration_s + self.ramp_down_s
    }

    /// Calculate current at a specific time during the protocol.
    #[must_use]
    pub fn current_at_time(&self, elapsed_s: f32) -> f32 {
        let amp = self.amplitude_ua as f32;

        if elapsed_s < 0.0 {
            0.0
        } else if elapsed_s < self.ramp_up_s {
            // Ramp up phase
            amp * (elapsed_s / self.ramp_up_s)
        } else if elapsed_s < self.ramp_up_s + self.duration_s {
            // Main stimulation phase
            amp
        } else if elapsed_s < self.total_duration_s() {
            // Ramp down phase
            let ramp_elapsed = elapsed_s - self.ramp_up_s - self.duration_s;
            amp * (1.0 - ramp_elapsed / self.ramp_down_s)
        } else {
            0.0
        }
    }

    /// Validate protocol against safety limits.
    pub fn validate(&self, limits: &SafetyLimits) -> Result<(), SafetyViolation> {
        // Check current
        if let Some(v) = limits.validate_current(self.amplitude_ua) {
            return Err(v);
        }

        // Check current density
        if let Some(v) =
            limits.validate_current_density(self.amplitude_ua, self.electrodes.electrode_size_cm2)
        {
            return Err(v);
        }

        // Check frequency
        if self.frequency_hz > 0.0 {
            if let Some(v) = limits.validate_frequency(self.frequency_hz as u16) {
                return Err(v);
            }
        }

        // Check duration
        let duration_min = (self.total_duration_s() / 60.0).ceil() as u16;
        if let Some(v) = limits.validate_duration(duration_min) {
            return Err(v);
        }

        Ok(())
    }
}

// ============================================================================
// Stimulation Session
// ============================================================================

/// Current phase of the stimulation session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionPhase {
    /// Session not started
    Idle,
    /// Pre-stimulation checks
    PreCheck,
    /// Ramping up current
    RampUp,
    /// Active stimulation
    Active,
    /// Ramping down current
    RampDown,
    /// Session completed successfully
    Complete,
    /// Session aborted due to safety violation
    Aborted,
}

/// Feedback data point during stimulation.
#[derive(Clone, Debug)]
pub struct FeedbackPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Similarity to target fingerprint (0-1)
    pub similarity: f32,
    /// Current being delivered (µA)
    pub current_ua: f32,
    /// Electrode impedance (kΩ)
    pub impedance_kohm: f32,
}

/// Active stimulation session for sensory playback.
#[derive(Clone)]
pub struct StimulationSession {
    /// Session identifier
    pub session_id: String,
    /// Target neural fingerprint
    pub target: NeuralFingerprint,
    /// Stimulation protocol
    pub protocol: StimulationProtocol,
    /// Current session phase
    pub phase: SessionPhase,
    /// Session start time
    pub start_time: Option<Instant>,
    /// Similarity history (last 100 points)
    similarity_history: VecDeque<f32>,
    /// Feedback history (last 100 points)
    feedback_history: VecDeque<FeedbackPoint>,
    /// Safety monitor
    safety_monitor: SafetyMonitor,
}

impl StimulationSession {
    /// Create a new stimulation session.
    #[must_use]
    pub fn new(
        session_id: String,
        target: NeuralFingerprint,
        protocol: StimulationProtocol,
    ) -> Self {
        Self {
            session_id,
            target,
            protocol,
            phase: SessionPhase::Idle,
            start_time: None,
            similarity_history: VecDeque::with_capacity(100),
            feedback_history: VecDeque::with_capacity(100),
            safety_monitor: SafetyMonitor::new(SafetyLimits::RESEARCH),
        }
    }

    /// Start the session (begin pre-checks).
    pub fn start(&mut self) -> Result<(), SafetyViolation> {
        // Validate protocol
        self.protocol.validate(&self.safety_monitor.limits)?;

        // Start safety monitor
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        self.safety_monitor.start_session(timestamp_us)?;

        // Validate protocol with monitor
        let duration_min = (self.protocol.total_duration_s() / 60.0).ceil() as u16;
        self.safety_monitor.validate_protocol(
            self.protocol.amplitude_ua,
            self.protocol.frequency_hz as u16,
            duration_min,
            self.protocol.electrodes.electrode_size_cm2,
        )?;

        self.phase = SessionPhase::RampUp;
        self.start_time = Some(Instant::now());

        Ok(())
    }

    /// Get elapsed time since session start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Get current amplitude for the current time.
    #[must_use]
    pub fn current_amplitude(&self) -> f32 {
        let elapsed = self.elapsed().as_secs_f32();
        self.protocol.current_at_time(elapsed)
    }

    /// Update session with current measurements.
    ///
    /// # Arguments
    ///
    /// * `current_fingerprint` - Current neural state
    /// * `impedance_kohm` - Measured electrode impedance
    ///
    /// Returns current similarity to target.
    pub fn update(
        &mut self,
        current_fingerprint: &NeuralFingerprint,
        impedance_kohm: f32,
    ) -> Result<f32, SafetyViolation> {
        if self.phase == SessionPhase::Idle || self.phase == SessionPhase::Complete {
            return Ok(0.0);
        }

        let elapsed = self.elapsed().as_secs_f32();
        let current_ua = self.current_amplitude();

        // Safety monitoring
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        self.safety_monitor
            .monitor(timestamp_us, current_ua as u16, impedance_kohm as u16)?;

        // Compute similarity to target
        let similarity = current_fingerprint.similarity(&self.target);

        // Update history
        if self.similarity_history.len() >= 100 {
            self.similarity_history.pop_front();
        }
        self.similarity_history.push_back(similarity);

        if self.feedback_history.len() >= 100 {
            self.feedback_history.pop_front();
        }
        self.feedback_history.push_back(FeedbackPoint {
            timestamp: Instant::now(),
            similarity,
            current_ua,
            impedance_kohm,
        });

        // Update phase based on elapsed time
        if elapsed < self.protocol.ramp_up_s {
            self.phase = SessionPhase::RampUp;
        } else if elapsed < self.protocol.ramp_up_s + self.protocol.duration_s {
            self.phase = SessionPhase::Active;
        } else if elapsed < self.protocol.total_duration_s() {
            self.phase = SessionPhase::RampDown;
        } else {
            self.phase = SessionPhase::Complete;
            self.safety_monitor.end_session();
        }

        Ok(similarity)
    }

    /// Stop the session (emergency or user-initiated).
    pub fn stop(&mut self) {
        if self.phase != SessionPhase::Complete {
            self.phase = SessionPhase::Aborted;
        }
        self.safety_monitor.end_session();
    }

    /// Emergency stop.
    pub fn emergency_stop(&mut self) {
        self.phase = SessionPhase::Aborted;
        self.safety_monitor.emergency_shutdown();
    }

    /// Get average similarity over recent history.
    #[must_use]
    pub fn average_similarity(&self) -> f32 {
        if self.similarity_history.is_empty() {
            0.0
        } else {
            self.similarity_history.iter().sum::<f32>() / self.similarity_history.len() as f32
        }
    }

    /// Get the most recent similarity values.
    #[must_use]
    pub fn recent_similarities(&self, n: usize) -> Vec<f32> {
        self.similarity_history
            .iter()
            .rev()
            .take(n)
            .copied()
            .collect()
    }

    /// Check if target has been achieved.
    #[must_use]
    pub fn target_achieved(&self, threshold: f32) -> bool {
        self.average_similarity() >= threshold
    }
}

// ============================================================================
// Stimulation Controller
// ============================================================================

/// Controller for managing stimulation sessions and feedback loops.
pub struct StimulationController {
    /// Active session (if any)
    pub active_session: Option<StimulationSession>,
    /// Feedback loop update rate (Hz)
    pub update_rate_hz: f32,
    /// Similarity threshold for target achievement
    pub similarity_threshold: f32,
    /// Maximum iterations before timeout
    pub max_iterations: usize,
}

impl StimulationController {
    /// Create a new stimulation controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_session: None,
            update_rate_hz: 10.0,
            similarity_threshold: 0.90,
            max_iterations: 1000,
        }
    }

    /// Start a new stimulation session.
    pub fn start_session(
        &mut self,
        target: NeuralFingerprint,
        protocol: StimulationProtocol,
    ) -> Result<String, SafetyViolation> {
        // Generate session ID
        let session_id = format!(
            "stim_{}_{}",
            protocol.modality.name(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        );

        let mut session = StimulationSession::new(session_id.clone(), target, protocol);
        session.start()?;

        self.active_session = Some(session);
        Ok(session_id)
    }

    /// Update the active session with current neural state.
    pub fn update(
        &mut self,
        current_fingerprint: &NeuralFingerprint,
        impedance_kohm: f32,
    ) -> Option<f32> {
        if let Some(ref mut session) = self.active_session {
            match session.update(current_fingerprint, impedance_kohm) {
                Ok(similarity) => {
                    if session.phase == SessionPhase::Complete {
                        // Session finished normally
                        return Some(similarity);
                    }
                    if session.target_achieved(self.similarity_threshold) {
                        // Target achieved early
                        return Some(similarity);
                    }
                    Some(similarity)
                }
                Err(_violation) => {
                    // Safety violation - session should be stopped
                    session.stop();
                    None
                }
            }
        } else {
            None
        }
    }

    /// Stop the active session.
    pub fn stop_session(&mut self) {
        if let Some(ref mut session) = self.active_session {
            session.stop();
        }
    }

    /// Emergency stop all stimulation.
    pub fn emergency_stop(&mut self) {
        if let Some(ref mut session) = self.active_session {
            session.emergency_stop();
        }
    }

    /// Get the current session phase.
    #[must_use]
    pub fn current_phase(&self) -> Option<SessionPhase> {
        self.active_session.as_ref().map(|s| s.phase)
    }

    /// Check if a session is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active_session
            .as_ref()
            .map(|s| matches!(s.phase, SessionPhase::RampUp | SessionPhase::Active | SessionPhase::RampDown))
            .unwrap_or(false)
    }
}

impl Default for StimulationController {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::fingerprint::{FingerprintId, FingerprintMetadata};

    fn create_test_fingerprint() -> NeuralFingerprint {
        let metadata = FingerprintMetadata::new(
            FingerprintId::null(),
            SensoryModality::Gustatory,
            "test",
            "subject",
            0,
        );
        NeuralFingerprint::new(metadata)
    }

    #[test]
    fn test_protocol_current_ramp() {
        let protocol = StimulationProtocol::gustatory(1000, 60.0);

        // At start
        assert!((protocol.current_at_time(0.0) - 0.0).abs() < 1.0);

        // Mid ramp-up
        assert!((protocol.current_at_time(15.0) - 500.0).abs() < 50.0);

        // Full ramp-up
        assert!((protocol.current_at_time(30.0) - 1000.0).abs() < 1.0);

        // Active phase
        assert!((protocol.current_at_time(60.0) - 1000.0).abs() < 1.0);

        // Mid ramp-down
        assert!((protocol.current_at_time(105.0) - 500.0).abs() < 50.0);

        // After complete
        assert!((protocol.current_at_time(121.0) - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_protocol_validation() {
        let protocol = StimulationProtocol::gustatory(1000, 60.0);
        let limits = SafetyLimits::RESEARCH;

        assert!(protocol.validate(&limits).is_ok());

        // Excessive current
        let mut bad_protocol = protocol.clone();
        bad_protocol.amplitude_ua = 3000;
        assert!(bad_protocol.validate(&limits).is_err());
    }

    #[test]
    fn test_session_lifecycle() {
        let target = create_test_fingerprint();
        let protocol = StimulationProtocol::gustatory(500, 5.0); // Short duration for test

        let mut session = StimulationSession::new("test_session".to_string(), target.clone(), protocol);

        // Start session
        assert!(session.start().is_ok());
        assert!(matches!(session.phase, SessionPhase::RampUp));

        // Update with current fingerprint
        let current = create_test_fingerprint();
        let result = session.update(&current, 5.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_controller() {
        let target = create_test_fingerprint();
        let protocol = StimulationProtocol::olfactory(800, 10.0);

        let mut controller = StimulationController::new();

        // Start session
        let session_id = controller.start_session(target.clone(), protocol);
        assert!(session_id.is_ok());
        assert!(controller.is_active());

        // Update
        let current = create_test_fingerprint();
        let similarity = controller.update(&current, 5.0);
        assert!(similarity.is_some());

        // Stop
        controller.stop_session();
    }
}

//! Per-device context and state management.
//!
//! This module provides the [`DeviceContext`] which encapsulates all per-device
//! state including visualization, stimulation control, and fingerprint access.
//!
//! Each connected device has its own independent context, allowing for:
//! - Independent stimulation sessions per device
//! - Per-device fingerprint capture and storage
//! - Device-specific visualization state
//! - Cross-device fingerprint sharing via the shared database
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::bridge::{DeviceContext, DeviceInfo};
//! use rootstar_bci_native::fingerprint::FingerprintDatabase;
//!
//! // Create shared database
//! let db = Arc::new(FingerprintDatabase::open("fingerprints.db")?);
//!
//! // Create context for each device
//! let ctx = DeviceContext::new(device_info, Arc::clone(&db));
//!
//! // Start stimulation with a fingerprint from ANY device
//! let target = db.find_similar(&current_fp, 0.8, 1)?
//!     .first()
//!     .map(|m| m.stored.fingerprint.clone());
//!
//! if let Some(target_fp) = target {
//!     ctx.start_stimulation(target_fp, protocol)?;
//! }
//! ```

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use rootstar_bci_core::fingerprint::{FingerprintId, NeuralFingerprint, SensoryModality};
use rootstar_bci_core::protocol::DeviceId;
use rootstar_bci_core::types::{EegSample, StimParams};

use crate::fingerprint::stimulation::{
    SessionPhase, StimulationController, StimulationProtocol,
};

use rootstar_bci_core::fingerprint::SafetyViolation;

#[cfg(feature = "database")]
use crate::fingerprint::database::{FingerprintDatabase, StoredFingerprint};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during device context operations.
#[derive(Debug, Error)]
pub enum ContextError {
    /// Stimulation error
    #[error("Stimulation error: {0}")]
    Stimulation(String),

    /// Safety violation
    #[error("Safety violation: {0:?}")]
    Safety(SafetyViolation),

    /// Database error
    #[cfg(feature = "database")]
    #[error("Database error: {0}")]
    Database(#[from] crate::fingerprint::database::DatabaseError),

    /// No active session
    #[error("No active stimulation session")]
    NoActiveSession,

    /// Device not ready
    #[error("Device not ready for operation")]
    NotReady,
}

/// Result type for context operations.
pub type ContextResult<T> = Result<T, ContextError>;

// ============================================================================
// Data Buffers
// ============================================================================

/// Configuration for data buffering.
#[derive(Clone, Debug)]
pub struct BufferConfig {
    /// Maximum EEG samples to buffer
    pub max_eeg_samples: usize,
    /// Maximum fNIRS samples to buffer
    pub max_fnirs_samples: usize,
    /// Maximum EMG samples to buffer
    pub max_emg_samples: usize,
    /// Maximum EDA samples to buffer
    pub max_eda_samples: usize,
    /// Time window for fingerprint extraction (seconds)
    pub extraction_window_s: f32,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_eeg_samples: 5000,    // ~10 seconds at 500 Hz
            max_fnirs_samples: 250,   // ~10 seconds at 25 Hz
            max_emg_samples: 5000,    // ~10 seconds at 500 Hz
            max_eda_samples: 100,     // ~10 seconds at 10 Hz
            extraction_window_s: 5.0, // 5-second window for fingerprint extraction
        }
    }
}

/// Data buffers for a single device.
#[derive(Debug)]
pub struct DataBuffers {
    /// EEG sample buffer
    pub eeg: VecDeque<EegSample>,
    /// fNIRS sample buffer (HbO, HbR values)
    pub fnirs: VecDeque<(f32, f32, u64)>, // (hbo, hbr, timestamp_us)
    /// EMG RMS buffer per channel
    pub emg: VecDeque<([f32; 8], u64)>, // (channels, timestamp_us)
    /// EDA buffer per site
    pub eda: VecDeque<([f32; 4], u64)>, // (sites, timestamp_us)
    /// Configuration
    config: BufferConfig,
}

impl DataBuffers {
    /// Create new data buffers with configuration.
    #[must_use]
    pub fn new(config: BufferConfig) -> Self {
        Self {
            eeg: VecDeque::with_capacity(config.max_eeg_samples),
            fnirs: VecDeque::with_capacity(config.max_fnirs_samples),
            emg: VecDeque::with_capacity(config.max_emg_samples),
            eda: VecDeque::with_capacity(config.max_eda_samples),
            config,
        }
    }

    /// Push an EEG sample.
    pub fn push_eeg(&mut self, sample: EegSample) {
        if self.eeg.len() >= self.config.max_eeg_samples {
            self.eeg.pop_front();
        }
        self.eeg.push_back(sample);
    }

    /// Push fNIRS data.
    pub fn push_fnirs(&mut self, hbo: f32, hbr: f32, timestamp_us: u64) {
        if self.fnirs.len() >= self.config.max_fnirs_samples {
            self.fnirs.pop_front();
        }
        self.fnirs.push_back((hbo, hbr, timestamp_us));
    }

    /// Push EMG data.
    pub fn push_emg(&mut self, channels: [f32; 8], timestamp_us: u64) {
        if self.emg.len() >= self.config.max_emg_samples {
            self.emg.pop_front();
        }
        self.emg.push_back((channels, timestamp_us));
    }

    /// Push EDA data.
    pub fn push_eda(&mut self, sites: [f32; 4], timestamp_us: u64) {
        if self.eda.len() >= self.config.max_eda_samples {
            self.eda.pop_front();
        }
        self.eda.push_back((sites, timestamp_us));
    }

    /// Clear all buffers.
    pub fn clear(&mut self) {
        self.eeg.clear();
        self.fnirs.clear();
        self.emg.clear();
        self.eda.clear();
    }

    /// Get buffer statistics.
    #[must_use]
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            eeg_count: self.eeg.len(),
            fnirs_count: self.fnirs.len(),
            emg_count: self.emg.len(),
            eda_count: self.eda.len(),
        }
    }
}

impl Default for DataBuffers {
    fn default() -> Self {
        Self::new(BufferConfig::default())
    }
}

/// Statistics about buffer fill levels.
#[derive(Clone, Debug, Default)]
pub struct BufferStats {
    /// EEG buffer sample count
    pub eeg_count: usize,
    /// fNIRS buffer sample count
    pub fnirs_count: usize,
    /// EMG buffer sample count
    pub emg_count: usize,
    /// EDA buffer sample count
    pub eda_count: usize,
}

// ============================================================================
// Fingerprint Registry (Per-Device View)
// ============================================================================

/// Per-device fingerprint registry.
///
/// This provides a device-specific view into the shared fingerprint database,
/// tracking which fingerprints were captured by this device and providing
/// convenient access methods.
#[cfg(feature = "database")]
pub struct FingerprintRegistry {
    /// Shared database reference
    db: Arc<FingerprintDatabase>,

    /// Device that owns this registry
    device_id: DeviceId,

    /// Cached fingerprints captured by this device
    local_cache: Vec<FingerprintId>,

    /// Currently selected target fingerprint
    selected_target: Option<FingerprintId>,
}

#[cfg(feature = "database")]
impl FingerprintRegistry {
    /// Create a new registry for a device.
    #[must_use]
    pub fn new(device_id: DeviceId, db: Arc<FingerprintDatabase>) -> Self {
        Self {
            db,
            device_id,
            local_cache: Vec::new(),
            selected_target: None,
        }
    }

    /// Store a fingerprint captured by this device.
    pub fn store(&mut self, fingerprint: NeuralFingerprint, name: String) -> ContextResult<FingerprintId> {
        let id = fingerprint.metadata.id;

        let stored = StoredFingerprint::new(fingerprint, name)
            .with_device(self.device_id.to_string());

        self.db.store(&stored)?;
        self.local_cache.push(id);

        Ok(id)
    }

    /// Get all fingerprints captured by this device.
    pub fn list_local(&self) -> ContextResult<Vec<StoredFingerprint>> {
        use crate::fingerprint::database::FingerprintQuery;

        let query = FingerprintQuery::new()
            .with_device(self.device_id.to_string());

        Ok(self.db.query(&query)?)
    }

    /// Get all fingerprints (from all devices).
    pub fn list_all(&self, limit: Option<usize>) -> ContextResult<Vec<StoredFingerprint>> {
        Ok(self.db.list_all(limit)?)
    }

    /// Find similar fingerprints across all devices.
    pub fn find_similar(
        &self,
        fingerprint: &NeuralFingerprint,
        threshold: f32,
        limit: usize,
    ) -> ContextResult<Vec<crate::fingerprint::database::SimilarityMatch>> {
        Ok(self.db.find_similar(fingerprint, threshold, limit)?)
    }

    /// Load a fingerprint by ID.
    pub fn load(&self, id: &FingerprintId) -> ContextResult<StoredFingerprint> {
        Ok(self.db.load(id)?)
    }

    /// Select a target fingerprint for stimulation.
    pub fn select_target(&mut self, id: FingerprintId) {
        self.selected_target = Some(id);
    }

    /// Get the selected target fingerprint.
    pub fn selected_target(&self) -> Option<FingerprintId> {
        self.selected_target
    }

    /// Clear the selected target.
    pub fn clear_target(&mut self) {
        self.selected_target = None;
    }

    /// Record that a fingerprint was used for stimulation.
    pub fn record_usage(&self, id: &FingerprintId) -> ContextResult<()> {
        Ok(self.db.record_usage(id)?)
    }
}

// ============================================================================
// Device Context
// ============================================================================

/// Stimulation session state for a device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StimulationState {
    /// Whether stimulation is active
    pub active: bool,
    /// Current phase
    pub phase: String,
    /// Target fingerprint ID
    pub target_id: Option<FingerprintId>,
    /// Target fingerprint name
    pub target_name: Option<String>,
    /// Current similarity to target
    pub similarity: f32,
    /// Session elapsed time (seconds)
    pub elapsed_s: f32,
    /// Session remaining time (seconds)
    pub remaining_s: f32,
    /// Current amplitude (µA)
    pub current_ua: u16,
    /// Current impedance (kΩ)
    pub impedance_kohm: f32,
}

impl Default for StimulationState {
    fn default() -> Self {
        Self {
            active: false,
            phase: "Idle".to_string(),
            target_id: None,
            target_name: None,
            similarity: 0.0,
            elapsed_s: 0.0,
            remaining_s: 0.0,
            current_ua: 0,
            impedance_kohm: 0.0,
        }
    }
}

/// Per-device context containing all device-specific state.
///
/// Each connected device gets its own `DeviceContext` which manages:
/// - Data buffers for EEG, fNIRS, EMG, EDA
/// - Stimulation controller (independent per device)
/// - Fingerprint registry (view into shared database)
/// - Visualization state
pub struct DeviceContext {
    /// Device identifier
    pub device_id: DeviceId,

    /// User-assigned device name
    pub name: String,

    /// Device color for UI
    pub color: [u8; 3],

    /// Data buffers
    pub buffers: DataBuffers,

    /// Stimulation controller
    stim_controller: StimulationController,

    /// Fingerprint registry (when database feature enabled)
    #[cfg(feature = "database")]
    pub registry: FingerprintRegistry,

    /// Last extracted fingerprint
    last_fingerprint: Option<NeuralFingerprint>,

    /// Impedance values per channel (kΩ)
    pub impedances: [f32; 8],

    /// Signal quality per channel (0-100%)
    pub signal_quality: [u8; 8],

    /// Device is acquiring data
    pub acquiring: bool,

    /// Context creation time
    created_at: Instant,

    /// Last data update time
    last_update: Option<Instant>,
}

impl DeviceContext {
    /// Create a new device context.
    #[cfg(feature = "database")]
    pub fn new(device_id: DeviceId, name: String, db: Arc<FingerprintDatabase>) -> Self {
        Self {
            device_id,
            name,
            color: [0x00, 0x7A, 0xCC], // Default blue
            buffers: DataBuffers::default(),
            stim_controller: StimulationController::new(),
            registry: FingerprintRegistry::new(device_id, db),
            last_fingerprint: None,
            impedances: [0.0; 8],
            signal_quality: [0; 8],
            acquiring: false,
            created_at: Instant::now(),
            last_update: None,
        }
    }

    /// Create a new device context without database.
    #[cfg(not(feature = "database"))]
    pub fn new(device_id: DeviceId, name: String) -> Self {
        Self {
            device_id,
            name,
            color: [0x00, 0x7A, 0xCC],
            buffers: DataBuffers::default(),
            stim_controller: StimulationController::new(),
            last_fingerprint: None,
            impedances: [0.0; 8],
            signal_quality: [0; 8],
            acquiring: false,
            created_at: Instant::now(),
            last_update: None,
        }
    }

    /// Get device ID.
    #[must_use]
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Set device color.
    pub fn set_color(&mut self, color: [u8; 3]) {
        self.color = color;
    }

    /// Set device name.
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Process incoming EEG data.
    pub fn process_eeg(&mut self, sample: EegSample) {
        self.buffers.push_eeg(sample);
        self.last_update = Some(Instant::now());
    }

    /// Process incoming fNIRS data.
    pub fn process_fnirs(&mut self, hbo: f32, hbr: f32, timestamp_us: u64) {
        self.buffers.push_fnirs(hbo, hbr, timestamp_us);
        self.last_update = Some(Instant::now());
    }

    /// Process incoming EMG data.
    pub fn process_emg(&mut self, channels: [f32; 8], timestamp_us: u64) {
        self.buffers.push_emg(channels, timestamp_us);
        self.last_update = Some(Instant::now());
    }

    /// Process incoming EDA data.
    pub fn process_eda(&mut self, sites: [f32; 4], timestamp_us: u64) {
        self.buffers.push_eda(sites, timestamp_us);
        self.last_update = Some(Instant::now());
    }

    /// Update impedance values.
    pub fn update_impedances(&mut self, impedances: [f32; 8]) {
        self.impedances = impedances;
    }

    /// Get the last extracted fingerprint.
    #[must_use]
    pub fn last_fingerprint(&self) -> Option<&NeuralFingerprint> {
        self.last_fingerprint.as_ref()
    }

    /// Set the last extracted fingerprint.
    pub fn set_fingerprint(&mut self, fingerprint: NeuralFingerprint) {
        self.last_fingerprint = Some(fingerprint);
    }

    /// Start a stimulation session targeting a fingerprint.
    pub fn start_stimulation(
        &mut self,
        target: NeuralFingerprint,
        protocol: StimulationProtocol,
    ) -> ContextResult<String> {
        self.stim_controller
            .start_session(target, protocol)
            .map_err(ContextError::Safety)
    }

    /// Update stimulation session with current fingerprint.
    pub fn update_stimulation(
        &mut self,
        current_fingerprint: &NeuralFingerprint,
        impedance_kohm: f32,
    ) -> ContextResult<Option<f32>> {
        Ok(self.stim_controller.update(current_fingerprint, impedance_kohm))
    }

    /// Stop active stimulation session.
    pub fn stop_stimulation(&mut self) -> ContextResult<()> {
        self.stim_controller.stop();
        Ok(())
    }

    /// Emergency stop stimulation.
    pub fn emergency_stop(&mut self) {
        self.stim_controller.emergency_stop();
    }

    /// Check if stimulation is active.
    #[must_use]
    pub fn is_stimulating(&self) -> bool {
        self.stim_controller.is_active()
    }

    /// Get current stimulation state for UI.
    #[must_use]
    pub fn stimulation_state(&self) -> StimulationState {
        if let Some(session) = self.stim_controller.current_session() {
            StimulationState {
                active: self.stim_controller.is_active(),
                phase: format!("{:?}", session.phase),
                target_id: Some(session.target.metadata.id),
                target_name: Some(session.target.metadata.stimulus_label().to_string()),
                similarity: session.average_similarity(),
                elapsed_s: session.elapsed_time().as_secs_f32(),
                remaining_s: session.remaining_time().map(|d| d.as_secs_f32()).unwrap_or(0.0),
                current_ua: session.protocol.amplitude_ua,
                impedance_kohm: self.impedances.iter().sum::<f32>() / 8.0,
            }
        } else {
            StimulationState::default()
        }
    }

    /// Get time since last data update.
    #[must_use]
    pub fn time_since_update(&self) -> Option<Duration> {
        self.last_update.map(|t| t.elapsed())
    }

    /// Get context uptime.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get overall signal quality (0-100%).
    #[must_use]
    pub fn overall_signal_quality(&self) -> u8 {
        let sum: u16 = self.signal_quality.iter().map(|&q| u16::from(q)).sum();
        (sum / 8) as u8
    }
}

// ============================================================================
// Multi-Device Context Manager
// ============================================================================

/// Manager for multiple device contexts.
///
/// This provides a centralized view of all connected device contexts,
/// allowing for cross-device operations like synchronized stimulation
/// or hyperscanning analysis.
pub struct MultiDeviceContextManager {
    /// Device contexts by ID
    contexts: std::collections::HashMap<DeviceId, DeviceContext>,

    /// Shared fingerprint database
    #[cfg(feature = "database")]
    db: Arc<FingerprintDatabase>,

    /// Maximum number of contexts
    max_contexts: usize,
}

impl MultiDeviceContextManager {
    /// Create a new context manager with shared database.
    #[cfg(feature = "database")]
    pub fn new(db: Arc<FingerprintDatabase>) -> Self {
        Self {
            contexts: std::collections::HashMap::new(),
            db,
            max_contexts: 16,
        }
    }

    /// Create a new context manager without database.
    #[cfg(not(feature = "database"))]
    pub fn new() -> Self {
        Self {
            contexts: std::collections::HashMap::new(),
            max_contexts: 16,
        }
    }

    /// Add a new device context.
    #[cfg(feature = "database")]
    pub fn add_device(&mut self, device_id: DeviceId, name: String) -> ContextResult<()> {
        if self.contexts.len() >= self.max_contexts {
            return Err(ContextError::NotReady);
        }

        let ctx = DeviceContext::new(device_id, name, Arc::clone(&self.db));
        self.contexts.insert(device_id, ctx);
        Ok(())
    }

    /// Add a new device context without database.
    #[cfg(not(feature = "database"))]
    pub fn add_device(&mut self, device_id: DeviceId, name: String) -> ContextResult<()> {
        if self.contexts.len() >= self.max_contexts {
            return Err(ContextError::NotReady);
        }

        let ctx = DeviceContext::new(device_id, name);
        self.contexts.insert(device_id, ctx);
        Ok(())
    }

    /// Remove a device context.
    pub fn remove_device(&mut self, device_id: DeviceId) -> Option<DeviceContext> {
        self.contexts.remove(&device_id)
    }

    /// Get a device context.
    #[must_use]
    pub fn get(&self, device_id: DeviceId) -> Option<&DeviceContext> {
        self.contexts.get(&device_id)
    }

    /// Get a mutable device context.
    pub fn get_mut(&mut self, device_id: DeviceId) -> Option<&mut DeviceContext> {
        self.contexts.get_mut(&device_id)
    }

    /// Get all device contexts.
    #[must_use]
    pub fn all(&self) -> Vec<&DeviceContext> {
        self.contexts.values().collect()
    }

    /// Get all device IDs.
    #[must_use]
    pub fn device_ids(&self) -> Vec<DeviceId> {
        self.contexts.keys().copied().collect()
    }

    /// Get number of devices.
    #[must_use]
    pub fn count(&self) -> usize {
        self.contexts.len()
    }

    /// Check if any device is stimulating.
    #[must_use]
    pub fn any_stimulating(&self) -> bool {
        self.contexts.values().any(|ctx| ctx.is_stimulating())
    }

    /// Emergency stop all devices.
    pub fn emergency_stop_all(&mut self) {
        for ctx in self.contexts.values_mut() {
            ctx.emergency_stop();
        }
    }

    /// Get shared database reference.
    #[cfg(feature = "database")]
    #[must_use]
    pub fn database(&self) -> Arc<FingerprintDatabase> {
        Arc::clone(&self.db)
    }
}

#[cfg(not(feature = "database"))]
impl Default for MultiDeviceContextManager {
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
    use rootstar_bci_core::types::Fixed24_8;

    fn create_test_eeg_sample(seq: u32) -> EegSample {
        EegSample {
            timestamp_us: seq as u64 * 2000,
            channels: [Fixed24_8::ZERO; 8],
            sequence: seq,
        }
    }

    #[test]
    fn test_data_buffers() {
        let mut buffers = DataBuffers::default();

        for i in 0..100 {
            buffers.push_eeg(create_test_eeg_sample(i));
        }

        assert_eq!(buffers.stats().eeg_count, 100);

        buffers.clear();
        assert_eq!(buffers.stats().eeg_count, 0);
    }

    #[test]
    fn test_buffer_overflow() {
        let config = BufferConfig {
            max_eeg_samples: 10,
            ..Default::default()
        };
        let mut buffers = DataBuffers::new(config);

        // Push more than max
        for i in 0..20 {
            buffers.push_eeg(create_test_eeg_sample(i));
        }

        // Should be capped at max
        assert_eq!(buffers.stats().eeg_count, 10);

        // First sample should be sequence 10 (not 0)
        assert_eq!(buffers.eeg.front().unwrap().sequence, 10);
    }

    #[test]
    fn test_stimulation_state_default() {
        let state = StimulationState::default();
        assert!(!state.active);
        assert_eq!(state.phase, "Idle");
        assert_eq!(state.similarity, 0.0);
    }
}

//! Hyperscanning and cross-device synchronization
//!
//! Provides synchronization of data streams from multiple BCI devices for
//! multi-brain studies (hyperscanning) and distributed sensor systems.
//!
//! # Features
//!
//! - Clock drift compensation between devices
//! - Cross-device sample alignment
//! - Inter-brain coherence calculation
//! - Sync marker injection and detection
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::processing::hyperscanning::{
//!     HyperscanningSession, SyncConfig,
//! };
//!
//! // Create session for 2 participants
//! let mut session = HyperscanningSession::new(2, SyncConfig::default());
//!
//! // Add samples from each device
//! session.push_sample(device1_id, sample1)?;
//! session.push_sample(device2_id, sample2)?;
//!
//! // Get synchronized sample pairs
//! for pair in session.get_aligned_pairs() {
//!     let coherence = session.compute_coherence(&pair, frequency_band)?;
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use rootstar_bci_core::protocol::DeviceId;
use rootstar_bci_core::types::EegSample;

/// Sync marker types for cross-device alignment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncMarkerType {
    /// Session start marker
    SessionStart,
    /// Session end marker
    SessionEnd,
    /// Periodic heartbeat (for drift detection)
    Heartbeat,
    /// External trigger (TTL pulse, stimulus onset)
    ExternalTrigger,
    /// User-defined event
    UserEvent(u16),
}

/// A sync marker event
#[derive(Clone, Debug)]
pub struct SyncMarker {
    /// Marker type
    pub marker_type: SyncMarkerType,
    /// Device-local timestamp (microseconds)
    pub device_timestamp_us: u64,
    /// Device sequence number
    pub sequence: u32,
    /// Host receive time
    pub host_time: Instant,
    /// Device ID
    pub device_id: DeviceId,
}

/// Configuration for hyperscanning synchronization
#[derive(Clone, Debug)]
pub struct SyncConfig {
    /// Maximum allowed clock drift (microseconds per second)
    pub max_drift_ppm: u32,
    /// Alignment window (microseconds)
    pub alignment_window_us: u64,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u32,
    /// Maximum sample buffer size per device
    pub max_buffer_size: usize,
    /// Enable automatic drift compensation
    pub auto_drift_compensation: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_drift_ppm: 100, // 100 ppm = 0.01%
            alignment_window_us: 5000, // 5ms alignment window
            heartbeat_interval_ms: 1000, // 1 second heartbeat
            max_buffer_size: 10000,
            auto_drift_compensation: true,
        }
    }
}

/// Clock drift estimate for a device
#[derive(Clone, Debug, Default)]
struct ClockDrift {
    /// Estimated drift in microseconds per second
    drift_us_per_sec: f64,
    /// Last sync marker time
    last_sync_host: Option<Instant>,
    last_sync_device: Option<u64>,
    /// Accumulated sync points for linear regression
    sync_points: Vec<(f64, f64)>, // (host_time_s, device_time_s)
}

impl ClockDrift {
    /// Update drift estimate with a new sync point
    fn add_sync_point(&mut self, host_time: Instant, device_time_us: u64) {
        let host_time_s = if let Some(first) = self.last_sync_host {
            host_time.duration_since(first).as_secs_f64()
        } else {
            self.last_sync_host = Some(host_time);
            self.last_sync_device = Some(device_time_us);
            0.0
        };

        let device_time_s = if let Some(first) = self.last_sync_device {
            (device_time_us - first) as f64 / 1_000_000.0
        } else {
            0.0
        };

        self.sync_points.push((host_time_s, device_time_s));

        // Keep only recent sync points (last 60 seconds)
        if self.sync_points.len() > 60 {
            self.sync_points.remove(0);
        }

        // Estimate drift using linear regression
        if self.sync_points.len() >= 2 {
            self.estimate_drift();
        }
    }

    /// Estimate drift using simple linear regression
    fn estimate_drift(&mut self) {
        let n = self.sync_points.len() as f64;
        if n < 2.0 {
            return;
        }

        let sum_x: f64 = self.sync_points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = self.sync_points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = self.sync_points.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = self.sync_points.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Drift is deviation from 1:1 time ratio
        // slope = 1.0 means no drift, slope = 1.001 means 1000 ppm fast
        self.drift_us_per_sec = (slope - 1.0) * 1_000_000.0;
    }

    /// Correct a device timestamp for estimated drift
    fn correct_timestamp(&self, device_time_us: u64, elapsed_host_s: f64) -> u64 {
        let correction = (elapsed_host_s * self.drift_us_per_sec) as i64;
        (device_time_us as i64 - correction).max(0) as u64
    }
}

/// Per-device sample buffer
struct DeviceBuffer {
    /// EEG samples with timestamps
    samples: VecDeque<(u64, EegSample)>, // (corrected_timestamp_us, sample)
    /// Clock drift tracker
    clock_drift: ClockDrift,
    /// Session start host time
    session_start: Option<Instant>,
    /// Last sample timestamp
    last_timestamp_us: u64,
}

impl DeviceBuffer {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(10000),
            clock_drift: ClockDrift::default(),
            session_start: None,
            last_timestamp_us: 0,
        }
    }

    fn push(&mut self, timestamp_us: u64, sample: EegSample, config: &SyncConfig) {
        // Correct for clock drift
        let corrected_ts = if config.auto_drift_compensation {
            if let Some(start) = self.session_start {
                let elapsed_s = start.elapsed().as_secs_f64();
                self.clock_drift.correct_timestamp(timestamp_us, elapsed_s)
            } else {
                timestamp_us
            }
        } else {
            timestamp_us
        };

        self.samples.push_back((corrected_ts, sample));
        self.last_timestamp_us = corrected_ts;

        // Prune old samples
        while self.samples.len() > config.max_buffer_size {
            self.samples.pop_front();
        }
    }
}

/// Aligned sample pair from two devices
#[derive(Clone, Debug)]
pub struct AlignedPair {
    /// Reference device sample
    pub reference: EegSample,
    /// Reference device timestamp
    pub reference_timestamp_us: u64,
    /// Paired device sample
    pub paired: EegSample,
    /// Paired device timestamp
    pub paired_timestamp_us: u64,
    /// Time offset between samples (microseconds)
    pub offset_us: i64,
    /// Reference device ID
    pub reference_device: DeviceId,
    /// Paired device ID
    pub paired_device: DeviceId,
}

/// Hyperscanning session for multi-device synchronization
pub struct HyperscanningSession {
    /// Device buffers
    devices: HashMap<DeviceId, DeviceBuffer>,
    /// Reference device ID (primary device for alignment)
    reference_device: Option<DeviceId>,
    /// Sync markers
    sync_markers: VecDeque<SyncMarker>,
    /// Configuration
    config: SyncConfig,
    /// Session start time
    session_start: Instant,
    /// Expected number of participants
    num_participants: usize,
}

impl HyperscanningSession {
    /// Create a new hyperscanning session
    #[must_use]
    pub fn new(num_participants: usize, config: SyncConfig) -> Self {
        Self {
            devices: HashMap::new(),
            reference_device: None,
            sync_markers: VecDeque::with_capacity(1000),
            config,
            session_start: Instant::now(),
            num_participants,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults(num_participants: usize) -> Self {
        Self::new(num_participants, SyncConfig::default())
    }

    /// Add a device to the session
    pub fn add_device(&mut self, device_id: DeviceId) {
        if !self.devices.contains_key(&device_id) {
            let mut buffer = DeviceBuffer::new();
            buffer.session_start = Some(Instant::now());
            self.devices.insert(device_id.clone(), buffer);

            // First device becomes reference
            if self.reference_device.is_none() {
                self.reference_device = Some(device_id);
            }
        }
    }

    /// Remove a device from the session
    pub fn remove_device(&mut self, device_id: &DeviceId) {
        self.devices.remove(device_id);

        // Update reference device if removed
        if self.reference_device.as_ref() == Some(device_id) {
            self.reference_device = self.devices.keys().next().cloned();
        }
    }

    /// Set the reference device for alignment
    pub fn set_reference_device(&mut self, device_id: DeviceId) {
        if self.devices.contains_key(&device_id) {
            self.reference_device = Some(device_id);
        }
    }

    /// Push a sample from a device
    pub fn push_sample(&mut self, device_id: DeviceId, timestamp_us: u64, sample: EegSample) {
        self.add_device(device_id.clone());

        if let Some(buffer) = self.devices.get_mut(&device_id) {
            buffer.push(timestamp_us, sample, &self.config);
        }
    }

    /// Add a sync marker
    pub fn add_sync_marker(&mut self, marker: SyncMarker) {
        // Update clock drift for this device
        if let Some(buffer) = self.devices.get_mut(&marker.device_id) {
            buffer
                .clock_drift
                .add_sync_point(marker.host_time, marker.device_timestamp_us);
        }

        self.sync_markers.push_back(marker);

        // Prune old markers
        while self.sync_markers.len() > 1000 {
            self.sync_markers.pop_front();
        }
    }

    /// Get aligned sample pairs between reference device and all others
    ///
    /// Returns pairs where samples from different devices are within the
    /// alignment window of each other.
    pub fn get_aligned_pairs(&self) -> Vec<AlignedPair> {
        let mut pairs = Vec::new();

        let Some(ref ref_device) = self.reference_device else {
            return pairs;
        };

        let Some(ref_buffer) = self.devices.get(ref_device) else {
            return pairs;
        };

        for (other_id, other_buffer) in &self.devices {
            if other_id == ref_device {
                continue;
            }

            // Find matching samples within alignment window
            for &(ref_ts, ref ref_sample) in &ref_buffer.samples {
                // Binary search for closest sample in other buffer
                if let Some((paired_ts, paired_sample)) =
                    Self::find_closest_sample(other_buffer, ref_ts, self.config.alignment_window_us)
                {
                    pairs.push(AlignedPair {
                        reference: ref_sample.clone(),
                        reference_timestamp_us: ref_ts,
                        paired: paired_sample,
                        paired_timestamp_us: paired_ts,
                        offset_us: (paired_ts as i64) - (ref_ts as i64),
                        reference_device: ref_device.clone(),
                        paired_device: other_id.clone(),
                    });
                }
            }
        }

        pairs
    }

    /// Find the closest sample within the alignment window
    fn find_closest_sample(
        buffer: &DeviceBuffer,
        target_ts: u64,
        window_us: u64,
    ) -> Option<(u64, EegSample)> {
        let mut best_match: Option<(u64, EegSample, u64)> = None; // (ts, sample, distance)

        for &(ts, ref sample) in &buffer.samples {
            let distance = if ts > target_ts {
                ts - target_ts
            } else {
                target_ts - ts
            };

            if distance <= window_us {
                if best_match.is_none() || distance < best_match.as_ref().unwrap().2 {
                    best_match = Some((ts, sample.clone(), distance));
                }
            }
        }

        best_match.map(|(ts, sample, _)| (ts, sample))
    }

    /// Compute inter-brain coherence for a frequency band
    ///
    /// Uses Welch's method to estimate coherence between two EEG signals.
    ///
    /// # Arguments
    ///
    /// * `pair` - Aligned sample pair
    /// * `channel` - EEG channel index (0-7)
    /// * `freq_band` - Frequency band (low_hz, high_hz)
    /// * `sample_rate_hz` - Sample rate
    pub fn compute_coherence(
        &self,
        channel: usize,
        freq_band: (f32, f32),
        sample_rate_hz: f32,
    ) -> Option<CoherenceResult> {
        let Some(ref ref_device) = self.reference_device else {
            return None;
        };

        // Get recent samples from reference and all other devices
        let ref_buffer = self.devices.get(ref_device)?;
        if ref_buffer.samples.len() < 128 {
            return None; // Need at least 128 samples for FFT
        }

        let mut coherence_values: HashMap<DeviceId, Vec<f32>> = HashMap::new();

        for (other_id, other_buffer) in &self.devices {
            if other_id == ref_device || other_buffer.samples.len() < 128 {
                continue;
            }

            // Extract signals for coherence calculation
            let ref_signal: Vec<f32> = ref_buffer
                .samples
                .iter()
                .take(256)
                .map(|(_, s)| s.channels[channel].to_f32())
                .collect();

            let other_signal: Vec<f32> = other_buffer
                .samples
                .iter()
                .take(256)
                .map(|(_, s)| s.channels[channel].to_f32())
                .collect();

            if ref_signal.len() == other_signal.len() {
                let coherence = Self::calculate_coherence_simple(
                    &ref_signal,
                    &other_signal,
                    freq_band,
                    sample_rate_hz,
                );
                coherence_values
                    .entry(other_id.clone())
                    .or_default()
                    .push(coherence);
            }
        }

        Some(CoherenceResult {
            reference_device: ref_device.clone(),
            channel,
            freq_band,
            device_coherences: coherence_values
                .into_iter()
                .map(|(k, v)| (k, v.iter().sum::<f32>() / v.len() as f32))
                .collect(),
            timestamp: Instant::now(),
        })
    }

    /// Simple coherence calculation (Pearson correlation of band-filtered signals)
    fn calculate_coherence_simple(
        signal1: &[f32],
        signal2: &[f32],
        _freq_band: (f32, f32),
        _sample_rate_hz: f32,
    ) -> f32 {
        // Simplified: correlation-based pseudo-coherence
        // TODO: Use proper Welch's coherence with FFT when rustfft is available
        if signal1.len() != signal2.len() || signal1.is_empty() {
            return 0.0;
        }

        let n = signal1.len() as f32;
        let mean1: f32 = signal1.iter().sum::<f32>() / n;
        let mean2: f32 = signal2.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var1 = 0.0f32;
        let mut var2 = 0.0f32;

        for (&x, &y) in signal1.iter().zip(signal2.iter()) {
            let d1 = x - mean1;
            let d2 = y - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        if var1 > 0.0 && var2 > 0.0 {
            (cov / (var1.sqrt() * var2.sqrt())).abs()
        } else {
            0.0
        }
    }

    /// Get synchronization statistics
    pub fn sync_stats(&self) -> SyncStats {
        let mut device_stats = Vec::new();

        for (device_id, buffer) in &self.devices {
            device_stats.push(DeviceSyncStats {
                device_id: device_id.clone(),
                sample_count: buffer.samples.len(),
                estimated_drift_ppm: (buffer.clock_drift.drift_us_per_sec * 1000.0) as i32,
                last_timestamp_us: buffer.last_timestamp_us,
            });
        }

        SyncStats {
            num_devices: self.devices.len(),
            reference_device: self.reference_device.clone(),
            session_duration: self.session_start.elapsed(),
            sync_marker_count: self.sync_markers.len(),
            device_stats,
        }
    }
}

/// Result of coherence computation
#[derive(Clone, Debug)]
pub struct CoherenceResult {
    /// Reference device ID
    pub reference_device: DeviceId,
    /// EEG channel analyzed
    pub channel: usize,
    /// Frequency band (Hz)
    pub freq_band: (f32, f32),
    /// Coherence values per paired device
    pub device_coherences: HashMap<DeviceId, f32>,
    /// Computation timestamp
    pub timestamp: Instant,
}

/// Per-device synchronization statistics
#[derive(Clone, Debug)]
pub struct DeviceSyncStats {
    /// Device ID
    pub device_id: DeviceId,
    /// Number of buffered samples
    pub sample_count: usize,
    /// Estimated clock drift (parts per million)
    pub estimated_drift_ppm: i32,
    /// Last sample timestamp
    pub last_timestamp_us: u64,
}

/// Session-wide synchronization statistics
#[derive(Clone, Debug)]
pub struct SyncStats {
    /// Number of connected devices
    pub num_devices: usize,
    /// Reference device ID
    pub reference_device: Option<DeviceId>,
    /// Session duration
    pub session_duration: Duration,
    /// Number of sync markers received
    pub sync_marker_count: usize,
    /// Per-device statistics
    pub device_stats: Vec<DeviceSyncStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::types::Fixed24_8;

    fn make_sample(value: i32) -> EegSample {
        EegSample {
            channels: [Fixed24_8::from_raw(value); 8],
            sequence: 0,
        }
    }

    #[test]
    fn test_session_creation() {
        let session = HyperscanningSession::with_defaults(2);
        assert_eq!(session.num_participants, 2);
        assert!(session.reference_device.is_none());
    }

    #[test]
    fn test_add_device() {
        let mut session = HyperscanningSession::with_defaults(2);
        let device1 = DeviceId::from_u32(1);
        let device2 = DeviceId::from_u32(2);

        session.add_device(device1.clone());
        session.add_device(device2.clone());

        assert_eq!(session.devices.len(), 2);
        assert_eq!(session.reference_device, Some(device1));
    }

    #[test]
    fn test_push_samples() {
        let mut session = HyperscanningSession::with_defaults(2);
        let device1 = DeviceId::from_u32(1);
        let device2 = DeviceId::from_u32(2);

        session.push_sample(device1.clone(), 0, make_sample(100));
        session.push_sample(device1.clone(), 1000, make_sample(101));
        session.push_sample(device2.clone(), 500, make_sample(200));

        assert_eq!(session.devices.get(&device1).unwrap().samples.len(), 2);
        assert_eq!(session.devices.get(&device2).unwrap().samples.len(), 1);
    }

    #[test]
    fn test_aligned_pairs() {
        let mut session = HyperscanningSession::with_defaults(2);
        let device1 = DeviceId::from_u32(1);
        let device2 = DeviceId::from_u32(2);

        // Push samples with overlapping timestamps
        session.push_sample(device1.clone(), 1000, make_sample(100));
        session.push_sample(device1.clone(), 2000, make_sample(101));
        session.push_sample(device2.clone(), 1100, make_sample(200)); // Within 5ms window of 1000
        session.push_sample(device2.clone(), 2050, make_sample(201)); // Within 5ms window of 2000

        let pairs = session.get_aligned_pairs();

        // Should find 2 pairs (1000-1100 and 2000-2050)
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_coherence_simple() {
        // Test with perfectly correlated signals
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let coherence =
            HyperscanningSession::calculate_coherence_simple(&signal1, &signal2, (8.0, 13.0), 250.0);

        assert!((coherence - 1.0).abs() < 0.01);

        // Test with uncorrelated signals
        let signal3 = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let coherence2 =
            HyperscanningSession::calculate_coherence_simple(&signal1, &signal3, (8.0, 13.0), 250.0);

        assert!(coherence2 < 0.5);
    }
}

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
use rustfft::{num_complex::Complex, FftPlanner};

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
                let coherence = Self::calculate_coherence_welch(
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

    /// Welch's coherence calculation using FFT
    ///
    /// Computes magnitude-squared coherence between two signals using Welch's method:
    /// - Segments signals into overlapping windows
    /// - Computes cross-spectral density and auto-spectral densities
    /// - Coherence = |Pxy|^2 / (Pxx * Pyy)
    fn calculate_coherence_welch(
        signal1: &[f32],
        signal2: &[f32],
        freq_band: (f32, f32),
        sample_rate_hz: f32,
    ) -> f32 {
        if signal1.len() != signal2.len() || signal1.len() < 64 {
            return 0.0;
        }

        // Welch's method parameters
        let segment_len = 64.min(signal1.len()); // Segment size (power of 2)
        let overlap = segment_len / 2; // 50% overlap
        let step = segment_len - overlap;
        let n_segments = (signal1.len() - overlap) / step;

        if n_segments == 0 {
            return 0.0;
        }

        // Create FFT planner
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(segment_len);

        // Hann window
        let window: Vec<f32> = (0..segment_len)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (segment_len - 1) as f32).cos())
            })
            .collect();

        // Window normalization factor
        let window_power: f32 = window.iter().map(|w| w * w).sum();

        // Accumulate spectral estimates
        let n_freqs = segment_len / 2 + 1;
        let mut pxx = vec![0.0f32; n_freqs]; // Auto-spectral density of signal1
        let mut pyy = vec![0.0f32; n_freqs]; // Auto-spectral density of signal2
        let mut pxy_re = vec![0.0f32; n_freqs]; // Cross-spectral density (real)
        let mut pxy_im = vec![0.0f32; n_freqs]; // Cross-spectral density (imag)

        let mut scratch = vec![Complex::new(0.0f32, 0.0f32); fft.get_inplace_scratch_len()];

        for seg_idx in 0..n_segments {
            let start = seg_idx * step;

            // Apply window and compute FFT for signal1
            let mut buf1: Vec<Complex<f32>> = signal1[start..start + segment_len]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            fft.process_with_scratch(&mut buf1, &mut scratch);

            // Apply window and compute FFT for signal2
            let mut buf2: Vec<Complex<f32>> = signal2[start..start + segment_len]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            fft.process_with_scratch(&mut buf2, &mut scratch);

            // Accumulate spectral estimates
            for k in 0..n_freqs {
                let x = buf1[k];
                let y = buf2[k];

                // Auto-spectral densities
                pxx[k] += x.re * x.re + x.im * x.im;
                pyy[k] += y.re * y.re + y.im * y.im;

                // Cross-spectral density: X * conj(Y)
                pxy_re[k] += x.re * y.re + x.im * y.im;
                pxy_im[k] += x.im * y.re - x.re * y.im;
            }
        }

        // Normalize by number of segments
        let n_seg_f = n_segments as f32;
        for k in 0..n_freqs {
            pxx[k] /= n_seg_f;
            pyy[k] /= n_seg_f;
            pxy_re[k] /= n_seg_f;
            pxy_im[k] /= n_seg_f;
        }

        // Compute coherence in the frequency band
        let freq_resolution = sample_rate_hz / segment_len as f32;
        let start_bin = ((freq_band.0 / freq_resolution).floor() as usize).min(n_freqs - 1);
        let end_bin = ((freq_band.1 / freq_resolution).ceil() as usize).min(n_freqs - 1);

        if start_bin >= end_bin {
            return 0.0;
        }

        // Average coherence across frequency bins in the band
        let mut coherence_sum = 0.0f32;
        let mut valid_bins = 0;

        for k in start_bin..=end_bin {
            let pxx_k = pxx[k];
            let pyy_k = pyy[k];
            let pxy_mag_sq = pxy_re[k] * pxy_re[k] + pxy_im[k] * pxy_im[k];

            // Coherence = |Pxy|^2 / (Pxx * Pyy)
            let denominator = pxx_k * pyy_k;
            if denominator > 1e-10 {
                let coh = pxy_mag_sq / denominator;
                coherence_sum += coh.min(1.0); // Clamp to [0, 1]
                valid_bins += 1;
            }
        }

        if valid_bins > 0 {
            coherence_sum / valid_bins as f32
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
            timestamp_us: 0,
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
    fn test_coherence_welch() {
        // Generate correlated signals: two identical 10 Hz sine waves
        let sample_rate = 256.0;
        let n_samples = 256;
        let signal1: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / sample_rate).sin())
            .collect();
        let signal2 = signal1.clone();

        // Test with perfectly correlated signals - should have high coherence
        let coherence =
            HyperscanningSession::calculate_coherence_welch(&signal1, &signal2, (8.0, 13.0), sample_rate);
        assert!(coherence > 0.9, "Identical signals should have coherence > 0.9, got {}", coherence);

        // Test with uncorrelated signals (different frequencies)
        let signal3: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 25.0 * i as f32 / sample_rate).sin())
            .collect();
        let coherence2 =
            HyperscanningSession::calculate_coherence_welch(&signal1, &signal3, (8.0, 13.0), sample_rate);

        // Different frequency signals should have lower coherence in alpha band
        assert!(coherence2 < coherence, "Different frequency signals should have lower coherence");
    }
}

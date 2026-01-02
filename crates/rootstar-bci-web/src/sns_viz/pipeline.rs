//! Real-time BCI to Visualization Pipeline (SNS-19)
//!
//! This module implements the data pipeline connecting BCI data streams
//! (EEG, fNIRS) to the SNS 3D visualization system. It handles:
//! - WebSocket-based data streaming
//! - Sample buffering and synchronization
//! - Receptor activation computation
//! - Real-time heatmap updates

use std::collections::VecDeque;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, WebSocket};

use rootstar_bci_core::sns::types::SensoryModality;

use super::heatmap::{ActivationHeatmap, Colormap};
use super::meshes::MeshId;
use super::SnsScene;

/// Pipeline configuration
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// WebSocket URL for BCI data stream
    pub ws_url: String,
    /// Buffer size for incoming samples (in samples)
    pub buffer_size: usize,
    /// Update rate for visualization (Hz)
    pub viz_update_rate: f32,
    /// Enable EEG data streaming
    pub enable_eeg: bool,
    /// Enable fNIRS data streaming
    pub enable_fnirs: bool,
    /// Activation smoothing factor (0.0-1.0, higher = more smoothing)
    pub smoothing_factor: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ws_url: "ws://localhost:8080/bci".to_string(),
            buffer_size: 256,
            viz_update_rate: 30.0,
            enable_eeg: true,
            enable_fnirs: true,
            smoothing_factor: 0.3,
        }
    }
}

/// Sample data from BCI stream
#[derive(Clone, Debug)]
pub struct BciSample {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Sample sequence number
    pub sequence: u32,
    /// Sensory modality
    pub modality: SensoryModality,
    /// Channel index
    pub channel: u8,
    /// Normalized activation value (0.0-1.0)
    pub activation: f32,
}

/// Decoded receptor activation for visualization
#[derive(Clone, Debug)]
pub struct ReceptorActivation {
    /// Mesh identifier
    pub mesh_id: MeshId,
    /// Receptor index within mesh
    pub receptor_index: usize,
    /// Current activation level (0.0-1.0)
    pub activation: f32,
    /// Smoothed activation (for display)
    pub smoothed_activation: f32,
}

/// Real-time BCI to visualization pipeline
#[wasm_bindgen]
pub struct BciVizPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// WebSocket connection (optional)
    websocket: Option<WebSocket>,
    /// Incoming sample buffer
    sample_buffer: VecDeque<BciSample>,
    /// Current receptor activations by mesh
    activations: Vec<ReceptorActivation>,
    /// Heatmap renderer
    heatmap: ActivationHeatmap,
    /// Connected state
    connected: bool,
    /// Last update timestamp
    last_update_us: u64,
    /// Accumulated samples since last viz update
    samples_since_update: usize,
    /// Error message (if any)
    last_error: Option<String>,
}

#[wasm_bindgen]
impl BciVizPipeline {
    /// Create new pipeline with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            websocket: None,
            sample_buffer: VecDeque::with_capacity(256),
            activations: Vec::new(),
            heatmap: ActivationHeatmap::new(Colormap::Viridis, 0.02),
            connected: false,
            last_update_us: 0,
            samples_since_update: 0,
            last_error: None,
        }
    }

    /// Create pipeline with custom WebSocket URL
    #[wasm_bindgen]
    pub fn with_url(url: &str) -> Self {
        let mut pipeline = Self::new();
        pipeline.config.ws_url = url.to_string();
        pipeline
    }

    /// Set the colormap for activation display
    #[wasm_bindgen]
    pub fn set_colormap(&mut self, colormap: &str) {
        let cmap = match colormap.to_lowercase().as_str() {
            "viridis" => Colormap::Viridis,
            "plasma" => Colormap::Plasma,
            "inferno" => Colormap::Inferno,
            "turbo" => Colormap::Turbo,
            "coolwarm" => Colormap::CoolWarm,
            _ => Colormap::Viridis,
        };
        self.heatmap.set_colormap(cmap);
    }

    /// Set activation value range
    #[wasm_bindgen]
    pub fn set_activation_range(&mut self, min_val: f32, max_val: f32) {
        self.heatmap.set_range(min_val, max_val);
    }

    /// Set smoothing factor (0.0-1.0)
    #[wasm_bindgen]
    pub fn set_smoothing(&mut self, factor: f32) {
        self.config.smoothing_factor = factor.clamp(0.0, 0.99);
    }

    /// Connect to BCI data stream
    #[wasm_bindgen]
    pub fn connect(&mut self) -> Result<(), JsValue> {
        let ws = WebSocket::new(&self.config.ws_url)?;
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        // Store reference for callbacks
        self.websocket = Some(ws);
        self.connected = false;

        Ok(())
    }

    /// Disconnect from BCI data stream
    #[wasm_bindgen]
    pub fn disconnect(&mut self) {
        if let Some(ws) = self.websocket.take() {
            let _ = ws.close();
        }
        self.connected = false;
    }

    /// Check if connected
    #[wasm_bindgen]
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get last error message
    #[wasm_bindgen]
    pub fn get_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    /// Clear error state
    #[wasm_bindgen]
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }

    /// Get number of buffered samples
    #[wasm_bindgen]
    pub fn buffered_samples(&self) -> usize {
        self.sample_buffer.len()
    }

    /// Process incoming message (called from JS callback)
    #[wasm_bindgen]
    pub fn on_message(&mut self, data: &[u8]) {
        if let Some(sample) = self.decode_sample(data) {
            self.sample_buffer.push_back(sample);
            self.samples_since_update += 1;

            // Limit buffer size
            while self.sample_buffer.len() > self.config.buffer_size {
                self.sample_buffer.pop_front();
            }
        }
    }

    /// Process buffered samples and update activations
    #[wasm_bindgen]
    pub fn update(&mut self, timestamp_us: u64) -> bool {
        // Check if it's time for a visualization update
        let update_interval_us = (1_000_000.0 / self.config.viz_update_rate) as u64;
        if timestamp_us - self.last_update_us < update_interval_us {
            return false;
        }

        self.last_update_us = timestamp_us;

        // Process all buffered samples
        while let Some(sample) = self.sample_buffer.pop_front() {
            self.process_sample(&sample);
        }

        self.samples_since_update = 0;
        true
    }

    /// Get activation value for a specific receptor
    #[wasm_bindgen]
    pub fn get_activation(&self, mesh_id: u32, receptor_index: usize) -> f32 {
        for activation in &self.activations {
            if activation.mesh_id.as_u32() == mesh_id
                && activation.receptor_index == receptor_index
            {
                return activation.smoothed_activation;
            }
        }
        0.0
    }

    /// Get all activations as flat array [mesh_id, receptor_idx, activation, ...]
    #[wasm_bindgen]
    pub fn get_all_activations(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.activations.len() * 3);
        for activation in &self.activations {
            result.push(activation.mesh_id.as_u32() as f32);
            result.push(activation.receptor_index as f32);
            result.push(activation.smoothed_activation);
        }
        result
    }

    /// Render activation heatmap to pixel buffer
    #[wasm_bindgen]
    pub fn render_heatmap(&self, mesh_id: u32, width: u32, height: u32) -> Vec<u8> {
        // Collect activations for this mesh
        let mut states = Vec::new();
        let mut uvs = Vec::new();

        for activation in &self.activations {
            if activation.mesh_id.as_u32() == mesh_id {
                states.push(activation.smoothed_activation);
                // Generate UV based on receptor index (simple linear distribution)
                let u = (activation.receptor_index as f32 % 16.0) / 16.0;
                let v = (activation.receptor_index as f32 / 16.0).floor() / 16.0;
                uvs.push([u, v]);
            }
        }

        self.heatmap.render_to_pixels(&states, &uvs, width, height)
    }

    /// Initialize receptor activations for a scene
    #[wasm_bindgen]
    pub fn initialize_for_scene(&mut self, num_receptors: usize, mesh_id: u32) {
        // Clear existing activations for this mesh
        self.activations.retain(|a| a.mesh_id.as_u32() != mesh_id);

        // Create new activation entries
        for i in 0..num_receptors {
            self.activations.push(ReceptorActivation {
                mesh_id: MeshId::from_u32(mesh_id),
                receptor_index: i,
                activation: 0.0,
                smoothed_activation: 0.0,
            });
        }
    }
}

impl BciVizPipeline {
    /// Decode binary sample from WebSocket message
    fn decode_sample(&mut self, data: &[u8]) -> Option<BciSample> {
        // Simple binary format:
        // [0-7]:   timestamp_us (u64 LE)
        // [8-11]:  sequence (u32 LE)
        // [12]:    modality (u8: 0=tactile, 1=auditory, 2=gustatory)
        // [13]:    channel (u8)
        // [14-17]: activation (f32 LE)

        if data.len() < 18 {
            return None;
        }

        let timestamp_us = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let sequence =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let modality = match data[12] {
            0 => SensoryModality::Tactile,
            1 => SensoryModality::Auditory,
            2 => SensoryModality::Gustatory,
            _ => return None,
        };
        let channel = data[13];
        let activation = f32::from_le_bytes([data[14], data[15], data[16], data[17]]);

        Some(BciSample {
            timestamp_us,
            sequence,
            modality,
            channel,
            activation: activation.clamp(0.0, 1.0),
        })
    }

    /// Process a single sample and update activations
    fn process_sample(&mut self, sample: &BciSample) {
        // Find corresponding activation entry
        for activation in &mut self.activations {
            // Match based on modality and channel
            let mesh_modality = match activation.mesh_id {
                MeshId::SkinPatch { .. } => SensoryModality::Tactile,
                MeshId::Cochlea { .. } => SensoryModality::Auditory,
                MeshId::Tongue => SensoryModality::Gustatory,
                MeshId::Retina { .. } => SensoryModality::Visual,
                MeshId::Olfactory { .. } => SensoryModality::Olfactory,
            };

            if mesh_modality == sample.modality
                && activation.receptor_index == sample.channel as usize
            {
                activation.activation = sample.activation;

                // Apply exponential smoothing
                let alpha = 1.0 - self.config.smoothing_factor;
                activation.smoothed_activation = alpha * sample.activation
                    + self.config.smoothing_factor * activation.smoothed_activation;
            }
        }
    }
}

impl Default for BciVizPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming mode for different data sources
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamMode {
    /// Live BCI data via WebSocket
    Live,
    /// Recorded data playback
    Playback,
    /// Simulated/synthetic data
    Simulation,
}

/// Playback controller for recorded data
#[wasm_bindgen]
pub struct PlaybackController {
    /// Recording data (timestamp_us, samples)
    recording: Vec<(u64, Vec<BciSample>)>,
    /// Current playback position
    position: usize,
    /// Playback speed multiplier
    speed: f32,
    /// Playback state
    playing: bool,
    /// Loop playback
    looping: bool,
    /// Start timestamp of current playback
    playback_start_us: u64,
}

#[wasm_bindgen]
impl PlaybackController {
    /// Create new playback controller
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            recording: Vec::new(),
            position: 0,
            speed: 1.0,
            playing: false,
            looping: false,
            playback_start_us: 0,
        }
    }

    /// Load recording from binary data
    #[wasm_bindgen]
    pub fn load_recording(&mut self, data: &[u8]) -> bool {
        self.recording.clear();
        self.position = 0;

        // Parse recording format
        // Header: [4 bytes magic "BCIR", 4 bytes version, 8 bytes total samples]
        // Frames: [8 bytes timestamp, 4 bytes sample count, samples...]

        if data.len() < 16 {
            return false;
        }

        let magic = &data[0..4];
        if magic != b"BCIR" {
            return false;
        }

        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let _total_samples =
            u64::from_le_bytes([data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]]);

        let mut offset = 16;
        while offset + 12 <= data.len() {
            let timestamp = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            let sample_count =
                u32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]])
                    as usize;

            offset += 12;

            let mut samples = Vec::with_capacity(sample_count);
            for _ in 0..sample_count {
                if offset + 18 > data.len() {
                    break;
                }

                // Reuse decode format from BciVizPipeline
                let modality = match data[offset + 12] {
                    0 => SensoryModality::Tactile,
                    1 => SensoryModality::Auditory,
                    2 => SensoryModality::Gustatory,
                    _ => continue,
                };

                samples.push(BciSample {
                    timestamp_us: timestamp,
                    sequence: u32::from_le_bytes([
                        data[offset + 8],
                        data[offset + 9],
                        data[offset + 10],
                        data[offset + 11],
                    ]),
                    modality,
                    channel: data[offset + 13],
                    activation: f32::from_le_bytes([
                        data[offset + 14],
                        data[offset + 15],
                        data[offset + 16],
                        data[offset + 17],
                    ])
                    .clamp(0.0, 1.0),
                });

                offset += 18;
            }

            if !samples.is_empty() {
                self.recording.push((timestamp, samples));
            }
        }

        !self.recording.is_empty()
    }

    /// Start playback
    #[wasm_bindgen]
    pub fn play(&mut self, current_time_us: u64) {
        self.playing = true;
        self.playback_start_us = current_time_us;
    }

    /// Pause playback
    #[wasm_bindgen]
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and reset
    #[wasm_bindgen]
    pub fn stop(&mut self) {
        self.playing = false;
        self.position = 0;
    }

    /// Set playback speed
    #[wasm_bindgen]
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.1, 10.0);
    }

    /// Enable/disable looping
    #[wasm_bindgen]
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Get next samples for current time
    #[wasm_bindgen]
    pub fn get_samples(&mut self, current_time_us: u64) -> Vec<u8> {
        if !self.playing || self.recording.is_empty() {
            return Vec::new();
        }

        let elapsed_us = ((current_time_us - self.playback_start_us) as f32 * self.speed) as u64;

        let mut result = Vec::new();

        while self.position < self.recording.len() {
            let (frame_time, samples) = &self.recording[self.position];

            if *frame_time <= elapsed_us {
                // Encode samples for pipeline
                for sample in samples {
                    let mut encoded = vec![0u8; 18];
                    encoded[0..8].copy_from_slice(&sample.timestamp_us.to_le_bytes());
                    encoded[8..12].copy_from_slice(&sample.sequence.to_le_bytes());
                    encoded[12] = match sample.modality {
                        SensoryModality::Tactile => 0,
                        SensoryModality::Auditory => 1,
                        SensoryModality::Gustatory => 2,
                        SensoryModality::Proprioceptive => 3,
                        SensoryModality::Nociceptive => 4,
                        SensoryModality::Thermoreceptive => 5,
                        SensoryModality::Visual => 6,
                        SensoryModality::Olfactory => 7,
                    };
                    encoded[13] = sample.channel;
                    encoded[14..18].copy_from_slice(&sample.activation.to_le_bytes());
                    result.extend_from_slice(&encoded);
                }
                self.position += 1;
            } else {
                break;
            }
        }

        // Handle looping
        if self.position >= self.recording.len() {
            if self.looping {
                self.position = 0;
                self.playback_start_us = current_time_us;
            } else {
                self.playing = false;
            }
        }

        result
    }

    /// Get playback progress (0.0-1.0)
    #[wasm_bindgen]
    pub fn progress(&self) -> f32 {
        if self.recording.is_empty() {
            return 0.0;
        }
        self.position as f32 / self.recording.len() as f32
    }

    /// Get recording duration in microseconds
    #[wasm_bindgen]
    pub fn duration_us(&self) -> u64 {
        if self.recording.is_empty() {
            return 0;
        }
        self.recording.last().map(|(t, _)| *t).unwrap_or(0)
            - self.recording.first().map(|(t, _)| *t).unwrap_or(0)
    }
}

impl Default for PlaybackController {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulation data generator for testing
#[wasm_bindgen]
pub struct SimulationGenerator {
    /// Simulation time
    time_us: u64,
    /// Sequence counter
    sequence: u32,
    /// Active modality
    modality: SensoryModality,
    /// Number of channels
    num_channels: u8,
    /// Base frequency for oscillations
    base_freq_hz: f32,
    /// Noise amplitude
    noise_amplitude: f32,
}

#[wasm_bindgen]
impl SimulationGenerator {
    /// Create new simulation generator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            time_us: 0,
            sequence: 0,
            modality: SensoryModality::Tactile,
            num_channels: 16,
            base_freq_hz: 1.0,
            noise_amplitude: 0.1,
        }
    }

    /// Set simulation modality
    #[wasm_bindgen]
    pub fn set_modality(&mut self, modality_str: &str) {
        self.modality = match modality_str.to_lowercase().as_str() {
            "tactile" => SensoryModality::Tactile,
            "auditory" => SensoryModality::Auditory,
            "gustatory" => SensoryModality::Gustatory,
            _ => SensoryModality::Tactile,
        };
    }

    /// Set number of channels
    #[wasm_bindgen]
    pub fn set_channels(&mut self, channels: u8) {
        self.num_channels = channels.clamp(1, 64);
    }

    /// Set base oscillation frequency
    #[wasm_bindgen]
    pub fn set_frequency(&mut self, freq_hz: f32) {
        self.base_freq_hz = freq_hz.clamp(0.1, 100.0);
    }

    /// Set noise amplitude
    #[wasm_bindgen]
    pub fn set_noise(&mut self, amplitude: f32) {
        self.noise_amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Generate samples for a time step
    #[wasm_bindgen]
    pub fn generate(&mut self, delta_us: u64) -> Vec<u8> {
        self.time_us += delta_us;

        let mut result = Vec::with_capacity(self.num_channels as usize * 18);
        let time_sec = self.time_us as f32 / 1_000_000.0;

        for channel in 0..self.num_channels {
            // Generate activation pattern: traveling wave + noise
            let phase = channel as f32 / self.num_channels as f32;
            let wave = 0.5 + 0.5 * ((2.0 * std::f32::consts::PI * self.base_freq_hz * time_sec - phase * std::f32::consts::PI).sin());

            // Simple pseudo-random noise using time and channel
            let noise_seed = (self.time_us as f32 * 0.0001 + channel as f32 * 17.3).sin();
            let noise = noise_seed * self.noise_amplitude;

            let activation = (wave + noise).clamp(0.0, 1.0);

            // Encode sample
            let mut encoded = vec![0u8; 18];
            encoded[0..8].copy_from_slice(&self.time_us.to_le_bytes());
            encoded[8..12].copy_from_slice(&self.sequence.to_le_bytes());
            encoded[12] = match self.modality {
                SensoryModality::Tactile => 0,
                SensoryModality::Auditory => 1,
                SensoryModality::Gustatory => 2,
                SensoryModality::Proprioceptive => 3,
                SensoryModality::Nociceptive => 4,
                SensoryModality::Thermoreceptive => 5,
                SensoryModality::Visual => 6,
                SensoryModality::Olfactory => 7,
            };
            encoded[13] = channel;
            encoded[14..18].copy_from_slice(&activation.to_le_bytes());
            result.extend_from_slice(&encoded);

            self.sequence = self.sequence.wrapping_add(1);
        }

        result
    }

    /// Reset generator state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.time_us = 0;
        self.sequence = 0;
    }
}

impl Default for SimulationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = BciVizPipeline::new();
        assert!(!pipeline.is_connected());
        assert_eq!(pipeline.buffered_samples(), 0);
    }

    #[test]
    fn test_sample_decode() {
        let mut pipeline = BciVizPipeline::new();

        // Create test sample data
        let mut data = vec![0u8; 18];
        let timestamp: u64 = 1_000_000;
        let sequence: u32 = 42;
        let modality: u8 = 0; // Tactile
        let channel: u8 = 5;
        let activation: f32 = 0.75;

        data[0..8].copy_from_slice(&timestamp.to_le_bytes());
        data[8..12].copy_from_slice(&sequence.to_le_bytes());
        data[12] = modality;
        data[13] = channel;
        data[14..18].copy_from_slice(&activation.to_le_bytes());

        let sample = pipeline.decode_sample(&data).unwrap();
        assert_eq!(sample.timestamp_us, 1_000_000);
        assert_eq!(sample.sequence, 42);
        assert_eq!(sample.modality, SensoryModality::Tactile);
        assert_eq!(sample.channel, 5);
        assert!((sample.activation - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_simulation_generator() {
        let mut generator = SimulationGenerator::new();
        generator.set_channels(8);
        generator.set_frequency(2.0);

        let samples = generator.generate(1000);
        assert_eq!(samples.len(), 8 * 18); // 8 channels * 18 bytes per sample
    }

    #[test]
    fn test_playback_controller() {
        let controller = PlaybackController::new();
        assert!(!controller.playing);
        assert_eq!(controller.progress(), 0.0);
    }
}

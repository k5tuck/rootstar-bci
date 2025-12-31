//! Rootstar BCI Web Visualization
//!
//! WASM-based visualization and control panel for the Rootstar BCI Platform.
//! Provides real-time EEG plots, topographic maps, fNIRS heatmaps, and
//! 3D sensory receptor visualization.
//!
//! # Multi-Device Support
//!
//! The [`dashboard`] module provides a multi-device dashboard for managing
//! and visualizing multiple BCI devices simultaneously. It supports:
//! - Device bar with connection status
//! - Single, side-by-side, overlay, and grid view modes
//! - Per-device visualization with color coding
//! - Stimulation status display

pub mod control;
pub mod dashboard;
pub mod sns_viz;
pub mod viz;

// Re-export dashboard types
pub use dashboard::{DashboardDevice, DeviceStatus, MultiDeviceDashboard, ViewMode};

use wasm_bindgen::prelude::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages in browser console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// BCI visualization application state
#[wasm_bindgen]
pub struct BciApp {
    /// EEG timeseries renderer
    timeseries: viz::TimeseriesRenderer,
    /// Topographic map renderer
    topomap: viz::TopomapRenderer,
    /// fNIRS heatmap renderer
    fnirs_map: viz::FnirsMapRenderer,
    /// EMG visualization renderer
    emg: viz::EmgRenderer,
    /// EDA visualization renderer
    eda: viz::EdaRenderer,
    /// VR Preview (integrated view)
    vr_preview: viz::VrPreviewRenderer,
    /// Stimulation control panel
    control_panel: control::StimControlPanel,
    /// Connection state
    connected: bool,
}

#[wasm_bindgen]
impl BciApp {
    /// Create a new BCI application instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BciApp, JsValue> {
        let timeseries = viz::TimeseriesRenderer::new()?;
        let topomap = viz::TopomapRenderer::new()?;
        let fnirs_map = viz::FnirsMapRenderer::new()?;
        let emg = viz::EmgRenderer::new()?;
        let eda = viz::EdaRenderer::new()?;
        let vr_preview = viz::VrPreviewRenderer::new()?;
        let control_panel = control::StimControlPanel::new();

        Ok(Self {
            timeseries,
            topomap,
            fnirs_map,
            emg,
            eda,
            vr_preview,
            control_panel,
            connected: false,
        })
    }

    /// Connect to WebSocket data stream
    pub fn connect(&mut self, url: &str) -> Result<(), JsValue> {
        // WebSocket connection would be established here
        self.connected = true;
        web_sys::console::log_1(&format!("Connecting to BCI stream at {}", url).into());
        Ok(())
    }

    /// Disconnect from data stream
    pub fn disconnect(&mut self) {
        self.connected = false;
        web_sys::console::log_1(&"Disconnected from BCI stream".into());
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Process incoming EEG data (raw protocol bytes)
    ///
    /// Parses a packet buffer and extracts EEG sample data.
    pub fn process_eeg_data(&mut self, data: &[u8]) -> Result<(), JsValue> {
        use rootstar_bci_core::protocol::{PacketHeader, PacketType, deserialize_eeg_sample};

        // Need at least header size
        if data.len() < PacketHeader::SIZE {
            return Ok(());
        }

        // Parse header
        if let Ok(header) = PacketHeader::from_bytes(data) {
            if header.packet_type == PacketType::EegData {
                let payload_start = PacketHeader::SIZE;
                let payload_end = payload_start + header.payload_len as usize;

                if data.len() >= payload_end {
                    let payload = &data[payload_start..payload_end];
                    if let Ok(sample) = deserialize_eeg_sample(payload) {
                        let values: Vec<f32> = sample.channels.iter()
                            .map(|c| c.to_f32())
                            .collect();
                        self.timeseries.push_raw(&values)?;
                        self.topomap.update_raw(&values)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Process incoming fNIRS data (raw protocol bytes)
    ///
    /// Parses a packet buffer and extracts fNIRS sample data.
    pub fn process_fnirs_data(&mut self, data: &[u8]) -> Result<(), JsValue> {
        use rootstar_bci_core::protocol::{PacketHeader, PacketType, deserialize_fnirs_sample};

        if data.len() < PacketHeader::SIZE {
            return Ok(());
        }

        if let Ok(header) = PacketHeader::from_bytes(data) {
            if header.packet_type == PacketType::FnirsData {
                let payload_start = PacketHeader::SIZE;
                let payload_end = payload_start + header.payload_len as usize;

                if data.len() >= payload_end {
                    let payload = &data[payload_start..payload_end];
                    if let Ok(sample) = deserialize_fnirs_sample(payload) {
                        // For now just use the raw intensities
                        // In a real app, you'd process through Beer-Lambert
                        let hbo2 = sample.intensity_760 as f32 / 1000.0;
                        let hbr = sample.intensity_850 as f32 / 1000.0;
                        self.fnirs_map.update_raw(&[hbo2], &[hbr])?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Push raw EEG values directly (8 channels in µV)
    pub fn push_eeg_raw(&mut self, channels: &[f32]) -> Result<(), JsValue> {
        self.timeseries.push_raw(channels)?;
        self.topomap.update_raw(channels)?;
        Ok(())
    }

    /// Push raw fNIRS values directly
    pub fn push_fnirs_raw(&mut self, hbo2: &[f32], hbr: &[f32]) -> Result<(), JsValue> {
        self.fnirs_map.update_raw(hbo2, hbr)?;
        Ok(())
    }

    /// Push EMG RMS values (8 channels in µV)
    pub fn push_emg_rms(&mut self, channels: &[f32]) -> Result<(), JsValue> {
        self.emg.push_rms(channels)?;
        Ok(())
    }

    /// Push raw EDA values directly (4 sites in µS)
    pub fn push_eda_raw(&mut self, sites: &[f32]) -> Result<(), JsValue> {
        self.eda.push_raw(sites)?;
        Ok(())
    }

    /// Push EDA decomposed values (SCL/SCR)
    ///
    /// `is_scr_peak` is a bitmask indicating which sites have an SCR peak.
    pub fn push_eda_decomposed(&mut self, scl: &[f32], scr: &[f32], is_scr_peak: u8) -> Result<(), JsValue> {
        self.eda.push_decomposed(scl, scr, is_scr_peak)?;
        Ok(())
    }

    /// Set VR preview EEG band power
    pub fn set_vr_eeg_bands(&mut self, delta: f32, theta: f32, alpha: f32, beta: f32, gamma: f32) {
        self.vr_preview.set_eeg_band_power(delta, theta, alpha, beta, gamma);
    }

    /// Set VR preview EEG topography
    pub fn set_vr_eeg_topography(&mut self, values: &[f32]) {
        self.vr_preview.set_eeg_topography(values);
    }

    /// Set VR preview fNIRS data
    pub fn set_vr_fnirs(&mut self, hbo: &[f32], hbr: &[f32]) {
        self.vr_preview.set_fnirs(hbo, hbr);
    }

    /// Set VR preview EMG data
    pub fn set_vr_emg(&mut self, rms: &[f32], valence: f32, arousal: f32) {
        self.vr_preview.set_emg(rms, valence, arousal);
    }

    /// Set VR preview EDA data
    pub fn set_vr_eda(&mut self, scl: &[f32], arousal: f32) {
        self.vr_preview.set_eda(scl, arousal);
    }

    /// Set VR preview fingerprint state
    pub fn set_vr_fingerprint(&mut self, similarity: f32, target: &str, modality: &str) {
        self.vr_preview.set_fingerprint_state(similarity, target, modality);
    }

    /// Tick VR preview animation
    pub fn tick_vr_preview(&mut self, dt: f64) {
        self.vr_preview.tick(dt);
    }

    /// Render all visualizations
    pub fn render(&mut self) -> Result<(), JsValue> {
        self.timeseries.render()?;
        self.topomap.render()?;
        self.fnirs_map.render()?;
        self.emg.render()?;
        self.eda.render()?;
        self.vr_preview.render()?;
        Ok(())
    }

    /// Get stimulation parameters from control panel as JSON
    pub fn get_stim_params(&self) -> Result<JsValue, JsValue> {
        self.control_panel.get_params_js()
    }

    /// Set stimulation parameters from JSON
    pub fn set_stim_params(&mut self, params_js: JsValue) -> Result<(), JsValue> {
        self.control_panel.set_params_js(params_js)
    }

    /// Get the timeseries canvas element
    pub fn get_timeseries_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.timeseries.canvas()
    }

    /// Get the topomap canvas element
    pub fn get_topomap_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.topomap.canvas()
    }

    /// Get the fNIRS map canvas element
    pub fn get_fnirs_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.fnirs_map.canvas()
    }

    /// Get the EMG canvas element
    pub fn get_emg_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.emg.canvas()
    }

    /// Get the EDA canvas element
    pub fn get_eda_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.eda.canvas()
    }

    /// Get the VR preview canvas element
    pub fn get_vr_preview_canvas(&self) -> web_sys::HtmlCanvasElement {
        self.vr_preview.canvas()
    }
}

impl Default for BciApp {
    fn default() -> Self {
        Self::new().expect("Failed to create BciApp")
    }
}

/// Data streaming configuration (not exported to JS directly due to String)
pub struct StreamConfig {
    /// WebSocket URL for data stream
    pub ws_url: String,
    /// EEG sample rate (Hz)
    pub eeg_sample_rate: u32,
    /// fNIRS sample rate (Hz)
    pub fnirs_sample_rate: u32,
    /// Enable EEG streaming
    pub enable_eeg: bool,
    /// Enable fNIRS streaming
    pub enable_fnirs: bool,
}

impl StreamConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            ws_url: String::from("ws://localhost:8080/bci"),
            eeg_sample_rate: 250,
            fnirs_sample_rate: 10,
            enable_eeg: true,
            enable_fnirs: true,
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// JS-accessible stream configuration builder
#[wasm_bindgen]
pub struct StreamConfigBuilder {
    config: StreamConfig,
}

#[wasm_bindgen]
impl StreamConfigBuilder {
    /// Create a new builder with defaults
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: StreamConfig::new(),
        }
    }

    /// Set WebSocket URL
    pub fn url(mut self, url: &str) -> Self {
        self.config.ws_url = url.to_string();
        self
    }

    /// Set EEG sample rate
    pub fn eeg_rate(mut self, rate: u32) -> Self {
        self.config.eeg_sample_rate = rate;
        self
    }

    /// Set fNIRS sample rate
    pub fn fnirs_rate(mut self, rate: u32) -> Self {
        self.config.fnirs_sample_rate = rate;
        self
    }

    /// Enable/disable EEG
    pub fn enable_eeg(mut self, enable: bool) -> Self {
        self.config.enable_eeg = enable;
        self
    }

    /// Enable/disable fNIRS
    pub fn enable_fnirs(mut self, enable: bool) -> Self {
        self.config.enable_fnirs = enable;
        self
    }

    /// Get the URL (for display)
    pub fn get_url(&self) -> String {
        self.config.ws_url.clone()
    }

    /// Get EEG rate
    pub fn get_eeg_rate(&self) -> u32 {
        self.config.eeg_sample_rate
    }

    /// Get fNIRS rate
    pub fn get_fnirs_rate(&self) -> u32 {
        self.config.fnirs_sample_rate
    }
}

impl Default for StreamConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

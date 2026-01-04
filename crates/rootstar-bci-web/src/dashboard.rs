//! Multi-device dashboard for web visualization.
//!
//! Provides a unified interface for managing and visualizing multiple
//! BCI devices simultaneously.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::viz::{
    DisplaySettings, EdaRenderer, EmgRenderer, FnirsMapRenderer, TimeseriesRenderer,
    TopomapRenderer, VrPreviewRenderer,
};

// ============================================================================
// Device Card
// ============================================================================

/// Connection status for a device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[wasm_bindgen]
pub enum DeviceStatus {
    /// Device discovered but not connected
    Discovered,
    /// Connecting to device
    Connecting,
    /// Connected and receiving data
    Connected,
    /// Device disconnected
    Disconnected,
    /// Error state
    Error,
}

/// Device information for the dashboard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardDevice {
    /// Unique device ID (hex string)
    pub id: String,
    /// User-assigned name
    pub name: String,
    /// Connection type ("USB" or "BLE")
    pub connection_type: String,
    /// Connection details (port or address)
    pub connection_info: String,
    /// Current status
    pub status: DeviceStatus,
    /// Signal quality (0-100%)
    pub signal_quality: u8,
    /// Device color [R, G, B]
    pub color: [u8; 3],
    /// Whether device is muted
    pub muted: bool,
    /// Stimulation active
    pub stimulating: bool,
    /// Current similarity (0-100%)
    pub similarity: u8,
}

impl DashboardDevice {
    /// Create a new dashboard device.
    #[must_use]
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            connection_type: "USB".to_string(),
            connection_info: String::new(),
            status: DeviceStatus::Discovered,
            signal_quality: 0,
            color: [0x00, 0x7A, 0xCC],
            muted: false,
            stimulating: false,
            similarity: 0,
        }
    }

    /// Get CSS color string.
    #[must_use]
    pub fn css_color(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.color[0], self.color[1], self.color[2])
    }
}

// ============================================================================
// View Modes
// ============================================================================

/// Dashboard view mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[wasm_bindgen]
pub enum ViewMode {
    /// Single device full view
    Single,
    /// Two devices side by side
    SideBySide,
    /// Multiple signals overlaid
    Overlay,
    /// Grid of all devices
    Grid,
}

// ============================================================================
// Multi-Device Dashboard
// ============================================================================

/// Multi-device dashboard manager.
#[wasm_bindgen]
pub struct MultiDeviceDashboard {
    /// All devices
    devices: HashMap<String, DashboardDevice>,

    /// Device order for display
    device_order: Vec<String>,

    /// Current view mode
    view_mode: ViewMode,

    /// Focused device ID (for single view)
    focused_device: Option<String>,

    /// Selected devices (for side-by-side/overlay)
    selected_devices: Vec<String>,

    /// Canvas for the device bar
    device_bar_canvas: HtmlCanvasElement,
    device_bar_ctx: CanvasRenderingContext2d,

    /// Per-device renderers
    device_renderers: HashMap<String, DeviceRenderers>,

    /// Main display canvas
    main_canvas: HtmlCanvasElement,
    main_ctx: CanvasRenderingContext2d,

    /// Settings
    settings: DashboardSettings,
}

/// Per-device renderer set.
pub struct DeviceRenderers {
    /// Timeseries renderer
    pub timeseries: TimeseriesRenderer,
    /// Topomap renderer
    pub topomap: TopomapRenderer,
    /// fNIRS renderer
    pub fnirs: FnirsMapRenderer,
    /// EMG renderer
    pub emg: EmgRenderer,
    /// EDA renderer
    pub eda: EdaRenderer,
    /// VR preview renderer
    pub vr_preview: VrPreviewRenderer,
}

impl DeviceRenderers {
    /// Create new renderers for a device.
    pub fn new() -> Result<Self, JsValue> {
        Ok(Self {
            timeseries: TimeseriesRenderer::new()?,
            topomap: TopomapRenderer::new()?,
            fnirs: FnirsMapRenderer::new()?,
            emg: EmgRenderer::new()?,
            eda: EdaRenderer::new()?,
            vr_preview: VrPreviewRenderer::new()?,
        })
    }
}

/// Dashboard display settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardSettings {
    /// Width of device bar
    pub device_bar_height: u32,
    /// Device card width
    pub device_card_width: u32,
    /// Device card height
    pub device_card_height: u32,
    /// Grid columns (for grid view)
    pub grid_columns: u32,
    /// Show signal quality bars
    pub show_signal_quality: bool,
    /// Show stimulation status
    pub show_stim_status: bool,
}

impl Default for DashboardSettings {
    fn default() -> Self {
        Self {
            device_bar_height: 100,
            device_card_width: 150,
            device_card_height: 80,
            grid_columns: 2,
            show_signal_quality: true,
            show_stim_status: true,
        }
    }
}

#[wasm_bindgen]
impl MultiDeviceDashboard {
    /// Create a new multi-device dashboard.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MultiDeviceDashboard, JsValue> {
        // Create device bar canvas
        let document = web_sys::window()
            .ok_or("no window")?
            .document()
            .ok_or("no document")?;

        let device_bar_canvas: HtmlCanvasElement = document
            .create_element("canvas")?
            .dyn_into()?;
        device_bar_canvas.set_width(1200);
        device_bar_canvas.set_height(100);

        let device_bar_ctx: CanvasRenderingContext2d = device_bar_canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into()?;

        // Create main canvas
        let main_canvas: HtmlCanvasElement = document
            .create_element("canvas")?
            .dyn_into()?;
        main_canvas.set_width(1200);
        main_canvas.set_height(800);

        let main_ctx: CanvasRenderingContext2d = main_canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into()?;

        Ok(Self {
            devices: HashMap::new(),
            device_order: Vec::new(),
            view_mode: ViewMode::Single,
            focused_device: None,
            selected_devices: Vec::new(),
            device_bar_canvas,
            device_bar_ctx,
            device_renderers: HashMap::new(),
            main_canvas,
            main_ctx,
            settings: DashboardSettings::default(),
        })
    }

    /// Get the device bar canvas element.
    #[wasm_bindgen(getter)]
    pub fn device_bar_canvas(&self) -> HtmlCanvasElement {
        self.device_bar_canvas.clone()
    }

    /// Get the main canvas element.
    #[wasm_bindgen(getter)]
    pub fn main_canvas(&self) -> HtmlCanvasElement {
        self.main_canvas.clone()
    }

    /// Add a device to the dashboard.
    #[wasm_bindgen]
    pub fn add_device(&mut self, id: String, name: String) -> Result<(), JsValue> {
        if self.devices.contains_key(&id) {
            return Err("Device already exists".into());
        }

        let device = DashboardDevice::new(id.clone(), name);
        let renderers = DeviceRenderers::new()?;

        self.devices.insert(id.clone(), device);
        self.device_renderers.insert(id.clone(), renderers);
        self.device_order.push(id.clone());

        // Auto-focus first device
        if self.focused_device.is_none() {
            self.focused_device = Some(id);
        }

        Ok(())
    }

    /// Remove a device from the dashboard.
    #[wasm_bindgen]
    pub fn remove_device(&mut self, id: &str) {
        self.devices.remove(id);
        self.device_renderers.remove(id);
        self.device_order.retain(|d| d != id);
        self.selected_devices.retain(|d| d != id);

        if self.focused_device.as_deref() == Some(id) {
            self.focused_device = self.device_order.first().cloned();
        }
    }

    /// Update device status.
    #[wasm_bindgen]
    pub fn update_device_status(&mut self, id: &str, status: DeviceStatus) {
        if let Some(device) = self.devices.get_mut(id) {
            device.status = status;
        }
    }

    /// Update device signal quality.
    #[wasm_bindgen]
    pub fn update_signal_quality(&mut self, id: &str, quality: u8) {
        if let Some(device) = self.devices.get_mut(id) {
            device.signal_quality = quality.min(100);
        }
    }

    /// Update device stimulation state.
    #[wasm_bindgen]
    pub fn update_stim_state(&mut self, id: &str, stimulating: bool, similarity: u8) {
        if let Some(device) = self.devices.get_mut(id) {
            device.stimulating = stimulating;
            device.similarity = similarity.min(100);
        }
    }

    /// Set device color.
    #[wasm_bindgen]
    pub fn set_device_color(&mut self, id: &str, r: u8, g: u8, b: u8) {
        if let Some(device) = self.devices.get_mut(id) {
            device.color = [r, g, b];
        }
    }

    /// Set device name.
    #[wasm_bindgen]
    pub fn set_device_name(&mut self, id: &str, name: String) {
        if let Some(device) = self.devices.get_mut(id) {
            device.name = name;
        }
    }

    /// Set device muted state.
    #[wasm_bindgen]
    pub fn set_device_muted(&mut self, id: &str, muted: bool) {
        if let Some(device) = self.devices.get_mut(id) {
            device.muted = muted;
        }
    }

    /// Focus on a specific device.
    #[wasm_bindgen]
    pub fn focus_device(&mut self, id: &str) {
        if self.devices.contains_key(id) {
            self.focused_device = Some(id.to_string());
            self.view_mode = ViewMode::Single;
        }
    }

    /// Set view mode.
    #[wasm_bindgen]
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.view_mode = mode;
    }

    /// Select devices for comparison.
    #[wasm_bindgen]
    pub fn select_devices(&mut self, ids: Vec<String>) {
        self.selected_devices = ids
            .into_iter()
            .filter(|id| self.devices.contains_key(id))
            .collect();
    }

    /// Get number of devices.
    #[wasm_bindgen]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device IDs as JSON array.
    #[wasm_bindgen]
    pub fn device_ids(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.device_order).unwrap_or(JsValue::NULL)
    }

    /// Get device info as JSON.
    #[wasm_bindgen]
    pub fn get_device_info(&self, id: &str) -> JsValue {
        self.devices
            .get(id)
            .map(|d| serde_wasm_bindgen::to_value(d).unwrap_or(JsValue::NULL))
            .unwrap_or(JsValue::NULL)
    }

    /// Get all devices as JSON.
    #[wasm_bindgen]
    pub fn get_all_devices(&self) -> JsValue {
        let devices: Vec<&DashboardDevice> = self.device_order
            .iter()
            .filter_map(|id| self.devices.get(id))
            .collect();
        serde_wasm_bindgen::to_value(&devices).unwrap_or(JsValue::NULL)
    }

    /// Push EEG data to a device.
    #[wasm_bindgen]
    pub fn push_eeg_data(&mut self, device_id: &str, channels: &[f32]) {
        if let Some(renderers) = self.device_renderers.get_mut(device_id) {
            let _ = renderers.timeseries.push_raw(channels);
            let _ = renderers.topomap.update_raw(channels);
        }
    }

    /// Push fNIRS data to a device.
    #[wasm_bindgen]
    pub fn push_fnirs_data(&mut self, device_id: &str, hbo: &[f32], hbr: &[f32]) {
        if let Some(renderers) = self.device_renderers.get_mut(device_id) {
            let _ = renderers.fnirs.update_raw(hbo, hbr);
        }
    }

    /// Push EMG data to a device.
    #[wasm_bindgen]
    pub fn push_emg_data(&mut self, device_id: &str, channels: &[f32]) {
        if let Some(renderers) = self.device_renderers.get_mut(device_id) {
            let _ = renderers.emg.push_rms(channels);
        }
    }

    /// Push EDA data to a device.
    #[wasm_bindgen]
    pub fn push_eda_data(&mut self, device_id: &str, scl: &[f32], scr: &[f32]) {
        if let Some(renderers) = self.device_renderers.get_mut(device_id) {
            // Use push_decomposed with no SCR peak indicators
            let _ = renderers.eda.push_decomposed(scl, scr, 0);
        }
    }

    /// Render the device bar.
    #[wasm_bindgen]
    pub fn render_device_bar(&self) -> Result<(), JsValue> {
        let ctx = &self.device_bar_ctx;
        let width = self.device_bar_canvas.width() as f64;
        let height = self.device_bar_canvas.height() as f64;

        // Clear
        ctx.set_fill_style(&"#1a1a2e".into());
        ctx.fill_rect(0.0, 0.0, width, height);

        // Draw device cards
        let card_w = self.settings.device_card_width as f64;
        let card_h = self.settings.device_card_height as f64;
        let padding = 10.0;

        for (i, id) in self.device_order.iter().enumerate() {
            if let Some(device) = self.devices.get(id) {
                let x = padding + (i as f64) * (card_w + padding);
                let y = (height - card_h) / 2.0;

                // Card background
                let is_focused = self.focused_device.as_deref() == Some(id);
                let bg_color = if is_focused { "#2a2a4e" } else { "#16162a" };
                ctx.set_fill_style(&bg_color.into());
                ctx.fill_rect(x, y, card_w, card_h);

                // Color accent bar
                ctx.set_fill_style(&device.css_color().into());
                ctx.fill_rect(x, y, 4.0, card_h);

                // Status indicator
                let status_color = match device.status {
                    DeviceStatus::Connected => "#00cc66",
                    DeviceStatus::Connecting => "#cccc00",
                    DeviceStatus::Discovered => "#6666cc",
                    DeviceStatus::Disconnected => "#666666",
                    DeviceStatus::Error => "#cc0000",
                };
                ctx.set_fill_style(&status_color.into());
                ctx.begin_path();
                ctx.arc(x + card_w - 15.0, y + 15.0, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
                ctx.fill();

                // Device name
                ctx.set_fill_style(&"#ffffff".into());
                ctx.set_font("12px sans-serif");
                ctx.fill_text(&device.name, x + 12.0, y + 20.0)?;

                // Connection info
                ctx.set_fill_style(&"#888888".into());
                ctx.set_font("10px sans-serif");
                let conn_text = format!("{}: {}", device.connection_type, &device.connection_info);
                ctx.fill_text(&conn_text, x + 12.0, y + 35.0)?;

                // Signal quality bar
                if self.settings.show_signal_quality {
                    let bar_w = card_w - 24.0;
                    let bar_h = 8.0;
                    let bar_y = y + card_h - 20.0;

                    // Background
                    ctx.set_fill_style(&"#333333".into());
                    ctx.fill_rect(x + 12.0, bar_y, bar_w, bar_h);

                    // Fill
                    let quality_pct = device.signal_quality as f64 / 100.0;
                    let quality_color = if device.signal_quality >= 80 {
                        "#00cc66"
                    } else if device.signal_quality >= 50 {
                        "#cccc00"
                    } else {
                        "#cc6600"
                    };
                    ctx.set_fill_style(&quality_color.into());
                    ctx.fill_rect(x + 12.0, bar_y, bar_w * quality_pct, bar_h);
                }

                // Stimulation indicator
                if self.settings.show_stim_status && device.stimulating {
                    ctx.set_fill_style(&"#ff6600".into());
                    ctx.set_font("bold 10px sans-serif");
                    ctx.fill_text("⚡ STIM", x + 12.0, y + 50.0)?;

                    let sim_text = format!("{}%", device.similarity);
                    ctx.fill_text(&sim_text, x + 70.0, y + 50.0)?;
                }

                // Muted indicator
                if device.muted {
                    ctx.set_fill_style(&"#ff0000".into());
                    ctx.set_font("bold 10px sans-serif");
                    ctx.fill_text("MUTED", x + card_w - 50.0, y + card_h - 8.0)?;
                }

                // Focused border
                if is_focused {
                    ctx.set_stroke_style(&device.css_color().into());
                    ctx.set_line_width(2.0);
                    ctx.stroke_rect(x, y, card_w, card_h);
                }
            }
        }

        // Add device button
        let add_x = padding + (self.device_order.len() as f64) * (card_w + padding);
        if add_x + 60.0 < width {
            ctx.set_fill_style(&"#333333".into());
            ctx.fill_rect(add_x, (height - 40.0) / 2.0, 50.0, 40.0);
            ctx.set_fill_style(&"#888888".into());
            ctx.set_font("24px sans-serif");
            ctx.fill_text("+", add_x + 18.0, (height + 12.0) / 2.0)?;
        }

        Ok(())
    }

    /// Render the main visualization area.
    #[wasm_bindgen]
    pub fn render_main(&self) -> Result<(), JsValue> {
        let ctx = &self.main_ctx;
        let width = self.main_canvas.width() as f64;
        let height = self.main_canvas.height() as f64;

        // Clear
        ctx.set_fill_style(&"#0a0a1a".into());
        ctx.fill_rect(0.0, 0.0, width, height);

        match self.view_mode {
            ViewMode::Single => self.render_single_view(width, height)?,
            ViewMode::SideBySide => self.render_side_by_side_view(width, height)?,
            ViewMode::Overlay => self.render_overlay_view(width, height)?,
            ViewMode::Grid => self.render_grid_view(width, height)?,
        }

        Ok(())
    }

    /// Render single device view.
    fn render_single_view(&self, width: f64, height: f64) -> Result<(), JsValue> {
        let ctx = &self.main_ctx;

        if let Some(device_id) = &self.focused_device {
            if let Some(renderers) = self.device_renderers.get(device_id) {
                // Render VR preview in the center
                let _ = renderers.vr_preview.render();

                // Copy VR preview canvas to main canvas
                let vr_canvas = renderers.vr_preview.canvas();
                ctx.draw_image_with_html_canvas_element(
                    &vr_canvas,
                    (width - vr_canvas.width() as f64) / 2.0,
                    (height - vr_canvas.height() as f64) / 2.0,
                )?;
            }
        } else {
            // No device focused
            ctx.set_fill_style(&"#888888".into());
            ctx.set_font("24px sans-serif");
            ctx.set_text_align("center");
            ctx.fill_text("No device selected", width / 2.0, height / 2.0)?;
            ctx.set_text_align("left");
        }

        Ok(())
    }

    /// Render side-by-side comparison view.
    fn render_side_by_side_view(&self, width: f64, height: f64) -> Result<(), JsValue> {
        let ctx = &self.main_ctx;
        let devices: Vec<&str> = if self.selected_devices.len() >= 2 {
            self.selected_devices.iter().take(2).map(|s| s.as_str()).collect()
        } else {
            self.device_order.iter().take(2).map(|s| s.as_str()).collect()
        };

        let half_width = width / 2.0;

        for (i, device_id) in devices.iter().enumerate() {
            let x_offset = (i as f64) * half_width;

            if let Some(renderers) = self.device_renderers.get(*device_id) {
                // Render VR preview
                let _ = renderers.vr_preview.render();

                // Copy to main canvas (scaled)
                let vr_canvas = renderers.vr_preview.canvas();
                let scale = (half_width - 20.0) / vr_canvas.width() as f64;
                let scaled_height = vr_canvas.height() as f64 * scale;

                ctx.draw_image_with_html_canvas_element_and_dw_and_dh(
                    &vr_canvas,
                    x_offset + 10.0,
                    (height - scaled_height) / 2.0,
                    half_width - 20.0,
                    scaled_height,
                )?;

                // Device label
                if let Some(device) = self.devices.get(*device_id) {
                    ctx.set_fill_style(&device.css_color().into());
                    ctx.set_font("16px sans-serif");
                    ctx.fill_text(&device.name, x_offset + 20.0, 30.0)?;
                }
            }

            // Divider
            if i == 0 {
                ctx.set_stroke_style(&"#333333".into());
                ctx.set_line_width(2.0);
                ctx.begin_path();
                ctx.move_to(half_width, 0.0);
                ctx.line_to(half_width, height);
                ctx.stroke();
            }
        }

        Ok(())
    }

    /// Render overlay comparison view with alpha-blended timeseries.
    fn render_overlay_view(&self, width: f64, height: f64) -> Result<(), JsValue> {
        let ctx = &self.main_ctx;

        // Title
        ctx.set_fill_style(&"#ffffff".into());
        ctx.set_font("18px sans-serif");
        ctx.fill_text("Multi-Device Overlay", 20.0, 30.0)?;

        // Get devices to overlay
        let devices: Vec<&str> = if !self.selected_devices.is_empty() {
            self.selected_devices.iter().map(|s| s.as_str()).collect()
        } else {
            self.device_order.iter().map(|s| s.as_str()).collect()
        };

        // Plot area
        let plot_x = 60.0;
        let plot_y = 60.0;
        let plot_width = width - 220.0;
        let plot_height = height - 100.0;

        // Draw plot background and grid
        ctx.set_fill_style(&"#0d0d1a".into());
        ctx.fill_rect(plot_x, plot_y, plot_width, plot_height);

        // Draw horizontal grid lines
        ctx.set_stroke_style(&"#1a1a2e".into());
        ctx.set_line_width(1.0);
        for i in 0..=4 {
            let y = plot_y + (i as f64) * plot_height / 4.0;
            ctx.begin_path();
            ctx.move_to(plot_x, y);
            ctx.line_to(plot_x + plot_width, y);
            ctx.stroke();
        }

        // Draw vertical grid lines (time markers)
        for i in 0..=10 {
            let x = plot_x + (i as f64) * plot_width / 10.0;
            ctx.begin_path();
            ctx.move_to(x, plot_y);
            ctx.line_to(x, plot_y + plot_height);
            ctx.stroke();
        }

        // Draw axis labels
        ctx.set_fill_style(&"#666666".into());
        ctx.set_font("10px sans-serif");

        // Y-axis labels (amplitude)
        for i in 0..=4 {
            let y = plot_y + (i as f64) * plot_height / 4.0;
            let label = format!("{}", 100 - i * 50);
            ctx.fill_text(&label, plot_x - 30.0, y + 4.0)?;
        }
        ctx.save();
        ctx.translate(15.0, plot_y + plot_height / 2.0)?;
        ctx.rotate(-std::f64::consts::FRAC_PI_2)?;
        ctx.set_text_align("center");
        ctx.fill_text("Amplitude (µV)", 0.0, 0.0)?;
        ctx.restore();
        ctx.set_text_align("left");

        // X-axis label
        ctx.set_text_align("center");
        ctx.fill_text("Time (s)", plot_x + plot_width / 2.0, height - 10.0)?;
        ctx.set_text_align("left");

        // Calculate alpha for each device (more devices = more transparency)
        let alpha = (0.8 / devices.len().max(1) as f64).max(0.3);

        // Render each device's timeseries with its color and alpha blending
        for (i, device_id) in devices.iter().enumerate() {
            if let Some(renderers) = self.device_renderers.get(*device_id) {
                if let Some(device) = self.devices.get(*device_id) {
                    // Get the timeseries canvas
                    let ts_canvas = renderers.timeseries.canvas();

                    // Set global alpha for blending
                    ctx.set_global_alpha(alpha);

                    // Draw the timeseries canvas onto the plot area
                    ctx.draw_image_with_html_canvas_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
                        &ts_canvas,
                        0.0,
                        0.0,
                        ts_canvas.width() as f64,
                        ts_canvas.height() as f64,
                        plot_x,
                        plot_y,
                        plot_width,
                        plot_height,
                    )?;

                    // Draw colored overlay tint
                    let [r, g, b] = device.color;
                    let color_str = format!("rgba({}, {}, {}, 0.15)", r, g, b);
                    ctx.set_fill_style(&color_str.into());
                    ctx.fill_rect(plot_x, plot_y, plot_width, plot_height);

                    // Reset alpha
                    ctx.set_global_alpha(1.0);

                    // Legend entry
                    let legend_x = width - 150.0;
                    let legend_y = plot_y + 20.0 + (i as f64) * 25.0;

                    // Color box
                    ctx.set_fill_style(&device.css_color().into());
                    ctx.fill_rect(legend_x, legend_y - 10.0, 15.0, 15.0);

                    // Device name
                    ctx.set_fill_style(&"#ffffff".into());
                    ctx.set_font("12px sans-serif");
                    ctx.fill_text(&device.name, legend_x + 20.0, legend_y)?;

                    // Signal quality indicator
                    let quality_color = if device.signal_quality >= 80 {
                        "#00cc66"
                    } else if device.signal_quality >= 50 {
                        "#cccc00"
                    } else {
                        "#cc6600"
                    };
                    ctx.set_fill_style(&quality_color.into());
                    ctx.set_font("10px sans-serif");
                    ctx.fill_text(
                        &format!("{}%", device.signal_quality),
                        legend_x + 80.0,
                        legend_y,
                    )?;
                }
            }
        }

        // Draw plot border
        ctx.set_stroke_style(&"#444444".into());
        ctx.set_line_width(1.0);
        ctx.stroke_rect(plot_x, plot_y, plot_width, plot_height);

        // Legend box
        let legend_box_x = width - 160.0;
        let legend_box_y = plot_y;
        let legend_box_h = 30.0 + devices.len() as f64 * 25.0;
        ctx.set_fill_style(&"#16162a".into());
        ctx.fill_rect(legend_box_x, legend_box_y, 155.0, legend_box_h);
        ctx.set_stroke_style(&"#333333".into());
        ctx.stroke_rect(legend_box_x, legend_box_y, 155.0, legend_box_h);

        // Legend title
        ctx.set_fill_style(&"#888888".into());
        ctx.set_font("11px sans-serif");
        ctx.fill_text("Devices", legend_box_x + 10.0, legend_box_y + 15.0)?;

        Ok(())
    }

    /// Render grid view of all devices.
    fn render_grid_view(&self, width: f64, height: f64) -> Result<(), JsValue> {
        let ctx = &self.main_ctx;

        let cols = self.settings.grid_columns.max(1) as usize;
        let rows = (self.device_order.len() + cols - 1) / cols;

        let cell_w = width / cols as f64;
        let cell_h = height / rows.max(1) as f64;

        for (i, device_id) in self.device_order.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let x = col as f64 * cell_w;
            let y = row as f64 * cell_h;

            if let Some(device) = self.devices.get(device_id) {
                // Cell border with device color
                ctx.set_stroke_style(&device.css_color().into());
                ctx.set_line_width(2.0);
                ctx.stroke_rect(x + 5.0, y + 5.0, cell_w - 10.0, cell_h - 10.0);

                // Device name
                ctx.set_fill_style(&device.css_color().into());
                ctx.set_font("14px sans-serif");
                ctx.fill_text(&device.name, x + 15.0, y + 25.0)?;

                // Status
                let status_text = format!("{:?}", device.status);
                ctx.set_fill_style(&"#888888".into());
                ctx.set_font("10px sans-serif");
                ctx.fill_text(&status_text, x + 15.0, y + 40.0)?;

                // Mini VR preview
                if let Some(renderers) = self.device_renderers.get(device_id) {
                    let _ = renderers.vr_preview.render();
                    let vr_canvas = renderers.vr_preview.canvas();

                    let preview_w = cell_w - 30.0;
                    let preview_h = cell_h - 60.0;
                    let scale = (preview_w / vr_canvas.width() as f64)
                        .min(preview_h / vr_canvas.height() as f64);

                    ctx.draw_image_with_html_canvas_element_and_dw_and_dh(
                        &vr_canvas,
                        x + 15.0,
                        y + 50.0,
                        vr_canvas.width() as f64 * scale,
                        vr_canvas.height() as f64 * scale,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Render the complete dashboard.
    #[wasm_bindgen]
    pub fn render(&self) -> Result<(), JsValue> {
        self.render_device_bar()?;
        self.render_main()?;
        Ok(())
    }
}

impl Default for MultiDeviceDashboard {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

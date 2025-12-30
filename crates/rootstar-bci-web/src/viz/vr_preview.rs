//! Integrated VR Preview visualization
//!
//! Combines all biosignal modalities (EEG, fNIRS, EMG, EDA) into a unified
//! display for VR-integrated sensory fingerprint experiences.

use std::collections::VecDeque;

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::{ColorScheme, DisplaySettings};

/// Maximum samples to buffer
const MAX_BUFFER_SIZE: usize = 1000;

/// Integrated VR preview panel combining all biosignal modalities
#[wasm_bindgen]
pub struct VrPreviewRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,

    // EEG state
    eeg_band_power: [f32; 5], // Delta, Theta, Alpha, Beta, Gamma
    eeg_topography: [f32; 8],

    // fNIRS state
    fnirs_hbo: [f32; 4],
    fnirs_hbr: [f32; 4],
    oxygenation_index: f32,

    // EMG state
    emg_rms: [f32; 8],
    emg_valence: f32,
    emg_arousal: f32,

    // EDA state
    eda_scl: [f32; 4],
    eda_arousal: f32,

    // Neural fingerprint state
    fingerprint_similarity: f32,
    target_stimulus: String,
    current_modality: String,
    stimulation_active: bool,
    stimulation_progress: f32,

    // Animation state
    time: f64,
    heartbeat_phase: f32,
}

#[wasm_bindgen]
impl VrPreviewRenderer {
    /// Create a new VR preview renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<VrPreviewRenderer, JsValue> {
        let document = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .document()
            .ok_or_else(|| JsValue::from_str("No document"))?;

        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;

        let ctx = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("No 2d context"))?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let mut settings = DisplaySettings::new();
        settings.width = 1200;
        settings.height = 800;
        canvas.set_width(settings.width);
        canvas.set_height(settings.height);

        Ok(Self {
            canvas,
            ctx,
            settings,
            eeg_band_power: [0.0; 5],
            eeg_topography: [0.0; 8],
            fnirs_hbo: [0.0; 4],
            fnirs_hbr: [0.0; 4],
            oxygenation_index: 0.5,
            emg_rms: [0.0; 8],
            emg_valence: 0.0,
            emg_arousal: 0.0,
            eda_scl: [5.0; 4],
            eda_arousal: 0.0,
            fingerprint_similarity: 0.0,
            target_stimulus: String::from("chocolate"),
            current_modality: String::from("gustatory"),
            stimulation_active: false,
            stimulation_progress: 0.0,
            time: 0.0,
            heartbeat_phase: 0.0,
        })
    }

    /// Get the canvas element
    pub fn canvas(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }

    // ========================================================================
    // Data update methods
    // ========================================================================

    /// Update EEG band power values
    pub fn set_eeg_band_power(&mut self, delta: f32, theta: f32, alpha: f32, beta: f32, gamma: f32) {
        self.eeg_band_power = [delta, theta, alpha, beta, gamma];
    }

    /// Update EEG topography (8 channels)
    pub fn set_eeg_topography(&mut self, values: &[f32]) {
        for (i, &v) in values.iter().take(8).enumerate() {
            self.eeg_topography[i] = v;
        }
    }

    /// Update fNIRS hemoglobin values
    pub fn set_fnirs(&mut self, hbo: &[f32], hbr: &[f32]) {
        for (i, &v) in hbo.iter().take(4).enumerate() {
            self.fnirs_hbo[i] = v;
        }
        for (i, &v) in hbr.iter().take(4).enumerate() {
            self.fnirs_hbr[i] = v;
        }
        // Calculate oxygenation index
        let total_hbo: f32 = self.fnirs_hbo.iter().sum();
        let total_hbr: f32 = self.fnirs_hbr.iter().map(|x| x.abs()).sum();
        self.oxygenation_index = if total_hbo + total_hbr > 0.01 {
            total_hbo / (total_hbo + total_hbr)
        } else {
            0.5
        };
    }

    /// Update EMG values
    pub fn set_emg(&mut self, rms: &[f32], valence: f32, arousal: f32) {
        for (i, &v) in rms.iter().take(8).enumerate() {
            self.emg_rms[i] = v;
        }
        self.emg_valence = valence;
        self.emg_arousal = arousal;
    }

    /// Update EDA values
    pub fn set_eda(&mut self, scl: &[f32], arousal: f32) {
        for (i, &v) in scl.iter().take(4).enumerate() {
            self.eda_scl[i] = v;
        }
        self.eda_arousal = arousal;
    }

    /// Update neural fingerprint state
    pub fn set_fingerprint_state(&mut self, similarity: f32, target: &str, modality: &str) {
        self.fingerprint_similarity = similarity;
        self.target_stimulus = target.to_string();
        self.current_modality = modality.to_string();
    }

    /// Update stimulation state
    pub fn set_stimulation(&mut self, active: bool, progress: f32) {
        self.stimulation_active = active;
        self.stimulation_progress = progress;
    }

    /// Advance animation time
    pub fn tick(&mut self, dt: f64) {
        self.time += dt;
        self.heartbeat_phase = (self.time * 1.2).sin() as f32 * 0.5 + 0.5;
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render the complete VR preview
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.settings.width as f64;
        let height = self.settings.height as f64;

        // Background gradient
        self.draw_background(width, height)?;

        // Layout: 4 quadrants + central fingerprint display
        let panel_margin = 20.0;
        let panel_width = (width - panel_margin * 3.0) / 2.0;
        let panel_height = (height - 140.0 - panel_margin * 3.0) / 2.0;

        // Top left: EEG
        self.draw_eeg_panel(panel_margin, panel_margin, panel_width, panel_height)?;

        // Top right: fNIRS
        self.draw_fnirs_panel(panel_width + panel_margin * 2.0, panel_margin, panel_width, panel_height)?;

        // Bottom left: EMG
        self.draw_emg_panel(panel_margin, panel_height + panel_margin * 2.0, panel_width, panel_height)?;

        // Bottom right: EDA
        self.draw_eda_panel(panel_width + panel_margin * 2.0, panel_height + panel_margin * 2.0, panel_width, panel_height)?;

        // Bottom: Fingerprint status bar
        self.draw_fingerprint_status(panel_margin, height - 120.0, width - panel_margin * 2.0, 100.0)?;

        Ok(())
    }

    fn draw_background(&self, width: f64, height: f64) -> Result<(), JsValue> {
        // Dark gradient background
        self.ctx.set_fill_style_str("#0a0a1a");
        self.ctx.fill_rect(0.0, 0.0, width, height);

        // Subtle grid pattern
        self.ctx.set_stroke_style_str("rgba(100, 100, 200, 0.1)");
        self.ctx.set_line_width(0.5);
        let grid_size = 40.0;

        for i in 0..((width / grid_size) as i32 + 1) {
            let x = i as f64 * grid_size;
            self.ctx.begin_path();
            self.ctx.move_to(x, 0.0);
            self.ctx.line_to(x, height);
            self.ctx.stroke();
        }

        for i in 0..((height / grid_size) as i32 + 1) {
            let y = i as f64 * grid_size;
            self.ctx.begin_path();
            self.ctx.move_to(0.0, y);
            self.ctx.line_to(width, y);
            self.ctx.stroke();
        }

        Ok(())
    }

    fn draw_panel_frame(&self, x: f64, y: f64, w: f64, h: f64, title: &str, color: &str) -> Result<(), JsValue> {
        // Panel background
        self.ctx.set_fill_style_str("rgba(20, 20, 40, 0.8)");
        self.ctx.fill_rect(x, y, w, h);

        // Border
        self.ctx.set_stroke_style_str(color);
        self.ctx.set_line_width(2.0);
        self.ctx.stroke_rect(x, y, w, h);

        // Title bar
        self.ctx.set_fill_style_str(color);
        self.ctx.set_global_alpha(0.3);
        self.ctx.fill_rect(x, y, w, 25.0);
        self.ctx.set_global_alpha(1.0);

        // Title text
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("bold 14px monospace");
        self.ctx.fill_text(title, x + 10.0, y + 17.0)?;

        Ok(())
    }

    fn draw_eeg_panel(&self, x: f64, y: f64, w: f64, h: f64) -> Result<(), JsValue> {
        self.draw_panel_frame(x, y, w, h, "EEG - Brain Activity", "#4ecdc4")?;

        let content_y = y + 35.0;
        let content_h = h - 45.0;

        // Band power bars
        let band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"];
        let band_colors = ["#9b59b6", "#3498db", "#27ae60", "#f1c40f", "#e74c3c"];
        let bar_height = 25.0;
        let bar_spacing = 30.0;

        for (i, (name, color)) in band_names.iter().zip(band_colors.iter()).enumerate() {
            let bar_y = content_y + 10.0 + i as f64 * bar_spacing;
            let power = self.eeg_band_power[i].clamp(0.0, 1.0);

            // Label
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.set_font("11px monospace");
            self.ctx.fill_text(name, x + 10.0, bar_y + 16.0)?;

            // Background
            self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.1)");
            self.ctx.fill_rect(x + 60.0, bar_y, w - 80.0, bar_height);

            // Power bar
            self.ctx.set_fill_style_str(color);
            self.ctx.fill_rect(x + 60.0, bar_y, (w - 80.0) * power as f64, bar_height);
        }

        // Mini topomap
        let topo_size = 100.0;
        let topo_x = x + w - topo_size - 20.0;
        let topo_y = content_y + content_h - topo_size - 10.0;
        self.draw_mini_topomap(topo_x, topo_y, topo_size)?;

        Ok(())
    }

    fn draw_mini_topomap(&self, x: f64, y: f64, size: f64) -> Result<(), JsValue> {
        let cx = x + size / 2.0;
        let cy = y + size / 2.0;
        let radius = size * 0.4;

        // Head outline
        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.5)");
        self.ctx.set_line_width(1.5);
        self.ctx.begin_path();
        self.ctx.arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.stroke();

        // Electrode positions (simplified 8-channel)
        let positions = [
            (0.0, -0.8),   // Fp1
            (0.0, -0.8),   // Fp2 (offset)
            (-0.6, 0.0),   // C3
            (0.6, 0.0),    // C4
            (-0.5, 0.5),   // P3
            (0.5, 0.5),    // P4
            (-0.3, 0.8),   // O1
            (0.3, 0.8),    // O2
        ];

        for (i, &(px, py)) in positions.iter().enumerate() {
            let ex = cx + px * radius * (if i < 2 { if i == 0 { -0.5 } else { 0.5 } } else { 1.0 });
            let ey = cy + py * radius;

            // Color based on activity
            let activity = (self.eeg_topography[i] + 50.0) / 100.0; // Normalize to 0-1
            let [r, g, b] = ColorScheme::BlueRed.to_rgb(activity);
            self.ctx.set_fill_style_str(&format!("rgb({},{},{})", r, g, b));

            self.ctx.begin_path();
            self.ctx.arc(ex, ey, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.ctx.fill();
        }

        Ok(())
    }

    fn draw_fnirs_panel(&self, x: f64, y: f64, w: f64, h: f64) -> Result<(), JsValue> {
        self.draw_panel_frame(x, y, w, h, "fNIRS - Hemodynamics", "#e74c3c")?;

        let content_y = y + 35.0;

        // Oxygenation gauge
        let gauge_cx = x + w / 2.0;
        let gauge_cy = content_y + 80.0;
        let gauge_r = 60.0;

        // Background arc
        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.2)");
        self.ctx.set_line_width(15.0);
        self.ctx.begin_path();
        self.ctx.arc(gauge_cx, gauge_cy, gauge_r, std::f64::consts::PI * 0.7, std::f64::consts::PI * 2.3)?;
        self.ctx.stroke();

        // Oxygenation arc (blue to red)
        let oxy = self.oxygenation_index;
        let [r, g, b] = ColorScheme::BlueRed.to_rgb(oxy);
        self.ctx.set_stroke_style_str(&format!("rgb({},{},{})", r, g, b));
        self.ctx.set_line_width(12.0);
        self.ctx.begin_path();
        let end_angle = std::f64::consts::PI * 0.7 + oxy as f64 * std::f64::consts::PI * 1.6;
        self.ctx.arc(gauge_cx, gauge_cy, gauge_r, std::f64::consts::PI * 0.7, end_angle)?;
        self.ctx.stroke();

        // Percentage text
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("bold 24px monospace");
        self.ctx.set_text_align("center");
        self.ctx.fill_text(&format!("{:.0}%", oxy * 100.0), gauge_cx, gauge_cy + 8.0)?;
        self.ctx.set_font("12px monospace");
        self.ctx.fill_text("Oxygenation", gauge_cx, gauge_cy + 25.0)?;
        self.ctx.set_text_align("left");

        // HbO/HbR bars
        let bar_y = gauge_cy + gauge_r + 30.0;
        self.ctx.set_fill_style_str("#e74c3c");
        self.ctx.fill_text("HbO", x + 20.0, bar_y + 15.0)?;
        self.ctx.set_fill_style_str("#3498db");
        self.ctx.fill_text("HbR", x + 20.0, bar_y + 40.0)?;

        let hbo_mean: f32 = self.fnirs_hbo.iter().sum::<f32>() / 4.0;
        let hbr_mean: f32 = self.fnirs_hbr.iter().sum::<f32>() / 4.0;

        self.ctx.set_fill_style_str("#e74c3c");
        self.ctx.fill_rect(x + 60.0, bar_y, ((hbo_mean + 5.0) / 10.0).clamp(0.0, 1.0) as f64 * (w - 90.0), 20.0);
        self.ctx.set_fill_style_str("#3498db");
        self.ctx.fill_rect(x + 60.0, bar_y + 25.0, ((hbr_mean.abs() + 5.0) / 10.0).clamp(0.0, 1.0) as f64 * (w - 90.0), 20.0);

        Ok(())
    }

    fn draw_emg_panel(&self, x: f64, y: f64, w: f64, h: f64) -> Result<(), JsValue> {
        self.draw_panel_frame(x, y, w, h, "EMG - Facial Expression", "#2ecc71")?;

        let content_y = y + 35.0;

        // Valence indicator (emoticon style)
        let face_cx = x + w / 2.0;
        let face_cy = content_y + 70.0;
        let face_r = 50.0;

        // Face circle
        self.ctx.set_stroke_style_str("#ffffff");
        self.ctx.set_line_width(3.0);
        self.ctx.begin_path();
        self.ctx.arc(face_cx, face_cy, face_r, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.stroke();

        // Eyes
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.begin_path();
        self.ctx.arc(face_cx - 18.0, face_cy - 15.0, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.arc(face_cx + 18.0, face_cy - 15.0, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.fill();

        // Mouth (curved based on valence)
        self.ctx.begin_path();
        let mouth_curve = self.emg_valence * 20.0;
        self.ctx.move_to(face_cx - 25.0, face_cy + 15.0);
        self.ctx.quadratic_curve_to(face_cx, face_cy + 15.0 + mouth_curve as f64, face_cx + 25.0, face_cy + 15.0);
        self.ctx.stroke();

        // Valence label
        let valence_text = if self.emg_valence > 0.3 {
            "Positive"
        } else if self.emg_valence < -0.3 {
            "Negative"
        } else {
            "Neutral"
        };
        let valence_color = if self.emg_valence > 0.3 {
            "#2ecc71"
        } else if self.emg_valence < -0.3 {
            "#e74c3c"
        } else {
            "#95a5a6"
        };

        self.ctx.set_fill_style_str(valence_color);
        self.ctx.set_font("bold 14px monospace");
        self.ctx.set_text_align("center");
        self.ctx.fill_text(valence_text, face_cx, face_cy + face_r + 25.0)?;
        self.ctx.set_text_align("left");

        // Arousal bar
        let arousal_y = content_y + h - 70.0;
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("11px monospace");
        self.ctx.fill_text("Arousal", x + 10.0, arousal_y)?;

        self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.2)");
        self.ctx.fill_rect(x + 70.0, arousal_y - 12.0, w - 90.0, 15.0);

        let arousal_color = if self.emg_arousal > 0.66 { "#e74c3c" } else if self.emg_arousal > 0.33 { "#f1c40f" } else { "#3498db" };
        self.ctx.set_fill_style_str(arousal_color);
        self.ctx.fill_rect(x + 70.0, arousal_y - 12.0, (w - 90.0) * self.emg_arousal as f64, 15.0);

        Ok(())
    }

    fn draw_eda_panel(&self, x: f64, y: f64, w: f64, h: f64) -> Result<(), JsValue> {
        self.draw_panel_frame(x, y, w, h, "EDA - Autonomic Response", "#9b59b6")?;

        let content_y = y + 35.0;

        // Arousal gauge (circular)
        let gauge_cx = x + w / 2.0;
        let gauge_cy = content_y + 80.0;
        let gauge_r = 55.0;

        // Pulsing effect based on arousal
        let pulse = 1.0 + self.heartbeat_phase * self.eda_arousal * 0.1;
        let pulse_r = gauge_r * pulse as f64;

        // Glow effect
        self.ctx.set_fill_style_str(&format!("rgba(155, 89, 182, {})", 0.2 * self.eda_arousal));
        self.ctx.begin_path();
        self.ctx.arc(gauge_cx, gauge_cy, pulse_r + 15.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.fill();

        // Main circle
        let arousal_color = if self.eda_arousal > 0.66 { "#e74c3c" } else if self.eda_arousal > 0.33 { "#f1c40f" } else { "#3498db" };
        self.ctx.set_stroke_style_str(arousal_color);
        self.ctx.set_line_width(8.0);
        self.ctx.begin_path();
        self.ctx.arc(gauge_cx, gauge_cy, pulse_r, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.stroke();

        // Arousal percentage
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("bold 20px monospace");
        self.ctx.set_text_align("center");
        self.ctx.fill_text(&format!("{:.0}%", self.eda_arousal * 100.0), gauge_cx, gauge_cy + 7.0)?;
        self.ctx.set_font("10px monospace");
        self.ctx.fill_text("Arousal", gauge_cx, gauge_cy + 22.0)?;
        self.ctx.set_text_align("left");

        // SCL values
        let scl_y = gauge_cy + gauge_r + 30.0;
        self.ctx.set_font("11px monospace");
        self.ctx.fill_text("SCL (µS)", x + 10.0, scl_y)?;

        let mean_scl: f32 = self.eda_scl.iter().sum::<f32>() / 4.0;
        self.ctx.set_fill_style_str("#9b59b6");
        self.ctx.fill_rect(x + 70.0, scl_y - 12.0, ((mean_scl / 20.0).clamp(0.0, 1.0)) as f64 * (w - 90.0), 15.0);

        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.fill_text(&format!("{:.1}", mean_scl), x + w - 40.0, scl_y)?;

        Ok(())
    }

    fn draw_fingerprint_status(&self, x: f64, y: f64, w: f64, h: f64) -> Result<(), JsValue> {
        // Background
        self.ctx.set_fill_style_str("rgba(20, 20, 40, 0.9)");
        self.ctx.fill_rect(x, y, w, h);

        // Border
        let border_color = if self.stimulation_active { "#f1c40f" } else { "#34495e" };
        self.ctx.set_stroke_style_str(border_color);
        self.ctx.set_line_width(2.0);
        self.ctx.stroke_rect(x, y, w, h);

        // Neural fingerprint icon (brain-like)
        let icon_x = x + 30.0;
        let icon_y = y + h / 2.0;
        self.draw_brain_icon(icon_x, icon_y, 25.0)?;

        // Target info
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("bold 16px monospace");
        self.ctx.fill_text("Neural Fingerprint", x + 70.0, y + 25.0)?;

        self.ctx.set_font("12px monospace");
        self.ctx.set_fill_style_str("#95a5a6");
        self.ctx.fill_text(&format!("Target: {} ({})", self.target_stimulus, self.current_modality), x + 70.0, y + 45.0)?;

        // Similarity gauge
        let gauge_x = x + 300.0;
        let gauge_w = w - 450.0;

        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.fill_text("Similarity", gauge_x, y + 25.0)?;

        // Background bar
        self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.2)");
        self.ctx.fill_rect(gauge_x, y + 35.0, gauge_w, 25.0);

        // Similarity bar
        let sim_color = if self.fingerprint_similarity > 0.9 { "#2ecc71" } else if self.fingerprint_similarity > 0.5 { "#f1c40f" } else { "#e74c3c" };
        self.ctx.set_fill_style_str(sim_color);
        self.ctx.fill_rect(gauge_x, y + 35.0, gauge_w * self.fingerprint_similarity as f64, 25.0);

        // Percentage
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("bold 14px monospace");
        self.ctx.fill_text(&format!("{:.1}%", self.fingerprint_similarity * 100.0), gauge_x + gauge_w + 10.0, y + 52.0)?;

        // Stimulation status
        let stim_x = x + w - 130.0;
        if self.stimulation_active {
            self.ctx.set_fill_style_str("#f1c40f");
            self.ctx.set_font("bold 12px monospace");
            self.ctx.fill_text("STIMULATING", stim_x, y + 25.0)?;

            // Progress bar
            self.ctx.set_fill_style_str("rgba(241, 196, 15, 0.3)");
            self.ctx.fill_rect(stim_x, y + 35.0, 100.0, 15.0);
            self.ctx.set_fill_style_str("#f1c40f");
            self.ctx.fill_rect(stim_x, y + 35.0, 100.0 * self.stimulation_progress as f64, 15.0);
        } else {
            self.ctx.set_fill_style_str("#95a5a6");
            self.ctx.set_font("12px monospace");
            self.ctx.fill_text("Standby", stim_x, y + 35.0)?;
        }

        Ok(())
    }

    fn draw_brain_icon(&self, cx: f64, cy: f64, size: f64) -> Result<(), JsValue> {
        self.ctx.set_stroke_style_str("#4ecdc4");
        self.ctx.set_line_width(2.0);

        // Simple brain-like curves
        self.ctx.begin_path();
        self.ctx.arc(cx - size * 0.3, cy, size * 0.5, std::f64::consts::PI * 0.5, std::f64::consts::PI * 1.5)?;
        self.ctx.stroke();

        self.ctx.begin_path();
        self.ctx.arc(cx + size * 0.3, cy, size * 0.5, std::f64::consts::PI * 1.5, std::f64::consts::PI * 2.5)?;
        self.ctx.stroke();

        // Neural activity sparkles
        self.ctx.set_fill_style_str("#4ecdc4");
        let sparkle_offset = (self.time * 3.0).sin() * 5.0;
        self.ctx.begin_path();
        self.ctx.arc(cx + sparkle_offset, cy - 10.0, 3.0, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.fill();

        Ok(())
    }
}

impl Default for VrPreviewRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create VrPreviewRenderer")
    }
}

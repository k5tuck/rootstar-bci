//! Real-time EDA (Electrodermal Activity) visualization
//!
//! Renders skin conductance with SCL/SCR decomposition and arousal indicator.

use std::collections::VecDeque;

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::DisplaySettings;

/// Maximum samples to buffer per site
const MAX_BUFFER_SIZE: usize = 5000;

/// EDA site names
const EDA_SITE_NAMES: [&str; 4] = [
    "Palm-L",   // Palmar Left
    "Palm-R",   // Palmar Right
    "Then-L",   // Thenar Left
    "Then-R",   // Thenar Right
];

/// Site colors
const EDA_SITE_COLORS: [&str; 4] = [
    "#a78bfa", // Purple
    "#8b5cf6",
    "#c4b5fd",
    "#a78bfa",
];

/// EDA visualization renderer
#[wasm_bindgen]
pub struct EdaRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,
    /// Raw conductance buffers per site
    conductance_buffers: [VecDeque<f32>; 4],
    /// SCL (tonic) buffers per site
    scl_buffers: [VecDeque<f32>; 4],
    /// SCR (phasic) buffers per site
    scr_buffers: [VecDeque<f32>; 4],
    /// Current SCL values
    current_scl: [f32; 4],
    /// SCR event count
    scr_count: [u32; 4],
    /// Current arousal score (0 to 1)
    arousal: f32,
    /// Sample rate (Hz)
    sample_rate: f32,
    /// Show raw signal
    show_raw: bool,
    /// Show SCL/SCR decomposition
    show_decomposition: bool,
    /// Show arousal gauge
    show_arousal: bool,
}

#[wasm_bindgen]
impl EdaRenderer {
    /// Create a new EDA renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<EdaRenderer, JsValue> {
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

        let settings = DisplaySettings::new();
        canvas.set_width(settings.width);
        canvas.set_height(settings.height);

        let create_buffers = || core::array::from_fn(|_| VecDeque::with_capacity(MAX_BUFFER_SIZE));

        Ok(Self {
            canvas,
            ctx,
            settings,
            conductance_buffers: create_buffers(),
            scl_buffers: create_buffers(),
            scr_buffers: create_buffers(),
            current_scl: [0.0; 4],
            scr_count: [0; 4],
            arousal: 0.0,
            sample_rate: 32.0, // EDA is typically 8-32 Hz
            show_raw: true,
            show_decomposition: true,
            show_arousal: true,
        })
    }

    /// Get the canvas element
    pub fn canvas(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }

    /// Set sample rate
    pub fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }

    /// Toggle raw signal display
    pub fn set_show_raw(&mut self, show: bool) {
        self.show_raw = show;
    }

    /// Toggle decomposition display
    pub fn set_show_decomposition(&mut self, show: bool) {
        self.show_decomposition = show;
    }

    /// Toggle arousal gauge
    pub fn set_show_arousal(&mut self, show: bool) {
        self.show_arousal = show;
    }

    /// Update display settings
    pub fn set_settings(&mut self, settings: DisplaySettings) {
        self.settings = settings;
        self.canvas.set_width(self.settings.width);
        self.canvas.set_height(self.settings.height);
    }

    /// Push raw conductance values (4 sites)
    pub fn push_raw(&mut self, conductance: &[f32]) -> Result<(), JsValue> {
        for (i, &value) in conductance.iter().take(4).enumerate() {
            self.conductance_buffers[i].push_back(value);

            while self.conductance_buffers[i].len() > MAX_BUFFER_SIZE {
                self.conductance_buffers[i].pop_front();
            }
        }
        Ok(())
    }

    /// Push decomposed SCL/SCR values
    ///
    /// `is_scr_peak` is a bitmask: bit 0 = site 0, bit 1 = site 1, etc.
    pub fn push_decomposed(
        &mut self,
        scl: &[f32],
        scr: &[f32],
        is_scr_peak: u8,
    ) -> Result<(), JsValue> {
        for i in 0..4.min(scl.len()) {
            self.current_scl[i] = scl[i];
            self.scl_buffers[i].push_back(scl[i]);
            self.scr_buffers[i].push_back(scr[i]);

            if (is_scr_peak >> i) & 1 == 1 {
                self.scr_count[i] += 1;
            }

            while self.scl_buffers[i].len() > MAX_BUFFER_SIZE {
                self.scl_buffers[i].pop_front();
            }
            while self.scr_buffers[i].len() > MAX_BUFFER_SIZE {
                self.scr_buffers[i].pop_front();
            }
        }
        Ok(())
    }

    /// Set arousal level
    pub fn set_arousal(&mut self, arousal: f32) {
        self.arousal = arousal.clamp(0.0, 1.0);
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for i in 0..4 {
            self.conductance_buffers[i].clear();
            self.scl_buffers[i].clear();
            self.scr_buffers[i].clear();
            self.scr_count[i] = 0;
        }
        self.current_scl = [0.0; 4];
        self.arousal = 0.0;
    }

    /// Render the EDA visualization
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.settings.width as f64;
        let height = self.settings.height as f64;

        // Clear canvas
        self.ctx.set_fill_style_str(&self.settings.get_background());
        self.ctx.fill_rect(0.0, 0.0, width, height);

        // Layout
        let arousal_width = if self.show_arousal { 120.0 } else { 0.0 };
        let signal_width = width - arousal_width - 20.0;

        let raw_height = if self.show_raw { height * 0.4 } else { 0.0 };
        let decomp_height = if self.show_decomposition { height - raw_height - 10.0 } else { 0.0 };

        // Draw raw signal
        if self.show_raw && raw_height > 50.0 {
            self.draw_raw_signal(10.0, 10.0, signal_width, raw_height - 20.0)?;
        }

        // Draw decomposition
        if self.show_decomposition && decomp_height > 50.0 {
            self.draw_decomposition(10.0, raw_height + 10.0, signal_width, decomp_height - 20.0)?;
        }

        // Draw arousal gauge
        if self.show_arousal {
            self.draw_arousal_gauge(signal_width + 20.0, 10.0, arousal_width - 10.0, height - 20.0)?;
        }

        Ok(())
    }

    /// Draw raw EDA signal
    fn draw_raw_signal(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        // Title
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("12px monospace");
        self.ctx.fill_text("Skin Conductance (µS)", x, y + 12.0)?;

        let plot_y = y + 25.0;
        let plot_height = height - 25.0;
        let samples_per_window = (self.sample_rate * self.settings.time_window_s) as usize;
        let x_scale = width / samples_per_window as f64;

        // Y scale: 0-25 µS typical range
        let y_min = 0.0;
        let y_max = 25.0;
        let y_scale = plot_height / (y_max - y_min);

        // Draw grid
        if self.settings.show_grid {
            self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.1)");
            self.ctx.set_line_width(0.5);

            for i in 0..=5 {
                let gy = plot_y + plot_height - (i as f64 * 5.0 * y_scale);
                self.ctx.begin_path();
                self.ctx.move_to(x, gy);
                self.ctx.line_to(x + width, gy);
                self.ctx.stroke();

                // Y-axis labels
                self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.5)");
                self.ctx.set_font("9px monospace");
                self.ctx.fill_text(&format!("{}", i * 5), x - 20.0, gy + 3.0)?;
            }
        }

        // Draw each site
        for (site, buffer) in self.conductance_buffers.iter().enumerate() {
            self.ctx.set_stroke_style_str(EDA_SITE_COLORS[site]);
            self.ctx.set_line_width(1.5);

            let start = buffer.len().saturating_sub(samples_per_window);

            self.ctx.begin_path();
            let mut first = true;

            for (i, &value) in buffer.iter().skip(start).enumerate() {
                let px = x + i as f64 * x_scale;
                let py = plot_y + plot_height - ((value as f64 - y_min) * y_scale);

                if first {
                    self.ctx.move_to(px, py);
                    first = false;
                } else {
                    self.ctx.line_to(px, py);
                }
            }

            self.ctx.stroke();
        }

        // Legend
        if self.settings.show_labels {
            for (i, name) in EDA_SITE_NAMES.iter().enumerate() {
                let legend_x = x + width - 250.0 + i as f64 * 60.0;
                self.ctx.set_fill_style_str(EDA_SITE_COLORS[i]);
                self.ctx.fill_rect(legend_x, plot_y, 10.0, 10.0);
                self.ctx.set_fill_style_str("#ffffff");
                self.ctx.set_font("9px monospace");
                self.ctx.fill_text(name, legend_x + 12.0, plot_y + 9.0)?;
            }
        }

        Ok(())
    }

    /// Draw SCL/SCR decomposition
    fn draw_decomposition(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        let half_height = (height - 30.0) / 2.0;

        // SCL section
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("12px monospace");
        self.ctx.fill_text("SCL (Tonic)", x, y + 12.0)?;

        let scl_y = y + 25.0;
        self.draw_signal_plot(&self.scl_buffers, x, scl_y, width, half_height - 10.0, 0.0, 25.0)?;

        // SCR section
        let scr_title_y = scl_y + half_height + 5.0;
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.fill_text("SCR (Phasic)", x, scr_title_y)?;

        let scr_y = scr_title_y + 12.0;
        self.draw_signal_plot(&self.scr_buffers, x, scr_y, width, half_height - 10.0, -1.0, 2.0)?;

        // SCR event counts
        self.ctx.set_font("10px monospace");
        for (i, &count) in self.scr_count.iter().enumerate() {
            let count_x = x + width - 200.0 + i as f64 * 50.0;
            self.ctx.set_fill_style_str(EDA_SITE_COLORS[i]);
            self.ctx.fill_text(&format!("{}:{}", EDA_SITE_NAMES[i].chars().next().unwrap(), count), count_x, scr_y + half_height - 5.0)?;
        }

        Ok(())
    }

    /// Helper to draw a signal plot
    fn draw_signal_plot(
        &self,
        buffers: &[VecDeque<f32>; 4],
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        y_min: f32,
        y_max: f32,
    ) -> Result<(), JsValue> {
        let samples_per_window = (self.sample_rate * self.settings.time_window_s) as usize;
        let x_scale = width / samples_per_window as f64;
        let y_scale = height as f64 / (y_max - y_min) as f64;

        for (site, buffer) in buffers.iter().enumerate() {
            self.ctx.set_stroke_style_str(EDA_SITE_COLORS[site]);
            self.ctx.set_line_width(1.5);

            let start = buffer.len().saturating_sub(samples_per_window);

            self.ctx.begin_path();
            let mut first = true;

            for (i, &value) in buffer.iter().skip(start).enumerate() {
                let px = x + i as f64 * x_scale;
                let py = y + height - (((value - y_min) as f64) * y_scale);

                if first {
                    self.ctx.move_to(px, py);
                    first = false;
                } else {
                    self.ctx.line_to(px, py);
                }
            }

            self.ctx.stroke();
        }

        Ok(())
    }

    /// Draw arousal gauge
    fn draw_arousal_gauge(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        let center_x = x + width / 2.0;
        let gauge_radius = (width.min(height * 0.4) / 2.0) - 10.0;
        let gauge_y = y + gauge_radius + 40.0;

        // Title
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("14px monospace");
        self.ctx.set_text_align("center");
        self.ctx.fill_text("Arousal", center_x, y + 20.0)?;
        self.ctx.set_text_align("left");

        // Draw gauge arc
        let start_angle = std::f64::consts::PI * 0.8;
        let end_angle = std::f64::consts::PI * 2.2;

        // Background arc
        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.2)");
        self.ctx.set_line_width(15.0);
        self.ctx.begin_path();
        self.ctx.arc(center_x, gauge_y, gauge_radius, start_angle, end_angle)?;
        self.ctx.stroke();

        // Arousal level arc
        let arousal_angle = start_angle + self.arousal as f64 * (end_angle - start_angle);
        let arousal_color = if self.arousal < 0.33 {
            "#60a5fa" // Low - blue
        } else if self.arousal < 0.66 {
            "#fbbf24" // Medium - yellow
        } else {
            "#f87171" // High - red
        };
        self.ctx.set_stroke_style_str(arousal_color);
        self.ctx.set_line_width(12.0);
        self.ctx.begin_path();
        self.ctx.arc(center_x, gauge_y, gauge_radius, start_angle, arousal_angle)?;
        self.ctx.stroke();

        // Center text
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("24px monospace");
        self.ctx.set_text_align("center");
        self.ctx.fill_text(&format!("{:.0}%", self.arousal * 100.0), center_x, gauge_y + 10.0)?;
        self.ctx.set_text_align("left");

        // Labels
        self.ctx.set_font("10px monospace");
        self.ctx.fill_text("Low", x, gauge_y + gauge_radius + 10.0)?;
        self.ctx.fill_text("High", x + width - 25.0, gauge_y + gauge_radius + 10.0)?;

        // SCL values
        let scl_y = gauge_y + gauge_radius + 40.0;
        self.ctx.set_font("11px monospace");
        self.ctx.fill_text("SCL (µS):", x, scl_y)?;

        for (i, &scl) in self.current_scl.iter().enumerate() {
            let row_y = scl_y + 15.0 + i as f64 * 18.0;
            self.ctx.set_fill_style_str(EDA_SITE_COLORS[i]);
            self.ctx.fill_text(EDA_SITE_NAMES[i], x, row_y)?;
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.fill_text(&format!("{:.1}", scl), x + 50.0, row_y)?;
        }

        Ok(())
    }
}

impl Default for EdaRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create EdaRenderer")
    }
}

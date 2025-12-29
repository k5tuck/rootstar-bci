//! fNIRS heatmap visualization
//!
//! Renders hemoglobin concentration changes as color-coded maps.

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::{ColorScheme, DisplaySettings};

/// Maximum number of channels to display
const MAX_FNIRS_CHANNELS: usize = 16;

/// fNIRS heatmap renderer
#[wasm_bindgen]
pub struct FnirsMapRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,
    /// HbO2 values per channel (µM)
    hbo2_values: Vec<f32>,
    /// HbR values per channel (µM)
    hbr_values: Vec<f32>,
    /// Channel positions (normalized 0-1)
    channel_positions: Vec<(f32, f32)>,
    /// Color scheme for HbO2
    hbo2_scheme: ColorScheme,
    /// Color scheme for HbR
    hbr_scheme: ColorScheme,
    /// Value range for HbO2 normalization
    hbo2_range: (f32, f32),
    /// Value range for HbR normalization
    hbr_range: (f32, f32),
    /// Display mode (HbO2, HbR, or both)
    display_mode: FnirsDisplayMode,
    /// Show channel labels
    show_labels: bool,
}

/// Display mode for fNIRS visualization
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FnirsDisplayMode {
    /// Show only HbO2 (oxygenated hemoglobin)
    Hbo2Only,
    /// Show only HbR (deoxygenated hemoglobin)
    HbrOnly,
    /// Show both side by side
    Both,
    /// Show total hemoglobin (HbO2 + HbR)
    Total,
    /// Show oxygenation index (HbO2 / (HbO2 + HbR))
    Oxygenation,
}

impl Default for FnirsDisplayMode {
    fn default() -> Self {
        Self::Hbo2Only
    }
}

#[wasm_bindgen]
impl FnirsMapRenderer {
    /// Create a new fNIRS heatmap renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<FnirsMapRenderer, JsValue> {
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

        canvas.set_width(600);
        canvas.set_height(400);

        // Default channel positions (prefrontal cortex grid)
        let channel_positions = vec![
            (0.25, 0.25),
            (0.5, 0.25),
            (0.75, 0.25),
            (0.25, 0.5),
            (0.5, 0.5),
            (0.75, 0.5),
            (0.25, 0.75),
            (0.5, 0.75),
            (0.75, 0.75),
        ];

        let n_channels = channel_positions.len();

        Ok(Self {
            canvas,
            ctx,
            settings: DisplaySettings::new(),
            hbo2_values: vec![0.0; n_channels],
            hbr_values: vec![0.0; n_channels],
            channel_positions,
            hbo2_scheme: ColorScheme::GreenRed,
            hbr_scheme: ColorScheme::BlueYellow,
            hbo2_range: (-5.0, 5.0),
            hbr_range: (-3.0, 3.0),
            display_mode: FnirsDisplayMode::Hbo2Only,
            show_labels: true,
        })
    }

    /// Get the canvas element
    pub fn canvas(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }

    /// Set display mode
    pub fn set_display_mode(&mut self, mode: FnirsDisplayMode) {
        self.display_mode = mode;
    }

    /// Set HbO2 value range
    pub fn set_hbo2_range(&mut self, min: f32, max: f32) {
        self.hbo2_range = (min, max);
    }

    /// Set HbR value range
    pub fn set_hbr_range(&mut self, min: f32, max: f32) {
        self.hbr_range = (min, max);
    }

    /// Toggle labels
    pub fn set_show_labels(&mut self, show: bool) {
        self.show_labels = show;
    }

    /// Set channel positions
    pub fn set_channel_positions(&mut self, positions: Vec<f32>) {
        // Positions come as flat array [x1, y1, x2, y2, ...]
        self.channel_positions = positions
            .chunks(2)
            .filter_map(|chunk| {
                if chunk.len() == 2 {
                    Some((chunk[0], chunk[1]))
                } else {
                    None
                }
            })
            .take(MAX_FNIRS_CHANNELS)
            .collect();

        let n = self.channel_positions.len();
        self.hbo2_values.resize(n, 0.0);
        self.hbr_values.resize(n, 0.0);
    }

    /// Update with raw HbO2 and HbR values (µM)
    pub fn update_raw(&mut self, hbo2: &[f32], hbr: &[f32]) -> Result<(), JsValue> {
        for (i, &v) in hbo2.iter().take(self.hbo2_values.len()).enumerate() {
            self.hbo2_values[i] = v;
        }
        for (i, &v) in hbr.iter().take(self.hbr_values.len()).enumerate() {
            self.hbr_values[i] = v;
        }
        Ok(())
    }

    /// Render the fNIRS map
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.canvas.width();
        let height = self.canvas.height();

        // Clear canvas
        self.ctx.set_fill_style_str(&self.settings.get_background());
        self.ctx.fill_rect(0.0, 0.0, width as f64, height as f64);

        match self.display_mode {
            FnirsDisplayMode::Hbo2Only => {
                self.render_single_map(
                    &self.hbo2_values,
                    &self.hbo2_scheme,
                    self.hbo2_range,
                    "HbO₂ (µM)",
                    0.0,
                    0.0,
                    width as f64,
                    height as f64,
                )?;
            }
            FnirsDisplayMode::HbrOnly => {
                self.render_single_map(
                    &self.hbr_values,
                    &self.hbr_scheme,
                    self.hbr_range,
                    "HbR (µM)",
                    0.0,
                    0.0,
                    width as f64,
                    height as f64,
                )?;
            }
            FnirsDisplayMode::Both => {
                let half_width = width as f64 / 2.0 - 10.0;
                self.render_single_map(
                    &self.hbo2_values,
                    &self.hbo2_scheme,
                    self.hbo2_range,
                    "HbO₂",
                    0.0,
                    0.0,
                    half_width,
                    height as f64,
                )?;
                self.render_single_map(
                    &self.hbr_values,
                    &self.hbr_scheme,
                    self.hbr_range,
                    "HbR",
                    half_width + 20.0,
                    0.0,
                    half_width,
                    height as f64,
                )?;
            }
            FnirsDisplayMode::Total => {
                let total: Vec<f32> = self
                    .hbo2_values
                    .iter()
                    .zip(self.hbr_values.iter())
                    .map(|(&o, &r)| o + r)
                    .collect();
                self.render_single_map(
                    &total,
                    &ColorScheme::Viridis,
                    (-8.0, 8.0),
                    "HbT (µM)",
                    0.0,
                    0.0,
                    width as f64,
                    height as f64,
                )?;
            }
            FnirsDisplayMode::Oxygenation => {
                let oxy: Vec<f32> = self
                    .hbo2_values
                    .iter()
                    .zip(self.hbr_values.iter())
                    .map(|(&o, &r)| {
                        let total = o.abs() + r.abs();
                        if total > 0.01 {
                            o / total
                        } else {
                            0.5
                        }
                    })
                    .collect();
                self.render_single_map(
                    &oxy,
                    &ColorScheme::BlueRed,
                    (0.0, 1.0),
                    "Oxygenation",
                    0.0,
                    0.0,
                    width as f64,
                    height as f64,
                )?;
            }
        }

        Ok(())
    }

    /// Render a single heatmap
    #[allow(clippy::too_many_arguments)]
    fn render_single_map(
        &self,
        values: &[f32],
        scheme: &ColorScheme,
        range: (f32, f32),
        title: &str,
        x_offset: f64,
        y_offset: f64,
        width: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        let margin = 40.0;
        let map_width = width - 2.0 * margin;
        let map_height = height - 2.0 * margin - 20.0; // Extra space for title

        // Draw title
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("14px monospace");
        self.ctx.fill_text(title, x_offset + margin, y_offset + 20.0)?;

        // Draw interpolated surface
        let resolution = 8;
        for py in (0..map_height as u32).step_by(resolution) {
            for px in (0..map_width as u32).step_by(resolution) {
                let nx = px as f32 / map_width as f32;
                let ny = py as f32 / map_height as f32;

                let value = self.interpolate_fnirs(nx, ny, values);
                let normalized = ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0);
                let color = scheme.to_css(normalized);

                self.ctx.set_fill_style_str(&color);
                self.ctx.fill_rect(
                    x_offset + margin + px as f64,
                    y_offset + margin + 20.0 + py as f64,
                    resolution as f64,
                    resolution as f64,
                );
            }
        }

        // Draw channel markers
        for (i, &(cx, cy)) in self.channel_positions.iter().enumerate() {
            let x = x_offset + margin + cx as f64 * map_width;
            let y = y_offset + margin + 20.0 + cy as f64 * map_height;

            // Draw circle
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.begin_path();
            self.ctx.arc(x, y, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.ctx.fill();

            self.ctx.set_stroke_style_str("#000000");
            self.ctx.set_line_width(1.0);
            self.ctx.begin_path();
            self.ctx.arc(x, y, 6.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.ctx.stroke();

            // Draw label
            if self.show_labels && i < values.len() {
                self.ctx.set_fill_style_str("#ffffff");
                self.ctx.set_font("9px monospace");
                let label = format!("{:.1}", values[i]);
                self.ctx.fill_text(&label, x + 8.0, y + 3.0)?;
            }
        }

        // Draw color bar
        self.draw_colorbar_fnirs(
            scheme,
            range,
            x_offset + width - 30.0,
            y_offset + margin + 20.0,
            map_height,
        )?;

        Ok(())
    }

    /// Interpolate fNIRS value at a point
    fn interpolate_fnirs(&self, x: f32, y: f32, values: &[f32]) -> f32 {
        let mut sum_weights = 0.0f32;
        let mut sum_values = 0.0f32;

        for (i, &(cx, cy)) in self.channel_positions.iter().enumerate() {
            if i >= values.len() {
                break;
            }

            let dx = x - cx;
            let dy = y - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.001 {
                return values[i];
            }

            let weight = 1.0 / dist.powi(2);
            sum_weights += weight;
            sum_values += weight * values[i];
        }

        if sum_weights > 0.0 {
            sum_values / sum_weights
        } else {
            0.0
        }
    }

    /// Draw color bar for fNIRS
    fn draw_colorbar_fnirs(
        &self,
        scheme: &ColorScheme,
        range: (f32, f32),
        x: f64,
        y: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        let bar_width = 15.0;

        // Draw gradient
        for i in 0..height as u32 {
            let normalized = 1.0 - (i as f32 / height as f32);
            let color = scheme.to_css(normalized);
            self.ctx.set_fill_style_str(&color);
            self.ctx.fill_rect(x, y + i as f64, bar_width, 1.0);
        }

        // Draw border
        self.ctx.set_stroke_style_str("#ffffff");
        self.ctx.set_line_width(1.0);
        self.ctx.stroke_rect(x, y, bar_width, height);

        // Draw labels
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("9px monospace");
        self.ctx
            .fill_text(&format!("{:.1}", range.1), x + bar_width + 3.0, y + 8.0)?;
        self.ctx
            .fill_text(&format!("{:.1}", range.0), x + bar_width + 3.0, y + height)?;

        Ok(())
    }

    /// Get current HbO2 values
    pub fn get_hbo2_values(&self) -> Vec<f32> {
        self.hbo2_values.clone()
    }

    /// Get current HbR values
    pub fn get_hbr_values(&self) -> Vec<f32> {
        self.hbr_values.clone()
    }

    /// Clear all values
    pub fn clear(&mut self) {
        for v in &mut self.hbo2_values {
            *v = 0.0;
        }
        for v in &mut self.hbr_values {
            *v = 0.0;
        }
    }
}

impl Default for FnirsMapRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create FnirsMapRenderer")
    }
}

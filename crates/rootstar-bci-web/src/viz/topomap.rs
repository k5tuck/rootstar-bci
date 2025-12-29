//! EEG topographic map visualization
//!
//! Renders scalp topography with interpolated potentials.

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::{ColorScheme, DisplaySettings, CHANNEL_NAMES, ELECTRODE_POSITIONS};

/// Topographic map renderer
#[wasm_bindgen]
pub struct TopomapRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,
    /// Current channel values (µV)
    values: [f32; 8],
    /// Color scheme for interpolation
    color_scheme: ColorScheme,
    /// Value range for normalization
    value_range: (f32, f32),
    /// Show electrode markers
    show_electrodes: bool,
    /// Show contour lines
    show_contours: bool,
    /// Interpolation resolution (pixels per cell)
    resolution: u32,
}

#[wasm_bindgen]
impl TopomapRenderer {
    /// Create a new topographic map renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<TopomapRenderer, JsValue> {
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
        canvas.set_width(400);
        canvas.set_height(400);

        Ok(Self {
            canvas,
            ctx,
            settings,
            values: [0.0; 8],
            color_scheme: ColorScheme::BlueRed,
            value_range: (-100.0, 100.0),
            show_electrodes: true,
            show_contours: true,
            resolution: 4,
        })
    }

    /// Get the canvas element
    pub fn canvas(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }

    /// Set color scheme
    pub fn set_color_scheme(&mut self, scheme: ColorScheme) {
        self.color_scheme = scheme;
    }

    /// Set value range for normalization
    pub fn set_value_range(&mut self, min: f32, max: f32) {
        self.value_range = (min, max);
    }

    /// Set interpolation resolution
    pub fn set_resolution(&mut self, resolution: u32) {
        self.resolution = resolution.max(1).min(20);
    }

    /// Toggle electrode markers
    pub fn set_show_electrodes(&mut self, show: bool) {
        self.show_electrodes = show;
    }

    /// Toggle contour lines
    pub fn set_show_contours(&mut self, show: bool) {
        self.show_contours = show;
    }

    /// Update with raw channel values (µV)
    pub fn update_raw(&mut self, values: &[f32]) -> Result<(), JsValue> {
        for (i, &value) in values.iter().take(8).enumerate() {
            self.values[i] = value;
        }
        Ok(())
    }

    /// Render the topographic map
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.canvas.width();
        let height = self.canvas.height();
        let size = width.min(height);

        // Clear canvas
        self.ctx.set_fill_style_str(&self.settings.get_background());
        self.ctx.fill_rect(0.0, 0.0, width as f64, height as f64);

        // Draw head outline
        self.draw_head_outline(size)?;

        // Draw interpolated surface
        self.draw_interpolated_surface(size)?;

        // Draw contours if enabled
        if self.show_contours {
            self.draw_contours(size)?;
        }

        // Draw electrode positions
        if self.show_electrodes {
            self.draw_electrodes(size)?;
        }

        // Draw color bar
        self.draw_colorbar(width, height)?;

        Ok(())
    }

    /// Draw head outline (nose at top)
    fn draw_head_outline(&self, size: u32) -> Result<(), JsValue> {
        let cx = size as f64 / 2.0;
        let cy = size as f64 / 2.0;
        let radius = size as f64 * 0.4;

        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.5)");
        self.ctx.set_line_width(2.0);

        // Head circle
        self.ctx.begin_path();
        self.ctx
            .arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.stroke();

        // Nose indicator
        self.ctx.begin_path();
        self.ctx.move_to(cx - 10.0, cy - radius);
        self.ctx.line_to(cx, cy - radius - 15.0);
        self.ctx.line_to(cx + 10.0, cy - radius);
        self.ctx.stroke();

        // Ears
        self.ctx.begin_path();
        self.ctx.ellipse(
            cx - radius - 5.0,
            cy,
            5.0,
            15.0,
            0.0,
            0.0,
            std::f64::consts::PI * 2.0,
        )?;
        self.ctx.stroke();

        self.ctx.begin_path();
        self.ctx.ellipse(
            cx + radius + 5.0,
            cy,
            5.0,
            15.0,
            0.0,
            0.0,
            std::f64::consts::PI * 2.0,
        )?;
        self.ctx.stroke();

        Ok(())
    }

    /// Draw interpolated surface using inverse distance weighting
    fn draw_interpolated_surface(&self, size: u32) -> Result<(), JsValue> {
        let cx = size as f64 / 2.0;
        let cy = size as f64 / 2.0;
        let radius = size as f64 * 0.4;
        let res = self.resolution;

        // Create pixel-wise interpolation
        for y in (0..size).step_by(res as usize) {
            for x in (0..size).step_by(res as usize) {
                let px = (x as f64 - cx) / radius;
                let py = (y as f64 - cy) / radius;

                // Check if inside head
                if px * px + py * py > 1.0 {
                    continue;
                }

                // Inverse distance weighted interpolation
                let value = self.interpolate_value(px as f32, py as f32);
                let normalized = self.normalize_value(value);
                let color = self.color_scheme.to_css(normalized);

                self.ctx.set_fill_style_str(&color);
                self.ctx.fill_rect(x as f64, y as f64, res as f64, res as f64);
            }
        }

        Ok(())
    }

    /// Interpolate value at a point using inverse distance weighting
    fn interpolate_value(&self, x: f32, y: f32) -> f32 {
        let mut sum_weights = 0.0f32;
        let mut sum_values = 0.0f32;
        let power = 2.0f32;

        for (i, &(ex, ey)) in ELECTRODE_POSITIONS.iter().enumerate() {
            // Convert electrode position to centered coordinates
            let ex_centered = ex * 2.0 - 1.0;
            let ey_centered = ey * 2.0 - 1.0;

            let dx = x - ex_centered;
            let dy = y - ey_centered;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.001 {
                // Very close to electrode, return its value directly
                return self.values[i];
            }

            let weight = 1.0 / dist.powf(power);
            sum_weights += weight;
            sum_values += weight * self.values[i];
        }

        if sum_weights > 0.0 {
            sum_values / sum_weights
        } else {
            0.0
        }
    }

    /// Normalize value to 0-1 range
    fn normalize_value(&self, value: f32) -> f32 {
        let (min, max) = self.value_range;
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Draw contour lines
    fn draw_contours(&self, size: u32) -> Result<(), JsValue> {
        let cx = size as f64 / 2.0;
        let cy = size as f64 / 2.0;
        let radius = size as f64 * 0.4;
        let n_contours = 5;

        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.3)");
        self.ctx.set_line_width(0.5);

        let (min, max) = self.value_range;
        let step = (max - min) / (n_contours + 1) as f32;

        for i in 1..=n_contours {
            let contour_value = min + step * i as f32;
            self.draw_single_contour(cx, cy, radius, contour_value)?;
        }

        Ok(())
    }

    /// Draw a single contour line (simplified marching squares)
    fn draw_single_contour(
        &self,
        cx: f64,
        cy: f64,
        radius: f64,
        _contour_value: f32,
    ) -> Result<(), JsValue> {
        // Simplified contour - just draw circles at different radii
        // A full implementation would use marching squares algorithm
        let contour_radius = radius * 0.3;
        self.ctx.begin_path();
        self.ctx
            .arc(cx, cy, contour_radius, 0.0, std::f64::consts::PI * 2.0)?;
        self.ctx.stroke();
        Ok(())
    }

    /// Draw electrode positions and labels
    fn draw_electrodes(&self, size: u32) -> Result<(), JsValue> {
        let cx = size as f64 / 2.0;
        let cy = size as f64 / 2.0;
        let radius = size as f64 * 0.4;

        for (i, &(ex, ey)) in ELECTRODE_POSITIONS.iter().enumerate() {
            // Convert to canvas coordinates
            let x = cx + (ex * 2.0 - 1.0) as f64 * radius;
            let y = cy + (ey * 2.0 - 1.0) as f64 * radius;

            // Draw electrode marker
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.begin_path();
            self.ctx.arc(x, y, 4.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.ctx.fill();

            // Draw outline
            self.ctx.set_stroke_style_str("#000000");
            self.ctx.set_line_width(1.0);
            self.ctx.begin_path();
            self.ctx.arc(x, y, 4.0, 0.0, std::f64::consts::PI * 2.0)?;
            self.ctx.stroke();

            // Draw label
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.set_font("10px monospace");
            self.ctx.fill_text(CHANNEL_NAMES[i], x + 6.0, y + 3.0)?;

            // Draw value
            let value_text = format!("{:.0}", self.values[i]);
            self.ctx.set_font("8px monospace");
            self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.7)");
            self.ctx.fill_text(&value_text, x + 6.0, y + 12.0)?;
        }

        Ok(())
    }

    /// Draw color scale bar
    fn draw_colorbar(&self, width: u32, height: u32) -> Result<(), JsValue> {
        let bar_width = 20.0;
        let bar_height = height as f64 * 0.6;
        let bar_x = width as f64 - 40.0;
        let bar_y = (height as f64 - bar_height) / 2.0;

        // Draw gradient bar
        for i in 0..bar_height as u32 {
            let normalized = 1.0 - (i as f32 / bar_height as f32);
            let color = self.color_scheme.to_css(normalized);
            self.ctx.set_fill_style_str(&color);
            self.ctx.fill_rect(bar_x, bar_y + i as f64, bar_width, 1.0);
        }

        // Draw border
        self.ctx.set_stroke_style_str("#ffffff");
        self.ctx.set_line_width(1.0);
        self.ctx.stroke_rect(bar_x, bar_y, bar_width, bar_height);

        // Draw labels
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("10px monospace");
        let (min, max) = self.value_range;
        self.ctx
            .fill_text(&format!("{:.0}", max), bar_x, bar_y - 5.0)?;
        self.ctx
            .fill_text(&format!("{:.0}", min), bar_x, bar_y + bar_height + 12.0)?;
        self.ctx
            .fill_text("µV", bar_x + 2.0, bar_y + bar_height / 2.0)?;

        Ok(())
    }
}

impl Default for TopomapRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create TopomapRenderer")
    }
}

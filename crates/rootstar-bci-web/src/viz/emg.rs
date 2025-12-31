//! Real-time EMG (Electromyography) visualization
//!
//! Renders facial muscle activity with bar graphs and valence indicator.

use std::collections::VecDeque;

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::DisplaySettings;

/// Maximum samples to buffer per channel
const MAX_BUFFER_SIZE: usize = 2000;

/// EMG channel names
const EMG_CHANNEL_NAMES: [&str; 8] = [
    "Zyg-L",   // Zygomaticus Left (smile)
    "Zyg-R",   // Zygomaticus Right
    "Cor-L",   // Corrugator Left (frown)
    "Cor-R",   // Corrugator Right
    "Mas-L",   // Masseter Left (jaw)
    "Mas-R",   // Masseter Right
    "Orb-U",   // Orbicularis Upper (lip)
    "Orb-D",   // Orbicularis Down
];

/// Channel colors (warm for smile muscles, cool for frown)
const EMG_CHANNEL_COLORS: [&str; 8] = [
    "#4ade80", // Zyg-L - Green (positive valence)
    "#22c55e", // Zyg-R - Green
    "#f87171", // Cor-L - Red (negative valence)
    "#ef4444", // Cor-R - Red
    "#60a5fa", // Mas-L - Blue (neutral)
    "#3b82f6", // Mas-R - Blue
    "#a78bfa", // Orb-U - Purple
    "#8b5cf6", // Orb-D - Purple
];

/// EMG visualization renderer
#[wasm_bindgen]
pub struct EmgRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,
    /// RMS buffers per channel
    rms_buffers: [VecDeque<f32>; 8],
    /// Current RMS values
    current_rms: [f32; 8],
    /// Current valence score (-1 to 1)
    valence: f32,
    /// Current arousal score (0 to 1)
    arousal: f32,
    /// Sample rate (Hz)
    sample_rate: f32,
    /// Show timeseries waveforms
    show_waveforms: bool,
    /// Show bar graph
    show_bars: bool,
    /// Show valence indicator
    show_valence: bool,
}

#[wasm_bindgen]
impl EmgRenderer {
    /// Create a new EMG renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<EmgRenderer, JsValue> {
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

        let rms_buffers: [VecDeque<f32>; 8] = core::array::from_fn(|_| {
            VecDeque::with_capacity(MAX_BUFFER_SIZE)
        });

        Ok(Self {
            canvas,
            ctx,
            settings,
            rms_buffers,
            current_rms: [0.0; 8],
            valence: 0.0,
            arousal: 0.0,
            sample_rate: 500.0,
            show_waveforms: true,
            show_bars: true,
            show_valence: true,
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

    /// Toggle waveform display
    pub fn set_show_waveforms(&mut self, show: bool) {
        self.show_waveforms = show;
    }

    /// Toggle bar graph display
    pub fn set_show_bars(&mut self, show: bool) {
        self.show_bars = show;
    }

    /// Toggle valence indicator
    pub fn set_show_valence(&mut self, show: bool) {
        self.show_valence = show;
    }

    /// Update display settings
    pub fn set_settings(&mut self, settings: DisplaySettings) {
        self.settings = settings;
        self.canvas.set_width(self.settings.width);
        self.canvas.set_height(self.settings.height);
    }

    /// Push EMG RMS values (8 channels)
    pub fn push_rms(&mut self, channels: &[f32]) -> Result<(), JsValue> {
        for (i, &value) in channels.iter().take(8).enumerate() {
            self.current_rms[i] = value;
            self.rms_buffers[i].push_back(value);

            while self.rms_buffers[i].len() > MAX_BUFFER_SIZE {
                self.rms_buffers[i].pop_front();
            }
        }
        Ok(())
    }

    /// Update valence and arousal scores
    pub fn set_valence_arousal(&mut self, valence: f32, arousal: f32) {
        self.valence = valence.clamp(-1.0, 1.0);
        self.arousal = arousal.clamp(0.0, 1.0);
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in &mut self.rms_buffers {
            buffer.clear();
        }
        self.current_rms = [0.0; 8];
        self.valence = 0.0;
        self.arousal = 0.0;
    }

    /// Render the EMG visualization
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.settings.width as f64;
        let height = self.settings.height as f64;

        // Clear canvas
        self.ctx.set_fill_style_str(&self.settings.get_background());
        self.ctx.fill_rect(0.0, 0.0, width, height);

        // Layout sections
        let bar_section_width = if self.show_bars { 150.0 } else { 0.0 };
        let valence_section_width = if self.show_valence { 100.0 } else { 0.0 };
        let waveform_width = width - bar_section_width - valence_section_width - 20.0;

        // Draw waveforms
        if self.show_waveforms && waveform_width > 100.0 {
            self.draw_waveforms(10.0, 10.0, waveform_width, height - 20.0)?;
        }

        // Draw bar graph
        if self.show_bars {
            let bar_x = waveform_width + 20.0;
            self.draw_bar_graph(bar_x, 10.0, bar_section_width - 10.0, height - 20.0)?;
        }

        // Draw valence indicator
        if self.show_valence {
            let valence_x = waveform_width + bar_section_width + 10.0;
            self.draw_valence_indicator(valence_x, 10.0, valence_section_width - 10.0, height - 20.0)?;
        }

        Ok(())
    }

    /// Draw EMG waveforms (stacked timeseries)
    fn draw_waveforms(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        let channel_height = height / 8.0;
        let samples_per_window = (self.sample_rate * self.settings.time_window_s) as usize;
        let x_scale = width / samples_per_window as f64;

        // Draw grid
        if self.settings.show_grid {
            self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.1)");
            self.ctx.set_line_width(0.5);

            for i in 0..=8 {
                let cy = y + i as f64 * channel_height;
                self.ctx.begin_path();
                self.ctx.move_to(x, cy);
                self.ctx.line_to(x + width, cy);
                self.ctx.stroke();
            }
        }

        // Draw each channel
        for (ch, buffer) in self.rms_buffers.iter().enumerate() {
            let y_offset = y + channel_height * (ch as f64 + 0.5);
            let y_scale = channel_height / (2.0 * 100.0); // 100µV scale

            self.ctx.set_stroke_style_str(EMG_CHANNEL_COLORS[ch]);
            self.ctx.set_line_width(1.5);

            let start = if buffer.len() > samples_per_window {
                buffer.len() - samples_per_window
            } else {
                0
            };

            self.ctx.begin_path();
            let mut first = true;

            for (i, &value) in buffer.iter().skip(start).enumerate() {
                let px = x + i as f64 * x_scale;
                let py = y_offset - (value as f64 * y_scale);

                if first {
                    self.ctx.move_to(px, py);
                    first = false;
                } else {
                    self.ctx.line_to(px, py);
                }
            }

            self.ctx.stroke();

            // Draw channel label
            if self.settings.show_labels {
                self.ctx.set_fill_style_str(EMG_CHANNEL_COLORS[ch]);
                self.ctx.set_font("10px monospace");
                self.ctx.fill_text(EMG_CHANNEL_NAMES[ch], x + 5.0, y_offset - channel_height * 0.3)?;
            }
        }

        Ok(())
    }

    /// Draw bar graph of current RMS values
    fn draw_bar_graph(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        let bar_height = (height - 20.0) / 8.0;
        let max_rms = 100.0; // 100µV max

        // Title
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("12px monospace");
        self.ctx.fill_text("RMS (µV)", x, y + 12.0)?;

        for (i, &rms) in self.current_rms.iter().enumerate() {
            let bar_y = y + 20.0 + i as f64 * bar_height;
            let bar_width = (rms / max_rms).min(1.0) as f64 * (width - 50.0);

            // Background
            self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.1)");
            self.ctx.fill_rect(x + 40.0, bar_y + 2.0, width - 50.0, bar_height - 4.0);

            // Bar
            self.ctx.set_fill_style_str(EMG_CHANNEL_COLORS[i]);
            self.ctx.fill_rect(x + 40.0, bar_y + 2.0, bar_width, bar_height - 4.0);

            // Label
            self.ctx.set_fill_style_str("#ffffff");
            self.ctx.set_font("9px monospace");
            self.ctx.fill_text(EMG_CHANNEL_NAMES[i], x, bar_y + bar_height * 0.7)?;
        }

        Ok(())
    }

    /// Draw valence/arousal indicator
    fn draw_valence_indicator(&self, x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        let center_x = x + width / 2.0;
        let indicator_height = height / 2.0 - 40.0;

        // Title
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("12px monospace");
        self.ctx.fill_text("Valence", x, y + 12.0)?;

        // Valence bar (vertical, centered)
        let bar_width = 30.0;
        let bar_y = y + 30.0;

        // Background gradient
        self.ctx.set_fill_style_str("#ef4444"); // Red at top
        self.ctx.fill_rect(center_x - bar_width / 2.0, bar_y, bar_width, indicator_height / 2.0);
        self.ctx.set_fill_style_str("#4ade80"); // Green at bottom
        self.ctx.fill_rect(center_x - bar_width / 2.0, bar_y + indicator_height / 2.0, bar_width, indicator_height / 2.0);

        // Valence marker
        let marker_y = bar_y + indicator_height / 2.0 - (self.valence as f64 * indicator_height / 2.0);
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.begin_path();
        self.ctx.move_to(center_x - bar_width / 2.0 - 5.0, marker_y);
        self.ctx.line_to(center_x + bar_width / 2.0 + 5.0, marker_y);
        self.ctx.set_line_width(3.0);
        self.ctx.stroke();

        // Valence labels
        self.ctx.set_fill_style_str("#ffffff");
        self.ctx.set_font("10px monospace");
        self.ctx.fill_text("+", center_x - 5.0, bar_y + 12.0)?;
        self.ctx.fill_text("-", center_x - 5.0, bar_y + indicator_height - 5.0)?;

        // Arousal indicator
        let arousal_y = y + indicator_height + 60.0;
        self.ctx.fill_text("Arousal", x, arousal_y)?;

        let arousal_bar_y = arousal_y + 15.0;
        self.ctx.set_fill_style_str("rgba(255, 255, 255, 0.2)");
        self.ctx.fill_rect(x, arousal_bar_y, width, 20.0);

        // Arousal level
        let arousal_fill = self.arousal as f64 * width;
        let arousal_color = if self.arousal < 0.33 {
            "#60a5fa" // Low - blue
        } else if self.arousal < 0.66 {
            "#fbbf24" // Medium - yellow
        } else {
            "#f87171" // High - red
        };
        self.ctx.set_fill_style_str(arousal_color);
        self.ctx.fill_rect(x, arousal_bar_y, arousal_fill, 20.0);

        // Arousal value
        self.ctx.set_fill_style_str("#ffffff");
        let arousal_text = format!("{:.0}%", self.arousal * 100.0);
        self.ctx.fill_text(&arousal_text, x + width / 2.0 - 15.0, arousal_bar_y + 14.0)?;

        Ok(())
    }
}

impl Default for EmgRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create EmgRenderer")
    }
}

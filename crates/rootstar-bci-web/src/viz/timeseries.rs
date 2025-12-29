//! Real-time EEG timeseries visualization
//!
//! Renders scrolling waveforms for each EEG channel.

use std::collections::VecDeque;

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use super::{DisplaySettings, CHANNEL_NAMES};

/// Maximum samples to buffer per channel
const MAX_BUFFER_SIZE: usize = 10000;

/// Channel colors for differentiation
const CHANNEL_COLORS: [&str; 8] = [
    "#ff6b6b", // Fp1 - Red
    "#4ecdc4", // Fp2 - Teal
    "#45b7d1", // C3 - Blue
    "#96ceb4", // C4 - Green
    "#ffeaa7", // P3 - Yellow
    "#dda0dd", // P4 - Plum
    "#98d8c8", // O1 - Mint
    "#f7dc6f", // O2 - Gold
];

/// EEG timeseries renderer
#[wasm_bindgen]
pub struct TimeseriesRenderer {
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// 2D rendering context
    ctx: CanvasRenderingContext2d,
    /// Display settings
    settings: DisplaySettings,
    /// Sample buffers per channel
    buffers: [VecDeque<f32>; 8],
    /// Sample rate (Hz)
    sample_rate: f32,
    /// Whether each channel is enabled
    channel_enabled: [bool; 8],
    /// Vertical offset per channel (for stacked display)
    channel_offsets: [f32; 8],
}

#[wasm_bindgen]
impl TimeseriesRenderer {
    /// Create a new timeseries renderer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<TimeseriesRenderer, JsValue> {
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

        // Initialize buffers
        let buffers: [VecDeque<f32>; 8] = core::array::from_fn(|_| {
            VecDeque::with_capacity(MAX_BUFFER_SIZE)
        });

        // Calculate vertical offsets for stacked display
        let channel_height = settings.height as f32 / 8.0;
        let channel_offsets: [f32; 8] = core::array::from_fn(|i| {
            channel_height * (i as f32 + 0.5)
        });

        Ok(Self {
            canvas,
            ctx,
            settings,
            buffers,
            sample_rate: 250.0,
            channel_enabled: [true; 8],
            channel_offsets,
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

    /// Enable/disable a channel
    pub fn set_channel_enabled(&mut self, channel: usize, enabled: bool) {
        if channel < 8 {
            self.channel_enabled[channel] = enabled;
        }
    }

    /// Update display settings
    pub fn set_settings(&mut self, settings: DisplaySettings) {
        self.settings = settings;
        self.canvas.set_width(self.settings.width);
        self.canvas.set_height(self.settings.height);

        // Recalculate offsets
        let channel_height = self.settings.height as f32 / 8.0;
        self.channel_offsets = core::array::from_fn(|i| {
            channel_height * (i as f32 + 0.5)
        });
    }

    /// Push raw sample data (8 channel values in µV)
    pub fn push_raw(&mut self, channels: &[f32]) -> Result<(), JsValue> {
        for (i, &value) in channels.iter().take(8).enumerate() {
            self.buffers[i].push_back(value);

            while self.buffers[i].len() > MAX_BUFFER_SIZE {
                self.buffers[i].pop_front();
            }
        }
        Ok(())
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
        }
    }

    /// Render the timeseries
    pub fn render(&self) -> Result<(), JsValue> {
        let width = self.settings.width as f64;
        let height = self.settings.height as f64;
        let channel_height = height / 8.0;

        // Clear canvas
        self.ctx.set_fill_style_str(&self.settings.get_background());
        self.ctx.fill_rect(0.0, 0.0, width, height);

        // Draw grid if enabled
        if self.settings.show_grid {
            self.draw_grid()?;
        }

        // Calculate samples to display
        let samples_per_window = (self.sample_rate * self.settings.time_window_s) as usize;
        let x_scale = width / samples_per_window as f64;

        // Draw each channel
        for (ch, buffer) in self.buffers.iter().enumerate() {
            if !self.channel_enabled[ch] {
                continue;
            }

            let y_offset = self.channel_offsets[ch] as f64;
            let y_scale = channel_height / (2.0 * self.settings.amplitude_scale as f64);

            // Set channel color
            self.ctx.set_stroke_style_str(CHANNEL_COLORS[ch]);
            self.ctx.set_line_width(1.5);

            // Get samples to display
            let start = if buffer.len() > samples_per_window {
                buffer.len() - samples_per_window
            } else {
                0
            };

            // Draw waveform
            self.ctx.begin_path();
            let mut first = true;

            for (i, &value) in buffer.iter().skip(start).enumerate() {
                let x = i as f64 * x_scale;
                let y = y_offset - (value as f64 * y_scale);

                if first {
                    self.ctx.move_to(x, y);
                    first = false;
                } else {
                    self.ctx.line_to(x, y);
                }
            }

            self.ctx.stroke();

            // Draw channel label
            if self.settings.show_labels {
                self.ctx.set_fill_style_str(CHANNEL_COLORS[ch]);
                self.ctx.set_font("12px monospace");
                self.ctx.fill_text(CHANNEL_NAMES[ch], 5.0, y_offset - channel_height * 0.3)?;
            }
        }

        Ok(())
    }

    /// Draw background grid
    fn draw_grid(&self) -> Result<(), JsValue> {
        let width = self.settings.width as f64;
        let height = self.settings.height as f64;
        let channel_height = height / 8.0;

        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.1)");
        self.ctx.set_line_width(0.5);

        // Horizontal lines (channel separators)
        for i in 0..9 {
            let y = i as f64 * channel_height;
            self.ctx.begin_path();
            self.ctx.move_to(0.0, y);
            self.ctx.line_to(width, y);
            self.ctx.stroke();
        }

        // Vertical lines (time divisions)
        let time_divisions = 10;
        let x_step = width / time_divisions as f64;
        for i in 0..=time_divisions {
            let x = i as f64 * x_step;
            self.ctx.begin_path();
            self.ctx.move_to(x, 0.0);
            self.ctx.line_to(x, height);
            self.ctx.stroke();
        }

        // Draw zero lines for each channel
        self.ctx.set_stroke_style_str("rgba(255, 255, 255, 0.2)");
        for y_offset in &self.channel_offsets {
            self.ctx.begin_path();
            self.ctx.move_to(0.0, *y_offset as f64);
            self.ctx.line_to(width, *y_offset as f64);
            self.ctx.stroke();
        }

        Ok(())
    }

    /// Get statistics for a channel
    pub fn get_channel_stats(&self, channel: usize) -> Option<ChannelStats> {
        if channel >= 8 || self.buffers[channel].is_empty() {
            return None;
        }

        let buffer = &self.buffers[channel];
        let n = buffer.len() as f32;

        let sum: f32 = buffer.iter().sum();
        let mean = sum / n;

        let variance: f32 = buffer.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let min = buffer.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = buffer.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Some(ChannelStats {
            mean,
            std_dev,
            min,
            max,
            sample_count: buffer.len(),
        })
    }
}

impl Default for TimeseriesRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create TimeseriesRenderer")
    }
}

/// Statistics for a single channel
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ChannelStats {
    /// Mean value (µV)
    pub mean: f32,
    /// Standard deviation (µV)
    pub std_dev: f32,
    /// Minimum value (µV)
    pub min: f32,
    /// Maximum value (µV)
    pub max: f32,
    /// Number of samples
    pub sample_count: usize,
}

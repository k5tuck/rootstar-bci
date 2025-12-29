//! Visualization modules for BCI data
//!
//! Provides real-time rendering of EEG and fNIRS data.

pub mod fnirs_map;
pub mod timeseries;
pub mod topomap;

pub use fnirs_map::FnirsMapRenderer;
pub use timeseries::TimeseriesRenderer;
pub use topomap::TopomapRenderer;

use wasm_bindgen::prelude::*;

/// Color scheme for visualizations
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorScheme {
    /// Blue to red gradient (default for EEG)
    BlueRed,
    /// Red to blue gradient (inverse)
    RedBlue,
    /// Green to red gradient (for fNIRS HbO2)
    GreenRed,
    /// Blue to yellow gradient (for fNIRS HbR)
    BlueYellow,
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::BlueRed
    }
}

impl ColorScheme {
    /// Convert a normalized value (0-1) to RGB color
    pub fn to_rgb(&self, value: f32) -> [u8; 3] {
        let v = value.clamp(0.0, 1.0);

        match self {
            ColorScheme::BlueRed => {
                // Blue (0) -> White (0.5) -> Red (1)
                if v < 0.5 {
                    let t = v * 2.0;
                    [
                        (t * 255.0) as u8,
                        (t * 255.0) as u8,
                        255,
                    ]
                } else {
                    let t = (v - 0.5) * 2.0;
                    [
                        255,
                        ((1.0 - t) * 255.0) as u8,
                        ((1.0 - t) * 255.0) as u8,
                    ]
                }
            }
            ColorScheme::RedBlue => {
                let rgb = Self::BlueRed.to_rgb(1.0 - v);
                rgb
            }
            ColorScheme::GreenRed => {
                // Green (0) -> Yellow (0.5) -> Red (1)
                if v < 0.5 {
                    let t = v * 2.0;
                    [
                        (t * 255.0) as u8,
                        255,
                        0,
                    ]
                } else {
                    let t = (v - 0.5) * 2.0;
                    [
                        255,
                        ((1.0 - t) * 255.0) as u8,
                        0,
                    ]
                }
            }
            ColorScheme::BlueYellow => {
                // Blue (0) -> Cyan (0.5) -> Yellow (1)
                if v < 0.5 {
                    let t = v * 2.0;
                    [
                        0,
                        (t * 255.0) as u8,
                        ((1.0 - t) * 255.0) as u8,
                    ]
                } else {
                    let t = (v - 0.5) * 2.0;
                    [
                        (t * 255.0) as u8,
                        255,
                        ((1.0 - t) * 255.0) as u8,
                    ]
                }
            }
            ColorScheme::Viridis => {
                // Simplified viridis approximation
                let r = (0.267 + v * (0.329 + v * (0.404 - v * 0.0))).min(1.0);
                let g = v * (0.675 + v * 0.325);
                let b = (0.329 + v * (0.533 - v * 0.862)).max(0.0);
                [
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                ]
            }
            ColorScheme::Plasma => {
                // Simplified plasma approximation
                let r = (0.050 + v * (0.850 + v * 0.100)).min(1.0);
                let g = v * v * 0.9;
                let b = (0.533 + v * (0.200 - v * 0.733)).max(0.0);
                [
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                ]
            }
        }
    }

    /// Convert a normalized value to CSS color string
    pub fn to_css(&self, value: f32) -> String {
        let [r, g, b] = self.to_rgb(value);
        format!("rgb({}, {}, {})", r, g, b)
    }
}

/// Visualization display settings
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DisplaySettings {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Time window in seconds (for timeseries)
    pub time_window_s: f32,
    /// Amplitude scale (µV per division)
    pub amplitude_scale: f32,
    /// Show grid lines
    pub show_grid: bool,
    /// Show channel labels
    pub show_labels: bool,
    /// Color scheme
    color_scheme: ColorScheme,
    /// Background color (CSS)
    background_color: String,
}

#[wasm_bindgen]
impl DisplaySettings {
    /// Create default display settings
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 600,
            time_window_s: 10.0,
            amplitude_scale: 100.0,
            show_grid: true,
            show_labels: true,
            color_scheme: ColorScheme::BlueRed,
            background_color: String::from("#1a1a2e"),
        }
    }

    /// Set color scheme
    pub fn set_color_scheme(&mut self, scheme: ColorScheme) {
        self.color_scheme = scheme;
    }

    /// Get color scheme
    pub fn get_color_scheme(&self) -> ColorScheme {
        self.color_scheme
    }

    /// Set background color
    pub fn set_background(&mut self, color: String) {
        self.background_color = color;
    }

    /// Get background color
    pub fn get_background(&self) -> String {
        self.background_color.clone()
    }
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self::new()
    }
}

/// 10-20 electrode positions (normalized coordinates)
pub const ELECTRODE_POSITIONS: [(f32, f32); 8] = [
    (0.15, 0.2),  // Fp1
    (0.85, 0.2),  // Fp2
    (0.25, 0.5),  // C3
    (0.75, 0.5),  // C4
    (0.25, 0.7),  // P3
    (0.75, 0.7),  // P4
    (0.15, 0.85), // O1
    (0.85, 0.85), // O2
];

/// Channel names for display
pub const CHANNEL_NAMES: [&str; 8] = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"];

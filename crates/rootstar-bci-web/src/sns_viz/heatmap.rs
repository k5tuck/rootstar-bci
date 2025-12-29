//! Activation Heatmap System
//!
//! Provides colormap-based visualization of receptor activation levels.

use std::f32::consts::PI;

/// Colormap for activation visualization
#[derive(Clone, Debug, PartialEq)]
pub enum Colormap {
    /// Perceptually uniform, good for scientific data
    Viridis,
    /// Purple to yellow, perceptually uniform
    Plasma,
    /// Black to yellow through red
    Inferno,
    /// Rainbow colormap (high perceptual variance)
    Turbo,
    /// Diverging blue-white-red for +/- values
    CoolWarm,
}

impl Colormap {
    /// Sample the colormap at parameter t (0.0 to 1.0)
    pub fn sample(&self, t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => Self::sample_viridis(t),
            Colormap::Plasma => Self::sample_plasma(t),
            Colormap::Inferno => Self::sample_inferno(t),
            Colormap::Turbo => Self::sample_turbo(t),
            Colormap::CoolWarm => Self::sample_coolwarm(t),
        }
    }

    fn sample_viridis(t: f32) -> [u8; 4] {
        // Viridis colormap approximation
        let r = (68.0 + t * (49.0 - 68.0 + t * (253.0 - 49.0))).clamp(0.0, 255.0) as u8;
        let g = (1.0 + t * (104.0 - 1.0 + t * (231.0 - 104.0))).clamp(0.0, 255.0) as u8;
        let b = (84.0 + t * (142.0 - 84.0 + t * (37.0 - 142.0))).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_plasma(t: f32) -> [u8; 4] {
        // Plasma colormap approximation
        let r = (13.0 + t * (240.0 - 13.0)).clamp(0.0, 255.0) as u8;
        let g = (8.0 + t * t * 240.0).clamp(0.0, 255.0) as u8;
        let b = (135.0 + t * (50.0 - 135.0)).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_inferno(t: f32) -> [u8; 4] {
        // Inferno colormap approximation
        let r = (t * 255.0).clamp(0.0, 255.0) as u8;
        let g = (t * t * 200.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - t) * 128.0 * (1.0 - t * t)).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_turbo(t: f32) -> [u8; 4] {
        // Turbo colormap approximation (rainbow-like)
        let r = (34.0 + 120.0 * (PI * (t - 0.3)).sin().max(0.0) * 2.0 + 135.0 * t * t).clamp(0.0, 255.0) as u8;
        let g = (30.0 + 220.0 * (PI * (t - 0.5) * 2.0).sin().max(0.0)).clamp(0.0, 255.0) as u8;
        let b = (130.0 + 125.0 * (PI * (t + 0.2)).sin().max(0.0) - 200.0 * t * t).clamp(0.0, 255.0) as u8;
        [r, g, b, 255]
    }

    fn sample_coolwarm(t: f32) -> [u8; 4] {
        // Blue (0) -> White (0.5) -> Red (1)
        let (r, g, b) = if t < 0.5 {
            let s = t * 2.0;
            (59.0 + s * 196.0, 76.0 + s * 179.0, 192.0 + s * 63.0)
        } else {
            let s = (t - 0.5) * 2.0;
            (255.0, 255.0 - s * 155.0, 255.0 - s * 195.0)
        };
        [r as u8, g as u8, b as u8, 255]
    }
}

impl Default for Colormap {
    fn default() -> Self {
        Colormap::Viridis
    }
}

/// Activation heatmap overlay system
#[derive(Clone, Debug)]
pub struct ActivationHeatmap {
    /// Colormap to use
    colormap: Colormap,
    /// Minimum activation value (maps to 0.0)
    min_value: f32,
    /// Maximum activation value (maps to 1.0)
    max_value: f32,
    /// Spatial smoothing radius (in UV space, 0.0-1.0)
    smoothing_radius: f32,
}

impl ActivationHeatmap {
    /// Create a new heatmap with specified colormap and smoothing
    pub fn new(colormap: Colormap, smoothing_radius: f32) -> Self {
        Self {
            colormap,
            min_value: 0.0,
            max_value: 100.0, // Default to 0-100 Hz firing rate range
            smoothing_radius: smoothing_radius.clamp(0.001, 0.2),
        }
    }

    /// Set the colormap
    pub fn set_colormap(&mut self, colormap: Colormap) {
        self.colormap = colormap;
    }

    /// Set the activation range
    pub fn set_range(&mut self, min_val: f32, max_val: f32) {
        self.min_value = min_val;
        self.max_value = max_val.max(min_val + 0.001);
    }

    /// Set smoothing radius
    pub fn set_smoothing(&mut self, radius: f32) {
        self.smoothing_radius = radius.clamp(0.001, 0.2);
    }

    /// Normalize an activation value to 0.0-1.0 range
    pub fn normalize(&self, activation: f32) -> f32 {
        ((activation - self.min_value) / (self.max_value - self.min_value)).clamp(0.0, 1.0)
    }

    /// Get the colormap
    pub fn colormap(&self) -> &Colormap {
        &self.colormap
    }

    /// Get smoothing radius
    pub fn smoothing_radius(&self) -> f32 {
        self.smoothing_radius
    }

    /// Get min value
    pub fn min_value(&self) -> f32 {
        self.min_value
    }

    /// Get max value
    pub fn max_value(&self) -> f32 {
        self.max_value
    }

    /// Render activation texture from receptor states
    pub fn render_to_pixels(
        &self,
        receptor_states: &[f32],
        receptor_uvs: &[[f32; 2]],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut pixels = vec![0u8; (width * height * 4) as usize];

        for (state, uv) in receptor_states.iter().zip(receptor_uvs.iter()) {
            let normalized = self.normalize(*state);
            let color = self.colormap.sample(normalized);

            let center_x = (uv[0] * width as f32) as i32;
            let center_y = (uv[1] * height as f32) as i32;
            let radius = (self.smoothing_radius * width as f32) as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = center_x + dx;
                    let py = center_y + dy;

                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        let dist = ((dx * dx + dy * dy) as f32).sqrt();
                        let sigma = self.smoothing_radius * width as f32;
                        let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();

                        let idx = ((py as u32 * width + px as u32) * 4) as usize;
                        // Alpha blend
                        for c in 0..3 {
                            let existing = pixels[idx + c] as f32;
                            let new_val = color[c] as f32;
                            pixels[idx + c] = (existing * (1.0 - weight) + new_val * weight) as u8;
                        }
                        pixels[idx + 3] = (pixels[idx + 3] as f32 + weight * 255.0).min(255.0) as u8;
                    }
                }
            }
        }

        pixels
    }
}

impl Default for ActivationHeatmap {
    fn default() -> Self {
        Self::new(Colormap::Viridis, 0.02)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_sample() {
        let viridis = Colormap::Viridis;
        let low = viridis.sample(0.0);
        let high = viridis.sample(1.0);

        // Viridis goes from dark purple to yellow
        assert!(low[2] > low[0]); // Blue > Red at low end
        assert!(high[0] > high[2]); // Red > Blue at high end
    }

    #[test]
    fn test_heatmap_normalize() {
        let mut heatmap = ActivationHeatmap::default();
        heatmap.set_range(0.0, 100.0);

        assert!((heatmap.normalize(0.0) - 0.0).abs() < 0.001);
        assert!((heatmap.normalize(50.0) - 0.5).abs() < 0.001);
        assert!((heatmap.normalize(100.0) - 1.0).abs() < 0.001);
        assert!((heatmap.normalize(-10.0) - 0.0).abs() < 0.001); // Clamped
        assert!((heatmap.normalize(200.0) - 1.0).abs() < 0.001); // Clamped
    }
}

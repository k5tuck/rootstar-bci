//! Feature extraction for ML models
//!
//! Extracts features from EEG and fNIRS data for classification.

use rootstar_bci_core::types::{EegChannel, EegSample};

use crate::processing::fft::{BandPowers, SpectralAnalyzer};
use crate::processing::fusion::AlignedSample;

/// Feature vector for ML classification
#[derive(Clone, Debug, Default)]
pub struct FeatureVector {
    /// EEG band powers (5 bands × 8 channels = 40 features)
    pub band_powers: [BandPowers; 8],
    /// Channel correlations (28 unique pairs)
    pub correlations: [f64; 28],
    /// fNIRS HbO₂ level
    pub hbo2: f64,
    /// fNIRS HbR level
    pub hbr: f64,
    /// Frontal alpha asymmetry
    pub frontal_asymmetry: f64,
    /// Timestamp
    pub timestamp_us: u64,
}

impl FeatureVector {
    /// Get total number of features
    #[must_use]
    pub const fn feature_count() -> usize {
        // 5 bands × 8 channels + 28 correlations + 3 fNIRS features
        40 + 28 + 3
    }

    /// Convert to flat feature array
    #[must_use]
    pub fn to_array(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(Self::feature_count());

        // Band powers
        for ch_powers in &self.band_powers {
            features.push(ch_powers.delta);
            features.push(ch_powers.theta);
            features.push(ch_powers.alpha);
            features.push(ch_powers.beta);
            features.push(ch_powers.gamma);
        }

        // Correlations
        features.extend_from_slice(&self.correlations);

        // fNIRS
        features.push(self.hbo2);
        features.push(self.hbr);
        features.push(self.frontal_asymmetry);

        features
    }
}

/// Feature extractor for BCI classification
pub struct FeatureExtractor {
    analyzer: SpectralAnalyzer,
    sample_buffer: Vec<[f64; 8]>,
    buffer_size: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    ///
    /// # Arguments
    ///
    /// * `fft_size` - FFT size for spectral analysis
    /// * `sample_rate` - EEG sample rate in Hz
    #[must_use]
    pub fn new(fft_size: usize, sample_rate: f64) -> Self {
        Self {
            analyzer: SpectralAnalyzer::new(fft_size, sample_rate),
            sample_buffer: Vec::with_capacity(fft_size),
            buffer_size: fft_size,
        }
    }

    /// Add an EEG sample to the buffer
    pub fn push_sample(&mut self, sample: &EegSample) {
        let channels: [f64; 8] = core::array::from_fn(|i| {
            f64::from(sample.channels[i].to_f32())
        });

        self.sample_buffer.push(channels);

        // Keep buffer at fixed size
        if self.sample_buffer.len() > self.buffer_size {
            self.sample_buffer.remove(0);
        }
    }

    /// Check if enough samples for feature extraction
    #[must_use]
    pub fn ready(&self) -> bool {
        self.sample_buffer.len() >= self.buffer_size
    }

    /// Extract features from buffered samples
    ///
    /// Returns None if not enough samples.
    pub fn extract(&mut self, aligned: Option<&AlignedSample>) -> Option<FeatureVector> {
        if !self.ready() {
            return None;
        }

        // Extract band powers per channel
        let mut band_powers = [BandPowers::default(); 8];
        for ch in 0..8 {
            let channel_data: Vec<f64> = self.sample_buffer.iter()
                .map(|s| s[ch])
                .collect();

            let psd = self.analyzer.compute_psd(&channel_data);
            band_powers[ch] = self.analyzer.all_band_powers(&psd);
        }

        // Compute frontal alpha asymmetry (Fp2 - Fp1) / (Fp2 + Fp1)
        let fp1_alpha = band_powers[EegChannel::Fp1.index()].alpha;
        let fp2_alpha = band_powers[EegChannel::Fp2.index()].alpha;
        let frontal_asymmetry = if fp1_alpha + fp2_alpha > 0.0 {
            (fp2_alpha - fp1_alpha) / (fp2_alpha + fp1_alpha)
        } else {
            0.0
        };

        // Compute channel correlations
        let correlations = self.compute_correlations();

        // Get fNIRS values if available
        let (hbo2, hbr) = aligned
            .map(|a| (
                f64::from(a.hemodynamic.delta_hbo2.to_f32()),
                f64::from(a.hemodynamic.delta_hbr.to_f32()),
            ))
            .unwrap_or((0.0, 0.0));

        Some(FeatureVector {
            band_powers,
            correlations,
            hbo2,
            hbr,
            frontal_asymmetry,
            timestamp_us: aligned.map(|a| a.timestamp_us).unwrap_or(0),
        })
    }

    fn compute_correlations(&self) -> [f64; 28] {
        let mut correlations = [0.0f64; 28];
        let mut idx = 0;

        // Compute correlation for each unique pair (8 choose 2 = 28)
        for i in 0..7 {
            for j in (i + 1)..8 {
                let ch_i: Vec<f64> = self.sample_buffer.iter().map(|s| s[i]).collect();
                let ch_j: Vec<f64> = self.sample_buffer.iter().map(|s| s[j]).collect();
                correlations[idx] = pearson_correlation(&ch_i, &ch_j);
                idx += 1;
            }
        }

        correlations
    }

    /// Clear the sample buffer
    pub fn clear(&mut self) {
        self.sample_buffer.clear();
    }
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_correlation(&x, &y_neg) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_feature_vector_size() {
        let fv = FeatureVector::default();
        assert_eq!(fv.to_array().len(), FeatureVector::feature_count());
    }
}

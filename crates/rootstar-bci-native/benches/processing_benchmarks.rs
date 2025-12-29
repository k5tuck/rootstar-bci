//! Benchmarks for signal processing modules

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use rootstar_bci_core::types::{EegSample, Fixed24_8};
use rootstar_bci_native::processing::{
    filters::{BiquadFilter, FilterCoefficients},
    fft::SpectralAnalyzer,
};

/// Generate synthetic EEG data (sinusoidal with noise)
fn generate_eeg_samples(n: usize, freq_hz: f64, sample_rate: f64) -> Vec<f64> {
    use std::f64::consts::PI;

    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let signal = (2.0 * PI * freq_hz * t).sin();
            let noise = (i as f64 * 0.123).sin() * 0.1; // Pseudo-noise
            (signal + noise) * 50.0 // Scale to ~50 µV
        })
        .collect()
}

fn bench_biquad_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_filter");

    // Create a lowpass filter at 30 Hz (typical EEG cutoff)
    let coeffs = FilterCoefficients::lowpass_2nd_order(250.0, 30.0);

    for size in [256, 512, 1024, 2048].iter() {
        let samples = generate_eeg_samples(*size, 10.0, 250.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                let mut filter = BiquadFilter::new(coeffs);
                b.iter(|| {
                    let mut output = Vec::with_capacity(samples.len());
                    for &sample in &samples {
                        output.push(filter.process(black_box(sample)));
                    }
                    filter.reset();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn bench_fft_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_analysis");

    for size in [256, 512, 1024].iter() {
        let samples = generate_eeg_samples(*size, 10.0, 250.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut analyzer = SpectralAnalyzer::new(size, 250.0);
                b.iter(|| {
                    let psd = analyzer.compute_psd(black_box(&samples));
                    black_box(psd)
                });
            },
        );
    }

    group.finish();
}

fn bench_band_power_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("band_power");

    let fft_size = 256;
    let samples = generate_eeg_samples(fft_size, 10.0, 250.0);
    let mut analyzer = SpectralAnalyzer::new(fft_size, 250.0);
    let psd = analyzer.compute_psd(&samples);

    group.bench_function("all_bands", |b| {
        b.iter(|| {
            let powers = analyzer.all_band_powers(black_box(&psd));
            black_box(powers)
        });
    });

    group.bench_function("relative_bands", |b| {
        b.iter(|| {
            let powers = analyzer.relative_band_powers(black_box(&psd));
            black_box(powers)
        });
    });

    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    use rootstar_bci_native::ml::features::FeatureExtractor;

    let mut group = c.benchmark_group("feature_extraction");

    let fft_size = 256;
    let mut extractor = FeatureExtractor::new(fft_size, 250.0);

    // Fill the buffer with samples
    for i in 0..fft_size {
        let sample = EegSample::new(
            i as u64 * 4000,
            i as u32,
        );
        extractor.push_sample(&sample);
    }

    group.bench_function("extract_features", |b| {
        b.iter(|| {
            let features = extractor.extract(None);
            black_box(features)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_biquad_filter,
    bench_fft_analysis,
    bench_band_power_extraction,
    bench_feature_extraction,
);

criterion_main!(benches);

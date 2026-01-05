/**
 * Rootstar BCI Web Visualization - Demo Application
 *
 * This module handles WASM loading and runs the demo visualization
 * without requiring any BCI hardware.
 */

// WASM module and demo runner
let wasmModule = null;
let bciApp = null;
let demoRunner = null;
let isRunning = false;
let animationId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fpsUpdateTime = 0;

// DOM elements
const loadingScreen = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const loadingProgress = document.getElementById('loading-progress');
const btnPlay = document.getElementById('btn-play');
const btnReset = document.getElementById('btn-reset');
const timeDisplay = document.getElementById('time-display');
const fpsDisplay = document.getElementById('fps');

// Canvas elements
const canvasEeg = document.getElementById('canvas-eeg');
const canvasTopomap = document.getElementById('canvas-topomap');
const canvasFnirs = document.getElementById('canvas-fnirs');
const canvasEmg = document.getElementById('canvas-emg');

// Stats elements
const statFp1 = document.getElementById('stat-fp1');
const statAlpha = document.getElementById('stat-alpha');
const statHbo2 = document.getElementById('stat-hbo2');
const statHbr = document.getElementById('stat-hbr');
const statScl = document.getElementById('stat-scl');
const statArousal = document.getElementById('stat-arousal');

// Band power elements
const bandDelta = document.getElementById('band-delta');
const bandTheta = document.getElementById('band-theta');
const bandAlpha = document.getElementById('band-alpha');
const bandBeta = document.getElementById('band-beta');
const bandGamma = document.getElementById('band-gamma');

// Control elements
const noiseSlider = document.getElementById('noise-level');
const noiseValue = document.getElementById('noise-value');
const timeWindowSlider = document.getElementById('time-window');
const timeValue = document.getElementById('time-value');
const amplitudeSlider = document.getElementById('amplitude-scale');
const amplitudeValue = document.getElementById('amplitude-value');
const scenarioButtons = document.querySelectorAll('.scenario-btn');

// Visualization state
let currentEegValues = [0, 0, 0, 0, 0, 0, 0, 0];
let currentFnirsHbo2 = [0, 0, 0, 0];
let currentFnirsHbr = [0, 0, 0, 0];
let currentEmgRms = [0, 0, 0, 0, 0, 0, 0, 0];
let currentEdaScl = [0, 0, 0, 0];

/**
 * Initialize the application
 */
async function init() {
    try {
        updateLoading('Initializing WebAssembly...', 10);

        // Try to load the WASM module for data generation
        // Always use JavaScript rendering for the canvas visualizations
        try {
            updateLoading('Loading WASM module...', 30);
            wasmModule = await import('./pkg/rootstar_bci_web.js');
            await wasmModule.default();
            updateLoading('Initializing demo runner...', 60);
            // Use WASM DemoRunner for data generation if available
            demoRunner = new wasmModule.DemoRunner();
            console.log('WASM DemoRunner loaded successfully');
        } catch (wasmError) {
            console.warn('WASM not available, using JavaScript demo runner:', wasmError);
            updateLoading('Using JavaScript demo mode...', 50);
            // Use JavaScript-based demo runner as fallback
            demoRunner = createJSDemoRunner();
        }

        // Always use JavaScript BCI app for canvas visualization
        // (WASM BciApp uses internal canvases that don't match our HTML)
        bciApp = createJSBciApp();

        updateLoading('Setting up visualizations...', 80);
        setupCanvases();
        setupEventListeners();

        updateLoading('Ready!', 100);

        // Hide loading screen after a short delay
        setTimeout(() => {
            loadingScreen.classList.add('hidden');
        }, 500);

        console.log('Rootstar BCI Web initialized successfully');

    } catch (error) {
        console.error('Failed to initialize:', error);
        loadingText.textContent = `Error: ${error.message}`;
        loadingText.style.color = '#cc0000';
    }
}

/**
 * Create a JavaScript-based demo runner (fallback when WASM isn't available)
 */
function createJSDemoRunner() {
    return {
        time_ms: 0,
        scenario: 'Relaxed',
        running: false,
        alpha_phase: 0,
        beta_phase: 0,
        theta_phase: 0,
        delta_phase: 0,
        noise_seed: 12345,

        scenarios: {
            Relaxed: { bands: [0.8, 0.9, 2.5, 0.5, 0.3], arousal: 0.2 },
            Focused: { bands: [0.6, 0.7, 0.8, 2.2, 1.5], arousal: 0.6 },
            Drowsy: { bands: [1.2, 2.0, 0.6, 0.4, 0.3], arousal: 0.1 },
            Thinking: { bands: [0.7, 0.8, 0.7, 1.5, 2.5], arousal: 0.5 },
            Meditation: { bands: [0.9, 1.8, 2.0, 0.6, 0.4], arousal: 0.15 },
            MotorImagery: { bands: [0.8, 1.0, 0.4, 1.8, 1.2], arousal: 0.4 },
            EmotionalResponse: { bands: [1.0, 1.2, 1.0, 1.3, 0.8], arousal: 0.85 }
        },

        start() { this.running = true; },
        stop() { this.running = false; },
        is_running() { return this.running; },

        set_scenario(scenario) {
            this.scenario = scenario;
        },

        current_time_ms() { return this.time_ms; },

        tick(delta_ms) {
            if (!this.running) return 0;
            this.time_ms += delta_ms;

            const dt = delta_ms / 1000;
            this.alpha_phase += 10 * Math.PI * 2 * dt;
            this.beta_phase += 20 * Math.PI * 2 * dt;
            this.theta_phase += 6 * Math.PI * 2 * dt;
            this.delta_phase += 2 * Math.PI * 2 * dt;

            // Keep phases bounded
            if (this.alpha_phase > Math.PI * 2) this.alpha_phase -= Math.PI * 2;
            if (this.beta_phase > Math.PI * 2) this.beta_phase -= Math.PI * 2;
            if (this.theta_phase > Math.PI * 2) this.theta_phase -= Math.PI * 2;
            if (this.delta_phase > Math.PI * 2) this.delta_phase -= Math.PI * 2;

            return Math.floor(delta_ms * 250 / 1000);
        },

        nextRandom() {
            this.noise_seed = (this.noise_seed * 1103515245 + 12345) >>> 0;
            return (this.noise_seed >>> 16) / 32768 - 1;
        },

        generate_eeg() {
            const baseline = [-5.2, 3.8, -12.4, 8.7, -2.1, 6.3, -8.9, 4.2];
            const mods = this.scenarios[this.scenario].bands;

            return baseline.map((b, i) => {
                const delta = 20 * mods[0] * Math.sin(this.delta_phase + i * 0.5);
                const theta = 15 * mods[1] * Math.sin(this.theta_phase + i * 0.4);
                const alpha = 25 * mods[2] * Math.sin(this.alpha_phase + i * 0.3);
                const beta = 8 * mods[3] * Math.sin(this.beta_phase + i * 0.2);
                const noise = this.nextRandom() * 5;
                return b + delta + theta + alpha + beta + noise;
            });
        },

        generate_fnirs() {
            const arousal = this.scenarios[this.scenario].arousal;
            const t = this.time_ms / 1000;

            const hbo2 = [0.045, 0.052, 0.038, 0.041].map((v, i) => {
                const wave = Math.sin(t * 0.5 + i * 0.8) * 0.01;
                return v * (1 + arousal * 0.3) + wave + this.nextRandom() * 0.003;
            });
            const hbr = [-0.012, -0.015, -0.008, -0.011].map((v, i) => {
                const wave = Math.sin(t * 0.5 + i * 0.8 + Math.PI) * 0.005;
                return v * (1 - arousal * 0.2) + wave + this.nextRandom() * 0.002;
            });
            return [...hbo2, ...hbr];
        },

        generate_emg() {
            const baseline = [2.5, 2.8, 1.2, 1.4, 8.5, 9.2, 0.8, 0.6];
            const arousal = this.scenarios[this.scenario].arousal;
            const t = this.time_ms / 1000;

            return baseline.map((v, i) => {
                const burst = Math.max(0, Math.sin(t * 2 + i) * arousal * 3);
                return v * (0.5 + arousal * 0.5) + burst + Math.abs(this.nextRandom()) * 2;
            });
        },

        generate_eda() {
            const arousal = this.scenarios[this.scenario].arousal;
            const t = this.time_ms / 1000;

            const scl = [5.2, 4.8, 5.5, 5.0].map((v, i) => {
                const drift = Math.sin(t * 0.1 + i) * 0.3;
                return v * (1 + arousal * 0.5) + drift + this.nextRandom() * 0.1;
            });
            const scr = [0.3, 0.1, 0.5, 0.2].map((v, i) => {
                // Occasional skin conductance responses
                const scrPeak = Math.max(0, Math.sin(t * 0.8 + i * 2) - 0.7) * 2;
                return v * (1 + arousal * 2) + scrPeak * arousal + Math.abs(this.nextRandom()) * arousal * 0.3;
            });
            return [...scl, ...scr];
        },

        generate_band_power() {
            const base = [0.25, 0.18, 0.35, 0.15, 0.07];
            const mods = this.scenarios[this.scenario].bands;
            return base.map((b, i) => b * mods[i]);
        },

        reset() {
            this.time_ms = 0;
            this.alpha_phase = 0;
            this.beta_phase = 0;
            this.theta_phase = 0;
            this.delta_phase = 0;
            this.noise_seed = 12345;
        }
    };
}

/**
 * Create a JavaScript-based BCI app (fallback visualization)
 */
function createJSBciApp() {
    return {
        eegBuffer: [],
        fnirsBuffer: [],
        emgBuffer: [],
        edaBuffer: [],
        maxSamples: 2500, // 10 seconds at 250Hz
        maxFnirsSamples: 100, // 10 seconds at 10Hz

        push_eeg_raw(channels) {
            this.eegBuffer.push([...channels]);
            if (this.eegBuffer.length > this.maxSamples) {
                this.eegBuffer.shift();
            }
            currentEegValues = [...channels];
        },

        push_fnirs_raw(hbo2, hbr) {
            this.fnirsBuffer.push({ hbo2: [...hbo2], hbr: [...hbr] });
            if (this.fnirsBuffer.length > this.maxFnirsSamples) {
                this.fnirsBuffer.shift();
            }
            currentFnirsHbo2 = [...hbo2];
            currentFnirsHbr = [...hbr];
        },

        push_emg_rms(channels) {
            this.emgBuffer.push([...channels]);
            if (this.emgBuffer.length > this.maxFnirsSamples) {
                this.emgBuffer.shift();
            }
            currentEmgRms = [...channels];
        },

        push_eda_raw(sites) {
            this.edaBuffer.push([...sites]);
            if (this.edaBuffer.length > this.maxFnirsSamples) {
                this.edaBuffer.shift();
            }
            currentEdaScl = [...sites];
        },

        render() {
            renderEegCanvas(this.eegBuffer);
            renderTopomapCanvas(currentEegValues);
            renderFnirsCanvas(this.fnirsBuffer, currentFnirsHbo2, currentFnirsHbr);
            renderEmgEdaCanvas(this.emgBuffer, this.edaBuffer, currentEmgRms, currentEdaScl);
        }
    };
}

/**
 * Render EEG timeseries to canvas
 */
function renderEegCanvas(buffer) {
    const ctx = canvasEeg.getContext('2d');
    const width = canvasEeg.width;
    const height = canvasEeg.height;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);

    const channelHeight = height / 8;
    const channelNames = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'];
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#a29bfe', '#fd79a8'];

    // Draw channels
    for (let ch = 0; ch < 8; ch++) {
        const yCenter = channelHeight * (ch + 0.5);

        // Draw zero line
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(40, yCenter);
        ctx.lineTo(width, yCenter);
        ctx.stroke();

        // Draw channel label
        ctx.fillStyle = colors[ch];
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(channelNames[ch], 5, yCenter + 4);

        if (buffer.length < 2) continue;

        // Draw waveform
        ctx.strokeStyle = colors[ch];
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const visibleSamples = Math.min(buffer.length, width - 45);
        const startIdx = buffer.length - visibleSamples;

        for (let i = 0; i < visibleSamples; i++) {
            const x = 45 + (i / visibleSamples) * (width - 50);
            const amplitude = parseFloat(amplitudeSlider.value) || 100;
            const y = yCenter - (buffer[startIdx + i][ch] / amplitude) * (channelHeight * 0.4);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    // Draw scale
    ctx.fillStyle = '#666';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    const amplitude = parseFloat(amplitudeSlider.value) || 100;
    ctx.fillText(`${amplitude}µV`, width - 5, 12);
}

/**
 * Render topographic map to canvas
 */
function renderTopomapCanvas(eegValues) {
    const ctx = canvasTopomap.getContext('2d');
    const w = canvasTopomap.width;
    const h = canvasTopomap.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.38;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Electrode positions (normalized to head radius)
    const electrodes = [
        { name: 'Fp1', x: -0.3, y: -0.8, idx: 0 },
        { name: 'Fp2', x: 0.3, y: -0.8, idx: 1 },
        { name: 'C3', x: -0.6, y: 0, idx: 2 },
        { name: 'C4', x: 0.6, y: 0, idx: 3 },
        { name: 'P3', x: -0.4, y: 0.5, idx: 4 },
        { name: 'P4', x: 0.4, y: 0.5, idx: 5 },
        { name: 'O1', x: -0.2, y: 0.85, idx: 6 },
        { name: 'O2', x: 0.2, y: 0.85, idx: 7 }
    ];

    // Create interpolated heatmap
    const resolution = 50;
    const imageData = ctx.createImageData(resolution, resolution);

    for (let py = 0; py < resolution; py++) {
        for (let px = 0; px < resolution; px++) {
            // Normalize to -1 to 1
            const nx = (px / resolution - 0.5) * 2.2;
            const ny = (py / resolution - 0.5) * 2.2;

            // Check if inside head
            const dist = Math.sqrt(nx * nx + ny * ny);
            if (dist > 1.05) {
                const idx = (py * resolution + px) * 4;
                imageData.data[idx] = 10;
                imageData.data[idx + 1] = 10;
                imageData.data[idx + 2] = 26;
                imageData.data[idx + 3] = 255;
                continue;
            }

            // Inverse distance weighted interpolation
            let weightSum = 0;
            let valueSum = 0;

            electrodes.forEach(e => {
                const dx = nx - e.x;
                const dy = ny - e.y;
                const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
                const weight = 1 / (d * d);
                weightSum += weight;
                valueSum += weight * (eegValues[e.idx] || 0);
            });

            const value = valueSum / weightSum;

            // Map to color (blue-white-red colormap)
            const normalized = Math.max(-1, Math.min(1, value / 50));
            let r_col, g_col, b_col;

            if (normalized < 0) {
                // Blue to white
                const t = normalized + 1;
                r_col = Math.floor(t * 255);
                g_col = Math.floor(t * 255);
                b_col = 255;
            } else {
                // White to red
                const t = 1 - normalized;
                r_col = 255;
                g_col = Math.floor(t * 255);
                b_col = Math.floor(t * 255);
            }

            const idx = (py * resolution + px) * 4;
            imageData.data[idx] = r_col;
            imageData.data[idx + 1] = g_col;
            imageData.data[idx + 2] = b_col;
            imageData.data[idx + 3] = dist < 1 ? 255 : 100;
        }
    }

    // Draw the interpolated data
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = resolution;
    tempCanvas.height = resolution;
    tempCanvas.getContext('2d').putImageData(imageData, 0, 0);

    // Scale up to head size
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(tempCanvas, cx - r * 1.1, cy - r * 1.1, r * 2.2, r * 2.2);
    ctx.restore();

    // Draw head outline
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Draw nose
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy - r);
    ctx.lineTo(cx, cy - r - 12);
    ctx.lineTo(cx + 8, cy - r);
    ctx.stroke();

    // Draw ears
    ctx.beginPath();
    ctx.ellipse(cx - r - 4, cy, 6, 12, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(cx + r + 4, cy, 6, 12, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw electrode positions
    electrodes.forEach(e => {
        const ex = cx + e.x * r;
        const ey = cy + e.y * r;

        ctx.fillStyle = '#000';
        ctx.beginPath();
        ctx.arc(ex, ey, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = '8px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(e.name, ex, ey + 14);
    });

    // Color scale
    const scaleX = w - 25;
    const scaleH = 60;
    const scaleY = (h - scaleH) / 2;

    for (let i = 0; i < scaleH; i++) {
        const t = 1 - i / scaleH;
        const normalized = t * 2 - 1;
        let r_col, g_col, b_col;
        if (normalized < 0) {
            r_col = Math.floor((normalized + 1) * 255);
            g_col = Math.floor((normalized + 1) * 255);
            b_col = 255;
        } else {
            r_col = 255;
            g_col = Math.floor((1 - normalized) * 255);
            b_col = Math.floor((1 - normalized) * 255);
        }
        ctx.fillStyle = `rgb(${r_col},${g_col},${b_col})`;
        ctx.fillRect(scaleX, scaleY + i, 15, 1);
    }

    ctx.fillStyle = '#888';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('+50µV', scaleX - 5, scaleY - 3);
    ctx.fillText('-50µV', scaleX - 5, scaleY + scaleH + 10);
}

/**
 * Render fNIRS heatmap to canvas
 */
function renderFnirsCanvas(buffer, hbo2, hbr) {
    const ctx = canvasFnirs.getContext('2d');
    const w = canvasFnirs.width;
    const h = canvasFnirs.height;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Draw optode grid (2x2 for 4 channels)
    const optodeSize = Math.min(w, h * 0.6) / 3;
    const startX = (w - optodeSize * 2.5) / 2;
    const startY = 20;

    // Optode positions
    const optodes = [
        { x: 0, y: 0, label: 'L-PFC' },
        { x: 1, y: 0, label: 'R-PFC' },
        { x: 0, y: 1, label: 'L-MC' },
        { x: 1, y: 1, label: 'R-MC' }
    ];

    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';

    optodes.forEach((opt, i) => {
        const x = startX + opt.x * optodeSize * 1.3;
        const y = startY + opt.y * optodeSize * 1.1;

        // HbO2 value (red-orange color)
        const hbo2Val = (hbo2[i] || 0) * 1000; // Convert to µM
        const hbo2Intensity = Math.min(1, Math.max(0, (hbo2Val + 20) / 100));

        // HbR value (blue color)
        const hbrVal = (hbr[i] || 0) * 1000;
        const hbrIntensity = Math.min(1, Math.max(0, (-hbrVal + 5) / 30));

        // Draw HbO2 circle (left half)
        const gradient1 = ctx.createRadialGradient(x + optodeSize * 0.25, y + optodeSize * 0.5, 0,
                                                    x + optodeSize * 0.25, y + optodeSize * 0.5, optodeSize * 0.4);
        gradient1.addColorStop(0, `rgba(255, ${Math.floor(100 + hbo2Intensity * 100)}, 50, 1)`);
        gradient1.addColorStop(1, `rgba(180, 50, 20, 0.3)`);

        ctx.fillStyle = gradient1;
        ctx.beginPath();
        ctx.arc(x + optodeSize * 0.3, y + optodeSize * 0.5, optodeSize * 0.35, 0, Math.PI * 2);
        ctx.fill();

        // Draw HbR circle (right half)
        const gradient2 = ctx.createRadialGradient(x + optodeSize * 0.7, y + optodeSize * 0.5, 0,
                                                    x + optodeSize * 0.7, y + optodeSize * 0.5, optodeSize * 0.4);
        gradient2.addColorStop(0, `rgba(50, ${Math.floor(100 + hbrIntensity * 100)}, 255, 1)`);
        gradient2.addColorStop(1, `rgba(20, 50, 180, 0.3)`);

        ctx.fillStyle = gradient2;
        ctx.beginPath();
        ctx.arc(x + optodeSize * 0.7, y + optodeSize * 0.5, optodeSize * 0.35, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = '#aaa';
        ctx.fillText(opt.label, x + optodeSize * 0.5, y + optodeSize + 15);

        // Values
        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#ff6b6b';
        ctx.fillText(`${hbo2Val.toFixed(1)}`, x + optodeSize * 0.3, y + optodeSize * 0.5 + 3);
        ctx.fillStyle = '#4ecdc4';
        ctx.fillText(`${hbrVal.toFixed(1)}`, x + optodeSize * 0.7, y + optodeSize * 0.5 + 3);
        ctx.font = '11px sans-serif';
    });

    // Legend
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ff6b6b';
    ctx.fillText('● HbO₂ (µM)', 10, h - 25);
    ctx.fillStyle = '#4ecdc4';
    ctx.fillText('● HbR (µM)', 10, h - 10);

    // Draw mini timeseries if we have buffer data
    if (buffer.length > 5) {
        const graphX = w * 0.55;
        const graphW = w * 0.4;
        const graphY = h * 0.55;
        const graphH = h * 0.35;

        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(graphX, graphY, graphW, graphH);

        // Draw HbO2 trend
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const visibleSamples = Math.min(buffer.length, 50);
        const startIdx = buffer.length - visibleSamples;
        for (let i = 0; i < visibleSamples; i++) {
            const x = graphX + (i / visibleSamples) * graphW;
            const val = buffer[startIdx + i].hbo2[0] * 1000;
            const y = graphY + graphH / 2 - (val / 100) * (graphH / 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw HbR trend
        ctx.strokeStyle = '#4ecdc4';
        ctx.beginPath();
        for (let i = 0; i < visibleSamples; i++) {
            const x = graphX + (i / visibleSamples) * graphW;
            const val = buffer[startIdx + i].hbr[0] * 1000;
            const y = graphY + graphH / 2 - (val / 50) * (graphH / 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.fillText('Trend (Ch1)', graphX, graphY - 5);
    }
}

/**
 * Render EMG and EDA to canvas
 */
function renderEmgEdaCanvas(emgBuffer, edaBuffer, emgRms, edaScl) {
    const ctx = canvasEmg.getContext('2d');
    const w = canvasEmg.width;
    const h = canvasEmg.height;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    const halfH = h / 2;

    // === EMG Section (top half) ===
    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('EMG - Facial Muscles', 10, 15);

    const emgLabels = ['Zyg-L', 'Zyg-R', 'Cor-L', 'Cor-R', 'Mas-L', 'Mas-R', 'Orb-U', 'Orb-D'];
    const barWidth = (w - 80) / 8;
    const maxEmg = 15;

    // Draw EMG bars
    for (let i = 0; i < 8; i++) {
        const x = 50 + i * barWidth;
        const barH = Math.min(1, (emgRms[i] || 0) / maxEmg) * (halfH - 50);

        // Bar gradient
        const gradient = ctx.createLinearGradient(x, halfH - 20, x, halfH - 20 - barH);
        gradient.addColorStop(0, '#45b7d1');
        gradient.addColorStop(1, '#96ceb4');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, halfH - 20 - barH, barWidth - 4, barH);

        // Value
        ctx.fillStyle = '#fff';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText((emgRms[i] || 0).toFixed(1), x + barWidth / 2 - 2, halfH - 25 - barH);

        // Label
        ctx.fillStyle = '#666';
        ctx.font = '8px sans-serif';
        ctx.save();
        ctx.translate(x + barWidth / 2 - 2, halfH - 8);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(emgLabels[i], 0, 0);
        ctx.restore();
    }

    // === EDA Section (bottom half) ===
    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('EDA - Skin Conductance', 10, halfH + 20);

    const edaLabels = ['Palm-L', 'Palm-R', 'Fing-L', 'Fing-R'];
    const edaBarWidth = (w - 80) / 4;
    const maxEda = 10;

    // Draw EDA bars
    for (let i = 0; i < 4; i++) {
        const x = 50 + i * edaBarWidth;
        const barH = Math.min(1, (edaScl[i] || 0) / maxEda) * (halfH - 60);

        // Bar gradient
        const gradient = ctx.createLinearGradient(x, h - 20, x, h - 20 - barH);
        gradient.addColorStop(0, '#ffeaa7');
        gradient.addColorStop(1, '#fdcb6e');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, h - 20 - barH, edaBarWidth - 8, barH);

        // Value
        ctx.fillStyle = '#fff';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText((edaScl[i] || 0).toFixed(1) + ' µS', x + edaBarWidth / 2 - 4, h - 28 - barH);

        // Label
        ctx.fillStyle = '#666';
        ctx.fillText(edaLabels[i], x + edaBarWidth / 2 - 4, h - 5);
    }

    // Draw mini timeseries for EDA
    if (edaBuffer.length > 5) {
        const graphX = w * 0.6;
        const graphW = w * 0.35;
        const graphY = halfH + 35;
        const graphH = halfH * 0.4;

        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(graphX, graphY, graphW, graphH);

        ctx.strokeStyle = '#ffeaa7';
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const visibleSamples = Math.min(edaBuffer.length, 50);
        const startIdx = edaBuffer.length - visibleSamples;
        for (let i = 0; i < visibleSamples; i++) {
            const x = graphX + (i / visibleSamples) * graphW;
            const val = edaBuffer[startIdx + i][0];
            const y = graphY + graphH - (val / maxEda) * graphH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('SCL Trend', graphX, graphY - 5);
    }
}

/**
 * Setup canvas dimensions
 */
function setupCanvases() {
    const canvases = [canvasEeg, canvasTopomap, canvasFnirs, canvasEmg];
    canvases.forEach(canvas => {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = Math.max(200, rect.width - 16);
        canvas.height = Math.max(150, rect.height - 16);
    });

    // Initial render with empty data
    renderEegCanvas([]);
    renderTopomapCanvas([0, 0, 0, 0, 0, 0, 0, 0]);
    renderFnirsCanvas([], [0, 0, 0, 0], [0, 0, 0, 0]);
    renderEmgEdaCanvas([], [], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0]);
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Play/pause button
    btnPlay.addEventListener('click', togglePlayback);

    // Reset button
    btnReset.addEventListener('click', resetDemo);

    // Scenario buttons
    scenarioButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            scenarioButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const scenario = btn.dataset.scenario;
            if (demoRunner.set_scenario) {
                demoRunner.set_scenario(scenario);
            } else {
                demoRunner.scenario = scenario;
            }
        });
    });

    // Sliders
    noiseSlider.addEventListener('input', (e) => {
        noiseValue.textContent = e.target.value;
    });

    timeWindowSlider.addEventListener('input', (e) => {
        timeValue.textContent = e.target.value;
    });

    amplitudeSlider.addEventListener('input', (e) => {
        amplitudeValue.textContent = e.target.value;
    });

    // Resize handler
    window.addEventListener('resize', () => {
        setupCanvases();
    });
}

/**
 * Toggle playback
 */
function togglePlayback() {
    if (isRunning) {
        stopDemo();
    } else {
        startDemo();
    }
}

/**
 * Start the demo
 */
function startDemo() {
    isRunning = true;
    btnPlay.textContent = '⏸ Pause';

    // Call start method on demo runner
    callDemoMethod('start');
    // Also set running flag directly for JS fallback
    if ('running' in demoRunner) {
        demoRunner.running = true;
    }

    lastFrameTime = performance.now();
    fpsUpdateTime = performance.now();
    animationId = requestAnimationFrame(renderLoop);
    console.log('Demo started');
}

/**
 * Stop the demo
 */
function stopDemo() {
    isRunning = false;
    btnPlay.textContent = '▶ Start';

    // Call stop method on demo runner
    callDemoMethod('stop');
    // Also set running flag directly for JS fallback
    if ('running' in demoRunner) {
        demoRunner.running = false;
    }

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    console.log('Demo stopped');
}

/**
 * Reset the demo
 */
function resetDemo() {
    stopDemo();

    if (demoRunner.reset) {
        demoRunner.reset();
    }

    if (bciApp.eegBuffer) {
        bciApp.eegBuffer = [];
        bciApp.fnirsBuffer = [];
        bciApp.emgBuffer = [];
        bciApp.edaBuffer = [];
    }

    currentEegValues = [0, 0, 0, 0, 0, 0, 0, 0];
    currentFnirsHbo2 = [0, 0, 0, 0];
    currentFnirsHbr = [0, 0, 0, 0];
    currentEmgRms = [0, 0, 0, 0, 0, 0, 0, 0];
    currentEdaScl = [0, 0, 0, 0];

    timeDisplay.textContent = '00:00.000';
    updateBandPower([0.25, 0.18, 0.35, 0.15, 0.07]);
    setupCanvases();
}

/**
 * Wrapper to call demo runner methods (handles both WASM and JS versions)
 */
function callDemoMethod(methodName, ...args) {
    // WASM uses snake_case, JS uses the same
    const method = demoRunner[methodName];
    if (typeof method === 'function') {
        return method.call(demoRunner, ...args);
    }
    // Some methods might return arrays that need conversion from WASM
    return null;
}

/**
 * Convert WASM array-like to JS array if needed
 */
function toArray(wasmResult) {
    if (!wasmResult) return [];
    if (Array.isArray(wasmResult)) return wasmResult;
    // WASM might return typed arrays or array-like objects
    if (wasmResult.length !== undefined) {
        return Array.from(wasmResult);
    }
    return [];
}

/**
 * Main render loop
 */
function renderLoop(timestamp) {
    if (!isRunning) return;

    const deltaMs = timestamp - lastFrameTime;
    lastFrameTime = timestamp;

    // Update FPS counter
    frameCount++;
    if (timestamp - fpsUpdateTime >= 1000) {
        fpsDisplay.textContent = `${frameCount} FPS`;
        frameCount = 0;
        fpsUpdateTime = timestamp;
    }

    // Tick the demo runner
    const samples = callDemoMethod('tick', deltaMs) || 0;

    // Generate and push data
    for (let i = 0; i < Math.max(1, samples); i++) {
        const eeg = toArray(callDemoMethod('generate_eeg'));
        if (eeg.length > 0) {
            bciApp.push_eeg_raw(eeg);

            // Update stats occasionally
            if (i === samples - 1 || samples === 0) {
                statFp1.textContent = `${eeg[0].toFixed(1)} µV`;
            }
        }
    }

    // Generate fNIRS data (lower rate)
    const fnirs = toArray(callDemoMethod('generate_fnirs'));
    if (fnirs.length >= 8) {
        bciApp.push_fnirs_raw(fnirs.slice(0, 4), fnirs.slice(4, 8));
        statHbo2.textContent = `${(fnirs[0] * 1000).toFixed(1)} µM`;
        statHbr.textContent = `${(fnirs[4] * 1000).toFixed(1)} µM`;
    }

    // Generate EMG and EDA
    const emg = toArray(callDemoMethod('generate_emg'));
    const eda = toArray(callDemoMethod('generate_eda'));
    if (emg.length > 0) {
        bciApp.push_emg_rms(emg);
    }
    if (eda.length >= 4) {
        bciApp.push_eda_raw(eda.slice(0, 4));
        statScl.textContent = `${eda[0].toFixed(1)} µS`;
    }

    // Get arousal from JS fallback scenarios if available
    if (demoRunner.scenarios && demoRunner.scenario) {
        statArousal.textContent = `${(demoRunner.scenarios[demoRunner.scenario].arousal * 100).toFixed(0)}%`;
    } else {
        statArousal.textContent = '--';
    }

    // Update band power display
    const bands = toArray(callDemoMethod('generate_band_power'));
    if (bands.length >= 5) {
        updateBandPower(bands);
        statAlpha.textContent = `${(bands[2] * 100).toFixed(0)}%`;
    }

    // Render all visualizations
    bciApp.render();

    // Update time display
    const timeMs = callDemoMethod('current_time_ms') || demoRunner.time_ms || 0;
    const mins = Math.floor(timeMs / 60000);
    const secs = Math.floor((timeMs % 60000) / 1000);
    const ms = Math.floor(timeMs % 1000);
    timeDisplay.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;

    animationId = requestAnimationFrame(renderLoop);
}

/**
 * Update band power display
 */
function updateBandPower(bands) {
    const normalize = bands.reduce((a, b) => a + b, 0);
    const normalized = bands.map(b => (b / normalize) * 100);

    bandDelta.style.height = `${normalized[0]}%`;
    bandTheta.style.height = `${normalized[1]}%`;
    bandAlpha.style.height = `${normalized[2]}%`;
    bandBeta.style.height = `${normalized[3]}%`;
    bandGamma.style.height = `${normalized[4]}%`;
}

/**
 * Update loading progress
 */
function updateLoading(text, progress) {
    loadingText.textContent = text;
    loadingProgress.style.width = `${progress}%`;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

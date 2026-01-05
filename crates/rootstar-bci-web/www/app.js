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

/**
 * Initialize the application
 */
async function init() {
    try {
        updateLoading('Initializing WebAssembly...', 10);

        // Try to load the WASM module
        // In production, this would load the actual compiled WASM
        // For demo purposes, we'll use a JavaScript fallback
        try {
            updateLoading('Loading WASM module...', 30);
            wasmModule = await import('./pkg/rootstar_bci_web.js');
            await wasmModule.default();
            updateLoading('Initializing BCI application...', 60);
            bciApp = new wasmModule.BciApp();
            demoRunner = new wasmModule.DemoRunner();
        } catch (wasmError) {
            console.warn('WASM not available, using JavaScript demo mode:', wasmError);
            updateLoading('Using JavaScript demo mode...', 50);
            // Use JavaScript-based demo (fallback for when WASM isn't built)
            demoRunner = createJSDemoRunner();
            bciApp = createJSBciApp();
        }

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
                const alpha = 15 * mods[2] * Math.sin(this.alpha_phase + i * 0.3);
                const beta = 5 * mods[3] * Math.sin(this.beta_phase + i * 0.2);
                const theta = 10 * mods[1] * Math.sin(this.theta_phase + i * 0.4);
                const noise = this.nextRandom() * 3;
                return b + alpha + beta + theta + noise;
            });
        },

        generate_fnirs() {
            const arousal = this.scenarios[this.scenario].arousal;
            const hbo2 = [0.045, 0.052, 0.038, 0.041].map(v =>
                v * (1 + arousal * 0.3) + this.nextRandom() * 0.005
            );
            const hbr = [-0.012, -0.015, -0.008, -0.011].map(v =>
                v * (1 - arousal * 0.2) + this.nextRandom() * 0.002
            );
            return [...hbo2, ...hbr];
        },

        generate_emg() {
            const baseline = [2.5, 2.8, 1.2, 1.4, 8.5, 9.2, 0.8, 0.6];
            return baseline.map(v => v + this.nextRandom() * 0.5);
        },

        generate_eda() {
            const arousal = this.scenarios[this.scenario].arousal;
            const scl = [5.2, 4.8, 5.5, 5.0].map(v =>
                v * (1 + arousal * 0.5) + this.nextRandom() * 0.1
            );
            const scr = [0.3, 0.1, 0.5, 0.2].map(v =>
                v * (1 + arousal * 2) + Math.abs(this.nextRandom()) * arousal
            );
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
        maxSamples: 2500, // 10 seconds at 250Hz

        push_eeg_raw(channels) {
            this.eegBuffer.push([...channels]);
            if (this.eegBuffer.length > this.maxSamples) {
                this.eegBuffer.shift();
            }
        },

        push_fnirs_raw(hbo2, hbr) {},
        push_emg_rms(channels) {},
        push_eda_raw(sites) {},

        set_vr_eeg_bands(d, t, a, b, g) {},
        set_vr_fnirs(hbo, hbr) {},
        set_vr_emg(rms, v, a) {},
        set_vr_eda(scl, a) {},

        render() {
            renderEegCanvas(this.eegBuffer);
        },

        get_timeseries_canvas() { return canvasEeg; },
        get_topomap_canvas() { return canvasTopomap; },
        get_fnirs_canvas() { return canvasFnirs; },
        get_emg_canvas() { return canvasEmg; }
    };
}

/**
 * Render EEG data to canvas (JS fallback)
 */
function renderEegCanvas(buffer) {
    const ctx = canvasEeg.getContext('2d');
    const width = canvasEeg.width;
    const height = canvasEeg.height;

    // Clear
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);

    if (buffer.length < 2) return;

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
        ctx.moveTo(50, yCenter);
        ctx.lineTo(width, yCenter);
        ctx.stroke();

        // Draw channel label
        ctx.fillStyle = '#888888';
        ctx.font = '10px sans-serif';
        ctx.fillText(channelNames[ch], 5, yCenter + 4);

        // Draw waveform
        ctx.strokeStyle = colors[ch];
        ctx.lineWidth = 1;
        ctx.beginPath();

        const startIdx = Math.max(0, buffer.length - Math.floor(width - 50));
        for (let i = startIdx; i < buffer.length; i++) {
            const x = 50 + (i - startIdx);
            const y = yCenter - (buffer[i][ch] / 100) * (channelHeight * 0.4);

            if (i === startIdx) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
}

/**
 * Setup canvas dimensions
 */
function setupCanvases() {
    const canvases = [canvasEeg, canvasTopomap, canvasFnirs, canvasEmg];
    canvases.forEach(canvas => {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width - 16;
        canvas.height = rect.height - 16;
    });

    // Initial render
    renderPlaceholders();
}

/**
 * Render placeholder content for canvases
 */
function renderPlaceholders() {
    // Topomap placeholder
    const ctx = canvasTopomap.getContext('2d');
    const w = canvasTopomap.width;
    const h = canvasTopomap.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.4;

    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Draw head outline
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Draw nose
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy - r);
    ctx.lineTo(cx, cy - r - 15);
    ctx.lineTo(cx + 10, cy - r);
    ctx.stroke();

    // Draw ears
    ctx.beginPath();
    ctx.ellipse(cx - r - 5, cy, 8, 15, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(cx + r + 5, cy, 8, 15, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw electrode positions
    const electrodes = [
        { name: 'Fp1', x: cx - 25, y: cy - r * 0.7 },
        { name: 'Fp2', x: cx + 25, y: cy - r * 0.7 },
        { name: 'C3', x: cx - r * 0.5, y: cy },
        { name: 'C4', x: cx + r * 0.5, y: cy },
        { name: 'P3', x: cx - r * 0.35, y: cy + r * 0.4 },
        { name: 'P4', x: cx + r * 0.35, y: cy + r * 0.4 },
        { name: 'O1', x: cx - 20, y: cy + r * 0.8 },
        { name: 'O2', x: cx + 20, y: cy + r * 0.8 }
    ];

    electrodes.forEach(e => {
        ctx.fillStyle = '#007ACC';
        ctx.beginPath();
        ctx.arc(e.x, e.y, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#888';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(e.name, e.x, e.y + 20);
    });

    // fNIRS placeholder
    const fnCtx = canvasFnirs.getContext('2d');
    fnCtx.fillStyle = '#0a0a1a';
    fnCtx.fillRect(0, 0, canvasFnirs.width, canvasFnirs.height);
    fnCtx.fillStyle = '#333';
    fnCtx.font = '14px sans-serif';
    fnCtx.textAlign = 'center';
    fnCtx.fillText('fNIRS Visualization', canvasFnirs.width / 2, canvasFnirs.height / 2);

    // EMG placeholder
    const emgCtx = canvasEmg.getContext('2d');
    emgCtx.fillStyle = '#0a0a1a';
    emgCtx.fillRect(0, 0, canvasEmg.width, canvasEmg.height);
    emgCtx.fillStyle = '#333';
    emgCtx.font = '14px sans-serif';
    emgCtx.textAlign = 'center';
    emgCtx.fillText('EMG + EDA Visualization', canvasEmg.width / 2, canvasEmg.height / 2);
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

    if (demoRunner.start) {
        demoRunner.start();
    } else {
        demoRunner.running = true;
    }

    lastFrameTime = performance.now();
    animationId = requestAnimationFrame(renderLoop);
}

/**
 * Stop the demo
 */
function stopDemo() {
    isRunning = false;
    btnPlay.textContent = '▶ Start';

    if (demoRunner.stop) {
        demoRunner.stop();
    } else {
        demoRunner.running = false;
    }

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
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
    }

    timeDisplay.textContent = '00:00.000';
    updateBandPower([0.25, 0.18, 0.35, 0.15, 0.07]);
    renderPlaceholders();
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
    const samples = demoRunner.tick ? demoRunner.tick(deltaMs) : demoRunner.tick(deltaMs);

    // Generate and push data
    for (let i = 0; i < samples; i++) {
        const eeg = demoRunner.generate_eeg();
        bciApp.push_eeg_raw(eeg);

        // Update stats occasionally
        if (i === samples - 1) {
            statFp1.textContent = `${eeg[0].toFixed(1)} µV`;
        }
    }

    // Generate fNIRS data (lower rate)
    const fnirs = demoRunner.generate_fnirs();
    bciApp.push_fnirs_raw(fnirs.slice(0, 4), fnirs.slice(4, 8));
    statHbo2.textContent = `${(fnirs[0] * 1000).toFixed(1)} µM`;
    statHbr.textContent = `${(fnirs[4] * 1000).toFixed(1)} µM`;

    // Generate EMG and EDA
    const emg = demoRunner.generate_emg();
    const eda = demoRunner.generate_eda();
    bciApp.push_emg_rms(emg);
    bciApp.push_eda_raw(eda.slice(0, 4));
    statScl.textContent = `${eda[0].toFixed(1)} µS`;
    statArousal.textContent = demoRunner.scenarios ?
        `${(demoRunner.scenarios[demoRunner.scenario].arousal * 100).toFixed(0)}%` : '--';

    // Update band power display
    const bands = demoRunner.generate_band_power();
    updateBandPower(bands);
    statAlpha.textContent = `${(bands[2] * 100).toFixed(0)}%`;

    // Render
    bciApp.render();

    // Update time display
    const timeMs = demoRunner.current_time_ms ? demoRunner.current_time_ms() : demoRunner.time_ms;
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

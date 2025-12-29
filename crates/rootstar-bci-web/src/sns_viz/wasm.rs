//! WASM Web Deployment Utilities (SNS-20)
//!
//! This module provides utilities for deploying the SNS visualization
//! to the web via WebAssembly, including:
//! - WebGPU initialization with WebGL2 fallback
//! - Canvas and context management
//! - Animation loop integration
//! - Error handling for web environments

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use super::pipeline::BciVizPipeline;
use super::SnsVizApp;

/// GPU backend type
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    /// WebGPU (preferred)
    WebGpu,
    /// WebGL2 (fallback)
    WebGl2,
    /// No GPU available (CPU only)
    CpuOnly,
}

/// Web deployment configuration
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WebDeployConfig {
    /// Canvas element ID
    canvas_id: String,
    /// Target frame rate
    target_fps: u32,
    /// Enable WebGPU (with fallback to WebGL2)
    prefer_webgpu: bool,
    /// Enable high-DPI rendering
    high_dpi: bool,
    /// Background color (CSS format)
    background_color: String,
    /// Enable performance monitoring
    enable_perf_monitor: bool,
    /// Canvas width
    width: u32,
    /// Canvas height
    height: u32,
}

#[wasm_bindgen]
impl WebDeployConfig {
    /// Create new configuration with defaults
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            canvas_id: "sns-canvas".to_string(),
            target_fps: 60,
            prefer_webgpu: true,
            high_dpi: true,
            background_color: "#1a1a2e".to_string(),
            enable_perf_monitor: false,
            width: 800,
            height: 600,
        }
    }

    /// Set canvas element ID
    #[wasm_bindgen]
    pub fn set_canvas_id(&mut self, id: &str) {
        self.canvas_id = id.to_string();
    }

    /// Set target frame rate
    #[wasm_bindgen]
    pub fn set_target_fps(&mut self, fps: u32) {
        self.target_fps = fps.clamp(1, 120);
    }

    /// Enable/disable WebGPU preference
    #[wasm_bindgen]
    pub fn set_prefer_webgpu(&mut self, prefer: bool) {
        self.prefer_webgpu = prefer;
    }

    /// Enable/disable high-DPI rendering
    #[wasm_bindgen]
    pub fn set_high_dpi(&mut self, enabled: bool) {
        self.high_dpi = enabled;
    }

    /// Set background color
    #[wasm_bindgen]
    pub fn set_background_color(&mut self, color: &str) {
        self.background_color = color.to_string();
    }

    /// Enable/disable performance monitor
    #[wasm_bindgen]
    pub fn set_perf_monitor(&mut self, enabled: bool) {
        self.enable_perf_monitor = enabled;
    }

    /// Set canvas size
    #[wasm_bindgen]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

impl Default for WebDeployConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Web renderer abstraction for GPU backends
pub struct WebRenderer {
    /// Backend type
    backend: GpuBackend,
    /// Canvas element
    canvas: HtmlCanvasElement,
    /// Canvas width
    width: u32,
    /// Canvas height
    height: u32,
    /// Device pixel ratio
    pixel_ratio: f64,
    /// Frame count
    frame_count: u64,
    /// Last frame timestamp
    last_frame_ms: f64,
    /// FPS accumulator
    fps_accumulator: f64,
    /// Frame time accumulator
    frame_time_acc: f64,
}

impl WebRenderer {
    /// Create new web renderer
    pub fn new(canvas: HtmlCanvasElement, backend: GpuBackend) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        Self {
            backend,
            canvas,
            width,
            height,
            pixel_ratio: 1.0,
            frame_count: 0,
            last_frame_ms: 0.0,
            fps_accumulator: 0.0,
            frame_time_acc: 0.0,
        }
    }

    /// Get current backend
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Resize canvas
    pub fn resize(&mut self, width: u32, height: u32, pixel_ratio: f64) {
        self.width = (width as f64 * pixel_ratio) as u32;
        self.height = (height as f64 * pixel_ratio) as u32;
        self.pixel_ratio = pixel_ratio;

        self.canvas.set_width(self.width);
        self.canvas.set_height(self.height);
    }

    /// Begin frame
    pub fn begin_frame(&mut self, timestamp_ms: f64) {
        let delta = timestamp_ms - self.last_frame_ms;
        self.last_frame_ms = timestamp_ms;
        self.frame_count += 1;

        self.frame_time_acc += delta;
        if self.frame_time_acc >= 1000.0 {
            self.fps_accumulator = self.frame_count as f64 * 1000.0 / self.frame_time_acc;
            self.frame_time_acc = 0.0;
            self.frame_count = 0;
        }
    }

    /// Get current FPS
    pub fn fps(&self) -> f64 {
        self.fps_accumulator
    }
}

/// Main web application entry point
#[wasm_bindgen]
pub struct SnsWebApp {
    /// Configuration
    config: WebDeployConfig,
    /// Renderer
    renderer: Option<WebRenderer>,
    /// Visualization app
    viz_app: SnsVizApp,
    /// BCI pipeline
    pipeline: BciVizPipeline,
    /// Animation frame ID
    animation_id: Option<i32>,
    /// Running state
    running: bool,
    /// Error message
    last_error: Option<String>,
    /// Last mouse X position
    last_mouse_x: f32,
    /// Last mouse Y position
    last_mouse_y: f32,
    /// Current mouse buttons state
    mouse_buttons: u32,
}

#[wasm_bindgen]
impl SnsWebApp {
    /// Create new web application
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: WebDeployConfig::new(),
            renderer: None,
            viz_app: SnsVizApp::new(800, 600),
            pipeline: BciVizPipeline::new(),
            animation_id: None,
            running: false,
            last_error: None,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            mouse_buttons: 0,
        }
    }

    /// Initialize with configuration
    #[wasm_bindgen]
    pub fn init(&mut self, config: WebDeployConfig) -> Result<GpuBackend, JsValue> {
        // Get window and document
        let window = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("No document object"))?;

        // Get canvas element
        let canvas = document
            .get_element_by_id(&config.canvas_id)
            .ok_or_else(|| JsValue::from_str("Canvas element not found"))?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("Element is not a canvas"))?;

        // Update viz app with dimensions
        self.viz_app = SnsVizApp::new(config.width, config.height);

        // Determine GPU backend (simplified - assume WebGL2 for now)
        let backend = if config.prefer_webgpu {
            // Try WebGPU first via JS reflection
            let gpu = js_sys::Reflect::get(&window, &JsValue::from_str("gpu"))
                .unwrap_or(JsValue::UNDEFINED);
            if !gpu.is_undefined() && !gpu.is_null() {
                GpuBackend::WebGpu
            } else {
                GpuBackend::WebGl2
            }
        } else {
            GpuBackend::WebGl2
        };

        // Handle high-DPI
        let pixel_ratio = if config.high_dpi {
            window.device_pixel_ratio()
        } else {
            1.0
        };

        // Create renderer
        let mut renderer = WebRenderer::new(canvas, backend);
        renderer.resize(config.width, config.height, pixel_ratio);

        self.config = config;
        self.renderer = Some(renderer);

        Ok(backend)
    }

    /// Start animation loop
    #[wasm_bindgen]
    pub fn start(&mut self) -> Result<(), JsValue> {
        if self.renderer.is_none() {
            return Err(JsValue::from_str("Not initialized"));
        }

        self.running = true;
        Ok(())
    }

    /// Stop animation loop
    #[wasm_bindgen]
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Check if running
    #[wasm_bindgen]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Update frame (called from animation loop)
    #[wasm_bindgen]
    pub fn update(&mut self, timestamp_ms: f64) -> bool {
        if !self.running {
            return false;
        }

        if let Some(renderer) = &mut self.renderer {
            renderer.begin_frame(timestamp_ms);

            // Update pipeline
            let timestamp_us = (timestamp_ms * 1000.0) as u64;
            self.pipeline.update(timestamp_us);

            true
        } else {
            false
        }
    }

    /// Get current FPS
    #[wasm_bindgen]
    pub fn get_fps(&self) -> f64 {
        self.renderer.as_ref().map(|r| r.fps()).unwrap_or(0.0)
    }

    /// Get GPU backend name
    #[wasm_bindgen]
    pub fn get_backend_name(&self) -> String {
        match self.renderer.as_ref().map(|r| r.backend()) {
            Some(GpuBackend::WebGpu) => "WebGPU".to_string(),
            Some(GpuBackend::WebGl2) => "WebGL2".to_string(),
            Some(GpuBackend::CpuOnly) => "CPU".to_string(),
            None => "Not initialized".to_string(),
        }
    }

    /// Connect to BCI data stream
    #[wasm_bindgen]
    pub fn connect_bci(&mut self, url: &str) -> Result<(), JsValue> {
        self.pipeline = BciVizPipeline::with_url(url);
        self.pipeline.connect()
    }

    /// Disconnect from BCI stream
    #[wasm_bindgen]
    pub fn disconnect_bci(&mut self) {
        self.pipeline.disconnect();
    }

    /// Check BCI connection status
    #[wasm_bindgen]
    pub fn is_bci_connected(&self) -> bool {
        self.pipeline.is_connected()
    }

    /// Load tactile mesh
    #[wasm_bindgen]
    pub fn load_tactile_mesh(&mut self, region: &str) -> bool {
        self.viz_app.load_tactile_mesh(region).is_ok()
    }

    /// Load cochlea mesh
    #[wasm_bindgen]
    pub fn load_cochlea_mesh(&mut self, ear: &str) -> bool {
        self.viz_app.load_cochlea_mesh(ear, false).is_ok()
    }

    /// Load tongue mesh
    #[wasm_bindgen]
    pub fn load_tongue_mesh(&mut self) -> bool {
        self.viz_app.load_tongue_mesh().is_ok()
    }

    /// Set colormap
    #[wasm_bindgen]
    pub fn set_colormap(&mut self, colormap: &str) {
        self.pipeline.set_colormap(colormap);
        self.viz_app.set_colormap(colormap);
    }

    /// Handle mouse move
    #[wasm_bindgen]
    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        let dx = x - self.last_mouse_x;
        let dy = y - self.last_mouse_y;
        self.last_mouse_x = x;
        self.last_mouse_y = y;
        self.viz_app.on_mouse_move(dx, dy, self.mouse_buttons);
    }

    /// Handle mouse down
    #[wasm_bindgen]
    pub fn on_mouse_down(&mut self, x: f32, y: f32, button: u32) {
        self.last_mouse_x = x;
        self.last_mouse_y = y;
        self.mouse_buttons |= 1 << button;
        self.viz_app.on_mouse_down(x, y, button);
    }

    /// Handle mouse up
    #[wasm_bindgen]
    pub fn on_mouse_up(&mut self, x: f32, y: f32) {
        self.mouse_buttons = 0;
        self.viz_app.on_mouse_up(x, y);
    }

    /// Handle mouse wheel
    #[wasm_bindgen]
    pub fn on_wheel(&mut self, delta: f32) {
        self.viz_app.on_wheel(delta);
    }

    /// Get last error message
    #[wasm_bindgen]
    pub fn get_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    /// Clear error state
    #[wasm_bindgen]
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }
}

impl Default for SnsWebApp {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate JavaScript module loader code
pub fn generate_js_loader() -> &'static str {
    r#"
// SNS Visualization Web Loader
// Generated by rootstar-bci-web

import init, { SnsWebApp, WebDeployConfig, GpuBackend } from './rootstar_bci_web.js';

class SnsVisualization {
    constructor() {
        this.app = null;
        this.animationId = null;
        this.initialized = false;
    }

    async initialize(canvasId = 'sns-canvas', options = {}) {
        // Initialize WASM module
        await init();

        // Create configuration
        const config = new WebDeployConfig();
        config.set_canvas_id(canvasId);

        if (options.fps) config.set_target_fps(options.fps);
        if (options.preferWebGpu !== undefined) config.set_prefer_webgpu(options.preferWebGpu);
        if (options.highDpi !== undefined) config.set_high_dpi(options.highDpi);
        if (options.backgroundColor) config.set_background_color(options.backgroundColor);
        if (options.perfMonitor) config.set_perf_monitor(options.perfMonitor);
        if (options.width && options.height) config.set_size(options.width, options.height);

        // Create and initialize app
        this.app = new SnsWebApp();
        const backend = this.app.init(config);

        console.log(`SNS Visualization initialized with ${this.app.get_backend_name()} backend`);

        this.initialized = true;
        return backend;
    }

    start() {
        if (!this.initialized || !this.app) {
            throw new Error('Not initialized');
        }

        this.app.start();
        this._animate();
    }

    stop() {
        if (this.app) {
            this.app.stop();
        }
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    _animate() {
        if (!this.app || !this.app.is_running()) return;

        const timestamp = performance.now();
        this.app.update(timestamp);

        this.animationId = requestAnimationFrame(() => this._animate());
    }

    async connectBci(url) {
        if (!this.app) throw new Error('Not initialized');
        return this.app.connect_bci(url);
    }

    disconnectBci() {
        if (this.app) this.app.disconnect_bci();
    }

    loadTactileMesh(region) {
        if (!this.app) return false;
        return this.app.load_tactile_mesh(region);
    }

    loadCochleaMesh(ear) {
        if (!this.app) return false;
        return this.app.load_cochlea_mesh(ear);
    }

    loadTongueMesh() {
        if (!this.app) return false;
        return this.app.load_tongue_mesh();
    }

    setColormap(colormap) {
        if (this.app) this.app.set_colormap(colormap);
    }

    getFps() {
        return this.app ? this.app.get_fps() : 0;
    }

    getBackendName() {
        return this.app ? this.app.get_backend_name() : 'Not initialized';
    }
}

// Export for module use
export { SnsVisualization, GpuBackend };

// Also attach to window for script use
if (typeof window !== 'undefined') {
    window.SnsVisualization = SnsVisualization;
}
"#
}

/// Generate HTML template for web deployment
pub fn generate_html_template() -> &'static str {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rootstar BCI - SNS Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background-color: #1a1a2e;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
            overflow: hidden;
        }
        .container { display: flex; flex-direction: column; height: 100vh; }
        header {
            padding: 10px 20px;
            background: linear-gradient(90deg, #16213e, #0f3460);
            display: flex; justify-content: space-between; align-items: center;
        }
        header h1 { font-size: 1.5em; color: #00d9ff; }
        .main-content { flex: 1; display: flex; }
        .sidebar { width: 250px; background: #0f3460; padding: 15px; }
        .canvas-container { flex: 1; position: relative; }
        #sns-canvas { width: 100%; height: 100%; display: block; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Rootstar BCI - SNS Visualization</h1>
        </header>
        <div class="main-content">
            <div class="sidebar">
                <p>SNS Visualization Controls</p>
            </div>
            <div class="canvas-container">
                <canvas id="sns-canvas" width="800" height="600"></canvas>
            </div>
        </div>
    </div>
    <script type="module">
        import { SnsVisualization } from './sns_loader.js';
        const viz = new SnsVisualization();
        async function init() {
            await viz.initialize('sns-canvas', { width: 800, height: 600 });
            viz.loadTactileMesh('fingertip');
            viz.start();
        }
        init();
    </script>
</body>
</html>
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = WebDeployConfig::new();
        assert_eq!(config.canvas_id, "sns-canvas");
        assert_eq!(config.target_fps, 60);
        assert!(config.prefer_webgpu);
    }

    #[test]
    fn test_js_loader_generation() {
        let js = generate_js_loader();
        assert!(js.contains("SnsVisualization"));
        assert!(js.contains("initialize"));
    }

    #[test]
    fn test_html_template_generation() {
        let html = generate_html_template();
        assert!(html.contains("sns-canvas"));
        assert!(html.contains("Rootstar BCI"));
    }
}

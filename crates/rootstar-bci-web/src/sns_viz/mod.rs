//! SNS 3D Visualization Module
//!
//! Provides interactive 3D visualization of sensory receptor populations
//! with real-time activation heatmaps. Supports:
//!
//! - Tactile: Skin surface with mechanoreceptors
//! - Auditory: Cochlea with hair cells
//! - Gustatory: Tongue with papillae and taste buds
//! - Visual: Retina with photoreceptors and V1 cortex with electrode mapping
//! - Olfactory: Olfactory epithelium and bulb with glomeruli
//!
//! Note: Full WebGPU rendering is implemented in the renderer submodule.
//! For WASM deployment, see SNS-20.

pub mod heatmap;
pub mod interaction;
pub mod meshes;
pub mod pipeline;
pub mod wasm;

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use rootstar_bci_core::sns::types::{BodyRegion, Ear, Finger, Hand, Side};

pub use heatmap::{ActivationHeatmap, Colormap};
pub use interaction::{InteractionEvent, InteractionHandler, ProbeResult};
pub use meshes::{Eye, MeshData, MeshId, OlfactoryView, ReceptorPosition, RetinaView, Vertex};
pub use pipeline::{BciVizPipeline, PlaybackController, SimulationGenerator};
pub use wasm::{GpuBackend, SnsWebApp, WebDeployConfig};

/// Configuration for SNS visualization
#[derive(Clone, Debug)]
pub struct SnsVizConfig {
    /// Canvas width in pixels
    pub width: u32,
    /// Canvas height in pixels
    pub height: u32,
    /// Background color (RGBA)
    pub background_color: [f32; 4],
    /// Enable antialiasing
    pub msaa_samples: u32,
    /// Colormap for activation overlay
    pub colormap: Colormap,
    /// Activation smoothing radius (in UV space)
    pub smoothing_radius: f32,
    /// Enable receptor markers
    pub show_receptors: bool,
    /// Receptor marker size
    pub receptor_size: f32,
}

impl Default for SnsVizConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            background_color: [0.1, 0.1, 0.12, 1.0],
            msaa_samples: 4,
            colormap: Colormap::Viridis,
            smoothing_radius: 0.02,
            show_receptors: true,
            receptor_size: 3.0,
        }
    }
}

/// 3D camera for scene viewing
#[derive(Clone, Debug)]
pub struct Camera3D {
    /// Eye position
    pub position: [f32; 3],
    /// Look-at target
    pub target: [f32; 3],
    /// Up vector
    pub up: [f32; 3],
    /// Field of view (radians)
    pub fov: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
}

impl Camera3D {
    /// Create a new camera with default settings
    #[must_use]
    pub fn new(aspect: f32) -> Self {
        Self {
            position: [0.0, -5.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            aspect,
        }
    }

    /// Compute view matrix
    #[must_use]
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let f = normalize(sub(self.target, self.position));
        let s = normalize(cross(f, self.up));
        let u = cross(s, f);

        [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [
                -dot(s, self.position),
                -dot(u, self.position),
                dot(f, self.position),
                1.0,
            ],
        ]
    }

    /// Compute projection matrix (perspective)
    #[must_use]
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (self.fov / 2.0).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) * nf, -1.0],
            [0.0, 0.0, 2.0 * self.far * self.near * nf, 0.0],
        ]
    }

    /// Orbit camera around target
    pub fn orbit(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        let rel = sub(self.position, self.target);
        let r = length(rel);
        let theta = rel[0].atan2(rel[1]) + delta_azimuth;
        let phi = (rel[2] / r).asin().clamp(-1.4, 1.4) + delta_elevation;

        self.position = [
            self.target[0] + r * phi.cos() * theta.sin(),
            self.target[1] + r * phi.cos() * theta.cos(),
            self.target[2] + r * phi.sin(),
        ];
    }

    /// Zoom camera (dolly)
    pub fn zoom(&mut self, factor: f32) {
        let rel = sub(self.position, self.target);
        let new_rel = scale(rel, factor.clamp(0.1, 10.0));
        self.position = add(self.target, new_rel);
    }

    /// Pan camera
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let f = normalize(sub(self.target, self.position));
        let s = normalize(cross(f, self.up));
        let u = cross(s, f);

        let offset = add(scale(s, delta_x), scale(u, delta_y));
        self.position = add(self.position, offset);
        self.target = add(self.target, offset);
    }
}

// Vector math helpers
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = length(v);
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Camera uniform buffer layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position (for lighting)
    pub camera_pos: [f32; 4],
}

impl CameraUniform {
    /// Create from camera
    #[must_use]
    pub fn from_camera(camera: &Camera3D) -> Self {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let view_proj = mat4_mul(proj, view);

        Self {
            view_proj,
            camera_pos: [
                camera.position[0],
                camera.position[1],
                camera.position[2],
                1.0,
            ],
        }
    }
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// SNS visualization scene (CPU-side data)
#[derive(Clone, Debug)]
pub struct SnsScene {
    /// Mesh data by ID
    meshes: HashMap<MeshId, MeshData>,
    /// Activation states per mesh
    activations: HashMap<MeshId, Vec<f32>>,
    /// Heatmap overlay settings
    heatmap: ActivationHeatmap,
    /// Interaction handler
    interaction: InteractionHandler,
    /// Camera
    camera: Camera3D,
    /// Configuration
    config: SnsVizConfig,
}

impl SnsScene {
    /// Create a new empty scene
    #[must_use]
    pub fn new(config: SnsVizConfig) -> Self {
        let aspect = config.width as f32 / config.height as f32;
        Self {
            meshes: HashMap::new(),
            activations: HashMap::new(),
            heatmap: ActivationHeatmap::new(config.colormap.clone(), config.smoothing_radius),
            interaction: InteractionHandler::new(),
            camera: Camera3D::new(aspect),
            config,
        }
    }

    /// Add a mesh to the scene
    pub fn add_mesh(&mut self, id: MeshId, mesh_data: MeshData) {
        let receptor_count = mesh_data.receptor_uvs.len();
        self.activations
            .insert(id.clone(), vec![0.0; receptor_count]);
        self.meshes.insert(id, mesh_data);
    }

    /// Update receptor activations for a mesh
    pub fn update_activations(&mut self, id: &MeshId, activations: &[f32]) {
        if let Some(stored) = self.activations.get_mut(id) {
            let len = stored.len().min(activations.len());
            stored[..len].copy_from_slice(&activations[..len]);
        }
    }

    /// Get current activations for a mesh
    #[must_use]
    pub fn get_activations(&self, id: &MeshId) -> Option<&[f32]> {
        self.activations.get(id).map(|v| v.as_slice())
    }

    /// Get mutable reference to camera
    pub fn camera_mut(&mut self) -> &mut Camera3D {
        &mut self.camera
    }

    /// Get camera reference
    #[must_use]
    pub fn camera(&self) -> &Camera3D {
        &self.camera
    }

    /// Get all mesh IDs
    pub fn mesh_ids(&self) -> impl Iterator<Item = &MeshId> {
        self.meshes.keys()
    }

    /// Get a mesh by ID
    #[must_use]
    pub fn get_mesh(&self, id: &MeshId) -> Option<&MeshData> {
        self.meshes.get(id)
    }

    /// Get heatmap settings
    #[must_use]
    pub fn heatmap(&self) -> &ActivationHeatmap {
        &self.heatmap
    }

    /// Get mutable heatmap settings
    pub fn heatmap_mut(&mut self) -> &mut ActivationHeatmap {
        &mut self.heatmap
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &SnsVizConfig {
        &self.config
    }

    /// Render activation heatmap to pixel buffer
    #[must_use]
    pub fn render_heatmap(&self, mesh_id: &MeshId, width: u32, height: u32) -> Option<Vec<u8>> {
        let mesh = self.meshes.get(mesh_id)?;
        let activations = self.activations.get(mesh_id)?;

        let uvs: Vec<[f32; 2]> = mesh.receptor_uvs.iter().map(|r| r.uv).collect();
        Some(self.heatmap.render_to_pixels(activations, &uvs, width, height))
    }
}

/// WASM-exported SNS visualization application
#[wasm_bindgen]
pub struct SnsVizApp {
    /// Scene data
    scene: SnsScene,
    /// Currently selected mesh
    selected_mesh: Option<MeshId>,
}

#[wasm_bindgen]
impl SnsVizApp {
    /// Create a new SNS visualization app
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        let config = SnsVizConfig {
            width,
            height,
            ..Default::default()
        };

        Self {
            scene: SnsScene::new(config),
            selected_mesh: None,
        }
    }

    /// Load tactile mesh for a body region
    pub fn load_tactile_mesh(&mut self, region_name: &str) -> Result<(), JsValue> {
        let region = match region_name.to_lowercase().as_str() {
            "fingertip" => BodyRegion::Fingertip(Finger::Index),
            "palm" => BodyRegion::Palm(Hand::Right),
            "forearm" => BodyRegion::Forearm(Side::Right),
            _ => return Err(JsValue::from_str("Unknown body region")),
        };

        let mesh_data = meshes::skin::generate_skin_mesh(region, 20);
        let id = MeshId::SkinPatch { region };
        self.scene.add_mesh(id.clone(), mesh_data);
        self.selected_mesh = Some(id);

        Ok(())
    }

    /// Load cochlea mesh
    pub fn load_cochlea_mesh(&mut self, ear: &str, unrolled: bool) -> Result<(), JsValue> {
        let ear_enum = match ear.to_lowercase().as_str() {
            "left" => Ear::Left,
            "right" => Ear::Right,
            _ => return Err(JsValue::from_str("Invalid ear: use 'left' or 'right'")),
        };

        let mesh_data = meshes::cochlea::generate_cochlea_mesh(ear_enum, unrolled);
        let id = MeshId::Cochlea { ear: ear_enum };
        self.scene.add_mesh(id.clone(), mesh_data);
        self.selected_mesh = Some(id);

        Ok(())
    }

    /// Load tongue mesh
    pub fn load_tongue_mesh(&mut self) -> Result<(), JsValue> {
        let mesh_data = meshes::tongue::generate_tongue_mesh();
        let id = MeshId::Tongue;
        self.scene.add_mesh(id.clone(), mesh_data);
        self.selected_mesh = Some(id);

        Ok(())
    }

    /// Load retina mesh (visual system)
    ///
    /// # Arguments
    /// * `eye` - "left" or "right"
    /// * `view` - "flat" (unrolled), "curved" (anatomical), or "cortex" (V1 mapping)
    /// * `detail` - Level of detail (1-10, default 5)
    pub fn load_retina_mesh(&mut self, eye: &str, view: &str, detail: u32) -> Result<(), JsValue> {
        let eye_enum = match eye.to_lowercase().as_str() {
            "left" => Eye::Left,
            "right" => Eye::Right,
            _ => return Err(JsValue::from_str("Invalid eye: use 'left' or 'right'")),
        };

        let view_enum = match view.to_lowercase().as_str() {
            "flat" => RetinaView::Flat,
            "curved" => RetinaView::Curved,
            "cortex" | "v1" => RetinaView::Cortex,
            _ => return Err(JsValue::from_str("Invalid view: use 'flat', 'curved', or 'cortex'")),
        };

        let mesh_data = meshes::retina::generate_retina_mesh(eye_enum, view_enum, detail);
        let id = MeshId::Retina { eye: eye_enum, view: view_enum };
        self.scene.add_mesh(id.clone(), mesh_data);
        self.selected_mesh = Some(id);

        Ok(())
    }

    /// Load olfactory mesh (smell system)
    ///
    /// # Arguments
    /// * `view` - "epithelium" (receptor neurons), "bulb" (glomeruli), or "combined"
    /// * `detail` - Level of detail (1-10, default 5)
    pub fn load_olfactory_mesh(&mut self, view: &str, detail: u32) -> Result<(), JsValue> {
        let view_enum = match view.to_lowercase().as_str() {
            "epithelium" | "epi" => OlfactoryView::Epithelium,
            "bulb" => OlfactoryView::Bulb,
            "combined" | "both" => OlfactoryView::Combined,
            _ => return Err(JsValue::from_str("Invalid view: use 'epithelium', 'bulb', or 'combined'")),
        };

        let mesh_data = meshes::olfactory::generate_olfactory_mesh(view_enum, detail);
        let id = MeshId::Olfactory { view: view_enum };
        self.scene.add_mesh(id.clone(), mesh_data);
        self.selected_mesh = Some(id);

        Ok(())
    }

    /// Update receptor activations (flat array, matches receptor order)
    pub fn update_activations(&mut self, activations: &[f32]) {
        if let Some(ref id) = self.selected_mesh {
            self.scene.update_activations(id, activations);
        }
    }

    /// Handle mouse move (for orbit)
    pub fn on_mouse_move(&mut self, dx: f32, dy: f32, buttons: u32) {
        if buttons & 1 != 0 {
            // Left button: orbit
            self.scene.camera_mut().orbit(dx * 0.01, dy * 0.01);
        } else if buttons & 2 != 0 {
            // Right button: pan
            self.scene.camera_mut().pan(dx * 0.01, dy * 0.01);
        }
    }

    /// Handle mouse wheel (for zoom)
    pub fn on_wheel(&mut self, delta: f32) {
        let factor = if delta > 0.0 { 1.1 } else { 0.9 };
        self.scene.camera_mut().zoom(factor);
    }

    /// Set colormap
    pub fn set_colormap(&mut self, name: &str) {
        let colormap = match name.to_lowercase().as_str() {
            "viridis" => Colormap::Viridis,
            "plasma" => Colormap::Plasma,
            "inferno" => Colormap::Inferno,
            "turbo" => Colormap::Turbo,
            "coolwarm" => Colormap::CoolWarm,
            _ => Colormap::Viridis,
        };
        self.scene.heatmap_mut().set_colormap(colormap);
    }

    /// Set activation range for colormap
    pub fn set_activation_range(&mut self, min_val: f32, max_val: f32) {
        self.scene.heatmap_mut().set_range(min_val, max_val);
    }

    /// Get receptor count for selected mesh
    #[must_use]
    pub fn get_receptor_count(&self) -> usize {
        if let Some(ref id) = self.selected_mesh {
            self.scene
                .get_activations(id)
                .map(|a| a.len())
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Render heatmap to pixel buffer (returns RGBA bytes)
    #[must_use]
    pub fn render_heatmap(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        if let Some(ref id) = self.selected_mesh {
            self.scene.render_heatmap(id, width, height)
        } else {
            None
        }
    }

    /// Get mesh vertex count
    #[must_use]
    pub fn get_vertex_count(&self) -> usize {
        if let Some(ref id) = self.selected_mesh {
            self.scene
                .get_mesh(id)
                .map(|m| m.vertices.len())
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Get mesh triangle count
    #[must_use]
    pub fn get_triangle_count(&self) -> usize {
        if let Some(ref id) = self.selected_mesh {
            self.scene
                .get_mesh(id)
                .map(|m| m.indices.len() / 3)
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Handle mouse down event
    pub fn on_mouse_down(&mut self, _x: f32, _y: f32, _button: u32) {
        // Store initial position for drag calculations
        // Button 0 = left (orbit), Button 2 = right (pan)
    }

    /// Handle mouse up event
    pub fn on_mouse_up(&mut self, _x: f32, _y: f32) {
        // End drag operation
    }
}

impl Default for SnsVizApp {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

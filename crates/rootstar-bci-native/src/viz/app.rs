//! Native SNS Visualization Application
//!
//! Provides a native desktop application for visualizing BCI data using
//! winit + wgpu. egui UI integration is optional and can be added later.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use rootstar_bci_core::sns::types::{BodyRegion, Ear, Finger};

use super::mesh::{generate_cochlea_mesh, generate_skin_mesh, generate_tongue_mesh, MeshId};
use super::renderer::{Camera3D, Colormap, SnsRenderer};

/// Configuration for the visualization application
#[derive(Clone, Debug)]
pub struct VizConfig {
    /// Window width
    pub width: u32,
    /// Window height
    pub height: u32,
    /// Window title
    pub title: String,
    /// Initial colormap
    pub colormap: Colormap,
    /// Background color
    pub background_color: [f32; 4],
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            title: "Rootstar BCI - SNS Visualization".to_string(),
            colormap: Colormap::Viridis,
            background_color: [0.1, 0.1, 0.12, 1.0],
        }
    }
}

/// State for the running application
struct AppState {
    window: Arc<Window>,
    renderer: SnsRenderer,
    camera: Camera3D,
    selected_mesh: Option<MeshId>,
    activations: Vec<f32>,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_buttons: u32,
    simulation_running: bool,
    simulation_time: f32,
}

/// SNS Visualization Application
pub struct SnsVizApp {
    config: VizConfig,
    state: Option<AppState>,
}

impl SnsVizApp {
    /// Create a new visualization app with configuration
    #[must_use]
    pub fn new(config: VizConfig) -> Self {
        Self { config, state: None }
    }

    fn create_window(&self, event_loop: &ActiveEventLoop) -> Arc<Window> {
        let window_attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));

        Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        )
    }

    fn load_mesh(&mut self, mesh_type: MeshType) {
        if let Some(ref mut state) = self.state {
            let (mesh_id, mesh_data) = match mesh_type {
                MeshType::Fingertip => {
                    let region = BodyRegion::Fingertip(Finger::Index);
                    (
                        MeshId::SkinPatch { region },
                        generate_skin_mesh(region, 20),
                    )
                }
                MeshType::Palm => {
                    let region = BodyRegion::Palm(rootstar_bci_core::sns::types::Hand::Right);
                    (
                        MeshId::SkinPatch { region },
                        generate_skin_mesh(region, 30),
                    )
                }
                MeshType::Forearm => {
                    let region = BodyRegion::Forearm(rootstar_bci_core::sns::types::Side::Right);
                    (
                        MeshId::SkinPatch { region },
                        generate_skin_mesh(region, 25),
                    )
                }
                MeshType::CochleaLeft => (
                    MeshId::Cochlea { ear: Ear::Left },
                    generate_cochlea_mesh(Ear::Left, false),
                ),
                MeshType::CochleaRight => (
                    MeshId::Cochlea { ear: Ear::Right },
                    generate_cochlea_mesh(Ear::Right, false),
                ),
                MeshType::CochleaUnrolled => (
                    MeshId::Cochlea { ear: Ear::Left },
                    generate_cochlea_mesh(Ear::Left, true),
                ),
                MeshType::Tongue => (MeshId::Tongue, generate_tongue_mesh()),
            };

            state.renderer.add_mesh(mesh_id.clone(), &mesh_data);
            state.selected_mesh = Some(mesh_id);
            state.activations = vec![0.0; mesh_data.receptor_count()];
        }
    }

    fn update_simulation(&mut self, dt: f32) {
        if let Some(ref mut state) = self.state {
            if state.simulation_running {
                state.simulation_time += dt;

                // Generate simulated activations
                let t = state.simulation_time;
                for (i, activation) in state.activations.iter_mut().enumerate() {
                    let phase = i as f32 * 0.1;
                    *activation = 50.0 + 40.0 * (t * 2.0 + phase).sin();
                }

                // Update renderer
                if let Some(ref mesh_id) = state.selected_mesh {
                    state
                        .renderer
                        .update_activations(mesh_id, &state.activations);
                }
            }
        }
    }
}

/// Mesh type selection
#[derive(Clone, Copy, Debug)]
pub enum MeshType {
    /// Fingertip skin mesh
    Fingertip,
    /// Palm skin mesh
    Palm,
    /// Forearm skin mesh
    Forearm,
    /// Left cochlea (spiral)
    CochleaLeft,
    /// Right cochlea (spiral)
    CochleaRight,
    /// Cochlea (unrolled view)
    CochleaUnrolled,
    /// Tongue surface
    Tongue,
}

impl ApplicationHandler for SnsVizApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = self.create_window(event_loop);
        let size = window.inner_size();

        // Create renderer
        let renderer = pollster::block_on(SnsRenderer::new(window.clone()))
            .expect("Failed to create renderer");

        let camera = Camera3D::new(size.width as f32 / size.height as f32);

        self.state = Some(AppState {
            window,
            renderer,
            camera,
            selected_mesh: None,
            activations: Vec::new(),
            last_mouse_pos: None,
            mouse_buttons: 0,
            simulation_running: false,
            simulation_time: 0.0,
        });

        // Load default mesh
        self.load_mesh(MeshType::Fingertip);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(ref mut state) = self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.camera.aspect = size.width as f32 / size.height as f32;
            }

            WindowEvent::RedrawRequested => {
                // Update simulation
                self.update_simulation(1.0 / 60.0);

                // Update camera
                if let Some(ref state) = self.state {
                    state.renderer.update_camera(&state.camera);
                }

                // Render scene
                if let Some(ref mut state) = self.state {
                    if let Err(e) = state.renderer.render() {
                        eprintln!("Render error: {e}");
                    }
                    state.window.request_redraw();
                }
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                let mask = match button {
                    MouseButton::Left => 1,
                    MouseButton::Right => 2,
                    MouseButton::Middle => 4,
                    _ => 0,
                };

                match button_state {
                    ElementState::Pressed => state.mouse_buttons |= mask,
                    ElementState::Released => state.mouse_buttons &= !mask,
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some((last_x, last_y)) = state.last_mouse_pos {
                    let dx = (position.x - last_x) as f32;
                    let dy = (position.y - last_y) as f32;

                    if state.mouse_buttons & 1 != 0 {
                        // Left button: orbit
                        state.camera.orbit(dx * 0.01, dy * 0.01);
                    } else if state.mouse_buttons & 2 != 0 {
                        // Right button: pan
                        state.camera.pan(dx * 0.01, dy * 0.01);
                    }
                }

                state.last_mouse_pos = Some((position.x, position.y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };

                let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                state.camera.zoom(factor);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                        match key_code {
                            winit::keyboard::KeyCode::Space => {
                                state.simulation_running = !state.simulation_running;
                                println!(
                                    "Simulation {}",
                                    if state.simulation_running {
                                        "started"
                                    } else {
                                        "paused"
                                    }
                                );
                            }
                            winit::keyboard::KeyCode::Digit1 => {
                                println!("Loading fingertip mesh...");
                            }
                            winit::keyboard::KeyCode::Digit2 => {
                                println!("Loading cochlea mesh...");
                            }
                            winit::keyboard::KeyCode::Digit3 => {
                                println!("Loading tongue mesh...");
                            }
                            winit::keyboard::KeyCode::KeyV => {
                                println!("Switching to Viridis colormap");
                                state.renderer.set_colormap(Colormap::Viridis);
                            }
                            winit::keyboard::KeyCode::KeyP => {
                                println!("Switching to Plasma colormap");
                                state.renderer.set_colormap(Colormap::Plasma);
                            }
                            winit::keyboard::KeyCode::KeyI => {
                                println!("Switching to Inferno colormap");
                                state.renderer.set_colormap(Colormap::Inferno);
                            }
                            winit::keyboard::KeyCode::Escape => {
                                event_loop.exit();
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

/// Run the visualization application
///
/// # Errors
///
/// Returns an error if the event loop fails to start.
pub fn run_app(config: VizConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting SNS Visualization...");
    println!("Controls:");
    println!("  Left drag: Orbit camera");
    println!("  Right drag: Pan camera");
    println!("  Scroll: Zoom");
    println!("  Space: Toggle simulation");
    println!("  1/2/3: Load fingertip/cochlea/tongue");
    println!("  V/P/I: Viridis/Plasma/Inferno colormap");
    println!("  Escape: Quit");

    let event_loop = EventLoop::new()?;
    let mut app = SnsVizApp::new(config);

    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viz_config_default() {
        let config = VizConfig::default();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }

    #[test]
    fn test_mesh_type() {
        let _fingertip = MeshType::Fingertip;
        let _cochlea = MeshType::CochleaLeft;
        let _tongue = MeshType::Tongue;
    }
}

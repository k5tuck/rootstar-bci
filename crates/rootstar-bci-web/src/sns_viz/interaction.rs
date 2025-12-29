//! User Interaction for SNS Visualization
//!
//! Handles mouse/touch interaction with receptor meshes for probing and stimulus application.

use std::collections::HashMap;

use super::meshes::{GpuMesh, MeshId};

/// Interaction event types
#[derive(Clone, Debug)]
pub enum InteractionEvent {
    /// Mouse/touch click at screen coordinates
    Click { x: f32, y: f32 },
    /// Mouse/touch drag (delta movement)
    Drag { dx: f32, dy: f32, buttons: u32 },
    /// Mouse wheel/pinch zoom
    Zoom { delta: f32 },
    /// Hover at screen coordinates
    Hover { x: f32, y: f32 },
}

/// Result of probing a receptor
#[derive(Clone, Debug)]
pub struct ProbeResult {
    /// Mesh that was hit
    pub mesh_id: MeshId,
    /// Index of the receptor
    pub receptor_index: usize,
    /// Position on the mesh (UV coordinates)
    pub position: [f32; 2],
    /// Current activation value
    pub activation: f32,
    /// Receptor type (if available)
    pub receptor_type: Option<String>,
}

/// Interaction mode
#[derive(Clone, Debug, PartialEq)]
pub enum InteractionMode {
    /// View only (orbit, zoom, pan)
    View,
    /// Probe receptors (click to get info)
    Probe,
    /// Apply stimulus (click and drag)
    Stimulus,
    /// Select region
    Select,
}

impl Default for InteractionMode {
    fn default() -> Self {
        InteractionMode::View
    }
}

/// Handler for user interactions
#[derive(Clone, Debug)]
pub struct InteractionHandler {
    /// Current interaction mode
    mode: InteractionMode,
    /// Last mouse position
    last_position: Option<[f32; 2]>,
    /// Currently hovered receptor
    hovered_receptor: Option<(MeshId, usize)>,
    /// Selected receptors
    selected_receptors: Vec<(MeshId, usize)>,
    /// Stimulus brush radius (in UV space)
    brush_radius: f32,
    /// Stimulus intensity
    stimulus_intensity: f32,
}

impl InteractionHandler {
    /// Create a new interaction handler
    pub fn new() -> Self {
        Self {
            mode: InteractionMode::View,
            last_position: None,
            hovered_receptor: None,
            selected_receptors: Vec::new(),
            brush_radius: 0.05,
            stimulus_intensity: 1.0,
        }
    }

    /// Set interaction mode
    pub fn set_mode(&mut self, mode: InteractionMode) {
        self.mode = mode;
    }

    /// Get current mode
    pub fn mode(&self) -> &InteractionMode {
        &self.mode
    }

    /// Set brush radius for stimulus mode
    pub fn set_brush_radius(&mut self, radius: f32) {
        self.brush_radius = radius.clamp(0.01, 0.5);
    }

    /// Set stimulus intensity
    pub fn set_stimulus_intensity(&mut self, intensity: f32) {
        self.stimulus_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Handle an interaction event
    pub fn handle_event(
        &mut self,
        event: InteractionEvent,
        meshes: &HashMap<MeshId, GpuMesh>,
        activations: &HashMap<MeshId, Vec<f32>>,
    ) -> Option<ProbeResult> {
        match event {
            InteractionEvent::Click { x, y } => {
                self.last_position = Some([x, y]);
                match self.mode {
                    InteractionMode::Probe => {
                        return self.probe_at(x, y, meshes, activations);
                    }
                    InteractionMode::Select => {
                        if let Some(result) = self.probe_at(x, y, meshes, activations) {
                            self.selected_receptors.push((result.mesh_id.clone(), result.receptor_index));
                            return Some(result);
                        }
                    }
                    _ => {}
                }
            }
            InteractionEvent::Hover { x, y } => {
                self.last_position = Some([x, y]);
                // Update hovered receptor
                if let Some(result) = self.probe_at(x, y, meshes, activations) {
                    self.hovered_receptor = Some((result.mesh_id.clone(), result.receptor_index));
                } else {
                    self.hovered_receptor = None;
                }
            }
            InteractionEvent::Drag { dx, dy, buttons: _ } => {
                if let Some([lx, ly]) = self.last_position {
                    self.last_position = Some([lx + dx, ly + dy]);
                }
            }
            InteractionEvent::Zoom { delta: _ } => {
                // Zoom is handled by the camera directly
            }
        }
        None
    }

    /// Probe for a receptor at screen coordinates
    fn probe_at(
        &self,
        x: f32,
        y: f32,
        meshes: &HashMap<MeshId, GpuMesh>,
        activations: &HashMap<MeshId, Vec<f32>>,
    ) -> Option<ProbeResult> {
        // Simplified: treat x, y as UV coordinates (0-1 range)
        // In a real implementation, you'd ray-cast into the scene
        let uv_x = x;
        let uv_y = y;

        let mut closest: Option<(MeshId, usize, f32, [f32; 2])> = None;

        for (mesh_id, mesh) in meshes.iter() {
            for (i, receptor) in mesh.receptor_positions.iter().enumerate() {
                let dx = receptor.uv[0] - uv_x;
                let dy = receptor.uv[1] - uv_y;
                let dist_sq = dx * dx + dy * dy;

                let threshold_sq = 0.01 * 0.01; // 1% UV distance threshold
                if dist_sq < threshold_sq {
                    if closest.is_none() || dist_sq < closest.as_ref().unwrap().2 {
                        closest = Some((mesh_id.clone(), i, dist_sq, receptor.uv));
                    }
                }
            }
        }

        closest.map(|(mesh_id, idx, _, position)| {
            let activation = activations
                .get(&mesh_id)
                .and_then(|a| a.get(idx).copied())
                .unwrap_or(0.0);

            ProbeResult {
                mesh_id,
                receptor_index: idx,
                position,
                activation,
                receptor_type: None,
            }
        })
    }

    /// Get receptors within brush radius of a UV position
    pub fn get_receptors_in_brush(
        &self,
        uv: [f32; 2],
        mesh: &GpuMesh,
    ) -> Vec<(usize, f32)> {
        let mut result = Vec::new();

        for (i, receptor) in mesh.receptor_positions.iter().enumerate() {
            let dx = receptor.uv[0] - uv[0];
            let dy = receptor.uv[1] - uv[1];
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= self.brush_radius {
                // Weight falls off with distance from brush center
                let weight = 1.0 - (dist / self.brush_radius);
                result.push((i, weight * self.stimulus_intensity));
            }
        }

        result
    }

    /// Get currently hovered receptor
    pub fn hovered(&self) -> Option<&(MeshId, usize)> {
        self.hovered_receptor.as_ref()
    }

    /// Get selected receptors
    pub fn selected(&self) -> &[(MeshId, usize)] {
        &self.selected_receptors
    }

    /// Clear selection
    pub fn clear_selection(&mut self) {
        self.selected_receptors.clear();
    }
}

impl Default for InteractionHandler {
    fn default() -> Self {
        Self::new()
    }
}

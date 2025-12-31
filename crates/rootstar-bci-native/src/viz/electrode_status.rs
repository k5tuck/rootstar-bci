//! Electrode Status Visualization
//!
//! Provides a visual representation of electrode placement using the 10-20 system
//! with real-time connection and impedance status.

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, Vec2};

/// Electrode connection status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ElectrodeStatus {
    /// Not connected / not measured
    #[default]
    NotConnected,
    /// Connected with good signal
    Connected,
    /// High impedance warning
    HighImpedance,
    /// Very high impedance / poor contact
    PoorContact,
    /// Electrode saturated
    Saturated,
    /// Reference electrode
    Reference,
}

impl ElectrodeStatus {
    /// Get the color for this status
    #[must_use]
    pub fn color(&self) -> Color32 {
        match self {
            Self::NotConnected => Color32::from_rgb(156, 163, 175), // Gray
            Self::Connected => Color32::from_rgb(34, 197, 94),      // Green
            Self::HighImpedance => Color32::from_rgb(251, 191, 36), // Yellow/Amber
            Self::PoorContact => Color32::from_rgb(239, 68, 68),    // Red
            Self::Saturated => Color32::from_rgb(168, 85, 247),     // Purple
            Self::Reference => Color32::from_rgb(59, 130, 246),     // Blue
        }
    }

    /// Get status label
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotConnected => "Not Connected",
            Self::Connected => "Connected",
            Self::HighImpedance => "High Impedance",
            Self::PoorContact => "Poor Contact",
            Self::Saturated => "Saturated",
            Self::Reference => "Reference",
        }
    }
}

/// Electrode position in the 10-20 system
#[derive(Clone, Copy, Debug)]
pub struct ElectrodePosition {
    /// Channel index (0-7 for 8-channel system)
    pub channel: usize,
    /// Electrode label (e.g., "Fp1", "O2")
    pub label: &'static str,
    /// X position (0.0-1.0, relative to head)
    pub x: f32,
    /// Y position (0.0-1.0, relative to head)
    pub y: f32,
    /// Description of brain region
    pub region: &'static str,
}

/// Standard 8-channel electrode positions (10-20 system)
pub const ELECTRODE_POSITIONS: [ElectrodePosition; 8] = [
    ElectrodePosition {
        channel: 0,
        label: "Fp1",
        x: 0.35,
        y: 0.15,
        region: "Left Prefrontal",
    },
    ElectrodePosition {
        channel: 1,
        label: "Fp2",
        x: 0.65,
        y: 0.15,
        region: "Right Prefrontal",
    },
    ElectrodePosition {
        channel: 2,
        label: "F3",
        x: 0.30,
        y: 0.30,
        region: "Left Frontal",
    },
    ElectrodePosition {
        channel: 3,
        label: "F4",
        x: 0.70,
        y: 0.30,
        region: "Right Frontal",
    },
    ElectrodePosition {
        channel: 4,
        label: "C3",
        x: 0.25,
        y: 0.50,
        region: "Left Central",
    },
    ElectrodePosition {
        channel: 5,
        label: "C4",
        x: 0.75,
        y: 0.50,
        region: "Right Central",
    },
    ElectrodePosition {
        channel: 6,
        label: "O1",
        x: 0.35,
        y: 0.85,
        region: "Left Occipital",
    },
    ElectrodePosition {
        channel: 7,
        label: "O2",
        x: 0.65,
        y: 0.85,
        region: "Right Occipital",
    },
];

/// Reference electrode positions
pub const REFERENCE_POSITIONS: [ElectrodePosition; 2] = [
    ElectrodePosition {
        channel: 8,
        label: "A1",
        x: 0.05,
        y: 0.50,
        region: "Left Ear (Reference)",
    },
    ElectrodePosition {
        channel: 9,
        label: "A2",
        x: 0.95,
        y: 0.50,
        region: "Right Ear (Reference)",
    },
];

/// Per-channel electrode state
#[derive(Clone, Debug, Default)]
pub struct ElectrodeState {
    /// Connection status
    pub status: ElectrodeStatus,
    /// Impedance in kOhms (None if not measured)
    pub impedance_kohm: Option<f32>,
    /// Signal quality (0.0-1.0)
    pub signal_quality: f32,
    /// Is currently selected/highlighted
    pub selected: bool,
}

impl ElectrodeState {
    /// Create state from impedance measurement
    #[must_use]
    pub fn from_impedance(impedance_kohm: f32) -> Self {
        let status = if impedance_kohm > 100.0 {
            ElectrodeStatus::PoorContact
        } else if impedance_kohm > 50.0 {
            ElectrodeStatus::HighImpedance
        } else {
            ElectrodeStatus::Connected
        };

        // Signal quality inversely related to impedance
        let signal_quality = (1.0 - impedance_kohm / 100.0).clamp(0.0, 1.0);

        Self {
            status,
            impedance_kohm: Some(impedance_kohm),
            signal_quality,
            selected: false,
        }
    }
}

/// Electrode status panel for visualizing electrode placement and connection status
pub struct ElectrodeStatusPanel {
    /// Per-channel electrode states
    pub electrodes: [ElectrodeState; 8],
    /// Reference electrode states
    pub references: [ElectrodeState; 2],
    /// Show impedance values
    pub show_impedance: bool,
    /// Show reference electrodes
    pub show_references: bool,
    /// Currently hovered electrode
    hovered_electrode: Option<usize>,
    /// Is impedance check in progress
    pub checking_impedance: bool,
    /// Panel title
    pub title: String,
}

impl Default for ElectrodeStatusPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElectrodeStatusPanel {
    /// Create a new electrode status panel
    #[must_use]
    pub fn new() -> Self {
        Self {
            electrodes: Default::default(),
            references: [
                ElectrodeState {
                    status: ElectrodeStatus::Reference,
                    ..Default::default()
                },
                ElectrodeState {
                    status: ElectrodeStatus::Reference,
                    ..Default::default()
                },
            ],
            show_impedance: true,
            show_references: true,
            hovered_electrode: None,
            checking_impedance: false,
            title: "Electrode Status".to_string(),
        }
    }

    /// Update electrode status from impedance measurements
    pub fn update_from_impedance(&mut self, impedances: &[f32]) {
        for (i, &imp) in impedances.iter().enumerate() {
            if i < 8 {
                self.electrodes[i] = ElectrodeState::from_impedance(imp);
            } else if i < 10 {
                self.references[i - 8] = ElectrodeState::from_impedance(imp);
                self.references[i - 8].status = if imp > 50.0 {
                    ElectrodeStatus::HighImpedance
                } else {
                    ElectrodeStatus::Reference
                };
            }
        }
    }

    /// Set all electrodes to connected (for simulation/testing)
    pub fn set_all_connected(&mut self) {
        for electrode in &mut self.electrodes {
            electrode.status = ElectrodeStatus::Connected;
            electrode.impedance_kohm = Some(15.0);
            electrode.signal_quality = 0.85;
        }
    }

    /// Get overall connection status summary
    #[must_use]
    pub fn connection_summary(&self) -> (usize, usize, usize) {
        let mut connected = 0;
        let mut warning = 0;
        let mut disconnected = 0;

        for electrode in &self.electrodes {
            match electrode.status {
                ElectrodeStatus::Connected => connected += 1,
                ElectrodeStatus::HighImpedance => warning += 1,
                ElectrodeStatus::PoorContact | ElectrodeStatus::NotConnected => disconnected += 1,
                ElectrodeStatus::Saturated => warning += 1,
                ElectrodeStatus::Reference => connected += 1,
            }
        }

        (connected, warning, disconnected)
    }

    /// Render the electrode status panel
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            // Title and status summary
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (connected, warning, disconnected) = self.connection_summary();
                    if disconnected > 0 {
                        ui.label(
                            RichText::new(format!("{} offline", disconnected))
                                .color(Color32::RED)
                                .small(),
                        );
                    }
                    if warning > 0 {
                        ui.label(
                            RichText::new(format!("{} warning", warning))
                                .color(Color32::YELLOW)
                                .small(),
                        );
                    }
                    ui.label(
                        RichText::new(format!("{}/8 connected", connected))
                            .color(Color32::GREEN)
                            .small(),
                    );
                });
            });

            ui.separator();

            // Head diagram
            let available_width = ui.available_width().min(300.0);
            let head_size = Vec2::new(available_width, available_width * 1.1);
            let (head_rect, response) = ui.allocate_exact_size(head_size, Sense::hover());

            self.draw_head_diagram(ui, head_rect);

            // Check for hover
            if let Some(pos) = response.hover_pos() {
                self.hovered_electrode = self.find_electrode_at(head_rect, pos);
            } else {
                self.hovered_electrode = None;
            }

            ui.separator();

            // Legend
            self.draw_legend(ui);

            // Impedance check button
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if self.checking_impedance {
                    ui.spinner();
                    ui.label("Checking impedance...");
                } else if ui.button("Check Impedance").clicked() {
                    self.checking_impedance = true;
                }

                ui.checkbox(&mut self.show_impedance, "Show Values");
                ui.checkbox(&mut self.show_references, "Show Refs");
            });

            // Selected electrode details
            if let Some(idx) = self.hovered_electrode {
                ui.add_space(10.0);
                self.draw_electrode_details(ui, idx);
            }
        });
    }

    /// Draw the head diagram with electrodes
    fn draw_head_diagram(&self, ui: &mut egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let head_radius = rect.width() * 0.42;

        // Background
        painter.rect_filled(rect, 8.0, Color32::from_gray(25));

        // Head outline (oval)
        let head_rect = Rect::from_center_size(
            center,
            Vec2::new(head_radius * 2.0, head_radius * 2.2),
        );
        painter.rect_stroke(head_rect, head_radius, Stroke::new(2.0, Color32::from_gray(80)));

        // Nose indicator (triangle at top)
        let nose_tip = Pos2::new(center.x, rect.min.y + 10.0);
        let nose_left = Pos2::new(center.x - 10.0, rect.min.y + 30.0);
        let nose_right = Pos2::new(center.x + 10.0, rect.min.y + 30.0);
        painter.add(egui::Shape::convex_polygon(
            vec![nose_tip, nose_left, nose_right],
            Color32::from_gray(60),
            Stroke::new(1.0, Color32::from_gray(80)),
        ));

        // Ears (reference positions)
        if self.show_references {
            // Left ear
            let left_ear = Pos2::new(rect.min.x + 15.0, center.y);
            painter.circle_stroke(left_ear, 12.0, Stroke::new(2.0, Color32::from_gray(80)));

            // Right ear
            let right_ear = Pos2::new(rect.max.x - 15.0, center.y);
            painter.circle_stroke(right_ear, 12.0, Stroke::new(2.0, Color32::from_gray(80)));

            // Reference electrodes
            for (i, ref_pos) in REFERENCE_POSITIONS.iter().enumerate() {
                let pos = self.electrode_screen_pos(rect, ref_pos.x, ref_pos.y);
                self.draw_electrode(&painter, pos, &self.references[i], ref_pos.label, i == 0);
            }
        }

        // Crosshairs (anterior-posterior and left-right lines)
        painter.line_segment(
            [
                Pos2::new(center.x, rect.min.y + head_radius * 0.3),
                Pos2::new(center.x, rect.max.y - head_radius * 0.2),
            ],
            Stroke::new(1.0, Color32::from_gray(50)),
        );
        painter.line_segment(
            [
                Pos2::new(rect.min.x + head_radius * 0.4, center.y),
                Pos2::new(rect.max.x - head_radius * 0.4, center.y),
            ],
            Stroke::new(1.0, Color32::from_gray(50)),
        );

        // Draw electrodes
        for (i, elec_pos) in ELECTRODE_POSITIONS.iter().enumerate() {
            let pos = self.electrode_screen_pos(rect, elec_pos.x, elec_pos.y);
            let is_hovered = self.hovered_electrode == Some(i);
            self.draw_electrode(&painter, pos, &self.electrodes[i], elec_pos.label, is_hovered);
        }
    }

    /// Convert normalized position to screen position
    fn electrode_screen_pos(&self, rect: Rect, x: f32, y: f32) -> Pos2 {
        Pos2::new(
            rect.min.x + x * rect.width(),
            rect.min.y + y * rect.height(),
        )
    }

    /// Draw a single electrode
    fn draw_electrode(
        &self,
        painter: &egui::Painter,
        pos: Pos2,
        state: &ElectrodeState,
        label: &str,
        is_hovered: bool,
    ) {
        let radius = if is_hovered { 18.0 } else { 14.0 };
        let color = state.status.color();

        // Outer ring (for hover/selection)
        if is_hovered || state.selected {
            painter.circle_stroke(pos, radius + 3.0, Stroke::new(2.0, Color32::WHITE));
        }

        // Main circle
        painter.circle_filled(pos, radius, color);

        // Border
        painter.circle_stroke(pos, radius, Stroke::new(1.5, Color32::from_gray(200)));

        // Label
        let label_color = if color.r() as u16 + color.g() as u16 + color.b() as u16 > 400 {
            Color32::BLACK
        } else {
            Color32::WHITE
        };
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(11.0),
            label_color,
        );

        // Impedance value (below electrode)
        if self.show_impedance {
            if let Some(imp) = state.impedance_kohm {
                let imp_text = format!("{:.0}k", imp);
                painter.text(
                    Pos2::new(pos.x, pos.y + radius + 10.0),
                    egui::Align2::CENTER_TOP,
                    imp_text,
                    egui::FontId::proportional(9.0),
                    Color32::from_gray(180),
                );
            }
        }
    }

    /// Draw the status legend
    fn draw_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let statuses = [
                ElectrodeStatus::Connected,
                ElectrodeStatus::HighImpedance,
                ElectrodeStatus::PoorContact,
                ElectrodeStatus::NotConnected,
            ];

            for status in statuses {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, status.color());
                ui.label(RichText::new(status.label()).small());
                ui.add_space(8.0);
            }
        });
    }

    /// Draw details for a specific electrode
    fn draw_electrode_details(&self, ui: &mut egui::Ui, channel: usize) {
        if channel >= 8 {
            return;
        }

        let pos = &ELECTRODE_POSITIONS[channel];
        let state = &self.electrodes[channel];

        ui.group(|ui| {
            ui.horizontal(|ui| {
                // Color indicator
                let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 7.0, state.status.color());

                ui.vertical(|ui| {
                    ui.label(RichText::new(pos.label).strong());
                    ui.label(RichText::new(pos.region).small().color(Color32::GRAY));
                });
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(RichText::new(state.status.label()).color(state.status.color()));
            });

            if let Some(imp) = state.impedance_kohm {
                ui.horizontal(|ui| {
                    ui.label("Impedance:");
                    let imp_color = if imp > 50.0 {
                        Color32::RED
                    } else if imp > 20.0 {
                        Color32::YELLOW
                    } else {
                        Color32::GREEN
                    };
                    ui.label(RichText::new(format!("{:.1} kΩ", imp)).color(imp_color));
                });
            }

            ui.horizontal(|ui| {
                ui.label("Signal Quality:");
                let quality_pct = (state.signal_quality * 100.0) as i32;
                let quality_color = if quality_pct > 80 {
                    Color32::GREEN
                } else if quality_pct > 50 {
                    Color32::YELLOW
                } else {
                    Color32::RED
                };
                ui.label(RichText::new(format!("{}%", quality_pct)).color(quality_color));

                // Quality bar
                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(60.0, 8.0), Sense::hover());
                ui.painter().rect_filled(bar_rect, 2.0, Color32::from_gray(40));
                let filled_width = bar_rect.width() * state.signal_quality;
                let filled_rect = Rect::from_min_size(
                    bar_rect.min,
                    Vec2::new(filled_width, bar_rect.height()),
                );
                ui.painter().rect_filled(filled_rect, 2.0, quality_color);
            });
        });
    }

    /// Find which electrode is at the given screen position
    fn find_electrode_at(&self, head_rect: Rect, pos: Pos2) -> Option<usize> {
        let hit_radius = 20.0;

        for (i, elec_pos) in ELECTRODE_POSITIONS.iter().enumerate() {
            let elec_screen_pos = self.electrode_screen_pos(head_rect, elec_pos.x, elec_pos.y);
            let dist = pos.distance(elec_screen_pos);
            if dist < hit_radius {
                return Some(i);
            }
        }

        None
    }
}

/// Compact electrode status bar (for use in headers/toolbars)
pub struct ElectrodeStatusBar {
    /// Electrode states
    pub electrodes: [ElectrodeState; 8],
}

impl Default for ElectrodeStatusBar {
    fn default() -> Self {
        Self::new()
    }
}

impl ElectrodeStatusBar {
    /// Create a new status bar
    #[must_use]
    pub fn new() -> Self {
        Self {
            electrodes: Default::default(),
        }
    }

    /// Update from impedance array
    pub fn update_from_impedance(&mut self, impedances: &[f32]) {
        for (i, &imp) in impedances.iter().take(8).enumerate() {
            self.electrodes[i] = ElectrodeState::from_impedance(imp);
        }
    }

    /// Render compact status bar
    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (i, state) in self.electrodes.iter().enumerate() {
                let label = ELECTRODE_POSITIONS[i].label;
                let (rect, response) = ui.allocate_exact_size(Vec2::new(24.0, 24.0), Sense::hover());

                // Background circle
                ui.painter().circle_filled(rect.center(), 10.0, state.status.color());
                ui.painter().circle_stroke(
                    rect.center(),
                    10.0,
                    Stroke::new(1.0, Color32::from_gray(180)),
                );

                // Label
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(8.0),
                    Color32::WHITE,
                );

                // Tooltip
                if response.hovered() {
                    let tooltip = format!(
                        "{}\n{}\nImpedance: {}",
                        label,
                        state.status.label(),
                        state
                            .impedance_kohm
                            .map_or("N/A".to_string(), |i| format!("{:.1} kΩ", i))
                    );
                    response.on_hover_text(tooltip);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_electrode_status_colors() {
        assert_eq!(ElectrodeStatus::Connected.color(), Color32::from_rgb(34, 197, 94));
        assert_eq!(ElectrodeStatus::PoorContact.color(), Color32::from_rgb(239, 68, 68));
    }

    #[test]
    fn test_electrode_state_from_impedance() {
        let good = ElectrodeState::from_impedance(10.0);
        assert_eq!(good.status, ElectrodeStatus::Connected);

        let warning = ElectrodeState::from_impedance(60.0);
        assert_eq!(warning.status, ElectrodeStatus::HighImpedance);

        let bad = ElectrodeState::from_impedance(150.0);
        assert_eq!(bad.status, ElectrodeStatus::PoorContact);
    }

    #[test]
    fn test_connection_summary() {
        let mut panel = ElectrodeStatusPanel::new();
        panel.set_all_connected();
        let (connected, warning, disconnected) = panel.connection_summary();
        assert_eq!(connected, 8);
        assert_eq!(warning, 0);
        assert_eq!(disconnected, 0);
    }

    #[test]
    fn test_electrode_positions() {
        assert_eq!(ELECTRODE_POSITIONS.len(), 8);
        assert_eq!(ELECTRODE_POSITIONS[0].label, "Fp1");
        assert_eq!(ELECTRODE_POSITIONS[7].label, "O2");
    }
}

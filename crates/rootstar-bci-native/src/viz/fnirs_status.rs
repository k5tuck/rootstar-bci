//! fNIRS Optode Status Visualization
//!
//! Provides a visual representation of fNIRS source (LED) and detector (photodiode)
//! placement with real-time signal quality and light intensity monitoring.

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, Vec2};

/// Optode type (source LED or detector photodiode)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptodeType {
    /// Light source (NIR LED)
    Source,
    /// Light detector (photodiode)
    Detector,
}

/// LED wavelength
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wavelength {
    /// 760nm - sensitive to deoxyhemoglobin (HbR)
    Nm760,
    /// 850nm - sensitive to oxyhemoglobin (HbO2)
    Nm850,
    /// Dual wavelength source
    Dual,
}

impl Wavelength {
    /// Get display color for wavelength
    #[must_use]
    pub fn color(&self) -> Color32 {
        match self {
            Self::Nm760 => Color32::from_rgb(180, 0, 0),     // Deep red (invisible but represented)
            Self::Nm850 => Color32::from_rgb(139, 0, 0),     // Darker red (even more IR)
            Self::Dual => Color32::from_rgb(160, 0, 50),     // Blend
        }
    }

    /// Get wavelength label
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Nm760 => "760nm",
            Self::Nm850 => "850nm",
            Self::Dual => "Dual",
        }
    }
}

/// Optode connection/signal status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OptodeStatus {
    /// Not connected / not detected
    #[default]
    NotConnected,
    /// Connected with good signal
    Good,
    /// Low signal (poor contact or high ambient light)
    LowSignal,
    /// Saturated (too much light)
    Saturated,
    /// No light detected (source off or blocked)
    NoSignal,
    /// Active/emitting (for sources)
    Active,
}

impl OptodeStatus {
    /// Get status color
    #[must_use]
    pub fn color(&self) -> Color32 {
        match self {
            Self::NotConnected => Color32::from_rgb(107, 114, 128), // Gray
            Self::Good => Color32::from_rgb(34, 197, 94),           // Green
            Self::LowSignal => Color32::from_rgb(251, 191, 36),     // Amber
            Self::Saturated => Color32::from_rgb(168, 85, 247),     // Purple
            Self::NoSignal => Color32::from_rgb(239, 68, 68),       // Red
            Self::Active => Color32::from_rgb(59, 130, 246),        // Blue
        }
    }

    /// Get status label
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotConnected => "Not Connected",
            Self::Good => "Good Signal",
            Self::LowSignal => "Low Signal",
            Self::Saturated => "Saturated",
            Self::NoSignal => "No Signal",
            Self::Active => "Active",
        }
    }
}

/// Single optode position and state
#[derive(Clone, Debug)]
pub struct OptodePosition {
    /// Optode ID
    pub id: usize,
    /// Optode type
    pub optode_type: OptodeType,
    /// Wavelength (for sources)
    pub wavelength: Option<Wavelength>,
    /// Label (e.g., "S1", "D1")
    pub label: String,
    /// X position (0.0-1.0, relative to head)
    pub x: f32,
    /// Y position (0.0-1.0, relative to head)
    pub y: f32,
    /// Brain region
    pub region: String,
}

/// Source-detector channel (measurement path)
#[derive(Clone, Debug)]
pub struct FnirsChannel {
    /// Channel index
    pub index: usize,
    /// Source optode index
    pub source_idx: usize,
    /// Detector optode index
    pub detector_idx: usize,
    /// Source-detector separation (mm)
    pub separation_mm: f32,
    /// Brain region being measured
    pub region: String,
}

/// Per-optode state
#[derive(Clone, Debug, Default)]
pub struct OptodeState {
    /// Connection/signal status
    pub status: OptodeStatus,
    /// Light intensity (0.0-1.0 normalized)
    pub intensity: f32,
    /// Signal-to-noise ratio (dB)
    pub snr_db: Option<f32>,
    /// Is currently selected
    pub selected: bool,
}

impl OptodeState {
    /// Create state from intensity measurement
    #[must_use]
    pub fn from_intensity(intensity: f32) -> Self {
        let status = if intensity < 0.05 {
            OptodeStatus::NoSignal
        } else if intensity < 0.2 {
            OptodeStatus::LowSignal
        } else if intensity > 0.95 {
            OptodeStatus::Saturated
        } else {
            OptodeStatus::Good
        };

        Self {
            status,
            intensity,
            snr_db: None,
            selected: false,
        }
    }
}

/// Per-channel state
#[derive(Clone, Debug, Default)]
pub struct ChannelState {
    /// Signal quality (0.0-1.0)
    pub signal_quality: f32,
    /// HbO value (micromolar)
    pub hbo_um: f32,
    /// HbR value (micromolar)
    pub hbr_um: f32,
    /// Is channel enabled
    pub enabled: bool,
    /// Is currently selected
    pub selected: bool,
}

/// Standard 4-channel fNIRS optode layout (prefrontal)
pub fn default_prefrontal_layout() -> (Vec<OptodePosition>, Vec<FnirsChannel>) {
    let sources = vec![
        OptodePosition {
            id: 0,
            optode_type: OptodeType::Source,
            wavelength: Some(Wavelength::Dual),
            label: "S1".to_string(),
            x: 0.35,
            y: 0.20,
            region: "Left Prefrontal".to_string(),
        },
        OptodePosition {
            id: 1,
            optode_type: OptodeType::Source,
            wavelength: Some(Wavelength::Dual),
            label: "S2".to_string(),
            x: 0.65,
            y: 0.20,
            region: "Right Prefrontal".to_string(),
        },
    ];

    let detectors = vec![
        OptodePosition {
            id: 2,
            optode_type: OptodeType::Detector,
            wavelength: None,
            label: "D1".to_string(),
            x: 0.25,
            y: 0.28,
            region: "Left Prefrontal".to_string(),
        },
        OptodePosition {
            id: 3,
            optode_type: OptodeType::Detector,
            wavelength: None,
            label: "D2".to_string(),
            x: 0.45,
            y: 0.28,
            region: "Left Prefrontal".to_string(),
        },
        OptodePosition {
            id: 4,
            optode_type: OptodeType::Detector,
            wavelength: None,
            label: "D3".to_string(),
            x: 0.55,
            y: 0.28,
            region: "Right Prefrontal".to_string(),
        },
        OptodePosition {
            id: 5,
            optode_type: OptodeType::Detector,
            wavelength: None,
            label: "D4".to_string(),
            x: 0.75,
            y: 0.28,
            region: "Right Prefrontal".to_string(),
        },
    ];

    let mut optodes = sources;
    optodes.extend(detectors);

    let channels = vec![
        FnirsChannel {
            index: 0,
            source_idx: 0,
            detector_idx: 2,
            separation_mm: 30.0,
            region: "Left Lateral PFC".to_string(),
        },
        FnirsChannel {
            index: 1,
            source_idx: 0,
            detector_idx: 3,
            separation_mm: 30.0,
            region: "Left Medial PFC".to_string(),
        },
        FnirsChannel {
            index: 2,
            source_idx: 1,
            detector_idx: 4,
            separation_mm: 30.0,
            region: "Right Medial PFC".to_string(),
        },
        FnirsChannel {
            index: 3,
            source_idx: 1,
            detector_idx: 5,
            separation_mm: 30.0,
            region: "Right Lateral PFC".to_string(),
        },
    ];

    (optodes, channels)
}

/// fNIRS optode status panel
pub struct FnirsStatusPanel {
    /// Optode positions
    pub optodes: Vec<OptodePosition>,
    /// Source-detector channels
    pub channels: Vec<FnirsChannel>,
    /// Per-optode states
    pub optode_states: Vec<OptodeState>,
    /// Per-channel states
    pub channel_states: Vec<ChannelState>,
    /// Show channel lines
    pub show_channels: bool,
    /// Show intensity values
    pub show_intensity: bool,
    /// Currently hovered optode
    hovered_optode: Option<usize>,
    /// Currently hovered channel
    hovered_channel: Option<usize>,
    /// LED emission active
    pub leds_active: bool,
    /// Panel title
    pub title: String,
}

impl Default for FnirsStatusPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl FnirsStatusPanel {
    /// Create a new fNIRS status panel with default prefrontal layout
    #[must_use]
    pub fn new() -> Self {
        let (optodes, channels) = default_prefrontal_layout();
        let optode_count = optodes.len();
        let channel_count = channels.len();

        Self {
            optodes,
            channels,
            optode_states: vec![OptodeState::default(); optode_count],
            channel_states: vec![
                ChannelState {
                    enabled: true,
                    ..Default::default()
                };
                channel_count
            ],
            show_channels: true,
            show_intensity: true,
            hovered_optode: None,
            hovered_channel: None,
            leds_active: false,
            title: "fNIRS Status".to_string(),
        }
    }

    /// Update optode states from intensity measurements
    pub fn update_intensities(&mut self, intensities: &[f32]) {
        for (i, &intensity) in intensities.iter().enumerate() {
            if i < self.optode_states.len() {
                self.optode_states[i] = OptodeState::from_intensity(intensity);
            }
        }
    }

    /// Update channel HbO/HbR values
    pub fn update_hemoglobin(&mut self, channel: usize, hbo: f32, hbr: f32) {
        if channel < self.channel_states.len() {
            self.channel_states[channel].hbo_um = hbo;
            self.channel_states[channel].hbr_um = hbr;
            // Simple quality metric based on signal range
            self.channel_states[channel].signal_quality =
                (1.0 - (hbo.abs() + hbr.abs()) / 20.0).clamp(0.0, 1.0);
        }
    }

    /// Set LED emission state
    pub fn set_leds_active(&mut self, active: bool) {
        self.leds_active = active;
        // Update source optode status
        for (i, optode) in self.optodes.iter().enumerate() {
            if optode.optode_type == OptodeType::Source {
                if active {
                    self.optode_states[i].status = OptodeStatus::Active;
                } else if self.optode_states[i].status == OptodeStatus::Active {
                    self.optode_states[i].status = OptodeStatus::Good;
                }
            }
        }
    }

    /// Get channel quality summary
    #[must_use]
    pub fn quality_summary(&self) -> (usize, usize, usize) {
        let mut good = 0;
        let mut warning = 0;
        let mut bad = 0;

        for state in &self.channel_states {
            if !state.enabled {
                continue;
            }
            if state.signal_quality > 0.7 {
                good += 1;
            } else if state.signal_quality > 0.4 {
                warning += 1;
            } else {
                bad += 1;
            }
        }

        (good, warning, bad)
    }

    /// Render the fNIRS status panel
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            // Title and status summary
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // LED status indicator
                    let led_color = if self.leds_active {
                        Color32::from_rgb(239, 68, 68) // Red when active
                    } else {
                        Color32::GRAY
                    };
                    let led_text = if self.leds_active { "LEDs ON" } else { "LEDs OFF" };
                    ui.label(RichText::new(led_text).color(led_color).small());

                    ui.separator();

                    let (good, warning, bad) = self.quality_summary();
                    if bad > 0 {
                        ui.label(RichText::new(format!("{} poor", bad)).color(Color32::RED).small());
                    }
                    if warning > 0 {
                        ui.label(
                            RichText::new(format!("{} warning", warning))
                                .color(Color32::YELLOW)
                                .small(),
                        );
                    }
                    ui.label(
                        RichText::new(format!("{}/{} good", good, self.channels.len()))
                            .color(Color32::GREEN)
                            .small(),
                    );
                });
            });

            ui.separator();

            // Head diagram with optodes
            let available_width = ui.available_width().min(300.0);
            let head_size = Vec2::new(available_width, available_width * 0.6);
            let (head_rect, response) = ui.allocate_exact_size(head_size, Sense::hover());

            self.draw_head_diagram(ui, head_rect);

            // Check for hover
            if let Some(pos) = response.hover_pos() {
                self.hovered_optode = self.find_optode_at(head_rect, pos);
                if self.hovered_optode.is_none() {
                    self.hovered_channel = self.find_channel_at(head_rect, pos);
                } else {
                    self.hovered_channel = None;
                }
            } else {
                self.hovered_optode = None;
                self.hovered_channel = None;
            }

            ui.separator();

            // Channel values display
            self.draw_channel_values(ui);

            ui.separator();

            // Legend
            self.draw_legend(ui);

            // Controls
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_channels, "Show Channels");
                ui.checkbox(&mut self.show_intensity, "Show Values");
            });

            // Hovered element details
            if let Some(idx) = self.hovered_optode {
                ui.add_space(10.0);
                self.draw_optode_details(ui, idx);
            } else if let Some(idx) = self.hovered_channel {
                ui.add_space(10.0);
                self.draw_channel_details(ui, idx);
            }
        });
    }

    /// Draw the head diagram with optodes
    fn draw_head_diagram(&self, ui: &mut egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let center_x = rect.center().x;
        let head_width = rect.width() * 0.85;
        let head_height = rect.height() * 0.9;

        // Background
        painter.rect_filled(rect, 8.0, Color32::from_gray(25));

        // Head outline (forehead region - partial oval)
        let head_center = Pos2::new(center_x, rect.max.y);
        let head_rect = Rect::from_center_size(
            head_center,
            Vec2::new(head_width, head_height * 2.0),
        );

        // Draw partial ellipse for forehead
        painter.rect_stroke(
            Rect::from_min_max(
                Pos2::new(rect.min.x + (rect.width() - head_width) / 2.0, rect.min.y + 10.0),
                Pos2::new(rect.max.x - (rect.width() - head_width) / 2.0, rect.max.y),
            ),
            head_width / 2.0,
            Stroke::new(2.0, Color32::from_gray(80)),
        );

        // Hairline indicator
        let hairline_y = rect.min.y + 15.0;
        painter.line_segment(
            [
                Pos2::new(rect.min.x + rect.width() * 0.2, hairline_y),
                Pos2::new(rect.max.x - rect.width() * 0.2, hairline_y),
            ],
            Stroke::new(1.0, Color32::from_gray(60)),
        );
        painter.text(
            Pos2::new(rect.max.x - 30.0, hairline_y),
            egui::Align2::RIGHT_CENTER,
            "hairline",
            egui::FontId::proportional(9.0),
            Color32::from_gray(100),
        );

        // Draw channel lines first (behind optodes)
        if self.show_channels {
            for (i, channel) in self.channels.iter().enumerate() {
                let source_pos = &self.optodes[channel.source_idx];
                let detector_pos = &self.optodes[channel.detector_idx];

                let p1 = self.optode_screen_pos(rect, source_pos.x, source_pos.y);
                let p2 = self.optode_screen_pos(rect, detector_pos.x, detector_pos.y);

                let quality = self.channel_states.get(i).map(|s| s.signal_quality).unwrap_or(0.0);
                let line_color = if quality > 0.7 {
                    Color32::from_rgba_unmultiplied(34, 197, 94, 100)
                } else if quality > 0.4 {
                    Color32::from_rgba_unmultiplied(251, 191, 36, 100)
                } else {
                    Color32::from_rgba_unmultiplied(239, 68, 68, 100)
                };

                let is_hovered = self.hovered_channel == Some(i);
                let stroke_width = if is_hovered { 4.0 } else { 2.0 };

                painter.line_segment([p1, p2], Stroke::new(stroke_width, line_color));

                // Channel label at midpoint
                let mid = Pos2::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
                painter.text(
                    mid,
                    egui::Align2::CENTER_CENTER,
                    format!("Ch{}", i + 1),
                    egui::FontId::proportional(9.0),
                    Color32::from_gray(150),
                );
            }
        }

        // Draw optodes
        for (i, optode) in self.optodes.iter().enumerate() {
            let pos = self.optode_screen_pos(rect, optode.x, optode.y);
            let state = self.optode_states.get(i).cloned().unwrap_or_default();
            let is_hovered = self.hovered_optode == Some(i);

            self.draw_optode(&painter, pos, optode, &state, is_hovered);
        }
    }

    /// Convert normalized position to screen position
    fn optode_screen_pos(&self, rect: Rect, x: f32, y: f32) -> Pos2 {
        Pos2::new(rect.min.x + x * rect.width(), rect.min.y + y * rect.height())
    }

    /// Draw a single optode
    fn draw_optode(
        &self,
        painter: &egui::Painter,
        pos: Pos2,
        optode: &OptodePosition,
        state: &OptodeState,
        is_hovered: bool,
    ) {
        let radius = if is_hovered { 16.0 } else { 12.0 };

        // Different shapes for sources vs detectors
        match optode.optode_type {
            OptodeType::Source => {
                // Source: filled circle with LED color
                let color = if self.leds_active {
                    optode.wavelength.map(|w| w.color()).unwrap_or(Color32::RED)
                } else {
                    state.status.color()
                };

                // Glow effect when active
                if self.leds_active {
                    painter.circle_filled(pos, radius + 4.0, Color32::from_rgba_unmultiplied(255, 0, 0, 50));
                }

                if is_hovered || state.selected {
                    painter.circle_stroke(pos, radius + 3.0, Stroke::new(2.0, Color32::WHITE));
                }

                painter.circle_filled(pos, radius, color);
                painter.circle_stroke(pos, radius, Stroke::new(1.5, Color32::from_gray(200)));

                // LED symbol (small circle inside)
                painter.circle_filled(pos, radius * 0.4, Color32::from_gray(40));
            }
            OptodeType::Detector => {
                // Detector: square shape
                let half = radius * 0.85;
                let det_rect = Rect::from_center_size(pos, Vec2::splat(half * 2.0));

                if is_hovered || state.selected {
                    let outer_rect = Rect::from_center_size(pos, Vec2::splat((half + 3.0) * 2.0));
                    painter.rect_stroke(outer_rect, 2.0, Stroke::new(2.0, Color32::WHITE));
                }

                painter.rect_filled(det_rect, 3.0, state.status.color());
                painter.rect_stroke(det_rect, 3.0, Stroke::new(1.5, Color32::from_gray(200)));
            }
        }

        // Label
        let label_color = Color32::WHITE;
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            &optode.label,
            egui::FontId::proportional(9.0),
            label_color,
        );

        // Intensity value below
        if self.show_intensity {
            let intensity_text = format!("{:.0}%", state.intensity * 100.0);
            painter.text(
                Pos2::new(pos.x, pos.y + radius + 8.0),
                egui::Align2::CENTER_TOP,
                intensity_text,
                egui::FontId::proportional(8.0),
                Color32::from_gray(180),
            );
        }
    }

    /// Draw channel values
    fn draw_channel_values(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for (i, channel) in self.channels.iter().enumerate() {
                let state = self.channel_states.get(i).cloned().unwrap_or_default();

                ui.group(|ui| {
                    ui.set_min_width(60.0);
                    ui.vertical(|ui| {
                        // Channel header
                        let quality_color = if state.signal_quality > 0.7 {
                            Color32::GREEN
                        } else if state.signal_quality > 0.4 {
                            Color32::YELLOW
                        } else {
                            Color32::RED
                        };

                        ui.horizontal(|ui| {
                            let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 8.0), Sense::hover());
                            ui.painter().circle_filled(rect.center(), 3.0, quality_color);
                            ui.label(RichText::new(format!("Ch{}", i + 1)).small().strong());
                        });

                        // HbO/HbR values
                        let hbo_color = Color32::from_rgb(239, 68, 68);  // Red for HbO
                        let hbr_color = Color32::from_rgb(59, 130, 246); // Blue for HbR

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("O₂").small().color(hbo_color));
                            ui.label(RichText::new(format!("{:+.1}", state.hbo_um)).small());
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Hb").small().color(hbr_color));
                            ui.label(RichText::new(format!("{:+.1}", state.hbr_um)).small());
                        });
                    });
                });
            }
        });
    }

    /// Draw legend
    fn draw_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            // Source symbol
            let (rect, _) = ui.allocate_exact_size(Vec2::new(14.0, 14.0), Sense::hover());
            ui.painter().circle_filled(rect.center(), 6.0, Color32::from_rgb(180, 0, 0));
            ui.label(RichText::new("Source").small());

            ui.add_space(8.0);

            // Detector symbol
            let (rect, _) = ui.allocate_exact_size(Vec2::new(14.0, 14.0), Sense::hover());
            ui.painter().rect_filled(
                Rect::from_center_size(rect.center(), Vec2::splat(10.0)),
                2.0,
                Color32::from_rgb(34, 197, 94),
            );
            ui.label(RichText::new("Detector").small());

            ui.add_space(8.0);

            // Channel line
            let (rect, _) = ui.allocate_exact_size(Vec2::new(20.0, 14.0), Sense::hover());
            ui.painter().line_segment(
                [
                    Pos2::new(rect.min.x, rect.center().y),
                    Pos2::new(rect.max.x, rect.center().y),
                ],
                Stroke::new(2.0, Color32::from_rgba_unmultiplied(34, 197, 94, 150)),
            );
            ui.label(RichText::new("Channel").small());
        });
    }

    /// Draw optode details
    fn draw_optode_details(&self, ui: &mut egui::Ui, idx: usize) {
        if idx >= self.optodes.len() {
            return;
        }

        let optode = &self.optodes[idx];
        let state = self.optode_states.get(idx).cloned().unwrap_or_default();

        ui.group(|ui| {
            ui.horizontal(|ui| {
                // Type indicator
                let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                match optode.optode_type {
                    OptodeType::Source => {
                        ui.painter().circle_filled(rect.center(), 7.0, state.status.color());
                    }
                    OptodeType::Detector => {
                        ui.painter().rect_filled(
                            Rect::from_center_size(rect.center(), Vec2::splat(12.0)),
                            2.0,
                            state.status.color(),
                        );
                    }
                }

                ui.vertical(|ui| {
                    ui.label(RichText::new(&optode.label).strong());
                    let type_str = match optode.optode_type {
                        OptodeType::Source => "LED Source",
                        OptodeType::Detector => "Photodetector",
                    };
                    ui.label(RichText::new(type_str).small().color(Color32::GRAY));
                });
            });

            ui.separator();

            // Wavelength (for sources)
            if let Some(wavelength) = &optode.wavelength {
                ui.horizontal(|ui| {
                    ui.label("Wavelength:");
                    ui.label(RichText::new(wavelength.label()).color(wavelength.color()));
                });
            }

            ui.horizontal(|ui| {
                ui.label("Region:");
                ui.label(&optode.region);
            });

            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(RichText::new(state.status.label()).color(state.status.color()));
            });

            ui.horizontal(|ui| {
                ui.label("Intensity:");
                let pct = (state.intensity * 100.0) as i32;
                let color = if pct > 80 {
                    Color32::GREEN
                } else if pct > 40 {
                    Color32::YELLOW
                } else {
                    Color32::RED
                };
                ui.label(RichText::new(format!("{}%", pct)).color(color));

                // Intensity bar
                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(60.0, 8.0), Sense::hover());
                ui.painter().rect_filled(bar_rect, 2.0, Color32::from_gray(40));
                let filled = Rect::from_min_size(
                    bar_rect.min,
                    Vec2::new(bar_rect.width() * state.intensity, bar_rect.height()),
                );
                ui.painter().rect_filled(filled, 2.0, color);
            });
        });
    }

    /// Draw channel details
    fn draw_channel_details(&self, ui: &mut egui::Ui, idx: usize) {
        if idx >= self.channels.len() {
            return;
        }

        let channel = &self.channels[idx];
        let state = self.channel_states.get(idx).cloned().unwrap_or_default();

        ui.group(|ui| {
            ui.horizontal(|ui| {
                let quality_color = if state.signal_quality > 0.7 {
                    Color32::GREEN
                } else if state.signal_quality > 0.4 {
                    Color32::YELLOW
                } else {
                    Color32::RED
                };

                let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 7.0, quality_color);

                ui.vertical(|ui| {
                    ui.label(RichText::new(format!("Channel {}", idx + 1)).strong());
                    ui.label(RichText::new(&channel.region).small().color(Color32::GRAY));
                });
            });

            ui.separator();

            let source = &self.optodes[channel.source_idx];
            let detector = &self.optodes[channel.detector_idx];

            ui.horizontal(|ui| {
                ui.label("Path:");
                ui.label(format!("{} → {}", source.label, detector.label));
            });

            ui.horizontal(|ui| {
                ui.label("Separation:");
                ui.label(format!("{:.0} mm", channel.separation_mm));
            });

            ui.horizontal(|ui| {
                ui.label("Quality:");
                let pct = (state.signal_quality * 100.0) as i32;
                ui.label(format!("{}%", pct));
            });

            ui.horizontal(|ui| {
                ui.label("HbO₂:");
                ui.label(
                    RichText::new(format!("{:+.2} µM", state.hbo_um))
                        .color(Color32::from_rgb(239, 68, 68)),
                );
            });

            ui.horizontal(|ui| {
                ui.label("HbR:");
                ui.label(
                    RichText::new(format!("{:+.2} µM", state.hbr_um))
                        .color(Color32::from_rgb(59, 130, 246)),
                );
            });
        });
    }

    /// Find optode at screen position
    fn find_optode_at(&self, rect: Rect, pos: Pos2) -> Option<usize> {
        let hit_radius = 18.0;

        for (i, optode) in self.optodes.iter().enumerate() {
            let optode_pos = self.optode_screen_pos(rect, optode.x, optode.y);
            if pos.distance(optode_pos) < hit_radius {
                return Some(i);
            }
        }

        None
    }

    /// Find channel at screen position (check if near the line)
    fn find_channel_at(&self, rect: Rect, pos: Pos2) -> Option<usize> {
        let hit_distance = 8.0;

        for (i, channel) in self.channels.iter().enumerate() {
            let source_pos = &self.optodes[channel.source_idx];
            let detector_pos = &self.optodes[channel.detector_idx];

            let p1 = self.optode_screen_pos(rect, source_pos.x, source_pos.y);
            let p2 = self.optode_screen_pos(rect, detector_pos.x, detector_pos.y);

            // Distance from point to line segment
            let dist = point_to_line_distance(pos, p1, p2);
            if dist < hit_distance {
                return Some(i);
            }
        }

        None
    }
}

/// Calculate distance from point to line segment
fn point_to_line_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = Vec2::new(b.x - a.x, b.y - a.y);
    let ap = Vec2::new(p.x - a.x, p.y - a.y);

    let ab_len_sq = ab.x * ab.x + ab.y * ab.y;
    if ab_len_sq < 0.0001 {
        return ap.length();
    }

    let t = ((ap.x * ab.x + ap.y * ab.y) / ab_len_sq).clamp(0.0, 1.0);
    let closest = Pos2::new(a.x + t * ab.x, a.y + t * ab.y);

    p.distance(closest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optode_status_colors() {
        assert_eq!(OptodeStatus::Good.color(), Color32::from_rgb(34, 197, 94));
        assert_eq!(OptodeStatus::NoSignal.color(), Color32::from_rgb(239, 68, 68));
    }

    #[test]
    fn test_optode_state_from_intensity() {
        let low = OptodeState::from_intensity(0.1);
        assert_eq!(low.status, OptodeStatus::LowSignal);

        let good = OptodeState::from_intensity(0.5);
        assert_eq!(good.status, OptodeStatus::Good);

        let saturated = OptodeState::from_intensity(0.98);
        assert_eq!(saturated.status, OptodeStatus::Saturated);
    }

    #[test]
    fn test_default_layout() {
        let (optodes, channels) = default_prefrontal_layout();
        assert_eq!(optodes.len(), 6); // 2 sources + 4 detectors
        assert_eq!(channels.len(), 4);
    }

    #[test]
    fn test_quality_summary() {
        let mut panel = FnirsStatusPanel::new();
        panel.channel_states[0].signal_quality = 0.9;
        panel.channel_states[1].signal_quality = 0.5;
        panel.channel_states[2].signal_quality = 0.2;
        panel.channel_states[3].signal_quality = 0.8;

        let (good, warning, bad) = panel.quality_summary();
        assert_eq!(good, 2);
        assert_eq!(warning, 1);
        assert_eq!(bad, 1);
    }

    #[test]
    fn test_point_to_line_distance() {
        let a = Pos2::new(0.0, 0.0);
        let b = Pos2::new(10.0, 0.0);
        let p = Pos2::new(5.0, 5.0);

        let dist = point_to_line_distance(p, a, b);
        assert!((dist - 5.0).abs() < 0.01);
    }
}

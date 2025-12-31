//! Native Multi-Device Dashboard
//!
//! Provides an egui-based dashboard for viewing and managing multiple BCI devices
//! simultaneously with real-time data visualization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use tokio::sync::mpsc;

use crate::bridge::{ConnectionState, DeviceEvent, DeviceInfo, DeviceManager, DeviceState};
use crate::fingerprint::StimulationController;
use crate::processing::fnirs::FnirsProcessor;
use crate::viz::electrode_status::{ElectrodeStatusPanel, ElectrodeState};
use rootstar_bci_core::protocol::DeviceId;
use rootstar_bci_core::types::EegSample;

/// View mode for the dashboard
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// Single device focused view
    #[default]
    Single,
    /// Two devices side by side
    SideBySide,
    /// Signals overlaid on same plot
    Overlay,
    /// Grid of all devices
    Grid,
}

impl ViewMode {
    /// Get display name for the view mode
    pub fn name(&self) -> &'static str {
        match self {
            Self::Single => "Single",
            Self::SideBySide => "Side-by-Side",
            Self::Overlay => "Overlay",
            Self::Grid => "Grid",
        }
    }

    /// Get all available view modes
    pub fn all() -> &'static [ViewMode] {
        &[Self::Single, Self::SideBySide, Self::Overlay, Self::Grid]
    }
}

/// Colors for device visualization
const DEVICE_COLORS: [Color32; 8] = [
    Color32::from_rgb(59, 130, 246),  // Blue
    Color32::from_rgb(34, 197, 94),   // Green
    Color32::from_rgb(168, 85, 247),  // Purple
    Color32::from_rgb(249, 115, 22),  // Orange
    Color32::from_rgb(236, 72, 153),  // Pink
    Color32::from_rgb(34, 211, 238),  // Cyan
    Color32::from_rgb(251, 191, 36),  // Amber
    Color32::from_rgb(248, 113, 113), // Red
];

/// Data buffers for a device
pub struct DeviceDataBuffers {
    /// EEG sample ring buffer (per channel)
    pub eeg_channels: Vec<Vec<f32>>,
    /// fNIRS HbO values
    pub fnirs_hbo: Vec<f32>,
    /// fNIRS HbR values
    pub fnirs_hbr: Vec<f32>,
    /// Band power values (delta, theta, alpha, beta, gamma)
    pub band_powers: [f32; 5],
    /// Buffer capacity
    capacity: usize,
}

impl DeviceDataBuffers {
    /// Create new data buffers
    pub fn new(eeg_channels: usize, fnirs_channels: usize, capacity: usize) -> Self {
        Self {
            eeg_channels: vec![Vec::with_capacity(capacity); eeg_channels],
            fnirs_hbo: Vec::with_capacity(fnirs_channels * capacity),
            fnirs_hbr: Vec::with_capacity(fnirs_channels * capacity),
            band_powers: [0.0; 5],
            capacity,
        }
    }

    /// Push an EEG sample
    pub fn push_eeg(&mut self, sample: &EegSample) {
        for (i, channel) in self.eeg_channels.iter_mut().enumerate() {
            if i < 8 {
                let value = sample.channels[i].to_f32();
                if channel.len() >= self.capacity {
                    channel.remove(0);
                }
                channel.push(value);
            }
        }
    }

    /// Push fNIRS sample
    pub fn push_fnirs(&mut self, hbo: f32, hbr: f32) {
        if self.fnirs_hbo.len() >= self.capacity {
            self.fnirs_hbo.remove(0);
            self.fnirs_hbr.remove(0);
        }
        self.fnirs_hbo.push(hbo);
        self.fnirs_hbr.push(hbr);
    }

    /// Update band powers
    pub fn update_band_powers(&mut self, powers: [f32; 5]) {
        self.band_powers = powers;
    }
}

/// Per-device state in the dashboard
pub struct DashboardDevice {
    /// Device identifier
    pub id: DeviceId,
    /// Device info
    pub info: DeviceInfo,
    /// Connection state
    pub connection_state: ConnectionState,
    /// Data buffers
    pub buffers: DeviceDataBuffers,
    /// fNIRS processor
    pub fnirs_processor: FnirsProcessor,
    /// Stimulation controller
    pub stim_controller: StimulationController,
    /// Electrode status panel
    pub electrode_status: ElectrodeStatusPanel,
    /// Per-channel impedance values (kOhms)
    pub impedances: [f32; 8],
    /// Last update time
    pub last_update: Instant,
    /// Device color for visualization
    pub color: Color32,
    /// Is device selected/focused
    pub selected: bool,
    /// Is streaming data
    pub is_streaming: bool,
}

impl DashboardDevice {
    /// Create a new dashboard device
    pub fn new(id: DeviceId, info: DeviceInfo, color_index: usize) -> Self {
        let color = DEVICE_COLORS[color_index % DEVICE_COLORS.len()];
        let mut electrode_status = ElectrodeStatusPanel::new();
        electrode_status.title = info.name.clone();

        Self {
            id,
            info,
            connection_state: ConnectionState::Connected,
            buffers: DeviceDataBuffers::new(8, 4, 500),
            fnirs_processor: FnirsProcessor::new(30, 25), // 30mm separation, age 25
            stim_controller: StimulationController::new(),
            electrode_status,
            impedances: [0.0; 8],
            last_update: Instant::now(),
            color,
            selected: false,
            is_streaming: false,
        }
    }

    /// Update electrode impedance values
    pub fn update_impedances(&mut self, impedances: &[f32]) {
        for (i, &imp) in impedances.iter().take(8).enumerate() {
            self.impedances[i] = imp;
        }
        self.electrode_status.update_from_impedance(&self.impedances);
    }
}

/// Multi-device dashboard
pub struct MultiDeviceDashboard {
    /// Connected devices
    devices: HashMap<DeviceId, DashboardDevice>,
    /// Device ordering (for consistent display)
    device_order: Vec<DeviceId>,
    /// Current view mode
    view_mode: ViewMode,
    /// Selected device ID (for single view)
    selected_device: Option<DeviceId>,
    /// Show device connection panel
    show_connection_panel: bool,
    /// Show electrode status panel (right sidebar)
    show_electrode_panel: bool,
    /// Device manager (optional, for scanning)
    device_manager: Option<Arc<DeviceManager>>,
    /// Event receiver
    event_rx: Option<mpsc::Receiver<DeviceEvent>>,
    /// Device counter for color assignment
    device_counter: usize,
}

impl Default for MultiDeviceDashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDeviceDashboard {
    /// Create a new multi-device dashboard
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            device_order: Vec::new(),
            view_mode: ViewMode::Single,
            selected_device: None,
            show_connection_panel: false,
            show_electrode_panel: true, // Show by default
            device_manager: None,
            event_rx: None,
            device_counter: 0,
        }
    }

    /// Toggle electrode panel visibility
    pub fn toggle_electrode_panel(&mut self) {
        self.show_electrode_panel = !self.show_electrode_panel;
    }

    /// Set the device manager for connection handling
    pub fn set_device_manager(&mut self, manager: Arc<DeviceManager>) {
        self.device_manager = Some(manager);
    }

    /// Set the event receiver
    pub fn set_event_receiver(&mut self, rx: mpsc::Receiver<DeviceEvent>) {
        self.event_rx = Some(rx);
    }

    /// Add a device to the dashboard
    pub fn add_device(&mut self, id: DeviceId, info: DeviceInfo) {
        if !self.devices.contains_key(&id) {
            let device = DashboardDevice::new(id.clone(), info, self.device_counter);
            self.device_counter += 1;

            self.devices.insert(id.clone(), device);
            self.device_order.push(id.clone());

            // Auto-select first device
            if self.selected_device.is_none() {
                self.selected_device = Some(id);
            }
        }
    }

    /// Remove a device from the dashboard
    pub fn remove_device(&mut self, id: &DeviceId) {
        self.devices.remove(id);
        self.device_order.retain(|d| d != id);

        if self.selected_device.as_ref() == Some(id) {
            self.selected_device = self.device_order.first().cloned();
        }
    }

    /// Get mutable reference to a device
    pub fn device_mut(&mut self, id: &DeviceId) -> Option<&mut DashboardDevice> {
        self.devices.get_mut(id)
    }

    /// Get all connected device IDs
    pub fn device_ids(&self) -> &[DeviceId] {
        &self.device_order
    }

    /// Set the view mode
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.view_mode = mode;
    }

    /// Select a device
    pub fn select_device(&mut self, id: DeviceId) {
        if self.devices.contains_key(&id) {
            // Deselect previous
            if let Some(prev_id) = &self.selected_device {
                if let Some(dev) = self.devices.get_mut(prev_id) {
                    dev.selected = false;
                }
            }

            // Select new
            if let Some(dev) = self.devices.get_mut(&id) {
                dev.selected = true;
            }
            self.selected_device = Some(id);
        }
    }

    /// Toggle connection panel visibility
    pub fn toggle_connection_panel(&mut self) {
        self.show_connection_panel = !self.show_connection_panel;
    }

    /// Process pending device events
    pub fn process_events(&mut self) {
        // Collect events first to avoid borrow checker issues
        let events: Vec<DeviceEvent> = if let Some(ref mut rx) = self.event_rx {
            let mut events = Vec::new();
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
            events
        } else {
            Vec::new()
        };

        // Now process the collected events
        for event in events {
            match event {
                DeviceEvent::Discovered { device_id, info } => {
                    self.add_device(device_id, info);
                }
                DeviceEvent::Connected { device_id } => {
                    if let Some(device) = self.devices.get_mut(&device_id) {
                        device.connection_state = ConnectionState::Connected;
                    }
                }
                DeviceEvent::Disconnected { device_id, .. } => {
                    self.remove_device(&device_id);
                }
                DeviceEvent::Data { device_id, .. } => {
                    if let Some(device) = self.devices.get_mut(&device_id) {
                        device.is_streaming = true;
                        device.last_update = Instant::now();
                    }
                }
                DeviceEvent::Error { device_id, message } => {
                    tracing::warn!("Device {:?} error: {}", device_id, message);
                    if let Some(device) = self.devices.get_mut(&device_id) {
                        device.connection_state = ConnectionState::Error { message };
                    }
                }
                DeviceEvent::SignalQuality { .. } => {
                    // Ignore signal quality events for now
                }
            }
        }
    }

    /// Render the dashboard UI
    pub fn ui(&mut self, ctx: &egui::Context) {
        // Process any pending events
        self.process_events();

        // Top toolbar
        egui::TopBottomPanel::top("dashboard_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Rootstar BCI Dashboard");
                ui.separator();

                // View mode selector
                ui.label("View:");
                for mode in ViewMode::all() {
                    if ui
                        .selectable_label(self.view_mode == *mode, mode.name())
                        .clicked()
                    {
                        self.view_mode = *mode;
                    }
                }

                ui.separator();

                // Connection button
                if ui.button("+ Connect Device").clicked() {
                    self.show_connection_panel = true;
                }

                // Electrode status toggle
                let electrode_label = if self.show_electrode_panel { "Hide Electrodes" } else { "Show Electrodes" };
                if ui.button(electrode_label).clicked() {
                    self.show_electrode_panel = !self.show_electrode_panel;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{} devices connected", self.devices.len()));
                });
            });
        });

        // Device bar (left panel) - collect interactions then apply
        let mut device_to_select: Option<DeviceId> = None;
        let mut device_to_remove: Option<DeviceId> = None;
        let mut show_scan_panel = false;

        egui::SidePanel::left("device_bar")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Devices");
                ui.separator();

                let device_ids: Vec<_> = self.device_order.clone();
                let selected_id = self.selected_device.clone();

                for id in &device_ids {
                    if let Some(device) = self.devices.get(id) {
                        let is_selected = selected_id.as_ref() == Some(id);
                        let (select, remove) = Self::render_device_card_static(ui, device, is_selected);
                        if select {
                            device_to_select = Some(device.id.clone());
                        }
                        if remove {
                            device_to_remove = Some(device.id.clone());
                        }
                    }
                }

                if self.devices.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label(RichText::new("No devices connected").italics());
                        ui.add_space(10.0);
                        if ui.button("Scan for Devices").clicked() {
                            show_scan_panel = true;
                        }
                    });
                }
            });

        // Apply deferred interactions
        if let Some(id) = device_to_select {
            self.select_device(id);
        }
        if let Some(id) = device_to_remove {
            self.remove_device(&id);
        }
        if show_scan_panel {
            self.show_connection_panel = true;
        }

        // Electrode status panel (right sidebar)
        if self.show_electrode_panel {
            let selected_id = self.selected_device.clone();
            egui::SidePanel::right("electrode_panel")
                .resizable(true)
                .default_width(320.0)
                .min_width(280.0)
                .show(ctx, |ui| {
                    if let Some(ref id) = selected_id {
                        if let Some(device) = self.devices.get_mut(id) {
                            device.electrode_status.ui(ui);
                        }
                    } else if !self.devices.is_empty() {
                        // Show first device's electrode status if nothing selected
                        if let Some(first_id) = self.device_order.first() {
                            if let Some(device) = self.devices.get_mut(first_id) {
                                device.electrode_status.ui(ui);
                            }
                        }
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(50.0);
                            ui.heading("Electrode Status");
                            ui.separator();
                            ui.add_space(20.0);
                            ui.label(RichText::new("No device selected").italics());
                            ui.add_space(10.0);
                            ui.label("Connect a device to view\nelectrode status and impedance.");
                        });
                    }
                });
        }

        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.view_mode {
                ViewMode::Single => self.render_single_view(ui),
                ViewMode::SideBySide => self.render_side_by_side_view(ui),
                ViewMode::Overlay => self.render_overlay_view(ui),
                ViewMode::Grid => self.render_grid_view(ui),
            }
        });

        // Connection panel (modal)
        if self.show_connection_panel {
            self.render_connection_panel(ctx);
        }
    }

    /// Render a device card in the sidebar (static version that returns interaction flags)
    fn render_device_card_static(ui: &mut egui::Ui, device: &DashboardDevice, is_selected: bool) -> (bool, bool) {
        let mut should_select = false;
        let mut should_remove = false;

        let card_response = ui.group(|ui| {
            ui.horizontal(|ui| {
                // Color indicator
                let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 40.0), Sense::hover());
                ui.painter().rect_filled(rect, 2.0, device.color);

                ui.vertical(|ui| {
                    // Device name (name is a String, not Option<String>)
                    let name = device.info.name.as_str();
                    let name_text = if is_selected {
                        RichText::new(name).strong()
                    } else {
                        RichText::new(name)
                    };
                    ui.label(name_text);

                    // Connection state
                    let (state_text, state_color) = if device.is_streaming {
                        ("Streaming", Color32::from_rgb(59, 130, 246))
                    } else {
                        match &device.connection_state {
                            ConnectionState::Connected => ("Connected", Color32::GREEN),
                            ConnectionState::Connecting => ("Connecting", Color32::YELLOW),
                            ConnectionState::Error { .. } => ("Error", Color32::RED),
                            _ => ("Disconnected", Color32::GRAY),
                        }
                    };
                    ui.label(RichText::new(state_text).color(state_color).small());

                    // Connection type
                    let conn_type = format!("{:?}", device.info.connection_type);
                    ui.label(RichText::new(conn_type).small().color(Color32::GRAY));
                });
            });
        });

        // Handle click to select
        if card_response.response.clicked() {
            should_select = true;
        }

        // Context menu
        card_response.response.context_menu(|ui| {
            if ui.button("Disconnect").clicked() {
                should_remove = true;
                ui.close_menu();
            }
            if ui.button("Start Streaming").clicked() {
                // Would call device_manager.start_streaming()
                ui.close_menu();
            }
            if ui.button("Stop Streaming").clicked() {
                // Would call device_manager.stop_streaming()
                ui.close_menu();
            }
        });

        ui.add_space(4.0);

        (should_select, should_remove)
    }

    /// Render single device view
    fn render_single_view(&self, ui: &mut egui::Ui) {
        if let Some(ref selected_id) = self.selected_device {
            if let Some(device) = self.devices.get(selected_id) {
                self.render_device_visualization(ui, device, ui.available_rect_before_wrap());
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a device from the sidebar");
            });
        }
    }

    /// Render side-by-side view (first two devices)
    fn render_side_by_side_view(&self, ui: &mut egui::Ui) {
        let available = ui.available_rect_before_wrap();
        let half_width = available.width() / 2.0 - 5.0;

        ui.horizontal(|ui| {
            for (i, id) in self.device_order.iter().take(2).enumerate() {
                if let Some(device) = self.devices.get(id) {
                    let rect = Rect::from_min_size(
                        Pos2::new(
                            available.min.x + i as f32 * (half_width + 10.0),
                            available.min.y,
                        ),
                        Vec2::new(half_width, available.height()),
                    );

                    ui.allocate_ui_at_rect(rect, |ui| {
                        self.render_device_visualization(ui, device, rect);
                    });
                }
            }
        });
    }

    /// Render overlay view (all signals on same plot)
    fn render_overlay_view(&self, ui: &mut egui::Ui) {
        let available = ui.available_rect_before_wrap();

        // Draw combined EEG plot with all devices overlaid
        ui.group(|ui| {
            ui.heading("EEG Overlay (Channel 0)");

            let plot_rect = Rect::from_min_size(
                available.min,
                Vec2::new(available.width(), available.height() - 40.0),
            );

            let painter = ui.painter_at(plot_rect);

            // Background
            painter.rect_filled(plot_rect, 4.0, Color32::from_gray(20));

            // Draw each device's data
            for id in &self.device_order {
                if let Some(device) = self.devices.get(id) {
                    if !device.buffers.eeg_channels.is_empty() {
                        let channel = &device.buffers.eeg_channels[0];
                        self.draw_signal_line(&painter, plot_rect, channel, device.color);
                    }
                }
            }

            // Legend
            ui.horizontal(|ui| {
                for id in &self.device_order {
                    if let Some(device) = self.devices.get(id) {
                        let name = device.info.name.as_str();
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
                        ui.painter().rect_filled(rect, 2.0, device.color);
                        ui.label(name);
                        ui.add_space(10.0);
                    }
                }
            });
        });
    }

    /// Render grid view (all devices in grid)
    fn render_grid_view(&self, ui: &mut egui::Ui) {
        let device_count = self.devices.len();
        if device_count == 0 {
            ui.centered_and_justified(|ui| {
                ui.label("No devices connected");
            });
            return;
        }

        let available = ui.available_rect_before_wrap();

        // Calculate grid dimensions
        let cols = (device_count as f32).sqrt().ceil() as usize;
        let rows = (device_count + cols - 1) / cols;

        let cell_width = available.width() / cols as f32 - 5.0;
        let cell_height = available.height() / rows as f32 - 5.0;

        for (i, id) in self.device_order.iter().enumerate() {
            if let Some(device) = self.devices.get(id) {
                let row = i / cols;
                let col = i % cols;

                let rect = Rect::from_min_size(
                    Pos2::new(
                        available.min.x + col as f32 * (cell_width + 5.0),
                        available.min.y + row as f32 * (cell_height + 5.0),
                    ),
                    Vec2::new(cell_width, cell_height),
                );

                ui.allocate_ui_at_rect(rect, |ui| {
                    self.render_device_visualization(ui, device, rect);
                });
            }
        }
    }

    /// Render visualization for a single device
    fn render_device_visualization(&self, ui: &mut egui::Ui, device: &DashboardDevice, rect: Rect) {
        ui.group(|ui| {
            // Device header
            ui.horizontal(|ui| {
                let (color_rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 20.0), Sense::hover());
                ui.painter().rect_filled(color_rect, 2.0, device.color);

                let name = device.info.name.as_str();
                ui.heading(name);

                // Status indicator
                let (status_text, status_color) = if device.is_streaming {
                    ("LIVE", Color32::GREEN)
                } else if matches!(device.connection_state, ConnectionState::Connected) {
                    ("READY", Color32::YELLOW)
                } else {
                    ("OFFLINE", Color32::GRAY)
                };
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(status_text).color(status_color).strong());
                });
            });

            ui.separator();

            // EEG plot
            let plot_height = (rect.height() - 100.0).max(100.0) / 2.0;
            ui.group(|ui| {
                ui.label(RichText::new("EEG").small());
                let (plot_rect, _) = ui.allocate_exact_size(
                    Vec2::new(rect.width() - 20.0, plot_height),
                    Sense::hover(),
                );

                let painter = ui.painter_at(plot_rect);
                painter.rect_filled(plot_rect, 2.0, Color32::from_gray(20));

                // Draw first channel
                if !device.buffers.eeg_channels.is_empty() {
                    self.draw_signal_line(&painter, plot_rect, &device.buffers.eeg_channels[0], device.color);
                }
            });

            // fNIRS plot
            ui.group(|ui| {
                ui.label(RichText::new("fNIRS (HbO/HbR)").small());
                let (plot_rect, _) = ui.allocate_exact_size(
                    Vec2::new(rect.width() - 20.0, plot_height),
                    Sense::hover(),
                );

                let painter = ui.painter_at(plot_rect);
                painter.rect_filled(plot_rect, 2.0, Color32::from_gray(20));

                // Draw HbO (red) and HbR (blue)
                self.draw_signal_line(&painter, plot_rect, &device.buffers.fnirs_hbo, Color32::from_rgb(239, 68, 68));
                self.draw_signal_line(&painter, plot_rect, &device.buffers.fnirs_hbr, Color32::from_rgb(59, 130, 246));
            });

            // Band powers
            ui.horizontal(|ui| {
                let bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"];
                for (i, band) in bands.iter().enumerate() {
                    let power = device.buffers.band_powers[i];
                    ui.vertical(|ui| {
                        ui.label(RichText::new(*band).small());
                        let bar_height = (power * 50.0).min(50.0);
                        let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(30.0, 50.0), Sense::hover());

                        // Background
                        ui.painter().rect_filled(bar_rect, 2.0, Color32::from_gray(40));

                        // Filled portion
                        let filled_rect = Rect::from_min_size(
                            Pos2::new(bar_rect.min.x, bar_rect.max.y - bar_height),
                            Vec2::new(30.0, bar_height),
                        );
                        ui.painter().rect_filled(filled_rect, 2.0, device.color);
                    });
                }
            });
        });
    }

    /// Draw a signal line on a plot
    fn draw_signal_line(&self, painter: &egui::Painter, rect: Rect, data: &[f32], color: Color32) {
        if data.len() < 2 {
            return;
        }

        // Find data range
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(0.001);

        // Generate points
        let points: Vec<Pos2> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.min.x + (i as f32 / data.len() as f32) * rect.width();
                let y = rect.max.y - ((v - min_val) / range) * rect.height();
                Pos2::new(x, y)
            })
            .collect();

        // Draw line
        painter.add(egui::Shape::line(points, Stroke::new(1.5, color)));
    }

    /// Render the device connection panel
    fn render_connection_panel(&mut self, ctx: &egui::Context) {
        egui::Window::new("Connect Device")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.heading("Available Devices");
                ui.separator();

                ui.label("USB Devices:");
                // Would list discovered USB devices here
                ui.label(RichText::new("  Scanning...").italics().color(Color32::GRAY));

                ui.add_space(10.0);

                ui.label("Bluetooth Devices:");
                // Would list discovered BLE devices here
                ui.label(RichText::new("  Scanning...").italics().color(Color32::GRAY));

                ui.add_space(20.0);

                ui.horizontal(|ui| {
                    if ui.button("Refresh").clicked() {
                        // Trigger device scan
                    }
                    if ui.button("Close").clicked() {
                        self.show_connection_panel = false;
                    }
                });
            });
    }
}

/// Stimulation control panel for a device
pub struct StimulationPanel {
    /// Target device ID
    device_id: DeviceId,
    /// Selected protocol index
    selected_protocol: usize,
    /// Custom amplitude (uA)
    custom_amplitude: f32,
    /// Custom frequency (Hz)
    custom_frequency: f32,
    /// Custom duration (minutes)
    custom_duration: f32,
}

impl StimulationPanel {
    /// Create a new stimulation panel for a device
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            selected_protocol: 0,
            custom_amplitude: 1000.0,
            custom_frequency: 0.0,
            custom_duration: 10.0,
        }
    }

    /// Render the stimulation control UI
    pub fn ui(&mut self, ui: &mut egui::Ui, controller: &mut StimulationController) {
        ui.heading("Stimulation Control");
        ui.separator();

        // Current session status
        if let Some(session) = controller.current_session() {
            ui.group(|ui| {
                ui.label(RichText::new("Active Session").strong().color(Color32::GREEN));

                let elapsed = session.elapsed_time();
                let remaining = session.remaining_time();

                ui.label(format!("Elapsed: {:.0}s", elapsed.as_secs_f32()));
                if let Some(rem) = remaining {
                    ui.label(format!("Remaining: {:.0}s", rem.as_secs_f32()));

                    // Progress bar
                    let progress = elapsed.as_secs_f32()
                        / (elapsed.as_secs_f32() + rem.as_secs_f32());
                    ui.add(egui::ProgressBar::new(progress).text("Progress"));
                }
            });

            if ui.button("Stop Stimulation").clicked() {
                controller.stop();
            }
        } else {
            // Protocol selection
            ui.label("Protocol:");
            let protocols = ["tDCS (anodal)", "tDCS (cathodal)", "tACS (10Hz)", "Custom"];
            egui::ComboBox::from_id_salt("protocol_select")
                .selected_text(protocols[self.selected_protocol])
                .show_ui(ui, |ui| {
                    for (i, protocol) in protocols.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_protocol, i, *protocol);
                    }
                });

            // Custom parameters
            if self.selected_protocol == 3 {
                ui.add_space(10.0);
                ui.label("Custom Parameters:");

                ui.horizontal(|ui| {
                    ui.label("Amplitude (uA):");
                    ui.add(egui::Slider::new(&mut self.custom_amplitude, 100.0..=2000.0));
                });

                ui.horizontal(|ui| {
                    ui.label("Frequency (Hz):");
                    ui.add(egui::Slider::new(&mut self.custom_frequency, 0.0..=100.0));
                });

                ui.horizontal(|ui| {
                    ui.label("Duration (min):");
                    ui.add(egui::Slider::new(&mut self.custom_duration, 1.0..=30.0));
                });
            }

            ui.add_space(10.0);

            if ui.button("Start Stimulation").clicked() {
                // Would create and start a stimulation session
                tracing::info!("Starting stimulation for device {:?}", self.device_id);
            }
        }

        // Safety status
        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("Safety Status").small());
        ui.label(RichText::new("All limits within range").small().color(Color32::GREEN));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_mode_names() {
        assert_eq!(ViewMode::Single.name(), "Single");
        assert_eq!(ViewMode::SideBySide.name(), "Side-by-Side");
        assert_eq!(ViewMode::Overlay.name(), "Overlay");
        assert_eq!(ViewMode::Grid.name(), "Grid");
    }

    #[test]
    fn test_dashboard_creation() {
        let dashboard = MultiDeviceDashboard::new();
        assert!(dashboard.devices.is_empty());
        assert_eq!(dashboard.view_mode, ViewMode::Single);
    }

    #[test]
    fn test_device_buffers() {
        let mut buffers = DeviceDataBuffers::new(8, 4, 100);
        assert_eq!(buffers.eeg_channels.len(), 8);

        buffers.push_fnirs(1.0, 0.5);
        assert_eq!(buffers.fnirs_hbo.len(), 1);
        assert_eq!(buffers.fnirs_hbr.len(), 1);
    }
}

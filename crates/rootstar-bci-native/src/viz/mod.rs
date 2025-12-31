//! Native SNS Visualization Module
//!
//! Provides a native desktop application for visualizing BCI data using
//! egui + wgpu. This is an alternative to the WASM-based web visualization.
//!
//! Enable with the `viz` feature:
//! ```toml
//! rootstar-bci-native = { version = "0.1", features = ["viz"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use rootstar_bci_native::viz::{run_app, VizConfig};
//!
//! fn main() {
//!     let config = VizConfig::default();
//!     run_app(config).expect("Failed to run visualization");
//! }
//! ```
//!
//! # Multi-Device Dashboard
//!
//! ```ignore
//! use rootstar_bci_native::viz::MultiDeviceDashboard;
//!
//! let mut dashboard = MultiDeviceDashboard::new();
//! dashboard.add_device(device_id, device_info);
//! dashboard.set_view_mode(ViewMode::Grid);
//! ```

#[cfg(feature = "viz")]
mod app;

#[cfg(feature = "viz")]
mod renderer;

#[cfg(feature = "viz")]
mod mesh;

#[cfg(feature = "viz")]
mod dashboard;

#[cfg(feature = "viz")]
mod electrode_status;

#[cfg(feature = "viz")]
mod fnirs_status;

// Re-export application types
#[cfg(feature = "viz")]
pub use app::{run_app, MeshType, SnsVizApp, VizConfig};

// Re-export renderer types
#[cfg(feature = "viz")]
pub use renderer::{Camera3D, Colormap, SnsRenderer};

// Re-export mesh types
#[cfg(feature = "viz")]
pub use mesh::{
    generate_cochlea_mesh, generate_skin_mesh, generate_tongue_mesh,
    MeshData, MeshId, ReceptorPosition, ReceptorType, Vertex,
};

// Re-export dashboard types
#[cfg(feature = "viz")]
pub use dashboard::{
    DashboardDevice, DeviceDataBuffers, MultiDeviceDashboard, StimulationPanel, ViewMode,
};

// Re-export electrode status types
#[cfg(feature = "viz")]
pub use electrode_status::{
    ElectrodePosition, ElectrodeState, ElectrodeStatus, ElectrodeStatusBar, ElectrodeStatusPanel,
    ELECTRODE_POSITIONS, REFERENCE_POSITIONS,
};

// Re-export fNIRS status types
#[cfg(feature = "viz")]
pub use fnirs_status::{
    ChannelState, FnirsChannel, FnirsStatusPanel, OptodePosition, OptodeState, OptodeStatus,
    OptodeType, Wavelength, default_prefrontal_layout,
};

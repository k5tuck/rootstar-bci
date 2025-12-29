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

#[cfg(feature = "viz")]
mod app;

#[cfg(feature = "viz")]
mod renderer;

#[cfg(feature = "viz")]
mod mesh;

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

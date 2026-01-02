//! Rootstar Physics Mesh - Body Mesh Coordinate System
//!
//! Provides a standardized virtual body mesh with anatomical region mapping
//! for VR neural stimulation. Maps UV coordinates to body regions for
//! neural fingerprint lookup.
//!
//! # Body Region Map
//!
//! The mesh uses UV coordinates (0.0-1.0) to identify body parts:
//!
//! ```text
//!     ┌─────────────────────────────────────────┐
//! 1.0 │            HEAD (0.90-0.95)             │
//!     │     ┌───────────────────────────┐       │
//! 0.9 │     │     Forehead/Cheeks       │       │
//!     │     └───────────────────────────┘       │
//! 0.8 │                                         │
//!     │  ┌──────┐             ┌──────┐          │
//! 0.7 │  │ L Arm│   TORSO    │ R Arm│          │
//!     │  │      │             │      │          │
//! 0.6 │  │Upper │  (Chest)   │Upper │          │
//!     │  └──────┘             └──────┘          │
//! 0.5 │  ┌──────┐             ┌──────┐          │
//!     │  │Forearm│ (Abdomen) │Forearm│         │
//! 0.4 │  └──────┘             └──────┘          │
//!     │  ┌──────┐             ┌──────┐          │
//! 0.3 │  │ Hand │             │ Hand │          │
//!     │  └──────┘             └──────┘          │
//! 0.2 │       ┌─────┐   ┌─────┐                 │
//!     │       │L Leg│   │R Leg│                 │
//! 0.1 │       │     │   │     │                 │
//!     │       └─────┘   └─────┘                 │
//! 0.0 │       (Foot)     (Foot)                 │
//!     └─────────────────────────────────────────┘
//!    0.0                                       1.0
//! ```

#![warn(missing_docs)]

pub mod region;
pub mod vertex;

pub use region::{BodyRegion, BodyRegionId, BodyRegionMap, SensitivityLevel};
pub use vertex::{MeshVertex, VertexBuffer};

//! Rootstar Physics Core - VR Neural Mapping Physics Engine
//!
//! This crate provides core physics types and simulation for translating virtual
//! world phenomena (wind, temperature, pressure) into neural stimulation coordinates.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        VR Physics Engine                                 │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │
//! │  │ Wind Simulation │    │ Temperature     │    │ Collision Detection │  │
//! │  │                 │    │ Gradients       │    │                     │  │
//! │  └────────┬────────┘    └────────┬────────┘    └──────────┬──────────┘  │
//! │           │                      │                        │             │
//! │           └──────────────────────┼────────────────────────┘             │
//! │                                  ▼                                      │
//! │                    ┌─────────────────────────┐                          │
//! │                    │ Affected Vertices List  │                          │
//! │                    │ (per-frame output)      │                          │
//! │                    └─────────────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - `std`: Standard library support (default)
//! - `no_std`: Embedded/WASM support for VR headsets
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_physics_core::{WindVector, EnvironmentalEffect, PhysicsFrame};
//!
//! // Create a wind simulation
//! let wind = WindVector::new(5.0, 0.0, 1.0)  // 5 m/s from the east
//!     .with_turbulence(0.2)
//!     .with_temperature(22.0);
//!
//! // Get affected vertices for current frame
//! let frame = PhysicsFrame::new(60.0);  // 60 Hz
//! let affected = wind.compute_affected_vertices(&body_mesh, &frame);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod collision;
pub mod effects;
pub mod types;
pub mod wind;

pub use collision::{BoundingBox, CollisionResult, Ray};
pub use effects::{EnvironmentalEffect, TemperatureField};
pub use types::{FrameTime, Intensity, PhysicsFrame, Velocity};
pub use wind::{Turbulence, WindVector};

/// Physics engine version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum frame rate for smooth sensation (Hz).
pub const MIN_FRAME_RATE: f32 = 60.0;

/// Maximum wind velocity in m/s.
pub const MAX_WIND_VELOCITY: f32 = 50.0;

/// Temperature range (Celsius).
pub const TEMP_RANGE: (f32, f32) = (-20.0, 50.0);

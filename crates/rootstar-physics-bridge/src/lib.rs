//! Rootstar Physics Bridge - VR to Neural Stimulation
//!
//! This crate bridges the VR physics simulation to the BCI stimulation system.
//! It implements the **Spatial Translation Layer** from the VR Neural Mapping spec:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Physics Bridge Pipeline                          │
//! │                                                                         │
//! │  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
//! │  │ Physics Core │    │ Spatial Translator │    │ Sensation Limiter  │   │
//! │  │              │───▶│                    │───▶│                    │   │
//! │  │ WindVector   │    │ UV → BodyRegion   │    │ Global/Region/Type │   │
//! │  │ Temperature  │    │ Intensity calc    │    │ limits + ramp      │   │
//! │  │ Collision    │    │                    │    │                    │   │
//! │  └──────────────┘    └───────────────────┘    └────────────────────┘   │
//! │                                                         │              │
//! │                                                         ▼              │
//! │                                              ┌────────────────────┐    │
//! │                                              │ Fingerprint DB +   │    │
//! │                                              │ Stimulation Ctrl   │    │
//! │                                              └────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Sensation Limiting
//!
//! The `SensationLimiter` provides multiple layers of intensity control:
//!
//! - **Global limit**: Caps all sensations (e.g., 50% max)
//! - **Per-region limits**: Different limits for face, hands, etc.
//! - **Per-sensation limits**: Different limits for cold, wind, etc.
//! - **Ramp-up/down**: Gradual intensity changes when starting/stopping
//! - **Rate limiting**: Prevents sudden intensity spikes
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_physics_bridge::{SpatialTranslator, SensationLimiter, LimiterPreset};
//! use rootstar_physics_core::WindVector;
//! use rootstar_physics_mesh::BodyRegionMap;
//!
//! // Create translator and limiter
//! let mut translator = SpatialTranslator::default();
//! let mut limiter = SensationLimiter::new(LimiterPreset::Conservative);
//!
//! // Start session with ramp-up
//! limiter.start_session();
//!
//! // Process physics frame
//! let commands = translator.process_frame(&frame);
//!
//! // Apply intensity limits
//! let limited_commands = limiter.apply(&commands);
//!
//! // Execute stimulations (with hardware safety limits in BCI core)
//! for cmd in limited_commands {
//!     controller.trigger(cmd);
//! }
//! ```

#![warn(missing_docs)]

pub mod intensity;
pub mod limiter;
pub mod translator;

pub use intensity::IntensityCalculator;
pub use limiter::{LimiterConfig, LimiterPreset, SensationLimiter};
pub use translator::{SensationType, SpatialTranslator, StimulationCommand};

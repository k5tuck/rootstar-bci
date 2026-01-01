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
//! │  │ Physics Core │    │ Spatial Translator │    │ Fingerprint DB     │   │
//! │  │              │───▶│                    │───▶│                    │   │
//! │  │ WindVector   │    │ UV → BodyRegion   │    │ Region → Pattern   │   │
//! │  │ Temperature  │    │ Intensity calc    │    │                    │   │
//! │  │ Collision    │    │                    │    │                    │   │
//! │  └──────────────┘    └───────────────────┘    └────────────────────┘   │
//! │                                                         │              │
//! │                                                         ▼              │
//! │                                              ┌────────────────────┐    │
//! │                                              │ Stimulation Ctrl   │    │
//! │                                              │                    │    │
//! │                                              │ Execute patterns   │    │
//! │                                              └────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_physics_bridge::{SpatialTranslator, FramePipeline};
//! use rootstar_physics_core::WindVector;
//! use rootstar_physics_mesh::BodyRegionMap;
//!
//! // Create translator with body region map
//! let translator = SpatialTranslator::new(BodyRegionMap::standard());
//!
//! // Process wind hitting the body
//! let wind = WindVector::gentle_breeze(Velocity::new(1.0, 0.0, 0.0));
//! let stimulations = translator.translate_wind(&wind, &affected_vertices);
//!
//! // Execute stimulations
//! for stim in stimulations {
//!     controller.trigger(stim.region_id, stim.intensity, stim.sensation_type);
//! }
//! ```

#![warn(missing_docs)]

pub mod intensity;
pub mod translator;

pub use intensity::IntensityCalculator;
pub use translator::{SensationType, SpatialTranslator, StimulationCommand};

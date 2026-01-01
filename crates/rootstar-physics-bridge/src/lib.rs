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
//! # Game Engine Integration
//!
//! The `engine` module provides a trait-based API for game engines:
//!
//! ```rust,ignore
//! use rootstar_physics_bridge::engine::{SensoryBridge, SensoryEvent, WindEvent};
//!
//! // Create bridge
//! let mut bridge = SensoryBridge::new();
//!
//! // Send event from game engine
//! let stimuli = bridge.process_event(SensoryEvent::Wind(WindEvent {
//!     velocity: [10.0, 0.0, 0.0],
//!     body_regions: vec![BodyRegionId::ArmLeftUpper],
//!     ..Default::default()
//! }));
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

#![warn(missing_docs)]

pub mod engine;
pub mod intensity;
pub mod limiter;
pub mod translator;

pub use engine::{
    GameEngineAdapter, HapticEvent, ProcessedStimulus, SensoryBridge, SensoryEvent,
    SensoryModality, SoundEvent, TasteEvent, VisualEvent, WindEvent,
};
pub use intensity::IntensityCalculator;
pub use limiter::{LimiterConfig, LimiterPreset, SensationLimiter};
pub use translator::{SensationType, SpatialTranslator, StimulationCommand};

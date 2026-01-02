//! Game Engine Integration API
//!
//! This module provides traits and types for integrating the VR neural mapping
//! system with game engines (Unity, Unreal, Godot, Bevy, etc.).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Game Engine (Unity/Unreal/Godot/Bevy)            │
//! │                                                                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
//! │  │ Wind Zone   │  │ Audio Zone  │  │ Taste Item  │  │ Visual FX   │     │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
//! └─────────┼────────────────┼────────────────┼────────────────┼────────────┘
//!           │                │                │                │
//!           ▼                ▼                ▼                ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         GameEngineAdapter (implement this)               │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │ SensoryEvent: Wind | Sound | Taste | Visual | Haptic            │    │
//! │  └─────────────────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         SensoryBridge                                    │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
//! │  │ WindEngine  │  │ AudioEngine │  │ TasteEngine │  │ VisualEngine│     │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Neural Stimulation                               │
//! │                    (Fingerprint DB → BCI Hardware)                       │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: Unity Integration
//!
//! ```rust,ignore
//! // In Unity C# (via FFI):
//! //
//! // [DllImport("rootstar_physics")]
//! // static extern void send_sensory_event(SensoryEvent event);
//! //
//! // void OnTriggerEnter(Collider other) {
//! //     if (other.CompareTag("WindZone")) {
//! //         var wind = other.GetComponent<WindZone>();
//! //         send_sensory_event(new SensoryEvent {
//! //             type = EventType.Wind,
//! //             wind = new WindEvent {
//! //                 velocity = wind.velocity,
//! //                 temperature = wind.temperature
//! //             }
//! //         });
//! //     }
//! // }
//!
//! // In Rust:
//! use rootstar_physics_bridge::engine::{SensoryBridge, SensoryEvent};
//!
//! let mut bridge = SensoryBridge::new();
//!
//! // Receive events from game engine
//! bridge.process_event(SensoryEvent::Wind {
//!     velocity: [5.0, 0.0, 0.0],
//!     temperature_c: 18.0,
//!     body_regions: vec![BodyRegionId::ArmLeftUpper],
//! });
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

use rootstar_physics_core::types::Intensity;
use rootstar_physics_mesh::BodyRegionId;

// ============================================================================
// Sensory Event Types
// ============================================================================

/// Event from game engine representing a sensory stimulus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SensoryEvent {
    /// Wind/air movement effect.
    Wind(WindEvent),
    /// Temperature change.
    Temperature(TemperatureEvent),
    /// Sound/audio effect.
    Sound(SoundEvent),
    /// Taste/gustatory effect.
    Taste(TasteEvent),
    /// Visual effect (phosphenes, etc.).
    Visual(VisualEvent),
    /// Olfactory/smell effect.
    Olfactory(OlfactoryEvent),
    /// Direct haptic/touch effect.
    Haptic(HapticEvent),
    /// Combined multi-sensory event.
    MultiSensory(Vec<SensoryEvent>),
}

/// Wind/air movement event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindEvent {
    /// Velocity vector [x, y, z] in m/s.
    pub velocity: [f32; 3],
    /// Air temperature in Celsius.
    pub temperature_c: f32,
    /// Turbulence level (0.0 - 1.0).
    pub turbulence: f32,
    /// Affected body regions.
    pub body_regions: Vec<BodyRegionId>,
    /// Duration (None = continuous until stopped).
    pub duration: Option<Duration>,
}

impl Default for WindEvent {
    fn default() -> Self {
        Self {
            velocity: [0.0, 0.0, 0.0],
            temperature_c: 20.0,
            turbulence: 0.0,
            body_regions: Vec::new(),
            duration: None,
        }
    }
}

/// Temperature change event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemperatureEvent {
    /// Temperature in Celsius.
    pub temperature_c: f32,
    /// Rate of temperature change (°C/s).
    pub rate: f32,
    /// Affected body regions.
    pub body_regions: Vec<BodyRegionId>,
    /// Duration.
    pub duration: Option<Duration>,
}

/// Sound/audio event for auditory neural stimulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoundEvent {
    /// Frequency in Hz.
    pub frequency_hz: f32,
    /// Amplitude in dB SPL.
    pub amplitude_db: f32,
    /// Left/right balance (-1.0 = left, 0.0 = center, 1.0 = right).
    pub pan: f32,
    /// Waveform type.
    pub waveform: Waveform,
    /// Duration.
    pub duration: Option<Duration>,
    /// Whether this is spatial audio (3D positioned).
    pub spatial: bool,
    /// 3D position if spatial [x, y, z].
    pub position: Option<[f32; 3]>,
}

impl Default for SoundEvent {
    fn default() -> Self {
        Self {
            frequency_hz: 440.0,
            amplitude_db: 60.0,
            pan: 0.0,
            waveform: Waveform::Sine,
            duration: None,
            spatial: false,
            position: None,
        }
    }
}

/// Waveform type for audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Waveform {
    /// Pure sine wave.
    Sine,
    /// Square wave.
    Square,
    /// Sawtooth wave.
    Sawtooth,
    /// Triangle wave.
    Triangle,
    /// White noise.
    Noise,
    /// Complex/harmonic (music, speech).
    Complex,
}

/// Taste/gustatory event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TasteEvent {
    /// Taste quality concentrations [sweet, salty, sour, bitter, umami] (0.0-1.0).
    pub concentrations: [f32; 5],
    /// Tongue region.
    pub region: TongueRegion,
    /// Duration.
    pub duration: Option<Duration>,
    /// Temperature (affects perception).
    pub temperature_c: f32,
}

impl Default for TasteEvent {
    fn default() -> Self {
        Self {
            concentrations: [0.0; 5],
            region: TongueRegion::Anterior,
            duration: None,
            temperature_c: 37.0,
        }
    }
}

/// Tongue region for taste events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TongueRegion {
    /// Tip of tongue.
    TipAnterior,
    /// Front/middle.
    Anterior,
    /// Sides.
    Lateral,
    /// Back.
    Posterior,
    /// Roof of mouth.
    SoftPalate,
}

/// Visual event for visual cortex stimulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualEvent {
    /// Type of visual effect.
    pub effect_type: VisualEffectType,
    /// Intensity (0.0 - 1.0).
    pub intensity: f32,
    /// Position in visual field [x, y] (-1.0 to 1.0).
    pub position: [f32; 2],
    /// Size/radius in visual field (0.0 - 1.0).
    pub size: f32,
    /// Color [r, g, b] (0.0 - 1.0).
    pub color: [f32; 3],
    /// Duration.
    pub duration: Option<Duration>,
}

impl Default for VisualEvent {
    fn default() -> Self {
        Self {
            effect_type: VisualEffectType::Phosphene,
            intensity: 0.5,
            position: [0.0, 0.0],
            size: 0.1,
            color: [1.0, 1.0, 1.0],
            duration: None,
        }
    }
}

/// Type of visual effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualEffectType {
    /// Simple phosphene (spot of light).
    Phosphene,
    /// Edge/contour perception.
    Edge,
    /// Motion perception.
    Motion,
    /// Flash/sudden brightness.
    Flash,
    /// Color perception.
    Color,
    /// Pattern (basic shapes).
    Pattern,
}

/// Direct haptic/touch event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HapticEvent {
    /// Body region.
    pub region: BodyRegionId,
    /// Pressure intensity (0.0 - 1.0).
    pub pressure: f32,
    /// Vibration frequency (Hz), None = no vibration.
    pub vibration_hz: Option<f32>,
    /// Texture roughness (0.0 = smooth, 1.0 = rough).
    pub texture: f32,
    /// Duration.
    pub duration: Option<Duration>,
}

/// Olfactory/smell event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OlfactoryEvent {
    /// Odorant class/type.
    pub odorant_class: OdorantType,
    /// Concentration/intensity (0.0 - 1.0).
    pub concentration: f32,
    /// Whether the smell is pleasant (affects processing).
    pub pleasant: bool,
    /// Duration.
    pub duration: Option<Duration>,
    /// 3D position if spatial (for source localization).
    pub position: Option<[f32; 3]>,
}

impl Default for OlfactoryEvent {
    fn default() -> Self {
        Self {
            odorant_class: OdorantType::Floral,
            concentration: 0.5,
            pleasant: true,
            duration: None,
            position: None,
        }
    }
}

/// Odorant classification for game engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdorantType {
    /// Floral scents (rose, jasmine, lavender).
    Floral,
    /// Fruity scents (citrus, berry, apple).
    Fruity,
    /// Woody scents (cedar, pine, sandalwood).
    Woody,
    /// Minty/herbal scents (mint, eucalyptus).
    Minty,
    /// Sweet scents (vanilla, caramel, honey).
    Sweet,
    /// Pungent/sharp scents (ammonia, vinegar).
    Pungent,
    /// Decay/unpleasant scents (smoke, sulfur).
    Decay,
    /// Musky/earthy scents (soil, moss).
    Musky,
    /// Food/cooking scents (bread, coffee).
    Food,
    /// Clean/soapy scents (detergent, fresh linen).
    Clean,
}

// ============================================================================
// Game Engine Adapter Trait
// ============================================================================

/// Trait for game engine integration.
///
/// Implement this trait to connect your game engine to the neural stimulation system.
pub trait GameEngineAdapter {
    /// Get the current frame time delta.
    fn delta_time(&self) -> Duration;

    /// Get the player's body position in world space.
    fn player_position(&self) -> [f32; 3];

    /// Get the player's body rotation (quaternion).
    fn player_rotation(&self) -> [f32; 4];

    /// Query active wind zones affecting the player.
    fn query_wind_zones(&self) -> Vec<WindEvent>;

    /// Query active sound sources.
    fn query_sound_sources(&self) -> Vec<SoundEvent>;

    /// Query active taste sources (food items, etc.).
    fn query_taste_sources(&self) -> Vec<TasteEvent>;

    /// Query active visual effects.
    fn query_visual_effects(&self) -> Vec<VisualEvent>;

    /// Query active olfactory sources.
    fn query_olfactory_sources(&self) -> Vec<OlfactoryEvent>;

    /// Query active haptic contacts.
    fn query_haptic_contacts(&self) -> Vec<HapticEvent>;

    /// Get collision points on player body mesh.
    fn get_body_collisions(&self) -> Vec<BodyCollision>;

    /// Log a message to the game engine console.
    fn log(&self, message: &str);
}

/// Body collision information from game engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyCollision {
    /// Body region hit.
    pub region: BodyRegionId,
    /// UV coordinates on body mesh.
    pub uv: [f32; 2],
    /// Contact normal.
    pub normal: [f32; 3],
    /// Impact velocity.
    pub velocity: f32,
    /// Material type of colliding object.
    pub material: MaterialType,
}

/// Material types for haptic feedback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialType {
    /// Soft material (skin, fabric).
    Soft,
    /// Hard material (wood, plastic).
    Hard,
    /// Metal.
    Metal,
    /// Water/liquid.
    Liquid,
    /// Sand/granular.
    Granular,
    /// Sharp edge.
    Sharp,
    /// Hot surface.
    Hot,
    /// Cold surface.
    Cold,
}

// ============================================================================
// Sensory Bridge (Main Integration Point)
// ============================================================================

/// Main bridge between game engine and neural stimulation.
///
/// This is the primary integration point for game engines.
pub struct SensoryBridge {
    /// Wind processing enabled.
    pub wind_enabled: bool,
    /// Sound processing enabled.
    pub sound_enabled: bool,
    /// Taste processing enabled.
    pub taste_enabled: bool,
    /// Visual processing enabled.
    pub visual_enabled: bool,
    /// Olfactory processing enabled.
    pub olfactory_enabled: bool,
    /// Haptic processing enabled.
    pub haptic_enabled: bool,
    /// Global intensity scale.
    pub intensity_scale: f32,
    /// Event queue.
    event_queue: Vec<SensoryEvent>,
}

impl SensoryBridge {
    /// Create a new sensory bridge with all modalities enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            wind_enabled: true,
            sound_enabled: true,
            taste_enabled: true,
            visual_enabled: true,
            olfactory_enabled: true,
            haptic_enabled: true,
            intensity_scale: 1.0,
            event_queue: Vec::new(),
        }
    }

    /// Create with only specific modalities enabled.
    #[must_use]
    pub fn with_modalities(wind: bool, sound: bool, taste: bool, visual: bool, olfactory: bool, haptic: bool) -> Self {
        Self {
            wind_enabled: wind,
            sound_enabled: sound,
            taste_enabled: taste,
            visual_enabled: visual,
            olfactory_enabled: olfactory,
            haptic_enabled: haptic,
            intensity_scale: 1.0,
            event_queue: Vec::new(),
        }
    }

    /// Set global intensity scale.
    pub fn set_intensity_scale(&mut self, scale: f32) {
        self.intensity_scale = scale.clamp(0.0, 2.0);
    }

    /// Queue a sensory event for processing.
    pub fn queue_event(&mut self, event: SensoryEvent) {
        self.event_queue.push(event);
    }

    /// Process a single sensory event immediately.
    pub fn process_event(&mut self, event: SensoryEvent) -> Vec<ProcessedStimulus> {
        let mut stimuli = Vec::new();

        match event {
            SensoryEvent::Wind(wind) if self.wind_enabled => {
                stimuli.extend(self.process_wind(&wind));
            }
            SensoryEvent::Temperature(temp) if self.wind_enabled => {
                stimuli.extend(self.process_temperature(&temp));
            }
            SensoryEvent::Sound(sound) if self.sound_enabled => {
                stimuli.extend(self.process_sound(&sound));
            }
            SensoryEvent::Taste(taste) if self.taste_enabled => {
                stimuli.extend(self.process_taste(&taste));
            }
            SensoryEvent::Visual(visual) if self.visual_enabled => {
                stimuli.extend(self.process_visual(&visual));
            }
            SensoryEvent::Olfactory(olfactory) if self.olfactory_enabled => {
                stimuli.extend(self.process_olfactory(&olfactory));
            }
            SensoryEvent::Haptic(haptic) if self.haptic_enabled => {
                stimuli.extend(self.process_haptic(&haptic));
            }
            SensoryEvent::MultiSensory(events) => {
                for e in events {
                    stimuli.extend(self.process_event(e));
                }
            }
            _ => {} // Disabled modality
        }

        stimuli
    }

    /// Process all queued events.
    pub fn process_queue(&mut self) -> Vec<ProcessedStimulus> {
        let events: Vec<_> = self.event_queue.drain(..).collect();
        let mut stimuli = Vec::new();

        for event in events {
            stimuli.extend(self.process_event(event));
        }

        stimuli
    }

    /// Poll a game engine adapter and process events.
    pub fn poll_adapter<A: GameEngineAdapter>(&mut self, adapter: &A) -> Vec<ProcessedStimulus> {
        let mut stimuli = Vec::new();

        // Wind
        if self.wind_enabled {
            for wind in adapter.query_wind_zones() {
                stimuli.extend(self.process_wind(&wind));
            }
        }

        // Sound
        if self.sound_enabled {
            for sound in adapter.query_sound_sources() {
                stimuli.extend(self.process_sound(&sound));
            }
        }

        // Taste
        if self.taste_enabled {
            for taste in adapter.query_taste_sources() {
                stimuli.extend(self.process_taste(&taste));
            }
        }

        // Visual
        if self.visual_enabled {
            for visual in adapter.query_visual_effects() {
                stimuli.extend(self.process_visual(&visual));
            }
        }

        // Olfactory
        if self.olfactory_enabled {
            for olfactory in adapter.query_olfactory_sources() {
                stimuli.extend(self.process_olfactory(&olfactory));
            }
        }

        // Haptic from collisions
        if self.haptic_enabled {
            for collision in adapter.get_body_collisions() {
                let haptic = HapticEvent {
                    region: collision.region,
                    pressure: (collision.velocity / 10.0).min(1.0),
                    vibration_hz: None,
                    texture: match collision.material {
                        MaterialType::Soft => 0.1,
                        MaterialType::Hard => 0.5,
                        MaterialType::Metal => 0.7,
                        MaterialType::Granular => 0.9,
                        _ => 0.3,
                    },
                    duration: Some(Duration::from_millis(100)),
                };
                stimuli.extend(self.process_haptic(&haptic));
            }

            for haptic in adapter.query_haptic_contacts() {
                stimuli.extend(self.process_haptic(&haptic));
            }
        }

        stimuli
    }

    fn process_wind(&self, wind: &WindEvent) -> Vec<ProcessedStimulus> {
        let speed = (wind.velocity[0].powi(2) + wind.velocity[1].powi(2) + wind.velocity[2].powi(2)).sqrt();
        let intensity = Intensity::new((speed / 20.0).min(1.0) * self.intensity_scale);

        wind.body_regions
            .iter()
            .map(|&region| ProcessedStimulus {
                modality: SensoryModality::Tactile,
                region: Some(region),
                intensity,
                parameters: StimParameters::Wind {
                    temperature_c: wind.temperature_c,
                    turbulence: wind.turbulence,
                },
            })
            .collect()
    }

    fn process_sound(&self, sound: &SoundEvent) -> Vec<ProcessedStimulus> {
        // Convert dB to normalized intensity (60 dB = 0.5, 90 dB = 1.0)
        let intensity = Intensity::new(((sound.amplitude_db - 30.0) / 60.0).clamp(0.0, 1.0) * self.intensity_scale);

        vec![ProcessedStimulus {
            modality: SensoryModality::Auditory,
            region: None,
            intensity,
            parameters: StimParameters::Sound {
                frequency_hz: sound.frequency_hz,
                pan: sound.pan,
            },
        }]
    }

    fn process_taste(&self, taste: &TasteEvent) -> Vec<ProcessedStimulus> {
        let total: f32 = taste.concentrations.iter().sum();
        let intensity = Intensity::new((total / 5.0).min(1.0) * self.intensity_scale);

        vec![ProcessedStimulus {
            modality: SensoryModality::Gustatory,
            region: None,
            intensity,
            parameters: StimParameters::Taste {
                concentrations: taste.concentrations,
            },
        }]
    }

    fn process_visual(&self, visual: &VisualEvent) -> Vec<ProcessedStimulus> {
        vec![ProcessedStimulus {
            modality: SensoryModality::Visual,
            region: None,
            intensity: Intensity::new(visual.intensity * self.intensity_scale),
            parameters: StimParameters::Visual {
                position: visual.position,
                color: visual.color,
            },
        }]
    }

    fn process_haptic(&self, haptic: &HapticEvent) -> Vec<ProcessedStimulus> {
        vec![ProcessedStimulus {
            modality: SensoryModality::Tactile,
            region: Some(haptic.region),
            intensity: Intensity::new(haptic.pressure * self.intensity_scale),
            parameters: StimParameters::Haptic {
                vibration_hz: haptic.vibration_hz,
                texture: haptic.texture,
            },
        }]
    }

    fn process_temperature(&self, temp: &TemperatureEvent) -> Vec<ProcessedStimulus> {
        // Convert temperature deviation from neutral (32°C skin temp) to intensity
        let deviation = (temp.temperature_c - 32.0).abs();
        let intensity = Intensity::new((deviation / 20.0).min(1.0) * self.intensity_scale);

        temp.body_regions
            .iter()
            .map(|&region| ProcessedStimulus {
                modality: SensoryModality::Tactile, // Thermoreceptive maps to tactile pathway
                region: Some(region),
                intensity,
                parameters: StimParameters::Temperature {
                    temperature_c: temp.temperature_c,
                    rate: temp.rate,
                },
            })
            .collect()
    }

    fn process_olfactory(&self, olfactory: &OlfactoryEvent) -> Vec<ProcessedStimulus> {
        vec![ProcessedStimulus {
            modality: SensoryModality::Olfactory,
            region: None,
            intensity: Intensity::new(olfactory.concentration * self.intensity_scale),
            parameters: StimParameters::Olfactory {
                odorant_type: olfactory.odorant_class,
                pleasant: olfactory.pleasant,
            },
        }]
    }
}

impl Default for SensoryBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Processed Stimulus Output
// ============================================================================

/// Processed stimulus ready for neural encoding.
#[derive(Clone, Debug)]
pub struct ProcessedStimulus {
    /// Sensory modality.
    pub modality: SensoryModality,
    /// Body region (if applicable).
    pub region: Option<BodyRegionId>,
    /// Stimulus intensity.
    pub intensity: Intensity,
    /// Modality-specific parameters.
    pub parameters: StimParameters,
}

/// Sensory modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensoryModality {
    /// Touch, pressure, vibration.
    Tactile,
    /// Sound, music, speech.
    Auditory,
    /// Taste.
    Gustatory,
    /// Sight (limited - phosphenes, basic patterns).
    Visual,
    /// Smell (future).
    Olfactory,
}

/// Modality-specific stimulus parameters.
#[derive(Clone, Debug)]
pub enum StimParameters {
    /// Wind/tactile parameters.
    Wind { temperature_c: f32, turbulence: f32 },
    /// Temperature parameters.
    Temperature { temperature_c: f32, rate: f32 },
    /// Sound parameters.
    Sound { frequency_hz: f32, pan: f32 },
    /// Taste parameters.
    Taste { concentrations: [f32; 5] },
    /// Visual parameters.
    Visual { position: [f32; 2], color: [f32; 3] },
    /// Olfactory parameters.
    Olfactory { odorant_type: OdorantType, pleasant: bool },
    /// Haptic parameters.
    Haptic { vibration_hz: Option<f32>, texture: f32 },
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_event() {
        let mut bridge = SensoryBridge::new();

        let event = SensoryEvent::Wind(WindEvent {
            velocity: [10.0, 0.0, 0.0],
            temperature_c: 15.0,
            turbulence: 0.2,
            body_regions: vec![BodyRegionId::ArmLeftUpper, BodyRegionId::ArmLeftForearm],
            duration: None,
        });

        let stimuli = bridge.process_event(event);
        assert_eq!(stimuli.len(), 2);
        assert!(stimuli[0].intensity.value() > 0.0);
    }

    #[test]
    fn test_taste_event() {
        let mut bridge = SensoryBridge::new();

        let event = SensoryEvent::Taste(TasteEvent {
            concentrations: [0.8, 0.0, 0.2, 0.0, 0.3], // Sweet-sour
            region: TongueRegion::TipAnterior,
            duration: Some(Duration::from_secs(2)),
            temperature_c: 25.0,
        });

        let stimuli = bridge.process_event(event);
        assert_eq!(stimuli.len(), 1);
        assert_eq!(stimuli[0].modality, SensoryModality::Gustatory);
    }

    #[test]
    fn test_disabled_modality() {
        let mut bridge = SensoryBridge::with_modalities(true, false, false, false, false, false);

        let sound_event = SensoryEvent::Sound(SoundEvent::default());
        let stimuli = bridge.process_event(sound_event);
        assert!(stimuli.is_empty()); // Sound disabled

        let wind_event = SensoryEvent::Wind(WindEvent {
            velocity: [5.0, 0.0, 0.0],
            body_regions: vec![BodyRegionId::TorsoChest],
            ..Default::default()
        });
        let stimuli = bridge.process_event(wind_event);
        assert_eq!(stimuli.len(), 1); // Wind enabled
    }

    #[test]
    fn test_intensity_scale() {
        let mut bridge = SensoryBridge::new();
        bridge.set_intensity_scale(0.5);

        let event = SensoryEvent::Wind(WindEvent {
            velocity: [20.0, 0.0, 0.0], // Would be 1.0 intensity
            body_regions: vec![BodyRegionId::TorsoChest],
            ..Default::default()
        });

        let stimuli = bridge.process_event(event);
        assert!(stimuli[0].intensity.value() <= 0.55); // Scaled down
    }
}

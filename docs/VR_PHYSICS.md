# VR Physics Engine for Neural Mapping

This document describes the VR physics system that translates virtual world phenomena into neural stimulation patterns.

## Overview

The VR physics engine enables users to **feel** virtual experiences (wind, temperature, touch) through direct neural stimulation via the BEAST BCI platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VR Neural Mapping System                                 │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    rootstar-physics-core                              │   │
│  │  ┌─────────────┐  ┌───────────────┐  ┌────────────────┐              │   │
│  │  │ WindVector  │  │ Temperature   │  │ Collision      │              │   │
│  │  │ Turbulence  │  │ Field         │  │ Detection      │              │   │
│  │  └─────────────┘  └───────────────┘  └────────────────┘              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    rootstar-physics-mesh                              │   │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌────────────────────┐     │   │
│  │  │ BodyRegionMap   │  │ UV Coordinates │  │ Sensitivity Levels │     │   │
│  │  │ (19 regions)    │  │ Mapping        │  │                    │     │   │
│  │  └─────────────────┘  └────────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   rootstar-physics-bridge                             │   │
│  │  ┌─────────────────────┐  ┌───────────────────────────────────────┐  │   │
│  │  │ SpatialTranslator   │  │ IntensityCalculator                   │  │   │
│  │  │ - UV → BodyRegion   │  │ - Wind intensity formula              │  │   │
│  │  │ - Temporal smooth   │  │ - Temperature mapping                 │  │   │
│  │  │ - Priority queue    │  │ - Temporal smoothing                  │  │   │
│  │  └─────────────────────┘  └───────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │             rootstar-bci-native (existing)                            │   │
│  │  ┌─────────────────────┐  ┌───────────────────────────────────────┐  │   │
│  │  │ FingerprintDatabase │  │ StimulationController                 │  │   │
│  │  │ - Neural patterns   │  │ - Safety limits                       │  │   │
│  │  │ - Similarity search │  │ - Session management                  │  │   │
│  │  └─────────────────────┘  └───────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Crate Descriptions

### `rootstar-physics-core`

Core physics types for VR environmental simulation:

- **WindVector**: 3D wind simulation with velocity, turbulence, temperature
- **TemperatureField**: Thermal gradients and body temperature deviation
- **BoundingBox/Ray**: Collision detection primitives
- **PhysicsFrame**: Per-frame affected vertices with intensities

**Features**:
- `std` (default): Standard library support
- `no_std`: Embedded/WASM support for VR headsets

### `rootstar-physics-mesh`

Body mesh coordinate system:

- **BodyRegionMap**: 19 anatomical regions with UV bounds
- **BodyRegionId**: Identifiers matching database schema (e.g., `ARM_L_03`)
- **SensitivityLevel**: Per-region sensitivity multipliers
- **VertexBuffer**: Mesh vertex storage

### `rootstar-physics-bridge`

Connects physics to BCI stimulation:

- **SpatialTranslator**: UV → BodyRegion mapping with temporal smoothing
- **IntensityCalculator**: Wind/temperature intensity formulas
- **StimulationCommand**: Commands for fingerprint database lookup

## Body Region Map

| Region ID | Body Part | UV Range | Sensitivity |
|-----------|-----------|----------|-------------|
| HEAD_01 | Forehead | (0.45-0.55, 0.90-0.95) | High (2.5x) |
| HEAD_02 | Left Cheek | (0.40-0.45, 0.85-0.90) | High |
| HEAD_03 | Right Cheek | (0.55-0.60, 0.85-0.90) | High |
| ARM_L_01 | Left Upper Arm | (0.15-0.25, 0.60-0.75) | Standard (1.0x) |
| ARM_L_02 | Left Forearm | (0.10-0.20, 0.45-0.60) | Medium (1.2x) |
| ARM_L_03 | Left Hand | (0.05-0.15, 0.35-0.45) | High (2.5x) |
| TORSO_01 | Chest | (0.35-0.65, 0.55-0.75) | Standard |
| TORSO_02 | Abdomen | (0.35-0.65, 0.40-0.55) | Standard |
| LEG_L_01 | Left Thigh | (0.30-0.45, 0.20-0.40) | Low (0.8x) |
| ... | ... | ... | ... |

## Intensity Calculation

The intensity formula from the VR Neural Mapping specification:

```
intensity = (affected_vertices / total_region_vertices) *
            (wind_velocity / max_velocity) *
            cos(angle_of_incidence) *
            temporal_smoothing_factor
```

## Example: Wind on Left Arm

```rust
use rootstar_physics_core::{WindVector, PhysicsFrame};
use rootstar_physics_mesh::BodyRegionMap;
use rootstar_physics_bridge::SpatialTranslator;

// 1. Create wind blowing from the east at 5 m/s
let wind = WindVector::new(5.0, 0.0, 0.0)
    .with_turbulence(Turbulence::new(0.2))
    .with_temperature(22.0);

// 2. Compute affected vertices (from VR engine collision)
let mut frame = PhysicsFrame::new(60.0);
for vertex in collision_results {
    frame.add_vertex(AffectedVertex::new(
        vertex.index,
        vertex.uv_x,
        vertex.uv_y,
        wind.intensity_at_surface(&vertex.normal),
        EffectType::Wind,
    ));
}

// 3. Translate to stimulation commands
let mut translator = SpatialTranslator::new(BodyRegionMap::standard());
let commands = translator.process_frame(&frame);

// 4. Execute stimulation
for cmd in commands {
    // cmd.region_str = "ARM_L_01" or "ARM_L_02"
    // cmd.sensation_type = SensationType::WindBreeze
    // cmd.adjusted_intensity = 0.26 (with sensitivity multiplier)

    db.lookup_fingerprint(cmd.region_str, cmd.sensation_type.as_str());
    controller.trigger(fingerprint, cmd.adjusted_intensity);
}
```

## Relationship to FutureEndeavors

**FutureEndeavors** is a separate repository containing educational physics simulations:

| FutureEndeavors | rootstar-physics-* |
|-----------------|---------------------|
| N-body gravity (celestial) | VR haptics (body sensations) |
| Black hole geodesics | Wind/temperature gradients |
| Molecular dynamics | Body mesh collision |
| Visualization output | Neural stimulation output |

These are **different types of physics** with different purposes:
- FutureEndeavors: Visualize physics concepts
- rootstar-physics: Translate VR to neural patterns

## Building

```bash
# Build all physics crates
cargo build -p rootstar-physics-core
cargo build -p rootstar-physics-mesh
cargo build -p rootstar-physics-bridge

# Run tests
cargo test -p rootstar-physics-core
cargo test -p rootstar-physics-mesh
cargo test -p rootstar-physics-bridge
```

## Future Extensions

1. **Olfactory/Gustatory**: Apply coordinate mapping to chemical senses
2. **Full Body Haptic Suit**: Expand electrode coverage
3. **Personalized Calibration**: ML-optimized patterns per user
4. **Multi-User VR**: Synchronized sensations in shared spaces

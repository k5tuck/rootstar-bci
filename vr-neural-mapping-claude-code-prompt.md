# Claude Code Prompt: VR Neural Mapping System for BEAST Platform

## Project Overview

Design and implement a **VR Neural Mapping System** that enables direct neurostimulation-based sensory experiences in virtual reality environments. This system will translate virtual world physics (such as wind, temperature, and touch) into neural fingerprint patterns that can be delivered via the BEAST (Bidirectional Electromagnetic Adaptive Sensory Technology) BCI platform.

The goal is to allow users to **feel** virtual experiences—such as wind blowing on their skin—through direct neural stimulation rather than physical peripherals.

---

## System Architecture

### Core Components

1. **Virtual World Physics Engine** - Real-time simulation of environmental effects
2. **Body Mesh Coordinate System** - Virtual body representation with precise coordinate mapping
3. **Spatial Translation Layer** - Maps virtual coordinates to body part regions
4. **Neural Fingerprint Database** - SQLite database storing neural patterns for each body region
5. **Stimulation Controller** - Retrieves and triggers neural patterns in real-time

---

## Component Specifications

### 1. Virtual World Physics Engine

The physics engine must track environmental effects and their interaction with the user's virtual body.

**Requirements:**
- Real-time wind vector simulation (direction, velocity, turbulence)
- Collision detection between environmental effects and body mesh
- Temperature gradient tracking
- Pressure differential calculations
- Frame-rate synchronized output (minimum 60Hz for smooth sensation)

**Wind Simulation Parameters:**
```
- Direction: 3D vector (x, y, z)
- Velocity: m/s (0-50 range typical)
- Turbulence: 0.0-1.0 (laminar to chaotic)
- Temperature: Celsius (-20 to 50 range)
- Humidity: 0-100%
```

**Output:** List of body mesh vertices affected per frame, with intensity values

---

### 2. Body Mesh Coordinate System

A standardized virtual body mesh that maps to human anatomy.

**Mesh Structure:**
- High-resolution humanoid mesh (10,000+ vertices minimum)
- Anatomically segmented regions (see Body Region Map below)
- UV-mapped for consistent coordinate referencing
- Supports real-time deformation for movement tracking

**Body Region Map:**

| Region ID | Body Part | Coordinate Range (UV) | Vertex Count |
|-----------|-----------|----------------------|--------------|
| HEAD_01 | Forehead | (0.45-0.55, 0.90-0.95) | ~200 |
| HEAD_02 | Left Cheek | (0.40-0.45, 0.85-0.90) | ~150 |
| HEAD_03 | Right Cheek | (0.55-0.60, 0.85-0.90) | ~150 |
| ARM_L_01 | Left Upper Arm | (0.15-0.25, 0.60-0.75) | ~400 |
| ARM_L_02 | Left Forearm | (0.10-0.20, 0.45-0.60) | ~350 |
| ARM_L_03 | Left Hand | (0.05-0.15, 0.35-0.45) | ~500 |
| ARM_R_01 | Right Upper Arm | (0.75-0.85, 0.60-0.75) | ~400 |
| ARM_R_02 | Right Forearm | (0.80-0.90, 0.45-0.60) | ~350 |
| ARM_R_03 | Right Hand | (0.85-0.95, 0.35-0.45) | ~500 |
| TORSO_01 | Chest | (0.35-0.65, 0.55-0.75) | ~800 |
| TORSO_02 | Abdomen | (0.35-0.65, 0.40-0.55) | ~600 |
| TORSO_03 | Upper Back | (0.35-0.65, 0.55-0.75) | ~700 |
| TORSO_04 | Lower Back | (0.35-0.65, 0.40-0.55) | ~500 |
| LEG_L_01 | Left Thigh | (0.30-0.45, 0.20-0.40) | ~500 |
| LEG_L_02 | Left Calf | (0.30-0.45, 0.05-0.20) | ~400 |
| LEG_L_03 | Left Foot | (0.30-0.40, 0.00-0.05) | ~300 |
| LEG_R_01 | Right Thigh | (0.55-0.70, 0.20-0.40) | ~500 |
| LEG_R_02 | Right Calf | (0.55-0.70, 0.05-0.20) | ~400 |
| LEG_R_03 | Right Foot | (0.60-0.70, 0.00-0.05) | ~300 |

---

### 3. Spatial Translation Layer

Converts virtual world collision data into body region queries.

**Translation Process:**
1. Receive affected vertices from physics engine
2. Map vertices to UV coordinates
3. Look up body region from coordinate ranges
4. Calculate intensity based on:
   - Number of affected vertices in region
   - Wind velocity at impact point
   - Angle of incidence
   - Duration of exposure

**Intensity Calculation Formula:**
```
intensity = (affected_vertices / total_region_vertices) * 
            (wind_velocity / max_velocity) * 
            cos(angle_of_incidence) * 
            temporal_smoothing_factor
```

**Output:** Body region IDs with normalized intensity values (0.0-1.0)

---

### 4. Neural Fingerprint Database (SQLite)

Database schema for storing and retrieving neural stimulation patterns.

**Schema Design:**

```sql
-- Body regions lookup table
CREATE TABLE body_regions (
    region_id TEXT PRIMARY KEY,
    region_name TEXT NOT NULL,
    parent_region TEXT,
    uv_min_x REAL NOT NULL,
    uv_max_x REAL NOT NULL,
    uv_min_y REAL NOT NULL,
    uv_max_y REAL NOT NULL,
    vertex_count INTEGER,
    sensitivity_multiplier REAL DEFAULT 1.0
);

-- Sensation types
CREATE TABLE sensation_types (
    sensation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensation_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, -- 'tactile', 'thermal', 'pressure', 'pain'
    description TEXT
);

-- Neural fingerprints - the core stimulation patterns
CREATE TABLE neural_fingerprints (
    fingerprint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL,
    sensation_id INTEGER NOT NULL,
    pattern_data BLOB NOT NULL, -- Serialized stimulation waveform
    frequency_hz REAL NOT NULL,
    amplitude_base REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    electrode_config TEXT NOT NULL, -- JSON array of electrode positions
    calibration_date TIMESTAMP,
    subject_id TEXT, -- For personalized patterns
    confidence_score REAL DEFAULT 0.0,
    FOREIGN KEY (region_id) REFERENCES body_regions(region_id),
    FOREIGN KEY (sensation_id) REFERENCES sensation_types(sensation_id)
);

-- Mapping table: Virtual coordinates to neural fingerprints
CREATE TABLE coordinate_fingerprint_map (
    map_id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL,
    sensation_id INTEGER NOT NULL,
    fingerprint_id INTEGER NOT NULL,
    intensity_min REAL DEFAULT 0.0,
    intensity_max REAL DEFAULT 1.0,
    priority INTEGER DEFAULT 0, -- For overlapping sensations
    FOREIGN KEY (region_id) REFERENCES body_regions(region_id),
    FOREIGN KEY (sensation_id) REFERENCES sensation_types(sensation_id),
    FOREIGN KEY (fingerprint_id) REFERENCES neural_fingerprints(fingerprint_id)
);

-- Composite sensations (e.g., "cold wind" = wind + temperature)
CREATE TABLE composite_sensations (
    composite_id INTEGER PRIMARY KEY AUTOINCREMENT,
    composite_name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE composite_components (
    composite_id INTEGER NOT NULL,
    sensation_id INTEGER NOT NULL,
    weight REAL DEFAULT 1.0,
    phase_offset_ms INTEGER DEFAULT 0,
    FOREIGN KEY (composite_id) REFERENCES composite_sensations(composite_id),
    FOREIGN KEY (sensation_id) REFERENCES sensation_types(sensation_id),
    PRIMARY KEY (composite_id, sensation_id)
);

-- Indexes for fast lookup
CREATE INDEX idx_fingerprint_region ON neural_fingerprints(region_id);
CREATE INDEX idx_fingerprint_sensation ON neural_fingerprints(sensation_id);
CREATE INDEX idx_map_region_sensation ON coordinate_fingerprint_map(region_id, sensation_id);
CREATE INDEX idx_body_region_coords ON body_regions(uv_min_x, uv_max_x, uv_min_y, uv_max_y);
```

**Sample Data Insertion:**

```sql
-- Insert sensation types
INSERT INTO sensation_types (sensation_name, category, description) VALUES
    ('light_touch', 'tactile', 'Gentle surface contact'),
    ('pressure', 'tactile', 'Firm pressing sensation'),
    ('wind_breeze', 'tactile', 'Moving air across skin'),
    ('cold', 'thermal', 'Below body temperature'),
    ('warm', 'thermal', 'Above body temperature'),
    ('vibration', 'tactile', 'Oscillating pressure');

-- Insert body regions
INSERT INTO body_regions (region_id, region_name, uv_min_x, uv_max_x, uv_min_y, uv_max_y, vertex_count, sensitivity_multiplier) VALUES
    ('ARM_L_01', 'Left Upper Arm', 0.15, 0.25, 0.60, 0.75, 400, 1.0),
    ('ARM_L_02', 'Left Forearm', 0.10, 0.20, 0.45, 0.60, 350, 1.2),
    ('ARM_L_03', 'Left Hand', 0.05, 0.15, 0.35, 0.45, 500, 2.5),
    ('ARM_R_01', 'Right Upper Arm', 0.75, 0.85, 0.60, 0.75, 400, 1.0),
    ('ARM_R_02', 'Right Forearm', 0.80, 0.90, 0.45, 0.60, 350, 1.2),
    ('ARM_R_03', 'Right Hand', 0.85, 0.95, 0.35, 0.45, 500, 2.5);
```

---

### 5. Stimulation Controller

Real-time system that retrieves and executes neural patterns.

**Query Flow:**

```sql
-- Query to get neural fingerprint for a body region and sensation
SELECT 
    nf.fingerprint_id,
    nf.pattern_data,
    nf.frequency_hz,
    nf.amplitude_base * br.sensitivity_multiplier * :intensity AS adjusted_amplitude,
    nf.duration_ms,
    nf.electrode_config
FROM coordinate_fingerprint_map cfm
JOIN neural_fingerprints nf ON cfm.fingerprint_id = nf.fingerprint_id
JOIN body_regions br ON cfm.region_id = br.region_id
WHERE cfm.region_id = :region_id
    AND cfm.sensation_id = :sensation_id
    AND :intensity BETWEEN cfm.intensity_min AND cfm.intensity_max
ORDER BY cfm.priority DESC
LIMIT 1;
```

**Coordinate-to-Region Lookup:**

```sql
-- Find body region from UV coordinates
SELECT region_id, region_name, sensitivity_multiplier
FROM body_regions
WHERE :uv_x BETWEEN uv_min_x AND uv_max_x
    AND :uv_y BETWEEN uv_min_y AND uv_max_y;
```

---

## Implementation Tasks for Claude Code

### Task 1: Core Data Structures (Rust)

Create Rust structs and enums for:
- `BodyRegion` with coordinate bounds
- `SensationType` enum
- `NeuralFingerprint` with pattern data
- `StimulationCommand` for real-time output

### Task 2: Database Layer (Rust + SQLite)

Implement:
- Database connection pool using `rusqlite` or `sqlx`
- CRUD operations for all tables
- Optimized batch queries for real-time lookup
- Caching layer for frequently accessed fingerprints

### Task 3: Spatial Translation Module

Implement:
- UV coordinate to body region mapping
- Intensity calculation from physics data
- Multi-region blending for boundary cases
- Temporal smoothing for natural sensation flow

### Task 4: Physics Integration Interface

Create trait/interface for:
- Receiving collision data from physics engine
- Processing wind vectors against body mesh
- Outputting affected regions with intensities
- Supporting multiple simultaneous effects

### Task 5: Real-Time Stimulation Pipeline

Implement:
- Frame-synchronized processing loop
- Priority queue for overlapping sensations
- Latency-optimized database queries
- Direct integration with BEAST hardware interface

---

## Example: Wind on Left Arm Flow

```
1. Physics Engine detects wind vector (5 m/s, direction: east)
   
2. Collision detection finds affected vertices on left arm mesh
   - 120 vertices affected in ARM_L_01 region
   - 80 vertices affected in ARM_L_02 region

3. Spatial Translation calculates intensities:
   - ARM_L_01: 120/400 * 5/50 * 0.85 = 0.0255 (normalized: 0.26)
   - ARM_L_02: 80/350 * 5/50 * 0.92 = 0.0211 (normalized: 0.21)

4. Database Query retrieves fingerprints:
   SELECT fingerprint for (ARM_L_01, wind_breeze, intensity=0.26)
   SELECT fingerprint for (ARM_L_02, wind_breeze, intensity=0.21)

5. Stimulation Controller executes:
   - ARM_L_01: 12Hz, amplitude 0.26 * base, electrode config [3,7,12]
   - ARM_L_02: 12Hz, amplitude 0.21 * base, electrode config [5,9,14]

6. User feels wind blowing on their left upper arm and forearm
```

---

## Integration with BEAST Platform

This system should integrate with the existing BEAST architecture:

- **Cerelog ESP-EEG Platform**: Hardware interface for stimulation delivery
- **Neural Fingerprint Detection**: Use existing fingerprint capture system to populate database
- **EEG/fNIRS Feedback**: Monitor brain response to calibrate and refine patterns

---

## Future Extensions

1. **Olfactory and Gustatory Integration**: Apply same coordinate mapping principles to chemical sense neural patterns
2. **Full Body Haptic Suit**: Expand electrode coverage for whole-body sensation
3. **Personalized Calibration**: Machine learning to optimize patterns per user
4. **Multi-User Environments**: Synchronized sensations in shared VR spaces

---

## Development Priority

1. **Phase 1**: Database schema and basic CRUD operations
2. **Phase 2**: Body mesh coordinate system and region mapping
3. **Phase 3**: Spatial translation with intensity calculations
4. **Phase 4**: Real-time stimulation pipeline
5. **Phase 5**: Physics engine integration
6. **Phase 6**: BEAST hardware interface

---

## Notes for Claude Code

When implementing this system:

1. Use Rust for all performance-critical components
2. Ensure `no_std` compatibility for embedded deployment options
3. Design for real-time constraints (< 16ms frame budget)
4. Include comprehensive error handling for hardware failures
5. Add telemetry for debugging stimulation timing issues
6. Consider WASM compilation for web-based simulation testing

This prompt provides the complete specification for building a VR-to-neural-stimulation mapping system that can make users feel virtual wind (and other sensations) through direct brain stimulation.

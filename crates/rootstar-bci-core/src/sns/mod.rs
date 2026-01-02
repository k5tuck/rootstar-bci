//! Sensory Neural Simulation (SNS) Module
//!
//! This module provides bidirectional mapping between real cortical signals (EEG/fNIRS)
//! and simulated peripheral sensory receptors.
//!
//! # Modalities
//!
//! - **Tactile**: Mechanoreceptors (Meissner, Merkel, Pacinian, Ruffini)
//! - **Auditory**: Hair cells (inner and outer) with tonotopic organization
//! - **Gustatory**: Taste receptor cells (sweet, salty, sour, bitter, umami)
//! - **Visual**: Photoreceptors (rods, cones) and retinal ganglion cells
//! - **Olfactory**: Olfactory receptor neurons and olfactory bulb circuits
//!
//! # Architecture
//!
//! The SNS module supports two model tiers:
//! - **Phenomenological**: Transfer function approach for real-time simulation
//! - **Spiking**: LIF neurons with conductance-based synapses
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::{
//!     SensoryModality, ReceptorType, StimulusField,
//!     tactile::MeissnerReceptor,
//! };
//!
//! // Create a Meissner corpuscle receptor
//! let mut receptor = MeissnerReceptor::default();
//!
//! // Compute firing rate from stimulus
//! let rate = receptor.compute_rate(50.0, 1.0);
//! assert!(rate > 0.0);
//! ```

pub mod auditory;
pub mod encoding;
pub mod error;
pub mod gustatory;
pub mod olfactory;
pub mod spiking;
pub mod tactile;
pub mod types;
pub mod visual;

// Re-export commonly used types
pub use encoding::{
    NeuralEncoder, PopulationCode, RateCode, SpikeEncoder, SpikePattern, TemporalCode,
};
pub use error::SnsError;
pub use types::{
    BodyRegion, OdorantClass, Position2D, Position3D, ReceptiveField, ReceptorPopulation,
    ReceptorType, SensoryModality, SpatialExtent, SpatialReceptor, StimulusDescriptor,
    StimulusField, StimulusType,
};

// Re-export receptor models
pub use auditory::{
    AuditoryReceptor, BasilarMembrane, CochlearPosition, HairCellType, InnerHairCell,
    OuterHairCell,
};
pub use gustatory::{GustatoryReceptor, TasteBud, TasteQuality, TasteReceptorCell};
pub use olfactory::{
    GranuleCell, MitralCell, Odorant, OdorantAffinities, OlfactoryBulb, OlfactoryReceptor,
    OlfactoryReceptorId, OlfactoryReceptorNeuron, TuftedCell,
};
pub use spiking::{LifNeuron, SpikingNetwork, Synapse, SynapseType};
pub use tactile::{
    MeissnerReceptor, MerkelReceptor, PacinianReceptor, RuffiniReceptor, TactileReceptor,
};
pub use visual::{
    ConePhotoreceptor, ConeType, GanglionCellType, Phosphene, PhospheneModel, RetinotopicMap,
    RetinalGanglionCell, RodPhotoreceptor, VisualCortexRegion, VisualReceptor,
};

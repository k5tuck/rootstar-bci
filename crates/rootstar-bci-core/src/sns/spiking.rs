//! Spiking Neural Network Models
//!
//! This module provides spiking neuron models compatible with `no_std`:
//! - **LIF**: Leaky Integrate-and-Fire neurons
//! - **Synapses**: Conductance-based synaptic connections
//! - **Networks**: Population dynamics and connectivity
//!
//! All models use fixed-point arithmetic for deterministic embedded execution.
//!
//! # Example
//!
//! ```rust
//! use rootstar_bci_core::sns::spiking::{LifNeuron, Synapse, SynapseType};
//! use rootstar_bci_core::types::Fixed24_8;
//!
//! let mut neuron = LifNeuron::default();
//! let current = Fixed24_8::from_f32(20.0);
//!
//! // Simulate for 1ms
//! let spike = neuron.step(current, 1000, 0);
//! ```

use serde::{Deserialize, Serialize};

use crate::types::Fixed24_8;

// ============================================================================
// Physical Constants (in Fixed-Point)
// ============================================================================

/// Resting membrane potential (-70 mV)
pub const V_REST: Fixed24_8 = Fixed24_8::from_raw(-70 * 256);

/// Spike threshold (-55 mV)
pub const V_THRESHOLD: Fixed24_8 = Fixed24_8::from_raw(-55 * 256);

/// Reset potential after spike (-75 mV)
pub const V_RESET: Fixed24_8 = Fixed24_8::from_raw(-75 * 256);

/// Default membrane time constant (15 ms)
pub const TAU_M_DEFAULT: Fixed24_8 = Fixed24_8::from_raw(15 * 256);

/// Default refractory period (2 ms)
pub const T_REFRAC_DEFAULT: Fixed24_8 = Fixed24_8::from_raw(2 * 256);

/// Excitatory reversal potential (0 mV)
pub const E_EXC: Fixed24_8 = Fixed24_8::from_raw(0);

/// Inhibitory reversal potential (-80 mV)
pub const E_INH: Fixed24_8 = Fixed24_8::from_raw(-80 * 256);

// ============================================================================
// Leaky Integrate-and-Fire Neuron
// ============================================================================

/// Leaky Integrate-and-Fire neuron model
///
/// Uses fixed-point arithmetic for deterministic no_std execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifNeuron {
    /// Resting membrane potential (mV)
    pub v_rest: Fixed24_8,
    /// Spike threshold (mV)
    pub v_thresh: Fixed24_8,
    /// Reset potential after spike (mV)
    pub v_reset: Fixed24_8,
    /// Membrane time constant (ms)
    pub tau_m: Fixed24_8,
    /// Absolute refractory period (ms)
    pub t_refrac: Fixed24_8,
    /// Remaining refractory time (ms)
    refrac_remaining: Fixed24_8,
    /// Current membrane potential (mV)
    pub v_membrane: Fixed24_8,
    /// Time of last spike (us), None if never spiked
    last_spike_time: Option<u64>,
    /// Neuron identifier
    pub id: u16,
}

impl LifNeuron {
    /// Create a new LIF neuron with default parameters
    #[inline]
    #[must_use]
    pub fn new(id: u16) -> Self {
        Self {
            v_rest: V_REST,
            v_thresh: V_THRESHOLD,
            v_reset: V_RESET,
            tau_m: TAU_M_DEFAULT,
            t_refrac: T_REFRAC_DEFAULT,
            refrac_remaining: Fixed24_8::ZERO,
            v_membrane: V_REST,
            last_spike_time: None,
            id,
        }
    }

    /// Create with custom parameters
    #[must_use]
    pub fn custom(
        id: u16,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_m: f32,
        t_refrac: f32,
    ) -> Self {
        Self {
            v_rest: Fixed24_8::from_f32(v_rest),
            v_thresh: Fixed24_8::from_f32(v_thresh),
            v_reset: Fixed24_8::from_f32(v_reset),
            tau_m: Fixed24_8::from_f32(tau_m),
            t_refrac: Fixed24_8::from_f32(t_refrac),
            refrac_remaining: Fixed24_8::ZERO,
            v_membrane: Fixed24_8::from_f32(v_rest),
            last_spike_time: None,
            id,
        }
    }

    /// Create a fast-spiking interneuron
    #[must_use]
    pub fn fast_spiking(id: u16) -> Self {
        Self::custom(id, -70.0, -50.0, -70.0, 10.0, 1.0)
    }

    /// Create a regular-spiking excitatory neuron
    #[must_use]
    pub fn regular_spiking(id: u16) -> Self {
        Self::custom(id, -70.0, -55.0, -75.0, 20.0, 2.0)
    }

    /// Integrate input current and check for spike
    ///
    /// # Arguments
    /// * `i_input` - Input current (arbitrary units)
    /// * `dt_us` - Time step in microseconds
    /// * `current_time_us` - Current simulation time in microseconds
    ///
    /// # Returns
    /// Some(spike_time) if the neuron fired, None otherwise
    pub fn step(&mut self, i_input: Fixed24_8, dt_us: u64, current_time_us: u64) -> Option<u64> {
        // Convert dt from microseconds to milliseconds (approximate)
        // dt_ms ≈ dt_us / 1000 ≈ dt_us >> 10
        let dt_ms = Fixed24_8::from_raw((dt_us as i32) >> 2); // Scale factor adjustment

        // Check refractory period
        if self.refrac_remaining > Fixed24_8::ZERO {
            let new_refrac = Fixed24_8::from_raw(
                (self.refrac_remaining.to_raw() - dt_ms.to_raw()).max(0)
            );
            self.refrac_remaining = new_refrac;
            return None;
        }

        // Membrane dynamics: dV/dt = (V_rest - V + R*I) / tau_m
        // Euler integration: V(t+dt) = V(t) + dt * dV/dt
        // Simplified: dV = dt * (V_rest - V + I) / tau_m

        let v_diff = Fixed24_8::from_raw(
            self.v_rest.to_raw() - self.v_membrane.to_raw() + i_input.to_raw()
        );

        // Prevent division by zero
        let tau_raw = self.tau_m.to_raw().max(1);

        // dV = (dt_ms * v_diff) / tau_m
        let dv_raw = (dt_ms.to_raw() as i64 * v_diff.to_raw() as i64) / tau_raw as i64;
        let dv = Fixed24_8::from_raw(dv_raw.clamp(i32::MIN as i64, i32::MAX as i64) as i32);

        self.v_membrane = Fixed24_8::from_raw(
            self.v_membrane.to_raw().saturating_add(dv.to_raw())
        );

        // Spike check
        if self.v_membrane >= self.v_thresh {
            self.v_membrane = self.v_reset;
            self.refrac_remaining = self.t_refrac;
            self.last_spike_time = Some(current_time_us);
            Some(current_time_us)
        } else {
            None
        }
    }

    /// Get the time since last spike in microseconds
    #[inline]
    #[must_use]
    pub fn time_since_spike(&self, current_time_us: u64) -> Option<u64> {
        self.last_spike_time.map(|t| current_time_us.saturating_sub(t))
    }

    /// Check if neuron is in refractory period
    #[inline]
    #[must_use]
    pub fn is_refractory(&self) -> bool {
        self.refrac_remaining > Fixed24_8::ZERO
    }

    /// Reset neuron state
    pub fn reset(&mut self) {
        self.v_membrane = self.v_rest;
        self.refrac_remaining = Fixed24_8::ZERO;
        self.last_spike_time = None;
    }

    /// Get membrane potential as f32
    #[inline]
    #[must_use]
    pub fn membrane_potential_mv(&self) -> f32 {
        self.v_membrane.to_f32()
    }
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self::new(0)
    }
}

// ============================================================================
// Synapse Types
// ============================================================================

/// Type of synapse (excitatory or inhibitory)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SynapseType {
    /// Excitatory synapse (AMPA-like)
    Excitatory,
    /// Inhibitory synapse (GABA-like)
    Inhibitory,
    /// NMDA-like (slow excitatory)
    Nmda,
    /// Electrical gap junction
    GapJunction,
}

impl SynapseType {
    /// Get the reversal potential for this synapse type
    #[inline]
    #[must_use]
    pub const fn reversal_potential(self) -> Fixed24_8 {
        match self {
            Self::Excitatory | Self::Nmda => E_EXC,
            Self::Inhibitory => E_INH,
            Self::GapJunction => Fixed24_8::ZERO, // Not applicable
        }
    }

    /// Get the default time constant (ms)
    #[inline]
    #[must_use]
    pub const fn default_tau_ms(self) -> f32 {
        match self {
            Self::Excitatory => 2.0,
            Self::Inhibitory => 5.0,
            Self::Nmda => 50.0,
            Self::GapJunction => 0.1,
        }
    }
}

// ============================================================================
// Synapse
// ============================================================================

/// Conductance-based synapse with exponential decay
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Synapse {
    /// Synaptic weight (determines PSP amplitude)
    pub weight: Fixed24_8,
    /// Synaptic time constant (ms)
    pub tau_syn: Fixed24_8,
    /// Reversal potential (mV)
    pub e_syn: Fixed24_8,
    /// Synapse type
    pub synapse_type: SynapseType,
    /// Current conductance state
    conductance: Fixed24_8,
    /// Presynaptic neuron ID
    pub pre_id: u16,
    /// Postsynaptic neuron ID
    pub post_id: u16,
    /// Transmission delay (us)
    pub delay_us: u32,
}

impl Synapse {
    /// Create a new synapse
    #[must_use]
    pub fn new(
        pre_id: u16,
        post_id: u16,
        weight: f32,
        synapse_type: SynapseType,
    ) -> Self {
        Self {
            weight: Fixed24_8::from_f32(weight),
            tau_syn: Fixed24_8::from_f32(synapse_type.default_tau_ms()),
            e_syn: synapse_type.reversal_potential(),
            synapse_type,
            conductance: Fixed24_8::ZERO,
            pre_id,
            post_id,
            delay_us: 1000, // 1ms default delay
        }
    }

    /// Create with custom delay
    #[must_use]
    pub fn with_delay(
        pre_id: u16,
        post_id: u16,
        weight: f32,
        synapse_type: SynapseType,
        delay_us: u32,
    ) -> Self {
        let mut syn = Self::new(pre_id, post_id, weight, synapse_type);
        syn.delay_us = delay_us;
        syn
    }

    /// Process incoming spike and update conductance
    pub fn receive_spike(&mut self) {
        // Instantaneous conductance increase
        self.conductance = Fixed24_8::from_raw(
            self.conductance.to_raw().saturating_add(self.weight.to_raw())
        );
    }

    /// Decay conductance and compute synaptic current
    ///
    /// # Arguments
    /// * `v_post` - Postsynaptic membrane potential (mV)
    /// * `dt_ms` - Time step (ms)
    ///
    /// # Returns
    /// Synaptic current contribution
    pub fn compute_current(&mut self, v_post: Fixed24_8, dt_ms: Fixed24_8) -> Fixed24_8 {
        // Exponential decay: g(t) = g(t-dt) * exp(-dt/tau)
        // Approximate: g(t) = g(t-dt) * (1 - dt/tau) for small dt

        let tau_raw = self.tau_syn.to_raw().max(1);
        let decay_factor = 256 - (dt_ms.to_raw() * 256 / tau_raw).min(256);

        self.conductance = Fixed24_8::from_raw(
            ((self.conductance.to_raw() as i64 * decay_factor as i64) >> 8) as i32
        );

        // I = g * (E_syn - V)
        let driving_force = Fixed24_8::from_raw(
            self.e_syn.to_raw() - v_post.to_raw()
        );

        Fixed24_8::from_raw(
            ((self.conductance.to_raw() as i64 * driving_force.to_raw() as i64) >> 8) as i32
        )
    }

    /// Get current conductance
    #[inline]
    #[must_use]
    pub fn conductance(&self) -> f32 {
        self.conductance.to_f32()
    }

    /// Reset synapse state
    pub fn reset(&mut self) {
        self.conductance = Fixed24_8::ZERO;
    }
}

// ============================================================================
// Spiking Network
// ============================================================================

/// Network of spiking neurons with synaptic connections
#[derive(Clone, Debug)]
pub struct SpikingNetwork<const N_NEURONS: usize, const N_SYNAPSES: usize> {
    /// Neurons in the network
    pub neurons: heapless::Vec<LifNeuron, N_NEURONS>,
    /// Synaptic connections (flat list)
    pub synapses: heapless::Vec<Synapse, N_SYNAPSES>,
    /// Recent spike buffer per neuron (last spike time)
    spike_times: heapless::Vec<Option<u64>, N_NEURONS>,
    /// Current simulation time (us)
    current_time_us: u64,
}

impl<const N_NEURONS: usize, const N_SYNAPSES: usize> SpikingNetwork<N_NEURONS, N_SYNAPSES> {
    /// Create a new empty network
    #[must_use]
    pub fn new() -> Self {
        Self {
            neurons: heapless::Vec::new(),
            synapses: heapless::Vec::new(),
            spike_times: heapless::Vec::new(),
            current_time_us: 0,
        }
    }

    /// Add a neuron to the network
    pub fn add_neuron(&mut self, neuron: LifNeuron) -> Result<u16, LifNeuron> {
        let id = self.neurons.len() as u16;
        self.neurons.push(neuron)?;
        self.spike_times.push(None).map_err(|_| self.neurons.pop().unwrap())?;
        Ok(id)
    }

    /// Add a synapse to the network
    pub fn add_synapse(&mut self, synapse: Synapse) -> Result<(), Synapse> {
        self.synapses.push(synapse)
    }

    /// Connect two neurons
    pub fn connect(
        &mut self,
        pre_id: u16,
        post_id: u16,
        weight: f32,
        synapse_type: SynapseType,
    ) -> Result<(), Synapse> {
        let synapse = Synapse::new(pre_id, post_id, weight, synapse_type);
        self.add_synapse(synapse)
    }

    /// Simulate one timestep
    ///
    /// # Arguments
    /// * `external_currents` - External input current for each neuron
    /// * `dt_us` - Time step in microseconds
    ///
    /// # Returns
    /// Array indicating which neurons spiked
    pub fn step(
        &mut self,
        external_currents: &[Fixed24_8],
        dt_us: u64,
    ) -> heapless::Vec<bool, N_NEURONS> {
        let mut spiked = heapless::Vec::new();
        let dt_ms = Fixed24_8::from_f32(dt_us as f32 / 1000.0);

        // Collect synaptic currents for each neuron
        let mut syn_currents: heapless::Vec<Fixed24_8, N_NEURONS> = heapless::Vec::new();
        for _ in 0..self.neurons.len() {
            let _ = syn_currents.push(Fixed24_8::ZERO);
        }

        // Process synapses: check for delayed spikes and compute currents
        for synapse in &mut self.synapses {
            let pre_idx = synapse.pre_id as usize;
            let post_idx = synapse.post_id as usize;

            if pre_idx >= self.neurons.len() || post_idx >= self.neurons.len() {
                continue;
            }

            // Check if presynaptic spike should arrive (accounting for delay)
            if let Some(spike_time) = self.spike_times.get(pre_idx).copied().flatten() {
                let arrival_time = spike_time + synapse.delay_us as u64;
                if arrival_time <= self.current_time_us &&
                   arrival_time > self.current_time_us.saturating_sub(dt_us) {
                    synapse.receive_spike();
                }
            }

            // Compute synaptic current
            let v_post = self.neurons[post_idx].v_membrane;
            let i_syn = synapse.compute_current(v_post, dt_ms);

            if let Some(current) = syn_currents.get_mut(post_idx) {
                *current = Fixed24_8::from_raw(current.to_raw().saturating_add(i_syn.to_raw()));
            }
        }

        // Update all neurons
        for i in 0..self.neurons.len() {
            // Total input current
            let i_ext = external_currents.get(i).copied().unwrap_or(Fixed24_8::ZERO);
            let i_syn = syn_currents.get(i).copied().unwrap_or(Fixed24_8::ZERO);
            let i_total = Fixed24_8::from_raw(i_ext.to_raw().saturating_add(i_syn.to_raw()));

            // Step neuron
            let spike = self.neurons[i].step(i_total, dt_us, self.current_time_us);

            // Record spike
            if spike.is_some() {
                self.spike_times[i] = spike;
                let _ = spiked.push(true);
            } else {
                let _ = spiked.push(false);
            }
        }

        // Advance time
        self.current_time_us += dt_us;

        spiked
    }

    /// Get current simulation time
    #[inline]
    #[must_use]
    pub fn time_us(&self) -> u64 {
        self.current_time_us
    }

    /// Get number of neurons
    #[inline]
    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Get number of synapses
    #[inline]
    #[must_use]
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Reset all neurons and synapses
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for synapse in &mut self.synapses {
            synapse.reset();
        }
        for spike_time in &mut self.spike_times {
            *spike_time = None;
        }
        self.current_time_us = 0;
    }

    /// Get mean firing rate across population (Hz)
    #[must_use]
    pub fn mean_firing_rate(&self, window_us: u64) -> f32 {
        if self.neurons.is_empty() || window_us == 0 {
            return 0.0;
        }

        let cutoff = self.current_time_us.saturating_sub(window_us);
        let spike_count: usize = self.spike_times.iter()
            .filter(|&&t| t.map(|t| t >= cutoff).unwrap_or(false))
            .count();

        let window_s = window_us as f32 / 1_000_000.0;
        spike_count as f32 / (self.neurons.len() as f32 * window_s)
    }
}

impl<const N_NEURONS: usize, const N_SYNAPSES: usize> Default for SpikingNetwork<N_NEURONS, N_SYNAPSES> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Afferent Population Builders
// ============================================================================

/// Create a tactile afferent population
pub fn create_tactile_afferents<const N: usize, const S: usize>() -> SpikingNetwork<N, S> {
    let mut network = SpikingNetwork::new();

    // Create afferent neurons with varying properties
    for i in 0..N.min(64) {
        let neuron = if i % 4 == 0 {
            LifNeuron::fast_spiking(i as u16)
        } else {
            LifNeuron::regular_spiking(i as u16)
        };
        let _ = network.add_neuron(neuron);
    }

    // Add lateral inhibition (sparse connectivity)
    for i in 0..network.neuron_count() {
        for j in 0..network.neuron_count() {
            if i != j && (i as i32 - j as i32).abs() <= 2 {
                let _ = network.connect(
                    i as u16,
                    j as u16,
                    0.1, // Weak inhibition
                    SynapseType::Inhibitory,
                );
            }
        }
    }

    network
}

/// Create an auditory nerve fiber population
pub fn create_auditory_fibers<const N: usize, const S: usize>(n_fibers: usize) -> SpikingNetwork<N, S> {
    let mut network = SpikingNetwork::new();

    // Create fibers with tonotopic organization
    for i in 0..n_fibers.min(N) {
        // High-SR fibers are more common
        let neuron = if i % 3 == 0 {
            // Low-SR fiber (higher threshold, wider dynamic range)
            LifNeuron::custom(i as u16, -70.0, -50.0, -75.0, 15.0, 1.0)
        } else {
            // High-SR fiber
            LifNeuron::custom(i as u16, -65.0, -55.0, -70.0, 10.0, 1.0)
        };
        let _ = network.add_neuron(neuron);
    }

    // Auditory nerve fibers typically don't have local connectivity
    // (connections are to cochlear nucleus, not modeled here)

    network
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron_spike() {
        let mut neuron = LifNeuron::default();

        // Strong input should cause spike
        let spike = neuron.step(Fixed24_8::from_f32(50.0), 1000, 0);
        assert!(spike.is_some());

        // Should be in refractory period
        assert!(neuron.is_refractory());

        // Weak input during refractory should not spike
        let spike2 = neuron.step(Fixed24_8::from_f32(50.0), 1000, 1000);
        assert!(spike2.is_none());
    }

    #[test]
    fn test_lif_neuron_membrane_dynamics() {
        let mut neuron = LifNeuron::default();

        // No input should decay toward rest
        let initial = neuron.membrane_potential_mv();
        neuron.v_membrane = Fixed24_8::from_f32(-60.0); // Above rest

        for _ in 0..100 {
            neuron.step(Fixed24_8::ZERO, 1000, 0);
        }

        let final_v = neuron.membrane_potential_mv();
        assert!(final_v < -60.0); // Should have decayed toward rest
    }

    #[test]
    fn test_synapse_current() {
        let mut synapse = Synapse::new(0, 1, 1.0, SynapseType::Excitatory);

        synapse.receive_spike();
        assert!(synapse.conductance() > 0.0);

        let v_post = Fixed24_8::from_f32(-70.0);
        let dt_ms = Fixed24_8::from_f32(1.0);
        let current = synapse.compute_current(v_post, dt_ms);

        // Excitatory current should be positive (depolarizing)
        assert!(current.to_f32() > 0.0);
    }

    #[test]
    fn test_synapse_decay() {
        let mut synapse = Synapse::new(0, 1, 1.0, SynapseType::Excitatory);

        synapse.receive_spike();
        let g1 = synapse.conductance();

        let v_post = Fixed24_8::from_f32(-70.0);
        let dt_ms = Fixed24_8::from_f32(5.0);

        for _ in 0..10 {
            synapse.compute_current(v_post, dt_ms);
        }

        let g2 = synapse.conductance();
        assert!(g2 < g1); // Conductance should decay
    }

    #[test]
    fn test_network_simulation() {
        let mut network: SpikingNetwork<4, 8> = SpikingNetwork::new();

        // Add neurons
        for i in 0..4 {
            let _ = network.add_neuron(LifNeuron::new(i));
        }

        // Connect: 0 -> 1, 1 -> 2, 2 -> 3
        let _ = network.connect(0, 1, 2.0, SynapseType::Excitatory);
        let _ = network.connect(1, 2, 2.0, SynapseType::Excitatory);
        let _ = network.connect(2, 3, 2.0, SynapseType::Excitatory);

        // Drive first neuron
        let mut inputs = [Fixed24_8::ZERO; 4];
        inputs[0] = Fixed24_8::from_f32(30.0);

        // Simulate
        let mut any_spike = false;
        for _ in 0..100 {
            let spiked = network.step(&inputs, 1000);
            if spiked.iter().any(|&s| s) {
                any_spike = true;
            }
        }

        assert!(any_spike);
    }

    #[test]
    fn test_afferent_population() {
        let network: SpikingNetwork<16, 64> = create_tactile_afferents();

        assert!(network.neuron_count() > 0);
        assert!(network.synapse_count() > 0);
    }
}

"""Real-time feedback controller for sensory simulation.

This module provides:
- Bidirectional feedback loop for neural pattern matching
- Session management for stimulation
- Real-time state tracking
"""

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from .database import FingerprintDatabase, FingerprintIndex


class SystemState(Enum):
    """Current state of the neural interface system."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    STIMULATING = "stimulating"
    FEEDBACK = "feedback"
    ERROR = "error"


@dataclass
class StimulationProtocol:
    """Stimulation parameters for sensory playback."""

    # Current parameters
    current_amplitude_ua: float  # 100-2000 µA
    waveform: str  # "DC", "AC", "pulsed"
    frequency_hz: Optional[float] = None  # For AC/pulsed (0.1-100 Hz)
    duty_cycle: Optional[float] = None  # For pulsed (0-1)

    # Duration
    ramp_up_s: float = 30.0
    stimulation_duration_s: float = 300.0
    ramp_down_s: float = 30.0

    # Electrode configuration
    anode_electrodes: List[str] = field(default_factory=lambda: ["Cz"])
    cathode_electrodes: List[str] = field(default_factory=lambda: ["Fp1", "Fp2"])

    # Safety limits
    max_current_density_ua_cm2: float = 25.0
    max_charge_density_uc_cm2: float = 40.0

    def validate(self) -> bool:
        """Check protocol is within safety limits."""
        if self.current_amplitude_ua > 2000:
            return False
        if self.frequency_hz and self.frequency_hz > 100:
            return False
        if self.duty_cycle and not 0 <= self.duty_cycle <= 1:
            return False
        return True


@dataclass
class BidirectionalSession:
    """Active neural interface session for sensory simulation."""

    session_id: str
    target_experience: str

    # State tracking
    state: SystemState = SystemState.IDLE
    current_embedding: Optional[np.ndarray] = None
    target_embedding: Optional[np.ndarray] = None

    # Feedback metrics
    similarity_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    stimulation_history: List[StimulationProtocol] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    last_update_time: float = 0.0

    def elapsed_time(self) -> float:
        """Get elapsed time since session start."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time


class AcquisitionSystem:
    """Interface to neural acquisition hardware.

    This is a placeholder that should be replaced with actual
    hardware interface (e.g., Cerelog, OpenBCI).
    """

    def __init__(self, eeg_channels: int = 8, fnirs_channels: int = 4):
        self.eeg_channels = eeg_channels
        self.fnirs_channels = fnirs_channels
        self._running = False

    async def start(self) -> None:
        """Start data acquisition."""
        self._running = True

    async def stop(self) -> None:
        """Stop data acquisition."""
        self._running = False

    async def get_current_data(
        self,
        duration_s: float = 0.1,
    ) -> tuple:
        """Get current EEG and fNIRS data.

        Args:
            duration_s: Duration of data to acquire

        Returns:
            Tuple of (eeg_data, fnirs_data)
        """
        # Placeholder - return random data
        samples = int(duration_s * 500)  # 500 Hz EEG
        fnirs_samples = int(duration_s * 25)  # 25 Hz fNIRS

        eeg_data = np.random.randn(self.eeg_channels, samples)
        fnirs_data = np.random.randn(self.fnirs_channels, 2, fnirs_samples)

        await asyncio.sleep(duration_s)
        return eeg_data, fnirs_data


class StimulationSystem:
    """Interface to transcranial stimulation hardware.

    Placeholder for actual hardware interface.
    """

    def __init__(self, n_channels: int = 8):
        self.n_channels = n_channels
        self._active = False
        self._current_protocol: Optional[StimulationProtocol] = None

    async def apply_protocol(self, protocol: StimulationProtocol) -> None:
        """Apply stimulation protocol.

        Args:
            protocol: Stimulation parameters
        """
        if not protocol.validate():
            raise ValueError("Protocol fails safety validation")

        self._current_protocol = protocol
        self._active = True

    async def ramp_down(self) -> None:
        """Gradually reduce stimulation to zero."""
        if self._current_protocol:
            # Simulate ramp down
            await asyncio.sleep(self._current_protocol.ramp_down_s)
        self._active = False
        self._current_protocol = None

    async def emergency_stop(self) -> None:
        """Immediately stop all stimulation."""
        self._active = False
        self._current_protocol = None

    @property
    def is_active(self) -> bool:
        """Check if stimulation is active."""
        return self._active


class RealTimeFeedbackController:
    """Controls bidirectional feedback loop for sensory simulation.

    The feedback loop:
    1. Acquire current brain state (EEG + fNIRS)
    2. Compare to target fingerprint
    3. Compute corrective stimulation
    4. Apply stimulation
    5. Measure response
    6. Repeat until target achieved or timeout
    """

    def __init__(
        self,
        acquisition_system: Optional[AcquisitionSystem] = None,
        stimulation_system: Optional[StimulationSystem] = None,
        fingerprint_db: Optional[FingerprintDatabase] = None,
        fingerprint_index: Optional[FingerprintIndex] = None,
    ):
        self.acquisition = acquisition_system or AcquisitionSystem()
        self.stimulation = stimulation_system or StimulationSystem()
        self.fingerprint_db = fingerprint_db
        self.fingerprint_index = fingerprint_index or FingerprintIndex()

        self.session: Optional[BidirectionalSession] = None
        self.running = False

        # Control parameters
        self.update_rate_hz = 10.0
        self.similarity_threshold = 0.90
        self.max_iterations = 100

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_similarity_update: Optional[Callable] = None

    def set_callbacks(
        self,
        on_state_change: Optional[Callable[[SystemState], None]] = None,
        on_similarity_update: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Set callback functions for state updates.

        Args:
            on_state_change: Called when system state changes
            on_similarity_update: Called when similarity is computed
        """
        self._on_state_change = on_state_change
        self._on_similarity_update = on_similarity_update

    async def start_session(self, target_experience: str) -> str:
        """Initialize new sensory simulation session.

        Args:
            target_experience: Label of target sensory experience

        Returns:
            Session ID

        Raises:
            ValueError: If no fingerprint found for target experience
        """
        session_id = str(uuid.uuid4())[:8]

        # Load target fingerprint
        target_embedding = self._load_target_embedding(target_experience)
        if target_embedding is None:
            raise ValueError(f"No fingerprint found for: {target_experience}")

        self.session = BidirectionalSession(
            session_id=session_id,
            target_experience=target_experience,
            target_embedding=target_embedding,
            start_time=time.time(),
        )

        self.running = True

        # Start acquisition
        await self.acquisition.start()

        # Start feedback loop
        asyncio.create_task(self._feedback_loop())

        return session_id

    async def _feedback_loop(self) -> None:
        """Main feedback control loop."""
        if self.session is None:
            return

        iteration = 0

        while self.running and iteration < self.max_iterations:
            try:
                self._set_state(SystemState.RECORDING)

                # 1. Acquire current brain state
                eeg_data, fnirs_data = await self.acquisition.get_current_data(
                    duration_s=0.1
                )

                self._set_state(SystemState.PROCESSING)

                # 2. Extract current embedding
                current_embedding = self._extract_embedding(eeg_data, fnirs_data)
                self.session.current_embedding = current_embedding
                self.session.last_update_time = time.time()

                # 3. Compute similarity to target
                similarity = self._compute_similarity(
                    current_embedding, self.session.target_embedding
                )
                self.session.similarity_history.append(similarity)

                if self._on_similarity_update:
                    self._on_similarity_update(similarity)

                # 4. Check if target achieved
                if similarity >= self.similarity_threshold:
                    print(f"Target achieved! Similarity: {similarity:.3f}")
                    break

                # 5. Compute corrective stimulation
                protocol = self._compute_stimulation_params(
                    self.session.target_embedding,
                    current_embedding,
                )

                self._set_state(SystemState.STIMULATING)

                # 6. Apply stimulation
                await self.stimulation.apply_protocol(protocol)
                self.session.stimulation_history.append(protocol)

                self._set_state(SystemState.FEEDBACK)

                # 7. Wait for next iteration
                await asyncio.sleep(1.0 / self.update_rate_hz)
                iteration += 1

            except Exception as e:
                print(f"Error in feedback loop: {e}")
                self._set_state(SystemState.ERROR)
                break

        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up after session ends."""
        self._set_state(SystemState.IDLE)
        self.running = False

        await self.stimulation.ramp_down()
        await self.acquisition.stop()

    def _set_state(self, state: SystemState) -> None:
        """Update system state and notify callbacks."""
        if self.session:
            self.session.state = state
        if self._on_state_change:
            self._on_state_change(state)

    def _load_target_embedding(self, experience: str) -> Optional[np.ndarray]:
        """Load target embedding from database."""
        if self.fingerprint_db:
            return self.fingerprint_db.get_reference_embedding(experience)
        else:
            # Return random embedding for testing
            return np.random.randn(256).astype(np.float32)

    def _extract_embedding(
        self,
        eeg_data: np.ndarray,
        fnirs_data: np.ndarray,
    ) -> np.ndarray:
        """Extract embedding from current neural data.

        This is a placeholder - should use the ML encoder model.
        """
        # Simplified: concatenate features and project
        eeg_features = np.mean(eeg_data, axis=1)  # Mean across time
        fnirs_features = np.mean(fnirs_data, axis=(1, 2))  # Mean across channels/time

        combined = np.concatenate([eeg_features, fnirs_features])

        # Project to embedding dimension (simplified)
        embedding = np.random.randn(256).astype(np.float32)
        embedding[:len(combined)] = combined[:256]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _compute_similarity(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if current is None or target is None:
            return 0.0

        dot_product = np.dot(current.flatten(), target.flatten())
        norm_a = np.linalg.norm(current)
        norm_b = np.linalg.norm(target)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _compute_stimulation_params(
        self,
        target_embedding: np.ndarray,
        current_embedding: np.ndarray,
    ) -> StimulationProtocol:
        """Compute stimulation parameters to drive brain toward target.

        This is a simplified placeholder - should use the inverse model.
        """
        delta = target_embedding - current_embedding
        magnitude = np.linalg.norm(delta)

        # Scale amplitude based on distance
        amplitude = 500 + magnitude * 500  # 500-1000 µA
        amplitude = min(amplitude, 1500)

        return StimulationProtocol(
            current_amplitude_ua=amplitude,
            waveform="AC",
            frequency_hz=40.0,  # Gamma
            ramp_up_s=0.5,
            stimulation_duration_s=0.1,
            ramp_down_s=0.1,
        )

    async def stop_session(self) -> None:
        """Gracefully stop current session."""
        self.running = False
        if self.stimulation.is_active:
            await self.stimulation.ramp_down()

    async def emergency_stop(self) -> None:
        """Immediately stop all activity."""
        self.running = False
        await self.stimulation.emergency_stop()
        self._set_state(SystemState.IDLE)

    def get_session_stats(self) -> Optional[Dict]:
        """Get current session statistics."""
        if self.session is None:
            return None

        history = list(self.session.similarity_history)

        return {
            "session_id": self.session.session_id,
            "target": self.session.target_experience,
            "state": self.session.state.value,
            "elapsed_s": self.session.elapsed_time(),
            "current_similarity": history[-1] if history else 0.0,
            "avg_similarity": np.mean(history) if history else 0.0,
            "max_similarity": max(history) if history else 0.0,
            "iterations": len(history),
            "stimulations": len(self.session.stimulation_history),
        }


async def demo_feedback_loop():
    """Demonstrate the feedback controller."""
    print("Starting Neural Fingerprint Feedback Demo...")

    controller = RealTimeFeedbackController()

    # Set callbacks
    def on_state(state: SystemState):
        print(f"State: {state.value}")

    def on_similarity(sim: float):
        print(f"Similarity: {sim:.3f}")

    controller.set_callbacks(on_state, on_similarity)
    controller.max_iterations = 5  # Short demo

    try:
        session_id = await controller.start_session("apple_taste")
        print(f"Session started: {session_id}")

        # Wait for session to complete
        while controller.running:
            await asyncio.sleep(0.5)
            stats = controller.get_session_stats()
            if stats:
                print(f"Iteration {stats['iterations']}, Similarity: {stats['current_similarity']:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await controller.stop_session()
        print("Session ended")
        stats = controller.get_session_stats()
        if stats:
            print(f"Final stats: {stats}")


if __name__ == "__main__":
    asyncio.run(demo_feedback_loop())

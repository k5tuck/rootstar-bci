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

    Supports multiple backends:
    - 'simulation': Generate synthetic data for testing
    - 'serial': Direct serial connection to Rootstar BCI device
    - 'lsl': Lab Streaming Layer inlet (requires pylsl)
    - 'brainflow': BrainFlow board connection (requires brainflow)

    Example:
        # Use simulation mode
        acq = AcquisitionSystem(backend='simulation')

        # Use serial connection
        acq = AcquisitionSystem(backend='serial', port='/dev/ttyUSB0')

        # Use LSL inlet
        acq = AcquisitionSystem(backend='lsl', stream_name='RootstarEEG')
    """

    SUPPORTED_BACKENDS = ('simulation', 'serial', 'lsl', 'brainflow')

    def __init__(
        self,
        eeg_channels: int = 8,
        fnirs_channels: int = 4,
        backend: str = 'simulation',
        eeg_sample_rate: float = 500.0,
        fnirs_sample_rate: float = 25.0,
        **kwargs,
    ):
        self.eeg_channels = eeg_channels
        self.fnirs_channels = fnirs_channels
        self.backend = backend
        self.eeg_sample_rate = eeg_sample_rate
        self.fnirs_sample_rate = fnirs_sample_rate
        self._running = False
        self._backend_config = kwargs

        # Data buffers
        self._eeg_buffer: Deque[np.ndarray] = deque(maxlen=int(eeg_sample_rate * 30))
        self._fnirs_buffer: Deque[np.ndarray] = deque(maxlen=int(fnirs_sample_rate * 30))

        # Backend-specific handles
        self._serial_port = None
        self._lsl_inlet = None
        self._brainflow_board = None
        self._acquisition_task = None

        # Simulation parameters
        self._sim_alpha_power = kwargs.get('alpha_power', 0.5)
        self._sim_noise_level = kwargs.get('noise_level', 0.1)
        self._sim_time = 0.0

    async def start(self) -> None:
        """Start data acquisition from configured backend."""
        if self._running:
            return

        self._running = True

        if self.backend == 'serial':
            await self._start_serial()
        elif self.backend == 'lsl':
            await self._start_lsl()
        elif self.backend == 'brainflow':
            await self._start_brainflow()
        else:  # simulation
            self._acquisition_task = asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        """Stop data acquisition."""
        self._running = False

        if self._acquisition_task:
            self._acquisition_task.cancel()
            try:
                await self._acquisition_task
            except asyncio.CancelledError:
                pass

        if self._serial_port:
            self._serial_port.close()
            self._serial_port = None

        if self._lsl_inlet:
            self._lsl_inlet = None

        if self._brainflow_board:
            try:
                self._brainflow_board.stop_stream()
                self._brainflow_board.release_session()
            except Exception:
                pass
            self._brainflow_board = None

    async def _start_serial(self) -> None:
        """Initialize serial port connection."""
        try:
            import serial
            port = self._backend_config.get('port', '/dev/ttyUSB0')
            baud = self._backend_config.get('baud', 921600)
            self._serial_port = serial.Serial(port, baud, timeout=0.1)
            self._acquisition_task = asyncio.create_task(self._serial_read_loop())
        except ImportError:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")

    async def _start_lsl(self) -> None:
        """Initialize LSL inlet connection."""
        try:
            import pylsl
            stream_name = self._backend_config.get('stream_name', 'RootstarEEG')
            streams = pylsl.resolve_stream('name', stream_name, timeout=5.0)
            if not streams:
                raise RuntimeError(f"No LSL stream found with name: {stream_name}")
            self._lsl_inlet = pylsl.StreamInlet(streams[0])
            self._acquisition_task = asyncio.create_task(self._lsl_read_loop())
        except ImportError:
            raise RuntimeError("pylsl not installed. Run: pip install pylsl")

    async def _start_brainflow(self) -> None:
        """Initialize BrainFlow board connection."""
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            board_id = self._backend_config.get('board_id', BoardIds.SYNTHETIC_BOARD)
            params = BrainFlowInputParams()
            if 'serial_port' in self._backend_config:
                params.serial_port = self._backend_config['serial_port']
            self._brainflow_board = BoardShim(board_id, params)
            self._brainflow_board.prepare_session()
            self._brainflow_board.start_stream()
            self._acquisition_task = asyncio.create_task(self._brainflow_read_loop())
        except ImportError:
            raise RuntimeError("brainflow not installed. Run: pip install brainflow")

    async def _simulation_loop(self) -> None:
        """Generate realistic simulated EEG/fNIRS data."""
        dt = 1.0 / self.eeg_sample_rate

        while self._running:
            # Generate EEG with realistic frequency components
            eeg_sample = np.zeros(self.eeg_channels)
            for ch in range(self.eeg_channels):
                # Alpha rhythm (8-13 Hz)
                alpha = self._sim_alpha_power * np.sin(
                    2 * np.pi * 10 * self._sim_time + ch * 0.5
                )
                # Beta rhythm (13-30 Hz)
                beta = 0.3 * np.sin(2 * np.pi * 20 * self._sim_time + ch * 0.3)
                # Theta rhythm (4-8 Hz)
                theta = 0.2 * np.sin(2 * np.pi * 6 * self._sim_time + ch * 0.7)
                # Pink noise
                noise = self._sim_noise_level * np.random.randn()
                eeg_sample[ch] = alpha + beta + theta + noise

            self._eeg_buffer.append(eeg_sample)

            # Generate fNIRS at lower rate (hemodynamic response)
            fnirs_idx_now = int(self._sim_time * self.fnirs_sample_rate)
            fnirs_idx_prev = int((self._sim_time - dt) * self.fnirs_sample_rate)
            if fnirs_idx_now > fnirs_idx_prev:
                fnirs_sample = np.zeros((self.fnirs_channels, 2))
                for ch in range(self.fnirs_channels):
                    # HbO2 with slow hemodynamic oscillations
                    hbo = 0.5 + 0.2 * np.sin(2 * np.pi * 0.1 * self._sim_time + ch)
                    # HbR typically anticorrelated
                    hbr = -0.15 - 0.08 * np.sin(2 * np.pi * 0.1 * self._sim_time + ch)
                    fnirs_sample[ch, 0] = hbo + 0.05 * np.random.randn()
                    fnirs_sample[ch, 1] = hbr + 0.02 * np.random.randn()
                self._fnirs_buffer.append(fnirs_sample)

            self._sim_time += dt
            await asyncio.sleep(dt)

    async def _serial_read_loop(self) -> None:
        """Read data from serial port."""
        while self._running and self._serial_port:
            try:
                if self._serial_port.in_waiting >= 2 + self.eeg_channels * 3:
                    # Read sync bytes
                    sync = self._serial_port.read(2)
                    if sync == b'\xAA\x55':
                        # Read EEG data (24-bit per channel)
                        raw = self._serial_port.read(self.eeg_channels * 3)
                        eeg_sample = np.zeros(self.eeg_channels)
                        for i in range(self.eeg_channels):
                            # Convert 24-bit signed to float
                            val = int.from_bytes(raw[i*3:(i+1)*3], 'big', signed=True)
                            eeg_sample[i] = val * 0.0223e-6  # Convert to microvolts
                        self._eeg_buffer.append(eeg_sample)
                else:
                    await asyncio.sleep(0.001)
            except Exception:
                await asyncio.sleep(0.01)

    async def _lsl_read_loop(self) -> None:
        """Read data from LSL inlet."""
        while self._running and self._lsl_inlet:
            try:
                sample, _ = self._lsl_inlet.pull_sample(timeout=0.1)
                if sample:
                    self._eeg_buffer.append(np.array(sample[:self.eeg_channels]))
            except Exception:
                await asyncio.sleep(0.01)

    async def _brainflow_read_loop(self) -> None:
        """Read data from BrainFlow board."""
        while self._running and self._brainflow_board:
            try:
                data = self._brainflow_board.get_board_data(256)
                if data.size > 0:
                    eeg_channels = self._brainflow_board.get_eeg_channels(
                        self._brainflow_board.board_id
                    )
                    for i in range(data.shape[1]):
                        eeg_sample = data[eeg_channels[:self.eeg_channels], i]
                        self._eeg_buffer.append(eeg_sample)
                await asyncio.sleep(0.05)
            except Exception:
                await asyncio.sleep(0.1)

    async def get_current_data(
        self,
        duration_s: float = 0.1,
    ) -> tuple:
        """Get current EEG and fNIRS data.

        Args:
            duration_s: Duration of data to acquire in seconds

        Returns:
            Tuple of (eeg_data, fnirs_data) where:
            - eeg_data: ndarray of shape (n_channels, n_samples)
            - fnirs_data: ndarray of shape (n_channels, 2, n_samples) for HbO2/HbR
        """
        n_eeg_samples = int(duration_s * self.eeg_sample_rate)
        n_fnirs_samples = int(duration_s * self.fnirs_sample_rate)

        # Wait for enough data
        start_time = time.time()
        while len(self._eeg_buffer) < n_eeg_samples:
            if time.time() - start_time > duration_s + 1.0:
                break  # Timeout
            await asyncio.sleep(0.01)

        # Extract data from buffers
        eeg_samples = list(self._eeg_buffer)[-n_eeg_samples:]
        fnirs_samples = list(self._fnirs_buffer)[-n_fnirs_samples:]

        # Convert to arrays
        if eeg_samples:
            eeg_data = np.array(eeg_samples).T  # (channels, samples)
        else:
            eeg_data = np.zeros((self.eeg_channels, n_eeg_samples))

        if fnirs_samples:
            fnirs_arr = np.array(fnirs_samples)  # (samples, channels, 2)
            fnirs_data = fnirs_arr.transpose(1, 2, 0)  # (channels, 2, samples)
        else:
            fnirs_data = np.zeros((self.fnirs_channels, 2, n_fnirs_samples))

        return eeg_data, fnirs_data

    @property
    def is_running(self) -> bool:
        """Check if acquisition is running."""
        return self._running


class StimulationSystem:
    """Interface to transcranial stimulation hardware.

    Supports multiple backends:
    - 'simulation': Simulate stimulation for testing
    - 'serial': Direct serial connection to Rootstar BCI stimulator
    - 'neuroconn': Neuroconn DC-STIMULATOR (requires serial)
    - 'starstim': Neuroelectrics Starstim (requires NIC2 API)

    Safety Features:
    - Software current limits (2mA max)
    - Duration limits (30 min max)
    - Impedance monitoring
    - Automatic ramp-up/down

    Example:
        stim = StimulationSystem(backend='serial', port='/dev/ttyUSB1')
        await stim.apply_protocol(protocol)
        await stim.ramp_down()
    """

    SAFETY_MAX_CURRENT_UA = 2000
    SAFETY_MAX_DURATION_S = 1800  # 30 minutes
    SAFETY_MAX_FREQUENCY_HZ = 100

    def __init__(
        self,
        n_channels: int = 8,
        backend: str = 'simulation',
        **kwargs,
    ):
        self.n_channels = n_channels
        self.backend = backend
        self._backend_config = kwargs
        self._active = False
        self._current_protocol: Optional[StimulationProtocol] = None
        self._start_time: float = 0.0
        self._current_amplitude: float = 0.0

        # Hardware handles
        self._serial_port = None
        self._ramp_task = None

        # Impedance tracking
        self._electrode_impedances: Dict[str, float] = {}
        self._impedance_ok = True

    async def initialize(self) -> None:
        """Initialize hardware connection."""
        if self.backend == 'serial':
            try:
                import serial
                port = self._backend_config.get('port', '/dev/ttyUSB1')
                baud = self._backend_config.get('baud', 115200)
                self._serial_port = serial.Serial(port, baud, timeout=1.0)
                # Send initialization command
                self._serial_port.write(b'\xAA\x01\x00')  # Init command
                await asyncio.sleep(0.1)
            except ImportError:
                raise RuntimeError("pyserial not installed")

    async def measure_impedance(self) -> Dict[str, float]:
        """Measure electrode impedances.

        Returns:
            Dict mapping electrode names to impedance in kOhm
        """
        if self.backend == 'simulation':
            # Simulate reasonable impedances
            electrodes = ['Cz', 'Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1']
            self._electrode_impedances = {
                e: 5.0 + np.random.rand() * 10.0 for e in electrodes[:self.n_channels]
            }
        elif self._serial_port:
            # Send impedance measurement command
            self._serial_port.write(b'\xAA\x02\x00')  # Impedance command
            await asyncio.sleep(0.5)
            # Parse response (simplified)
            response = self._serial_port.read(self.n_channels * 2)
            electrodes = ['Cz', 'Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1']
            for i in range(min(len(response) // 2, self.n_channels)):
                val = int.from_bytes(response[i*2:(i+1)*2], 'little')
                self._electrode_impedances[electrodes[i]] = val / 100.0  # Convert to kOhm

        # Check impedance limits (typically < 25 kOhm)
        self._impedance_ok = all(z < 25.0 for z in self._electrode_impedances.values())
        return self._electrode_impedances

    async def apply_protocol(self, protocol: StimulationProtocol) -> None:
        """Apply stimulation protocol with safety checks.

        Args:
            protocol: Stimulation parameters

        Raises:
            ValueError: If protocol fails safety validation
            RuntimeError: If impedance check fails
        """
        # Safety validation
        if not protocol.validate():
            raise ValueError("Protocol fails safety validation")

        if protocol.current_amplitude_ua > self.SAFETY_MAX_CURRENT_UA:
            raise ValueError(
                f"Current {protocol.current_amplitude_ua} µA exceeds max {self.SAFETY_MAX_CURRENT_UA} µA"
            )

        if protocol.stimulation_duration_s > self.SAFETY_MAX_DURATION_S:
            raise ValueError(
                f"Duration {protocol.stimulation_duration_s}s exceeds max {self.SAFETY_MAX_DURATION_S}s"
            )

        # Check impedance before starting
        if not self._electrode_impedances:
            await self.measure_impedance()

        if not self._impedance_ok:
            raise RuntimeError("Electrode impedance too high. Check electrode placement.")

        self._current_protocol = protocol
        self._start_time = time.time()

        # Start ramp-up
        self._ramp_task = asyncio.create_task(
            self._ramp_amplitude(0, protocol.current_amplitude_ua, protocol.ramp_up_s)
        )
        await self._ramp_task

        self._active = True

        # Schedule automatic stop after duration
        asyncio.create_task(self._auto_stop(protocol.stimulation_duration_s))

    async def _ramp_amplitude(
        self,
        start_ua: float,
        end_ua: float,
        duration_s: float,
    ) -> None:
        """Gradually change stimulation amplitude."""
        steps = max(int(duration_s * 10), 1)  # 10 Hz update rate
        step_time = duration_s / steps

        for i in range(steps + 1):
            if not self._active and end_ua > 0:
                break  # Cancelled during ramp-up
            progress = i / steps
            self._current_amplitude = start_ua + (end_ua - start_ua) * progress

            if self._serial_port:
                # Send amplitude update command
                amp_bytes = int(self._current_amplitude).to_bytes(2, 'little')
                self._serial_port.write(b'\xAA\x03' + amp_bytes)

            await asyncio.sleep(step_time)

    async def _auto_stop(self, duration_s: float) -> None:
        """Automatically stop after duration expires."""
        await asyncio.sleep(duration_s)
        if self._active:
            await self.ramp_down()

    async def ramp_down(self) -> None:
        """Gradually reduce stimulation to zero."""
        if self._current_protocol and self._active:
            self._active = False  # Prevent further stimulation
            await self._ramp_amplitude(
                self._current_amplitude,
                0,
                self._current_protocol.ramp_down_s
            )
        self._current_amplitude = 0
        self._current_protocol = None

    async def emergency_stop(self) -> None:
        """Immediately stop all stimulation (no ramp)."""
        self._active = False
        self._current_amplitude = 0

        if self._ramp_task:
            self._ramp_task.cancel()

        if self._serial_port:
            # Send emergency stop command
            self._serial_port.write(b'\xAA\xFF\x00')

        self._current_protocol = None

    @property
    def is_active(self) -> bool:
        """Check if stimulation is active."""
        return self._active

    @property
    def current_amplitude_ua(self) -> float:
        """Get current stimulation amplitude."""
        return self._current_amplitude

    @property
    def elapsed_time_s(self) -> float:
        """Get elapsed stimulation time."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    async def close(self) -> None:
        """Clean up hardware connection."""
        await self.emergency_stop()
        if self._serial_port:
            self._serial_port.close()
            self._serial_port = None


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
        """Extract embedding from current neural data using feature extraction.

        Extracts multi-scale features from EEG and fNIRS data:
        - EEG: Band power (delta, theta, alpha, beta, gamma), entropy, asymmetry
        - fNIRS: HbO2/HbR activation, temporal dynamics

        Args:
            eeg_data: EEG data of shape (n_channels, n_samples)
            fnirs_data: fNIRS data of shape (n_channels, 2, n_samples)

        Returns:
            Normalized embedding vector of shape (256,)
        """
        features = []

        # EEG Feature Extraction
        n_eeg_channels = eeg_data.shape[0]
        n_eeg_samples = eeg_data.shape[1]

        # 1. Band power using Welch's method
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
        }

        fs = self.acquisition.eeg_sample_rate
        for ch in range(n_eeg_channels):
            signal = eeg_data[ch]

            # Compute power spectral density
            nperseg = min(256, n_eeg_samples)
            freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
            fft_vals = np.abs(np.fft.rfft(signal[:nperseg])) ** 2

            # Extract band powers
            total_power = np.sum(fft_vals) + 1e-10
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(fft_vals[mask]) / total_power
                features.append(band_power)

        # 2. Spectral entropy per channel
        for ch in range(n_eeg_channels):
            signal = eeg_data[ch]
            nperseg = min(256, n_eeg_samples)
            psd = np.abs(np.fft.rfft(signal[:nperseg])) ** 2
            psd_norm = psd / (np.sum(psd) + 1e-10)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            features.append(entropy)

        # 3. Hemispheric asymmetry (assuming standard 10-20: Fp1/Fp2, C3/C4, O1/O2)
        if n_eeg_channels >= 8:
            # Alpha asymmetry for frontal (index 0,1) and central (2,3)
            for left, right in [(0, 1), (2, 3)]:
                left_power = np.var(eeg_data[left])
                right_power = np.var(eeg_data[right])
                asymmetry = np.log(right_power + 1e-10) - np.log(left_power + 1e-10)
                features.append(asymmetry)

        # fNIRS Feature Extraction
        n_fnirs_channels = fnirs_data.shape[0]

        # 4. HbO2 and HbR activation levels
        for ch in range(n_fnirs_channels):
            hbo = fnirs_data[ch, 0]  # HbO2
            hbr = fnirs_data[ch, 1]  # HbR

            features.append(np.mean(hbo))
            features.append(np.std(hbo))
            features.append(np.mean(hbr))
            features.append(np.std(hbr))

            # Oxygenation index
            hbt = np.abs(hbo) + np.abs(hbr) + 1e-10
            features.append(np.mean(hbo / hbt))

        # 5. Temporal dynamics (slope of HbO2)
        for ch in range(n_fnirs_channels):
            hbo = fnirs_data[ch, 0]
            if len(hbo) > 1:
                slope = np.polyfit(np.arange(len(hbo)), hbo, 1)[0]
            else:
                slope = 0.0
            features.append(slope)

        # Convert to numpy array
        features = np.array(features, dtype=np.float32)

        # Pad or truncate to embedding dimension (256)
        embedding_dim = 256
        if len(features) < embedding_dim:
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            embedding[:len(features)] = features
        else:
            embedding = features[:embedding_dim]

        # L2 normalize
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

        Uses an inverse model approach based on:
        1. Embedding difference analysis to determine target brain regions
        2. Frequency selection based on desired neural oscillation changes
        3. Electrode selection based on target cortical areas
        4. Amplitude scaling based on distance to target

        The model maps embedding differences to stimulation montages:
        - Frontal features (indices 0-15) → Fp1/Fp2 montage
        - Central features (indices 16-31) → C3/C4/Cz montage
        - Posterior features (indices 32-47) → O1/O2 montage
        - fNIRS features → HD-tDCS spatial patterns

        Args:
            target_embedding: Target neural fingerprint embedding
            current_embedding: Current measured embedding

        Returns:
            StimulationProtocol with computed parameters
        """
        delta = target_embedding - current_embedding
        magnitude = np.linalg.norm(delta)

        # Analyze which embedding components need adjustment
        # Split embedding into regions (assuming 256-dim with 8 EEG + 4 fNIRS channels)
        # Each EEG channel has 5 bands + entropy = 6 features = 48 total
        # fNIRS has 5 features per channel = 20 total

        frontal_delta = np.linalg.norm(delta[:12])  # First 2 channels
        central_delta = np.linalg.norm(delta[12:24])  # Middle channels
        posterior_delta = np.linalg.norm(delta[24:48])  # Back channels
        fnirs_delta = np.linalg.norm(delta[48:68])  # fNIRS features

        # Determine primary electrode montage based on largest delta
        deltas = {
            'frontal': frontal_delta,
            'central': central_delta,
            'posterior': posterior_delta,
        }
        primary_region = max(deltas, key=deltas.get)

        # Select electrodes based on region
        electrode_configs = {
            'frontal': (['Fp1', 'Fp2'], ['Cz']),
            'central': (['Cz'], ['Fp1', 'Fp2']),
            'posterior': (['O1', 'O2'], ['Cz']),
        }
        anode, cathode = electrode_configs[primary_region]

        # Determine frequency based on band power changes needed
        # Analyze which frequency bands need most change
        band_indices = {
            'delta': 0,
            'theta': 1,
            'alpha': 2,
            'beta': 3,
            'gamma': 4,
        }

        # Calculate average change needed per band across channels
        band_changes = {}
        for band, idx in band_indices.items():
            band_delta = 0.0
            for ch in range(8):
                feature_idx = ch * 6 + idx  # 6 features per channel
                if feature_idx < len(delta):
                    band_delta += abs(delta[feature_idx])
            band_changes[band] = band_delta

        # Select stimulation frequency based on target band
        dominant_band = max(band_changes, key=band_changes.get)
        frequency_map = {
            'delta': 0.0,  # tDCS (no oscillation)
            'theta': 6.0,
            'alpha': 10.0,
            'beta': 20.0,
            'gamma': 40.0,
        }
        stim_frequency = frequency_map[dominant_band]

        # Determine waveform
        if stim_frequency == 0:
            waveform = "DC"
        else:
            waveform = "AC"

        # Scale amplitude based on distance (larger delta = higher amplitude)
        # Use sigmoid-like scaling for smooth control
        base_amplitude = 500  # Minimum amplitude
        max_amplitude = 1500  # Maximum amplitude
        scaling_factor = 1.0 / (1.0 + np.exp(-5 * (magnitude - 0.5)))
        amplitude = base_amplitude + (max_amplitude - base_amplitude) * scaling_factor

        # Adjust amplitude based on fNIRS activation needs
        # If fNIRS shows we need more hemodynamic response, increase duration
        if fnirs_delta > 0.1:
            stimulation_duration = 1.0  # Longer for hemodynamic effects
        else:
            stimulation_duration = 0.5

        # Add polarity based on direction of change needed
        # Positive delta means we need to increase, negative means decrease
        mean_delta = np.mean(delta[:48])  # EEG features
        if mean_delta < 0:
            # Swap anode/cathode for opposite effect
            anode, cathode = cathode, anode

        return StimulationProtocol(
            current_amplitude_ua=float(amplitude),
            waveform=waveform,
            frequency_hz=stim_frequency if waveform == "AC" else None,
            ramp_up_s=0.3,
            stimulation_duration_s=stimulation_duration,
            ramp_down_s=0.2,
            anode_electrodes=anode,
            cathode_electrodes=cathode,
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

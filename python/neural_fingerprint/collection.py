"""Data collection management for building fingerprint libraries.

This module provides tools for systematic collection of neural fingerprints
from real sensory stimuli. It handles:
- Stimulus presentation protocols
- Neural data acquisition and storage
- Trial management and randomization
- Quality control and averaging
"""

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .controller import AcquisitionSystem
from .database import FingerprintDatabase


@dataclass
class StimulusSession:
    """Protocol configuration for collecting neural fingerprints.

    Defines timing parameters and stimulus presentation order for
    a data collection session.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject_id: str = ""
    modality: str = "gustatory"
    stimuli: List[str] = field(default_factory=list)

    # Timing parameters (seconds)
    baseline_duration_s: float = 10.0
    stimulus_duration_s: float = 5.0
    washout_duration_s: float = 30.0

    # Repetitions
    repetitions_per_stimulus: int = 10

    # Randomization
    randomize_order: bool = True
    include_catch_trials: bool = True
    catch_trial_fraction: float = 0.1  # 10% catch trials

    # Callbacks
    on_trial_start: Optional[Callable[[str, int], None]] = None
    on_trial_end: Optional[Callable[[str, int, float], None]] = None
    on_session_progress: Optional[Callable[[float], None]] = None


@dataclass
class TrialData:
    """Data from a single stimulus trial."""

    trial_id: str
    stimulus_label: str
    trial_number: int
    timestamp: float

    # Raw data
    baseline_eeg: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_fnirs: np.ndarray = field(default_factory=lambda: np.array([]))
    response_eeg: np.ndarray = field(default_factory=lambda: np.array([]))
    response_fnirs: np.ndarray = field(default_factory=lambda: np.array([]))

    # Quality metrics
    signal_quality: float = 0.0
    is_catch_trial: bool = False
    subject_rating: Optional[int] = None  # 1-5 subjective rating


@dataclass
class CollectionResults:
    """Results from a data collection session."""

    session_id: str
    subject_id: str
    modality: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Trial data
    trials: List[TrialData] = field(default_factory=list)

    # Summary statistics
    total_trials: int = 0
    valid_trials: int = 0
    catch_trial_accuracy: float = 0.0
    avg_signal_quality: float = 0.0

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "modality": self.modality,
            "total_trials": self.total_trials,
            "valid_trials": self.valid_trials,
            "catch_accuracy": self.catch_trial_accuracy,
            "avg_quality": self.avg_signal_quality,
            "duration_min": (
                (self.end_time - self.start_time).total_seconds() / 60
                if self.end_time
                else 0
            ),
        }


class StimulusPresenter:
    """Interface for presenting sensory stimuli.

    Supports multiple backends for stimulus delivery:
    - 'simulation': Print messages (testing)
    - 'serial': Serial-controlled stimulus delivery
    - 'psychopy': PsychoPy for visual/auditory stimuli
    - 'gustometer': Taste delivery system (serial)
    - 'olfactometer': Smell delivery system (serial)

    Hardware Protocol (Serial):
    - Present: 0xAA <channel> <intensity_byte> <duration_ms_16bit>
    - Clear: 0xAA 0xFF 0x00 0x00 0x00

    Example:
        # Gustatory with serial hardware
        presenter = StimulusPresenter(modality='gustatory', backend='serial', port='/dev/ttyUSB2')

        # Visual with PsychoPy
        presenter = StimulusPresenter(modality='visual', backend='psychopy')
    """

    # Stimulus channel mappings for gustatory
    TASTE_CHANNELS = {
        'sweet': 0,
        'salty': 1,
        'sour': 2,
        'bitter': 3,
        'umami': 4,
        'water': 5,  # Rinse
    }

    # Stimulus channel mappings for olfactory
    ODOR_CHANNELS = {
        'rose': 0,
        'lemon': 1,
        'coffee': 2,
        'vanilla': 3,
        'peppermint': 4,
        'clean_air': 5,  # Rinse
    }

    def __init__(
        self,
        modality: str = "gustatory",
        backend: str = "simulation",
        **kwargs,
    ):
        self.modality = modality
        self.backend = backend
        self._backend_config = kwargs
        self._current_stimulus: Optional[str] = None
        self._serial_port = None
        self._psychopy_win = None

        # Channel mappings based on modality
        if modality == "gustatory":
            self._channel_map = self.TASTE_CHANNELS
        elif modality == "olfactory":
            self._channel_map = self.ODOR_CHANNELS
        else:
            self._channel_map = {}

    async def initialize(self) -> None:
        """Initialize hardware connection."""
        if self.backend == 'serial':
            try:
                import serial
                port = self._backend_config.get('port', '/dev/ttyUSB2')
                baud = self._backend_config.get('baud', 9600)
                self._serial_port = serial.Serial(port, baud, timeout=1.0)
                await asyncio.sleep(0.5)  # Wait for hardware init
            except ImportError:
                raise RuntimeError("pyserial not installed")
        elif self.backend == 'psychopy':
            try:
                from psychopy import visual
                self._psychopy_win = visual.Window(
                    size=self._backend_config.get('size', (800, 600)),
                    fullscr=self._backend_config.get('fullscreen', False),
                )
            except ImportError:
                raise RuntimeError("psychopy not installed")

    async def present(
        self,
        stimulus: str,
        intensity: float = 1.0,
        duration_ms: int = 2000,
    ) -> None:
        """Present a sensory stimulus.

        Args:
            stimulus: Stimulus identifier (e.g., 'sweet', 'rose')
            intensity: Intensity level (0-1)
            duration_ms: Presentation duration in milliseconds
        """
        self._current_stimulus = stimulus

        if self.backend == 'simulation':
            print(f"  [STIMULUS] Presenting: {stimulus} ({self.modality}) "
                  f"intensity={intensity:.2f}")
            await asyncio.sleep(duration_ms / 1000.0)

        elif self.backend == 'serial' and self._serial_port:
            channel = self._channel_map.get(stimulus.lower(), 0)
            intensity_byte = int(intensity * 255)

            # Build command: 0xAA <channel> <intensity> <duration_lo> <duration_hi>
            cmd = bytes([
                0xAA,
                channel,
                intensity_byte,
                duration_ms & 0xFF,
                (duration_ms >> 8) & 0xFF,
            ])
            self._serial_port.write(cmd)
            await asyncio.sleep(duration_ms / 1000.0)

        elif self.backend == 'psychopy' and self._psychopy_win:
            await self._present_psychopy(stimulus, intensity, duration_ms)

    async def _present_psychopy(
        self,
        stimulus: str,
        intensity: float,
        duration_ms: int,
    ) -> None:
        """Present stimulus using PsychoPy."""
        from psychopy import visual, core

        if self.modality == 'visual':
            # Create visual stimulus (e.g., colored square, image)
            stim = visual.Rect(
                self._psychopy_win,
                width=0.5, height=0.5,
                fillColor=[intensity, intensity, intensity],
            )
            stim.draw()
            self._psychopy_win.flip()
            await asyncio.sleep(duration_ms / 1000.0)
            self._psychopy_win.flip()  # Clear

        elif self.modality == 'auditory':
            # Play audio stimulus
            from psychopy import sound
            freq = self._backend_config.get('frequencies', {}).get(stimulus, 440)
            tone = sound.Sound(value=freq, secs=duration_ms / 1000.0)
            tone.play()
            await asyncio.sleep(duration_ms / 1000.0)

    async def clear(self) -> None:
        """Clear/wash out current stimulus."""
        self._current_stimulus = None

        if self.backend == 'simulation':
            print("  [STIMULUS] Cleared")
            await asyncio.sleep(0.1)

        elif self.backend == 'serial' and self._serial_port:
            # Send clear/rinse command
            if self.modality == 'gustatory':
                rinse_channel = self._channel_map.get('water', 5)
            else:
                rinse_channel = self._channel_map.get('clean_air', 5)

            cmd = bytes([0xAA, rinse_channel, 128, 0xE8, 0x03])  # 1000ms rinse
            self._serial_port.write(cmd)
            await asyncio.sleep(1.0)

            # Stop all
            self._serial_port.write(bytes([0xAA, 0xFF, 0x00, 0x00, 0x00]))
            await asyncio.sleep(0.1)

        elif self.backend == 'psychopy' and self._psychopy_win:
            self._psychopy_win.flip()  # Clear screen
            await asyncio.sleep(0.1)

    @property
    def current_stimulus(self) -> Optional[str]:
        """Get currently presented stimulus."""
        return self._current_stimulus

    async def close(self) -> None:
        """Clean up resources."""
        await self.clear()
        if self._serial_port:
            self._serial_port.close()
            self._serial_port = None
        if self._psychopy_win:
            self._psychopy_win.close()
            self._psychopy_win = None


class DataCollectionManager:
    """Manages collection of training data for fingerprint library.

    Coordinates stimulus presentation, neural acquisition, and data
    storage for systematic collection of neural fingerprints.

    Args:
        acquisition_system: Neural acquisition interface
        database: Fingerprint database for storage
        presenter: Stimulus presentation interface
    """

    def __init__(
        self,
        acquisition_system: Optional[AcquisitionSystem] = None,
        database: Optional[FingerprintDatabase] = None,
        presenter: Optional[StimulusPresenter] = None,
    ):
        self.acquisition = acquisition_system or AcquisitionSystem()
        self.database = database
        self.presenter = presenter or StimulusPresenter()

        self._current_session: Optional[StimulusSession] = None
        self._results: Optional[CollectionResults] = None
        self._running = False

    async def run_collection_session(
        self,
        session: StimulusSession,
    ) -> CollectionResults:
        """Run full data collection session.

        For each stimulus:
        1. Record baseline
        2. Present stimulus
        3. Record neural response
        4. Store fingerprint
        5. Washout period

        Args:
            session: Session configuration

        Returns:
            Collection results with all trials
        """
        self._current_session = session
        self._running = True

        # Initialize results
        self._results = CollectionResults(
            session_id=session.session_id,
            subject_id=session.subject_id,
            modality=session.modality,
            start_time=datetime.now(),
        )

        # Build trial list
        trial_list = self._build_trial_list(session)
        total_trials = len(trial_list)

        print(f"\n{'='*60}")
        print(f"Starting Collection Session: {session.session_id}")
        print(f"Subject: {session.subject_id}")
        print(f"Modality: {session.modality}")
        print(f"Stimuli: {', '.join(session.stimuli)}")
        print(f"Total trials: {total_trials}")
        print(f"{'='*60}\n")

        # Start acquisition
        await self.acquisition.start()

        try:
            for trial_idx, (stimulus, rep, is_catch) in enumerate(trial_list):
                if not self._running:
                    break

                # Update progress
                progress = (trial_idx + 1) / total_trials
                if session.on_session_progress:
                    session.on_session_progress(progress)

                # Run single trial
                trial_data = await self._run_trial(
                    stimulus=stimulus,
                    trial_number=rep,
                    is_catch=is_catch,
                    session=session,
                )

                self._results.trials.append(trial_data)
                self._results.total_trials += 1

                if trial_data.signal_quality > 0.5:
                    self._results.valid_trials += 1

                # Print progress
                print(
                    f"  Trial {trial_idx + 1}/{total_trials}: "
                    f"{stimulus} (rep {rep}) - "
                    f"Quality: {trial_data.signal_quality:.2f}"
                )

        finally:
            await self.acquisition.stop()

        # Compute final statistics
        self._results.end_time = datetime.now()
        self._compute_session_stats()

        # Store averaged fingerprints
        if self.database:
            await self._store_averaged_fingerprints()

        print(f"\n{'='*60}")
        print("Session Complete!")
        print(f"Valid trials: {self._results.valid_trials}/{self._results.total_trials}")
        print(f"Average quality: {self._results.avg_signal_quality:.2f}")
        print(f"{'='*60}\n")

        return self._results

    async def _run_trial(
        self,
        stimulus: str,
        trial_number: int,
        is_catch: bool,
        session: StimulusSession,
    ) -> TrialData:
        """Run a single data collection trial.

        Args:
            stimulus: Stimulus to present
            trial_number: Repetition number
            is_catch: Whether this is a catch trial
            session: Session configuration

        Returns:
            Trial data with neural recordings
        """
        trial_id = f"{session.session_id}_{stimulus}_{trial_number}"

        trial = TrialData(
            trial_id=trial_id,
            stimulus_label=stimulus,
            trial_number=trial_number,
            timestamp=time.time(),
            is_catch_trial=is_catch,
        )

        # Callback: trial start
        if session.on_trial_start:
            session.on_trial_start(stimulus, trial_number)

        # 1. Record baseline
        print(f"    Recording baseline ({session.baseline_duration_s}s)...")
        baseline_eeg, baseline_fnirs = await self.acquisition.get_current_data(
            duration_s=session.baseline_duration_s
        )
        trial.baseline_eeg = baseline_eeg
        trial.baseline_fnirs = baseline_fnirs

        # 2. Present stimulus (or nothing for catch trial)
        if not is_catch:
            print(f"    Presenting: {stimulus}")
            await self.presenter.present(stimulus)

        # 3. Record response
        print(f"    Recording response ({session.stimulus_duration_s}s)...")
        response_eeg, response_fnirs = await self.acquisition.get_current_data(
            duration_s=session.stimulus_duration_s
        )
        trial.response_eeg = response_eeg
        trial.response_fnirs = response_fnirs

        # 4. Compute signal quality
        trial.signal_quality = self._compute_signal_quality(
            baseline_eeg, response_eeg
        )

        # 5. Clear stimulus and washout
        await self.presenter.clear()
        print(f"    Washout ({session.washout_duration_s}s)...")
        await asyncio.sleep(session.washout_duration_s)

        # Callback: trial end
        if session.on_trial_end:
            session.on_trial_end(stimulus, trial_number, trial.signal_quality)

        return trial

    def _build_trial_list(
        self,
        session: StimulusSession,
    ) -> List[tuple]:
        """Build randomized list of trials.

        Returns:
            List of (stimulus, repetition, is_catch) tuples
        """
        trials = []

        # Add regular trials
        for stimulus in session.stimuli:
            for rep in range(session.repetitions_per_stimulus):
                trials.append((stimulus, rep, False))

        # Add catch trials
        if session.include_catch_trials:
            n_catch = int(len(trials) * session.catch_trial_fraction)
            for i in range(n_catch):
                trials.append(("CATCH", i, True))

        # Randomize
        if session.randomize_order:
            random.shuffle(trials)

        return trials

    def _compute_signal_quality(
        self,
        baseline: np.ndarray,
        response: np.ndarray,
    ) -> float:
        """Compute signal quality metric.

        Simple SNR-based quality metric.
        """
        if baseline.size == 0 or response.size == 0:
            return 0.0

        # Compute variance ratio (simplified SNR)
        baseline_var = np.var(baseline)
        response_var = np.var(response)

        if baseline_var == 0:
            return 1.0

        snr = response_var / baseline_var

        # Map to 0-1 quality score
        quality = min(1.0, snr / 10.0)

        return quality

    def _compute_session_stats(self) -> None:
        """Compute session-level statistics including catch trial accuracy."""
        if not self._results or not self._results.trials:
            return

        # Average signal quality
        qualities = [t.signal_quality for t in self._results.trials]
        self._results.avg_signal_quality = np.mean(qualities) if qualities else 0.0

        # Catch trial detection accuracy
        # Catch trials should have lower neural response (no actual stimulus)
        catch_trials = [t for t in self._results.trials if t.is_catch_trial]
        regular_trials = [t for t in self._results.trials if not t.is_catch_trial]

        if catch_trials and regular_trials:
            # Compute response magnitude for catch vs regular trials
            catch_magnitudes = []
            for t in catch_trials:
                if t.response_eeg.size > 0:
                    # Response magnitude = RMS of response minus baseline
                    response_rms = np.sqrt(np.mean(t.response_eeg ** 2))
                    baseline_rms = np.sqrt(np.mean(t.baseline_eeg ** 2)) if t.baseline_eeg.size > 0 else 0
                    catch_magnitudes.append(response_rms - baseline_rms)

            regular_magnitudes = []
            for t in regular_trials:
                if t.response_eeg.size > 0:
                    response_rms = np.sqrt(np.mean(t.response_eeg ** 2))
                    baseline_rms = np.sqrt(np.mean(t.baseline_eeg ** 2)) if t.baseline_eeg.size > 0 else 0
                    regular_magnitudes.append(response_rms - baseline_rms)

            if catch_magnitudes and regular_magnitudes:
                # Threshold: catch trials should have lower magnitude than median of regular
                threshold = np.median(regular_magnitudes) * 0.5
                correct_catches = sum(1 for m in catch_magnitudes if m < threshold)
                self._results.catch_trial_accuracy = correct_catches / len(catch_magnitudes)
            else:
                self._results.catch_trial_accuracy = 0.0
        elif catch_trials:
            # No regular trials to compare
            self._results.catch_trial_accuracy = 0.0
        else:
            # No catch trials
            self._results.catch_trial_accuracy = 1.0

    def _extract_trial_embedding(
        self,
        trial: TrialData,
        sample_rate: float = 500.0,
    ) -> np.ndarray:
        """Extract ML embedding from trial data.

        Uses the same feature extraction as the real-time controller:
        - EEG band power (delta, theta, alpha, beta, gamma)
        - Spectral entropy
        - Hemispheric asymmetry
        - fNIRS activation features

        Args:
            trial: Trial data with EEG and fNIRS recordings
            sample_rate: EEG sample rate in Hz

        Returns:
            Normalized 256-dimensional embedding vector
        """
        features = []

        # Get response data (subtract baseline for evoked response)
        eeg_data = trial.response_eeg
        fnirs_data = trial.response_fnirs

        if eeg_data.size == 0:
            return np.zeros(256, dtype=np.float32)

        n_eeg_channels = eeg_data.shape[0] if eeg_data.ndim > 1 else 1
        n_eeg_samples = eeg_data.shape[1] if eeg_data.ndim > 1 else eeg_data.size

        # Baseline correction
        if trial.baseline_eeg.size > 0:
            baseline_mean = np.mean(trial.baseline_eeg, axis=1 if trial.baseline_eeg.ndim > 1 else 0)
            if eeg_data.ndim > 1:
                eeg_data = eeg_data - baseline_mean[:, np.newaxis]
            else:
                eeg_data = eeg_data - baseline_mean

        # EEG Band Power Features
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
        }

        for ch in range(n_eeg_channels):
            signal = eeg_data[ch] if eeg_data.ndim > 1 else eeg_data

            # Compute FFT-based power spectrum
            nperseg = min(256, n_eeg_samples)
            freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate)
            fft_vals = np.abs(np.fft.rfft(signal[:nperseg])) ** 2

            total_power = np.sum(fft_vals) + 1e-10
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(fft_vals[mask]) / total_power
                features.append(band_power)

        # Spectral Entropy
        for ch in range(n_eeg_channels):
            signal = eeg_data[ch] if eeg_data.ndim > 1 else eeg_data
            nperseg = min(256, n_eeg_samples)
            psd = np.abs(np.fft.rfft(signal[:nperseg])) ** 2
            psd_norm = psd / (np.sum(psd) + 1e-10)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            features.append(entropy)

        # Hemispheric Asymmetry (if enough channels)
        if n_eeg_channels >= 4:
            for left, right in [(0, 1), (2, 3)]:
                left_power = np.var(eeg_data[left] if eeg_data.ndim > 1 else eeg_data)
                right_power = np.var(eeg_data[right] if eeg_data.ndim > 1 else eeg_data)
                asymmetry = np.log(right_power + 1e-10) - np.log(left_power + 1e-10)
                features.append(asymmetry)

        # fNIRS Features
        if fnirs_data.size > 0:
            n_fnirs_channels = fnirs_data.shape[0] if fnirs_data.ndim > 2 else 1
            for ch in range(n_fnirs_channels):
                if fnirs_data.ndim > 2:
                    hbo = fnirs_data[ch, 0]
                    hbr = fnirs_data[ch, 1]
                else:
                    hbo = fnirs_data[0] if fnirs_data.ndim > 1 else fnirs_data
                    hbr = fnirs_data[1] if fnirs_data.ndim > 1 and fnirs_data.shape[0] > 1 else np.zeros_like(hbo)

                features.extend([
                    np.mean(hbo),
                    np.std(hbo),
                    np.mean(hbr),
                    np.std(hbr),
                ])

                # Oxygenation index
                hbt = np.abs(hbo) + np.abs(hbr) + 1e-10
                features.append(np.mean(hbo / hbt))

                # Temporal slope
                if len(hbo) > 1:
                    slope = np.polyfit(np.arange(len(hbo)), hbo, 1)[0]
                else:
                    slope = 0.0
                features.append(slope)

        # Convert and pad/truncate to 256 dimensions
        features = np.array(features, dtype=np.float32)
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

    async def _store_averaged_fingerprints(self) -> None:
        """Compute and store averaged fingerprints per stimulus using ML embedding."""
        if not self._results or not self.database:
            return

        # Group trials by stimulus
        trials_by_stimulus: Dict[str, List[TrialData]] = {}
        for trial in self._results.trials:
            if trial.is_catch_trial:
                continue
            if trial.stimulus_label not in trials_by_stimulus:
                trials_by_stimulus[trial.stimulus_label] = []
            trials_by_stimulus[trial.stimulus_label].append(trial)

        # Compute average embedding for each stimulus
        for stimulus, trials in trials_by_stimulus.items():
            if not trials:
                continue

            # Extract embeddings for each trial
            embeddings = []
            for trial in trials:
                if trial.signal_quality > 0.3:  # Only use good quality trials
                    emb = self._extract_trial_embedding(trial)
                    embeddings.append(emb)

            if not embeddings:
                print(f"  Warning: No valid trials for {stimulus}")
                continue

            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

            # Re-normalize after averaging
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            # Compute confidence from embedding consistency
            # Higher consistency = embeddings are more similar = higher confidence
            if len(embeddings) > 1:
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j])
                        similarities.append(sim)
                consistency = np.mean(similarities)
            else:
                consistency = 0.5

            # Compute average quality
            avg_quality = np.mean([t.signal_quality for t in trials])

            # Combined confidence
            confidence = 0.6 * avg_quality + 0.4 * consistency

            # Store in database
            fingerprint_id = f"{self._results.session_id}_{stimulus}_avg"
            self.database.store_fingerprint(
                fingerprint_id=fingerprint_id,
                embedding=avg_embedding,
                modality=self._results.modality,
                stimulus_label=stimulus,
                subject_id=self._results.subject_id,
                confidence=confidence,
                signal_quality=avg_quality,
            )

            print(f"  Stored fingerprint: {fingerprint_id} "
                  f"(trials={len(embeddings)}, confidence={confidence:.2f})")

    def stop(self) -> None:
        """Stop the current collection session."""
        self._running = False


async def demo_data_collection():
    """Demonstrate data collection workflow."""
    import tempfile

    print("Starting Data Collection Demo...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create database
        db_path = Path(tmpdir) / "demo.db"
        db = FingerprintDatabase(str(db_path))

        # Create collection manager
        manager = DataCollectionManager(database=db)

        # Define session
        session = StimulusSession(
            subject_id="demo_subject",
            modality="gustatory",
            stimuli=["sweet", "salty"],
            baseline_duration_s=0.5,  # Short for demo
            stimulus_duration_s=0.5,
            washout_duration_s=0.5,
            repetitions_per_stimulus=2,
            include_catch_trials=False,
        )

        # Run collection
        results = await manager.run_collection_session(session)

        # Print summary
        print("\nResults Summary:")
        for key, value in results.summary().items():
            print(f"  {key}: {value}")

        db.close()

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(demo_data_collection())

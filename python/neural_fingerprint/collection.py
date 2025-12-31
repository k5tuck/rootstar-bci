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

    This is a placeholder that should be replaced with actual
    hardware interface for stimulus delivery.
    """

    def __init__(self, modality: str = "gustatory"):
        self.modality = modality
        self._current_stimulus: Optional[str] = None

    async def present(self, stimulus: str, intensity: float = 1.0) -> None:
        """Present a sensory stimulus.

        Args:
            stimulus: Stimulus identifier
            intensity: Intensity level (0-1)
        """
        self._current_stimulus = stimulus
        print(f"  [STIMULUS] Presenting: {stimulus} ({self.modality})")

        # Placeholder - actual implementation would control hardware
        # e.g., taste delivery system, olfactometer, etc.
        await asyncio.sleep(0.1)

    async def clear(self) -> None:
        """Clear/wash out current stimulus."""
        self._current_stimulus = None
        print("  [STIMULUS] Cleared")
        await asyncio.sleep(0.1)


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
        """Compute session-level statistics."""
        if not self._results or not self._results.trials:
            return

        # Average signal quality
        qualities = [t.signal_quality for t in self._results.trials]
        self._results.avg_signal_quality = np.mean(qualities) if qualities else 0.0

        # Catch trial accuracy (placeholder - would need subject responses)
        catch_trials = [t for t in self._results.trials if t.is_catch_trial]
        if catch_trials:
            # For now, assume all correctly identified
            self._results.catch_trial_accuracy = 1.0

    async def _store_averaged_fingerprints(self) -> None:
        """Compute and store averaged fingerprints per stimulus."""
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

        # Compute average for each stimulus
        for stimulus, trials in trials_by_stimulus.items():
            if not trials:
                continue

            # Average the response EEG (simplified)
            avg_eeg = np.mean([t.response_eeg for t in trials], axis=0)

            # Create simplified embedding (placeholder)
            # Real implementation would use the ML encoder
            embedding = np.mean(avg_eeg, axis=1)
            if len(embedding) < 256:
                embedding = np.pad(embedding, (0, 256 - len(embedding)))
            embedding = embedding[:256].astype(np.float32)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Compute average quality
            avg_quality = np.mean([t.signal_quality for t in trials])

            # Store in database
            fingerprint_id = f"{self._results.session_id}_{stimulus}_avg"
            self.database.store_fingerprint(
                fingerprint_id=fingerprint_id,
                embedding=embedding,
                modality=self._results.modality,
                stimulus_label=stimulus,
                subject_id=self._results.subject_id,
                confidence=avg_quality,
                signal_quality=avg_quality,
            )

            print(f"  Stored averaged fingerprint: {fingerprint_id}")

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

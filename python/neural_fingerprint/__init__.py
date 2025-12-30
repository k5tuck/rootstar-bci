"""Neural Fingerprint Detection & Sensory Simulation System.

This package provides the Python ML stack for the Neural Fingerprint system,
including:

- Deep learning models for fingerprint encoding
- Database storage with similarity search
- Real-time feedback control
- VR integration via WebSocket

Example:
    >>> from neural_fingerprint import (
    ...     NeuralFingerprintEncoder,
    ...     FingerprintDatabase,
    ...     RealTimeFeedbackController,
    ... )
    >>>
    >>> # Create encoder
    >>> encoder = NeuralFingerprintEncoder()
    >>>
    >>> # Connect to database
    >>> db = FingerprintDatabase("fingerprints.db")
"""

from .models import (
    NeuralFingerprintEncoder,
    ContrastiveFingerprintLoss,
    InverseStimulationModel,
)
from .database import (
    FingerprintRecord,
    StimulusLibrary,
    FingerprintDatabase,
    FingerprintIndex,
)
from .controller import (
    SystemState,
    BidirectionalSession,
    RealTimeFeedbackController,
)
from .vr_bridge import VRSensoryBridge
from .collection import (
    StimulusSession,
    DataCollectionManager,
)

__version__ = "0.1.0"
__all__ = [
    # Models
    "NeuralFingerprintEncoder",
    "ContrastiveFingerprintLoss",
    "InverseStimulationModel",
    # Database
    "FingerprintRecord",
    "StimulusLibrary",
    "FingerprintDatabase",
    "FingerprintIndex",
    # Controller
    "SystemState",
    "BidirectionalSession",
    "RealTimeFeedbackController",
    # VR Bridge
    "VRSensoryBridge",
    # Collection
    "StimulusSession",
    "DataCollectionManager",
]

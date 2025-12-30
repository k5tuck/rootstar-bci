"""Database storage and similarity search for neural fingerprints.

This module provides:
- SQLAlchemy models for fingerprint storage
- FAISS-based similarity search index
- High-level database interface
"""

import pickle
import zlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class FingerprintRecord(Base):
    """SQLAlchemy model for storing neural fingerprints."""

    __tablename__ = "neural_fingerprints"

    id = Column(Integer, primary_key=True)
    fingerprint_id = Column(String(64), unique=True, index=True)

    # Sensory classification
    modality = Column(String(32), index=True)  # gustatory, olfactory, etc.
    stimulus_label = Column(String(128), index=True)
    stimulus_category = Column(String(64))  # fruit, spice, floral, etc.

    # Subject info
    subject_id = Column(String(64), index=True)
    session_id = Column(String(64))
    timestamp = Column(DateTime, index=True)

    # Embedding vector (stored as compressed binary)
    embedding = Column(LargeBinary)  # Compressed numpy array
    embedding_dim = Column(Integer)

    # Compressed feature data
    features_blob = Column(LargeBinary)  # Compressed full feature dict

    # Quality metrics
    signal_quality = Column(Float)
    confidence = Column(Float)
    validation_score = Column(Float)  # Cross-validation accuracy

    # Hardware config
    eeg_channels = Column(Integer)
    fnirs_channels = Column(Integer)
    hardware_version = Column(String(32))

    def set_embedding(self, embedding: np.ndarray) -> None:
        """Compress and store embedding vector."""
        self.embedding = zlib.compress(embedding.astype(np.float32).tobytes())
        self.embedding_dim = len(embedding)

    def get_embedding(self) -> np.ndarray:
        """Decompress and return embedding vector."""
        if self.embedding is None:
            return np.array([])
        data = zlib.decompress(self.embedding)
        return np.frombuffer(data, dtype=np.float32)

    def set_features(self, features: dict) -> None:
        """Compress and store full feature dictionary."""
        self.features_blob = zlib.compress(pickle.dumps(features))

    def get_features(self) -> dict:
        """Decompress and return feature dictionary."""
        if self.features_blob is None:
            return {}
        return pickle.loads(zlib.decompress(self.features_blob))


class StimulusLibrary(Base):
    """Reference library of standard sensory stimuli."""

    __tablename__ = "stimulus_library"

    id = Column(Integer, primary_key=True)
    stimulus_label = Column(String(128), unique=True, index=True)
    modality = Column(String(32), index=True)
    category = Column(String(64))

    # Semantic descriptors (stored as comma-separated)
    descriptors = Column(String(512))  # "sweet,crisp,fresh"
    intensity_level = Column(Float)  # 0-1 normalized

    # Reference fingerprint (average across subjects)
    reference_embedding = Column(LargeBinary)
    reference_embedding_dim = Column(Integer)

    # VR Integration
    vr_asset_id = Column(String(64))
    stimulation_protocol_id = Column(String(64))

    def set_descriptors(self, desc_list: List[str]) -> None:
        """Store list of descriptors."""
        self.descriptors = ",".join(desc_list)

    def get_descriptors(self) -> List[str]:
        """Get list of descriptors."""
        if not self.descriptors:
            return []
        return self.descriptors.split(",")

    def set_reference_embedding(self, embedding: np.ndarray) -> None:
        """Store reference embedding."""
        self.reference_embedding = zlib.compress(
            embedding.astype(np.float32).tobytes()
        )
        self.reference_embedding_dim = len(embedding)

    def get_reference_embedding(self) -> np.ndarray:
        """Get reference embedding."""
        if self.reference_embedding is None:
            return np.array([])
        data = zlib.decompress(self.reference_embedding)
        return np.frombuffer(data, dtype=np.float32)


class FingerprintIndex:
    """Fast similarity search over neural fingerprint embeddings using FAISS.

    Args:
        embedding_dim: Dimension of embedding vectors (default: 256)
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.id_map: List[str] = []

        if FAISS_AVAILABLE:
            # Use Inner Product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            # Fallback to numpy-based search
            self.index = None
            self._embeddings: List[np.ndarray] = []

    def add_fingerprint(
        self,
        fingerprint_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Add a fingerprint embedding to the index.

        Args:
            fingerprint_id: Unique identifier for the fingerprint
            embedding: Embedding vector (will be L2-normalized)
        """
        embedding = embedding.reshape(1, -1).astype("float32")

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embedding)
        else:
            self._embeddings.append(embedding.flatten())

        self.id_map.append(fingerprint_id)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find k most similar fingerprints to query.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (fingerprint_id, similarity_score) tuples
        """
        query = query_embedding.reshape(1, -1).astype("float32")

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.id_map):
                    results.append((self.id_map[idx], float(score)))
            return results
        else:
            # Numpy fallback
            if not self._embeddings:
                return []

            embeddings = np.array(self._embeddings)
            scores = embeddings @ query.flatten()
            top_k = np.argsort(scores)[::-1][:k]

            return [(self.id_map[i], float(scores[i])) for i in top_k]

    def find_matching_experience(
        self,
        current_fingerprint: np.ndarray,
        threshold: float = 0.85,
    ) -> Optional[str]:
        """Find best matching stored experience for playback.

        Args:
            current_fingerprint: Current neural state embedding
            threshold: Minimum similarity threshold

        Returns:
            fingerprint_id if similarity > threshold, else None
        """
        results = self.search(current_fingerprint, k=1)

        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    def save(self, path: str) -> None:
        """Save index to file."""
        data = {
            "id_map": self.id_map,
            "embedding_dim": self.embedding_dim,
        }

        if FAISS_AVAILABLE and self.index is not None:
            # Save FAISS index separately
            faiss.write_index(self.index, f"{path}.faiss")
        else:
            data["embeddings"] = self._embeddings

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "FingerprintIndex":
        """Load index from file."""
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)

        index = cls(data["embedding_dim"])
        index.id_map = data["id_map"]

        if FAISS_AVAILABLE and Path(f"{path}.faiss").exists():
            index.index = faiss.read_index(f"{path}.faiss")
        elif "embeddings" in data:
            index._embeddings = data["embeddings"]

        return index

    def __len__(self) -> int:
        """Return number of fingerprints in index."""
        return len(self.id_map)


class FingerprintDatabase:
    """High-level interface for fingerprint storage and retrieval.

    Combines SQLAlchemy for metadata storage and FAISS for similarity search.

    Args:
        db_path: Path to SQLite database file
        index_path: Optional path for FAISS index (defaults to db_path + ".index")
    """

    def __init__(
        self,
        db_path: str,
        index_path: Optional[str] = None,
        embedding_dim: int = 256,
    ):
        self.db_path = db_path
        self.index_path = index_path or f"{db_path}.index"
        self.embedding_dim = embedding_dim

        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Load or create FAISS index
        if Path(f"{self.index_path}.pkl").exists():
            self.index = FingerprintIndex.load(self.index_path)
        else:
            self.index = FingerprintIndex(embedding_dim)

    def store_fingerprint(
        self,
        fingerprint_id: str,
        embedding: np.ndarray,
        modality: str,
        stimulus_label: str,
        subject_id: str,
        features: Optional[dict] = None,
        confidence: float = 0.0,
        signal_quality: float = 0.0,
        eeg_channels: int = 8,
        fnirs_channels: int = 4,
    ) -> None:
        """Store a neural fingerprint.

        Args:
            fingerprint_id: Unique identifier
            embedding: Embedding vector
            modality: Sensory modality (gustatory, olfactory, etc.)
            stimulus_label: Label for the stimulus
            subject_id: Subject identifier
            features: Optional full feature dictionary
            confidence: Confidence score (0-1)
            signal_quality: Signal quality score (0-1)
            eeg_channels: Number of EEG channels used
            fnirs_channels: Number of fNIRS channels used
        """
        session = self.Session()

        try:
            record = FingerprintRecord(
                fingerprint_id=fingerprint_id,
                modality=modality,
                stimulus_label=stimulus_label,
                subject_id=subject_id,
                timestamp=datetime.utcnow(),
                confidence=confidence,
                signal_quality=signal_quality,
                eeg_channels=eeg_channels,
                fnirs_channels=fnirs_channels,
            )

            record.set_embedding(embedding)

            if features:
                record.set_features(features)

            session.add(record)
            session.commit()

            # Add to similarity index
            self.index.add_fingerprint(fingerprint_id, embedding)

        finally:
            session.close()

    def get_fingerprint(self, fingerprint_id: str) -> Optional[FingerprintRecord]:
        """Retrieve a fingerprint by ID."""
        session = self.Session()
        try:
            return (
                session.query(FingerprintRecord)
                .filter_by(fingerprint_id=fingerprint_id)
                .first()
            )
        finally:
            session.close()

    def find_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        modality: Optional[str] = None,
    ) -> List[Tuple[FingerprintRecord, float]]:
        """Find similar fingerprints.

        Args:
            query_embedding: Query vector
            k: Number of results
            modality: Optional filter by modality

        Returns:
            List of (record, similarity) tuples
        """
        # Get candidates from FAISS
        candidates = self.index.search(query_embedding, k * 2)

        session = self.Session()
        try:
            results = []
            for fp_id, score in candidates:
                record = (
                    session.query(FingerprintRecord)
                    .filter_by(fingerprint_id=fp_id)
                    .first()
                )

                if record is None:
                    continue

                if modality and record.modality != modality:
                    continue

                results.append((record, score))

                if len(results) >= k:
                    break

            return results

        finally:
            session.close()

    def get_by_stimulus(
        self,
        stimulus_label: str,
        subject_id: Optional[str] = None,
    ) -> List[FingerprintRecord]:
        """Get all fingerprints for a stimulus.

        Args:
            stimulus_label: Stimulus to search for
            subject_id: Optional subject filter

        Returns:
            List of matching fingerprint records
        """
        session = self.Session()
        try:
            query = session.query(FingerprintRecord).filter_by(
                stimulus_label=stimulus_label
            )

            if subject_id:
                query = query.filter_by(subject_id=subject_id)

            return query.all()

        finally:
            session.close()

    def get_reference_embedding(
        self,
        stimulus_label: str,
    ) -> Optional[np.ndarray]:
        """Get average embedding for a stimulus across subjects.

        Args:
            stimulus_label: Stimulus to get reference for

        Returns:
            Average embedding vector, or None if not found
        """
        records = self.get_by_stimulus(stimulus_label)

        if not records:
            return None

        embeddings = [r.get_embedding() for r in records]
        embeddings = [e for e in embeddings if len(e) > 0]

        if not embeddings:
            return None

        return np.mean(embeddings, axis=0)

    def save_index(self) -> None:
        """Save the similarity index to disk."""
        self.index.save(self.index_path)

    def close(self) -> None:
        """Close database connection and save index."""
        self.save_index()
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import tempfile

    print("Testing FingerprintDatabase...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"

        # Create database
        db = FingerprintDatabase(db_path, embedding_dim=256)

        # Store some test fingerprints
        for i in range(5):
            embedding = np.random.randn(256).astype(np.float32)
            db.store_fingerprint(
                fingerprint_id=f"test_{i}",
                embedding=embedding,
                modality="gustatory",
                stimulus_label="apple" if i < 3 else "orange",
                subject_id="subject_01",
                confidence=0.8,
            )

        print(f"Stored {len(db.index)} fingerprints")

        # Search for similar
        query = np.random.randn(256).astype(np.float32)
        results = db.find_similar(query, k=3)
        print(f"Found {len(results)} similar fingerprints")

        for record, score in results:
            print(f"  {record.fingerprint_id}: {score:.3f}")

        # Get by stimulus
        apple_fps = db.get_by_stimulus("apple")
        print(f"Apple fingerprints: {len(apple_fps)}")

        # Get reference embedding
        ref = db.get_reference_embedding("apple")
        print(f"Reference embedding shape: {ref.shape if ref is not None else None}")

        db.close()

    print("\nAll database tests passed!")

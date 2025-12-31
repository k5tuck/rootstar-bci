//! Shared Fingerprint Database for cross-device access.
//!
//! This module provides a SQLite-backed database for storing and retrieving
//! neural fingerprints. The database is shared across all connected devices,
//! allowing fingerprints captured on one device to be used for stimulation
//! on another device.
//!
//! # Features
//!
//! - CRUD operations for neural fingerprints
//! - Similarity search using cached embeddings
//! - Subject and modality-based filtering
//! - Cross-device fingerprint sharing
//! - Automatic schema migration
//!
//! # Example
//!
//! ```rust,ignore
//! use rootstar_bci_native::fingerprint::database::FingerprintDatabase;
//!
//! // Open or create database
//! let db = FingerprintDatabase::open("fingerprints.db")?;
//!
//! // Store a fingerprint
//! db.store(&fingerprint)?;
//!
//! // Find similar fingerprints
//! let matches = db.find_similar(&current_fp, 0.8, 10)?;
//!
//! // Load by ID
//! let fp = db.load(&fingerprint_id)?;
//! ```

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use rootstar_bci_core::fingerprint::{
    FingerprintId, FingerprintMetadata, NeuralFingerprint, QualityMetrics, SensoryModality,
};
use rootstar_bci_core::types::Fixed24_8;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during database operations.
#[derive(Debug, Error)]
pub enum DatabaseError {
    /// SQLite error
    #[error("Database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Fingerprint not found
    #[error("Fingerprint not found: {0}")]
    NotFound(String),

    /// Invalid fingerprint data
    #[error("Invalid fingerprint data: {0}")]
    InvalidData(String),

    /// Database is locked
    #[error("Database is locked by another process")]
    Locked,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for database operations.
pub type DbResult<T> = Result<T, DatabaseError>;

// ============================================================================
// Stored Fingerprint Record
// ============================================================================

/// A stored fingerprint with additional metadata for database queries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredFingerprint {
    /// The neural fingerprint data
    pub fingerprint: NeuralFingerprint,

    /// User-assigned name for easy identification
    pub name: String,

    /// Device ID that captured this fingerprint
    pub source_device_id: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Notes or description
    pub notes: Option<String>,

    /// Number of times this fingerprint has been used for stimulation
    pub usage_count: u32,

    /// Last time this fingerprint was used (Unix timestamp)
    pub last_used_at: Option<u64>,

    /// Creation timestamp (Unix timestamp)
    pub created_at: u64,

    /// Last modification timestamp
    pub updated_at: u64,
}

impl StoredFingerprint {
    /// Create a new stored fingerprint from a neural fingerprint.
    #[must_use]
    pub fn new(fingerprint: NeuralFingerprint, name: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            fingerprint,
            name,
            source_device_id: None,
            tags: Vec::new(),
            notes: None,
            usage_count: 0,
            last_used_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the source device ID.
    #[must_use]
    pub fn with_device(mut self, device_id: String) -> Self {
        self.source_device_id = Some(device_id);
        self
    }

    /// Add tags for categorization.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add notes.
    #[must_use]
    pub fn with_notes(mut self, notes: String) -> Self {
        self.notes = Some(notes);
        self
    }

    /// Get the fingerprint ID.
    #[must_use]
    pub fn id(&self) -> FingerprintId {
        self.fingerprint.metadata.id
    }
}

// ============================================================================
// Search Results
// ============================================================================

/// A search result with similarity score.
#[derive(Clone, Debug)]
pub struct SimilarityMatch {
    /// The stored fingerprint
    pub stored: StoredFingerprint,

    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
}

/// Query filter for fingerprint searches.
#[derive(Clone, Debug, Default)]
pub struct FingerprintQuery {
    /// Filter by sensory modality
    pub modality: Option<SensoryModality>,

    /// Filter by subject ID
    pub subject_id: Option<String>,

    /// Filter by source device
    pub device_id: Option<String>,

    /// Filter by tags (any match)
    pub tags: Vec<String>,

    /// Minimum quality score (0-100)
    pub min_quality: Option<u8>,

    /// Maximum results to return
    pub limit: Option<usize>,

    /// Offset for pagination
    pub offset: Option<usize>,
}

impl FingerprintQuery {
    /// Create a new empty query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by modality.
    #[must_use]
    pub fn with_modality(mut self, modality: SensoryModality) -> Self {
        self.modality = Some(modality);
        self
    }

    /// Filter by subject.
    #[must_use]
    pub fn with_subject(mut self, subject_id: String) -> Self {
        self.subject_id = Some(subject_id);
        self
    }

    /// Filter by device.
    #[must_use]
    pub fn with_device(mut self, device_id: String) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Limit results.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

// ============================================================================
// Database Implementation
// ============================================================================

/// Schema version for migrations.
const SCHEMA_VERSION: i32 = 1;

/// Shared fingerprint database with cross-device access.
///
/// This database stores neural fingerprints in SQLite, allowing them to be
/// shared across multiple connected BCI devices. A fingerprint captured on
/// Device A can be loaded and used for stimulation on Device B.
pub struct FingerprintDatabase {
    /// Database connection (thread-safe)
    conn: Arc<RwLock<Connection>>,

    /// In-memory cache of fingerprint embeddings for fast similarity search
    embedding_cache: Arc<RwLock<HashMap<FingerprintId, Vec<f32>>>>,
}

impl FingerprintDatabase {
    /// Open or create a fingerprint database at the given path.
    ///
    /// The database file will be created if it doesn't exist.
    /// Schema migrations are applied automatically.
    pub fn open<P: AsRef<Path>>(path: P) -> DbResult<Self> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        let db = Self {
            conn: Arc::new(RwLock::new(conn)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        db.initialize_schema()?;
        db.load_embedding_cache()?;

        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory()?;

        let db = Self {
            conn: Arc::new(RwLock::new(conn)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        db.initialize_schema()?;

        Ok(db)
    }

    /// Initialize database schema.
    fn initialize_schema(&self) -> DbResult<()> {
        let conn = self.conn.write().map_err(|_| DatabaseError::Locked)?;

        // Check current schema version
        let version: i32 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap_or(0);

        if version < SCHEMA_VERSION {
            // Create tables
            conn.execute_batch(
                r"
                -- Main fingerprint storage
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id BLOB PRIMARY KEY,
                    name TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    stimulus_label TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    source_device_id TEXT,
                    quality_score INTEGER NOT NULL,
                    confidence INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    embedding BLOB,
                    tags TEXT,
                    notes TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used_at INTEGER,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_fingerprints_modality ON fingerprints(modality);
                CREATE INDEX IF NOT EXISTS idx_fingerprints_subject ON fingerprints(subject_id);
                CREATE INDEX IF NOT EXISTS idx_fingerprints_device ON fingerprints(source_device_id);
                CREATE INDEX IF NOT EXISTS idx_fingerprints_quality ON fingerprints(quality_score);
                CREATE INDEX IF NOT EXISTS idx_fingerprints_created ON fingerprints(created_at);

                -- Associated stimulation protocols
                CREATE TABLE IF NOT EXISTS stimulation_protocols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id BLOB NOT NULL,
                    protocol_json TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (fingerprint_id) REFERENCES fingerprints(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_protocols_fingerprint ON stimulation_protocols(fingerprint_id);
                ",
            )?;

            // Update schema version
            conn.execute(&format!("PRAGMA user_version = {SCHEMA_VERSION}"), [])?;
        }

        Ok(())
    }

    /// Load embedding cache from database.
    fn load_embedding_cache(&self) -> DbResult<()> {
        let conn = self.conn.read().map_err(|_| DatabaseError::Locked)?;
        let mut cache = self.embedding_cache.write().map_err(|_| DatabaseError::Locked)?;

        let mut stmt = conn.prepare("SELECT id, embedding FROM fingerprints WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let embedding_bytes: Vec<u8> = row.get(1)?;
            Ok((id_bytes, embedding_bytes))
        })?;

        for row in rows {
            let (id_bytes, embedding_bytes) = row?;
            if id_bytes.len() == 16 {
                let mut id_arr = [0u8; 16];
                id_arr.copy_from_slice(&id_bytes);
                let id = FingerprintId::from_bytes(id_arr);

                // Deserialize embedding (f32 array)
                let embedding: Vec<f32> = embedding_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                cache.insert(id, embedding);
            }
        }

        Ok(())
    }

    /// Generate embedding vector from fingerprint for similarity search.
    fn generate_embedding(fp: &NeuralFingerprint) -> Vec<f32> {
        let mut embedding = Vec::with_capacity(512);

        // Add EEG band power (normalized)
        for val in &fp.eeg_band_power {
            embedding.push(val.to_f32());
        }

        // Add EEG topography
        for val in &fp.eeg_topography {
            embedding.push(val.to_f32());
        }

        // Add fNIRS activation
        for val in &fp.fnirs_hbo_activation {
            embedding.push(val.to_f32());
        }
        for val in &fp.fnirs_hbr_activation {
            embedding.push(val.to_f32());
        }

        // Add EMG features
        for val in &fp.emg_rms_activation {
            embedding.push(val.to_f32());
        }
        embedding.push(fp.emg_valence_score.to_f32());
        embedding.push(fp.emg_arousal_score.to_f32());

        // Add EDA features
        for val in &fp.eda_scl {
            embedding.push(val.to_f32());
        }
        embedding.push(fp.eda_arousal_score.to_f32());

        embedding
    }

    /// Compute cosine similarity between two embeddings.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let norm_product = norm_a.sqrt() * norm_b.sqrt();
        if norm_product > 1e-8 {
            dot / norm_product
        } else {
            0.0
        }
    }

    /// Store a fingerprint in the database.
    ///
    /// If a fingerprint with the same ID already exists, it will be updated.
    pub fn store(&self, stored: &StoredFingerprint) -> DbResult<()> {
        let conn = self.conn.write().map_err(|_| DatabaseError::Locked)?;

        let fp = &stored.fingerprint;
        let id_bytes = fp.metadata.id.as_bytes().to_vec();
        let modality = fp.metadata.modality.name();
        let stimulus_label = fp.metadata.stimulus_label();
        let subject_id = fp.metadata.subject_id();

        // Serialize fingerprint data
        let data_json = serde_json::to_string(stored)?;

        // Generate embedding
        let embedding = Self::generate_embedding(fp);
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Serialize tags
        let tags_json = serde_json::to_string(&stored.tags)?;

        conn.execute(
            r"
            INSERT INTO fingerprints (
                id, name, modality, stimulus_label, subject_id, source_device_id,
                quality_score, confidence, data_json, embedding, tags, notes,
                usage_count, last_used_at, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                data_json = excluded.data_json,
                embedding = excluded.embedding,
                tags = excluded.tags,
                notes = excluded.notes,
                usage_count = excluded.usage_count,
                last_used_at = excluded.last_used_at,
                updated_at = excluded.updated_at
            ",
            params![
                id_bytes,
                stored.name,
                modality,
                stimulus_label,
                subject_id,
                stored.source_device_id,
                fp.metadata.quality.overall_score,
                fp.confidence,
                data_json,
                embedding_bytes,
                tags_json,
                stored.notes,
                stored.usage_count,
                stored.last_used_at.map(|t| t as i64),
                stored.created_at as i64,
                stored.updated_at as i64,
            ],
        )?;

        // Update cache
        drop(conn);
        let mut cache = self.embedding_cache.write().map_err(|_| DatabaseError::Locked)?;
        cache.insert(fp.metadata.id, embedding);

        Ok(())
    }

    /// Load a fingerprint by ID.
    pub fn load(&self, id: &FingerprintId) -> DbResult<StoredFingerprint> {
        let conn = self.conn.read().map_err(|_| DatabaseError::Locked)?;

        let id_bytes = id.as_bytes().to_vec();

        let data_json: String = conn
            .query_row(
                "SELECT data_json FROM fingerprints WHERE id = ?1",
                params![id_bytes],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    DatabaseError::NotFound(format!("{id:?}"))
                }
                other => DatabaseError::Sqlite(other),
            })?;

        let stored: StoredFingerprint = serde_json::from_str(&data_json)?;
        Ok(stored)
    }

    /// Delete a fingerprint by ID.
    pub fn delete(&self, id: &FingerprintId) -> DbResult<bool> {
        let conn = self.conn.write().map_err(|_| DatabaseError::Locked)?;

        let id_bytes = id.as_bytes().to_vec();
        let rows_affected = conn.execute(
            "DELETE FROM fingerprints WHERE id = ?1",
            params![id_bytes],
        )?;

        // Remove from cache
        drop(conn);
        let mut cache = self.embedding_cache.write().map_err(|_| DatabaseError::Locked)?;
        cache.remove(id);

        Ok(rows_affected > 0)
    }

    /// Find fingerprints similar to the given fingerprint.
    ///
    /// Returns matches with similarity >= threshold, sorted by similarity (descending).
    pub fn find_similar(
        &self,
        fingerprint: &NeuralFingerprint,
        threshold: f32,
        limit: usize,
    ) -> DbResult<Vec<SimilarityMatch>> {
        let query_embedding = Self::generate_embedding(fingerprint);
        let cache = self.embedding_cache.read().map_err(|_| DatabaseError::Locked)?;

        // Compute similarities against all cached embeddings
        let mut matches: Vec<(FingerprintId, f32)> = cache
            .iter()
            .map(|(id, emb)| (*id, Self::cosine_similarity(&query_embedding, emb)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        // Sort by similarity (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(limit);

        // Load full fingerprint data for matches
        drop(cache);
        let mut results = Vec::with_capacity(matches.len());
        for (id, similarity) in matches {
            if let Ok(stored) = self.load(&id) {
                results.push(SimilarityMatch { stored, similarity });
            }
        }

        Ok(results)
    }

    /// Query fingerprints with filters.
    pub fn query(&self, filter: &FingerprintQuery) -> DbResult<Vec<StoredFingerprint>> {
        let conn = self.conn.read().map_err(|_| DatabaseError::Locked)?;

        let mut sql = String::from("SELECT data_json FROM fingerprints WHERE 1=1");
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(modality) = &filter.modality {
            sql.push_str(" AND modality = ?");
            params_vec.push(Box::new(modality.name().to_string()));
        }

        if let Some(subject_id) = &filter.subject_id {
            sql.push_str(" AND subject_id = ?");
            params_vec.push(Box::new(subject_id.clone()));
        }

        if let Some(device_id) = &filter.device_id {
            sql.push_str(" AND source_device_id = ?");
            params_vec.push(Box::new(device_id.clone()));
        }

        if let Some(min_quality) = filter.min_quality {
            sql.push_str(" AND quality_score >= ?");
            params_vec.push(Box::new(i32::from(min_quality)));
        }

        sql.push_str(" ORDER BY created_at DESC");

        if let Some(limit) = filter.limit {
            sql.push_str(&format!(" LIMIT {limit}"));
        }

        if let Some(offset) = filter.offset {
            sql.push_str(&format!(" OFFSET {offset}"));
        }

        let mut stmt = conn.prepare(&sql)?;

        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            let data_json: String = row.get(0)?;
            Ok(data_json)
        })?;

        let mut results = Vec::new();
        for row in rows {
            let data_json = row?;
            if let Ok(stored) = serde_json::from_str::<StoredFingerprint>(&data_json) {
                results.push(stored);
            }
        }

        Ok(results)
    }

    /// List all fingerprints (with optional limit).
    pub fn list_all(&self, limit: Option<usize>) -> DbResult<Vec<StoredFingerprint>> {
        self.query(&FingerprintQuery {
            limit,
            ..Default::default()
        })
    }

    /// Get count of stored fingerprints.
    pub fn count(&self) -> DbResult<usize> {
        let conn = self.conn.read().map_err(|_| DatabaseError::Locked)?;

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM fingerprints",
            [],
            |row| row.get(0),
        )?;

        Ok(count as usize)
    }

    /// Record usage of a fingerprint (for statistics).
    pub fn record_usage(&self, id: &FingerprintId) -> DbResult<()> {
        let conn = self.conn.write().map_err(|_| DatabaseError::Locked)?;

        let id_bytes = id.as_bytes().to_vec();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        conn.execute(
            "UPDATE fingerprints SET usage_count = usage_count + 1, last_used_at = ?1 WHERE id = ?2",
            params![now, id_bytes],
        )?;

        Ok(())
    }

    /// Generate a new unique fingerprint ID.
    #[must_use]
    pub fn generate_id() -> FingerprintId {
        let uuid = Uuid::new_v4();
        FingerprintId::from_bytes(*uuid.as_bytes())
    }

    /// Get database statistics.
    pub fn stats(&self) -> DbResult<DatabaseStats> {
        let conn = self.conn.read().map_err(|_| DatabaseError::Locked)?;

        let total_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM fingerprints",
            [],
            |row| row.get(0),
        )?;

        let mut modality_counts = HashMap::new();
        let mut stmt = conn.prepare("SELECT modality, COUNT(*) FROM fingerprints GROUP BY modality")?;
        let rows = stmt.query_map([], |row| {
            let modality: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((modality, count))
        })?;

        for row in rows {
            let (modality, count) = row?;
            modality_counts.insert(modality, count as usize);
        }

        let cache_size = self.embedding_cache.read()
            .map(|c| c.len())
            .unwrap_or(0);

        Ok(DatabaseStats {
            total_fingerprints: total_count as usize,
            modality_counts,
            embedding_cache_size: cache_size,
        })
    }
}

/// Database statistics.
#[derive(Clone, Debug)]
pub struct DatabaseStats {
    /// Total number of stored fingerprints
    pub total_fingerprints: usize,

    /// Count per modality
    pub modality_counts: HashMap<String, usize>,

    /// Number of embeddings in cache
    pub embedding_cache_size: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rootstar_bci_core::fingerprint::FingerprintMetadata;

    fn create_test_fingerprint(label: &str) -> NeuralFingerprint {
        let id = FingerprintDatabase::generate_id();
        let metadata = FingerprintMetadata::new(
            id,
            SensoryModality::Gustatory,
            label,
            "test_subject",
            0,
        );

        let mut fp = NeuralFingerprint::new(metadata);

        // Add some test data
        for i in 0..8 {
            let _ = fp.eeg_topography.push(Fixed24_8::from_f32(i as f32 * 0.1));
        }

        fp.set_confidence_f32(0.85);
        fp
    }

    #[test]
    fn test_database_open_and_store() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        let fp = create_test_fingerprint("apple_taste");
        let stored = StoredFingerprint::new(fp, "Apple Taste Test".to_string());

        db.store(&stored).unwrap();

        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn test_database_load() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        let fp = create_test_fingerprint("rose_smell");
        let id = fp.metadata.id;
        let stored = StoredFingerprint::new(fp, "Rose Smell".to_string())
            .with_notes("First capture".to_string());

        db.store(&stored).unwrap();

        let loaded = db.load(&id).unwrap();
        assert_eq!(loaded.name, "Rose Smell");
        assert_eq!(loaded.notes, Some("First capture".to_string()));
    }

    #[test]
    fn test_similarity_search() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        // Store a few fingerprints
        for i in 0..5 {
            let fp = create_test_fingerprint(&format!("test_{i}"));
            let stored = StoredFingerprint::new(fp, format!("Test {i}"));
            db.store(&stored).unwrap();
        }

        // Search for similar
        let query_fp = create_test_fingerprint("query");
        let matches = db.find_similar(&query_fp, 0.5, 10).unwrap();

        // Should find matches (all test fingerprints are similar)
        assert!(!matches.is_empty());

        // Verify sorted by similarity
        for window in matches.windows(2) {
            assert!(window[0].similarity >= window[1].similarity);
        }
    }

    #[test]
    fn test_query_by_modality() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        // Store fingerprints with different modalities
        let fp1 = create_test_fingerprint("taste1");
        let stored1 = StoredFingerprint::new(fp1, "Taste 1".to_string());
        db.store(&stored1).unwrap();

        // Query by modality
        let query = FingerprintQuery::new()
            .with_modality(SensoryModality::Gustatory)
            .with_limit(10);

        let results = db.query(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Taste 1");
    }

    #[test]
    fn test_delete() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        let fp = create_test_fingerprint("to_delete");
        let id = fp.metadata.id;
        let stored = StoredFingerprint::new(fp, "To Delete".to_string());

        db.store(&stored).unwrap();
        assert_eq!(db.count().unwrap(), 1);

        let deleted = db.delete(&id).unwrap();
        assert!(deleted);
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn test_usage_tracking() {
        let db = FingerprintDatabase::open_in_memory().unwrap();

        let fp = create_test_fingerprint("usage_test");
        let id = fp.metadata.id;
        let stored = StoredFingerprint::new(fp, "Usage Test".to_string());

        db.store(&stored).unwrap();

        // Record usage
        db.record_usage(&id).unwrap();
        db.record_usage(&id).unwrap();

        let loaded = db.load(&id).unwrap();
        assert_eq!(loaded.usage_count, 2);
        assert!(loaded.last_used_at.is_some());
    }
}

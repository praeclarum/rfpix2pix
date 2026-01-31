"""
Persistent embedding cache using SQLite.

Stores image MD5 -> embedding mappings to avoid recomputing structure
embeddings across training runs.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def get_cache_dir() -> Path:
    """Get the cache directory, respecting RFPIX2PIX_CACHE_DIR env var."""
    cache_dir = os.environ.get("RFPIX2PIX_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "rfpix2pix"


class EmbeddingCache:
    """
    SQLite-backed cache for image structure embeddings.
    
    Maps image MD5 hashes to embedding vectors. Thread-safe for reads,
    uses write-ahead logging for concurrent access across processes.
    """
    
    def __init__(self, db_name: str = "dino_embeddings.db"):
        """
        Initialize the embedding cache.
        
        Args:
            db_name: Name of the SQLite database file.
        """
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / db_name
        
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    md5 TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """)
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get(self, md5: str) -> Optional[np.ndarray]:
        """
        Get an embedding by MD5 hash.
        
        Args:
            md5: MD5 hash of the image file.
            
        Returns:
            Embedding vector as numpy array, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE md5 = ?",
                (md5.lower(),)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return np.frombuffer(row[0], dtype=np.float32)
        finally:
            conn.close()
    
    def put(self, md5: str, embedding: np.ndarray):
        """
        Store an embedding.
        
        Args:
            md5: MD5 hash of the image file.
            embedding: Embedding vector as numpy array.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (md5, embedding) VALUES (?, ?)",
                (md5.lower(), embedding.astype(np.float32).tobytes())
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_batch(self, md5_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple embeddings by MD5 hash.
        
        Args:
            md5_list: List of MD5 hashes to look up.
            
        Returns:
            Dict mapping found MD5s to their embeddings.
        """
        if not md5_list:
            return {}
        
        conn = self._get_connection()
        try:
            # Use parameterized query with IN clause
            placeholders = ",".join("?" * len(md5_list))
            cursor = conn.execute(
                f"SELECT md5, embedding FROM embeddings WHERE md5 IN ({placeholders})",
                [m.lower() for m in md5_list]
            )
            result = {}
            for row in cursor:
                result[row[0]] = np.frombuffer(row[1], dtype=np.float32)
            return result
        finally:
            conn.close()
    
    def put_batch(self, embeddings: Dict[str, np.ndarray]):
        """
        Store multiple embeddings.
        
        Args:
            embeddings: Dict mapping MD5 hashes to embedding vectors.
        """
        if not embeddings:
            return
        
        conn = self._get_connection()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (md5, embedding) VALUES (?, ?)",
                [(md5.lower(), emb.astype(np.float32).tobytes()) for md5, emb in embeddings.items()]
            )
            conn.commit()
        finally:
            conn.close()
    
    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def __repr__(self) -> str:
        return f"EmbeddingCache({self.db_path}, {len(self)} entries)"

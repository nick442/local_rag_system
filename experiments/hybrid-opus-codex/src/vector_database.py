"""
Vector Database for RAG System
SQLite-based vector database with sqlite-vec extension for efficient similarity search.
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import uuid

import numpy as np

# sqlite-vec is optional; if not installed as a Python package, we'll attempt vendor dylib below
try:
    import sqlite_vec  # type: ignore
except Exception:  # pragma: no cover - handled by runtime fallback
    sqlite_vec = None
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_ingestion import DocumentChunk
from .interfaces.vector_index_interface import VectorIndexInterface


class VectorDatabase(VectorIndexInterface):
    """SQLite-based vector database with sqlite-vec extension."""
    
    def __init__(self, db_path: str, embedding_dimension: int = 384):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the SQLite database file
            embedding_dimension: Dimension of embedding vectors
        """
        self.db_path = Path(db_path)
        self.embedding_dimension = embedding_dimension
        self.logger = logging.getLogger(__name__)
        
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with sqlite-vec loaded."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            # Apply PRAGMA optimizations
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            # Negative cache size means kibibytes of cache in memory
            cur.execute("PRAGMA cache_size=-64000")
            cur.close()
        except Exception:
            # Non-fatal if PRAGMAs fail (e.g., read-only connection)
            pass
        
        try:
            # Try Python package first
            if sqlite_vec is not None:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self.logger.info("Successfully loaded sqlite-vec extension using Python package")
            else:
                raise RuntimeError("sqlite_vec module not available")
        except Exception as e:
            # Attempt vendor dylib fallback unless disabled by env
            try:
                disable_vendor = os.getenv('RAG_DISABLE_SQLITE_VEC_VENDOR') == '1' or os.getenv('RAG_SQLITE_VEC_TRY_VENDOR', '1') == '0'
                if not disable_vendor:
                    conn.enable_load_extension(True)
                    # Resolve vendor dylib path relative to project root
                    vendor_path = (Path(__file__).resolve().parents[1] / 'vendor' / 'sqlite-vec' / 'vec0.dylib')
                    if vendor_path.exists():
                        conn.load_extension(str(vendor_path))
                        self.logger.info(f"Loaded sqlite-vec from vendor dylib: {vendor_path}")
                    else:
                        # Try by name if on system path
                        conn.load_extension('vec0')
                        self.logger.info("Loaded sqlite-vec via 'vec0' name")
                    conn.enable_load_extension(False)
                else:
                    raise RuntimeError("Vendor sqlite-vec load disabled by environment")
            except Exception as e2:
                try:
                    conn.enable_load_extension(False)
                except Exception:
                    pass
                self.logger.warning(
                    f"Failed to load sqlite-vec (package/vendor): {e2}. Vector search will use fallback."
                )
        
        return conn
    
    def init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Persist and validate embedding dimension in metadata
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cursor.execute("SELECT value FROM db_metadata WHERE key = 'embedding_dimension'")
            row = cursor.fetchone()
            if row is None:
                # First-time init: store the configured embedding dimension
                cursor.execute(
                    "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('embedding_dimension', ?)",
                    (str(int(self.embedding_dimension)),),
                )
            else:
                try:
                    stored_dim = int(row["value"]) if isinstance(row, sqlite3.Row) else int(row[0])
                except Exception:
                    stored_dim = None
                if stored_dim is not None and stored_dim != int(self.embedding_dimension):
                    raise ValueError(
                        f"Database embedding dimension mismatch: stored={stored_dim}, requested={self.embedding_dimension}. "
                        f"Use a database initialized with the same embedding dimension, or reindex your data."
                    )

            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT,
                    file_size INTEGER,
                    total_chunks INTEGER DEFAULT 0,
                    collection_id TEXT NOT NULL DEFAULT 'default'
                )
            """)
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    collection_id TEXT NOT NULL DEFAULT 'default',
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                )
            """)
            
            # Create embeddings table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding_vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
                )
            """)
            
            # Add collection_id columns to existing tables if they don't exist
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN collection_id TEXT NOT NULL DEFAULT 'default'")
                self.logger.info("Added collection_id column to documents table")
            except sqlite3.OperationalError:
                # Column already exists
                pass
                
            try:
                cursor.execute("ALTER TABLE chunks ADD COLUMN collection_id TEXT NOT NULL DEFAULT 'default'")  
                self.logger.info("Added collection_id column to chunks table")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks (doc_id, chunk_index)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks (collection_id)")
            
            # Create FTS5 virtual table for keyword search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)
            
            # Create triggers to keep FTS5 in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(chunk_id, content) VALUES (new.chunk_id, new.content);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, chunk_id, content) VALUES('delete', old.chunk_id, old.content);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, chunk_id, content) VALUES('delete', old.chunk_id, old.content);
                    INSERT INTO chunks_fts(chunk_id, content) VALUES (new.chunk_id, new.content);
                END
            """)
            
            # Try to create vector search table if sqlite-vec is available
            try:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[{self.embedding_dimension}]
                    )
                """)
                self.logger.info("sqlite-vec vector search table created successfully")
            except sqlite3.OperationalError as e:
                self.logger.warning(f"Could not create vector search table: {e}")

            conn.commit()
    
    def insert_document(self, doc_id: str, source_path: str, metadata: Dict[str, Any], collection_id: str = "default") -> bool:
        """
        Insert a document record.
        
        Args:
            doc_id: Unique document identifier
            source_path: Path to the source document
            metadata: Document metadata
            collection_id: Collection to associate the document with
            
        Returns:
            True if inserted, False if already exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if document already exists
            cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
            if cursor.fetchone():
                return False
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (doc_id, source_path, ingested_at, metadata_json, 
                                     content_hash, file_size, collection_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                source_path,
                datetime.now().isoformat(),
                json.dumps(metadata),
                metadata.get('content_hash', ''),
                metadata.get('size', 0),
                collection_id
            ))
            
            conn.commit()
            return True
    
    def insert_chunk(self, chunk: 'DocumentChunk', embedding: np.ndarray, collection_id: str = "default") -> bool:
        """
        Insert a document chunk with its embedding.
        
        Args:
            chunk: DocumentChunk object
            embedding: Embedding vector as numpy array
            collection_id: Collection to associate the chunk with
            
        Returns:
            True if inserted successfully
        """
        if embedding.shape[0] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected {self.embedding_dimension}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert chunk
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, content, 
                                             token_count, metadata_json, created_at, collection_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                json.dumps(chunk.metadata),
                datetime.now().isoformat(),
                collection_id
            ))
            
            # Insert embedding
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (chunk_id, embedding_vector, created_at)
                VALUES (?, ?, ?)
            """, (
                chunk.chunk_id,
                embedding_blob,
                datetime.now().isoformat()
            ))
            
            # Insert into vector search table if available
            try:
                # Convert embedding to JSON string format for sqlite-vec
                embedding_json = f"[{','.join(map(str, embedding.tolist()))}]"
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings_vec (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk.chunk_id, embedding_json))
            except sqlite3.OperationalError:
                # Vector table not available, skip
                pass
            
            conn.commit()
            return True
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      metadata_filter: Optional[Dict[str, Any]] = None, collection_id: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            metadata_filter: Optional metadata filters
            collection_id: Optional collection filter
            
        Returns:
            List of (chunk_id, similarity_score, chunk_data) tuples
        """
        if query_embedding.shape[0] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match expected {self.embedding_dimension}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Try vector search first if available
            try:
                # Convert query embedding to JSON string format
                query_json = f"[{','.join(map(str, query_embedding.tolist()))}]"
                
                # Build query with optional collection filter
                base_query = """
                    SELECT 
                        v.chunk_id,
                        v.distance,
                        c.content,
                        c.metadata_json,
                        c.doc_id,
                        c.chunk_index,
                        c.collection_id
                    FROM embeddings_vec v
                    JOIN chunks c ON v.chunk_id = c.chunk_id
                    WHERE v.embedding match ?
                """
                
                params = [query_json]
                
                if collection_id:
                    base_query += " AND c.collection_id = ?"
                    params.append(collection_id)
                
                base_query += " ORDER BY v.distance LIMIT ?"
                params.append(k)
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                # sqlite-vec returns distance directly (lower is better)
                return [
                    (
                        row['chunk_id'],
                        # Convert distance to similarity score
                        1.0 / (1.0 + row['distance']),
                        {
                            'content': row['content'],
                            'metadata': json.loads(row['metadata_json']),
                            'doc_id': row['doc_id'],
                            'chunk_index': row['chunk_index'],
                            'collection_id': row['collection_id']
                        }
                    )
                    for row in results
                ]
                
            except sqlite3.OperationalError:
                # Fallback to manual similarity calculation
                return self._manual_similarity_search(query_embedding, k, metadata_filter, collection_id)
    
    def _manual_similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_id: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Fallback method for similarity search without sqlite-vec."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all embeddings
            query = """
                SELECT e.chunk_id, e.embedding_vector, c.content, c.metadata_json, 
                       c.doc_id, c.chunk_index, c.collection_id
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
            """
            
            # Add filters
            conditions = []
            params = []
            
            if collection_id:
                conditions.append("c.collection_id = ?")
                params.append(collection_id)
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    conditions.append(f"json_extract(c.metadata_json, '$.{key}') = ?")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Calculate similarities
            similarities = []
            for row in rows:
                # Convert blob back to numpy array
                embedding_vector = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                similarity = np.dot(query_embedding, embedding_vector)
                
                similarities.append((
                    row['chunk_id'],
                    float(similarity),
                    {
                        'content': row['content'],
                        'metadata': json.loads(row['metadata_json']),
                        'doc_id': row['doc_id'],
                        'chunk_index': row['chunk_index'],
                        'collection_id': row['collection_id']
                    }
                ))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
    
    def keyword_search(self, query: str, k: int = 5, collection_id: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for chunks using keyword search (FTS5).
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (chunk_id, relevance_score, chunk_data) tuples
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                base_query = (
                    "SELECT f.chunk_id, bm25(f) as score, c.content, c.metadata_json, c.doc_id, c.chunk_index "
                    "FROM chunks_fts f JOIN chunks c ON f.chunk_id = c.chunk_id "
                    "WHERE chunks_fts MATCH ?"
                )
                params: List[Any] = [query]
                if collection_id:
                    base_query += " AND c.collection_id = ?"
                    params.append(collection_id)
                base_query += " ORDER BY score LIMIT ?"
                params.append(k)
                cursor.execute(base_query, params)
                
                results = cursor.fetchall()
                
                return [
                    (
                        row['chunk_id'],
                        row['score'],
                        {
                            'content': row['content'],
                            'metadata': json.loads(row['metadata_json']),
                            'doc_id': row['doc_id'],
                            'chunk_index': row['chunk_index']
                        }
                    )
                    for row in results
                ]
                
            except sqlite3.OperationalError as e:
                self.logger.warning(f"FTS5 search failed: {e}")
                return []
    
    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, k: int = 5, 
                     alpha: float = 0.7, collection_id: Optional[str] = None,
                     candidate_multiplier: int = 5,
                     fusion_method: str = "maxnorm",
                     rrf_k: int = 60) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Query text for keyword search
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            
        Returns:
            List of (chunk_id, combined_score, chunk_data) tuples
        """
        # Candidate pools (fetch more from each method, then fuse)
        try:
            cm = int(candidate_multiplier)
        except Exception:
            cm = 5
        fetch_n = max(k * cm, k)
        
        # Get vector search results
        vector_results = self.search_similar(query_embedding, fetch_n, collection_id=collection_id)
        
        # Get keyword search results
        keyword_results = self.keyword_search(query_text, fetch_n, collection_id=collection_id)
        
        # Combine and score results using selected fusion
        combined_scores: Dict[str, Dict[str, Any]] = {}
        fusion = (fusion_method or "maxnorm").lower()

        # Build maps for scores and ranks
        vec_scores = {cid: s for cid, s, _ in vector_results}
        key_scores = {cid: s for cid, s, _ in keyword_results}
        vec_data = {cid: d for cid, _, d in vector_results}
        key_data = {cid: d for cid, _, d in keyword_results}
        all_ids = set(vec_scores.keys()) | set(key_scores.keys())

        if fusion == "zscore":
            import numpy as np
            v_vals = np.array(list(vec_scores.values())) if vec_scores else np.array([])
            k_vals = np.array(list(key_scores.values())) if key_scores else np.array([])
            v_mean, v_std = (float(v_vals.mean()), float(v_vals.std())) if v_vals.size else (0.0, 1.0)
            k_mean, k_std = (float(k_vals.mean()), float(k_vals.std())) if k_vals.size else (0.0, 1.0)
            if v_std == 0.0:
                v_std = 1.0
            if k_std == 0.0:
                k_std = 1.0
            for cid in all_ids:
                zv = ((vec_scores.get(cid, 0.0) - v_mean) / v_std)
                zk = ((key_scores.get(cid, 0.0) - k_mean) / k_std)
                score = alpha * zv + (1 - alpha) * zk
                data = vec_data.get(cid, key_data.get(cid))
                combined_scores[cid] = { 'score': float(score), 'data': data }

        elif fusion == "rrf":
            # Reciprocal Rank Fusion
            v_ranks = {cid: idx+1 for idx, (cid, _, _) in enumerate(vector_results)}
            k_ranks = {cid: idx+1 for idx, (cid, _, _) in enumerate(keyword_results)}
            K = int(rrf_k) if rrf_k and rrf_k > 0 else 60
            for cid in all_ids:
                rv = v_ranks.get(cid, len(v_ranks) + K)
                rk = k_ranks.get(cid, len(k_ranks) + K)
                sv = 1.0 / (K + rv)
                sk = 1.0 / (K + rk)
                score = alpha * sv + (1 - alpha) * sk
                data = vec_data.get(cid, key_data.get(cid))
                combined_scores[cid] = { 'score': float(score), 'data': data }

        else:
            # Default: per-list max normalization (legacy)
            if vector_results:
                max_vector_score = max(score for _, score, _ in vector_results)
                for chunk_id, score, data in vector_results:
                    normalized_score = score / max_vector_score if max_vector_score > 0 else 0
                    combined_scores[chunk_id] = {
                        'score': alpha * normalized_score,
                        'data': data
                    }
            if keyword_results:
                max_keyword_score = max(score for _, score, _ in keyword_results)
                for chunk_id, score, data in keyword_results:
                    normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                    if chunk_id in combined_scores:
                        combined_scores[chunk_id]['score'] += (1 - alpha) * normalized_score
                    else:
                        combined_scores[chunk_id] = {
                            'score': (1 - alpha) * normalized_score,
                            'data': data
                        }

        # Sort by combined score and return top k
        sorted_results = sorted(
            [(chunk_id, item['score'], item['data']) for chunk_id, item in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data by chunk ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_id, doc_id, chunk_index, content, token_count, 
                       metadata_json, created_at
                FROM chunks
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'token_count': row['token_count'],
                    'metadata': json.loads(row['metadata_json']),
                    'created_at': row['created_at']
                }
            return None
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_id, chunk_index, content, token_count, metadata_json
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_index
            """, (doc_id,))
            
            return [
                {
                    'chunk_id': row['chunk_id'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'token_count': row['token_count'],
                    'metadata': json.loads(row['metadata_json'])
                }
                for row in cursor.fetchall()
            ]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            deleted_rows = cursor.rowcount
            
            conn.commit()
            return deleted_rows > 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()['count']
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            chunk_count = cursor.fetchone()['count']
            
            # Count embeddings
            cursor.execute("SELECT COUNT(*) as count FROM embeddings")
            embedding_count = cursor.fetchone()['count']
            
            # Database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'documents': doc_count,
                'chunks': chunk_count,
                'embeddings': embedding_count,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'embedding_dimension': self.embedding_dimension
            }


def create_vector_database(db_path: str, embedding_dimension: int = 384) -> VectorDatabase:
    """
    Factory function to create a VectorDatabase instance.
    
    Args:
        db_path: Path to the SQLite database file
        embedding_dimension: Dimension of embedding vectors
        
    Returns:
        Configured VectorDatabase instance
    """
    return VectorDatabase(db_path, embedding_dimension)


def create_vector_index(
    db_path: str,
    embedding_dimension: int = 384,
    backend: str = "sqlite",
    **kwargs,
) -> VectorIndexInterface:
    """
    Factory for vector index implementations.

    Currently supports:
    - backend='sqlite': returns VectorDatabase (sqlite-vec)
    """
    backend = (backend or "sqlite").lower()
    if backend in ("sqlite", "sqlite-vec", "sqlite_vec"):
        return VectorDatabase(db_path, embedding_dimension)
    raise NotImplementedError(f"Unknown vector index backend: {backend}")

"""
Re-indexing Tools - Corpus maintenance and optimization utilities

This module provides comprehensive re-indexing capabilities including:
- Update embeddings with new models
- Re-chunk documents with different parameters
- Refresh corrupted indices and vector tables
- Database optimization and maintenance
- Schema migration and upgrades
"""

import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .vector_database import VectorDatabase
from .embedding_service import EmbeddingService
from .document_ingestion import DocumentIngestionService, DocumentChunker


@dataclass
class ReindexStats:
    """Statistics for re-indexing operations"""
    operation: str
    collection_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    documents_processed: int = 0
    chunks_processed: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    processing_time: float = 0.0
    success: bool = False
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'collection_id': self.collection_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'documents_processed': self.documents_processed,
            'chunks_processed': self.chunks_processed,
            'embeddings_generated': self.embeddings_generated,
            'errors': self.errors,
            'processing_time': self.processing_time,
            'success': self.success,
            'details': self.details
        }


class ReindexTool:
    """
    Advanced corpus re-indexing and maintenance system.
    
    Features:
    - Re-embed documents with new embedding models
    - Re-chunk documents with different chunk parameters
    - Rebuild vector indices for performance optimization
    - Database schema migration and upgrades
    - Corruption detection and repair
    - Performance optimization and maintenance
    """
    
    def __init__(
        self,
        db_path: str = "data/rag_vectors.db",
        embedding_service: Optional[EmbeddingService] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128
    ):
        """
        Initialize re-indexing tool.
        
        Args:
            db_path: Path to vector database
            embedding_service: Embedding service for re-embedding
            chunk_size: Default chunk size for re-chunking
            chunk_overlap: Default chunk overlap for re-chunking
        """
        self.db_path = Path(db_path)
        self.db = VectorDatabase(str(db_path))
        self.embedding_service = embedding_service or EmbeddingService()
        self.ingestion_service = DocumentIngestionService()
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        
        self.logger = logging.getLogger(__name__)
    
    def _get_documents_in_collection(self, collection_id: str = "default") -> List[Dict[str, Any]]:
        """Get all documents in a collection."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, source_path, ingested_at, metadata_json, content_hash, file_size, total_chunks
                FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'doc_id': row[0],
                    'source_path': row[1],
                    'ingested_at': row[2],
                    'metadata_json': row[3],
                    'content_hash': row[4],
                    'file_size': row[5],
                    'total_chunks': row[6],
                    'collection_id': collection_id
                })
            
            return documents
    
    def _get_document_chunks(self, doc_id: str, collection_id: str = "default") -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, chunk_index, content, token_count, metadata_json, created_at
                FROM chunks WHERE doc_id = ? AND collection_id = ?
                ORDER BY chunk_index
            """, (doc_id, collection_id))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'chunk_id': row[0],
                    'chunk_index': row[1],
                    'content': row[2],
                    'token_count': row[3],
                    'metadata_json': row[4],
                    'created_at': row[5],
                    'doc_id': doc_id,
                    'collection_id': collection_id
                })
            
            return chunks
    
    def _backup_database(self) -> Path:
        """Create a backup of the database before major operations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_{timestamp}.db"
        
        with sqlite3.connect(str(self.db_path)) as source:
            with sqlite3.connect(str(backup_path)) as backup:
                source.backup(backup)
        
        self.logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def reembed_collection(
        self,
        collection_id: str = "default",
        new_embedding_service: Optional[EmbeddingService] = None,
        batch_size: int = 32,
        backup: bool = True
    ) -> ReindexStats:
        """
        Re-generate embeddings for all documents in a collection.
        
        Args:
            collection_id: Collection to re-embed
            new_embedding_service: New embedding service (uses current if None)
            batch_size: Batch size for embedding generation
            backup: Whether to backup database before operation
            
        Returns:
            ReindexStats with operation results
        """
        self.logger.info(f"Starting re-embedding for collection: {collection_id}")
        
        stats = ReindexStats(
            operation="reembed",
            collection_id=collection_id,
            start_time=datetime.now()
        )
        
        try:
            # Create backup if requested
            if backup:
                backup_path = self._backup_database()
                stats.details['backup_path'] = str(backup_path)
            
            # Use new embedding service if provided
            embedding_service = new_embedding_service or self.embedding_service
            
            # Get all documents in collection
            documents = self._get_documents_in_collection(collection_id)
            stats.details['total_documents'] = len(documents)
            
            if not documents:
                self.logger.warning(f"No documents found in collection: {collection_id}")
                stats.success = True
                return stats
            
            # Process documents in batches
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                with tqdm(total=len(documents), desc="Re-embedding documents") as pbar:
                    for doc in documents:
                        try:
                            # Get all chunks for document
                            chunks = self._get_document_chunks(doc['doc_id'], collection_id)
                            
                            if not chunks:
                                self.logger.warning(f"No chunks found for document: {doc['doc_id']}")
                                continue
                            
                            # Extract chunk content
                            chunk_texts = [chunk['content'] for chunk in chunks]
                            
                            # Generate new embeddings
                            embeddings = embedding_service.embed_texts(chunk_texts)
                            
                            # Update embeddings in database
                            for chunk, embedding in zip(chunks, embeddings):
                                embedding_bytes = embedding.astype(np.float32).tobytes()
                                
                                # Update embeddings table
                                cursor.execute("""
                                    UPDATE embeddings SET embedding_vector = ?, created_at = ?
                                    WHERE chunk_id = ?
                                """, (embedding_bytes, datetime.now().isoformat(), chunk['chunk_id']))
                                
                                # Update vector table if it exists
                                try:
                                    embedding_json = f"[{','.join(map(str, embedding.tolist()))}]"
                                    cursor.execute("""
                                        UPDATE embeddings_vec SET embedding = ?
                                        WHERE chunk_id = ?
                                    """, (embedding_json, chunk['chunk_id']))
                                except sqlite3.OperationalError:
                                    pass  # Vector table may not exist
                            
                            stats.documents_processed += 1
                            stats.chunks_processed += len(chunks)
                            stats.embeddings_generated += len(embeddings)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to re-embed document {doc['doc_id']}: {e}")
                            stats.errors += 1
                        
                        pbar.update(1)
                
                conn.commit()
            
            stats.success = True
            self.logger.info(f"Re-embedding complete: {stats.documents_processed} documents, {stats.embeddings_generated} embeddings")
            
        except Exception as e:
            self.logger.error(f"Re-embedding failed: {e}")
            stats.success = False
            stats.details['error'] = str(e)
        
        finally:
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
        
        return stats
    
    def rechunk_documents(
        self,
        collection_id: str = "default",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        reembed: bool = True,
        backup: bool = True
    ) -> ReindexStats:
        """
        Re-chunk all documents in a collection with new parameters.
        
        Args:
            collection_id: Collection to re-chunk
            chunk_size: New chunk size (uses default if None)
            chunk_overlap: New chunk overlap (uses default if None)
            reembed: Whether to regenerate embeddings for new chunks
            backup: Whether to backup database before operation
            
        Returns:
            ReindexStats with operation results
        """
        self.logger.info(f"Starting re-chunking for collection: {collection_id}")
        
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        stats = ReindexStats(
            operation="rechunk",
            collection_id=collection_id,
            start_time=datetime.now(),
            details={
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'reembed': reembed
            }
        )
        
        try:
            # Create backup if requested
            if backup:
                backup_path = self._backup_database()
                stats.details['backup_path'] = str(backup_path)
            
            # Get all documents in collection
            documents = self._get_documents_in_collection(collection_id)
            stats.details['total_documents'] = len(documents)
            
            if not documents:
                self.logger.warning(f"No documents found in collection: {collection_id}")
                stats.success = True
                return stats
            
            # Create new chunker with specified parameters
            chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                with tqdm(total=len(documents), desc="Re-chunking documents") as pbar:
                    for doc in documents:
                        try:
                            # Get original document content by concatenating chunks
                            old_chunks = self._get_document_chunks(doc['doc_id'], collection_id)
                            if not old_chunks:
                                continue
                            
                            original_content = '\n'.join(chunk['content'] for chunk in old_chunks)
                            
                            # Generate new chunks
                            new_chunks = chunker.chunk_text(original_content)
                            
                            # Delete old chunks and embeddings
                            chunk_ids = [chunk['chunk_id'] for chunk in old_chunks]
                            if chunk_ids:
                                placeholders = ','.join(['?'] * len(chunk_ids))
                                
                                # Delete embeddings
                                cursor.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
                                
                                # Delete from vector table if exists
                                try:
                                    cursor.execute(f"DELETE FROM embeddings_vec WHERE chunk_id IN ({placeholders})", chunk_ids)
                                except sqlite3.OperationalError:
                                    pass
                                
                                # Delete chunks
                                cursor.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
                            
                            # Insert new chunks
                            for i, chunk_content in enumerate(new_chunks):
                                chunk_id = f"{doc['doc_id']}_chunk_{i}"
                                
                                # Insert chunk
                                cursor.execute("""
                                    INSERT INTO chunks 
                                    (chunk_id, doc_id, chunk_index, content, token_count, metadata_json, created_at, collection_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    chunk_id,
                                    doc['doc_id'],
                                    i,
                                    chunk_content,
                                    len(chunk_content.split()),  # Rough token count
                                    '{}',
                                    datetime.now().isoformat(),
                                    collection_id
                                ))
                                
                                # Generate embedding if requested
                                if reembed:
                                    embedding = self.embedding_service.embed_text(chunk_content)
                                    embedding_bytes = embedding.astype(np.float32).tobytes()
                                    
                                    # Insert embedding
                                    cursor.execute("""
                                        INSERT INTO embeddings (chunk_id, embedding_vector, created_at, collection_id)
                                        VALUES (?, ?, ?, ?)
                                    """, (chunk_id, embedding_bytes, datetime.now().isoformat(), collection_id))
                                    
                                    # Insert into vector table if exists
                                    try:
                                        embedding_json = f"[{','.join(map(str, embedding.tolist()))}]"
                                        cursor.execute("""
                                            INSERT INTO embeddings_vec (chunk_id, embedding)
                                            VALUES (?, ?)
                                        """, (chunk_id, embedding_json))
                                    except sqlite3.OperationalError:
                                        pass
                                    
                                    stats.embeddings_generated += 1
                            
                            # Update document total_chunks
                            cursor.execute("""
                                UPDATE documents SET total_chunks = ? WHERE doc_id = ? AND collection_id = ?
                            """, (len(new_chunks), doc['doc_id'], collection_id))
                            
                            stats.documents_processed += 1
                            stats.chunks_processed += len(new_chunks)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to re-chunk document {doc['doc_id']}: {e}")
                            stats.errors += 1
                        
                        pbar.update(1)
                
                conn.commit()
            
            stats.success = True
            self.logger.info(f"Re-chunking complete: {stats.documents_processed} documents, {stats.chunks_processed} new chunks")
            
        except Exception as e:
            self.logger.error(f"Re-chunking failed: {e}")
            stats.success = False
            stats.details['error'] = str(e)
        
        finally:
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
        
        return stats
    
    def rebuild_indices(self, backup: bool = True) -> ReindexStats:
        """
        Rebuild all database indices for performance optimization.
        
        Args:
            backup: Whether to backup database before operation
            
        Returns:
            ReindexStats with operation results
        """
        self.logger.info("Starting index rebuild")
        
        stats = ReindexStats(
            operation="rebuild_indices",
            collection_id="all",
            start_time=datetime.now()
        )
        
        try:
            # Create backup if requested
            if backup:
                backup_path = self._backup_database()
                stats.details['backup_path'] = str(backup_path)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Drop and recreate standard indices
                indices = [
                    "DROP INDEX IF EXISTS idx_chunks_doc_id",
                    "DROP INDEX IF EXISTS idx_chunks_chunk_index", 
                    "DROP INDEX IF EXISTS idx_documents_source",
                    "DROP INDEX IF EXISTS idx_documents_collection",
                    "DROP INDEX IF EXISTS idx_chunks_collection",
                    "DROP INDEX IF EXISTS idx_embeddings_collection",
                    
                    "CREATE INDEX idx_chunks_doc_id ON chunks (doc_id)",
                    "CREATE INDEX idx_chunks_chunk_index ON chunks (doc_id, chunk_index)",
                    "CREATE INDEX idx_documents_source ON documents (source_path)",
                    "CREATE INDEX idx_documents_collection ON documents (collection_id)",
                    "CREATE INDEX idx_chunks_collection ON chunks (collection_id)",
                    "CREATE INDEX idx_embeddings_collection ON embeddings (collection_id)"
                ]
                
                for sql in indices:
                    cursor.execute(sql)
                
                # Rebuild FTS index
                cursor.execute("DELETE FROM chunks_fts")
                cursor.execute("INSERT INTO chunks_fts(chunk_id, content) SELECT chunk_id, content FROM chunks")
                
                # Rebuild vector table if it exists
                try:
                    cursor.execute("DELETE FROM embeddings_vec")
                    
                    # Re-populate vector table from embeddings table
                    cursor.execute("""
                        SELECT e.chunk_id, e.embedding_vector 
                        FROM embeddings e
                        JOIN chunks c ON e.chunk_id = c.chunk_id
                    """)
                    
                    for row in cursor.fetchall():
                        chunk_id, embedding_bytes = row
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        embedding_json = f"[{','.join(map(str, embedding.tolist()))}]"
                        
                        cursor.execute("INSERT INTO embeddings_vec (chunk_id, embedding) VALUES (?, ?)", 
                                     (chunk_id, embedding_json))
                    
                    stats.details['vector_table_rebuilt'] = True
                    
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not rebuild vector table: {e}")
                    stats.details['vector_table_rebuilt'] = False
                
                conn.commit()
            
            stats.success = True
            self.logger.info("Index rebuild complete")
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {e}")
            stats.success = False
            stats.details['error'] = str(e)
        
        finally:
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
        
        return stats
    
    def vacuum_database(self, backup: bool = True) -> ReindexStats:
        """
        Optimize database storage by running VACUUM operation.
        
        Args:
            backup: Whether to backup database before operation
            
        Returns:
            ReindexStats with operation results
        """
        self.logger.info("Starting database optimization (VACUUM)")
        
        stats = ReindexStats(
            operation="vacuum",
            collection_id="all",
            start_time=datetime.now()
        )
        
        try:
            # Get initial database size
            initial_size = self.db_path.stat().st_size
            stats.details['initial_size_mb'] = initial_size / (1024 * 1024)
            
            # Create backup if requested
            if backup:
                backup_path = self._backup_database()
                stats.details['backup_path'] = str(backup_path)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Run VACUUM to reclaim space and optimize storage
                conn.execute("VACUUM")
                
                # Run ANALYZE to update query planner statistics
                conn.execute("ANALYZE")
            
            # Get final database size
            final_size = self.db_path.stat().st_size
            stats.details['final_size_mb'] = final_size / (1024 * 1024)
            stats.details['space_saved_mb'] = (initial_size - final_size) / (1024 * 1024)
            
            stats.success = True
            self.logger.info(f"Database optimization complete: {stats.details['space_saved_mb']:.2f}MB saved")
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            stats.success = False
            stats.details['error'] = str(e)
        
        finally:
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
        
        return stats
    
    def validate_integrity(self, collection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate database integrity and check for corruption.
        
        Args:
            collection_id: Specific collection to check (all if None)
            
        Returns:
            Integrity validation report
        """
        self.logger.info("Starting integrity validation")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'collection_id': collection_id or 'all',
            'checks_passed': 0,
            'checks_failed': 0,
            'issues': [],
            'statistics': {}
        }
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check 1: Foreign key constraints
            try:
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                if fk_violations:
                    report['issues'].append(f"Foreign key violations: {len(fk_violations)}")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1
            except Exception as e:
                report['issues'].append(f"Foreign key check failed: {e}")
                report['checks_failed'] += 1
            
            # Check 2: Database integrity
            try:
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result == "ok":
                    report['checks_passed'] += 1
                else:
                    report['issues'].append(f"Integrity check failed: {integrity_result}")
                    report['checks_failed'] += 1
            except Exception as e:
                report['issues'].append(f"Integrity check failed: {e}")
                report['checks_failed'] += 1
            
            # Check 3: Orphaned chunks (chunks without documents)
            try:
                where_clause = f"AND c.collection_id = '{collection_id}'" if collection_id else ""
                cursor.execute(f"""
                    SELECT COUNT(*) FROM chunks c 
                    LEFT JOIN documents d ON c.doc_id = d.doc_id 
                    WHERE d.doc_id IS NULL {where_clause}
                """)
                orphaned_chunks = cursor.fetchone()[0]
                report['statistics']['orphaned_chunks'] = orphaned_chunks
                if orphaned_chunks > 0:
                    report['issues'].append(f"Orphaned chunks: {orphaned_chunks}")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1
            except Exception as e:
                report['issues'].append(f"Orphaned chunks check failed: {e}")
                report['checks_failed'] += 1
            
            # Check 4: Missing embeddings (chunks without embeddings)
            try:
                where_clause = f"AND c.collection_id = '{collection_id}'" if collection_id else ""
                cursor.execute(f"""
                    SELECT COUNT(*) FROM chunks c
                    LEFT JOIN embeddings e ON c.chunk_id = e.chunk_id
                    WHERE e.chunk_id IS NULL {where_clause}
                """)
                missing_embeddings = cursor.fetchone()[0]
                report['statistics']['missing_embeddings'] = missing_embeddings
                if missing_embeddings > 0:
                    report['issues'].append(f"Missing embeddings: {missing_embeddings}")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1
            except Exception as e:
                report['issues'].append(f"Missing embeddings check failed: {e}")
                report['checks_failed'] += 1
            
            # Check 5: Inconsistent chunk counts
            try:
                where_clause = f"WHERE collection_id = '{collection_id}'" if collection_id else ""
                cursor.execute(f"""
                    SELECT d.doc_id, d.total_chunks, COUNT(c.chunk_id) as actual_chunks
                    FROM documents d
                    LEFT JOIN chunks c ON d.doc_id = c.doc_id
                    {where_clause}
                    GROUP BY d.doc_id, d.total_chunks
                    HAVING d.total_chunks != COUNT(c.chunk_id)
                """)
                
                inconsistent_counts = cursor.fetchall()
                report['statistics']['inconsistent_chunk_counts'] = len(inconsistent_counts)
                if inconsistent_counts:
                    report['issues'].append(f"Inconsistent chunk counts: {len(inconsistent_counts)} documents")
                    report['checks_failed'] += 1
                else:
                    report['checks_passed'] += 1
            except Exception as e:
                report['issues'].append(f"Chunk count check failed: {e}")
                report['checks_failed'] += 1
        
        report['overall_status'] = 'PASS' if report['checks_failed'] == 0 else 'FAIL'
        
        self.logger.info(f"Integrity validation complete: {report['overall_status']} ({report['checks_passed']} passed, {report['checks_failed']} failed)")
        return report
    
    def repair_database(self, collection_id: Optional[str] = None, backup: bool = True) -> ReindexStats:
        """
        Attempt to repair database issues found during integrity validation.
        
        Args:
            collection_id: Specific collection to repair (all if None)
            backup: Whether to backup database before operation
            
        Returns:
            ReindexStats with repair results
        """
        self.logger.info("Starting database repair")
        
        stats = ReindexStats(
            operation="repair",
            collection_id=collection_id or "all",
            start_time=datetime.now()
        )
        
        try:
            # Create backup if requested
            if backup:
                backup_path = self._backup_database()
                stats.details['backup_path'] = str(backup_path)
            
            # First run validation to identify issues
            validation_report = self.validate_integrity(collection_id)
            stats.details['initial_validation'] = validation_report
            
            repairs_made = []
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Repair 1: Remove orphaned chunks
                where_clause = f"AND c.collection_id = '{collection_id}'" if collection_id else ""
                cursor.execute(f"""
                    DELETE FROM chunks WHERE chunk_id IN (
                        SELECT c.chunk_id FROM chunks c 
                        LEFT JOIN documents d ON c.doc_id = d.doc_id 
                        WHERE d.doc_id IS NULL {where_clause}
                    )
                """)
                orphaned_removed = cursor.rowcount
                if orphaned_removed > 0:
                    repairs_made.append(f"Removed {orphaned_removed} orphaned chunks")
                
                # Repair 2: Fix inconsistent chunk counts
                where_clause = f"WHERE d.collection_id = '{collection_id}'" if collection_id else ""
                cursor.execute(f"""
                    UPDATE documents SET total_chunks = (
                        SELECT COUNT(*) FROM chunks c WHERE c.doc_id = documents.doc_id
                    ) {where_clause}
                """)
                counts_fixed = cursor.rowcount
                if counts_fixed > 0:
                    repairs_made.append(f"Fixed chunk counts for {counts_fixed} documents")
                
                conn.commit()
            
            # Run rebuild indices and vacuum for optimization
            rebuild_stats = self.rebuild_indices(backup=False)
            if rebuild_stats.success:
                repairs_made.append("Rebuilt database indices")
            
            vacuum_stats = self.vacuum_database(backup=False)
            if vacuum_stats.success:
                repairs_made.append(f"Optimized database ({vacuum_stats.details.get('space_saved_mb', 0):.2f}MB saved)")
            
            # Final validation
            final_validation = self.validate_integrity(collection_id)
            stats.details['final_validation'] = final_validation
            stats.details['repairs_made'] = repairs_made
            
            stats.success = final_validation['checks_failed'] < validation_report['checks_failed']
            
            self.logger.info(f"Database repair complete: {len(repairs_made)} repairs made")
            
        except Exception as e:
            self.logger.error(f"Database repair failed: {e}")
            stats.success = False
            stats.details['error'] = str(e)
        
        finally:
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
        
        return stats


# Utility functions

def create_reindex_tool(
    db_path: str = "data/rag_vectors.db",
    **kwargs
) -> ReindexTool:
    """
    Factory function to create a ReindexTool instance.
    
    Args:
        db_path: Path to vector database
        **kwargs: Additional arguments for ReindexTool
        
    Returns:
        Configured ReindexTool instance
    """
    return ReindexTool(db_path=db_path, **kwargs)


def quick_maintenance(
    db_path: str = "data/rag_vectors.db",
    backup: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for quick database maintenance.
    
    Args:
        db_path: Vector database path
        backup: Whether to backup before operations
        
    Returns:
        Maintenance report
    """
    tool = create_reindex_tool(db_path=db_path)
    
    # Run validation
    validation = tool.validate_integrity()
    
    # Run optimization
    vacuum_stats = tool.vacuum_database(backup=backup)
    
    return {
        'validation': validation,
        'optimization': vacuum_stats.to_dict()
    }
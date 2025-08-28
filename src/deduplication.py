"""
Deduplication - Intelligent duplicate detection for document corpus

This module provides comprehensive duplicate detection capabilities including:
- Content hash comparison for exact duplicates
- Fuzzy matching for near-duplicates using MinHash LSH
- Semantic similarity using embeddings
- Metadata-based detection (source, date, size)
- User confirmation workflows for ambiguous cases
"""

import hashlib
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

import numpy as np
from datasketch import MinHashLSH, MinHash

from .vector_database import VectorDatabase
from .embedding_service import EmbeddingService


@dataclass
class DuplicateGroup:
    """Group of documents identified as duplicates"""
    group_id: str
    detection_method: str  # 'exact', 'fuzzy', 'semantic', 'metadata'
    similarity_score: float
    documents: List[Dict[str, Any]]
    recommended_action: str  # 'keep_first', 'manual_review', 'merge'
    confidence: float


@dataclass
class DeduplicationReport:
    """Report of deduplication analysis"""
    total_documents: int
    unique_documents: int
    duplicate_groups: List[DuplicateGroup]
    exact_duplicates: int
    fuzzy_duplicates: int
    semantic_duplicates: int
    metadata_duplicates: int
    space_saved_mb: float
    processing_time: float


class DocumentDeduplicator:
    """
    Advanced document deduplication system with multiple detection strategies.
    
    Features:
    - Exact duplicate detection via SHA-256 content hashing
    - Near-duplicate detection using MinHash LSH
    - Semantic similarity using document embeddings
    - Metadata-based detection (filename, size, modification date)
    - Configurable similarity thresholds
    - Interactive confirmation workflow
    """
    
    def __init__(
        self,
        db_path: str = "data/rag_vectors.db",
        embedding_service: Optional[EmbeddingService] = None,
        fuzzy_threshold: float = 0.8,
        semantic_threshold: float = 0.95,
        minhash_threshold: float = 0.8,
        num_perm: int = 128
    ):
        """
        Initialize deduplicator with configuration.
        
        Args:
            db_path: Path to vector database
            embedding_service: Embedding service for semantic comparison
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
            semantic_threshold: Threshold for semantic similarity (0-1)
            minhash_threshold: Threshold for MinHash LSH (0-1)
            num_perm: Number of permutations for MinHash
        """
        self.db_path = Path(db_path)
        self.db = VectorDatabase(str(db_path))
        self.embedding_service = embedding_service or EmbeddingService()
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.minhash_threshold = minhash_threshold
        self.num_perm = num_perm
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MinHash LSH
        self.lsh = MinHashLSH(threshold=minhash_threshold, num_perm=num_perm)
    
    def _get_document_content(self, doc_id: str, collection_id: str = "default") -> Optional[str]:
        """Get full document content by concatenating chunks."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content FROM chunks 
                WHERE doc_id = ? AND collection_id = ?
                ORDER BY chunk_index
            """, (doc_id, collection_id))
            
            chunks = cursor.fetchall()
            if not chunks:
                return None
            
            return '\n'.join(chunk[0] for chunk in chunks)
    
    def _get_document_metadata(self, doc_id: str, collection_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source_path, ingested_at, metadata_json, content_hash, file_size, total_chunks
                FROM documents WHERE doc_id = ? AND collection_id = ?
            """, (doc_id, collection_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'doc_id': doc_id,
                'source_path': row[0],
                'ingested_at': row[1],
                'metadata_json': row[2],
                'content_hash': row[3],
                'file_size': row[4],
                'total_chunks': row[5]
            }
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _create_minhash(self, content: str) -> MinHash:
        """Create MinHash signature for content."""
        minhash = MinHash(num_perm=self.num_perm)
        
        # Tokenize content (simple word-based tokenization)
        words = content.lower().split()
        for word in words:
            minhash.update(word.encode('utf-8'))
        
        return minhash
    
    def _get_document_embedding(self, doc_id: str, collection_id: str = "default") -> Optional[np.ndarray]:
        """Get document embedding by averaging chunk embeddings."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.embedding_vector FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                WHERE c.doc_id = ? AND c.collection_id = ?
            """, (doc_id, collection_id))
            
            embeddings = []
            for row in cursor.fetchall():
                embedding_bytes = row[0]
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
            
            if not embeddings:
                return None
            
            # Average embeddings to get document-level representation
            return np.mean(embeddings, axis=0)
    
    def detect_exact_duplicates(self, collection_id: str = "default") -> List[DuplicateGroup]:
        """Detect exact duplicates using content hashes."""
        self.logger.info("Detecting exact duplicates...")
        
        hash_to_docs = defaultdict(list)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Group documents by content hash
        for doc_id in doc_ids:
            content = self._get_document_content(doc_id, collection_id)
            if content:
                content_hash = self._compute_content_hash(content)
                metadata = self._get_document_metadata(doc_id, collection_id)
                if metadata:
                    metadata['content'] = content
                    hash_to_docs[content_hash].append(metadata)
        
        # Create duplicate groups for hashes with multiple documents
        duplicate_groups = []
        for content_hash, docs in hash_to_docs.items():
            if len(docs) > 1:
                group = DuplicateGroup(
                    group_id=f"exact_{content_hash[:8]}",
                    detection_method="exact",
                    similarity_score=1.0,
                    documents=docs,
                    recommended_action="keep_first",
                    confidence=1.0
                )
                duplicate_groups.append(group)
        
        self.logger.info(f"Found {len(duplicate_groups)} exact duplicate groups")
        return duplicate_groups
    
    def detect_fuzzy_duplicates(self, collection_id: str = "default") -> List[DuplicateGroup]:
        """Detect near-duplicates using MinHash LSH."""
        self.logger.info("Detecting fuzzy duplicates...")
        
        # Clear previous LSH data
        self.lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.num_perm)
        
        doc_to_minhash = {}
        doc_metadata = {}
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Create MinHash signatures for all documents
        for doc_id in doc_ids:
            content = self._get_document_content(doc_id, collection_id)
            metadata = self._get_document_metadata(doc_id, collection_id)
            
            if content and metadata:
                minhash = self._create_minhash(content)
                doc_to_minhash[doc_id] = minhash
                doc_metadata[doc_id] = metadata
                doc_metadata[doc_id]['content'] = content
                
                # Insert into LSH
                self.lsh.insert(doc_id, minhash)
        
        # Find similar documents
        duplicate_groups = []
        processed_docs = set()
        
        for doc_id, minhash in doc_to_minhash.items():
            if doc_id in processed_docs:
                continue
            
            # Query for similar documents
            similar_docs = self.lsh.query(minhash)
            similar_docs = [d for d in similar_docs if d != doc_id]  # Exclude self
            
            if similar_docs:
                # Calculate actual similarity scores
                group_docs = [doc_metadata[doc_id]]
                similarities = []
                
                for similar_doc in similar_docs:
                    if similar_doc not in processed_docs:
                        similar_minhash = doc_to_minhash[similar_doc]
                        similarity = minhash.jaccard(similar_minhash)
                        
                        if similarity >= self.fuzzy_threshold:
                            group_docs.append(doc_metadata[similar_doc])
                            similarities.append(similarity)
                            processed_docs.add(similar_doc)
                
                if len(group_docs) > 1:
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    
                    group = DuplicateGroup(
                        group_id=f"fuzzy_{doc_id[:8]}",
                        detection_method="fuzzy",
                        similarity_score=avg_similarity,
                        documents=group_docs,
                        recommended_action="manual_review" if avg_similarity < 0.9 else "keep_first",
                        confidence=avg_similarity
                    )
                    duplicate_groups.append(group)
                    processed_docs.add(doc_id)
        
        self.logger.info(f"Found {len(duplicate_groups)} fuzzy duplicate groups")
        return duplicate_groups
    
    def detect_semantic_duplicates(self, collection_id: str = "default") -> List[DuplicateGroup]:
        """Detect semantic duplicates using embedding similarity."""
        self.logger.info("Detecting semantic duplicates...")
        
        doc_embeddings = {}
        doc_metadata = {}
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Get embeddings for all documents
        for doc_id in doc_ids:
            embedding = self._get_document_embedding(doc_id, collection_id)
            metadata = self._get_document_metadata(doc_id, collection_id)
            
            if embedding is not None and metadata:
                doc_embeddings[doc_id] = embedding
                doc_metadata[doc_id] = metadata
        
        # Find similar document pairs
        duplicate_groups = []
        processed_docs = set()
        
        doc_list = list(doc_embeddings.keys())
        embeddings_matrix = np.array([doc_embeddings[doc_id] for doc_id in doc_list])
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
        
        for i, doc_id_i in enumerate(doc_list):
            if doc_id_i in processed_docs:
                continue
            
            similar_docs = []
            similarities = []
            
            for j, doc_id_j in enumerate(doc_list):
                if i != j and doc_id_j not in processed_docs:
                    similarity = similarity_matrix[i, j]
                    if similarity >= self.semantic_threshold:
                        similar_docs.append(doc_id_j)
                        similarities.append(similarity)
            
            if similar_docs:
                group_docs = [doc_metadata[doc_id_i]]
                for doc_id in similar_docs:
                    group_docs.append(doc_metadata[doc_id])
                    processed_docs.add(doc_id)
                
                avg_similarity = np.mean(similarities)
                
                group = DuplicateGroup(
                    group_id=f"semantic_{doc_id_i[:8]}",
                    detection_method="semantic",
                    similarity_score=avg_similarity,
                    documents=group_docs,
                    recommended_action="manual_review",
                    confidence=avg_similarity
                )
                duplicate_groups.append(group)
                processed_docs.add(doc_id_i)
        
        self.logger.info(f"Found {len(duplicate_groups)} semantic duplicate groups")
        return duplicate_groups
    
    def detect_metadata_duplicates(self, collection_id: str = "default") -> List[DuplicateGroup]:
        """Detect duplicates based on metadata (filename, size)."""
        self.logger.info("Detecting metadata duplicates...")
        
        # Group by filename and file size
        metadata_groups = defaultdict(list)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, source_path, file_size FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            for row in cursor.fetchall():
                doc_id, source_path, file_size = row
                filename = Path(source_path).name
                key = (filename, file_size)
                
                metadata = self._get_document_metadata(doc_id, collection_id)
                if metadata:
                    metadata_groups[key].append(metadata)
        
        # Create duplicate groups
        duplicate_groups = []
        for (filename, file_size), docs in metadata_groups.items():
            if len(docs) > 1:
                group = DuplicateGroup(
                    group_id=f"metadata_{hash((filename, file_size)) % 10000:04d}",
                    detection_method="metadata",
                    similarity_score=1.0,  # Same filename + size = high confidence
                    documents=docs,
                    recommended_action="manual_review",
                    confidence=0.8
                )
                duplicate_groups.append(group)
        
        self.logger.info(f"Found {len(duplicate_groups)} metadata duplicate groups")
        return duplicate_groups
    
    def analyze_duplicates(self, collection_id: str = "default") -> DeduplicationReport:
        """Comprehensive duplicate analysis using all detection methods."""
        self.logger.info(f"Starting comprehensive duplicate analysis for collection: {collection_id}")
        
        start_time = datetime.now()
        
        # Get total document count
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            total_docs = cursor.fetchone()[0]
        
        # Run all detection methods
        exact_groups = self.detect_exact_duplicates(collection_id)
        fuzzy_groups = self.detect_fuzzy_duplicates(collection_id)
        semantic_groups = self.detect_semantic_duplicates(collection_id)
        metadata_groups = self.detect_metadata_duplicates(collection_id)
        
        # Combine all groups
        all_groups = exact_groups + fuzzy_groups + semantic_groups + metadata_groups
        
        # Calculate statistics
        duplicate_docs = set()
        for group in all_groups:
            for doc in group.documents:
                duplicate_docs.add(doc['doc_id'])
        
        unique_docs = total_docs - len(duplicate_docs)
        
        # Estimate space savings (very rough estimate)
        total_size = 0
        duplicate_size = 0
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE collection_id = ?", (collection_id,))
            total_size = cursor.fetchone()[0] or 0
            
            if duplicate_docs:
                placeholders = ','.join(['?'] * len(duplicate_docs))
                cursor.execute(f"""
                    SELECT SUM(file_size) FROM documents 
                    WHERE collection_id = ? AND doc_id IN ({placeholders})
                """, [collection_id] + list(duplicate_docs))
                duplicate_size = cursor.fetchone()[0] or 0
        
        space_saved_mb = duplicate_size / (1024 * 1024) if duplicate_size else 0
        processing_time = (datetime.now() - start_time).total_seconds()
        
        report = DeduplicationReport(
            total_documents=total_docs,
            unique_documents=unique_docs,
            duplicate_groups=all_groups,
            exact_duplicates=len(exact_groups),
            fuzzy_duplicates=len(fuzzy_groups),
            semantic_duplicates=len(semantic_groups),
            metadata_duplicates=len(metadata_groups),
            space_saved_mb=space_saved_mb,
            processing_time=processing_time
        )
        
        self.logger.info(f"Duplicate analysis complete: {len(all_groups)} groups, {len(duplicate_docs)} duplicates")
        return report
    
    def resolve_duplicates(
        self,
        duplicate_groups: List[DuplicateGroup],
        action_map: Optional[Dict[str, str]] = None,
        collection_id: str = "default",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Resolve duplicate groups by removing or merging documents.
        
        Args:
            duplicate_groups: List of duplicate groups to resolve
            action_map: Manual action mapping {group_id: action}
            collection_id: Target collection
            dry_run: Preview mode without making changes
            
        Returns:
            Resolution statistics
        """
        if dry_run:
            self.logger.info("DRY RUN MODE - No changes will be made")
        
        stats = {
            'groups_processed': 0,
            'documents_removed': 0,
            'documents_kept': 0,
            'space_saved_mb': 0.0,
            'actions_taken': defaultdict(int)
        }
        
        for group in duplicate_groups:
            action = 'keep_first'  # Default action
            
            # Use manual action if provided
            if action_map and group.group_id in action_map:
                action = action_map[group.group_id]
            elif group.recommended_action:
                action = group.recommended_action
            
            stats['actions_taken'][action] += 1
            
            if action == 'keep_first' and len(group.documents) > 1:
                # Keep first document, remove others
                docs_to_remove = group.documents[1:]
                
                for doc in docs_to_remove:
                    if not dry_run:
                        self._remove_document(doc['doc_id'], collection_id)
                    
                    stats['documents_removed'] += 1
                    stats['space_saved_mb'] += (doc.get('file_size', 0) or 0) / (1024 * 1024)
                
                stats['documents_kept'] += 1
            
            elif action == 'remove_all':
                # Remove all documents in group
                for doc in group.documents:
                    if not dry_run:
                        self._remove_document(doc['doc_id'], collection_id)
                    
                    stats['documents_removed'] += 1
                    stats['space_saved_mb'] += (doc.get('file_size', 0) or 0) / (1024 * 1024)
            
            elif action == 'manual_review':
                self.logger.info(f"Group {group.group_id} requires manual review")
                stats['documents_kept'] += len(group.documents)
            
            stats['groups_processed'] += 1
        
        self.logger.info(f"Resolution complete: {stats['documents_removed']} documents removed, {stats['space_saved_mb']:.2f}MB saved")
        return dict(stats)
    
    def _remove_document(self, doc_id: str, collection_id: str):
        """Remove a document and all its associated data."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Remove embeddings
            cursor.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ? AND collection_id = ?)", (doc_id, collection_id))
            
            # Remove from vector table if it exists
            try:
                cursor.execute("DELETE FROM embeddings_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ? AND collection_id = ?)", (doc_id, collection_id))
            except sqlite3.OperationalError:
                pass  # Table may not exist
            
            # Remove chunks
            cursor.execute("DELETE FROM chunks WHERE doc_id = ? AND collection_id = ?", (doc_id, collection_id))
            
            # Remove document
            cursor.execute("DELETE FROM documents WHERE doc_id = ? AND collection_id = ?", (doc_id, collection_id))
            
            conn.commit()


# Utility functions

def create_deduplicator(
    db_path: str = "data/rag_vectors.db",
    **kwargs
) -> DocumentDeduplicator:
    """
    Factory function to create a DocumentDeduplicator instance.
    
    Args:
        db_path: Path to vector database
        **kwargs: Additional arguments for DocumentDeduplicator
        
    Returns:
        Configured DocumentDeduplicator instance
    """
    return DocumentDeduplicator(db_path=db_path, **kwargs)


def quick_dedupe(
    collection_id: str = "default",
    db_path: str = "data/rag_vectors.db",
    dry_run: bool = False
) -> DeduplicationReport:
    """
    Convenience function for quick deduplication analysis.
    
    Args:
        collection_id: Collection to analyze
        db_path: Vector database path
        dry_run: Preview mode
        
    Returns:
        Deduplication report
    """
    deduplicator = create_deduplicator(db_path=db_path)
    return deduplicator.analyze_duplicates(collection_id)
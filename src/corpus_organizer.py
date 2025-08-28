"""
Corpus Organizer - Collection and namespace management for document corpus

This module provides comprehensive corpus organization capabilities including:
- Named collection creation and management
- Document tagging with metadata
- Collection switching and cross-collection search
- Collection merging and export/import
- Hierarchical namespace organization
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict

from .vector_database import VectorDatabase


@dataclass
class Collection:
    """Data structure representing a document collection"""
    collection_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    chunk_count: int = 0
    size_mb: float = 0.0
    tags: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['tags'] = list(self.tags)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Collection':
        data = data.copy()
        data['tags'] = set(data.get('tags', []))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class CorpusOrganizer:
    """
    Advanced corpus organization system with collection management.
    
    Features:
    - Named collection creation and management
    - Document organization by collections/namespaces
    - Cross-collection search and filtering
    - Collection statistics and analytics
    - Export/import capabilities
    - Hierarchical organization support
    """
    
    def __init__(self, db_path: str = "data/rag_vectors.db"):
        """
        Initialize corpus organizer.
        
        Args:
            db_path: Path to vector database
        """
        self.db_path = Path(db_path)
        self.db = VectorDatabase(str(db_path))
        self.logger = logging.getLogger(__name__)
        self.current_collection = "default"
        
        # Initialize collection schema
        self._init_collection_schema()
    
    def _init_collection_schema(self):
        """Initialize collection management schema in database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create collections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    size_mb REAL DEFAULT 0.0,
                    tags_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            
            # Add collection_id to existing tables if not present
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN collection_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE chunks ADD COLUMN collection_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE embeddings ADD COLUMN collection_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Create indexes for collection filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks (collection_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings (collection_id)")
            
            # Create default collection if it doesn't exist
            cursor.execute("""
                INSERT OR IGNORE INTO collections 
                (collection_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                'default',
                'Default Collection',
                'Default document collection',
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            self.logger.info("Collection schema initialized")
    
    def create_collection(
        self,
        name: str,
        description: str = "",
        collection_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new document collection.
        
        Args:
            name: Human-readable collection name
            description: Collection description
            collection_id: Unique collection ID (auto-generated if None)
            tags: List of tags for organization
            metadata: Additional metadata
            
        Returns:
            Collection ID
        """
        if collection_id is None:
            # Generate collection ID from name
            collection_id = name.lower().replace(' ', '_').replace('-', '_')
            # Ensure uniqueness
            counter = 1
            original_id = collection_id
            while self.collection_exists(collection_id):
                collection_id = f"{original_id}_{counter}"
                counter += 1
        
        if self.collection_exists(collection_id):
            raise ValueError(f"Collection '{collection_id}' already exists")
        
        now = datetime.now()
        collection = Collection(
            collection_id=collection_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            tags=set(tags or []),
            metadata=metadata or {}
        )
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collections 
                (collection_id, name, description, created_at, updated_at, tags_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                collection.collection_id,
                collection.name,
                collection.description,
                collection.created_at.isoformat(),
                collection.updated_at.isoformat(),
                json.dumps(list(collection.tags)),
                json.dumps(collection.metadata)
            ))
            conn.commit()
        
        self.logger.info(f"Created collection '{collection_id}': {name}")
        return collection_id
    
    def collection_exists(self, collection_id: str) -> bool:
        """Check if a collection exists."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM collections WHERE collection_id = ?", (collection_id,))
            return cursor.fetchone() is not None
    
    def get_collection(self, collection_id: str) -> Optional[Collection]:
        """Get collection information by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT collection_id, name, description, created_at, updated_at,
                       document_count, chunk_count, size_mb, tags_json, metadata_json
                FROM collections WHERE collection_id = ?
            """, (collection_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Collection(
                collection_id=row[0],
                name=row[1],
                description=row[2],
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                document_count=row[5],
                chunk_count=row[6],
                size_mb=row[7],
                tags=set(json.loads(row[8] or '[]')),
                metadata=json.loads(row[9] or '{}')
            )
    
    def list_collections(self) -> List[Collection]:
        """List all collections."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT collection_id, name, description, created_at, updated_at,
                       document_count, chunk_count, size_mb, tags_json, metadata_json
                FROM collections ORDER BY created_at
            """)
            
            collections = []
            for row in cursor.fetchall():
                collection = Collection(
                    collection_id=row[0],
                    name=row[1],
                    description=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]),
                    document_count=row[5],
                    chunk_count=row[6],
                    size_mb=row[7],
                    tags=set(json.loads(row[8] or '[]')),
                    metadata=json.loads(row[9] or '{}')
                )
                collections.append(collection)
            
            return collections
    
    def switch_collection(self, collection_id: str):
        """Switch to a different collection as the current working collection."""
        if not self.collection_exists(collection_id):
            raise ValueError(f"Collection '{collection_id}' does not exist")
        
        self.current_collection = collection_id
        self.logger.info(f"Switched to collection: {collection_id}")
    
    def delete_collection(self, collection_id: str, confirm: bool = False) -> bool:
        """
        Delete a collection and all its documents.
        
        Args:
            collection_id: Collection to delete
            confirm: Safety confirmation flag
            
        Returns:
            True if deleted, False if cancelled
        """
        if not confirm:
            collection = self.get_collection(collection_id)
            if collection and (collection.document_count > 0 or collection.chunk_count > 0):
                self.logger.warning(f"Collection '{collection_id}' contains {collection.document_count} documents. Use confirm=True to proceed.")
                return False
        
        if collection_id == "default":
            raise ValueError("Cannot delete the default collection")
        
        if not self.collection_exists(collection_id):
            self.logger.warning(f"Collection '{collection_id}' does not exist")
            return False
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Delete embeddings first (due to foreign key constraints)
            cursor.execute("DELETE FROM embeddings WHERE collection_id = ?", (collection_id,))
            
            # Delete from vector table if it exists
            try:
                cursor.execute("DELETE FROM embeddings_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE collection_id = ?)", (collection_id,))
            except sqlite3.OperationalError:
                pass  # Table may not exist
            
            # Delete chunks
            cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
            
            # Delete documents
            cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            
            # Delete collection record
            cursor.execute("DELETE FROM collections WHERE collection_id = ?", (collection_id,))
            
            conn.commit()
        
        # Switch to default if this was the current collection
        if self.current_collection == collection_id:
            self.current_collection = "default"
        
        self.logger.info(f"Deleted collection: {collection_id}")
        return True
    
    def update_collection_stats(self, collection_id: str):
        """Update collection statistics (document count, chunk count, size)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Count documents and chunks
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection_id,))
            chunk_count = cursor.fetchone()[0]
            
            # Calculate approximate size (rough estimate based on content)
            cursor.execute("""
                SELECT SUM(LENGTH(content)) FROM chunks WHERE collection_id = ?
            """, (collection_id,))
            content_size = cursor.fetchone()[0] or 0
            size_mb = content_size / (1024 * 1024)
            
            # Update collection record
            cursor.execute("""
                UPDATE collections 
                SET document_count = ?, chunk_count = ?, size_mb = ?, updated_at = ?
                WHERE collection_id = ?
            """, (doc_count, chunk_count, size_mb, datetime.now().isoformat(), collection_id))
            
            conn.commit()
    
    def merge_collections(self, source_id: str, target_id: str) -> bool:
        """
        Merge source collection into target collection.
        
        Args:
            source_id: Source collection ID
            target_id: Target collection ID
            
        Returns:
            True if successful
        """
        if not self.collection_exists(source_id):
            raise ValueError(f"Source collection '{source_id}' does not exist")
        if not self.collection_exists(target_id):
            raise ValueError(f"Target collection '{target_id}' does not exist")
        if source_id == target_id:
            raise ValueError("Source and target collections cannot be the same")
        
        self.logger.info(f"Merging collection '{source_id}' into '{target_id}'")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Update all records to target collection
            cursor.execute("UPDATE documents SET collection_id = ? WHERE collection_id = ?", (target_id, source_id))
            cursor.execute("UPDATE chunks SET collection_id = ? WHERE collection_id = ?", (target_id, source_id))
            cursor.execute("UPDATE embeddings SET collection_id = ? WHERE collection_id = ?", (target_id, source_id))
            
            # Update vector table if it exists
            try:
                cursor.execute("""
                    UPDATE embeddings_vec SET collection_id = ? 
                    WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE collection_id = ?)
                """, (target_id, target_id))
            except sqlite3.OperationalError:
                pass  # Table may not exist
            
            # Delete source collection
            cursor.execute("DELETE FROM collections WHERE collection_id = ?", (source_id,))
            
            conn.commit()
        
        # Update statistics for target collection
        self.update_collection_stats(target_id)
        
        # Update current collection if needed
        if self.current_collection == source_id:
            self.current_collection = target_id
        
        self.logger.info(f"Merged '{source_id}' into '{target_id}'")
        return True
    
    def tag_documents(self, document_ids: List[str], tags: List[str], collection_id: Optional[str] = None):
        """Add tags to documents (stored in metadata)."""
        if collection_id is None:
            collection_id = self.current_collection
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            for doc_id in document_ids:
                # Get current metadata
                cursor.execute("""
                    SELECT metadata_json FROM documents 
                    WHERE doc_id = ? AND collection_id = ?
                """, (doc_id, collection_id))
                
                row = cursor.fetchone()
                if not row:
                    continue
                
                metadata = json.loads(row[0] or '{}')
                doc_tags = set(metadata.get('tags', []))
                doc_tags.update(tags)
                metadata['tags'] = list(doc_tags)
                
                # Update metadata
                cursor.execute("""
                    UPDATE documents SET metadata_json = ? 
                    WHERE doc_id = ? AND collection_id = ?
                """, (json.dumps(metadata), doc_id, collection_id))
            
            conn.commit()
        
        self.logger.info(f"Tagged {len(document_ids)} documents with tags: {tags}")
    
    def search_by_tags(self, tags: List[str], collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents by tags."""
        if collection_id is None:
            collection_id = self.current_collection
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, source_path, metadata_json 
                FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            matching_docs = []
            for row in cursor.fetchall():
                doc_id, source_path, metadata_json = row
                metadata = json.loads(metadata_json or '{}')
                doc_tags = set(metadata.get('tags', []))
                
                # Check if document has any of the requested tags
                if doc_tags.intersection(set(tags)):
                    matching_docs.append({
                        'doc_id': doc_id,
                        'source_path': source_path,
                        'metadata': metadata,
                        'matching_tags': list(doc_tags.intersection(set(tags)))
                    })
            
            return matching_docs
    
    def export_collection(self, collection_id: str, export_path: str, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export collection to a file for backup or transfer.
        
        Args:
            collection_id: Collection to export
            export_path: Output file path
            include_embeddings: Whether to include embeddings (large)
            
        Returns:
            Export statistics
        """
        if not self.collection_exists(collection_id):
            raise ValueError(f"Collection '{collection_id}' does not exist")
        
        export_data = {
            'collection': self.get_collection(collection_id).to_dict(),
            'documents': [],
            'chunks': [],
            'embeddings': [] if include_embeddings else None,
            'export_timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Export documents
            cursor.execute("""
                SELECT doc_id, source_path, ingested_at, metadata_json, content_hash, file_size, total_chunks
                FROM documents WHERE collection_id = ?
            """, (collection_id,))
            
            for row in cursor.fetchall():
                export_data['documents'].append({
                    'doc_id': row[0],
                    'source_path': row[1],
                    'ingested_at': row[2],
                    'metadata_json': row[3],
                    'content_hash': row[4],
                    'file_size': row[5],
                    'total_chunks': row[6]
                })
            
            # Export chunks
            cursor.execute("""
                SELECT chunk_id, doc_id, chunk_index, content, token_count, metadata_json, created_at
                FROM chunks WHERE collection_id = ?
            """, (collection_id,))
            
            for row in cursor.fetchall():
                export_data['chunks'].append({
                    'chunk_id': row[0],
                    'doc_id': row[1],
                    'chunk_index': row[2],
                    'content': row[3],
                    'token_count': row[4],
                    'metadata_json': row[5],
                    'created_at': row[6]
                })
            
            # Export embeddings if requested
            if include_embeddings:
                cursor.execute("""
                    SELECT chunk_id, embedding_vector, created_at
                    FROM embeddings WHERE collection_id = ?
                """, (collection_id,))
                
                for row in cursor.fetchall():
                    export_data['embeddings'].append({
                        'chunk_id': row[0],
                        'embedding_vector': row[1].hex(),  # Convert binary to hex
                        'created_at': row[2]
                    })
        
        # Write to file
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        stats = {
            'collection_id': collection_id,
            'documents_exported': len(export_data['documents']),
            'chunks_exported': len(export_data['chunks']),
            'embeddings_exported': len(export_data['embeddings']) if include_embeddings else 0,
            'file_size_mb': export_path.stat().st_size / (1024 * 1024),
            'export_path': str(export_path)
        }
        
        self.logger.info(f"Exported collection '{collection_id}' to {export_path}")
        return stats
    
    def get_current_collection(self) -> str:
        """Get the current working collection ID."""
        return self.current_collection
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all collections."""
        collections = self.list_collections()
        
        # Update stats for all collections
        for collection in collections:
            self.update_collection_stats(collection.collection_id)
        
        # Refresh data after stats update
        collections = self.list_collections()
        
        total_docs = sum(c.document_count for c in collections)
        total_chunks = sum(c.chunk_count for c in collections)
        total_size = sum(c.size_mb for c in collections)
        
        return {
            'total_collections': len(collections),
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'total_size_mb': total_size,
            'current_collection': self.current_collection,
            'collections': [c.to_dict() for c in collections]
        }


# Utility functions

def create_corpus_organizer(db_path: str = "data/rag_vectors.db") -> CorpusOrganizer:
    """
    Factory function to create a CorpusOrganizer instance.
    
    Args:
        db_path: Path to vector database
        
    Returns:
        Configured CorpusOrganizer instance
    """
    return CorpusOrganizer(db_path=db_path)
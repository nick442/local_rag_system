"""
Corpus Manager - Bulk document processing and ingestion system

This module provides comprehensive corpus management capabilities including:
- Recursive directory scanning and bulk processing
- Parallel document processing with progress tracking
- Duplicate detection and incremental updates
- Checkpointing and resume functionality
- Statistics reporting and performance monitoring
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import aiofiles
from tqdm import tqdm

from .document_ingestion import DocumentIngestionService, Document, DocumentChunk
from .embedding_service import EmbeddingService
from .vector_database import VectorDatabase


@dataclass
class ProcessingStats:
    """Statistics for document processing operations"""
    files_scanned: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'files_scanned': self.files_scanned,
            'files_processed': self.files_processed,
            'files_skipped': self.files_skipped,
            'files_failed': self.files_failed,
            'chunks_created': self.chunks_created,
            'chunks_embedded': self.chunks_embedded,
            'processing_time': self.processing_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class CheckpointData:
    """Data structure for checkpointing progress"""
    processed_files: Set[str]
    stats: ProcessingStats
    collection_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'processed_files': list(self.processed_files),
            'stats': self.stats.to_dict(),
            'collection_id': self.collection_id,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        # Reconstruct ProcessingStats with proper datetime handling
        stats_data = data['stats'].copy()
        if stats_data.get('start_time'):
            stats_data['start_time'] = datetime.fromisoformat(stats_data['start_time'])
        if stats_data.get('end_time'):
            stats_data['end_time'] = datetime.fromisoformat(stats_data['end_time'])
        
        return cls(
            processed_files=set(data['processed_files']),
            stats=ProcessingStats(**stats_data),
            collection_id=data['collection_id'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


class CorpusManager:
    """
    Comprehensive corpus management system for bulk document processing.
    
    Features:
    - Parallel processing with configurable workers
    - Progress tracking with detailed statistics  
    - Checkpointing for resume on failure
    - Duplicate detection via content hashing
    - Incremental updates and dry-run mode
    """
    
    def __init__(
        self,
        db_path: str = "data/rag_vectors.db",
        embedding_model_path: Optional[str] = None,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        batch_size: int = 32
    ):
        """
        Initialize corpus manager with processing configuration.
        
        Args:
            db_path: Path to vector database
            embedding_model_path: Path to embedding model (auto-detected if None)
            max_workers: Number of parallel processing workers
            checkpoint_interval: Save checkpoint every N processed files
            batch_size: Batch size for embedding generation
        """
        self.db_path = Path(db_path)
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        
        # Initialize services
        self.db = VectorDatabase(str(db_path))
        self.embedding_service = EmbeddingService(
            model_path=embedding_model_path,
            batch_size=batch_size
        )
        self.ingestion_service = DocumentIngestionService()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def scan_directory(
        self,
        path: Path,
        pattern: str = "**/*",
        supported_extensions: Optional[Set[str]] = None
    ) -> List[Path]:
        """
        Scan directory for documents matching pattern and supported formats.
        
        Args:
            path: Directory to scan
            pattern: Glob pattern for file matching
            supported_extensions: Set of supported extensions (auto-detected if None)
            
        Returns:
            List of valid document file paths
        """
        if supported_extensions is None:
            supported_extensions = set(self.ingestion_service.get_supported_extensions())
        
        # Convert extensions to lowercase for comparison
        supported_extensions = {ext.lower() for ext in supported_extensions}
        
        files = []
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return sorted(files)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash of file content for duplicate detection."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to hash {file_path}: {e}")
            return ""
    
    def check_duplicates(self, file_paths: List[Path]) -> Tuple[List[Path], Dict[str, List[Path]]]:
        """
        Check for duplicate files based on content hash.
        
        Args:
            file_paths: List of file paths to check
            
        Returns:
            Tuple of (unique_files, duplicates_map)
        """
        hash_to_files = {}
        
        # Group files by hash
        for file_path in tqdm(file_paths, desc="Checking duplicates"):
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append(file_path)
        
        # Separate unique files from duplicates
        unique_files = []
        duplicates_map = {}
        
        for file_hash, files in hash_to_files.items():
            if len(files) == 1:
                unique_files.extend(files)
            else:
                # Keep first file, mark others as duplicates
                unique_files.append(files[0])
                duplicates_map[file_hash] = files[1:]
        
        return unique_files, duplicates_map
    
    def load_checkpoint(self, checkpoint_path: Path) -> Optional[CheckpointData]:
        """Load checkpoint data from file."""
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return CheckpointData.from_dict(data)
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def save_checkpoint(self, checkpoint_data: CheckpointData, checkpoint_path: Path):
        """Save checkpoint data to file."""
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2)
            self.logger.debug(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
    
    def process_single_document(self, file_path: Path, collection_id: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """
        Process a single document: load, chunk, embed, and store.
        
        Args:
            file_path: Path to document file
            collection_id: Collection to store document in
            
        Returns:
            Tuple of (success, stats_dict)
        """
        start_time = time.time()
        stats = {
            'file_path': str(file_path),
            'success': False,
            'chunks_created': 0,
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            # Generate unique document ID
            import uuid
            document_id = str(uuid.uuid4())
            
            # Ingest document - returns list of chunks directly
            chunks = self.ingestion_service.ingest_document(file_path)
            if not chunks:
                stats['error'] = "No chunks generated"
                return False, stats
            
            # Set document ID on all chunks
            for i, chunk in enumerate(chunks):
                chunk.doc_id = document_id
                chunk.chunk_index = i
                chunk.chunk_id = f"{document_id}_chunk_{i}"
            
            stats['chunks_created'] = len(chunks)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            
            # Create document metadata
            document_metadata = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'content_hash': self.get_file_hash(file_path),
                'size': file_path.stat().st_size
            }
            
            # Insert document record with collection_id
            self.db.insert_document(document_id, str(file_path), document_metadata, collection_id)
            
            # Insert chunks with embeddings and collection_id
            for chunk, embedding in zip(chunks, embeddings):
                self.db.insert_chunk(chunk, embedding, collection_id)
            
            stats['success'] = True
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            stats['error'] = str(e)
        
        stats['processing_time'] = time.time() - start_time
        return stats['success'], stats
    
    async def ingest_directory(
        self,
        path: str | Path,
        pattern: str = "**/*",
        collection_id: str = "default",
        dry_run: bool = False,
        resume: bool = True,
        deduplicate: bool = True
    ) -> ProcessingStats:
        """
        Bulk ingest documents from directory with parallel processing.
        
        Args:
            path: Directory path to process  
            pattern: Glob pattern for file matching
            collection_id: Target collection ID
            dry_run: Preview mode without actual processing
            resume: Resume from checkpoint if available
            deduplicate: Skip duplicate files based on content hash
            
        Returns:
            ProcessingStats object with detailed results
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        # Initialize processing stats
        stats = ProcessingStats(start_time=datetime.now())
        
        # Setup checkpoint
        checkpoint_path = self.checkpoint_dir / f"ingest_{collection_id}_{int(time.time())}.json"
        processed_files: Set[str] = set()
        
        # Try to resume from checkpoint
        if resume:
            latest_checkpoint = self._find_latest_checkpoint(collection_id)
            if latest_checkpoint:
                checkpoint_data = self.load_checkpoint(latest_checkpoint)
                if checkpoint_data:
                    processed_files = checkpoint_data.processed_files
                    stats = checkpoint_data.stats
                    self.logger.info(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        
        try:
            # Scan for files
            self.logger.info(f"Scanning directory: {path}")
            files = self.scan_directory(path, pattern)
            stats.files_scanned = len(files)
            
            if not files:
                self.logger.warning("No supported files found")
                return stats
            
            self.logger.info(f"Found {len(files)} supported files")
            
            # Remove already processed files
            remaining_files = [f for f in files if str(f) not in processed_files]
            self.logger.info(f"Files to process: {len(remaining_files)} (skipping {len(files) - len(remaining_files)} already processed)")
            
            # Deduplicate if requested
            if deduplicate and remaining_files:
                self.logger.info("Checking for duplicates...")
                unique_files, duplicates = self.check_duplicates(remaining_files)
                
                if duplicates:
                    total_duplicates = sum(len(dups) for dups in duplicates.values())
                    self.logger.info(f"Found {total_duplicates} duplicate files, processing {len(unique_files)} unique files")
                
                remaining_files = unique_files
            
            # Dry run mode
            if dry_run:
                self.logger.info("DRY RUN MODE - No files will be processed")
                stats.files_processed = len(remaining_files)
                return stats
            
            # Process files in parallel
            self.logger.info(f"Processing {len(remaining_files)} files with {self.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_document, file_path, collection_id): file_path
                    for file_path in remaining_files
                }
                
                # Process results with progress bar
                with tqdm(total=len(remaining_files), desc="Processing documents") as pbar:
                    checkpoint_counter = 0
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        
                        try:
                            success, file_stats = future.result()
                            
                            if success:
                                stats.files_processed += 1
                                stats.chunks_created += file_stats['chunks_created']
                                stats.chunks_embedded += file_stats['chunks_created']  # All chunks get embedded
                            else:
                                stats.files_failed += 1
                                self.logger.warning(f"Failed to process {file_path}: {file_stats.get('error', 'Unknown error')}")
                            
                            processed_files.add(str(file_path))
                            
                        except Exception as e:
                            stats.files_failed += 1
                            self.logger.error(f"Exception processing {file_path}: {e}")
                        
                        pbar.update(1)
                        checkpoint_counter += 1
                        
                        # Save checkpoint periodically
                        if checkpoint_counter >= self.checkpoint_interval:
                            checkpoint_data = CheckpointData(
                                processed_files=processed_files,
                                stats=stats,
                                collection_id=collection_id,
                                timestamp=datetime.now()
                            )
                            self.save_checkpoint(checkpoint_data, checkpoint_path)
                            checkpoint_counter = 0
            
            # Final statistics
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
            
            self.logger.info(f"Processing complete: {stats.files_processed} files, {stats.chunks_created} chunks, {stats.processing_time:.2f}s")
            
            # Save final checkpoint
            checkpoint_data = CheckpointData(
                processed_files=processed_files,
                stats=stats,
                collection_id=collection_id,
                timestamp=datetime.now()
            )
            self.save_checkpoint(checkpoint_data, checkpoint_path)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Directory ingestion failed: {e}")
            stats.end_time = datetime.now()
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()
            raise
    
    def _find_latest_checkpoint(self, collection_id: str) -> Optional[Path]:
        """Find the most recent checkpoint file for a collection."""
        pattern = f"ingest_{collection_id}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by modification time and return newest
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def get_processing_stats(self, collection_id: str = "default") -> Dict[str, Any]:
        """Get processing statistics for a collection."""
        db_stats = self.db.get_database_stats()
        
        # Add collection-specific stats if available
        stats = {
            'collection_id': collection_id,
            'database_stats': db_stats,
            'supported_formats': list(self.ingestion_service.get_supported_extensions()),
            'embedding_model': self.embedding_service.get_model_info(),
            'processing_config': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'checkpoint_interval': self.checkpoint_interval
            }
        }
        
        return stats
    
    def clean_checkpoints(self, collection_id: Optional[str] = None, keep_latest: int = 3):
        """Clean up old checkpoint files, keeping only the most recent ones."""
        if collection_id:
            pattern = f"ingest_{collection_id}_*.json"
        else:
            pattern = "ingest_*.json"
        
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= keep_latest:
            return
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_latest:]:
            try:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


# Utility functions for corpus management

def create_corpus_manager(
    db_path: str = "data/rag_vectors.db",
    embedding_model_path: Optional[str] = None,
    **kwargs
) -> CorpusManager:
    """
    Factory function to create a CorpusManager instance.
    
    Args:
        db_path: Path to vector database
        embedding_model_path: Path to embedding model
        **kwargs: Additional arguments for CorpusManager
        
    Returns:
        Configured CorpusManager instance
    """
    return CorpusManager(
        db_path=db_path,
        embedding_model_path=embedding_model_path,
        **kwargs
    )


async def bulk_ingest_directory(
    directory: str | Path,
    db_path: str = "data/rag_vectors.db",
    pattern: str = "**/*",
    collection_id: str = "default",
    max_workers: int = 4,
    dry_run: bool = False
) -> ProcessingStats:
    """
    Convenience function for bulk directory ingestion.
    
    Args:
        directory: Directory to process
        db_path: Vector database path
        pattern: File matching pattern
        collection_id: Target collection
        max_workers: Number of parallel workers
        dry_run: Preview mode
        
    Returns:
        Processing statistics
    """
    manager = create_corpus_manager(db_path=db_path, max_workers=max_workers)
    
    return await manager.ingest_directory(
        path=directory,
        pattern=pattern,
        collection_id=collection_id,
        dry_run=dry_run
    )
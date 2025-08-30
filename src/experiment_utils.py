"""
Experiment Utilities for RAG System
Collection management and experiment utilities for chunking optimization experiments.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from dataclasses import dataclass


@dataclass
class CollectionInfo:
    """Information about a collection."""
    collection_id: str
    document_count: int
    chunk_count: int
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    created_at: Optional[str] = None
    size_mb: Optional[float] = None


class ExperimentCollectionManager:
    """Manage collections for chunking experiments."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
    
    def create_experiment_collection(self, base_collection: str, chunk_size: int, 
                                   chunk_overlap: int, reembed: bool = True) -> str:
        """
        Create collection with specific chunking parameters.
        
        Args:
            base_collection: Source collection to copy from
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens  
            reembed: Whether to re-embed the chunks
            
        Returns:
            New collection ID
        """
        collection_id = f"exp_cs{chunk_size}_co{chunk_overlap}"
        
        try:
            # Check if collection already exists
            if self._collection_exists(collection_id):
                self.logger.info(f"Collection {collection_id} already exists, skipping creation")
                return collection_id
            
            # Use ReindexTool to create the collection
            from .reindex import ReindexTool
            reindex_tool = ReindexTool(str(self.db_path))
            
            self.logger.info(f"Creating experimental collection: {collection_id}")
            stats = reindex_tool.rechunk_documents(
                collection_id=collection_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reembed=reembed,
                backup=False
            )
            
            if stats.success:
                self.logger.info(f"Successfully created {collection_id}: "
                               f"{stats.documents_processed} docs, {stats.chunks_processed} chunks")
                return collection_id
            else:
                raise RuntimeError(f"Failed to create collection: {stats.details.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_id}: {e}")
            raise
    
    def cleanup_experiment_collections(self, pattern: str = "exp_*", 
                                     confirm: bool = True) -> List[str]:
        """
        Clean up experimental collections matching pattern.
        
        Args:
            pattern: Pattern to match collection IDs (default: "exp_*")
            confirm: Whether to require confirmation before deletion
            
        Returns:
            List of deleted collection IDs
        """
        collections = self.list_experiment_collections(pattern)
        
        if not collections:
            self.logger.info("No experimental collections found to clean up")
            return []
        
        if confirm:
            collection_list = ", ".join([c.collection_id for c in collections])
            response = input(f"Delete {len(collections)} collections ({collection_list})? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                self.logger.info("Collection cleanup cancelled")
                return []
        
        deleted = []
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                for collection in collections:
                    try:
                        # Delete chunks
                        cursor.execute("DELETE FROM chunks WHERE collection_id = ?", 
                                     (collection.collection_id,))
                        # Delete documents  
                        cursor.execute("DELETE FROM documents WHERE collection_id = ?", 
                                     (collection.collection_id,))
                        
                        deleted.append(collection.collection_id)
                        self.logger.info(f"Deleted collection: {collection.collection_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to delete collection {collection.collection_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database error during cleanup: {e}")
            raise
        
        self.logger.info(f"Successfully deleted {len(deleted)} experimental collections")
        return deleted
    
    def list_experiment_collections(self, pattern: str = "exp_*") -> List[CollectionInfo]:
        """
        List experimental collections matching pattern.
        
        Args:
            pattern: Pattern to match collection IDs
            
        Returns:
            List of CollectionInfo objects
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get collections matching pattern
                if pattern == "exp_*":
                    cursor.execute("""
                        SELECT collection_id, COUNT(DISTINCT doc_id) as doc_count, COUNT(*) as chunk_count
                        FROM chunks 
                        WHERE collection_id LIKE 'exp_%'
                        GROUP BY collection_id
                        ORDER BY collection_id
                    """)
                else:
                    # Simple pattern matching - could be enhanced
                    cursor.execute("""
                        SELECT collection_id, COUNT(DISTINCT doc_id) as doc_count, COUNT(*) as chunk_count
                        FROM chunks 
                        WHERE collection_id LIKE ?
                        GROUP BY collection_id
                        ORDER BY collection_id
                    """, (pattern.replace('*', '%'),))
                
                collections = []
                for row in cursor.fetchall():
                    collection_id, doc_count, chunk_count = row
                    
                    # Extract chunking parameters from collection ID
                    chunk_size, chunk_overlap = self._parse_collection_id(collection_id)
                    
                    collections.append(CollectionInfo(
                        collection_id=collection_id,
                        document_count=doc_count,
                        chunk_count=chunk_count,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    ))
                
                return collections
                
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    def validate_collection_parameters(self, collection_id: str) -> Dict[str, Any]:
        """
        Validate that collection has expected chunking parameters.
        
        Args:
            collection_id: Collection to validate
            
        Returns:
            Validation results and collection statistics
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get collection statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT doc_id) as total_docs,
                        AVG(LENGTH(content)) as avg_content_length,
                        MIN(LENGTH(content)) as min_content_length,
                        MAX(LENGTH(content)) as max_content_length
                    FROM chunks 
                    WHERE collection_id = ?
                """, (collection_id,))
                
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return {
                        'valid': False,
                        'error': f'Collection {collection_id} not found or empty'
                    }
                
                total_chunks, total_docs, avg_length, min_length, max_length = row
                
                # Parse expected parameters from collection ID
                expected_chunk_size, expected_overlap = self._parse_collection_id(collection_id)
                
                # Basic validation - actual content length should be reasonably close to expected
                # (accounting for tokenization differences)
                length_ratio = avg_length / (expected_chunk_size * 4) if expected_chunk_size else 1.0  # ~4 chars per token
                
                validation = {
                    'valid': True,
                    'collection_id': collection_id,
                    'total_chunks': total_chunks,
                    'total_documents': total_docs,
                    'expected_chunk_size': expected_chunk_size,
                    'expected_overlap': expected_overlap,
                    'statistics': {
                        'avg_content_length': round(avg_length, 2) if avg_length else 0,
                        'min_content_length': min_length,
                        'max_content_length': max_length,
                        'length_ratio_to_expected': round(length_ratio, 2)
                    },
                    'warnings': []
                }
                
                # Add warnings for unusual values
                if length_ratio < 0.5 or length_ratio > 2.0:
                    validation['warnings'].append(
                        f"Content length ratio {length_ratio:.2f} suggests chunking may not match expected size"
                    )
                
                if total_chunks / total_docs > 50:  # Very high chunk-to-doc ratio
                    validation['warnings'].append(
                        f"High chunk/document ratio ({total_chunks/total_docs:.1f}) - check overlap settings"
                    )
                
                return validation
                
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}'
            }
    
    def get_collection_size(self, collection_id: str) -> float:
        """
        Get collection storage size in MB.
        
        Args:
            collection_id: Collection to measure
            
        Returns:
            Size in megabytes
        """
        try:
            # This is an approximation based on content length
            # In practice, you might want to measure actual storage
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT SUM(LENGTH(content) + LENGTH(COALESCE(metadata_json, ''))) as total_size
                    FROM chunks 
                    WHERE collection_id = ?
                """, (collection_id,))
                
                row = cursor.fetchone()
                total_bytes = row[0] if row and row[0] else 0
                return total_bytes / (1024 * 1024)  # Convert to MB
                
        except Exception as e:
            self.logger.error(f"Error calculating collection size: {e}")
            return 0.0
    
    def _collection_exists(self, collection_id: str) -> bool:
        """Check if collection exists."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection_id,))
                return cursor.fetchone()[0] > 0
        except Exception:
            return False
    
    def _parse_collection_id(self, collection_id: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse chunk_size and chunk_overlap from collection ID.
        
        Args:
            collection_id: Collection ID like "exp_cs256_co64"
            
        Returns:
            Tuple of (chunk_size, chunk_overlap) or (None, None) if not parseable
        """
        try:
            if not collection_id.startswith('exp_'):
                return None, None
            
            parts = collection_id.split('_')
            chunk_size = None
            chunk_overlap = None
            
            for part in parts:
                if part.startswith('cs'):
                    chunk_size = int(part[2:])
                elif part.startswith('co'):
                    chunk_overlap = int(part[2:])
            
            return chunk_size, chunk_overlap
            
        except (ValueError, IndexError):
            return None, None


class ExperimentConfigGenerator:
    """Generate experiment configurations for chunking parameter sweeps."""
    
    @staticmethod
    def generate_chunking_configs(chunk_sizes: List[int], 
                                overlap_ratios: List[float]) -> List[Dict[str, Any]]:
        """
        Generate configurations for chunking parameter sweep.
        
        Args:
            chunk_sizes: List of chunk sizes to test
            overlap_ratios: List of overlap ratios (0.1 = 10% of chunk_size)
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                chunk_overlap = int(chunk_size * overlap_ratio)
                
                config = {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'overlap_ratio': overlap_ratio,
                    'collection_id': f"exp_cs{chunk_size}_co{chunk_overlap}",
                    'profile': f"chunking_cs{chunk_size}_co{chunk_overlap}"
                }
                
                configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_experiment_plan(chunk_sizes: List[int] = [128, 256, 512, 768, 1024],
                               overlap_ratios: List[float] = [0.10, 0.15, 0.20, 0.25],
                               repetitions: int = 10,
                               queries_file: str = "test_data/enhanced_evaluation_queries.json") -> Dict[str, Any]:
        """
        Generate complete experiment plan.
        
        Args:
            chunk_sizes: Chunk sizes to test
            overlap_ratios: Overlap ratios to test
            repetitions: Number of repetitions per configuration
            queries_file: Path to query dataset
            
        Returns:
            Complete experiment plan
        """
        configs = ExperimentConfigGenerator.generate_chunking_configs(chunk_sizes, overlap_ratios)
        
        plan = {
            'experiment_name': 'chunking_optimization_v2',
            'description': 'Document chunking parameter optimization with proper isolation',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'chunk_sizes': chunk_sizes,
                'overlap_ratios': overlap_ratios,
                'repetitions': repetitions,
                'total_configurations': len(configs),
                'total_runs': len(configs) * repetitions
            },
            'data_sources': {
                'queries_file': queries_file,
                'base_collection': 'realistic_full_production'
            },
            'configurations': configs,
            'expected_runtime_hours': len(configs) * repetitions * 0.1,  # Rough estimate
            'success_criteria': {
                'statistical_significance': 'p < 0.05',
                'effect_size_threshold': 0.3,
                'minimum_improvement': '10%'
            }
        }
        
        return plan
    
    @staticmethod
    def save_experiment_plan(plan: Dict[str, Any], output_file: str) -> None:
        """Save experiment plan to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"✅ Saved experiment plan to {output_path}")
        print(f"   {plan['parameters']['total_configurations']} configurations")
        print(f"   {plan['parameters']['total_runs']} total runs")
        print(f"   Estimated runtime: {plan['expected_runtime_hours']:.1f} hours")
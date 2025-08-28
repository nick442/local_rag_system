"""
Corpus Analytics - Statistical analysis and visualization for document corpus

This module provides comprehensive corpus analysis capabilities including:
- Document distribution analysis by type, size, and collection
- Token count statistics and content analysis
- Coverage analysis and retrieval pattern insights
- Embedding space visualization and clustering
- Growth trends and usage analytics
"""

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from .vector_database import VectorDatabase


@dataclass
class CorpusStats:
    """Comprehensive corpus statistics"""
    collection_id: str
    total_documents: int
    total_chunks: int
    total_tokens: int
    avg_document_size: float
    avg_chunks_per_doc: float
    size_mb: float
    file_types: Dict[str, int]
    ingestion_timeline: Dict[str, int]
    most_similar_pairs: List[Tuple[str, str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'collection_id': self.collection_id,
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'total_tokens': self.total_tokens,
            'avg_document_size': self.avg_document_size,
            'avg_chunks_per_doc': self.avg_chunks_per_doc,
            'size_mb': self.size_mb,
            'file_types': self.file_types,
            'ingestion_timeline': self.ingestion_timeline,
            'most_similar_pairs': [
                {'doc1': pair[0], 'doc2': pair[1], 'similarity': pair[2]}
                for pair in self.most_similar_pairs
            ]
        }


@dataclass
class DocumentInsights:
    """Insights for individual document"""
    doc_id: str
    source_path: str
    chunk_count: int
    token_count: int
    file_size: int
    ingested_at: datetime
    most_similar_docs: List[Tuple[str, float]]
    avg_chunk_similarity: float
    retrieval_frequency: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'source_path': self.source_path,
            'chunk_count': self.chunk_count,
            'token_count': self.token_count,
            'file_size': self.file_size,
            'ingested_at': self.ingested_at.isoformat(),
            'most_similar_docs': [
                {'doc_id': doc_id, 'similarity': sim}
                for doc_id, sim in self.most_similar_docs
            ],
            'avg_chunk_similarity': self.avg_chunk_similarity,
            'retrieval_frequency': self.retrieval_frequency
        }


class CorpusAnalyzer:
    """
    Advanced corpus analytics and visualization system.
    
    Features:
    - Comprehensive statistical analysis
    - Document type and size distribution
    - Temporal ingestion patterns
    - Similarity analysis and clustering
    - Content quality assessment
    - Usage pattern tracking
    """
    
    def __init__(self, db_path: str = "data/rag_vectors.db"):
        """
        Initialize corpus analyzer.
        
        Args:
            db_path: Path to vector database
        """
        self.db_path = Path(db_path)
        self.db = VectorDatabase(str(db_path))
        self.logger = logging.getLogger(__name__)
    
    def analyze_collection(self, collection_id: str = "default") -> CorpusStats:
        """
        Comprehensive analysis of a document collection.
        
        Args:
            collection_id: Collection to analyze
            
        Returns:
            CorpusStats with detailed analysis
        """
        self.logger.info(f"Analyzing collection: {collection_id}")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection_id,))
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(token_count) FROM chunks WHERE collection_id = ?", (collection_id,))
            total_tokens = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE collection_id = ?", (collection_id,))
            total_size = cursor.fetchone()[0] or 0
            
            # Document size statistics
            cursor.execute("SELECT AVG(file_size) FROM documents WHERE collection_id = ?", (collection_id,))
            avg_doc_size = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(total_chunks) FROM documents WHERE collection_id = ?", (collection_id,))
            avg_chunks_per_doc = cursor.fetchone()[0] or 0
            
            # File type distribution
            cursor.execute("SELECT source_path FROM documents WHERE collection_id = ?", (collection_id,))
            file_types = Counter()
            for row in cursor.fetchall():
                file_path = Path(row[0])
                extension = file_path.suffix.lower()
                file_types[extension or 'unknown'] += 1
            
            # Ingestion timeline (by day)
            cursor.execute("""
                SELECT DATE(ingested_at) as ingestion_date, COUNT(*) 
                FROM documents WHERE collection_id = ? 
                GROUP BY DATE(ingested_at)
                ORDER BY ingestion_date
            """, (collection_id,))
            
            ingestion_timeline = {}
            for row in cursor.fetchall():
                ingestion_timeline[row[0]] = row[1]
        
        # Find most similar document pairs (sample)
        most_similar_pairs = self._find_similar_documents(collection_id, limit=5)
        
        stats = CorpusStats(
            collection_id=collection_id,
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            avg_document_size=avg_doc_size,
            avg_chunks_per_doc=avg_chunks_per_doc,
            size_mb=total_size / (1024 * 1024),
            file_types=dict(file_types),
            ingestion_timeline=ingestion_timeline,
            most_similar_pairs=most_similar_pairs
        )
        
        self.logger.info(f"Collection analysis complete: {total_docs} docs, {total_chunks} chunks")
        return stats
    
    def _find_similar_documents(
        self,
        collection_id: str = "default",
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of similar documents using embeddings."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get document embeddings (average of chunk embeddings)
            cursor.execute("""
                SELECT d.doc_id, d.source_path, AVG(e.embedding_vector)
                FROM documents d
                JOIN chunks c ON d.doc_id = c.doc_id
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE d.collection_id = ?
                GROUP BY d.doc_id, d.source_path
                LIMIT 100
            """, (collection_id,))
            
            doc_embeddings = {}
            for row in cursor.fetchall():
                doc_id, source_path, embedding_blob = row
                if embedding_blob:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    doc_embeddings[doc_id] = {
                        'embedding': embedding,
                        'source_path': source_path
                    }
        
        # Calculate pairwise similarities
        similar_pairs = []
        doc_ids = list(doc_embeddings.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for j, doc_id2 in enumerate(doc_ids[i+1:], i+1):
                emb1 = doc_embeddings[doc_id1]['embedding']
                emb2 = doc_embeddings[doc_id2]['embedding']
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                if similarity >= min_similarity:
                    similar_pairs.append((doc_id1, doc_id2, float(similarity)))
        
        # Sort by similarity and return top pairs
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs[:limit]
    
    def analyze_document(self, doc_id: str, collection_id: str = "default") -> Optional[DocumentInsights]:
        """
        Detailed analysis of a specific document.
        
        Args:
            doc_id: Document ID to analyze
            collection_id: Collection containing the document
            
        Returns:
            DocumentInsights with detailed analysis
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get document metadata
            cursor.execute("""
                SELECT source_path, ingested_at, file_size, total_chunks
                FROM documents WHERE doc_id = ? AND collection_id = ?
            """, (doc_id, collection_id))
            
            doc_row = cursor.fetchone()
            if not doc_row:
                return None
            
            source_path, ingested_at, file_size, total_chunks = doc_row
            
            # Get token count
            cursor.execute("""
                SELECT SUM(token_count) FROM chunks 
                WHERE doc_id = ? AND collection_id = ?
            """, (doc_id, collection_id))
            token_count = cursor.fetchone()[0] or 0
            
            # Get average chunk embedding for similarity analysis
            cursor.execute("""
                SELECT embedding_vector FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                WHERE c.doc_id = ? AND c.collection_id = ?
            """, (doc_id, collection_id))
            
            embeddings = []
            for row in cursor.fetchall():
                embedding = np.frombuffer(row[0], dtype=np.float32)
                embeddings.append(embedding)
            
            # Calculate average embedding and chunk similarity
            avg_chunk_similarity = 0.0
            if len(embeddings) > 1:
                # Calculate pairwise similarities between chunks
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)
                avg_chunk_similarity = np.mean(similarities) if similarities else 0.0
        
        # Find most similar documents
        doc_embedding = np.mean(embeddings, axis=0) if embeddings else None
        most_similar_docs = []
        
        if doc_embedding is not None:
            most_similar_docs = self._find_similar_to_document(
                doc_id, doc_embedding, collection_id, limit=5
            )
        
        # TODO: Add retrieval frequency tracking (requires query log)
        retrieval_frequency = 0
        
        insights = DocumentInsights(
            doc_id=doc_id,
            source_path=source_path,
            chunk_count=total_chunks,
            token_count=token_count,
            file_size=file_size,
            ingested_at=datetime.fromisoformat(ingested_at),
            most_similar_docs=most_similar_docs,
            avg_chunk_similarity=avg_chunk_similarity,
            retrieval_frequency=retrieval_frequency
        )
        
        return insights
    
    def _find_similar_to_document(
        self,
        target_doc_id: str,
        target_embedding: np.ndarray,
        collection_id: str,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find documents similar to a target document."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get embeddings for all other documents
            cursor.execute("""
                SELECT d.doc_id, AVG(e.embedding_vector)
                FROM documents d
                JOIN chunks c ON d.doc_id = c.doc_id
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE d.collection_id = ? AND d.doc_id != ?
                GROUP BY d.doc_id
            """, (collection_id, target_doc_id))
            
            similarities = []
            for row in cursor.fetchall():
                doc_id, embedding_blob = row
                if embedding_blob:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    similarity = np.dot(target_embedding, embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append((doc_id, float(similarity)))
        
        # Sort by similarity and return top documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def get_collection_growth(
        self,
        collection_id: str = "default",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze collection growth over time.
        
        Args:
            collection_id: Collection to analyze
            days: Number of days to analyze
            
        Returns:
            Growth analysis data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Daily document counts
            cursor.execute("""
                SELECT DATE(ingested_at) as day, COUNT(*) as doc_count, SUM(file_size) as total_size
                FROM documents 
                WHERE collection_id = ? AND ingested_at >= ?
                GROUP BY DATE(ingested_at)
                ORDER BY day
            """, (collection_id, start_date.isoformat()))
            
            daily_stats = []
            cumulative_docs = 0
            cumulative_size = 0
            
            for row in cursor.fetchall():
                day, doc_count, total_size = row
                cumulative_docs += doc_count
                cumulative_size += total_size or 0
                
                daily_stats.append({
                    'date': day,
                    'documents_added': doc_count,
                    'size_added_mb': (total_size or 0) / (1024 * 1024),
                    'cumulative_documents': cumulative_docs,
                    'cumulative_size_mb': cumulative_size / (1024 * 1024)
                })
            
            # File type trends
            cursor.execute("""
                SELECT DATE(ingested_at) as day, source_path
                FROM documents 
                WHERE collection_id = ? AND ingested_at >= ?
                ORDER BY day
            """, (collection_id, start_date.isoformat()))
            
            type_trends = defaultdict(lambda: defaultdict(int))
            for row in cursor.fetchall():
                day, source_path = row
                extension = Path(source_path).suffix.lower() or 'unknown'
                type_trends[day][extension] += 1
        
        return {
            'collection_id': collection_id,
            'analysis_period_days': days,
            'daily_stats': daily_stats,
            'file_type_trends': dict(type_trends),
            'total_growth': {
                'documents': cumulative_docs,
                'size_mb': cumulative_size / (1024 * 1024)
            }
        }
    
    def generate_quality_report(self, collection_id: str = "default") -> Dict[str, Any]:
        """
        Generate a quality assessment report for the collection.
        
        Args:
            collection_id: Collection to assess
            
        Returns:
            Quality assessment report
        """
        self.logger.info(f"Generating quality report for collection: {collection_id}")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Basic quality metrics
            cursor.execute("""
                SELECT COUNT(*) FROM documents WHERE collection_id = ? AND total_chunks = 0
            """, (collection_id,))
            empty_docs = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM chunks WHERE collection_id = ? AND LENGTH(content) < 50
            """, (collection_id,))
            short_chunks = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM chunks WHERE collection_id = ? AND LENGTH(content) > 5000
            """, (collection_id,))
            long_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection_id,))
            total_chunks = cursor.fetchone()[0]
            
            # Missing embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM chunks c
                LEFT JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE c.collection_id = ? AND e.chunk_id IS NULL
            """, (collection_id,))
            missing_embeddings = cursor.fetchone()[0]
            
            # Token distribution
            cursor.execute("""
                SELECT MIN(token_count), MAX(token_count), AVG(token_count), 
                       COUNT(CASE WHEN token_count < 10 THEN 1 END) as very_short,
                       COUNT(CASE WHEN token_count > 1000 THEN 1 END) as very_long
                FROM chunks WHERE collection_id = ?
            """, (collection_id,))
            
            token_stats = cursor.fetchone()
            min_tokens, max_tokens, avg_tokens, very_short_chunks, very_long_chunks = token_stats
            
            # File format distribution and potential issues
            cursor.execute("SELECT source_path FROM documents WHERE collection_id = ?", (collection_id,))
            file_paths = [row[0] for row in cursor.fetchall()]
            
            format_issues = []
            unusual_extensions = Counter()
            
            for path in file_paths:
                ext = Path(path).suffix.lower()
                if ext not in ['.pdf', '.txt', '.md', '.html', '.htm']:
                    unusual_extensions[ext] += 1
        
        # Calculate quality scores
        quality_scores = {
            'completeness': max(0, 1 - (empty_docs + missing_embeddings) / max(1, total_chunks)),
            'chunk_size_consistency': max(0, 1 - (short_chunks + long_chunks) / max(1, total_chunks)),
            'token_distribution': max(0, 1 - (very_short_chunks + very_long_chunks) / max(1, total_chunks)),
            'format_consistency': max(0, 1 - len(unusual_extensions) / max(1, len(file_paths)))
        }
        
        overall_quality = np.mean(list(quality_scores.values()))
        
        report = {
            'collection_id': collection_id,
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': overall_quality,
            'quality_scores': quality_scores,
            'metrics': {
                'total_documents': len(file_paths),
                'total_chunks': total_chunks,
                'empty_documents': empty_docs,
                'missing_embeddings': missing_embeddings,
                'short_chunks': short_chunks,
                'long_chunks': long_chunks,
                'token_stats': {
                    'min': min_tokens,
                    'max': max_tokens,
                    'avg': avg_tokens,
                    'very_short': very_short_chunks,
                    'very_long': very_long_chunks
                }
            },
            'issues': {
                'unusual_file_formats': dict(unusual_extensions),
                'recommendations': []
            }
        }
        
        # Generate recommendations
        if empty_docs > 0:
            report['issues']['recommendations'].append(f"Remove or re-process {empty_docs} empty documents")
        
        if missing_embeddings > 0:
            report['issues']['recommendations'].append(f"Generate embeddings for {missing_embeddings} chunks")
        
        if short_chunks > total_chunks * 0.1:
            report['issues']['recommendations'].append(f"Review chunking strategy - {short_chunks} very short chunks")
        
        if unusual_extensions:
            report['issues']['recommendations'].append(f"Review unsupported formats: {list(unusual_extensions.keys())}")
        
        quality_rating = "Excellent" if overall_quality > 0.9 else \
                        "Good" if overall_quality > 0.7 else \
                        "Fair" if overall_quality > 0.5 else "Poor"
        
        report['quality_rating'] = quality_rating
        
        self.logger.info(f"Quality report complete: {quality_rating} ({overall_quality:.2f})")
        return report
    
    def compare_collections(self, collection_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple collections across various metrics.
        
        Args:
            collection_ids: List of collection IDs to compare
            
        Returns:
            Comparison analysis
        """
        self.logger.info(f"Comparing collections: {collection_ids}")
        
        comparison = {
            'collections': collection_ids,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for collection_id in collection_ids:
            stats = self.analyze_collection(collection_id)
            comparison['metrics'][collection_id] = stats.to_dict()
        
        # Calculate comparative metrics
        total_docs = sum(comparison['metrics'][cid]['total_documents'] for cid in collection_ids)
        total_size = sum(comparison['metrics'][cid]['size_mb'] for cid in collection_ids)
        
        comparison['summary'] = {
            'total_documents_across_collections': total_docs,
            'total_size_mb_across_collections': total_size,
            'largest_collection': max(collection_ids, key=lambda cid: comparison['metrics'][cid]['total_documents']),
            'most_diverse_formats': max(collection_ids, key=lambda cid: len(comparison['metrics'][cid]['file_types']))
        }
        
        return comparison
    
    def export_analytics_report(
        self,
        collection_id: str = "default",
        output_path: Optional[str] = None,
        include_quality: bool = True,
        include_growth: bool = True
    ) -> Dict[str, Any]:
        """
        Export comprehensive analytics report to file.
        
        Args:
            collection_id: Collection to analyze
            output_path: Output file path (auto-generated if None)
            include_quality: Include quality assessment
            include_growth: Include growth analysis
            
        Returns:
            Report data and export information
        """
        self.logger.info(f"Exporting analytics report for collection: {collection_id}")
        
        # Generate comprehensive report
        report = {
            'collection_id': collection_id,
            'generated_at': datetime.now().isoformat(),
            'corpus_stats': self.analyze_collection(collection_id).to_dict()
        }
        
        if include_quality:
            report['quality_assessment'] = self.generate_quality_report(collection_id)
        
        if include_growth:
            report['growth_analysis'] = self.get_collection_growth(collection_id)
        
        # Export to file
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analytics_report_{collection_id}_{timestamp}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        export_info = {
            'report_path': str(output_file),
            'file_size_mb': output_file.stat().st_size / (1024 * 1024),
            'sections_included': list(report.keys())
        }
        
        self.logger.info(f"Analytics report exported to: {output_file}")
        return {**report, 'export_info': export_info}


# Utility functions

def create_corpus_analyzer(db_path: str = "data/rag_vectors.db") -> CorpusAnalyzer:
    """
    Factory function to create a CorpusAnalyzer instance.
    
    Args:
        db_path: Path to vector database
        
    Returns:
        Configured CorpusAnalyzer instance
    """
    return CorpusAnalyzer(db_path=db_path)


def quick_stats(collection_id: str = "default", db_path: str = "data/rag_vectors.db") -> Dict[str, Any]:
    """
    Convenience function for quick collection statistics.
    
    Args:
        collection_id: Collection to analyze
        db_path: Vector database path
        
    Returns:
        Basic collection statistics
    """
    analyzer = create_corpus_analyzer(db_path=db_path)
    stats = analyzer.analyze_collection(collection_id)
    return stats.to_dict()
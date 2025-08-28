#!/usr/bin/env python3
"""
Phase 7 Testing Suite - Comprehensive Corpus Management Tests

This test suite validates all Phase 7 corpus management functionality including:
- Bulk document ingestion with parallel processing
- Collection management and organization
- Duplicate detection and resolution
- Re-indexing and maintenance operations
- Analytics and reporting capabilities
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_manager import CorpusManager, create_corpus_manager
from src.corpus_organizer import CorpusOrganizer, create_corpus_organizer
from src.deduplication import DocumentDeduplicator, create_deduplicator
from src.reindex import ReindexTool, create_reindex_tool
from src.corpus_analytics import CorpusAnalyzer, create_corpus_analyzer
from src.vector_database import VectorDatabase
from src.embedding_service import EmbeddingService


class Phase7TestSuite:
    """Comprehensive test suite for Phase 7 corpus management features"""
    
    def __init__(self):
        self.test_db_path = None
        self.temp_dir = None
        self.results = {
            'phase': 7,
            'timestamp': time.time(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {},
            'performance_metrics': {},
            'errors': []
        }
    
    def setup_test_environment(self):
        """Setup temporary test environment"""
        print("🔧 Setting up test environment...")
        
        # Create temporary directory for test database
        self.temp_dir = Path(tempfile.mkdtemp(prefix='phase7_test_'))
        self.test_db_path = self.temp_dir / "test_rag.db"
        
        # Create test data directories
        self.test_corpus_dir = self.temp_dir / "test_corpus"
        self.test_corpus_dir.mkdir()
        
        # Create sample test files
        self._create_test_documents()
        
        print(f"✓ Test environment created: {self.temp_dir}")
    
    def _create_test_documents(self):
        """Create sample test documents"""
        # Create different document types
        test_docs = [
            ("sample1.txt", "This is the first sample document about machine learning. It contains important information about neural networks and artificial intelligence."),
            ("sample2.txt", "This is the second sample document also about machine learning. It discusses deep learning and neural networks in detail."),  # Similar to sample1
            ("sample3.md", "# Markdown Document\n\nThis is a markdown document about data science. It covers statistical analysis and machine learning algorithms."),
            ("sample4.html", "<html><body><h1>HTML Document</h1><p>This document discusses web development and programming.</p></body></html>"),
            ("duplicate1.txt", "This is the first sample document about machine learning. It contains important information about neural networks and artificial intelligence."),  # Exact duplicate of sample1
            ("different.txt", "This document is completely different and talks about cooking recipes and food preparation techniques.")
        ]
        
        for filename, content in test_docs:
            doc_path = self.test_corpus_dir / filename
            doc_path.write_text(content, encoding='utf-8')
        
        print(f"✓ Created {len(test_docs)} test documents")
    
    def teardown_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    def test_corpus_manager_bulk_ingestion(self):
        """Test bulk document ingestion with parallel processing"""
        print("\n📥 Testing bulk document ingestion...")
        
        try:
            # Initialize corpus manager
            manager = create_corpus_manager(
                db_path=str(self.test_db_path),
                max_workers=2,
                batch_size=4
            )
            
            # Test directory ingestion
            start_time = time.time()
            stats = asyncio.run(manager.ingest_directory(
                path=self.test_corpus_dir,
                pattern="*.txt",
                collection_id="test_collection",
                dry_run=False,
                deduplicate=True
            ))
            processing_time = time.time() - start_time
            
            # Verify results
            assert stats.files_processed > 0, "No files were processed"
            assert stats.chunks_created > 0, "No chunks were created"
            assert stats.chunks_embedded > 0, "No embeddings were generated"
            
            self.results['performance_metrics']['ingestion_time'] = processing_time
            self.results['performance_metrics']['files_per_second'] = stats.files_processed / processing_time
            
            self.results['test_results']['bulk_ingestion'] = {
                'passed': True,
                'files_processed': stats.files_processed,
                'chunks_created': stats.chunks_created,
                'processing_time': processing_time
            }
            
            print(f"✓ Bulk ingestion test passed")
            print(f"  - Files processed: {stats.files_processed}")
            print(f"  - Chunks created: {stats.chunks_created}")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['bulk_ingestion'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Bulk ingestion test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Bulk ingestion test failed: {e}")
    
    def test_collection_management(self):
        """Test collection creation, management, and organization"""
        print("\n📚 Testing collection management...")
        
        try:
            organizer = create_corpus_organizer(str(self.test_db_path))
            
            # Test collection creation
            collection_id = organizer.create_collection(
                name="Test Collection",
                description="Test collection for Phase 7",
                tags=["test", "phase7"]
            )
            
            assert collection_id is not None, "Collection creation failed"
            
            # Test collection listing
            collections = organizer.list_collections()
            assert len(collections) >= 1, "Collection not found in list"
            
            test_collection = None
            for col in collections:
                if col.collection_id == collection_id:
                    test_collection = col
                    break
            
            assert test_collection is not None, "Created collection not found"
            assert test_collection.name == "Test Collection", "Collection name mismatch"
            
            # Test collection switching
            organizer.switch_collection(collection_id)
            current = organizer.get_current_collection()
            assert current == collection_id, "Collection switching failed"
            
            # Test collection stats update
            organizer.update_collection_stats(collection_id)
            
            self.results['test_results']['collection_management'] = {
                'passed': True,
                'collections_created': 1,
                'collections_listed': len(collections)
            }
            
            print(f"✓ Collection management test passed")
            print(f"  - Collection created: {collection_id}")
            print(f"  - Total collections: {len(collections)}")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['collection_management'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Collection management test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Collection management test failed: {e}")
    
    def test_duplicate_detection(self):
        """Test duplicate detection and resolution"""
        print("\n🔍 Testing duplicate detection...")
        
        try:
            # First, ingest all documents including duplicates
            manager = create_corpus_manager(db_path=str(self.test_db_path))
            
            # Ingest all files (including duplicates)
            asyncio.run(manager.ingest_directory(
                path=self.test_corpus_dir,
                collection_id="default",
                deduplicate=False  # Don't deduplicate during ingestion
            ))
            
            # Test duplicate detection
            deduplicator = create_deduplicator(str(self.test_db_path))
            
            start_time = time.time()
            report = deduplicator.analyze_duplicates("default")
            detection_time = time.time() - start_time
            
            # Verify duplicate detection results
            assert report.total_documents > 0, "No documents found for duplicate analysis"
            assert len(report.duplicate_groups) > 0, "No duplicates detected (expected at least one)"
            
            # We know we have exact duplicates (sample1.txt and duplicate1.txt)
            exact_duplicates_found = any(group.detection_method == 'exact' for group in report.duplicate_groups)
            assert exact_duplicates_found, "Exact duplicates not detected"
            
            self.results['performance_metrics']['duplicate_detection_time'] = detection_time
            
            self.results['test_results']['duplicate_detection'] = {
                'passed': True,
                'total_documents': report.total_documents,
                'duplicate_groups': len(report.duplicate_groups),
                'exact_duplicates': report.exact_duplicates,
                'detection_time': detection_time
            }
            
            print(f"✓ Duplicate detection test passed")
            print(f"  - Total documents: {report.total_documents}")
            print(f"  - Duplicate groups: {len(report.duplicate_groups)}")
            print(f"  - Detection time: {detection_time:.2f}s")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['duplicate_detection'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Duplicate detection test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Duplicate detection test failed: {e}")
    
    def test_reindexing_operations(self):
        """Test various re-indexing and maintenance operations"""
        print("\n🔧 Testing re-indexing operations...")
        
        try:
            tool = create_reindex_tool(str(self.test_db_path))
            
            # Test database validation
            validation_report = tool.validate_integrity()
            assert validation_report['overall_status'] in ['PASS', 'FAIL'], "Invalid validation status"
            
            # Test index rebuilding
            start_time = time.time()
            rebuild_stats = tool.rebuild_indices(backup=False)
            rebuild_time = time.time() - start_time
            
            assert rebuild_stats.success, f"Index rebuild failed: {rebuild_stats.details.get('error', 'Unknown error')}"
            
            # Test database vacuum
            start_time = time.time()
            vacuum_stats = tool.vacuum_database(backup=False)
            vacuum_time = time.time() - start_time
            
            assert vacuum_stats.success, f"Database vacuum failed: {vacuum_stats.details.get('error', 'Unknown error')}"
            
            self.results['performance_metrics']['rebuild_time'] = rebuild_time
            self.results['performance_metrics']['vacuum_time'] = vacuum_time
            
            self.results['test_results']['reindexing'] = {
                'passed': True,
                'validation_status': validation_report['overall_status'],
                'rebuild_success': rebuild_stats.success,
                'vacuum_success': vacuum_stats.success,
                'rebuild_time': rebuild_time,
                'vacuum_time': vacuum_time
            }
            
            print(f"✓ Re-indexing operations test passed")
            print(f"  - Validation: {validation_report['overall_status']}")
            print(f"  - Rebuild time: {rebuild_time:.2f}s")
            print(f"  - Vacuum time: {vacuum_time:.2f}s")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['reindexing'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Re-indexing test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Re-indexing operations test failed: {e}")
    
    def test_analytics_and_reporting(self):
        """Test corpus analytics and reporting capabilities"""
        print("\n📊 Testing analytics and reporting...")
        
        try:
            analyzer = create_corpus_analyzer(str(self.test_db_path))
            
            # Test collection analysis
            start_time = time.time()
            stats = analyzer.analyze_collection("default")
            analysis_time = time.time() - start_time
            
            assert stats.total_documents > 0, "No documents found in analysis"
            assert stats.total_chunks > 0, "No chunks found in analysis"
            assert stats.file_types, "No file type information"
            
            # Test quality report
            start_time = time.time()
            quality_report = analyzer.generate_quality_report("default")
            quality_time = time.time() - start_time
            
            assert 0 <= quality_report['overall_quality_score'] <= 1, "Invalid quality score"
            assert quality_report['quality_rating'] in ['Excellent', 'Good', 'Fair', 'Poor'], "Invalid quality rating"
            
            # Test growth analysis
            growth_analysis = analyzer.get_collection_growth("default", days=7)
            assert 'daily_stats' in growth_analysis, "Growth analysis missing daily stats"
            
            self.results['performance_metrics']['analysis_time'] = analysis_time
            self.results['performance_metrics']['quality_analysis_time'] = quality_time
            
            self.results['test_results']['analytics'] = {
                'passed': True,
                'documents_analyzed': stats.total_documents,
                'chunks_analyzed': stats.total_chunks,
                'quality_score': quality_report['overall_quality_score'],
                'quality_rating': quality_report['quality_rating'],
                'analysis_time': analysis_time
            }
            
            print(f"✓ Analytics and reporting test passed")
            print(f"  - Documents analyzed: {stats.total_documents}")
            print(f"  - Quality score: {quality_report['overall_quality_score']:.2f}")
            print(f"  - Quality rating: {quality_report['quality_rating']}")
            print(f"  - Analysis time: {analysis_time:.2f}s")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['analytics'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Analytics test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Analytics and reporting test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance characteristics and benchmarks"""
        print("\n⚡ Testing performance benchmarks...")
        
        try:
            # Create larger test dataset for performance testing
            perf_corpus_dir = self.temp_dir / "perf_corpus"
            perf_corpus_dir.mkdir()
            
            # Create multiple test files for performance testing
            for i in range(20):
                content = f"Performance test document {i}. " * 50  # ~1KB per document
                (perf_corpus_dir / f"perf_doc_{i}.txt").write_text(content)
            
            # Test bulk ingestion performance
            manager = create_corpus_manager(
                db_path=str(self.test_db_path),
                max_workers=4,
                batch_size=8
            )
            
            start_time = time.time()
            perf_stats = asyncio.run(manager.ingest_directory(
                path=perf_corpus_dir,
                collection_id="performance_test",
                deduplicate=False
            ))
            ingestion_time = time.time() - start_time
            
            # Calculate performance metrics
            docs_per_second = perf_stats.files_processed / ingestion_time
            chunks_per_second = perf_stats.chunks_created / ingestion_time
            
            # Performance assertions (reasonable thresholds)
            assert docs_per_second > 1, f"Document processing too slow: {docs_per_second:.2f} docs/sec"
            assert ingestion_time < 60, f"Ingestion took too long: {ingestion_time:.2f}s"
            
            self.results['performance_metrics']['perf_docs_per_second'] = docs_per_second
            self.results['performance_metrics']['perf_chunks_per_second'] = chunks_per_second
            self.results['performance_metrics']['perf_ingestion_time'] = ingestion_time
            
            self.results['test_results']['performance'] = {
                'passed': True,
                'documents_processed': perf_stats.files_processed,
                'processing_time': ingestion_time,
                'docs_per_second': docs_per_second,
                'chunks_per_second': chunks_per_second
            }
            
            print(f"✓ Performance benchmarks test passed")
            print(f"  - Documents processed: {perf_stats.files_processed}")
            print(f"  - Processing rate: {docs_per_second:.2f} docs/sec")
            print(f"  - Chunk rate: {chunks_per_second:.2f} chunks/sec")
            print(f"  - Total time: {ingestion_time:.2f}s")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['performance'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Performance test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Performance benchmarks test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n🛡️ Testing error handling...")
        
        try:
            # Test invalid database path
            try:
                invalid_manager = CorpusManager(db_path="/invalid/path/database.db")
                # Should handle gracefully
            except Exception:
                pass  # Expected to fail, but shouldn't crash
            
            # Test empty directory ingestion
            empty_dir = self.temp_dir / "empty"
            empty_dir.mkdir()
            
            manager = create_corpus_manager(str(self.test_db_path))
            empty_stats = asyncio.run(manager.ingest_directory(path=empty_dir))
            
            assert empty_stats.files_processed == 0, "Should process 0 files from empty directory"
            
            # Test duplicate detection on empty collection
            deduplicator = create_deduplicator(str(self.test_db_path))
            empty_report = deduplicator.analyze_duplicates("nonexistent_collection")
            
            assert empty_report.total_documents == 0, "Should find 0 documents in nonexistent collection"
            
            self.results['test_results']['error_handling'] = {
                'passed': True,
                'empty_directory_handled': True,
                'invalid_collection_handled': True
            }
            
            print(f"✓ Error handling test passed")
            print(f"  - Empty directory handling: ✓")
            print(f"  - Invalid collection handling: ✓")
            
            self.results['tests_passed'] += 1
            
        except Exception as e:
            self.results['test_results']['error_handling'] = {'passed': False, 'error': str(e)}
            self.results['errors'].append(f"Error handling test failed: {e}")
            self.results['tests_failed'] += 1
            print(f"❌ Error handling test failed: {e}")
    
    def run_all_tests(self):
        """Run the complete Phase 7 test suite"""
        print("🚀 Starting Phase 7 Comprehensive Test Suite")
        print("=" * 60)
        
        try:
            self.setup_test_environment()
            
            # Run all tests
            self.test_corpus_manager_bulk_ingestion()
            self.test_collection_management()
            self.test_duplicate_detection()
            self.test_reindexing_operations()
            self.test_analytics_and_reporting()
            self.test_performance_benchmarks()
            self.test_error_handling()
            
        except Exception as e:
            print(f"❌ Test suite setup failed: {e}")
            self.results['errors'].append(f"Test suite setup failed: {e}")
            return False
        
        finally:
            self.teardown_test_environment()
        
        # Calculate final results
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📋 Phase 7 Test Results Summary")
        print("=" * 60)
        print(f"✅ Tests passed: {self.results['tests_passed']}")
        print(f"❌ Tests failed: {self.results['tests_failed']}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n🚨 Errors encountered:")
            for error in self.results['errors']:
                print(f"  - {error}")
        
        print(f"\n⚡ Performance Metrics:")
        for metric, value in self.results['performance_metrics'].items():
            if isinstance(value, float):
                print(f"  - {metric}: {value:.2f}")
            else:
                print(f"  - {metric}: {value}")
        
        # Overall assessment
        if self.results['tests_failed'] == 0:
            print(f"\n🎉 All tests passed! Phase 7 corpus management is fully functional.")
            overall_success = True
        else:
            print(f"\n⚠️  Some tests failed. Review errors and fix issues before deployment.")
            overall_success = False
        
        self.results['overall_success'] = overall_success
        self.results['success_rate'] = success_rate
        
        return overall_success
    
    def export_results(self, output_path: str = "phase_7_test_results.json"):
        """Export test results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"📄 Test results exported to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to export results: {e}")


def main():
    """Main test runner"""
    print("Phase 7 Corpus Management Test Suite")
    print("Testing comprehensive document corpus management capabilities")
    print()
    
    # Ensure required packages are available
    try:
        import datasketch
        import scipy
        import rich
        import click
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install missing dependencies:")
        print("pip install datasketch scipy rich click")
        sys.exit(1)
    
    # Run the test suite
    test_suite = Phase7TestSuite()
    success = test_suite.run_all_tests()
    
    # Export results
    test_suite.export_results()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
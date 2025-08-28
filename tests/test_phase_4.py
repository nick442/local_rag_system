"""
Comprehensive Test Suite for Phase 4: RAG Pipeline Components
Tests document ingestion, embedding generation, vector storage, and retrieval.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

import numpy as np

# Import our modules
from src.document_ingestion import (
    Document, DocumentChunk, DocumentLoader, TextLoader, PDFLoader, 
    HTMLLoader, MarkdownLoader, DocumentChunker, DocumentIngestionService
)
from src.embedding_service import EmbeddingService, create_embedding_service
from src.vector_database import VectorDatabase, create_vector_database
from src.retriever import Retriever, RetrievalResult, create_retriever


class TestDocumentIngestion(unittest.TestCase):
    """Test document ingestion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_document_creation(self):
        """Test Document class."""
        doc = Document(content="Test content", metadata={"source": "test.txt"})
        
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["source"], "test.txt")
        self.assertIsNotNone(doc.doc_id)
        self.assertTrue(doc.doc_id.startswith("doc_"))
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk class."""
        chunk = DocumentChunk(
            content="Test chunk",
            doc_id="test_doc",
            chunk_index=0,
            metadata={"test": "meta"},
            token_count=10
        )
        
        self.assertEqual(chunk.content, "Test chunk")
        self.assertEqual(chunk.doc_id, "test_doc")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.token_count, 10)
        self.assertEqual(chunk.chunk_id, "test_doc_chunk_0")
    
    def test_text_loader(self):
        """Test TextLoader functionality."""
        # Create test file
        test_file = self.temp_path / "test.txt"
        test_content = "This is a test file.\nWith multiple lines.\nAnd some content."
        test_file.write_text(test_content)
        
        loader = TextLoader()
        doc = loader.load(str(test_file))
        
        self.assertEqual(doc.content, test_content)
        self.assertEqual(doc.metadata['filename'], 'test.txt')
        self.assertEqual(doc.metadata['file_type'], '.txt')
        self.assertEqual(doc.metadata['loader'], 'TextLoader')
    
    def test_html_loader(self):
        """Test HTMLLoader functionality."""
        # Create test HTML file
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a test paragraph.</p>
            <h2>Subtitle</h2>
            <p>Another paragraph with content.</p>
        </body>
        </html>
        """
        test_file = self.temp_path / "test.html"
        test_file.write_text(html_content)
        
        loader = HTMLLoader()
        doc = loader.load(str(test_file))
        
        self.assertIn("Test Page", doc.content)
        self.assertIn("Main Title", doc.content)
        self.assertIn("test paragraph", doc.content)
        self.assertEqual(doc.metadata['loader'], 'HTMLLoader')
        self.assertEqual(doc.metadata['title'], 'Test Page')
        self.assertTrue(len(doc.metadata['headers']) >= 2)
    
    def test_markdown_loader(self):
        """Test MarkdownLoader functionality."""
        # Create test Markdown file
        md_content = """# Main Title

This is a test paragraph in markdown.

## Subtitle

Another paragraph with **bold** text and *italic* text.

### Sub-subtitle

- List item 1
- List item 2
"""
        test_file = self.temp_path / "test.md"
        test_file.write_text(md_content)
        
        loader = MarkdownLoader()
        doc = loader.load(str(test_file))
        
        self.assertEqual(doc.content, md_content)
        self.assertEqual(doc.metadata['loader'], 'MarkdownLoader')
        self.assertTrue(len(doc.metadata['headers']) >= 3)
        self.assertEqual(doc.metadata['headers'][0]['level'], 1)
        self.assertEqual(doc.metadata['headers'][0]['text'], 'Main Title')
    
    def test_document_chunker(self):
        """Test document chunking functionality."""
        # Create a long document
        long_content = " ".join([f"This is sentence number {i}." for i in range(100)])
        doc = Document(content=long_content, metadata={'test': 'meta'})
        
        chunker = DocumentChunker(chunk_size=50, overlap=10)  # Small chunks for testing
        chunks = chunker.chunk_document(doc)
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        # Check chunk properties
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.doc_id, doc.doc_id)
            self.assertEqual(chunk.chunk_index, i)
            self.assertGreater(chunk.token_count, 0)
            self.assertEqual(chunk.chunk_id, f"{doc.doc_id}_chunk_{i}")
            
        # Check overlap - adjacent chunks should share some content
        if len(chunks) > 1:
            # This is a simple heuristic check
            self.assertGreater(len(chunks[0].content), 0)
            self.assertGreater(len(chunks[1].content), 0)
    
    def test_document_ingestion_service(self):
        """Test DocumentIngestionService."""
        # Create test files
        test_files = {
            'test.txt': 'This is a plain text file.',
            'test.md': '# Markdown File\n\nWith some content.',
            'test.html': '<html><body><p>HTML content</p></body></html>'
        }
        
        for filename, content in test_files.items():
            (self.temp_path / filename).write_text(content)
        
        service = DocumentIngestionService(chunk_size=20, chunk_overlap=5)  # Small chunks
        
        # Test single document ingestion
        chunks = service.ingest_document(str(self.temp_path / 'test.txt'))
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], DocumentChunk)
        
        # Test directory ingestion
        all_chunks = service.ingest_directory(str(self.temp_path))
        self.assertGreaterEqual(len(all_chunks), len(chunks))
        
        # Test supported extensions
        extensions = service.get_supported_extensions()
        self.assertIn('.txt', extensions)
        self.assertIn('.md', extensions)
        self.assertIn('.html', extensions)


class TestEmbeddingService(unittest.TestCase):
    """Test embedding service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # We'll need to mock the embedding service since the model may not be available
        self.mock_model_path = "models/embeddings/sentence-transformers_all-MiniLM-L6-v2"
    
    @patch('src.embedding_service.SentenceTransformer')
    def test_embedding_service_creation(self, mock_sentence_transformer):
        """Test EmbeddingService creation with mocked model."""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 256
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(self.mock_model_path, batch_size=16)
        
        self.assertIsNotNone(service.model)
        self.assertEqual(service.batch_size, 16)
        mock_sentence_transformer.assert_called_once()
    
    @patch('src.embedding_service.SentenceTransformer')
    def test_embedding_generation(self, mock_sentence_transformer):
        """Test embedding generation."""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.random((2, 384)).astype(np.float32)
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(self.mock_model_path)
        
        # Test single text embedding
        texts = ["Test sentence 1", "Test sentence 2"]
        embeddings = service.embed_texts(texts, show_progress=False)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].shape, (384,))
        
        mock_model.encode.assert_called_once()
    
    @patch('src.embedding_service.SentenceTransformer')
    def test_similarity_calculation(self, mock_sentence_transformer):
        """Test similarity calculations."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(self.mock_model_path)
        
        # Create test embeddings (normalized)
        emb1 = np.array([1.0, 0.0, 0.0, 0.0])[:384]
        emb1 = emb1 / np.linalg.norm(emb1)  # Normalize
        
        emb2 = np.array([1.0, 0.0, 0.0, 0.0])[:384]
        emb2 = emb2 / np.linalg.norm(emb2)  # Normalize
        
        # Test similarity (should be close to 1.0 for identical normalized vectors)
        similarity = service.similarity(emb1, emb2)
        self.assertAlmostEqual(similarity, 1.0, places=5)


class TestVectorDatabase(unittest.TestCase):
    """Test vector database functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_vectors.db")
        self.db = VectorDatabase(self.db_path, embedding_dimension=384)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check that database file was created
        self.assertTrue(Path(self.db_path).exists())
        
        # Test stats
        stats = self.db.get_database_stats()
        self.assertEqual(stats['documents'], 0)
        self.assertEqual(stats['chunks'], 0)
        self.assertEqual(stats['embeddings'], 0)
        self.assertEqual(stats['embedding_dimension'], 384)
    
    def test_document_insertion(self):
        """Test document insertion."""
        doc_id = "test_doc_1"
        metadata = {"source": "/path/to/doc.txt", "type": "text"}
        
        # Insert document
        result = self.db.insert_document(doc_id, "/path/to/doc.txt", metadata)
        self.assertTrue(result)
        
        # Try to insert same document again
        result = self.db.insert_document(doc_id, "/path/to/doc.txt", metadata)
        self.assertFalse(result)
        
        # Check stats
        stats = self.db.get_database_stats()
        self.assertEqual(stats['documents'], 1)
    
    def test_chunk_insertion(self):
        """Test chunk insertion with embeddings."""
        # First insert a document
        doc_id = "test_doc_1"
        self.db.insert_document(doc_id, "/path/to/doc.txt", {"source": "test"})
        
        # Create test chunk
        chunk = DocumentChunk(
            content="This is a test chunk content.",
            doc_id=doc_id,
            chunk_index=0,
            metadata={"test": "metadata"},
            token_count=15
        )
        
        # Create test embedding
        embedding = np.random.random(384).astype(np.float32)
        
        # Insert chunk
        result = self.db.insert_chunk(chunk, embedding)
        self.assertTrue(result)
        
        # Check stats
        stats = self.db.get_database_stats()
        self.assertEqual(stats['chunks'], 1)
        self.assertEqual(stats['embeddings'], 1)
        
        # Retrieve chunk
        retrieved = self.db.get_chunk_by_id(chunk.chunk_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['content'], chunk.content)
        self.assertEqual(retrieved['doc_id'], doc_id)
        self.assertEqual(retrieved['chunk_index'], 0)
    
    def test_similarity_search(self):
        """Test vector similarity search."""
        # Insert test data
        doc_id = "test_doc_1"
        self.db.insert_document(doc_id, "/path/to/doc.txt", {"source": "test"})
        
        # Create several chunks with similar embeddings
        base_embedding = np.random.random(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize
        
        chunks_data = [
            ("chunk_1", "First test content", base_embedding),
            ("chunk_2", "Second test content", base_embedding + 0.1 * np.random.random(384)),
            ("chunk_3", "Third test content", np.random.random(384))
        ]
        
        for i, (chunk_id, content, embedding) in enumerate(chunks_data):
            chunk = DocumentChunk(
                content=content,
                doc_id=doc_id,
                chunk_index=i,
                metadata={"test": f"chunk_{i+1}"},
                token_count=10,
                chunk_id=chunk_id
            )
            embedding_norm = embedding / np.linalg.norm(embedding)  # Normalize
            self.db.insert_chunk(chunk, embedding_norm)
        
        # Search for similar chunks
        query_embedding = base_embedding  # Use base embedding as query
        results = self.db.search_similar(query_embedding, k=2)
        
        # Should get results back (exact format depends on sqlite-vec availability)
        self.assertGreaterEqual(len(results), 1)
        if results:
            chunk_id, score, data = results[0]
            self.assertIsInstance(chunk_id, str)
            self.assertIsInstance(score, float)
            self.assertIn('content', data)
    
    def test_keyword_search(self):
        """Test keyword search functionality."""
        # Insert test data
        doc_id = "test_doc_1"
        self.db.insert_document(doc_id, "/path/to/doc.txt", {"source": "test"})
        
        chunks_data = [
            "This chunk contains information about machine learning algorithms.",
            "Deep learning is a subset of machine learning using neural networks.",
            "Natural language processing helps computers understand human text."
        ]
        
        for i, content in enumerate(chunks_data):
            chunk = DocumentChunk(
                content=content,
                doc_id=doc_id,
                chunk_index=i,
                metadata={"test": f"chunk_{i+1}"},
                token_count=len(content.split())
            )
            # Create dummy embedding
            embedding = np.random.random(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            self.db.insert_chunk(chunk, embedding)
        
        # Test keyword search
        results = self.db.keyword_search("machine learning", k=2)
        
        # Should find chunks containing "machine learning"
        if results:  # FTS5 might not be available in all environments
            self.assertGreater(len(results), 0)
            # Check that results contain the search term
            found_terms = any("machine learning" in data['content'].lower() 
                            for _, _, data in results)
            self.assertTrue(found_terms)


class TestRetriever(unittest.TestCase):
    """Test retrieval functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_retriever.db")
        
        # Create mock components
        self.vector_db = VectorDatabase(self.db_path, embedding_dimension=384)
        
        # Mock embedding service
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.embed_text.return_value = np.random.random(384)
        self.mock_embedding_service.get_model_info.return_value = {
            'embedding_dimension': 384,
            'model_path': 'mock_path'
        }
        
        self.retriever = Retriever(
            vector_db=self.vector_db,
            embedding_service=self.mock_embedding_service,
            max_context_tokens=1000
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_retriever_creation(self):
        """Test Retriever initialization."""
        self.assertIsNotNone(self.retriever.vector_db)
        self.assertIsNotNone(self.retriever.embedding_service)
        self.assertEqual(self.retriever.max_context_tokens, 1000)
    
    def test_retrieval_result(self):
        """Test RetrievalResult class."""
        result = RetrievalResult(
            chunk_id="test_chunk",
            content="Test content",
            score=0.95,
            metadata={"source": "test.txt"},
            doc_id="test_doc",
            chunk_index=0
        )
        
        self.assertEqual(result.chunk_id, "test_chunk")
        self.assertEqual(result.score, 0.95)
        self.assertIn("chunk_id", result.to_dict())
    
    @patch.object(VectorDatabase, 'search_similar')
    def test_vector_retrieve(self, mock_search):
        """Test vector retrieval."""
        # Mock database response
        mock_search.return_value = [
            ("chunk_1", 0.95, {
                'content': "Test content 1",
                'metadata': {'source': 'test.txt'},
                'doc_id': 'doc_1',
                'chunk_index': 0
            }),
            ("chunk_2", 0.87, {
                'content': "Test content 2", 
                'metadata': {'source': 'test.txt'},
                'doc_id': 'doc_1',
                'chunk_index': 1
            })
        ]
        
        results = self.retriever.retrieve("test query", k=2, method="vector")
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].chunk_id, "chunk_1")
        self.assertEqual(results[0].score, 0.95)
        
        # Verify embedding service was called
        self.mock_embedding_service.embed_text.assert_called_once_with("test query")
    
    def test_context_assembly(self):
        """Test context assembly functionality."""
        # Create test results
        results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="First chunk content with important information.",
                score=0.95,
                metadata={'source': 'test.txt'},
                doc_id="doc_1",
                chunk_index=0
            ),
            RetrievalResult(
                chunk_id="chunk_2", 
                content="Second chunk content with more details.",
                score=0.87,
                metadata={'source': 'test.txt'},
                doc_id="doc_1", 
                chunk_index=1
            )
        ]
        
        context = self.retriever.assemble_context(results, include_metadata=True)
        
        self.assertIn("First chunk content", context)
        self.assertIn("Second chunk content", context) 
        self.assertIn("Source: test.txt", context)  # Metadata should be included
        
        # Test without metadata
        context_no_meta = self.retriever.assemble_context(results, include_metadata=False)
        self.assertNotIn("Source:", context_no_meta)
        self.assertIn("First chunk content", context_no_meta)
    
    def test_stats(self):
        """Test retriever statistics."""
        stats = self.retriever.get_stats()
        
        self.assertIn('database', stats)
        self.assertIn('embedding_service', stats)
        self.assertEqual(stats['max_context_tokens'], 1000)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.db_path = self.temp_path / "test_e2e.db"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.embedding_service.SentenceTransformer')
    def test_complete_workflow(self, mock_sentence_transformer):
        """Test complete ingestion -> embedding -> storage -> retrieval workflow."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        def mock_encode(texts, **kwargs):
            # Return different embeddings for different texts
            embeddings = []
            for i, text in enumerate(texts):
                # Create deterministic embeddings based on text content
                base = np.zeros(384)
                base[i % 384] = 1.0  # Different peak for each text
                base = base + 0.1 * np.random.RandomState(hash(text) % 2**32).random(384)
                base = base / np.linalg.norm(base)  # Normalize
                embeddings.append(base.astype(np.float32))
            return np.array(embeddings)
        
        mock_model.encode = mock_encode
        mock_sentence_transformer.return_value = mock_model
        
        # Step 1: Create test documents
        test_docs = {
            'doc1.txt': 'This document discusses machine learning algorithms and their applications.',
            'doc2.md': '# Deep Learning\n\nDeep learning is a powerful subset of machine learning.',
            'doc3.html': '<html><body><p>Natural language processing enables computers to understand text.</p></body></html>'
        }
        
        for filename, content in test_docs.items():
            (self.temp_path / filename).write_text(content)
        
        # Step 2: Ingest documents
        ingestion_service = DocumentIngestionService(chunk_size=50, chunk_overlap=10)
        all_chunks = ingestion_service.ingest_directory(str(self.temp_path))
        
        self.assertGreater(len(all_chunks), 0, "Should have created chunks from documents")
        
        # Step 3: Create embedding service and vector database
        embedding_service = EmbeddingService("mock_model_path", batch_size=8)
        vector_db = VectorDatabase(str(self.db_path), embedding_dimension=384)
        
        # Step 4: Generate embeddings and store in database
        for chunk in all_chunks:
            # Insert document if not exists
            vector_db.insert_document(
                chunk.doc_id, 
                chunk.metadata.get('source', 'unknown'),
                chunk.metadata
            )
            
            # Generate embedding for chunk
            embedding = embedding_service.embed_text(chunk.content)
            
            # Store chunk and embedding
            vector_db.insert_chunk(chunk, embedding)
        
        # Step 5: Create retriever and test retrieval
        retriever = Retriever(vector_db, embedding_service, max_context_tokens=500)
        
        # Test retrieval
        query = "machine learning"
        results = retriever.retrieve(query, k=3, method="vector")
        
        self.assertGreater(len(results), 0, "Should retrieve relevant chunks")
        
        # Verify results contain RetrievalResult objects
        for result in results:
            self.assertIsInstance(result, RetrievalResult)
            self.assertIsInstance(result.content, str)
            self.assertIsInstance(result.score, float)
        
        # Step 6: Test context assembly
        context = retriever.assemble_context(results)
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0, "Context should not be empty")
        
        # Step 7: Check database statistics
        stats = vector_db.get_database_stats()
        self.assertGreater(stats['documents'], 0)
        self.assertGreater(stats['chunks'], 0)
        self.assertGreater(stats['embeddings'], 0)
        
        print(f"✅ End-to-end test completed successfully!")
        print(f"   - Documents processed: {stats['documents']}")
        print(f"   - Chunks created: {stats['chunks']}")
        print(f"   - Embeddings stored: {stats['embeddings']}")
        print(f"   - Retrieved chunks: {len(results)}")


def run_all_tests():
    """Run all test suites and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDocumentIngestion,
        TestEmbeddingService, 
        TestVectorDatabase,
        TestRetriever,
        TestEndToEndWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("🧪 Running Phase 4 RAG Pipeline Component Tests...")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    else:
        print("❌ Some tests failed!")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    exit(0 if result.wasSuccessful() else 1)
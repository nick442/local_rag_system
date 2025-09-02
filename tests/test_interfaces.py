import os
import tempfile
import unittest

import numpy as np

import importlib


class TestInterfaces(unittest.TestCase):
    def setUp(self):
        self.interfaces = importlib.import_module("src.interfaces")
        self.doc_ingestion = importlib.import_module("src.document_ingestion")
        self.vector_database = importlib.import_module("src.vector_database")
        self.retriever_mod = importlib.import_module("src.retriever")

    def test_chunker_interface_impl(self):
        chunker = self.doc_ingestion.DocumentChunker(chunk_size=64, overlap=16)
        self.assertIsInstance(chunker, self.interfaces.ChunkerInterface)

        # Quick smoke for chunking
        doc = self.doc_ingestion.Document(content="Hello world " * 200)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    def test_vector_index_interface_impl(self):
        with tempfile.TemporaryDirectory() as tmpd:
            db_path = os.path.join(tmpd, "test.db")
            vdb = self.vector_database.VectorDatabase(db_path, embedding_dimension=384)
            self.assertIsInstance(vdb, self.interfaces.VectorIndexInterface)

            # Insert minimal data path: insert_document + stats
            inserted = vdb.insert_document("doc-1", "/tmp/doc.txt", {"size": 10})
            self.assertTrue(inserted)
            stats = vdb.get_database_stats()
            self.assertIn("embedding_dimension", stats)

    def test_retrieval_interface_impl(self):
        # Create a fake vector index and embedding service to avoid loading real models
        class FakeVectorIndex(self.interfaces.VectorIndexInterface):
            def __init__(self):
                self.docs = {}

            def insert_document(self, doc_id, source_path, metadata, collection_id="default"):
                self.docs[doc_id] = {"source": source_path, **metadata}
                return True

            def insert_chunk(self, chunk, embedding, collection_id="default"):
                return True

            def search_similar(self, query_embedding, k=5, metadata_filter=None, collection_id=None):
                # Return deterministic fake results
                return [
                    ("c1", 0.9, {"content": "alpha", "metadata": {"filename": "a.txt"}, "doc_id": "doc-1", "chunk_index": 0}),
                    ("c2", 0.8, {"content": "beta", "metadata": {"filename": "b.txt"}, "doc_id": "doc-2", "chunk_index": 1}),
                ][:k]

            def keyword_search(self, query, k=5, collection_id=None):
                return self.search_similar(np.zeros(384, dtype=np.float32), k)

            def hybrid_search(self, query_embedding, query_text, k=5, alpha=0.7):
                return self.search_similar(query_embedding, k)

            def get_chunk_by_id(self, chunk_id):
                return None

            def get_document_chunks(self, doc_id):
                return []

            def delete_document(self, doc_id):
                return True

            def get_database_stats(self):
                return {"embedding_dimension": 384}

        class FakeEmbeddingService:
            def embed_text(self, text: str):
                return np.zeros(384, dtype=np.float32)

            def get_model_info(self):
                return {"model": "fake-embed", "dim": 384}

        retriever = self.retriever_mod.Retriever(FakeVectorIndex(), FakeEmbeddingService())
        self.assertIsInstance(retriever, self.interfaces.RetrievalInterface)

        results = retriever.retrieve("hello", k=2, method="vector")
        self.assertEqual(len(results), 2)
        self.assertTrue(hasattr(results[0], "chunk_id"))


if __name__ == "__main__":
    unittest.main()


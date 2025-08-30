import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def make_fake_modules():
    fake = {}

    # Fake sentence_transformers to prevent sklearn/scipy import chain via src.__init__
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            pass

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=None):
            import numpy as np
            arr = np.zeros((len(texts), 384), dtype=np.float32)
            if normalize_embeddings and len(texts) > 0:
                arr[:, 0] = 1.0
            return arr

        def get_sentence_embedding_dimension(self):
            return 384

    st_mod.SentenceTransformer = FakeSentenceTransformer
    fake["sentence_transformers"] = st_mod

    # Fake torch
    torch_mod = types.ModuleType("torch")

    class _Backends:
        class _MPS:
            @staticmethod
            def is_available():
                return False

        mps = _MPS()

    class _CUDA:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def memory_allocated():
            return 0

    class _MPS2:
        @staticmethod
        def empty_cache():
            return None

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    torch_mod.backends = _Backends()
    torch_mod.cuda = _CUDA()
    torch_mod.mps = _MPS2()
    torch_mod.no_grad = lambda: _NoGrad()
    fake["torch"] = torch_mod

    # Fake tiktoken
    tk_mod = types.ModuleType("tiktoken")

    class _Encoding:
        name = "fake-encoding"

        def encode(self, text):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    def get_encoding(_):
        return _Encoding()

    tk_mod.get_encoding = get_encoding
    fake["tiktoken"] = tk_mod

    return fake


def fake_sqlite_vec_raising():
    mod = types.ModuleType("sqlite_vec")

    def load(conn):
        raise RuntimeError("forced load failure for test")

    mod.load = load
    return mod


class TestVectorDatabaseFallback(unittest.TestCase):
    def setUp(self):
        # Ensure vendor sqlite-vec is not loaded during this test to avoid segfaults; force manual fallback
        os.environ['RAG_DISABLE_SQLITE_VEC_VENDOR'] = '1'

        # Patch sqlite_vec to raise on load, and stub heavy deps
        fakes = make_fake_modules()
        fakes["sqlite_vec"] = fake_sqlite_vec_raising()
        self.modules_patcher = mock.patch.dict(sys.modules, fakes, clear=False)
        self.modules_patcher.start()

        # Import after patching
        import importlib
        self.vector_database = importlib.import_module("src.vector_database")
        self.doc_ing = importlib.import_module("src.document_ingestion")

    def tearDown(self):
        self.modules_patcher.stop()
        os.environ.pop('RAG_DISABLE_SQLITE_VEC_VENDOR', None)

    def test_manual_similarity_search_fallback(self):
        import numpy as np
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "vec.db"
            vdb = self.vector_database.VectorDatabase(str(db_path), embedding_dimension=384)

            # Insert a single chunk+embedding
            doc_id = "doc-1"
            chunk = self.doc_ing.DocumentChunk(content="hello world", doc_id=doc_id, chunk_index=0, metadata={"filename": "a.txt"})
            embedding = np.ones(384, dtype=np.float32)

            inserted_doc = vdb.insert_document(doc_id, source_path="/tmp/a.txt", metadata={"size": 1}, collection_id="default")
            self.assertTrue(inserted_doc)
            inserted_chunk = vdb.insert_chunk(chunk, embedding, collection_id="default")
            self.assertTrue(inserted_chunk)

            # Query using identical vector; sqlite-vec failed to load, so should use manual dot-product fallback
            q = np.ones(384, dtype=np.float32)
            results = vdb.search_similar(q, k=1)
            self.assertEqual(len(results), 1)
            top_chunk_id, score, data = results[0]
            self.assertEqual(top_chunk_id, chunk.chunk_id)
            # score should be positive (dot product of ones)
            self.assertGreater(score, 0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from pathlib import Path
import sys
import numpy as np


class _FakeEmbeddingService:
    def get_model_info(self):
        return {"embedding_dimension": 384}

    def embed_text(self, text: str):
        # Return a fixed-size vector
        return np.zeros(384, dtype=np.float32)


class _FakeVectorDB:
    def __init__(self):
        self.last_call = None

    # Methods used by vector/keyword paths (not exercised here)
    def search_similar(self, embedding, k, collection_id=None):
        return []

    def keyword_search(self, query, k, collection_id=None):
        return []

    # Hybrid path that we want to verify
    def hybrid_search(self, query_embedding, query, k, alpha, *, collection_id=None,
                      candidate_multiplier=None, fusion_method=None, rrf_k=None):
        self.last_call = {
            "query": query,
            "k": k,
            "alpha": alpha,
            "collection_id": collection_id,
            "candidate_multiplier": candidate_multiplier,
            "fusion_method": fusion_method,
            "rrf_k": rrf_k,
        }
        # Return a minimal valid shape: list of (chunk_id, score, data)
        return [("c1", 1.0, {"content": "x", "metadata": {}, "doc_id": "d1", "chunk_index": 0})]


class TestRetrieverHybridParams(unittest.TestCase):
    def setUp(self):
        # Ensure project root on sys.path
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.retriever import Retriever
        self.Retriever = Retriever

    def test_env_overrides_applied_when_params_none(self):
        os.environ["RAG_HYBRID_CAND_MULT"] = "7"
        os.environ["RAG_HYBRID_FUSION"] = "zscore"
        os.environ["RAG_HYBRID_RRF_K"] = "42"

        try:
            fake_db = _FakeVectorDB()
            fake_emb = _FakeEmbeddingService()
            r = self.Retriever(fake_db, fake_emb)
            r._hybrid_retrieve("q", k=5, collection_id="fiqa", alpha=0.5, candidate_multiplier=None)

            self.assertIsNotNone(fake_db.last_call)
            call = fake_db.last_call
            self.assertEqual(call["candidate_multiplier"], 7)
            self.assertEqual(call["fusion_method"], "zscore")
            self.assertEqual(call["rrf_k"], 42)
        finally:
            for k in ["RAG_HYBRID_CAND_MULT", "RAG_HYBRID_FUSION", "RAG_HYBRID_RRF_K"]:
                os.environ.pop(k, None)

    def test_explicit_candidate_multiplier_takes_precedence(self):
        os.environ["RAG_HYBRID_CAND_MULT"] = "11"
        try:
            fake_db = _FakeVectorDB()
            fake_emb = _FakeEmbeddingService()
            r = self.Retriever(fake_db, fake_emb)
            r._hybrid_retrieve("q", k=10, collection_id=None, alpha=0.7, candidate_multiplier=3)

            call = fake_db.last_call
            self.assertEqual(call["candidate_multiplier"], 3)
            self.assertEqual(call["fusion_method"], os.getenv("RAG_HYBRID_FUSION", "maxnorm"))
            self.assertEqual(call["rrf_k"], int(os.getenv("RAG_HYBRID_RRF_K", "60")))
        finally:
            os.environ.pop("RAG_HYBRID_CAND_MULT", None)


if __name__ == "__main__":
    unittest.main()

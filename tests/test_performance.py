import os
import sys
import time
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def get_rss_mb_fallback():
    """Get current process resident memory in MB using psutil if available, else resource."""
    try:
        import psutil  # type: ignore
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        try:
            import resource  # Unix-only
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux kB; guard with min conversion
            if rss_kb > 10_000_000:  # looks like bytes
                return rss_kb / 1024 / 1024
            return rss_kb / 1024
        except Exception:
            return 0.0


def make_fake_modules():
    fake_modules = {}

    # Fake llama_cpp (no-op model)
    llama_mod = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "ok"}]}

        def tokenize(self, b):
            return list(range(max(1, len(b) // 4)))

    llama_mod.Llama = FakeLlama
    fake_modules["llama_cpp"] = llama_mod

    # Fake sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            self.max_seq_length = 256

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=None):
            import numpy as np
            arr = np.zeros((len(texts), 384), dtype=np.float32)
            if normalize_embeddings and len(texts) > 0:
                arr[:, 0] = 1.0
            return arr

        def get_sentence_embedding_dimension(self):
            return 384

    st_mod.SentenceTransformer = FakeSentenceTransformer
    fake_modules["sentence_transformers"] = st_mod

    # Fake torch minimal
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
    fake_modules["torch"] = torch_mod

    # Fake tiktoken
    tk_mod = types.ModuleType("tiktoken")

    class _Encoding:
        name = "fake-encoding"

        def encode(self, text):
            return list(text)

    def get_encoding(_):
        return _Encoding()

    tk_mod.get_encoding = get_encoding
    fake_modules["tiktoken"] = tk_mod

    return fake_modules


class TestPerformancePhase7(unittest.TestCase):
    def setUp(self):
        self.modules_patcher = mock.patch.dict(sys.modules, make_fake_modules(), clear=False)
        self.modules_patcher.start()

        import importlib
        self.rag_pipeline = importlib.import_module("src.rag_pipeline")

        # Patch retriever and prompt/llm to be lightweight
        class _FakeRetriever:
            def retrieve(self, query, k=5, method="vector", collection_id=None):
                retriever_mod = __import__("src.retriever", fromlist=["RetrievalResult"])
                return [
                    retriever_mod.RetrievalResult(
                        chunk_id="c1",
                        content="dummy",
                        score=1.0,
                        metadata={"filename": "a.txt"},
                        doc_id="d1",
                        chunk_index=0,
                    )
                ]

            def get_stats(self):
                return {"backend": "fake"}

        class _FakePromptBuilder:
            def truncate_contexts_to_fit(self, user_query, contexts, max_context, system_prompt, generation_buffer=2048):
                return contexts, "PROMPT"

            def count_prompt_tokens(self, prompt):
                return 10

            def get_template_info(self):
                return {"template_type": "fake"}

        class _FakeLLM:
            n_ctx = 4096

            def generate_with_stats(self, prompt, **kwargs):
                return {
                    "generated_text": "ok",
                    "prompt_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "generation_time": 0.002,
                    "tokens_per_second": 2000.0,
                    "context_remaining": 4096 - 15,
                }

            def get_model_info(self):
                return {"model": "fake"}

        self.ret_patch = mock.patch.object(self.rag_pipeline, "create_retriever", return_value=_FakeRetriever())
        self.pb_patch = mock.patch.object(self.rag_pipeline, "create_prompt_builder", return_value=_FakePromptBuilder())
        self.llm_patch = mock.patch.object(self.rag_pipeline, "create_llm_wrapper", return_value=_FakeLLM())
        self.ret_patch.start(); self.pb_patch.start(); self.llm_patch.start()

        # Use a temp database path for isolation and cleanup
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "test_perf.db")
        self.pipeline = self.rag_pipeline.RAGPipeline(
            db_path=self.db_path,
            embedding_model_path="models/embeddings/fake",
            llm_model_path="models/llm/fake.gguf",
        )

    def tearDown(self):
        self.ret_patch.stop(); self.pb_patch.stop(); self.llm_patch.stop()
        self.modules_patcher.stop()
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()

    def test_memory_usage_under_16gb(self):
        baseline_mb = get_rss_mb_fallback()
        # Perform a few operations to potentially grow memory
        for _ in range(3):
            _ = self.pipeline.query("what is ML?", k=1)
        after_mb = get_rss_mb_fallback()
        delta_mb = max(0.0, after_mb - baseline_mb)
        # Absolute guard and relative delta guard to reduce flakiness
        self.assertLess(after_mb, 16_384, msg=f"RSS too high: {after_mb:.1f}MB")
        self.assertLess(delta_mb, 512, msg=f"Memory delta unexpectedly large: {delta_mb:.1f}MB")

    def test_query_latency_under_reasonable_threshold(self):
        # Run a few queries and ensure latency is low with fakes
        times = []
        for _ in range(5):
            t0 = time.time()
            out = self.pipeline.query("what is ML?", k=1)
            self.assertIn('answer', out)
            times.append(time.time() - t0)

        avg = sum(times) / len(times)
        self.assertLess(avg, 0.2, msg=f"avg latency too high: {avg:.3f}s")


if __name__ == "__main__":
    unittest.main()

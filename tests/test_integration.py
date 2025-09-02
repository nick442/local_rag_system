import os
import sys
import types
import unittest
import tempfile
import importlib
from pathlib import Path
from unittest import mock
from contextlib import contextmanager


EMBEDDING_DIM = 384


def make_fake_third_party_modules(counters=None):
    counters = counters or {}
    fake_modules = {}

    # Fake sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            counters['st_loads'] = counters.get('st_loads', 0) + 1
            self._model_name = f"fake-embed:{Path(model_path).name}:{device}"
            self.max_seq_length = 256

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=None):
            import numpy as np
            arr = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
            if normalize_embeddings and len(texts) > 0:
                arr[:, 0] = 1.0
            return arr

        def get_sentence_embedding_dimension(self):
            return EMBEDDING_DIM

    st_mod.SentenceTransformer = FakeSentenceTransformer
    fake_modules["sentence_transformers"] = st_mod

    # Fake torch (minimal API used in embedding_service)
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

    # Fake llama_cpp with load counter
    llama_mod = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            counters['llm_loads'] = counters.get('llm_loads', 0) + 1

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "ok"}]}

        def tokenize(self, b):
            return list(range(max(1, len(b) // 4)))

    llama_mod.Llama = FakeLlama
    fake_modules["llama_cpp"] = llama_mod

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


class TestIntegrationPhase7(unittest.TestCase):
    def setUp(self):
        # Ensure repo root as cwd
        self.repo_root = Path(__file__).resolve().parents[1]
        os.chdir(self.repo_root)

        self.counters = {}
        self.fake_modules = make_fake_third_party_modules(self.counters)
        self.modules_patcher = mock.patch.dict(sys.modules, self.fake_modules, clear=False)
        self.modules_patcher.start()

        self.rag_pipeline = importlib.import_module("src.rag_pipeline")
        self.config_manager = importlib.import_module("src.config_manager")

    def tearDown(self):
        self.modules_patcher.stop()

    def _fake_retriever_factory(self):
        # Minimal fake retriever honoring RetrievalInterface
        class _FakeRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, query, k=5, method="vector", collection_id=None):
                self.calls.append({"k": k, "method": method, "collection_id": collection_id})
                # Minimal RetrievalResult payload compatible with RAGPipeline expectations
                return [
                    self._mk_result("c1", "alpha", 0.9, 0),
                    self._mk_result("c2", "beta", 0.8, 1),
                ][:k]

            def _mk_result(self, cid, content, score, idx):
                # Import inside to avoid type concerns on class body
                retriever_mod = __import__("src.retriever", fromlist=["RetrievalResult"])
                return retriever_mod.RetrievalResult(
                    chunk_id=cid,
                    content=content,
                    score=score,
                    metadata={"filename": f"doc{idx}.txt"},
                    doc_id=f"doc-{idx}",
                    chunk_index=idx,
                )

            def get_stats(self):
                return {"backend": "fake"}

        return _FakeRetriever()

    def _fake_prompt_builder_factory(self):
        class _FakePromptBuilder:
            def truncate_contexts_to_fit(self, user_query, contexts, max_context, system_prompt, generation_buffer=2048):
                return contexts, f"PROMPT: {user_query}"

            def count_prompt_tokens(self, prompt):
                return 10

            def get_template_info(self):
                return {"template_type": "fake"}

        return _FakePromptBuilder()

    def _fake_llm_wrapper_factory(self):
        class _FakeLLM:
            n_ctx = 4096

            def generate_with_stats(self, prompt, **kwargs):
                return {
                    "generated_text": "ok",
                    "prompt_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "generation_time": 0.005,
                    "tokens_per_second": 1000.0,
                    "context_remaining": self.n_ctx - 15,
                }

            def get_model_info(self):
                return {"model": "fake-llm"}

        return _FakeLLM()

    @contextmanager
    def pipeline_with_mocks(self, profile_config=None, db_path: str = None, *, mock_llm: bool = True, llm_model_path: str | None = None):
        """Context manager to create RAGPipeline with mocked components."""
        if db_path is None:
            # Use a temp database path per invocation
            tmpd = tempfile.TemporaryDirectory()
            db_path = str(Path(tmpd.name) / "test_integration.db")
        else:
            tmpd = None
        ret = self._fake_retriever_factory()
        pb = self._fake_prompt_builder_factory()
        patches = [
            mock.patch.object(self.rag_pipeline, "create_retriever", return_value=ret),
            mock.patch.object(self.rag_pipeline, "create_prompt_builder", return_value=pb),
        ]
        if mock_llm:
            patches.append(mock.patch.object(self.rag_pipeline, "create_llm_wrapper", return_value=self._fake_llm_wrapper_factory()))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            rp = self.rag_pipeline.RAGPipeline(
                db_path=db_path,
                embedding_model_path="models/embeddings/fake",
                llm_model_path=llm_model_path or "models/llm/fake.gguf",
                profile_config=profile_config,
            )
            try:
                yield rp, ret
            finally:
                # Cleanup temp dir if we created one
                if tmpd is not None:
                    tmpd.cleanup()

    def test_config_profile_propagation(self):
        profile = self.config_manager.ProfileConfig(
            retrieval_k=3,
            max_tokens=512,
            temperature=0.6,
            chunk_size=256,
            chunk_overlap=32,
            n_ctx=4096,
        )

        with self.pipeline_with_mocks(profile_config=profile) as (rp, _ret):
            cfg = rp.config
            # Verify propagation to nested blocks
            self.assertEqual(cfg['retrieval']['default_k'], 3)
            self.assertEqual(cfg['chunking']['chunk_size'], 256)
            self.assertEqual(cfg['chunking']['chunk_overlap'], 32)
            self.assertEqual(cfg['llm_params']['n_ctx'], 4096)
            self.assertEqual(cfg['llm_params']['max_tokens'], 512)
            self.assertAlmostEqual(cfg['llm_params']['temperature'], 0.6)

            # Ensure query uses retrieval method and returns expected structure
            out = rp.query("what is ML?", k=0, retrieval_method="hybrid", collection_id="demo")
            self.assertIn('answer', out)
            self.assertIn('sources', out)
            self.assertEqual(out['metadata']['retrieval_method'], 'hybrid')

    def test_model_caching_across_pipelines(self):
        # Create a temporary fake LLM model file
        with tempfile.TemporaryDirectory() as tmpd:
            model_path = Path(tmpd) / "fake.gguf"
            model_path.write_text("fake")

            # First pipeline
            with self.pipeline_with_mocks(
                profile_config=self.config_manager.ProfileConfig(
                    retrieval_k=5, max_tokens=512, temperature=0.7, chunk_size=256, chunk_overlap=64, n_ctx=4096
                ),
                db_path=str(Path(model_path).with_suffix('.db')),
                mock_llm=False,
                llm_model_path=str(model_path),
            ) as (rp1, _):
                pass

            # Second pipeline with identical LLM params should reuse cache
            with self.pipeline_with_mocks(
                profile_config=self.config_manager.ProfileConfig(
                    retrieval_k=5, max_tokens=512, temperature=0.7, chunk_size=256, chunk_overlap=64, n_ctx=4096
                ),
                db_path=str(Path(model_path).with_suffix('.db')),
                mock_llm=False,
                llm_model_path=str(model_path),
            ) as (rp2, _):
                pass

            # Verify ModelCache stats reflect reuse
            mc = importlib.import_module("src.model_cache").ModelCache.instance()
            stats = mc.stats()
            self.assertEqual(stats["llm_models"], 1)
            # One miss for first load, one hit for second
            self.assertEqual(stats["llm_misses"], 1)
            self.assertGreaterEqual(stats["llm_hits"], 1)

    def test_interface_swapping_method_and_collection(self):
        with self.pipeline_with_mocks() as (rp, fake_ret):
            rp.query("hello", k=2, retrieval_method="keyword", collection_id="colA")
            rp.query("hello", k=2, retrieval_method="hybrid", collection_id="colB")

            # Verify retriever was called with requested methods and collection ids
            methods = [c["method"] for c in fake_ret.calls]
            cols = [c["collection_id"] for c in fake_ret.calls]
            self.assertEqual(methods, ["keyword", "hybrid"])
            self.assertEqual(cols, ["colA", "colB"])


if __name__ == "__main__":
    unittest.main()

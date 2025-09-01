import os
import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock


def make_fake_modules(counter_dict):
    """Create lightweight fake third-party modules with counters to verify loads."""
    fake_modules = {}

    # Fake sentence_transformers with load counter
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            counter_dict['st_loads'] = counter_dict.get('st_loads', 0) + 1
            self._model_name = f"fake-embed:{Path(model_path).name}:{device}"
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

    # Fake llama_cpp with load counter
    llama_mod = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            counter_dict['llm_loads'] = counter_dict.get('llm_loads', 0) + 1

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

    # Fake bs4
    bs4_mod = types.ModuleType("bs4")

    class _FakeSoup:
        def __init__(self, html, parser):
            self.title = None

        def find_all(self, *args, **kwargs):
            return []

        def get_text(self):
            return ""

    bs4_mod.BeautifulSoup = _FakeSoup
    fake_modules["bs4"] = bs4_mod

    # Fake html2text
    h2t_mod = types.ModuleType("html2text")

    class _H2T:
        def __init__(self):
            self.ignore_links = False
            self.ignore_images = True
            self.body_width = 0

        def handle(self, html):
            return ""

    h2t_mod.HTML2Text = _H2T
    fake_modules["html2text"] = h2t_mod

    # Fake markdown
    md_mod = types.ModuleType("markdown")
    fake_modules["markdown"] = md_mod

    # Fake PyPDF2
    pypdf2_mod = types.ModuleType("PyPDF2")

    class _Page:
        def extract_text(self):
            return ""

    class _Reader:
        def __init__(self, f):
            self.pages = []

    pypdf2_mod.PdfReader = _Reader
    fake_modules["PyPDF2"] = pypdf2_mod

    return fake_modules


class TestModelCache(unittest.TestCase):
    def setUp(self):
        self.counters = {}
        self.fake_modules = make_fake_modules(self.counters)
        self.modules_patcher = mock.patch.dict(sys.modules, self.fake_modules, clear=False)
        self.modules_patcher.start()

        import importlib
        # Import after fakes are installed
        self.model_cache = importlib.import_module("src.model_cache")
        self.embedding_service = importlib.import_module("src.embedding_service")
        self.llm_wrapper = importlib.import_module("src.llm_wrapper")
        self.vector_database = importlib.import_module("src.vector_database")

        # Ensure clean cache between tests
        self.model_cache.ModelCache.instance().clear()

    def tearDown(self):
        self.modules_patcher.stop()

    def test_singleton_instance(self):
        a = self.model_cache.ModelCache.instance()
        b = self.model_cache.ModelCache.instance()
        self.assertIs(a, b)

    def test_embedding_model_cached(self):
        # Two services with same path/device should reuse same underlying model
        svc1 = self.embedding_service.EmbeddingService("models/embeddings/fake", device="cpu")
        svc2 = self.embedding_service.EmbeddingService("models/embeddings/fake", device="cpu")

        self.assertIsNotNone(svc1.model)
        self.assertIs(svc1.model, svc2.model)
        # Loader should be invoked only once
        self.assertEqual(self.counters.get('st_loads', 0), 1)

    def test_llm_model_cached_by_init_params(self):
        with tempfile.TemporaryDirectory() as tmpd:
            model_path = Path(tmpd) / "fake.gguf"
            model_path.write_text("fake")
            w1 = self.llm_wrapper.LLMWrapper(str(model_path), n_ctx=2048, n_gpu_layers=-1)
            w2 = self.llm_wrapper.LLMWrapper(str(model_path), n_ctx=2048, n_gpu_layers=-1)
        self.assertIs(w1.model, w2.model)
        self.assertEqual(self.counters.get('llm_loads', 0), 1)

        # Changing a construction-critical param should lead to a different cached instance
        with tempfile.TemporaryDirectory() as tmpd2:
            model_path2 = Path(tmpd2) / "fake.gguf"
            model_path2.write_text("fake")
            w3 = self.llm_wrapper.LLMWrapper(str(model_path2), n_ctx=4096, n_gpu_layers=-1)
        self.assertIsNot(w1.model, w3.model)
        self.assertEqual(self.counters.get('llm_loads', 0), 2)

    def test_vector_db_embedding_dimension_validation(self):
        # Initialize DB with one embedding dimension, then reopen with conflicting dimension
        with tempfile.TemporaryDirectory() as tmpd:
            db_path = Path(tmpd) / "test.db"
            # First init with 384
            v1 = self.vector_database.VectorDatabase(str(db_path), embedding_dimension=384)
            stats1 = v1.get_database_stats()
            self.assertEqual(stats1['embedding_dimension'], 384)

            # Reopen with same dimension should be fine
            v2 = self.vector_database.VectorDatabase(str(db_path), embedding_dimension=384)
            self.assertIsInstance(v2, self.vector_database.VectorDatabase)

            # Reopen with a different dimension should raise
            with self.assertRaises(ValueError):
                self.vector_database.VectorDatabase(str(db_path), embedding_dimension=768)


if __name__ == "__main__":
    unittest.main()

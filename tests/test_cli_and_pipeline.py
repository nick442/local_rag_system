import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def make_fake_modules():
    """Create lightweight fake third-party modules to satisfy imports."""
    fake_modules = {}

    # Fake sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            self._model_name = "fake-model"
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
    fake_modules["torch"] = torch_mod

    # Fake llama_cpp
    llama_mod = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, max_tokens=128, temperature=0.7, top_p=0.95, stop=None, stream=False, echo=False):
            if stream:
                def gen():
                    for ch in "Hello from fake LLM":
                        yield {"choices": [{"text": ch}]}
                return gen()
            return {"choices": [{"text": "Hello from fake LLM"}]}

        def tokenize(self, b):
            # simple token count proxy
            return list(range(max(1, len(b) // 4)))

    llama_mod.Llama = FakeLlama
    fake_modules["llama_cpp"] = llama_mod

    # Fake sqlite_vec
    sqlite_vec_mod = types.ModuleType("sqlite_vec")

    def fake_load(conn):
        # Default: succeed; individual tests may override
        return None

    sqlite_vec_mod.load = fake_load
    fake_modules["sqlite_vec"] = sqlite_vec_mod

    # Fake tiktoken
    tk_mod = types.ModuleType("tiktoken")

    class _Encoding:
        name = "fake-encoding"

        def encode(self, text):
            # naive char-based tokenization
            return list(text)

    def get_encoding(_):
        return _Encoding()

    tk_mod.get_encoding = get_encoding
    fake_modules["tiktoken"] = tk_mod

    # Fake datasketch (to avoid SciPy dependency during import of deduplication)
    ds_mod = types.ModuleType("datasketch")

    class FakeMinHash:
        def __init__(self, num_perm=128):
            self._tokens = set()
            self.num_perm = num_perm

        def update(self, b):
            # b is bytes of a token
            try:
                token = b.decode("utf-8")
            except Exception:
                token = str(b)
            self._tokens.add(token)

        def jaccard(self, other):
            a = self._tokens
            b = other._tokens
            if not a and not b:
                return 1.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

    class FakeMinHashLSH:
        def __init__(self, threshold=0.8, num_perm=128):
            self.threshold = threshold
            self.num_perm = num_perm
            self._store = {}

        def insert(self, key, minhash):
            self._store[key] = minhash

        def query(self, minhash):
            out = []
            for k, mh in self._store.items():
                if minhash.jaccard(mh) >= self.threshold:
                    out.append(k)
            return out

    ds_mod.MinHash = FakeMinHash
    ds_mod.MinHashLSH = FakeMinHashLSH
    fake_modules["datasketch"] = ds_mod

    return fake_modules


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        # Ensure cwd is repo root (tests can be run from project root)
        self.repo_root = Path(__file__).resolve().parents[1]
        os.chdir(self.repo_root)

        # Patch sys.modules with fakes before importing main
        self.fake_modules = make_fake_modules()
        self.modules_patcher = mock.patch.dict(sys.modules, self.fake_modules, clear=False)
        self.modules_patcher.start()

        # Import main after patching
        import importlib
        self.main = importlib.import_module("main")

    def tearDown(self):
        self.modules_patcher.stop()

    def test_status_command_no_db(self):
        from click.testing import CliRunner

        runner = CliRunner()
        # Use repo working dir so existing 'logs/' path avoids FileHandler errors
        db_path = Path("data/__nonexistent_status_test__.db")
        if db_path.exists():
            db_path.unlink()
        result = runner.invoke(self.main.cli, ["--db-path", str(db_path), "status"]) 
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Should mention Database path and not found
        self.assertIn("RAG System Status", result.output)
        self.assertTrue("Database not found" in result.output or "❌ Database not found" in result.output)

    def test_ingest_directory_dry_run_uses_manager(self):
        from click.testing import CliRunner

        class FakeStats:
            files_scanned = 3
            files_processed = 3
            files_failed = 0
            chunks_created = 9
            chunks_embedded = 9
            processing_time = 0.12

        class FakeManager:
            async def ingest_directory(self, **kwargs):
                return FakeStats()

        with mock.patch.object(self.main, "create_corpus_manager", return_value=FakeManager()):
            runner = CliRunner()
            # Use sample_corpus path (exists in repo)
            sample_dir = self.repo_root / "sample_corpus"
            result = runner.invoke(
                self.main.cli,
                ["ingest", "directory", str(sample_dir), "--collection", "demo", "--dry-run"],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("DRY RUN - Would process", result.output)

    def test_query_command_calls_rag_pipeline(self):
        from click.testing import CliRunner

        class FakeRAG:
            def __init__(self, db_path, embedding_model_path, llm_model_path):
                self.params = (db_path, embedding_model_path, llm_model_path)

            def query(self, question, k=5, collection_id="default"):
                # main.py expects key 'response' currently
                return {"response": "Hello from pipeline", "retrieval_results": []}

        with mock.patch.object(self.main, "RAGPipeline", FakeRAG):
            runner = CliRunner()
            result = runner.invoke(
                self.main.cli,
                [
                    "--db-path",
                    "data/test_cli.db",
                    "query",
                    "What is machine learning?",
                    "--collection",
                    "demo",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Answer:", result.output)
            self.assertIn("Hello from pipeline", result.output)


class TestRAGPipelineFlow(unittest.TestCase):
    def setUp(self):
        self.fake_modules = make_fake_modules()
        self.modules_patcher = mock.patch.dict(sys.modules, self.fake_modules, clear=False)
        self.modules_patcher.start()

        # Import after fakes are in place
        import importlib
        self.rag_pipeline = importlib.import_module("src.rag_pipeline")
        self.retriever = importlib.import_module("src.retriever")

    def tearDown(self):
        self.modules_patcher.stop()

    def test_pipeline_query_returns_expected_structure(self):
        # Prepare fakes for components created inside RAGPipeline
        class FakeRetriever:
            def __init__(self, *args, **kwargs):
                pass

            def retrieve(self, query, k=5, method="vector"):
                # Return minimal RetrievalResult objects
                return [
                    self._mk_result("c1", "content 1", 0.9, 0),
                    self._mk_result("c2", "content 2", 0.8, 1),
                ]

            def _mk_result(self, cid, content, score, idx):
                return self.retriever_cls.RetrievalResult(
                    chunk_id=cid,
                    content=content,
                    score=score,
                    metadata={"filename": f"doc{idx}.txt"},
                    doc_id=f"doc-{idx}",
                    chunk_index=idx,
                )

        class FakePromptBuilder:
            def __init__(self, *args, **kwargs):
                self.n = 0

            def truncate_contexts_to_fit(self, user_query, contexts, max_context, system_prompt, generation_buffer=2048):
                return contexts, "PROMPT"

            def count_prompt_tokens(self, prompt):
                return 10

            def get_template_info(self):
                return {"template_type": "fake"}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                self.n_ctx = 8192

            def generate_with_stats(self, prompt, **kwargs):
                return {
                    "generated_text": "This is an answer.",
                    "prompt_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "generation_time": 0.01,
                    "tokens_per_second": 500.0,
                    "context_remaining": 8000,
                }

            def get_model_info(self):
                return {"model": "fake"}

        # Bind retriever_cls on FakeRetriever for nested reference
        FakeRetriever.retriever_cls = self.retriever

        with mock.patch.object(self.rag_pipeline, "create_retriever", return_value=FakeRetriever()):
            with mock.patch.object(self.rag_pipeline, "create_prompt_builder", return_value=FakePromptBuilder()):
                with mock.patch.object(self.rag_pipeline, "create_llm_wrapper", return_value=FakeLLM()):
                    rp = self.rag_pipeline.RAGPipeline(
                        db_path="data/test_rag.db",
                        embedding_model_path="models/embeddings/fake",
                        llm_model_path="models/llm/fake.gguf",
                    )
                    out = rp.query("What is ML?", k=2)
                    # Validate structure from docs
                    self.assertIn("answer", out)
                    self.assertIn("sources", out)
                    self.assertIn("contexts", out)
                    self.assertIn("metadata", out)
                    self.assertEqual(out["answer"], "This is an answer.")


if __name__ == "__main__":
    unittest.main()

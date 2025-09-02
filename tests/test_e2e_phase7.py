import os
import sys
import types
import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def make_fakes():
    """Lightweight fakes for heavy third-party deps to enable E2E run."""
    fake_modules = {}

    # sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            self._model_name = f"fake-embed:{Path(str(model_path)).name}:{device}"
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

    # torch
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

    # llama_cpp
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

    # tiktoken
    tk_mod = types.ModuleType("tiktoken")

    class _Encoding:
        name = "fake-encoding"

        def encode(self, text):
            return list(text)

        def decode(self, tokens):
            try:
                return ''.join(tokens)
            except Exception:
                return ''

    def get_encoding(_):
        return _Encoding()

    tk_mod.get_encoding = get_encoding
    fake_modules["tiktoken"] = tk_mod

    # bs4
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

    # html2text
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

    # markdown
    md_mod = types.ModuleType("markdown")
    fake_modules["markdown"] = md_mod

    # PyPDF2
    pypdf2_mod = types.ModuleType("PyPDF2")

    class _Reader:
        def __init__(self, f):
            self.pages = []

    pypdf2_mod.PdfReader = _Reader
    fake_modules["PyPDF2"] = pypdf2_mod

    # sqlite_vec (optional): ensure no load error
    sqlite_vec = types.ModuleType("sqlite_vec")
    def _load(conn):
        return None
    sqlite_vec.load = _load
    fake_modules["sqlite_vec"] = sqlite_vec

    return fake_modules


class TestEndToEndPhase7(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        os.chdir(self.repo_root)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "e2e.db")
        self.metrics_path = str(Path(self.tmpdir.name) / "metrics.jsonl")
        self.out_path = str(Path(self.tmpdir.name) / "e2e_out.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_e2e_ingest_and_query_with_metrics(self):
        # Build a subprocess script that installs fakes, runs ingest+query, and writes results
        script_template = r"""
import os, sys, json
from pathlib import Path

# Fakes for heavy deps
def make_fakes():
    import types
    fake_modules = {}

    st_mod = types.ModuleType("sentence_transformers")
    class FakeSentenceTransformer:
        def __init__(self, model_path, device=None):
            self._model_name = f"fake-embed:{Path(str(model_path)).name}:{device}"
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

    tk_mod = types.ModuleType("tiktoken")
    class _Encoding:
        name = "fake-encoding"
        def encode(self, text):
            return list(text)
        def decode(self, tokens):
            try:
                return ''.join(tokens)
            except Exception:
                return ''
    def get_encoding(_):
        return _Encoding()
    tk_mod.get_encoding = get_encoding
    fake_modules["tiktoken"] = tk_mod

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

    md_mod = types.ModuleType("markdown")
    fake_modules["markdown"] = md_mod

    pypdf2_mod = types.ModuleType("PyPDF2")
    class _Reader:
        def __init__(self, f):
            self.pages = []
    pypdf2_mod.PdfReader = _Reader
    fake_modules["PyPDF2"] = pypdf2_mod

    sqlite_vec = types.ModuleType("sqlite_vec")
    def _load(conn):
        return None
    sqlite_vec.load = _load
    fake_modules["sqlite_vec"] = sqlite_vec
    return fake_modules

for name, mod in make_fakes().items():
    sys.modules[name] = mod

os.environ['RAG_ENABLE_METRICS'] = '1'
os.environ['RAG_METRICS_PATH'] = '@@METRICS_PATH@@'

sys.path.insert(0, '@@REPO_ROOT@@')
import importlib as _il
# Ensure embedding_service sees a global 'torch'
es = _il.import_module('src.embedding_service')
if not hasattr(es, 'torch'):
    es.torch = sys.modules['torch']
from src.corpus_manager import create_corpus_manager
from src.rag_pipeline import RAGPipeline
from src.config_manager import ProfileConfig

sample_dir = Path('@@SAMPLE_DIR@@')
db_path = '@@DB_PATH@@'

import asyncio
async def run_ingest():
    manager = create_corpus_manager(db_path=db_path, embedding_model_path='models/embeddings/fake', max_workers=2)
    return await manager.ingest_directory(path=str(sample_dir), collection_id='e2e_demo', dry_run=False, resume=False, deduplicate=True)

stats = asyncio.run(run_ingest())

fake_model_path = Path('@@TMPDIR@@') / 'fake.gguf'
fake_model_path.write_text('fake')

profile = ProfileConfig(retrieval_k=3, max_tokens=128, temperature=0.2, chunk_size=256, chunk_overlap=32, n_ctx=2048)
rp = RAGPipeline(db_path=db_path, embedding_model_path='models/embeddings/fake', llm_model_path=str(fake_model_path), profile_config=profile)
out = rp.query('sample', k=3, retrieval_method='vector', collection_id='e2e_demo', max_tokens=128)

res = {
  'files_processed': stats.files_processed,
  'chunks_created': stats.chunks_created,
  'contexts_len': len(out.get('contexts', [])),
  'sources_len': len(out.get('sources', [])),
}
Path('@@OUT_PATH@@').write_text(json.dumps(res))
print('OK')
        """

        py = (
            script_template
            .replace('@@REPO_ROOT@@', str(self.repo_root).replace('\\', '\\\\'))
            .replace('@@METRICS_PATH@@', self.metrics_path.replace('\\', '\\\\'))
            .replace('@@SAMPLE_DIR@@', str(self.repo_root / 'sample_corpus').replace('\\', '\\\\'))
            .replace('@@DB_PATH@@', self.db_path.replace('\\', '\\\\'))
            .replace('@@TMPDIR@@', self.tmpdir.name.replace('\\', '\\\\'))
            .replace('@@OUT_PATH@@', self.out_path.replace('\\', '\\\\'))
        )

        # Write the script to tmp and execute with current python
        script_path = Path(self.tmpdir.name) / "e2e_run.py"
        script_path.write_text(py)

        import subprocess
        proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stdout: {proc.stdout}\nstderr: {proc.stderr}")

        # Validate output JSON
        res = json.loads(Path(self.out_path).read_text())
        self.assertGreaterEqual(res['files_processed'], 1)
        self.assertGreater(res['chunks_created'], 0)
        self.assertGreater(res['contexts_len'], 0)
        self.assertGreater(res['sources_len'], 0)

        # Metrics file should exist and have at least one line
        self.assertTrue(Path(self.metrics_path).exists())
        with open(self.metrics_path, 'r', encoding='utf-8') as f:
            self.assertTrue(f.readline().strip().startswith('{'))


if __name__ == "__main__":
    unittest.main()

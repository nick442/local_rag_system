import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


def run_in_conda(cmd: str) -> subprocess.CompletedProcess:
    # Use bash -lc to allow `source` and conda activation from CLAUDE.md
    full_cmd = f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && {cmd}"
    return subprocess.run(["bash", "-lc", full_cmd], capture_output=True, text=True)


class TestEndToEndRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        os.chdir(cls.repo_root)
        cls.db_path = cls.repo_root / "data/test_200_e2e.db"
        cls.collection = "test200"
        cls.data_dir = cls.repo_root / "data/test_200"
        cls.model_path = cls.repo_root / "models/gemma-3-4b-it-q4_0.gguf"
        cls.embedding_dir = cls.repo_root / (
            "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        )

    def setUp(self):
        # Preconditions: data dir and local models exist; otherwise skip heavy e2e
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            self.skipTest(f"Real data not found at {self.data_dir}")
        if not self.model_path.exists():
            self.skipTest(f"LLM model not found at {self.model_path}")
        if not self.embedding_dir.exists():
            self.skipTest(f"Embedding model snapshot not found at {self.embedding_dir}")

        # Fresh DB
        if self.db_path.exists():
            self.db_path.unlink()

    def test_ingest_and_query_end_to_end(self):
        # Ingest directory with conservative settings to limit resource use
        ingest_cmd = (
            f"python main.py --db-path {self.db_path} ingest directory {self.data_dir} "
            f"--collection {self.collection} --max-workers 1 --batch-size 8"
        )
        r1 = run_in_conda(ingest_cmd)
        self.assertEqual(r1.returncode, 0, msg=f"ingest failed:\nSTDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}")
        self.assertIn("Processing complete", r1.stdout)

        # Status check
        status_cmd = f"python main.py --db-path {self.db_path} status"
        r2 = run_in_conda(status_cmd)
        self.assertEqual(r2.returncode, 0, msg=f"status failed:\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
        self.assertIn("RAG System Status", r2.stdout)
        self.assertTrue("Collections:" in r2.stdout or "📚 Collections:" in r2.stdout)

        # Queries
        queries = [
            "What is machine learning?",
            "Define a neural network in one sentence.",
            "List two supervised learning algorithms.",
        ]
        for q in queries:
            q_cmd = (
                f"python main.py --db-path {self.db_path} query \"{q}\" --collection {self.collection} --k 3"
            )
            rq = run_in_conda(q_cmd)
            self.assertEqual(rq.returncode, 0, msg=f"query failed for: {q}\nSTDOUT:\n{rq.stdout}\nSTDERR:\n{rq.stderr}")
            self.assertIn("Answer:", rq.stdout)


if __name__ == "__main__":
    unittest.main()


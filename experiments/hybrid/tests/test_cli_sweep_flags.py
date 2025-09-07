import unittest
from click.testing import CliRunner

# Import the click CLI from main
import importlib.util
import sys
from pathlib import Path


class TestCliSweepFlags(unittest.TestCase):
    def setUp(self):
        # Ensure the project root is on sys.path
        # Repo root: two levels up from this tests/ folder when run under experiments/hybrid CWD
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # Dynamically import main as a module to get the click CLI group
        spec = importlib.util.spec_from_file_location("rag_main", str(root / "main.py"))
        self.main_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.main_mod)

        self.runner = CliRunner()

    def test_experiment_sweep_help_includes_hybrid_flags(self):
        result = self.runner.invoke(self.main_mod.experiment.commands["sweep"], ["--help"])  # type: ignore
        self.assertEqual(result.exit_code, 0, msg=result.output)
        out = result.output
        self.assertIn("--fusion", out)
        self.assertIn("--cand-mult", out)
        self.assertIn("--rrf-k", out)


if __name__ == "__main__":
    unittest.main()

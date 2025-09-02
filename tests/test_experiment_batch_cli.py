import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestExperimentBatchCLI(unittest.TestCase):
    def setUp(self):
        # Ensure we run from repo root
        self.repo_root = Path(__file__).resolve().parents[1]
        os.chdir(self.repo_root)

        # Import main after ensuring cwd
        import importlib
        self.main = importlib.import_module('main')

    def test_batch_dry_run_jsonl(self):
        from click.testing import CliRunner
        runner = CliRunner()

        queries_path = Path('test_data/sample_batch_queries.jsonl')
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'out.jsonl'
            result = runner.invoke(
                self.main.cli,
                [
                    'experiment', 'batch',
                    '--queries', str(queries_path),
                    '--profile', 'fast',
                    '--collection', 'default',
                    '--k', '2',
                    '--dry-run',
                    '--output', str(out_path)
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            # File should be created and contain as many lines as queries
            content = out_path.read_text().strip().splitlines()
            self.assertGreaterEqual(len(content), 1)
            # Validate basic JSONL structure
            rec = json.loads(content[0])
            self.assertIn('query', rec)
            self.assertIn('answer', rec)
            self.assertIn('sources', rec)

    def test_batch_empty_queries_message(self):
        from click.testing import CliRunner
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / 'empty.jsonl'
            empty_file.write_text('\n')
            out_path = Path(tmpdir) / 'out.jsonl'
            result = runner.invoke(
                self.main.cli,
                [
                    'experiment', 'batch',
                    '--queries', str(empty_file),
                    '--dry-run',
                    '--output', str(out_path)
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn('No queries found in', result.output)

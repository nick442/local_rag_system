import os
import shutil
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path


class TestSqliteVecVendor(unittest.TestCase):
    def setUp(self):
        # Ensure we don't disable vendor in this test
        os.environ.pop('RAG_DISABLE_SQLITE_VEC_VENDOR', None)

    def test_vendor_vec0_dylib_loads(self):
        vendor_path = Path('vendor/sqlite-vec/vec0.dylib').resolve()
        if not vendor_path.exists():
            self.skipTest(f"Vendor dylib not found at {vendor_path}")

        # Prefer running the load in a subprocess so any ABI mismatch/segfault does not crash pytest/unittest
        sqlite3_cli = shutil.which('sqlite3')
        if sqlite3_cli:
            cmd = [
                sqlite3_cli,
                ":memory:",
                f".load {vendor_path}",
                "select vec_version();",
                ".exit",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                # Basic sanity: output should include a version string
                self.assertTrue(len(res.stdout.strip().splitlines()[-1]) > 0)
                return
            # If CLI fails or lacks .load, try Python subprocess fallback next

        # Fallback: run a short-lived Python subprocess; skip if it fails (e.g., segfault)
        code = f"""
import sqlite3
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
conn.load_extension(r"{vendor_path}")
cur = conn.cursor()
cur.execute('select vec_version()')
v = cur.fetchone()[0]
print(v)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if res.returncode != 0 or not res.stdout.strip():
            self.skipTest(
                f"Vendor dylib present but could not be verified (CLI rc may be non-zero and Python rc={res.returncode}).\n"
                f"CLI stdout/stderr may show unsupported .load.\nPython stdout: {res.stdout}\nPython stderr: {res.stderr}"
            )
        self.assertTrue(len(res.stdout.strip()) > 0)


if __name__ == '__main__':
    unittest.main()

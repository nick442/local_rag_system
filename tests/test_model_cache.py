import threading
import unittest
from pathlib import Path

from src.model_cache import ModelCache


class TestModelCache(unittest.TestCase):
    def setUp(self):
        self.cache = ModelCache(max_items=3, log_interval=0)

    def test_safe_path_resolution_nonexistent(self):
        key = self.cache.build_model_key("/path/that/does/not/exist/llm.gguf", device="CUDA", params={
            "temperature": 0.7,
            "max_tokens": 100,
            "unused_param": 123,
        })
        # Should build a hashable tuple even if the file does not exist
        self.assertIsInstance(key, tuple)
        # Device should be normalized lower-case
        self.assertEqual(key[1], "cuda")
        # Only configured params included
        filtered = dict(key[2])
        self.assertIn("temperature", filtered)
        self.assertIn("max_tokens", filtered)
        self.assertNotIn("unused_param", filtered)

    def test_put_get_and_stats(self):
        k1 = ("/tmp/a", "cpu", tuple())
        self.assertIsNone(self.cache.get(k1))
        self.cache.put(k1, 123)
        self.assertEqual(self.cache.get(k1), 123)
        stats = self.cache.get_statistics()
        self.assertEqual(stats["size"], 1)
        self.assertGreaterEqual(stats["hits"], 1)
        self.assertGreaterEqual(stats["misses"], 1)

    def test_lru_eviction(self):
        # Fill cache beyond capacity to trigger LRU eviction
        self.cache.put(("k1", "cpu", tuple()), 1)
        self.cache.put(("k2", "cpu", tuple()), 2)
        self.cache.put(("k3", "cpu", tuple()), 3)
        # Access k1 to make it MRU, then add k4 -> evict k2
        _ = self.cache.get(("k1", "cpu", tuple()))
        self.cache.put(("k4", "cpu", tuple()), 4)

        self.assertIsNotNone(self.cache.get(("k1", "cpu", tuple())))
        self.assertIsNone(self.cache.get(("k2", "cpu", tuple())))
        self.assertIsNotNone(self.cache.get(("k3", "cpu", tuple())))
        self.assertIsNotNone(self.cache.get(("k4", "cpu", tuple())))

    def test_evict_specific_key(self):
        k = ("k1", "cpu", tuple())
        self.cache.put(k, 42)
        removed = self.cache.evict(k)
        self.assertTrue(removed)
        self.assertIsNone(self.cache.get(k))
        # Evict again should be False
        self.assertFalse(self.cache.evict(k))

    def test_clear_cache(self):
        self.cache.put(("k1", "cpu", tuple()), 1)
        self.cache.put(("k2", "cpu", tuple()), 2)
        cleared = self.cache.clear_cache()
        self.assertGreaterEqual(cleared, 2)
        self.assertEqual(self.cache.get_statistics()["size"], 0)

    def test_thread_safety_under_contention(self):
        cache = ModelCache(max_items=50, log_interval=0)
        key = ("shared", "cpu", tuple())

        def writer():
            for i in range(500):
                cache.put(key, i)

        def reader():
            for _ in range(500):
                _ = cache.get(key)

        threads = ([threading.Thread(target=writer) for _ in range(3)] + 
                  [threading.Thread(target=reader) for _ in range(3)])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash; final value should be an int
        self.assertIsInstance(cache.get(key), int)


if __name__ == "__main__":
    unittest.main()


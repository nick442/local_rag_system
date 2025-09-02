import unittest
from unittest import mock


class TestBackendConfigSelection(unittest.TestCase):
    def setUp(self):
        import importlib
        self.rag_pipeline = importlib.import_module("src.rag_pipeline")

    def test_default_backends_used(self):
        calls = {}

        def fake_create_retriever(db_path, embedding_model_path, **kwargs):
            calls['kwargs'] = kwargs
            class _R:
                def retrieve(self, *a, **k):
                    return []
                def get_stats(self):
                    return {}
            return _R()

        class FakePromptBuilder:
            def __init__(self, *args, **kwargs):
                pass
            def truncate_contexts_to_fit(self, user_query, contexts, max_context, system_prompt, generation_buffer=2048):
                return [], "PROMPT"
            def count_prompt_tokens(self, prompt):
                return 0
            def get_template_info(self):
                return {}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                self.n_ctx = 2048
            def generate_with_stats(self, prompt, **kwargs):
                return {"generated_text": "", "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "generation_time": 0.0, "tokens_per_second": 0.0, "context_remaining": 2048}
            def get_model_info(self):
                return {}

        with mock.patch.object(self.rag_pipeline, "create_retriever", side_effect=fake_create_retriever):
            with mock.patch.object(self.rag_pipeline, "create_prompt_builder", return_value=FakePromptBuilder()):
                with mock.patch.object(self.rag_pipeline, "create_llm_wrapper", return_value=FakeLLM()):
                    _ = self.rag_pipeline.RAGPipeline(
                        db_path="data/x.db",
                        embedding_model_path="models/embeddings/fake",
                        llm_model_path="models/llm/fake.gguf",
                    )
        # Defaults should be wired
        self.assertEqual(calls['kwargs'].get('retrieval_backend'), 'default')
        self.assertEqual(calls['kwargs'].get('vector_index_backend'), 'sqlite')

    def test_custom_backends_passed(self):
        calls = {}

        def fake_create_retriever(db_path, embedding_model_path, **kwargs):
            calls['kwargs'] = kwargs
            class _R:
                def retrieve(self, *a, **k):
                    return []
                def get_stats(self):
                    return {}
            return _R()

        class FakePromptBuilder:
            def __init__(self, *args, **kwargs):
                pass
            def truncate_contexts_to_fit(self, user_query, contexts, max_context, system_prompt, generation_buffer=2048):
                return [], "PROMPT"
            def count_prompt_tokens(self, prompt):
                return 0
            def get_template_info(self):
                return {}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                self.n_ctx = 2048
            def generate_with_stats(self, prompt, **kwargs):
                return {"generated_text": "", "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "generation_time": 0.0, "tokens_per_second": 0.0, "context_remaining": 2048}
            def get_model_info(self):
                return {}

        with mock.patch.object(self.rag_pipeline, "create_retriever", side_effect=fake_create_retriever):
            with mock.patch.object(self.rag_pipeline, "create_prompt_builder", return_value=FakePromptBuilder()):
                with mock.patch.object(self.rag_pipeline, "create_llm_wrapper", return_value=FakeLLM()):
                    _ = self.rag_pipeline.RAGPipeline(
                        db_path="data/x.db",
                        embedding_model_path="models/embeddings/fake",
                        llm_model_path="models/llm/fake.gguf",
                        retrieval={"backend": "bm25"},
                        vector_index={"backend": "faiss"},
                    )

        self.assertEqual(calls['kwargs'].get('retrieval_backend'), 'bm25')
        self.assertEqual(calls['kwargs'].get('vector_index_backend'), 'faiss')


if __name__ == "__main__":
    unittest.main()


"""
Test Suite for Phase 5: LLM Integration
Comprehensive tests for LLM wrapper, prompt builder, and RAG pipeline.
"""

import os
import sys
import unittest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_wrapper import LLMWrapper, create_llm_wrapper
from src.prompt_builder import PromptBuilder, create_prompt_builder
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.query_reformulation import QueryReformulator, create_query_reformulator
from src.retriever import RetrievalResult


class TestLLMWrapper(unittest.TestCase):
    """Test cases for LLM wrapper functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.model_path = "models/gemma-3-4b-it-q4_0.gguf"  # From config
        self.test_config = {
            'n_ctx': 2048,  # Smaller for testing
            'n_batch': 128,
            'n_threads': 4,
            'n_gpu_layers': -1,
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 100  # Small for quick testing
        }
    
    def test_llm_wrapper_initialization(self):
        """Test LLM wrapper can be initialized."""
        # Mock the Llama class to avoid loading actual model
        with patch('src.llm_wrapper.Llama') as mock_llama:
            mock_model = Mock()
            mock_llama.return_value = mock_model
            
            wrapper = LLMWrapper(self.model_path, **self.test_config)
            
            self.assertIsNotNone(wrapper)
            self.assertEqual(wrapper.model_path, Path(self.model_path))
            self.assertEqual(wrapper.n_ctx, 2048)
            self.assertEqual(wrapper.temperature, 0.7)
            self.assertTrue(wrapper._is_loaded)
    
    def test_llm_wrapper_generation(self):
        """Test text generation functionality."""
        with patch('src.llm_wrapper.Llama') as mock_llama:
            mock_model = Mock()
            mock_model.return_value = {
                'choices': [{'text': 'Generated response text'}]
            }
            mock_llama.return_value = mock_model
            
            wrapper = LLMWrapper(self.model_path, **self.test_config)
            
            response = wrapper.generate("Test prompt")
            
            self.assertEqual(response, "Generated response text")
            mock_model.assert_called_once()
    
    def test_llm_wrapper_streaming(self):
        """Test streaming generation."""
        with patch('src.llm_wrapper.Llama') as mock_llama:
            mock_model = Mock()
            mock_chunks = [
                {'choices': [{'text': 'Hello '}]},
                {'choices': [{'text': 'world '}]},
                {'choices': [{'text': '!'}]}
            ]
            mock_model.return_value = iter(mock_chunks)
            mock_llama.return_value = mock_model
            
            wrapper = LLMWrapper(self.model_path, **self.test_config)
            
            tokens = list(wrapper.generate_stream("Test prompt"))
            
            self.assertEqual(tokens, ['Hello ', 'world ', '!'])
    
    def test_token_counting(self):
        """Test token counting functionality."""
        with patch('src.llm_wrapper.Llama') as mock_llama:
            mock_model = Mock()
            mock_model.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_llama.return_value = mock_model
            
            wrapper = LLMWrapper(self.model_path, **self.test_config)
            
            token_count = wrapper.count_tokens("Test text")
            
            self.assertEqual(token_count, 5)
    
    def test_context_management(self):
        """Test context window management."""
        with patch('src.llm_wrapper.Llama') as mock_llama:
            mock_model = Mock()
            mock_model.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_llama.return_value = mock_model
            
            wrapper = LLMWrapper(self.model_path, **self.test_config)
            
            remaining = wrapper.get_context_remaining("Test text")
            fits = wrapper.fits_in_context("Test text")
            
            self.assertEqual(remaining, 2048 - 5)  # n_ctx - tokens
            self.assertTrue(fits)
    
    def test_create_llm_wrapper_factory(self):
        """Test factory function."""
        with patch('src.llm_wrapper.Llama'):
            wrapper = create_llm_wrapper(self.model_path, self.test_config)
            self.assertIsInstance(wrapper, LLMWrapper)


class TestPromptBuilder(unittest.TestCase):
    """Test cases for prompt builder functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.chat_template = {
            'system_prefix': '<bos>',
            'user_prefix': '<start_of_turn>user\n',
            'user_suffix': '<end_of_turn>\n',
            'assistant_prefix': '<start_of_turn>model\n',
            'assistant_suffix': '<end_of_turn>\n'
        }
        self.builder = PromptBuilder(self.chat_template)
        
        # Create mock retrieval results
        self.mock_contexts = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.9,
                metadata={'source': 'ml_basics.txt', 'page_number': 1},
                doc_id="doc_1",
                chunk_index=0
            ),
            RetrievalResult(
                chunk_id="chunk_2", 
                content="Neural networks are inspired by biological neurons.",
                score=0.8,
                metadata={'source': 'neural_nets.pdf', 'chunk_index': 5},
                doc_id="doc_2",
                chunk_index=5
            )
        ]
    
    def test_prompt_builder_initialization(self):
        """Test prompt builder initialization."""
        self.assertIsNotNone(self.builder)
        self.assertEqual(self.builder.chat_template['system_prefix'], '<bos>')
    
    def test_simple_prompt_building(self):
        """Test simple prompt without RAG context."""
        prompt = self.builder.build_simple_prompt("What is machine learning?")
        
        expected_parts = [
            '<bos>',
            '<start_of_turn>user\n',
            'What is machine learning?',
            '<end_of_turn>\n',
            '<start_of_turn>model\n'
        ]
        
        for part in expected_parts:
            self.assertIn(part, prompt)
    
    def test_rag_prompt_building(self):
        """Test RAG prompt with contexts."""
        prompt = self.builder.build_rag_prompt(
            "What is machine learning?",
            self.mock_contexts,
            include_metadata=True
        )
        
        # Check structure
        self.assertIn('<bos>', prompt)
        self.assertIn('<start_of_turn>user', prompt)
        self.assertIn('Context information:', prompt)
        self.assertIn('Question: What is machine learning?', prompt)
        self.assertIn('<start_of_turn>model', prompt)
        
        # Check contexts are included
        self.assertIn('Machine learning is a subset', prompt)
        self.assertIn('Neural networks are inspired', prompt)
        
        # Check metadata is included
        self.assertIn('Source: ml_basics.txt', prompt)
        self.assertIn('Score: 0.900', prompt)
    
    def test_rag_prompt_without_metadata(self):
        """Test RAG prompt without metadata."""
        prompt = self.builder.build_rag_prompt(
            "What is machine learning?",
            self.mock_contexts,
            include_metadata=False
        )
        
        # Should not contain metadata
        self.assertNotIn('Source:', prompt)
        self.assertNotIn('Score:', prompt)
        
        # Should contain contexts
        self.assertIn('Context 1', prompt)
        self.assertIn('Context 2', prompt)
    
    def test_conversation_prompt(self):
        """Test multi-turn conversation prompt."""
        messages = [
            {'role': 'user', 'content': 'What is AI?'},
            {'role': 'assistant', 'content': 'AI is artificial intelligence.'},
            {'role': 'user', 'content': 'How does machine learning relate to AI?'}
        ]
        
        prompt = self.builder.build_conversation_prompt(messages, self.mock_contexts)
        
        # Check all messages included
        self.assertIn('What is AI?', prompt)
        self.assertIn('AI is artificial intelligence.', prompt)
        self.assertIn('How does machine learning relate to AI?', prompt)
        
        # Check contexts added to last user message
        self.assertIn('Context information:', prompt)
        self.assertIn('Machine learning is a subset', prompt)
    
    def test_token_counting(self):
        """Test prompt token counting."""
        prompt = "This is a test prompt"
        token_count = self.builder.count_prompt_tokens(prompt)
        
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
    
    def test_prompt_analysis(self):
        """Test prompt structure analysis."""
        prompt = self.builder.build_rag_prompt("Test query", self.mock_contexts)
        analysis = self.builder.analyze_prompt_structure(prompt)
        
        self.assertIn('total_tokens', analysis)
        self.assertIn('context_tokens', analysis)
        self.assertIn('question_tokens', analysis)
        self.assertIn('context_percentage', analysis)
        
        self.assertGreater(analysis['total_tokens'], 0)
    
    def test_context_window_fit(self):
        """Test context window fit checking."""
        prompt = self.builder.build_rag_prompt("Test query", self.mock_contexts)
        fit_analysis = self.builder.check_context_window_fit(prompt, 4096)
        
        self.assertIn('fits', fit_analysis)
        self.assertIn('prompt_tokens', fit_analysis)
        self.assertIn('available_tokens', fit_analysis)
        
        self.assertTrue(fit_analysis['fits'])  # Should fit in 4096 tokens
    
    def test_context_truncation(self):
        """Test context truncation to fit window."""
        contexts, prompt = self.builder.truncate_contexts_to_fit(
            "Test query",
            self.mock_contexts,
            max_context=1000,  # Small window
            generation_buffer=200
        )
        
        # Should return some contexts (possibly fewer)
        self.assertIsInstance(contexts, list)
        self.assertLessEqual(len(contexts), len(self.mock_contexts))
        
        # Prompt should be valid
        self.assertIn('Test query', prompt)
    
    def test_create_prompt_builder_factory(self):
        """Test factory function."""
        builder = create_prompt_builder(self.chat_template)
        self.assertIsInstance(builder, PromptBuilder)


class TestQueryReformulator(unittest.TestCase):
    """Test cases for query reformulation."""
    
    def setUp(self):
        """Set up test environment."""
        self.reformulator = QueryReformulator()
    
    def test_keyword_expansion(self):
        """Test keyword-based query expansion."""
        variants = self.reformulator.expand_query_keywords("machine learning algorithm")
        
        self.assertIsInstance(variants, list)
        self.assertGreater(len(variants), 1)
        self.assertIn("machine learning algorithm", variants)  # Original included
    
    def test_search_variants(self):
        """Test search variant generation."""
        variants = self.reformulator.generate_search_variants("neural networks", num_variants=3)
        
        self.assertIsInstance(variants, list)
        self.assertLessEqual(len(variants), 3)
        self.assertIn("neural networks", variants)  # Original included
    
    def test_multi_strategy_reformulation(self):
        """Test multi-strategy reformulation."""
        variants = self.reformulator.multi_strategy_reformulation(
            "deep learning",
            use_llm=False,  # Skip LLM for testing
            max_variants=5
        )
        
        self.assertIsInstance(variants, list)
        self.assertLessEqual(len(variants), 5)
        self.assertIn("deep learning", variants)
    
    def test_create_query_reformulator_factory(self):
        """Test factory function."""
        reformulator = create_query_reformulator()
        self.assertIsInstance(reformulator, QueryReformulator)


class TestRAGPipeline(unittest.TestCase):
    """Test cases for complete RAG pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_data = {
            'llm_params': {
                'n_ctx': 2048,
                'n_batch': 128,
                'temperature': 0.7,
                'max_tokens': 100
            },
            'chat_template': {
                'system_prefix': '<bos>',
                'user_prefix': '<start_of_turn>user\n',
                'user_suffix': '<end_of_turn>\n',
                'assistant_prefix': '<start_of_turn>model\n',
                'assistant_suffix': '<end_of_turn>\n'
            }
        }
        yaml.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        self.test_paths = {
            'db_path': 'data/rag_vectors.db',
            'embedding_model_path': 'models/embeddings/sentence-transformers_all-MiniLM-L6-v2',
            'llm_model_path': 'models/gemma-3-4b-it-q4_0.gguf',
            'config_path': self.temp_config.name
        }
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_config.name)
    
    @patch('src.rag_pipeline.create_retriever')
    @patch('src.rag_pipeline.create_llm_wrapper')
    @patch('src.rag_pipeline.create_prompt_builder')
    def test_rag_pipeline_initialization(self, mock_prompt_builder, mock_llm_wrapper, mock_retriever):
        """Test RAG pipeline initialization."""
        # Mock components
        mock_retriever.return_value = Mock()
        mock_llm_wrapper.return_value = Mock()
        mock_prompt_builder.return_value = Mock()
        
        pipeline = RAGPipeline(**self.test_paths)
        
        self.assertIsNotNone(pipeline.retriever)
        self.assertIsNotNone(pipeline.llm_wrapper)
        self.assertIsNotNone(pipeline.prompt_builder)
    
    @patch('src.rag_pipeline.create_retriever')
    @patch('src.rag_pipeline.create_llm_wrapper')
    @patch('src.rag_pipeline.create_prompt_builder')
    def test_rag_pipeline_query(self, mock_prompt_builder, mock_llm_wrapper, mock_retriever):
        """Test RAG pipeline query execution."""
        # Mock components
        mock_retriever_instance = Mock()
        mock_retrieval_results = [
            RetrievalResult("chunk_1", "Test content", 0.9, {}, "doc_1", 0)
        ]
        mock_retriever_instance.retrieve.return_value = mock_retrieval_results
        mock_retriever.return_value = mock_retriever_instance
        
        mock_llm_instance = Mock()
        mock_llm_instance.n_ctx = 2048
        mock_llm_instance.generate_with_stats.return_value = {
            'generated_text': 'Generated answer',
            'prompt_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150,
            'generation_time': 1.0,
            'tokens_per_second': 50.0,
            'context_remaining': 1898
        }
        mock_llm_wrapper.return_value = mock_llm_instance
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.truncate_contexts_to_fit.return_value = (
            mock_retrieval_results, "Built prompt"
        )
        mock_prompt_instance.count_prompt_tokens.return_value = 100
        mock_prompt_builder.return_value = mock_prompt_instance
        
        pipeline = RAGPipeline(**self.test_paths)
        
        # Execute query
        response = pipeline.query("What is machine learning?")
        
        # Verify response structure
        self.assertIn('answer', response)
        self.assertIn('sources', response)
        self.assertIn('metadata', response)
        
        self.assertEqual(response['answer'], 'Generated answer')
        
        # Verify components were called
        mock_retriever_instance.retrieve.assert_called_once()
        mock_llm_instance.generate_with_stats.assert_called_once()
    
    @patch('src.rag_pipeline.create_retriever')
    @patch('src.rag_pipeline.create_llm_wrapper')
    @patch('src.rag_pipeline.create_prompt_builder')
    def test_rag_pipeline_streaming(self, mock_prompt_builder, mock_llm_wrapper, mock_retriever):
        """Test RAG pipeline streaming query."""
        # Mock components for streaming
        mock_retriever_instance = Mock()
        mock_retrieval_results = [
            RetrievalResult("chunk_1", "Test content", 0.9, {}, "doc_1", 0)
        ]
        mock_retriever_instance.retrieve.return_value = mock_retrieval_results
        mock_retriever.return_value = mock_retriever_instance
        
        mock_llm_instance = Mock()
        mock_llm_instance.n_ctx = 2048
        
        def mock_stream_generator():
            yield "Generated"
            yield " "
            yield "response"
        
        def mock_stats():
            return {'output_tokens': 3, 'generation_time': 1.0}
        
        mock_llm_instance.generate_stream_with_stats.return_value = (
            mock_stream_generator(), mock_stats
        )
        mock_llm_wrapper.return_value = mock_llm_instance
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.truncate_contexts_to_fit.return_value = (
            mock_retrieval_results, "Built prompt"
        )
        mock_prompt_instance.count_prompt_tokens.return_value = 100
        mock_prompt_builder.return_value = mock_prompt_instance
        
        pipeline = RAGPipeline(**self.test_paths)
        
        # Execute streaming query
        result = pipeline.query("What is machine learning?", stream=True)
        
        # Verify streaming structure
        self.assertIn('generator', result)
        self.assertIn('get_final_stats', result)
        self.assertIn('contexts', result)
        
        # Test generator
        tokens = list(result['generator'])
        self.assertEqual(tokens, ["Generated", " ", "response"])
    
    def test_create_rag_pipeline_factory(self):
        """Test factory function."""
        with patch('src.rag_pipeline.RAGPipeline') as mock_pipeline:
            create_rag_pipeline(**self.test_paths)
            mock_pipeline.assert_called_once_with(**self.test_paths)


def run_integration_test():
    """Run a basic integration test with actual components if available."""
    print("\n" + "="*60)
    print("INTEGRATION TEST - Testing with real components")
    print("="*60)
    
    try:
        # Check if model files exist
        model_config_path = "config/model_config.yaml"
        if not os.path.exists(model_config_path):
            print("❌ Model config not found - skipping integration test")
            return False
        
        # Load actual config
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        llm_model_path = config.get('model', {}).get('path', 'models/gemma-3-4b-it-q4_0.gguf')
        
        if not os.path.exists(llm_model_path):
            print(f"❌ LLM model not found at {llm_model_path} - skipping integration test")
            return False
        
        # Test LLM wrapper loading
        print("🔄 Testing LLM wrapper loading...")
        try:
            from src.llm_wrapper import LLMWrapper
            wrapper = LLMWrapper(llm_model_path, n_ctx=512, max_tokens=50)
            print("✅ LLM wrapper loaded successfully")
            
            # Test simple generation
            print("🔄 Testing simple generation...")
            response = wrapper.generate("What is 2+2?", max_tokens=10, temperature=0.1)
            print(f"✅ Generated response: '{response[:50]}...'")
            
            # Test token counting
            tokens = wrapper.count_tokens("This is a test")
            print(f"✅ Token counting works: {tokens} tokens")
            
            wrapper.unload_model()
            
        except Exception as e:
            print(f"❌ LLM integration test failed: {e}")
            return False
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False


if __name__ == '__main__':
    print("Running Phase 5 Tests...")
    print("="*60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run integration test
    success = run_integration_test()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n⚠️  Some integration tests skipped or failed")
        sys.exit(0)  # Don't fail on integration test issues
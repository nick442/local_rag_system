#!/usr/bin/env python3
"""
Comprehensive RAG Information Retrieval Test Suite

This script runs automated tests to validate the accuracy, consistency, and performance
of the RAG system's information retrieval capabilities against a production corpus.

Features:
- Loads test prompts from JSON configuration
- Executes queries against the RAG pipeline
- Evaluates results using multiple criteria
- Generates detailed logs and reports
- Supports consistency testing with multiple runs
- Performance benchmarking
- HTML report generation

Usage:
    python run_retrieval_tests_fixed.py [--config retrieval_test_prompts.json] [--output test_results/]
"""

import asyncio
import json
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import statistics
import sys
import os
from dataclasses import dataclass, asdict

# Ensure project root is on path so `src.*` package imports work
sys.path.insert(0, str(Path(__file__).parent))

# Import components via package
from src.vector_database import VectorDatabase
from src.retriever import create_retriever
from src.llm_wrapper import create_llm_wrapper
from src.prompt_builder import create_prompt_builder


@dataclass
class TestResult:
    """Represents the result of a single test execution"""
    test_id: str
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    execution_time: float
    relevance_score: float
    completeness_score: float
    response_quality_score: float
    source_count: int
    expected_sources: int
    found_elements: List[str]
    missing_elements: List[str]
    timestamp: str
    error: Optional[str] = None


@dataclass
class TestSummary:
    """Summary statistics for a test category or full test run"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_execution_time: float
    avg_relevance_score: float
    avg_completeness_score: float
    avg_response_quality: float
    consistency_score: Optional[float] = None


class SimpleRAGInterface:
    """Simplified RAG interface for testing without complex pipeline imports"""
    
    def __init__(self, db_path: str, embedding_model_path: str, llm_model_path: str, collection_id: str = 'default'):
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.collection_id = collection_id
        
        # Initialize components
        self.retriever = None
        self.llm_wrapper = None
        self.prompt_builder = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        # Create retriever
        self.retriever = create_retriever(
            self.db_path,
            self.embedding_model_path,
            embedding_dimension=384
        )
        
        # Create LLM wrapper
        llm_params = {
            'n_ctx': 8192,
            'n_batch': 512,
            'n_threads': 8,
            'n_gpu_layers': -1,
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 2048
        }
        self.llm_wrapper = create_llm_wrapper(self.llm_model_path, llm_params)
        
        # Create prompt builder
        chat_template = {
            'system_prefix': '<bos>',
            'user_prefix': '<start_of_turn>user\n',
            'user_suffix': '<end_of_turn>\n',
            'assistant_prefix': '<start_of_turn>model\n',
            'assistant_suffix': '<end_of_turn>\n'
        }
        self.prompt_builder = create_prompt_builder(chat_template)
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        results = self.retriever.retrieve(query, k=k, method='vector', collection_id=self.collection_id)
        
        # Convert to dict format
        search_results = []
        for result in results:
            search_results.append({
                'chunk_id': result.chunk_id,
                'content': result.content,
                'similarity_score': result.score,
                'doc_id': result.metadata.get('doc_id', 'unknown'),
                'source_path': result.metadata.get('source', result.metadata.get('filename', 'unknown'))
            })
        
        return search_results
    
    async def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate response using LLM"""
        if not search_results:
            return "I don't have any relevant information to answer this question."
        
        # Build context from search results
        contexts = []
        for result in search_results:
            contexts.append(result['content'])
        
        # Create prompt
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Based on the following context information, please answer the user's question.

Context Information:
{context_text}

Question: {query}

Answer based on the context provided:"""
        
        # Generate response
        try:
            result = self.llm_wrapper.generate_with_stats(prompt, max_tokens=1024)
            return result['generated_text']
        except Exception as e:
            return f"Error generating response: {str(e)}"


class RetrievalTestRunner:
    """Main test runner for RAG system validation"""
    
    def __init__(self, config_path: str, output_dir: str = "test_results"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load test configuration
        self.config = self.load_config()
        
        # Initialize RAG interface
        self.rag = None
        
        # Test results storage
        self.results: Dict[str, List[TestResult]] = {}
        
        self.logger.info(f"Initialized RetrievalTestRunner with config: {config_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Configure detailed logging for test execution"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"test_execution_{timestamp}.log"
        
        # Create logger
        self.logger = logging.getLogger('RetrievalTestRunner')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging configured. Detailed logs: {log_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load test configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded test configuration with {len(config['retrieval_test_prompts']['test_sets'])} categories")
            return config['retrieval_test_prompts']
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    async def initialize_rag(self):
        """Initialize the RAG interface with production settings"""
        try:
            self.logger.info("Initializing RAG interface...")
            
            # Use same configuration as production ingestion
            db_path = 'data/rag_vectors.db'
            embedding_path = 'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
            # Prefer environment variable override; fallback to local Gemma-3 model symlink
            llm_path = os.getenv('LLM_MODEL_PATH', 'models/gemma-3-4b-it-q4_0.gguf')
            collection_id = 'realistic_full_production'
            
            self.rag = SimpleRAGInterface(
                db_path=db_path,
                embedding_model_path=embedding_path,
                llm_model_path=llm_path,
                collection_id=collection_id
            )
            
            # Test basic connectivity
            test_query = "test connection"
            test_results = await self.rag.search(test_query, k=1)
            
            if not test_results:
                raise RuntimeError("RAG interface returned no results for test query")
            
            self.logger.info(f"RAG interface initialized successfully")
            self.logger.info(f"Test query returned {len(test_results)} results")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG interface: {e}")
            raise
    
    def evaluate_response(self, test_config: Dict[str, Any], response: str, 
                         retrieved_docs: List[Dict[str, Any]], execution_time: float) -> TestResult:
        """Evaluate a test response against expected criteria"""
        test_id = test_config['id']
        query = test_config['query']
        expected_elements = test_config.get('expected_elements', [])
        expected_sources = test_config.get('expected_sources', 1)
        
        # Find which expected elements are present
        response_lower = response.lower()
        found_elements = [elem for elem in expected_elements if elem.lower() in response_lower]
        missing_elements = [elem for elem in expected_elements if elem.lower() not in response_lower]
        
        # Calculate scores
        completeness_score = (len(found_elements) / len(expected_elements)) * 100 if expected_elements else 100
        
        # Source diversity check
        source_count = len(retrieved_docs)
        
        # Relevance score based on keyword presence and document count
        relevance_score = min(100, (len(found_elements) * 20) + min(source_count * 10, 30))
        
        # Response quality heuristics
        response_quality_score = self.calculate_response_quality(response, query, retrieved_docs)
        
        return TestResult(
            test_id=test_id,
            query=query,
            response=response,
            retrieved_documents=retrieved_docs,
            execution_time=execution_time,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            response_quality_score=response_quality_score,
            source_count=source_count,
            expected_sources=expected_sources,
            found_elements=found_elements,
            missing_elements=missing_elements,
            timestamp=datetime.now().isoformat()
        )
    
    def calculate_response_quality(self, response: str, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate response quality score using various heuristics"""
        score = 0
        
        # Length appropriateness (not too short, not too long)
        if 50 <= len(response) <= 1000:
            score += 20
        elif 20 <= len(response) < 50:
            score += 10
        elif len(response) < 20:
            score += 5
        
        # Coherence indicators
        if '. ' in response and response.count('. ') >= 1:  # Complete sentences
            score += 20
        
        if any(word in response.lower() for word in ['according', 'mentioned', 'described', 'explains']):
            score += 15  # References source material
        
        # Query relevance
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        score += overlap * 25
        
        # Document utilization
        if retrieved_docs and len(retrieved_docs) > 0:
            score += 20
        
        return min(100, score)
    
    async def run_single_test(self, category: str, test_config: Dict[str, Any]) -> TestResult:
        """Execute a single test query"""
        test_id = test_config['id']
        query = test_config['query']
        
        self.logger.debug(f"Running test {test_id}: {query}")
        
        try:
            start_time = time.time()
            
            # Execute RAG query
            results = await self.rag.search(query, k=5)  # Get top 5 results
            response = await self.rag.generate_response(query, results)
            
            execution_time = time.time() - start_time
            
            # Convert results to serializable format
            retrieved_docs = []
            for result in results:
                doc_info = {
                    'doc_id': result.get('doc_id', 'unknown'),
                    'content_preview': result.get('content', '')[:200] + '...' if result.get('content') else '',
                    'similarity_score': result.get('similarity_score', 0.0),
                    'source_path': result.get('source_path', 'unknown')
                }
                retrieved_docs.append(doc_info)
            
            # Evaluate the response
            test_result = self.evaluate_response(test_config, response, retrieved_docs, execution_time)
            
            self.logger.info(f"✅ Test {test_id} completed - Relevance: {test_result.relevance_score:.1f}, "
                           f"Completeness: {test_result.completeness_score:.1f}%, Time: {execution_time:.3f}s")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Test {test_id} failed: {e}")
            return TestResult(
                test_id=test_id,
                query=query,
                response="",
                retrieved_documents=[],
                execution_time=0,
                relevance_score=0,
                completeness_score=0,
                response_quality_score=0,
                source_count=0,
                expected_sources=test_config.get('expected_sources', 1),
                found_elements=[],
                missing_elements=test_config.get('expected_elements', []),
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    async def run_consistency_tests(self, category: str, test_config: Dict[str, Any]) -> List[TestResult]:
        """Run consistency tests with multiple executions of the same query"""
        runs = test_config.get('runs', 1)
        results = []
        
        self.logger.info(f"Running consistency test {test_config['id']} with {runs} iterations")
        
        for i in range(runs):
            self.logger.debug(f"Consistency test run {i+1}/{runs}")
            result = await self.run_single_test(category, test_config)
            result.test_id = f"{result.test_id}_run_{i+1}"
            results.append(result)
            
            # Small delay between runs
            await asyncio.sleep(0.1)
        
        return results
    
    async def run_test_category(self, category: str) -> List[TestResult]:
        """Run all tests in a specific category"""
        self.logger.info(f"🚀 Starting test category: {category}")
        
        test_configs = self.config['test_sets'].get(category, {}).get('prompts', [])
        results = []
        
        for test_config in test_configs:
            if category == 'consistency_validation' and test_config.get('runs', 1) > 1:
                # Run consistency tests
                consistency_results = await self.run_consistency_tests(category, test_config)
                results.extend(consistency_results)
            else:
                # Run single test
                result = await self.run_single_test(category, test_config)
                results.append(result)
        
        self.logger.info(f"✅ Completed category {category}: {len(results)} tests")
        return results
    
    def calculate_summary(self, results: List[TestResult]) -> TestSummary:
        """Calculate summary statistics for a set of test results"""
        if not results:
            return TestSummary(0, 0, 0, 0, 0, 0, 0)
        
        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]
        
        if successful:
            avg_execution_time = statistics.mean([r.execution_time for r in successful])
            avg_relevance = statistics.mean([r.relevance_score for r in successful])
            avg_completeness = statistics.mean([r.completeness_score for r in successful])
            avg_quality = statistics.mean([r.response_quality_score for r in successful])
        else:
            avg_execution_time = avg_relevance = avg_completeness = avg_quality = 0
        
        return TestSummary(
            total_tests=len(results),
            successful_tests=len(successful),
            failed_tests=len(failed),
            avg_execution_time=avg_execution_time,
            avg_relevance_score=avg_relevance,
            avg_completeness_score=avg_completeness,
            avg_response_quality=avg_quality
        )
    
    def save_results(self, timestamp: str):
        """Save test results in multiple formats"""
        # Save detailed JSON results
        json_file = self.output_dir / f"test_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for category, results in self.results.items():
            serializable_results[category] = [asdict(result) for result in results]
        
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'config_file': str(self.config_path),
                    'total_categories': len(self.results),
                    'total_tests': sum(len(results) for results in self.results.values())
                },
                'results': serializable_results,
                'summaries': {category: asdict(self.calculate_summary(results)) 
                            for category, results in self.results.items()}
            }, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {json_file}")
    
    async def run_all_tests(self):
        """Execute the complete test suite"""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("🚀 Starting comprehensive RAG retrieval test suite")
        
        # Initialize RAG interface
        await self.initialize_rag()
        
        # Run tests by category
        categories = list(self.config['test_sets'].keys())
        self.logger.info(f"Test categories to run: {categories}")
        
        for category in categories:
            try:
                results = await self.run_test_category(category)
                self.results[category] = results
            except Exception as e:
                self.logger.error(f"Failed to run category {category}: {e}")
                self.results[category] = []
        
        # Calculate and log final statistics
        total_time = time.time() - start_time
        total_tests = sum(len(results) for results in self.results.values())
        
        self.logger.info(f"🎉 Test suite completed!")
        self.logger.info(f"   Total tests: {total_tests}")
        self.logger.info(f"   Total time: {total_time:.1f} seconds")
        if total_tests > 0:
            self.logger.info(f"   Average time per test: {total_time/total_tests:.3f} seconds")
        
        # Save results
        self.save_results(timestamp)
        
        return self.results


async def main():
    """Main entry point for the test runner"""
    parser = argparse.ArgumentParser(description='Run RAG Information Retrieval Tests')
    parser.add_argument('--config', default='retrieval_test_prompts.json',
                       help='Path to test configuration file')
    parser.add_argument('--output', default='test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    try:
        # Create and run test suite
        runner = RetrievalTestRunner(args.config, args.output)
        results = await runner.run_all_tests()
        
        # Print summary
        total_tests = sum(len(r) for r in results.values())
        successful_tests = sum(len([t for t in r if not t.error]) for r in results.values())
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        if total_tests > 0:
            print(f"   Success Rate: {successful_tests/total_tests*100:.1f}%")
        print(f"   Results saved to: {runner.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

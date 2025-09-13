#!/usr/bin/env python3
"""
Automated experiment runner for chunking optimization experiments.

This module handles the execution of parameter sweeps, data ingestion,
and experiment orchestration with comprehensive logging and error handling.
"""

import json
import time
import asyncio
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import psutil

from experiment_logger import ExperimentLogger, ExperimentMetrics, IngestionMetrics

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    chunk_size: int
    chunk_overlap: int
    collection: str
    query: str
    expected_chunks: int
    run_id: str

class ChunkingExperimentRunner:
    """Automated runner for chunking optimization experiments."""
    
    def __init__(self, base_dir: str = "/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/Opus-Experiments"):
        self.base_dir = Path(base_dir)
        self.logger = ExperimentLogger("chunking_optimization")
        
        # Experiment parameters
        self.chunk_sizes = [128, 256, 512, 768, 1024, 1536, 2048]
        self.overlap_ratios = [0, 64, 128, 192, 256]  # Fixed overlaps in tokens
        
        # System paths
        self.conda_activate = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env"
        self.python_cmd = f"{self.conda_activate} && python"
        self.main_script = self.base_dir / "main.py"
        
        # Results storage
        self.results_dir = self.base_dir / "experiments" / "chunking" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, command: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """Run shell command with timeout and logging."""
        self.logger.log_info(f"Executing command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_dir)
            )
            
            success = result.returncode == 0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if success:
                self.logger.log_info(f"Command succeeded", "command_execution")
            else:
                self.logger.log_warning(f"Command failed with return code {result.returncode}", "command_execution")
                self.logger.log_warning(f"STDERR: {stderr}", "command_execution")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            self.logger.log_error(Exception(f"Command timed out after {timeout}s"), "command_execution")
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            self.logger.log_error(e, "command_execution")
            return False, "", str(e)

    def test_system_health(self) -> bool:
        """Run system health check and return status."""
        self.logger.log_info("Running system health check", "health_check")
        
        success, stdout, stderr = self.run_command(f"{self.python_cmd} {self.main_script} doctor --format json")
        
        if not success:
            self.logger.log_warning("System health check failed", "health_check")
            return False
        
        # Parse health check results (assuming JSON output)
        try:
            # For now, just check if the command ran successfully
            self.logger.log_info("System health check passed", "health_check")
            return True
        except Exception as e:
            self.logger.log_error(e, "health_check_parsing")
            return False

    def prepare_test_corpus(self, collection_name: str, source_file: str, max_docs: int = 100) -> bool:
        """Create small test corpus from larger dataset."""
        self.logger.log_info(f"Preparing test corpus: {collection_name} ({max_docs} docs)", "test_corpus")
        
        source_path = self.base_dir / "corpus" / source_file
        test_dir = self.base_dir / "corpus" / "test" / collection_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Read source corpus and create smaller version
            with open(source_path, 'r') as f:
                lines = f.readlines()
            
            # Take first max_docs documents
            test_lines = lines[:max_docs]
            
            # Write test corpus
            test_file = test_dir / "corpus.jsonl"
            with open(test_file, 'w') as f:
                f.writelines(test_lines)
            
            self.logger.log_info(f"Created test corpus with {len(test_lines)} documents: {test_file}", "test_corpus")
            return True
            
        except Exception as e:
            self.logger.log_error(e, "test_corpus_creation")
            return False

    def ingest_collection(self, collection_name: str, source_path: str, chunk_size: int = 512, 
                         chunk_overlap: int = 128, test_mode: bool = False) -> IngestionMetrics:
        """Ingest documents into collection with specified chunking parameters."""
        
        start_time = datetime.now(timezone.utc)
        ingestion_id = self.logger.log_ingestion_start(collection_name, -1)  # Unknown doc count initially
        
        # Configure chunking parameters
        config_commands = [
            f"{self.python_cmd} {self.main_script} config set chunk_size {chunk_size} --type int",
            f"{self.python_cmd} {self.main_script} config set chunk_overlap {chunk_overlap} --type int"
        ]
        
        for cmd in config_commands:
            success, stdout, stderr = self.run_command(cmd, timeout=30)
            if not success:
                error_msg = f"Failed to set configuration: {stderr}"
                self.logger.log_error(Exception(error_msg), "ingestion_config")
                return self._create_failed_ingestion_metrics(collection_name, start_time, error_msg)
        
        # Create collection
        create_cmd = f"{self.python_cmd} {self.main_script} collection create {collection_name} --description 'Chunking experiment collection'"
        success, stdout, stderr = self.run_command(create_cmd, timeout=60)
        
        if not success and "already exists" not in stderr.lower():
            error_msg = f"Failed to create collection: {stderr}"
            return self._create_failed_ingestion_metrics(collection_name, start_time, error_msg)
        
        # Monitor memory usage during ingestion
        initial_memory = psutil.virtual_memory().used / (1024**3)
        peak_memory = initial_memory
        
        # Ingest documents
        ingest_cmd = f"{self.python_cmd} {self.main_script} ingest directory {source_path} --collection {collection_name} --batch-size 16 --max-workers 2"
        
        self.logger.log_info(f"Starting ingestion: {ingest_cmd}", "ingestion")
        ingestion_start = time.time()
        
        success, stdout, stderr = self.run_command(ingest_cmd, timeout=7200)  # 2 hour timeout
        
        ingestion_end = time.time()
        processing_time = ingestion_end - ingestion_start
        
        # Monitor peak memory
        current_memory = psutil.virtual_memory().used / (1024**3)
        peak_memory = max(peak_memory, current_memory)
        
        end_time = datetime.now(timezone.utc)
        
        if not success:
            error_msg = f"Ingestion failed: {stderr}"
            return self._create_failed_ingestion_metrics(collection_name, start_time, error_msg)
        
        # Parse ingestion statistics from output
        docs_processed = self._extract_stat_from_output(stdout, "files processed", default=0)
        chunks_created = self._extract_stat_from_output(stdout, "chunks created", default=0)
        
        # Create successful metrics
        metrics = IngestionMetrics(
            collection_name=collection_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            documents_processed=docs_processed,
            documents_failed=0,  # Would need to parse from output
            chunks_created=chunks_created,
            total_tokens=chunks_created * chunk_size,  # Estimate
            processing_time_seconds=processing_time,
            average_doc_size_bytes=0.0,  # Would need to calculate
            ingestion_rate_docs_per_second=docs_processed / processing_time if processing_time > 0 else 0,
            peak_memory_usage_gb=peak_memory,
            success=True
        )
        
        self.logger.log_ingestion_complete(metrics)
        return metrics

    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentMetrics:
        """Run a single experiment with given configuration."""
        start_time = datetime.now(timezone.utc)
        initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        # Set chunk parameters
        config_commands = [
            f"{self.python_cmd} {self.main_script} config set chunk_size {config.chunk_size} --type int",
            f"{self.python_cmd} {self.main_script} config set chunk_overlap {config.chunk_overlap} --type int"
        ]
        
        for cmd in config_commands:
            success, _, stderr = self.run_command(cmd, timeout=30)
            if not success:
                return self._create_failed_experiment_metrics(config, start_time, f"Config failed: {stderr}")
        
        # Run query
        query_start = time.time()
        query_cmd = f"{self.python_cmd} {self.main_script} query '{config.query}' --collection {config.collection} --metrics"
        success, stdout, stderr = self.run_command(query_cmd, timeout=300)
        query_end = time.time()
        
        current_memory = psutil.virtual_memory().used / (1024**2)  # MB
        memory_usage = current_memory - initial_memory
        
        if not success:
            return self._create_failed_experiment_metrics(config, start_time, f"Query failed: {stderr}")
        
        # Parse response metrics
        response_time = query_end - query_start
        response_length = len(stdout)
        response_words = len(stdout.split()) if stdout else 0
        
        # Create successful metrics
        metrics = ExperimentMetrics(
            run_id=config.run_id,
            timestamp=start_time.isoformat(),
            query=config.query,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            collection=config.collection,
            response_time_seconds=response_time,
            retrieval_time_seconds=response_time * 0.3,  # Estimate
            generation_time_seconds=response_time * 0.7,  # Estimate
            retrieved_chunks=config.expected_chunks,  # Use expected as placeholder
            response_length_chars=response_length,
            response_length_words=response_words,
            tokens_per_second=response_words / response_time if response_time > 0 else 0,
            memory_usage_mb=memory_usage,
            success=True
        )
        
        return metrics

    def run_test_experiments(self) -> bool:
        """Run small test experiments to validate pipeline."""
        self.logger.log_info("Starting test experiments", "test_phase")
        
        # Test with 3 chunk sizes and 2 queries
        test_chunk_sizes = [256, 512, 1024]
        test_queries = [
            "What are FSA health insurance premiums?",
            "What is the apparent diffusion coefficient?"
        ]
        
        test_configs = []
        for chunk_size in test_chunk_sizes:
            for query in test_queries:
                config = ExperimentConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=128,
                    collection="fiqa_test",
                    query=query,
                    expected_chunks=3,
                    run_id=f"test_{chunk_size}_{len(test_configs)}"
                )
                test_configs.append(config)
        
        self.logger.start_experiment(len(test_configs), "Test experiments to validate pipeline")
        
        success_count = 0
        for config in test_configs:
            try:
                metrics = self.run_single_experiment(config)
                self.logger.log_experiment_result(metrics)
                if metrics.success:
                    success_count += 1
            except Exception as e:
                self.logger.log_error(e, f"test_experiment_{config.run_id}")
        
        success_rate = success_count / len(test_configs) * 100
        self.logger.log_info(f"Test experiments complete: {success_count}/{len(test_configs)} successful ({success_rate:.1f}%)", "test_phase")
        
        return success_rate >= 80.0  # Require 80% success rate

    def run_parameter_sweep(self, collection: str, queries: List[str]) -> List[ExperimentMetrics]:
        """Run complete parameter sweep for chunk sizes and overlap ratios."""
        all_configs = []
        
        # Generate all combinations
        for chunk_size in self.chunk_sizes:
            for overlap in self.overlap_ratios:
                # Skip invalid combinations (overlap >= chunk_size)
                if overlap >= chunk_size:
                    continue
                    
                for i, query in enumerate(queries):
                    config = ExperimentConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        collection=collection,
                        query=query,
                        expected_chunks=3,  # Default expectation
                        run_id=f"sweep_{collection}_{chunk_size}_{overlap}_{i}"
                    )
                    all_configs.append(config)
        
        self.logger.log_info(f"Starting parameter sweep with {len(all_configs)} configurations", "parameter_sweep")
        self.logger.start_experiment(len(all_configs), f"Parameter sweep for {collection} collection")
        
        results = []
        for config in all_configs:
            try:
                metrics = self.run_single_experiment(config)
                self.logger.log_experiment_result(metrics)
                results.append(metrics)
                
                # Brief pause to prevent system overload
                time.sleep(1)
                
            except Exception as e:
                self.logger.log_error(e, f"parameter_sweep_{config.run_id}")
                # Create failed metrics
                failed_metrics = self._create_failed_experiment_metrics(
                    config, datetime.now(timezone.utc), str(e)
                )
                results.append(failed_metrics)
        
        return results

    def _create_failed_ingestion_metrics(self, collection_name: str, start_time: datetime, error_msg: str) -> IngestionMetrics:
        """Create failed ingestion metrics."""
        return IngestionMetrics(
            collection_name=collection_name,
            start_time=start_time.isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            documents_processed=0,
            documents_failed=0,
            chunks_created=0,
            total_tokens=0,
            processing_time_seconds=0,
            average_doc_size_bytes=0,
            ingestion_rate_docs_per_second=0,
            peak_memory_usage_gb=0,
            success=False,
            error_message=error_msg
        )

    def _create_failed_experiment_metrics(self, config: ExperimentConfig, start_time: datetime, error_msg: str) -> ExperimentMetrics:
        """Create failed experiment metrics."""
        return ExperimentMetrics(
            run_id=config.run_id,
            timestamp=start_time.isoformat(),
            query=config.query,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            collection=config.collection,
            response_time_seconds=0,
            retrieval_time_seconds=0,
            generation_time_seconds=0,
            retrieved_chunks=0,
            response_length_chars=0,
            response_length_words=0,
            tokens_per_second=0,
            memory_usage_mb=0,
            success=False,
            error_message=error_msg
        )

    def _extract_stat_from_output(self, output: str, stat_name: str, default: int = 0) -> int:
        """Extract numeric statistic from command output."""
        try:
            lines = output.lower().split('\n')
            for line in lines:
                if stat_name.lower() in line:
                    # Extract number from line
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[0])
            return default
        except Exception:
            return default

    def cleanup_test_data(self):
        """Clean up test data and temporary collections."""
        self.logger.log_info("Cleaning up test data", "cleanup")
        
        test_collections = ["fiqa_test", "scifact_test"]
        for collection in test_collections:
            cmd = f"{self.python_cmd} {self.main_script} collection delete {collection} --confirm"
            success, _, _ = self.run_command(cmd, timeout=60)
            if success:
                self.logger.log_info(f"Deleted test collection: {collection}", "cleanup")

if __name__ == "__main__":
    runner = ChunkingExperimentRunner()
    
    # Test the runner
    runner.logger.log_info("Testing experiment runner", "main")
    
    # Test system health
    if runner.test_system_health():
        runner.logger.log_info("System health check passed", "main")
    else:
        runner.logger.log_warning("System health check failed", "main")
    
    runner.logger.end_experiment()
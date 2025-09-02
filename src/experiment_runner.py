"""
Experiment Runner Framework for RAG System
Orchestrates parameter sweeps, A/B tests, and optimization experiments.
"""

import json
import sqlite3
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import traceback

from .config_manager import ExperimentConfig, ParameterRange, ExperimentTemplate, ConstraintValidator, ConfigManager, ProfileConfig
from .rag_pipeline import RAGPipeline


@dataclass
class ExperimentResult:
    """Container for individual experiment run results."""
    run_id: str
    config: ExperimentConfig
    query: str
    response: str
    metrics: Dict[str, float]
    duration_seconds: float
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class ExperimentResults:
    """Container for complete experiment results."""
    experiment_id: str
    experiment_type: str
    base_config: ExperimentConfig
    parameter_ranges: List[ParameterRange]
    results: List[ExperimentResult]
    total_runtime: float
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class ExperimentDatabase:
    """SQLite database for experiment results and metadata."""
    
    def __init__(self, db_path: str = "data/experiments.db"):
        """Initialize experiment database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_type TEXT NOT NULL,
                    base_config_json TEXT NOT NULL,
                    parameter_ranges_json TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_runs INTEGER DEFAULT 0,
                    completed_runs INTEGER DEFAULT 0,
                    total_runtime REAL DEFAULT 0.0,
                    error_message TEXT
                );
                
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    metrics_json TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_experiment_id ON experiment_runs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_experiment_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_experiment_type ON experiments(experiment_type);
            """)
    
    def create_experiment(self, experiment_id: str, experiment_type: str, 
                         base_config: ExperimentConfig, 
                         parameter_ranges: List[ParameterRange] = None) -> str:
        """Create a new experiment record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (
                    experiment_id, experiment_type, base_config_json, 
                    parameter_ranges_json, status, created_at
                ) VALUES (?, ?, ?, ?, 'created', ?)
            """, (
                experiment_id,
                experiment_type,
                json.dumps(asdict(base_config)),
                json.dumps([asdict(pr) for pr in parameter_ranges] if parameter_ranges else []),
                datetime.now(timezone.utc).isoformat()
            ))
        
        self.logger.info(f"Created experiment {experiment_id} of type {experiment_type}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str, total_runs: int):
        """Mark experiment as started."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET status = 'running', started_at = ?, total_runs = ?
                WHERE experiment_id = ?
            """, (datetime.now(timezone.utc).isoformat(), total_runs, experiment_id))
    
    def complete_experiment(self, experiment_id: str, total_runtime: float, 
                          status: str = 'completed', error_message: str = None):
        """Mark experiment as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET status = ?, completed_at = ?, total_runtime = ?, error_message = ?
                WHERE experiment_id = ?
            """, (status, datetime.now(timezone.utc).isoformat(), total_runtime, 
                  error_message, experiment_id))
    
    def save_run_result(self, result: ExperimentResult):
        """Save individual run result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiment_runs (
                    run_id, experiment_id, config_json, query, response,
                    metrics_json, duration_seconds, timestamp, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.run_id,
                result.run_id.split('_')[0],  # Extract experiment_id from run_id
                json.dumps(asdict(result.config)),
                result.query,
                result.response,
                json.dumps(result.metrics),
                result.duration_seconds,
                result.timestamp.isoformat(),
                result.error_message
            ))
            
            # Update completed runs count
            conn.execute("""
                UPDATE experiments 
                SET completed_runs = completed_runs + 1
                WHERE experiment_id = ?
            """, (result.run_id.split('_')[0],))
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Retrieve complete experiment results."""
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment metadata
            exp_row = conn.execute("""
                SELECT experiment_type, base_config_json, parameter_ranges_json,
                       total_runtime, status, created_at, completed_at
                FROM experiments WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()
            
            if not exp_row:
                return None
            
            # Get all run results
            run_rows = conn.execute("""
                SELECT run_id, config_json, query, response, metrics_json,
                       duration_seconds, timestamp, error_message
                FROM experiment_runs WHERE experiment_id = ?
                ORDER BY timestamp
            """, (experiment_id,)).fetchall()
            
            # Parse results
            results = []
            for row in run_rows:
                result = ExperimentResult(
                    run_id=row[0],
                    config=ExperimentConfig(**json.loads(row[1])),
                    query=row[2],
                    response=row[3] or "",
                    metrics=json.loads(row[4]),
                    duration_seconds=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    error_message=row[7]
                )
                results.append(result)
            
            return ExperimentResults(
                experiment_id=experiment_id,
                experiment_type=exp_row[0],
                base_config=ExperimentConfig(**json.loads(exp_row[1])),
                parameter_ranges=[ParameterRange(**pr) for pr in json.loads(exp_row[2])],
                results=results,
                total_runtime=exp_row[3],
                status=exp_row[4],
                created_at=datetime.fromisoformat(exp_row[5]),
                completed_at=datetime.fromisoformat(exp_row[6]) if exp_row[6] else None
            )


class ResourceManager:
    """Handles resource allocation and monitoring for experiments."""
    
    def __init__(self, max_memory_gb: float = 12.0, max_parallel_runs: int = 1):
        """Initialize resource manager."""
        self.max_memory_gb = max_memory_gb
        self.max_parallel_runs = max_parallel_runs
        self.logger = logging.getLogger(__name__)
        self.validator = ConstraintValidator()
    
    def check_resources(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Check if resources are available for configuration."""
        requirements = self.validator.check_resource_requirements(config)
        
        available = {
            "memory_available": requirements["memory_gb"] <= self.max_memory_gb,
            "requirements": requirements,
            "feasible": True
        }
        
        if not available["memory_available"]:
            available["feasible"] = False
            self.logger.warning(f"Configuration requires {requirements['memory_gb']:.1f}GB "
                              f"but only {self.max_memory_gb}GB available")
        
        return available


class ExperimentRunner:
    """Core experiment orchestration system."""
    
    def __init__(self, config_manager: ConfigManager = None, db_path: str = "data/experiments.db"):
        """Initialize experiment runner."""
        self.config_manager = config_manager or ConfigManager()
        self.db = ExperimentDatabase(db_path)
        self.resource_manager = ResourceManager()
        self.validator = ConstraintValidator()
        self.logger = logging.getLogger(__name__)
    
    def run_parameter_sweep(self, base_config: ExperimentConfig, 
                          parameter_ranges: List[ParameterRange],
                          queries: List[str],
                          experiment_id: str = None) -> ExperimentResults:
        """Execute systematic parameter variation experiment."""
        experiment_id = experiment_id or f"sweep_{int(time.time())}"
        
        self.logger.info(f"Starting parameter sweep experiment {experiment_id}")
        
        # Create experiment record
        self.db.create_experiment(experiment_id, "parameter_sweep", base_config, parameter_ranges)
        
        start_time = time.time()
        results = []
        
        try:
            # Generate all parameter combinations
            all_configs = self._generate_parameter_combinations(base_config, parameter_ranges)
            total_runs = len(all_configs) * len(queries)
            
            self.db.start_experiment(experiment_id, total_runs)
            self.logger.info(f"Generated {len(all_configs)} configurations × {len(queries)} queries = {total_runs} runs")
            
            # Run experiments
            for i, config in enumerate(all_configs):
                for j, query in enumerate(queries):
                    run_id = f"{experiment_id}_run_{i}_{j}"
                    
                    try:
                        result = self._run_single_experiment(run_id, config, query)
                        results.append(result)
                        self.db.save_run_result(result)
                        
                        self.logger.info(f"Completed run {len(results)}/{total_runs}: "
                                       f"{result.metrics.get('response_time', 0):.2f}s")
                        
                    except Exception as e:
                        error_result = ExperimentResult(
                            run_id=run_id,
                            config=config,
                            query=query,
                            response="",
                            metrics={"error": 1.0},
                            duration_seconds=0.0,
                            timestamp=datetime.now(timezone.utc),
                            error_message=str(e)
                        )
                        results.append(error_result)
                        self.db.save_run_result(error_result)
                        self.logger.error(f"Run {run_id} failed: {e}")
            
            total_runtime = time.time() - start_time
            self.db.complete_experiment(experiment_id, total_runtime, 'completed')
            
            self.logger.info(f"Parameter sweep completed in {total_runtime:.1f}s")
            
            return ExperimentResults(
                experiment_id=experiment_id,
                experiment_type="parameter_sweep",
                base_config=base_config,
                parameter_ranges=parameter_ranges,
                results=results,
                total_runtime=total_runtime,
                status="completed",
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            total_runtime = time.time() - start_time
            error_msg = f"Parameter sweep failed: {str(e)}\n{traceback.format_exc()}"
            self.db.complete_experiment(experiment_id, total_runtime, 'failed', error_msg)
            self.logger.error(error_msg)
            raise
    
    def run_ab_test(self, config_a: ExperimentConfig, config_b: ExperimentConfig,
                   queries: List[str], experiment_id: str = None,
                   significance_level: float = 0.05) -> ExperimentResults:
        """Execute A/B test between two configurations."""
        experiment_id = experiment_id or f"ab_test_{int(time.time())}"
        
        self.logger.info(f"Starting A/B test experiment {experiment_id}")
        
        # Create experiment record  
        self.db.create_experiment(experiment_id, "ab_test", config_a, [])
        
        start_time = time.time()
        results = []
        
        try:
            total_runs = len(queries) * 2  # Each query tested with both configs
            self.db.start_experiment(experiment_id, total_runs)
            
            # Test configuration A
            for i, query in enumerate(queries):
                run_id = f"{experiment_id}_A_{i}"
                result = self._run_single_experiment(run_id, config_a, query)
                result.config.target_corpus = "config_A"  # Mark which config
                results.append(result)
                self.db.save_run_result(result)
            
            # Test configuration B
            for i, query in enumerate(queries):
                run_id = f"{experiment_id}_B_{i}"
                result = self._run_single_experiment(run_id, config_b, query)
                result.config.target_corpus = "config_B"  # Mark which config
                results.append(result)
                self.db.save_run_result(result)
            
            # Perform statistical analysis
            a_metrics = [r.metrics for r in results[:len(queries)]]
            b_metrics = [r.metrics for r in results[len(queries):]]
            
            stats_results = self._perform_statistical_analysis(a_metrics, b_metrics, significance_level)
            
            total_runtime = time.time() - start_time
            self.db.complete_experiment(experiment_id, total_runtime, 'completed')
            
            self.logger.info(f"A/B test completed in {total_runtime:.1f}s")
            self.logger.info(f"Statistical significance: {stats_results}")
            
            return ExperimentResults(
                experiment_id=experiment_id,
                experiment_type="ab_test",
                base_config=config_a,
                parameter_ranges=[],
                results=results,
                total_runtime=total_runtime,
                status="completed",
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            total_runtime = time.time() - start_time
            error_msg = f"A/B test failed: {str(e)}\n{traceback.format_exc()}"
            self.db.complete_experiment(experiment_id, total_runtime, 'failed', error_msg)
            self.logger.error(error_msg)
            raise
    
    def run_template_experiment(self, template: ExperimentTemplate,
                              queries: List[str] = None,
                              experiment_id: str = None) -> ExperimentResults:
        """Run a pre-defined experiment template."""
        experiment_id = experiment_id or f"template_{template.name}_{int(time.time())}"
        
        # Use template's evaluation queries if none provided
        if not queries:
            queries = template.evaluation_queries or ["What is machine learning?", "How does AI work?"]
        
        self.logger.info(f"Running template experiment '{template.name}' ({experiment_id})")
        
        return self.run_parameter_sweep(
            base_config=template.base_config,
            parameter_ranges=template.parameter_ranges,
            queries=queries,
            experiment_id=experiment_id
        )
    
    def _generate_parameter_combinations(self, base_config: ExperimentConfig, 
                                       parameter_ranges: List[ParameterRange]) -> List[ExperimentConfig]:
        """Generate all parameter combinations."""
        import itertools
        
        # Extract parameter values
        param_names = [pr.param_name for pr in parameter_ranges]
        param_value_lists = [pr.generate_values() for pr in parameter_ranges]
        
        configurations = []
        
        for combination in itertools.product(*param_value_lists):
            # Create new config with parameter overrides
            config_dict = asdict(base_config)
            
            for param_name, param_value in zip(param_names, combination):
                config_dict[param_name] = param_value
            
            config = ExperimentConfig(**config_dict)
            
            # Validate configuration
            errors = self.validator.validate_config(config)
            if errors:
                self.logger.warning(f"Skipping invalid configuration: {errors}")
                continue
            
            # Check resource requirements
            resource_check = self.resource_manager.check_resources(config)
            if not resource_check["feasible"]:
                self.logger.warning(f"Skipping resource-intensive configuration: {config_dict}")
                continue
            
            configurations.append(config)
        
        return configurations
    
    def _run_single_experiment(self, run_id: str, config: ExperimentConfig, query: str) -> ExperimentResult:
        """Execute single experiment run."""
        start_time = time.time()
        
        try:
            # Initialize RAG pipeline with experimental config
            rag_pipeline = self._create_rag_pipeline(config)
            
            # Execute query and get full response dict
            result = rag_pipeline.query(
                query, 
                k=getattr(config, 'retrieval_k', 5),
                max_tokens=getattr(config, 'max_tokens', 1024),
                temperature=getattr(config, 'temperature', 0.7)
            )
            
            response = result['answer']
            sources = result['sources']
            
            duration = time.time() - start_time
            
            # Calculate metrics
            metrics = {
                "response_time": duration,
                "response_length": len(response.split()) if response else 0,
                "num_sources": len(sources) if sources else 0,
                "retrieval_success": 1.0 if sources else 0.0,
                "response_generated": 1.0 if response and response.strip() else 0.0
            }
            
            return ExperimentResult(
                run_id=run_id,
                config=config,
                query=query,
                response=response,
                metrics=metrics,
                duration_seconds=duration,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Experiment run {run_id} failed: {e}")
            
            return ExperimentResult(
                run_id=run_id,
                config=config,
                query=query,
                response="",
                metrics={"error": 1.0, "response_time": duration},
                duration_seconds=duration,
                timestamp=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    def _create_rag_pipeline(self, config: ExperimentConfig) -> RAGPipeline:
        """Create RAG pipeline with experimental configuration."""
        # Get current profile and create ProfileConfig from experiment config
        current_profile = self.config_manager.get_profile()
        
        # Create profile config with experiment parameters
        profile_config = ProfileConfig(
            retrieval_k=getattr(config, 'retrieval_k', current_profile.retrieval_k),
            max_tokens=getattr(config, 'max_tokens', current_profile.max_tokens),
            temperature=getattr(config, 'temperature', current_profile.temperature),
            chunk_size=getattr(config, 'chunk_size', current_profile.chunk_size),
            chunk_overlap=getattr(config, 'chunk_overlap', current_profile.chunk_overlap),
            n_ctx=getattr(config, 'n_ctx', current_profile.n_ctx)
        )
        
        # Get database and model paths from config manager
        db_path = self.config_manager.get_param('database.path', 'data/rag_vectors.db')

        # Prefer experiment config values; fall back to global config or sensible defaults
        try:
            from src.config_manager import ExperimentConfig as _DefaultConfig
            _default_llm = _DefaultConfig().llm_model_path
            _default_embed = _DefaultConfig().embedding_model_path
        except Exception:
            _default_llm = 'models/gemma-3-4b-it-q4_0.gguf'
            _default_embed = 'sentence-transformers/all-MiniLM-L6-v2'

        embedding_model_path = getattr(config, 'embedding_model_path', None) or \
                               self.config_manager.get_param('embedding_model_path', _default_embed)
        llm_model_path = getattr(config, 'llm_model_path', None) or \
                         self.config_manager.get_param('llm_model_path', _default_llm)
        
        # Create RAG pipeline with profile configuration
        rag_pipeline = RAGPipeline(
            db_path=db_path,
            embedding_model_path=embedding_model_path,
            llm_model_path=llm_model_path,
            profile_config=profile_config
        )
        
        # Handle chunking parameters by creating per-config collection
        if hasattr(config, 'chunk_size') or hasattr(config, 'chunk_overlap'):
            collection_id = self._ensure_chunked_collection(config)
            rag_pipeline.set_corpus(collection_id)
        
        return rag_pipeline

    def _ensure_chunked_collection(self, config: ExperimentConfig) -> str:
        """Ensure a collection exists with the specified chunking parameters."""
        # Generate unique collection ID for this config
        chunk_size = getattr(config, 'chunk_size', 512)
        chunk_overlap = getattr(config, 'chunk_overlap', 128)
        collection_id = f"exp_cs{chunk_size}_co{chunk_overlap}"
        
        # Check if collection already exists
        try:
            collections = self.db.list_collections()
            if any(c['collection_id'] == collection_id for c in collections):
                self.logger.info(f"Reusing existing collection: {collection_id}")
                return collection_id
        except Exception as e:
            self.logger.warning(f"Could not check existing collections: {e}")
        
        # Create new collection with proper chunking
        self.logger.info(f"Creating chunked collection: {collection_id} (size={chunk_size}, overlap={chunk_overlap})")
        
        # Use ReindexTool to create properly chunked collection
        try:
            from .reindex import ReindexTool
            db_path = self.config_manager.get_param('database.path', 'data/rag_vectors.db')
            reindex_tool = ReindexTool(db_path)
            
            # Source collection (configurable, default to production)
            source_collection = "realistic_full_production"
            
            # Copy and rechunk documents
            stats = reindex_tool.rechunk_documents(
                collection_id=collection_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reembed=True,
                backup=False
            )
            
            if not stats.success:
                raise RuntimeError(f"Rechunking failed: {stats.details.get('error', 'Unknown error')}")
                
            self.logger.info(f"Created collection {collection_id}: {stats.documents_processed} docs, {stats.chunks_processed} chunks")
            return collection_id
            
        except Exception as e:
            self.logger.error(f"Failed to create chunked collection {collection_id}: {e}")
            # Fallback to default collection
            return "default"
    
    def _perform_statistical_analysis(self, metrics_a: List[Dict], metrics_b: List[Dict], 
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """Perform statistical analysis for A/B test."""
        try:
            from scipy import stats
            import numpy as np
            
            # Extract response times for comparison
            times_a = [m.get("response_time", 0) for m in metrics_a]
            times_b = [m.get("response_time", 0) for m in metrics_b]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(times_a, times_b)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((np.std(times_a, ddof=1) ** 2) + (np.std(times_b, ddof=1) ** 2)) / 2)
            cohens_d = (np.mean(times_a) - np.mean(times_b)) / pooled_std if pooled_std > 0 else 0
            
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "cohens_d": cohens_d,
                "mean_a": np.mean(times_a),
                "mean_b": np.mean(times_b),
                "std_a": np.std(times_a, ddof=1),
                "std_b": np.std(times_b, ddof=1)
            }
            
        except ImportError:
            self.logger.warning("scipy not available - using basic statistics")
            import statistics
            
            times_a = [m.get("response_time", 0) for m in metrics_a]
            times_b = [m.get("response_time", 0) for m in metrics_b]
            
            return {
                "mean_a": statistics.mean(times_a),
                "mean_b": statistics.mean(times_b),
                "median_a": statistics.median(times_a),
                "median_b": statistics.median(times_b),
                "statistical_test": "basic_comparison"
            }


def create_experiment_runner(config_manager: ConfigManager = None) -> ExperimentRunner:
    """Factory function to create ExperimentRunner instance."""
    return ExperimentRunner(config_manager)
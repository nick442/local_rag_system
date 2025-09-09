#!/usr/bin/env python3
"""
Comprehensive experiment logging system for chunking optimization experiments.

This module provides structured logging, metrics collection, and progress tracking
for the document chunking strategy optimization experiments.
"""

import json
import time
import psutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading
import sys

@dataclass
class SystemMetrics:
    """System resource usage metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float

@dataclass
class ExperimentMetrics:
    """Experiment-specific metrics."""
    run_id: str
    timestamp: str
    query: str
    chunk_size: int
    chunk_overlap: int
    collection: str
    response_time_seconds: float
    retrieval_time_seconds: float
    generation_time_seconds: float
    retrieved_chunks: int
    response_length_chars: int
    response_length_words: int
    tokens_per_second: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class IngestionMetrics:
    """Document ingestion tracking metrics."""
    collection_name: str
    start_time: str
    end_time: Optional[str]
    documents_processed: int
    documents_failed: int
    chunks_created: int
    total_tokens: int
    processing_time_seconds: float
    average_doc_size_bytes: float
    ingestion_rate_docs_per_second: float
    peak_memory_usage_gb: float
    success: bool
    error_message: Optional[str] = None

class ExperimentLogger:
    """Comprehensive experiment logging and metrics collection."""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments/chunking/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log files
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{self.session_id}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics_{self.session_id}.jsonl"
        self.system_metrics_file = self.log_dir / f"system_metrics_{self.session_id}.jsonl"
        
        # Set up structured logging
        self.setup_logging()
        
        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.experiment_start_time = None
        
        # Experiment state tracking
        self.current_phase = "initialization"
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0
        
        self.logger.info(f"Experiment logger initialized for: {experiment_name}")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Log files: {self.log_file}, {self.metrics_file}")

    def setup_logging(self):
        """Configure structured logging."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(f"chunking_experiment.{self.experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs from parent loggers
        self.logger.propagate = False

    def start_experiment(self, total_experiments: int, description: str):
        """Start experiment session with comprehensive tracking."""
        self.experiment_start_time = datetime.now(timezone.utc)
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        self.failed_experiments = 0
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING EXPERIMENT: {self.experiment_name}")
        self.logger.info(f"Description: {description}")
        self.logger.info(f"Total experiments planned: {total_experiments}")
        self.logger.info(f"Start time: {self.experiment_start_time.isoformat()}")
        self.logger.info("=" * 80)
        
        # Start system monitoring
        self.start_system_monitoring()
        
        # Log initial system state
        self._log_system_state("experiment_start")

    def start_phase(self, phase_name: str, description: str):
        """Start a new experiment phase."""
        self.current_phase = phase_name
        phase_start_time = datetime.now(timezone.utc)
        
        self.logger.info("-" * 60)
        self.logger.info(f"PHASE START: {phase_name}")
        self.logger.info(f"Description: {description}")
        self.logger.info(f"Time: {phase_start_time.isoformat()}")
        self.logger.info("-" * 60)
        
        return phase_start_time

    def end_phase(self, phase_name: str, phase_start_time: datetime):
        """End current experiment phase."""
        phase_end_time = datetime.now(timezone.utc)
        duration = (phase_end_time - phase_start_time).total_seconds()
        
        self.logger.info("-" * 60)
        self.logger.info(f"PHASE COMPLETE: {phase_name}")
        self.logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        self.logger.info(f"End time: {phase_end_time.isoformat()}")
        self.logger.info("-" * 60)

    def log_ingestion_start(self, collection_name: str, estimated_docs: int) -> str:
        """Log start of corpus ingestion."""
        ingestion_id = f"ingestion_{collection_name}_{int(time.time())}"
        
        self.logger.info(f"INGESTION START: {collection_name}")
        self.logger.info(f"Ingestion ID: {ingestion_id}")
        self.logger.info(f"Estimated documents: {estimated_docs}")
        
        return ingestion_id

    def log_ingestion_complete(self, metrics: IngestionMetrics):
        """Log completion of corpus ingestion."""
        self.logger.info(f"INGESTION COMPLETE: {metrics.collection_name}")
        self.logger.info(f"Documents processed: {metrics.documents_processed}")
        self.logger.info(f"Documents failed: {metrics.documents_failed}")
        self.logger.info(f"Chunks created: {metrics.chunks_created}")
        self.logger.info(f"Processing time: {metrics.processing_time_seconds:.2f}s")
        self.logger.info(f"Rate: {metrics.ingestion_rate_docs_per_second:.2f} docs/sec")
        self.logger.info(f"Peak memory: {metrics.peak_memory_usage_gb:.2f}GB")
        
        # Write metrics to JSONL
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')

    def log_experiment_result(self, metrics: ExperimentMetrics):
        """Log individual experiment result."""
        if metrics.success:
            self.completed_experiments += 1
            self.logger.debug(f"Experiment {metrics.run_id}: SUCCESS - {metrics.response_time_seconds:.2f}s")
        else:
            self.failed_experiments += 1
            self.logger.warning(f"Experiment {metrics.run_id}: FAILED - {metrics.error_message}")
        
        # Write metrics to JSONL
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        # Log progress
        progress = (self.completed_experiments + self.failed_experiments) / self.total_experiments * 100
        self.logger.info(f"Progress: {progress:.1f}% ({self.completed_experiments + self.failed_experiments}/{self.total_experiments})")

    def start_system_monitoring(self, interval_seconds: int = 30):
        """Start continuous system resource monitoring."""
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                metrics = self._get_system_metrics()
                with open(self.system_metrics_file, 'a') as f:
                    f.write(json.dumps(asdict(metrics)) + '\n')
                time.sleep(interval_seconds)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("System monitoring started")

    def stop_system_monitoring(self):
        """Stop system resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")

    def end_experiment(self):
        """End experiment session with summary."""
        experiment_end_time = datetime.now(timezone.utc)
        if self.experiment_start_time is None:
            self.experiment_start_time = experiment_end_time
            duration = 0.0
        else:
            duration = (experiment_end_time - self.experiment_start_time).total_seconds()
        
        success_rate = self.completed_experiments / self.total_experiments * 100 if self.total_experiments > 0 else 0
        
        self.logger.info("=" * 80)
        self.logger.info(f"EXPERIMENT COMPLETE: {self.experiment_name}")
        self.logger.info(f"Total duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
        self.logger.info(f"Experiments completed: {self.completed_experiments}/{self.total_experiments}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Failed experiments: {self.failed_experiments}")
        self.logger.info(f"End time: {experiment_end_time.isoformat()}")
        self.logger.info("=" * 80)
        
        # Stop monitoring
        self.stop_system_monitoring()
        
        # Generate summary report
        self._generate_summary_report(duration, success_rate)

    def log_error(self, error: Exception, context: str):
        """Log error with full context."""
        self.logger.error(f"ERROR in {context}: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
        import traceback
        self.logger.error(f"Traceback: {traceback.format_exc()}")

    def log_warning(self, message: str, context: str = ""):
        """Log warning with context."""
        self.logger.warning(f"WARNING {context}: {message}")

    def log_info(self, message: str, context: str = ""):
        """Log info with context."""
        context_str = f"[{context}] " if context else ""
        self.logger.info(f"{context_str}{message}")

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            disk_percent=disk.used / disk.total * 100
        )

    def _log_system_state(self, event: str):
        """Log current system state for key events."""
        metrics = self._get_system_metrics()
        self.logger.info(f"SYSTEM STATE ({event}):")
        self.logger.info(f"  CPU: {metrics.cpu_percent:.1f}%")
        self.logger.info(f"  Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.2f}GB/{metrics.memory_total_gb:.2f}GB)")
        self.logger.info(f"  Disk: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}GB free)")

    def _generate_summary_report(self, duration_seconds: float, success_rate: float):
        """Generate experiment summary report."""
        summary = {
            "experiment_name": self.experiment_name,
            "session_id": self.session_id,
            "start_time": self.experiment_start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_seconds / 3600,
            "total_experiments": self.total_experiments,
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "success_rate_percent": success_rate,
            "log_files": {
                "main_log": str(self.log_file),
                "metrics": str(self.metrics_file),
                "system_metrics": str(self.system_metrics_file)
            }
        }
        
        summary_file = self.log_dir / f"experiment_summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved: {summary_file}")

    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current experiment progress statistics."""
        if self.total_experiments == 0:
            return {"progress_percent": 0, "completed": 0, "failed": 0, "total": 0}
        
        completed_total = self.completed_experiments + self.failed_experiments
        progress_percent = (completed_total / self.total_experiments) * 100
        
        return {
            "progress_percent": progress_percent,
            "completed": self.completed_experiments,
            "failed": self.failed_experiments,
            "total": self.total_experiments,
            "current_phase": self.current_phase,
            "session_id": self.session_id
        }

if __name__ == "__main__":
    # Test the logger
    logger = ExperimentLogger("test_experiment")
    logger.start_experiment(10, "Test experiment for logging system")
    
    # Simulate some experiments
    for i in range(3):
        time.sleep(1)
        metrics = ExperimentMetrics(
            run_id=f"test_{i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=f"Test query {i}",
            chunk_size=512,
            chunk_overlap=64,
            collection="test_collection",
            response_time_seconds=0.5 + i * 0.1,
            retrieval_time_seconds=0.2,
            generation_time_seconds=0.3,
            retrieved_chunks=5,
            response_length_chars=100,
            response_length_words=20,
            tokens_per_second=50.0,
            memory_usage_mb=200.0,
            success=True
        )
        logger.log_experiment_result(metrics)
    
    logger.end_experiment()
    print("Test completed - check logs directory")
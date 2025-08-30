#!/usr/bin/env python3
"""
Continuous Monitoring for RAG System

Runtime monitoring tools:
1. Query logging
2. Performance tracking
3. Error collection
4. Usage statistics
"""

import time
import json
import logging
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from datetime import datetime, timedelta
import statistics

@dataclass
class QueryLog:
    """Individual query log entry"""
    timestamp: str
    query: str
    response_length: int
    latency_ms: float
    sources_count: int
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: str
    metric_name: str
    value: float
    tags: Dict[str, str] = None

@dataclass
class SystemMetric:
    """System resource metric"""
    timestamp: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

class RAGMonitor:
    """Continuous monitoring for RAG system"""
    
    def __init__(self, log_dir: str = "logs", db_file: str = "monitoring.db"):
        """Initialize monitoring system"""
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for persistent storage
        self.db_file = self.log_dir / db_file
        self.init_database()
        
        # In-memory metrics storage
        self.query_logs = deque(maxlen=1000)  # Keep last 1000 queries in memory
        self.performance_metrics = defaultdict(lambda: deque(maxlen=100))
        self.error_counts = defaultdict(int)
        self.session_stats = defaultdict(dict)
        
        # System monitoring
        self.process = psutil.Process()
        self.system_metrics = deque(maxlen=60)  # Keep last 60 system snapshots
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Thresholds for alerting
        self.thresholds = {
            'slow_query_ms': 2000,
            'error_rate_threshold': 0.1,  # 10%
            'memory_threshold_mb': 1000,
            'cpu_threshold_percent': 80
        }
        
        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'total_errors': 0,
            'total_sessions': 0,
            'avg_latency_ms': 0,
            'avg_response_length': 0
        }
    
    def init_database(self) -> None:
        """Initialize SQLite database for persistent logging"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                # Query logs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response_length INTEGER,
                        latency_ms REAL,
                        sources_count INTEGER,
                        error TEXT,
                        user_id TEXT,
                        session_id TEXT
                    )
                """)
                
                # Performance metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        tags TEXT
                    )
                """)
                
                # System metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_percent REAL,
                        memory_mb REAL,
                        memory_percent REAL,
                        disk_io_read_mb REAL,
                        disk_io_write_mb REAL
                    )
                """)
                
                # Indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sys_timestamp ON system_metrics(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def log_query(self, query: str, response: Any, latency_ms: float,
                  error: str = None, user_id: str = None, session_id: str = None) -> None:
        """Log a query and its response"""
        timestamp = datetime.now().isoformat()
        
        # Extract response information
        response_length = 0
        sources_count = 0
        
        if response:
            if isinstance(response, dict):
                response_text = response.get('response', str(response))
                sources = response.get('sources', [])
                sources_count = len(sources)
            else:
                response_text = str(response)
            
            response_length = len(response_text)
        
        # Create log entry
        log_entry = QueryLog(
            timestamp=timestamp,
            query=query,
            response_length=response_length,
            latency_ms=latency_ms,
            sources_count=sources_count,
            error=error,
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to in-memory storage
        self.query_logs.append(log_entry)
        
        # Update statistics
        self.stats['total_queries'] += 1
        if error:
            self.stats['total_errors'] += 1
            self.error_counts[error] += 1
        
        # Check for slow query
        if latency_ms > self.thresholds['slow_query_ms']:
            self.logger.warning(f"Slow query detected: {latency_ms:.2f}ms - {query[:100]}")
        
        # Persist to database
        self._persist_query_log(log_entry)
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Continuous monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Continuous monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                system_metric = SystemMetric(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=self.process.memory_percent(),
                    disk_io_read_mb=0,  # Simplified for now
                    disk_io_write_mb=0
                )
                
                self.system_metrics.append(system_metric)
                
                # Check thresholds
                if memory_mb > self.thresholds['memory_threshold_mb']:
                    self.logger.warning(f"High memory usage: {memory_mb:.2f}MB")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _persist_query_log(self, log_entry: QueryLog) -> None:
        """Persist query log to database"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("""
                    INSERT INTO query_logs 
                    (timestamp, query, response_length, latency_ms, sources_count, error, user_id, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_entry.timestamp, log_entry.query, log_entry.response_length,
                    log_entry.latency_ms, log_entry.sources_count, log_entry.error,
                    log_entry.user_id, log_entry.session_id
                ))
        except Exception as e:
            self.logger.error(f"Error persisting query log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            'total_queries': self.stats['total_queries'],
            'total_errors': self.stats['total_errors'],
            'error_rate': self.stats['total_errors'] / max(self.stats['total_queries'], 1),
            'avg_latency_ms': self.stats['avg_latency_ms'],
            'monitoring_active': self.monitoring_active,
            'recent_queries': len(self.query_logs),
            'system_metrics_collected': len(self.system_metrics)
        }

# Global monitor instance
_monitor_instance = None

def get_monitor() -> RAGMonitor:
    """Get global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RAGMonitor()
    return _monitor_instance

def monitor_query(func):
    """Decorator to automatically monitor RAG queries"""
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        query = args[0] if args else "unknown"
        
        start_time = time.time()
        error = None
        response = None
        
        try:
            response = func(*args, **kwargs)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            monitor.log_query(query, response, latency_ms, error)
    
    return wrapper
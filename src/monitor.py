"""
System Monitoring and Statistics

Provides real-time system monitoring, performance tracking,
and session statistics using psutil.
"""

import psutil
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


@dataclass
class SystemStats:
    """System resource statistics."""
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class SessionStats:
    """RAG session performance statistics."""
    total_queries: int
    total_tokens_generated: int
    total_tokens_prompt: int
    avg_tokens_per_second: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    peak_memory_gb: float
    current_memory_gb: float
    session_duration_minutes: float
    cache_hit_rate: float
    errors_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryMetrics:
    """Individual query performance metrics."""
    
    def __init__(self):
        self.retrieval_time: float = 0.0
        self.generation_time: float = 0.0
        self.total_time: float = 0.0
        self.tokens_generated: int = 0
        self.tokens_prompt: int = 0
        self.memory_before_mb: float = 0.0
        self.memory_after_mb: float = 0.0
        self.timestamp: datetime = datetime.now()
        self.success: bool = True
        self.error_message: Optional[str] = None


class Monitor:
    """
    System monitoring and performance tracking.
    
    Features:
    - Real-time system resource monitoring
    - Query performance tracking
    - Session statistics
    - Memory usage analysis
    - Cache hit rate tracking
    """
    
    def __init__(self, enable_continuous_monitoring: bool = True):
        """
        Initialize monitor.
        
        Args:
            enable_continuous_monitoring: Whether to start background monitoring
        """
        self.logger = logging.getLogger(__name__)
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.query_metrics: List[QueryMetrics] = []
        self.system_stats_history: deque = deque(maxlen=1000)  # Keep last 1000 readings
        
        # Performance tracking
        self._total_queries = 0
        self._total_tokens_generated = 0
        self._total_tokens_prompt = 0
        self._peak_memory_gb = 0.0
        self._errors_count = 0
        self._cache_hits = 0
        self._cache_requests = 0
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_interval = 5.0  # seconds
        
        if enable_continuous_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Started background system monitoring")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        self.logger.info("Stopped background system monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                stats = self.get_current_system_stats()
                self.system_stats_history.append(stats)
                
                # Update peak memory
                if stats.memory_used_gb > self._peak_memory_gb:
                    self._peak_memory_gb = stats.memory_used_gb
                
                time.sleep(self._monitoring_interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self._monitoring_interval)
    
    def get_current_system_stats(self) -> SystemStats:
        """
        Get current system resource statistics.
        
        Returns:
            Current system statistics
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        
        # Disk usage for current directory
        disk = psutil.disk_usage('.')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        return SystemStats(
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            memory_percent=memory_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
            timestamp=datetime.now()
        )
    
    def start_query_tracking(self) -> QueryMetrics:
        """
        Start tracking a new query.
        
        Returns:
            QueryMetrics object for this query
        """
        metrics = QueryMetrics()
        metrics.memory_before_mb = psutil.Process().memory_info().rss / (1024**2)
        return metrics
    
    def end_query_tracking(self, metrics: QueryMetrics, 
                          tokens_generated: int = 0,
                          tokens_prompt: int = 0,
                          retrieval_time: float = 0.0,
                          generation_time: float = 0.0,
                          success: bool = True,
                          error_message: Optional[str] = None):
        """
        Complete query tracking and update statistics.
        
        Args:
            metrics: QueryMetrics object from start_query_tracking
            tokens_generated: Number of tokens generated
            tokens_prompt: Number of tokens in prompt
            retrieval_time: Time spent on retrieval (seconds)
            generation_time: Time spent on generation (seconds)
            success: Whether query was successful
            error_message: Error message if query failed
        """
        # Update metrics
        metrics.tokens_generated = tokens_generated
        metrics.tokens_prompt = tokens_prompt
        metrics.retrieval_time = retrieval_time
        metrics.generation_time = generation_time
        metrics.total_time = retrieval_time + generation_time
        metrics.success = success
        metrics.error_message = error_message
        metrics.memory_after_mb = psutil.Process().memory_info().rss / (1024**2)
        
        # Add to history
        self.query_metrics.append(metrics)
        
        # Update session totals
        self._total_queries += 1
        if success:
            self._total_tokens_generated += tokens_generated
            self._total_tokens_prompt += tokens_prompt
        else:
            self._errors_count += 1
        
        self.logger.debug(f"Query tracking complete: {tokens_generated} tokens in {generation_time:.2f}s")
    
    def record_cache_hit(self, hit: bool = True):
        """
        Record a cache hit or miss.
        
        Args:
            hit: True for cache hit, False for cache miss
        """
        self._cache_requests += 1
        if hit:
            self._cache_hits += 1
    
    def get_session_stats(self) -> SessionStats:
        """
        Get comprehensive session statistics.
        
        Returns:
            Session statistics object
        """
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        
        # Calculate averages from recent queries
        recent_queries = [q for q in self.query_metrics if q.success]
        
        if recent_queries:
            avg_retrieval_latency = sum(q.retrieval_time for q in recent_queries) / len(recent_queries) * 1000
            avg_generation_latency = sum(q.generation_time for q in recent_queries) / len(recent_queries) * 1000
            
            # Calculate tokens per second
            total_generation_time = sum(q.generation_time for q in recent_queries)
            if total_generation_time > 0:
                avg_tokens_per_second = self._total_tokens_generated / total_generation_time
            else:
                avg_tokens_per_second = 0.0
        else:
            avg_retrieval_latency = 0.0
            avg_generation_latency = 0.0
            avg_tokens_per_second = 0.0
        
        # Cache hit rate
        cache_hit_rate = (self._cache_hits / self._cache_requests) if self._cache_requests > 0 else 0.0
        
        # Current memory usage
        current_stats = self.get_current_system_stats()
        
        return SessionStats(
            total_queries=self._total_queries,
            total_tokens_generated=self._total_tokens_generated,
            total_tokens_prompt=self._total_tokens_prompt,
            avg_tokens_per_second=avg_tokens_per_second,
            avg_retrieval_latency_ms=avg_retrieval_latency,
            avg_generation_latency_ms=avg_generation_latency,
            peak_memory_gb=self._peak_memory_gb,
            current_memory_gb=current_stats.memory_used_gb,
            session_duration_minutes=session_duration,
            cache_hit_rate=cache_hit_rate,
            errors_count=self._errors_count
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Performance summary dictionary
        """
        session_stats = self.get_session_stats()
        current_stats = self.get_current_system_stats()
        
        # Query performance trends
        recent_queries = self.query_metrics[-10:] if self.query_metrics else []
        query_trends = {
            'recent_avg_response_time': sum(q.total_time for q in recent_queries) / len(recent_queries) if recent_queries else 0.0,
            'recent_avg_tokens_per_sec': sum(q.tokens_generated / q.generation_time for q in recent_queries if q.generation_time > 0) / len([q for q in recent_queries if q.generation_time > 0]) if recent_queries else 0.0,
            'recent_memory_trend': [q.memory_after_mb for q in recent_queries]
        }
        
        # System resource trends
        recent_system_stats = list(self.system_stats_history)[-20:] if self.system_stats_history else []
        system_trends = {
            'avg_cpu_usage': sum(s.cpu_percent for s in recent_system_stats) / len(recent_system_stats) if recent_system_stats else 0.0,
            'avg_memory_usage': sum(s.memory_percent for s in recent_system_stats) / len(recent_system_stats) if recent_system_stats else 0.0,
            'memory_trend': [s.memory_used_gb for s in recent_system_stats]
        }
        
        return {
            'session_stats': session_stats.to_dict(),
            'current_system': current_stats.to_dict(),
            'query_trends': query_trends,
            'system_trends': system_trends,
            'monitoring_active': self._monitoring_active,
            'total_queries_tracked': len(self.query_metrics),
            'system_readings_count': len(self.system_stats_history)
        }
    
    def format_stats_display(self) -> str:
        """
        Format statistics for display in CLI.
        
        Returns:
            Formatted statistics string
        """
        session_stats = self.get_session_stats()
        current_stats = self.get_current_system_stats()
        
        return f"""=== Session Statistics ===
Queries: {session_stats.total_queries}
Avg Speed: {session_stats.avg_tokens_per_second:.1f} tokens/sec
Avg Retrieval: {session_stats.avg_retrieval_latency_ms:.0f}ms
Avg Generation: {session_stats.avg_generation_latency_ms:.1f}s
Memory: {current_stats.memory_used_gb:.1f}GB / {current_stats.memory_total_gb:.1f}GB ({current_stats.memory_percent:.1f}%)
Peak Memory: {session_stats.peak_memory_gb:.1f}GB
Total Tokens: {session_stats.total_tokens_generated:,}
Cache Hit Rate: {session_stats.cache_hit_rate:.1%}
Errors: {session_stats.errors_count}
Session Duration: {session_stats.session_duration_minutes:.1f} minutes"""
    
    def reset_session(self):
        """Reset session statistics."""
        self.session_start_time = datetime.now()
        self.query_metrics.clear()
        self._total_queries = 0
        self._total_tokens_generated = 0
        self._total_tokens_prompt = 0
        self._peak_memory_gb = 0.0
        self._errors_count = 0
        self._cache_hits = 0
        self._cache_requests = 0
        
        self.logger.info("Session statistics reset")
    
    def export_stats(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export statistics in specified format.
        
        Args:
            format: Export format ('json', 'csv', etc.)
        
        Returns:
            Exported statistics data
        """
        if format.lower() == 'json':
            return self.get_performance_summary()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        if hasattr(self, '_monitoring_active'):
            self.stop_monitoring()


def create_monitor(enable_continuous_monitoring: bool = True) -> Monitor:
    """
    Factory function to create a Monitor instance.
    
    Args:
        enable_continuous_monitoring: Whether to start background monitoring
    
    Returns:
        Monitor instance
    """
    return Monitor(enable_continuous_monitoring=enable_continuous_monitoring)
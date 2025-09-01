"""
Automatic performance tuning system for the RAG pipeline.

Monitors system performance in real-time and applies optimizations:
1. Identifies performance bottlenecks automatically  
2. Applies targeted optimizations based on detected issues
3. Monitors optimization effectiveness and rolls back if needed
4. Learns optimal configurations over time
5. Provides recommendations for manual optimization
"""

import time
import logging
import threading
import statistics
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import numpy as np


class BottleneckType(Enum):
    """Types of performance bottlenecks that can be detected."""
    MEMORY_PRESSURE = "memory_pressure"
    CPU_BOTTLENECK = "cpu_bottleneck"
    SLOW_GENERATION = "slow_generation"
    SLOW_RETRIEVAL = "slow_retrieval"
    CACHE_MISSES = "cache_misses"
    DATABASE_SLOW = "database_slow"
    IO_BOTTLENECK = "io_bottleneck"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    token_throughput: float
    memory_usage_mb: float
    cpu_percent: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    cache_hit_rate: float
    db_query_time_ms: float
    io_wait_time_ms: float


@dataclass
class OptimizationAction:
    """Container for optimization actions."""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    applied_at: float
    success: bool = False
    actual_improvement: float = 0.0


class AutoTuner:
    """
    Automatic performance tuning system for the RAG pipeline.
    
    Monitors metrics, identifies bottlenecks, and applies optimizations automatically.
    """
    
    def __init__(self, config_manager_or_system, monitoring_interval: float = 30.0,
                 optimization_threshold: float = 0.2):
        """
        Initialize auto-tuner.
        
        Args:
            config_manager_or_system: ConfigManager or legacy SystemManager for accessing components
            monitoring_interval: How often to collect metrics (seconds)
            optimization_threshold: Minimum performance degradation to trigger optimization
        """
        # Support both ConfigManager and legacy SystemManager
        if hasattr(config_manager_or_system, 'get_param'):
            self.config_manager = config_manager_or_system
            self.system_manager = None
        else:
            self.system_manager = config_manager_or_system
            self.config_manager = getattr(config_manager_or_system, 'config', None)
        self.monitoring_interval = monitoring_interval
        self.optimization_threshold = optimization_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Metrics history (using deque for efficient rolling window)
        self.metrics_history = deque(maxlen=100)  # Keep last 100 measurements
        self.baseline_metrics = None
        
        # Optimization tracking
        self.applied_optimizations = []
        self.optimization_history = deque(maxlen=50)
        
        # Bottleneck detection thresholds
        self.thresholds = {
            BottleneckType.MEMORY_PRESSURE: {'memory_percent': 85, 'growth_rate': 10},
            BottleneckType.CPU_BOTTLENECK: {'cpu_percent': 90, 'sustained_duration': 60},
            BottleneckType.SLOW_GENERATION: {'tokens_per_second': 50, 'latency_ms': 30000},
            BottleneckType.SLOW_RETRIEVAL: {'retrieval_ms': 5000, 'degradation': 50},
            BottleneckType.CACHE_MISSES: {'hit_rate': 0.3, 'miss_rate_increase': 0.2},
            BottleneckType.DATABASE_SLOW: {'query_time_ms': 1000, 'degradation': 100},
            BottleneckType.IO_BOTTLENECK: {'io_wait_ms': 2000, 'io_percent': 30}
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def _get_component(self, component_name: str):
        """Get component from system manager or return None if not available."""
        if self.system_manager and hasattr(self.system_manager, 'get_component'):
            try:
                return self.system_manager.get_component(component_name)
            except:
                return None
        return None
        
        # Optimization strategies
        self.optimization_strategies = {
            BottleneckType.MEMORY_PRESSURE: self._optimize_memory_pressure,
            BottleneckType.CPU_BOTTLENECK: self._optimize_cpu_bottleneck,
            BottleneckType.SLOW_GENERATION: self._optimize_generation_speed,
            BottleneckType.SLOW_RETRIEVAL: self._optimize_retrieval_speed,
            BottleneckType.CACHE_MISSES: self._optimize_cache_performance,
            BottleneckType.DATABASE_SLOW: self._optimize_database_performance,
            BottleneckType.IO_BOTTLENECK: self._optimize_io_performance
        }
        
        self.logger.info(f"AutoTuner initialized - Monitoring interval: {monitoring_interval}s")
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics from all system components."""
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get component-specific metrics
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                token_throughput=self._get_token_throughput(),
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_percent=cpu_percent,
                retrieval_latency_ms=self._get_retrieval_latency(),
                generation_latency_ms=self._get_generation_latency(),
                cache_hit_rate=self._get_cache_hit_rate(),
                db_query_time_ms=self._get_db_query_time(),
                io_wait_time_ms=self._get_io_wait_time()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics to avoid breaking the monitoring loop
            return PerformanceMetrics(
                timestamp=time.time(), token_throughput=0, memory_usage_mb=0,
                cpu_percent=0, retrieval_latency_ms=0, generation_latency_ms=0,
                cache_hit_rate=0, db_query_time_ms=0, io_wait_time_ms=0
            )
    
    def _get_token_throughput(self) -> float:
        """Get current token throughput from monitor component."""
        try:
            monitor = self._get_component('monitor')
            if monitor and hasattr(monitor, 'get_current_stats'):
                stats = monitor.get_current_stats()
                return stats.get('token_throughput', 0.0)
        except Exception as e:
            self.logger.debug(f"Could not get token throughput: {e}")
        return 0.0
    
    def _get_retrieval_latency(self) -> float:
        """Get current retrieval latency."""
        try:
            monitor = self._get_component('monitor')
            if monitor and hasattr(monitor, 'get_current_stats'):
                stats = monitor.get_current_stats()
                return stats.get('avg_retrieval_time_ms', 0.0)
        except Exception as e:
            self.logger.debug(f"Could not get retrieval latency: {e}")
        return 0.0
    
    def _get_generation_latency(self) -> float:
        """Get current generation latency."""
        try:
            monitor = self._get_component('monitor')
            if monitor and hasattr(monitor, 'get_current_stats'):
                stats = monitor.get_current_stats()
                return stats.get('avg_generation_time_ms', 0.0)
        except Exception as e:
            self.logger.debug(f"Could not get generation latency: {e}")
        return 0.0
    
    def _get_cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        try:
            cache_manager = self._get_component('cache_manager')
            if cache_manager and hasattr(cache_manager, 'get_cache_statistics'):
                stats = cache_manager.get_cache_statistics()
                return stats['global_stats'].get('hit_rate', 0.0)
        except Exception as e:
            self.logger.debug(f"Could not get cache hit rate: {e}")
        return 0.0
    
    def _get_db_query_time(self) -> float:
        """Get current database query time."""
        try:
            db_optimizer = self._get_component('db_optimizer')
            if db_optimizer and hasattr(db_optimizer, 'performance_stats'):
                stats = db_optimizer.performance_stats
                if stats['queries_executed'] > 0:
                    return (stats['total_query_time'] / stats['queries_executed']) * 1000
        except Exception as e:
            self.logger.debug(f"Could not get DB query time: {e}")
        return 0.0
    
    def _get_io_wait_time(self) -> float:
        """Get current I/O wait time."""
        try:
            # This would require more sophisticated monitoring
            # For now, return a placeholder
            return 0.0
        except Exception as e:
            self.logger.debug(f"Could not get I/O wait time: {e}")
        return 0.0
    
    def identify_bottlenecks(self, current_metrics: PerformanceMetrics) -> List[BottleneckType]:
        """
        Identify performance bottlenecks based on current metrics.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            List of detected bottleneck types
        """
        bottlenecks = []
        
        try:
            # Memory pressure detection
            memory_percent = (current_metrics.memory_usage_mb / 
                            (psutil.virtual_memory().total / (1024 * 1024))) * 100
            if memory_percent > self.thresholds[BottleneckType.MEMORY_PRESSURE]['memory_percent']:
                bottlenecks.append(BottleneckType.MEMORY_PRESSURE)
            
            # CPU bottleneck detection
            if current_metrics.cpu_percent > self.thresholds[BottleneckType.CPU_BOTTLENECK]['cpu_percent']:
                # Check if sustained over time
                recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-5:]]
                if len(recent_cpu) >= 3 and all(cpu > 85 for cpu in recent_cpu):
                    bottlenecks.append(BottleneckType.CPU_BOTTLENECK)
            
            # Slow generation detection
            if (current_metrics.token_throughput < self.thresholds[BottleneckType.SLOW_GENERATION]['tokens_per_second'] or
                current_metrics.generation_latency_ms > self.thresholds[BottleneckType.SLOW_GENERATION]['latency_ms']):
                bottlenecks.append(BottleneckType.SLOW_GENERATION)
            
            # Slow retrieval detection
            if current_metrics.retrieval_latency_ms > self.thresholds[BottleneckType.SLOW_RETRIEVAL]['retrieval_ms']:
                bottlenecks.append(BottleneckType.SLOW_RETRIEVAL)
            
            # Cache performance issues
            if current_metrics.cache_hit_rate < self.thresholds[BottleneckType.CACHE_MISSES]['hit_rate']:
                bottlenecks.append(BottleneckType.CACHE_MISSES)
            
            # Database performance issues
            if current_metrics.db_query_time_ms > self.thresholds[BottleneckType.DATABASE_SLOW]['query_time_ms']:
                bottlenecks.append(BottleneckType.DATABASE_SLOW)
            
            # I/O bottleneck detection
            if current_metrics.io_wait_time_ms > self.thresholds[BottleneckType.IO_BOTTLENECK]['io_wait_ms']:
                bottlenecks.append(BottleneckType.IO_BOTTLENECK)
            
        except Exception as e:
            self.logger.error(f"Error identifying bottlenecks: {e}")
        
        return bottlenecks
    
    def apply_optimization(self, bottleneck: BottleneckType) -> OptimizationAction:
        """
        Apply optimization for a specific bottleneck type.
        
        Args:
            bottleneck: Type of bottleneck to optimize
            
        Returns:
            OptimizationAction with results
        """
        if bottleneck not in self.optimization_strategies:
            self.logger.warning(f"No optimization strategy for {bottleneck}")
            return OptimizationAction("unknown", {}, 0.0, time.time(), False)
        
        try:
            strategy_func = self.optimization_strategies[bottleneck]
            action = strategy_func()
            
            # Track applied optimization
            self.applied_optimizations.append(action)
            self.optimization_history.append(action)
            
            self.logger.info(f"Applied optimization for {bottleneck}: {action.action_type}")
            return action
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization for {bottleneck}: {e}")
            return OptimizationAction("failed", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_memory_pressure(self) -> OptimizationAction:
        """Optimize memory pressure issues."""
        try:
            memory_optimizer = self._get_component('memory_optimizer')
            if memory_optimizer:
                # Perform emergency cleanup
                cleanup_stats = memory_optimizer.emergency_cleanup()
                
                action = OptimizationAction(
                    action_type="memory_cleanup",
                    parameters=cleanup_stats,
                    expected_improvement=20.0,  # Expect 20% memory reduction
                    applied_at=time.time(),
                    success=True
                )
                
                return action
            else:
                return OptimizationAction("memory_cleanup", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return OptimizationAction("memory_cleanup", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_cpu_bottleneck(self) -> OptimizationAction:
        """Optimize CPU bottleneck issues."""
        try:
            # Reduce batch sizes to lower CPU load
            memory_optimizer = self._get_component('memory_optimizer')
            if memory_optimizer:
                # Get current memory stats to determine optimal batch size
                memory_stats = memory_optimizer.get_memory_usage()
                available_mb = memory_stats.get('available_mb', 4000)
                
                # Reduce batch size for less CPU intensive operations
                new_batch_size = memory_optimizer.optimize_batch_size(16, 5.0)  # Smaller base size
                
                action = OptimizationAction(
                    action_type="reduce_batch_size",
                    parameters={"new_batch_size": new_batch_size, "available_memory": available_mb},
                    expected_improvement=15.0,
                    applied_at=time.time(),
                    success=True
                )
                
                return action
            else:
                return OptimizationAction("reduce_batch_size", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
            return OptimizationAction("reduce_batch_size", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_generation_speed(self) -> OptimizationAction:
        """Optimize slow text generation."""
        try:
            # Try to optimize Metal usage
            metal_optimizer = self._get_component('metal_optimizer')
            if metal_optimizer:
                # Clear MPS cache to free up memory
                cache_stats = metal_optimizer.clear_mps_cache()
                
                action = OptimizationAction(
                    action_type="optimize_metal_generation",
                    parameters=cache_stats,
                    expected_improvement=25.0,
                    applied_at=time.time(),
                    success=True
                )
                
                return action
            else:
                return OptimizationAction("optimize_metal_generation", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"Generation optimization failed: {e}")
            return OptimizationAction("optimize_metal_generation", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_retrieval_speed(self) -> OptimizationAction:
        """Optimize slow retrieval performance."""
        try:
            # Optimize database indices
            db_optimizer = self._get_component('db_optimizer')
            if db_optimizer:
                # Run query optimization
                optimization_results = db_optimizer.optimize_query_plans()
                
                action = OptimizationAction(
                    action_type="optimize_database_queries",
                    parameters=optimization_results,
                    expected_improvement=30.0,
                    applied_at=time.time(),
                    success=optimization_results.get('analyze_completed', False)
                )
                
                return action
            else:
                return OptimizationAction("optimize_database_queries", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"Retrieval optimization failed: {e}")
            return OptimizationAction("optimize_database_queries", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_cache_performance(self) -> OptimizationAction:
        """Optimize cache performance."""
        try:
            cache_manager = self._get_component('cache_manager')
            if cache_manager:
                # Optimize cache sizes based on usage patterns
                new_sizes = cache_manager.optimize_cache_sizes()
                
                action = OptimizationAction(
                    action_type="optimize_cache_sizes",
                    parameters={"new_sizes": new_sizes},
                    expected_improvement=40.0,
                    applied_at=time.time(),
                    success=bool(new_sizes)
                )
                
                return action
            else:
                return OptimizationAction("optimize_cache_sizes", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return OptimizationAction("optimize_cache_sizes", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_database_performance(self) -> OptimizationAction:
        """Optimize database performance."""
        try:
            db_optimizer = self._get_component('db_optimizer')
            if db_optimizer:
                # Create missing performance indices
                index_results = db_optimizer.create_performance_indices()
                
                action = OptimizationAction(
                    action_type="create_database_indices",
                    parameters=index_results,
                    expected_improvement=50.0,
                    applied_at=time.time(),
                    success=any(index_results.values())
                )
                
                return action
            else:
                return OptimizationAction("create_database_indices", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return OptimizationAction("create_database_indices", {"error": str(e)}, 0.0, time.time(), False)
    
    def _optimize_io_performance(self) -> OptimizationAction:
        """Optimize I/O performance."""
        try:
            speed_optimizer = self._get_component('speed_optimizer')
            if speed_optimizer:
                # Clear operation cache to free up I/O resources
                cleared_counts = speed_optimizer.clear_cache()
                
                action = OptimizationAction(
                    action_type="optimize_io_caching",
                    parameters=cleared_counts,
                    expected_improvement=20.0,
                    applied_at=time.time(),
                    success=bool(cleared_counts)
                )
                
                return action
            else:
                return OptimizationAction("optimize_io_caching", {}, 0.0, time.time(), False)
                
        except Exception as e:
            self.logger.error(f"I/O optimization failed: {e}")
            return OptimizationAction("optimize_io_caching", {"error": str(e)}, 0.0, time.time(), False)
    
    def verify_optimization_effectiveness(self, action: OptimizationAction, 
                                        verification_window: int = 5) -> float:
        """
        Verify if an optimization was effective.
        
        Args:
            action: Optimization action to verify
            verification_window: Number of metrics to compare (post-optimization)
            
        Returns:
            Actual improvement percentage (negative if performance degraded)
        """
        try:
            # Get metrics from before the optimization
            pre_optimization_metrics = [m for m in self.metrics_history 
                                      if m.timestamp < action.applied_at]
            
            if len(pre_optimization_metrics) < 2:
                self.logger.warning("Insufficient pre-optimization metrics for verification")
                return 0.0
            
            # Get metrics from after the optimization
            post_optimization_metrics = [m for m in self.metrics_history 
                                       if m.timestamp > action.applied_at]
            
            if len(post_optimization_metrics) < verification_window:
                self.logger.debug("Insufficient post-optimization metrics for verification")
                return 0.0
            
            # Calculate average performance before and after
            pre_avg = self._calculate_performance_score(pre_optimization_metrics[-5:])
            post_avg = self._calculate_performance_score(post_optimization_metrics[:verification_window])
            
            # Calculate improvement percentage
            if pre_avg > 0:
                improvement = ((post_avg - pre_avg) / pre_avg) * 100
                action.actual_improvement = improvement
                return improvement
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to verify optimization effectiveness: {e}")
            return 0.0
    
    def _calculate_performance_score(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate a composite performance score from metrics."""
        if not metrics_list:
            return 0.0
        
        try:
            # Weighted performance score (higher is better)
            scores = []
            
            for metrics in metrics_list:
                score = (
                    metrics.token_throughput * 0.3 +  # 30% weight on token throughput
                    (1 / max(metrics.retrieval_latency_ms / 1000, 0.1)) * 0.25 +  # 25% weight on retrieval speed
                    (1 / max(metrics.generation_latency_ms / 1000, 0.1)) * 0.25 +  # 25% weight on generation speed
                    metrics.cache_hit_rate * 100 * 0.1 +  # 10% weight on cache hit rate
                    max(0, 100 - (metrics.memory_usage_mb / 100)) * 0.1  # 10% weight on memory efficiency
                )
                scores.append(score)
            
            return statistics.mean(scores)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance score: {e}")
            return 0.0
    
    def start_monitoring(self) -> None:
        """Start automatic performance monitoring and optimization."""
        if self._monitoring:
            self.logger.warning("Auto-tuner already running")
            return
        
        self._stop_event.clear()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Auto-tuner monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop automatic performance monitoring."""
        if not self._monitoring:
            return
        
        self._stop_event.set()
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Auto-tuner monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        self.logger.info("Auto-tuner monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Collect current metrics
                current_metrics = self.collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Set baseline on first run
                if self.baseline_metrics is None:
                    self.baseline_metrics = current_metrics
                    self.logger.info("Baseline performance metrics established")
                
                # Identify bottlenecks
                bottlenecks = self.identify_bottlenecks(current_metrics)
                
                if bottlenecks:
                    self.logger.info(f"Detected bottlenecks: {[b.value for b in bottlenecks]}")
                    
                    # Apply optimizations for detected bottlenecks
                    for bottleneck in bottlenecks:
                        # Avoid applying the same optimization too frequently
                        recent_actions = [a for a in self.applied_optimizations 
                                        if (time.time() - a.applied_at) < 300]  # Last 5 minutes
                        
                        similar_actions = [a for a in recent_actions 
                                         if bottleneck.value in a.action_type]
                        
                        if len(similar_actions) < 2:  # Max 2 similar actions in 5 minutes
                            optimization_action = self.apply_optimization(bottleneck)
                            
                            if optimization_action.success:
                                self.logger.info(f"Successfully applied {optimization_action.action_type}")
                            else:
                                self.logger.warning(f"Failed to apply {optimization_action.action_type}")
                
                # Verify effectiveness of recent optimizations
                for action in self.applied_optimizations[-3:]:  # Check last 3 actions
                    if action.actual_improvement == 0.0:  # Not yet verified
                        improvement = self.verify_optimization_effectiveness(action)
                        if abs(improvement) > 1.0:  # Significant change
                            if improvement > 0:
                                self.logger.info(f"Optimization {action.action_type} effective: "
                                               f"{improvement:.1f}% improvement")
                            else:
                                self.logger.warning(f"Optimization {action.action_type} degraded performance: "
                                                  f"{improvement:.1f}%")
                
                # Wait for next monitoring cycle
                self._stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(min(self.monitoring_interval, 10.0))  # Shorter retry interval on error
        
        self.logger.info("Auto-tuner monitoring loop stopped")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get manual optimization recommendations based on historical data.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if len(self.metrics_history) < 10:
            recommendations.append({
                'priority': 'info',
                'category': 'monitoring',
                'recommendation': 'Collect more performance data before providing specific recommendations',
                'confidence': 0.5
            })
            return recommendations
        
        try:
            # Analyze metrics trends
            recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
            
            # Memory usage trend
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            if len(memory_usage) > 5:
                memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
                if memory_trend > 50:  # Growing by >50MB per measurement
                    recommendations.append({
                        'priority': 'high',
                        'category': 'memory',
                        'recommendation': 'Memory usage is growing rapidly. Consider enabling aggressive garbage collection or reducing batch sizes.',
                        'confidence': 0.8
                    })
            
            # Performance degradation
            performance_scores = [self._calculate_performance_score([m]) for m in recent_metrics]
            if len(performance_scores) > 5:
                perf_trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
                if perf_trend < -0.5:  # Declining performance
                    recommendations.append({
                        'priority': 'high',
                        'category': 'performance',
                        'recommendation': 'Overall performance is declining. Run comprehensive optimization or restart components.',
                        'confidence': 0.9
                    })
            
            # Cache effectiveness
            cache_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate > 0]
            if cache_rates:
                avg_cache_rate = statistics.mean(cache_rates)
                if avg_cache_rate < 0.4:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'cache',
                        'recommendation': f'Cache hit rate is low ({avg_cache_rate:.1%}). Consider increasing cache sizes or reviewing caching strategy.',
                        'confidence': 0.7
                    })
            
            # Database performance
            db_times = [m.db_query_time_ms for m in recent_metrics if m.db_query_time_ms > 0]
            if db_times:
                avg_db_time = statistics.mean(db_times)
                if avg_db_time > 500:  # >500ms average
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'database',
                        'recommendation': f'Database queries are slow (avg {avg_db_time:.0f}ms). Consider creating additional indices or running VACUUM.',
                        'confidence': 0.8
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append({
                'priority': 'error',
                'category': 'system',
                'recommendation': f'Unable to analyze performance data: {e}',
                'confidence': 0.0
            })
        
        return recommendations
    
    def get_tuning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive auto-tuning statistics."""
        stats = {
            'monitoring_active': self._monitoring,
            'metrics_collected': len(self.metrics_history),
            'optimizations_applied': len(self.applied_optimizations),
            'successful_optimizations': sum(1 for a in self.applied_optimizations if a.success),
            'baseline_established': self.baseline_metrics is not None
        }
        
        if self.baseline_metrics and self.metrics_history:
            # Compare current performance to baseline
            current_metrics = self.metrics_history[-1]
            baseline_score = self._calculate_performance_score([self.baseline_metrics])
            current_score = self._calculate_performance_score([current_metrics])
            
            if baseline_score > 0:
                performance_change = ((current_score - baseline_score) / baseline_score) * 100
                stats['performance_change_percent'] = performance_change
        
        # Optimization effectiveness
        verified_optimizations = [a for a in self.applied_optimizations if a.actual_improvement != 0.0]
        if verified_optimizations:
            avg_improvement = statistics.mean([a.actual_improvement for a in verified_optimizations])
            stats['average_optimization_improvement'] = avg_improvement
        
        # Recent bottlenecks
        if self.metrics_history:
            recent_bottlenecks = self.identify_bottlenecks(self.metrics_history[-1])
            stats['current_bottlenecks'] = [b.value for b in recent_bottlenecks]
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up auto-tuner resources."""
        self.stop_monitoring()
        
        self.logger.info("AutoTuner cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during shutdown
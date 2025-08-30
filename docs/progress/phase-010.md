# Phase 10: Performance Optimization Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Review system performance baseline from Phase 8:
```bash
cat handoff/phase_8_complete.json | jq '.baseline_metrics'
```

## Your Mission
Optimize the RAG system for maximum performance on the Mac mini M4 with 16GB RAM. Focus on memory efficiency, speed improvements, and resource utilization.

## Prerequisites Check
1. Review current benchmarks: `python run_benchmarks.py --quick`
2. Check current memory usage: `python scripts/doctor.py --memory`
3. Profile current bottlenecks: `pip install py-spy; sudo py-spy record -o profile.svg -- python main.py query "test"`

## Implementation Tasks

### Task 10.1: Memory Optimization
Create `src/optimizations/memory_optimizer.py`:

```python
# Memory optimization strategies:
# 1. Dynamic batch sizing
# 2. Memory-mapped model loading
# 3. Aggressive garbage collection
# 4. Component unloading
# 5. Cache management
```

Implementation:
```python
import gc
import mmap
import weakref
from functools import lru_cache
import numpy as np

class MemoryOptimizer:
    def __init__(self, target_memory_mb=12000):
        self.target_memory = target_memory_mb
        self.component_registry = weakref.WeakValueDictionary()
        
    def optimize_batch_size(self, available_memory):
        """Dynamically adjust batch sizes"""
        # Calculate optimal batch size based on available memory
        if available_memory < 2000:  # <2GB
            return 8
        elif available_memory < 4000:  # <4GB
            return 16
        else:
            return 32
            
    def enable_memory_mapping(self, model_path):
        """Use mmap for model loading"""
        # Memory-mapped file access
        with open(model_path, 'rb') as f:
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
    def aggressive_gc(self):
        """Force garbage collection"""
        gc.collect()
        gc.collect()  # Second pass for circular refs
        
    def unload_component(self, name):
        """Unload unused components"""
        if name in self.component_registry:
            del self.component_registry[name]
            self.aggressive_gc()
            
    @lru_cache(maxsize=1000)
    def cached_computation(self, key):
        """LRU cache for expensive computations"""
        pass
```

Memory optimization targets:
- Model loading: Use mmap to reduce RAM copy
- Embedding cache: LRU with size limit
- Chunk processing: Stream instead of batch
- Vector index: Use disk-backed index for large corpus
- Context window: Dynamic sizing based on available memory

### Task 10.2: Speed Optimization
Create `src/optimizations/speed_optimizer.py`:

```python
# Speed optimization techniques:
# 1. Vectorized operations
# 2. Async I/O
# 3. Caching strategies
# 4. Query optimization
# 5. Parallel processing
```

Speed improvements:
```python
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import pickle

class SpeedOptimizer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        
    def vectorize_embeddings(self, texts):
        """Batch embedding computation"""
        # Use numpy for vectorized operations
        # Process in optimal batch sizes
        
    async def async_document_processing(self, documents):
        """Async I/O for document loading"""
        tasks = [self.process_doc_async(doc) for doc in documents]
        return await asyncio.gather(*tasks)
        
    def cache_results(self, func):
        """Decorator for caching expensive operations"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = pickle.dumps((args, kwargs))
            if key not in self.cache:
                self.cache[key] = func(*args, **kwargs)
            return self.cache[key]
        return wrapper
        
    def optimize_sql_queries(self, query):
        """SQL query optimization"""
        # Add indices
        # Use prepared statements
        # Batch operations
```

### Task 10.3: Metal/MPS Optimization
Create `src/optimizations/metal_optimizer.py`:

```python
# Apple Silicon specific optimizations:
# 1. MPS tensor operations
# 2. Unified memory utilization
# 3. Neural Engine usage (where possible)
# 4. Metal Performance Shaders
```

Metal optimizations:
```python
import torch
import torch.nn as nn

class MetalOptimizer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def optimize_embedding_model(self, model):
        """Move model to MPS and optimize"""
        model = model.to(self.device)
        model.eval()  # Disable dropout
        
        # Enable memory efficient attention if available
        if hasattr(model, 'enable_mem_efficient_attention'):
            model.enable_mem_efficient_attention()
            
        return model
        
    def optimize_tensor_operations(self, tensors):
        """Ensure tensors use MPS"""
        return [t.to(self.device) for t in tensors]
        
    def profile_mps_usage(self):
        """Profile MPS utilization"""
        if torch.backends.mps.is_available():
            # Get MPS statistics
            return torch.mps.current_allocated_memory()
```

### Task 10.4: Database Optimization
Create `src/optimizations/db_optimizer.py`:

```python
# Database performance improvements:
# 1. Index optimization
# 2. Query planning
# 3. Connection pooling
# 4. WAL mode
# 5. Vacuum and analyze
```

Database optimizations:
```python
import sqlite3
from contextlib import contextmanager

class DatabaseOptimizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.optimize_settings()
        
    def optimize_settings(self):
        """Apply optimal SQLite settings"""
        with self.get_connection() as conn:
            # Enable WAL mode for concurrent reads
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Increase cache size (in KB)
            conn.execute("PRAGMA cache_size=-64000")  # 64MB
            
            # Memory-mapped I/O
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            # Synchronous mode
            conn.execute("PRAGMA synchronous=NORMAL")
            
    def create_indices(self):
        """Create optimal indices"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)",
        ]
        
    def vacuum_database(self):
        """Defragment and optimize storage"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
```

### Task 10.5: Caching Strategy
Create `src/optimizations/cache_manager.py`:

```python
# Multi-level caching:
# 1. Query cache
# 2. Embedding cache
# 3. Result cache
# 4. Model output cache
```

Cache implementation:
```python
from collections import OrderedDict
import hashlib
import json

class CacheManager:
    def __init__(self, max_memory_mb=500):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_size = 0
        self.caches = {
            'query': LRUCache(maxsize=100),
            'embedding': LRUCache(maxsize=1000),
            'retrieval': LRUCache(maxsize=50),
            'generation': LRUCache(maxsize=20)
        }
        
    def get_or_compute(self, cache_type, key, compute_func):
        """Get from cache or compute"""
        cache = self.caches[cache_type]
        
        if key in cache:
            return cache[key]
            
        result = compute_func()
        cache[key] = result
        return result
        
    def clear_cache(self, cache_type=None):
        """Clear specific or all caches"""
        if cache_type:
            self.caches[cache_type].clear()
        else:
            for cache in self.caches.values():
                cache.clear()
```

### Task 10.6: Monitoring and Auto-tuning
Create `src/optimizations/auto_tuner.py`:

```python
# Automatic performance tuning:
# 1. Monitor metrics
# 2. Identify bottlenecks
# 3. Apply optimizations
# 4. Verify improvements
```

Auto-tuning implementation:
```python
class AutoTuner:
    def __init__(self, system_manager):
        self.system = system_manager
        self.metrics_history = []
        
    def collect_metrics(self):
        """Gather performance metrics"""
        return {
            'memory_usage': self.get_memory_usage(),
            'token_throughput': self.measure_throughput(),
            'retrieval_latency': self.measure_retrieval(),
            'cache_hit_rate': self.get_cache_stats()
        }
        
    def identify_bottleneck(self, metrics):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if metrics['memory_usage'] > 0.9:
            bottlenecks.append('memory')
        if metrics['token_throughput'] < 10:
            bottlenecks.append('generation')
        if metrics['retrieval_latency'] > 100:
            bottlenecks.append('retrieval')
            
        return bottlenecks
        
    def apply_optimization(self, bottleneck):
        """Apply targeted optimization"""
        strategies = {
            'memory': self.optimize_memory,
            'generation': self.optimize_generation,
            'retrieval': self.optimize_retrieval
        }
        
        if bottleneck in strategies:
            strategies[bottleneck]()
```

## Testing Requirements
Create `test_phase_10.py`:
1. Benchmark memory usage reduction
2. Measure speed improvements
3. Test cache effectiveness
4. Verify Metal utilization
5. Compare before/after metrics

## Output Requirements
Create `handoff/phase_10_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 10,
  "created_files": [
    "src/optimizations/memory_optimizer.py",
    "src/optimizations/speed_optimizer.py",
    "src/optimizations/metal_optimizer.py",
    "src/optimizations/db_optimizer.py",
    "src/optimizations/cache_manager.py",
    "src/optimizations/auto_tuner.py",
    "test_phase_10.py"
  ],
  "optimizations_applied": {
    "memory_mapping": true,
    "dynamic_batching": true,
    "mps_acceleration": true,
    "database_indices": true,
    "caching_layers": true,
    "auto_tuning": true
  },
  "performance_improvements": {
    "memory_reduction_percent": 0.0,
    "speed_increase_percent": 0.0,
    "cache_hit_rate": 0.0,
    "metal_utilization": 0.0
  },
  "before_metrics": {
    "tokens_per_second": 0.0,
    "memory_usage_mb": 0.0,
    "retrieval_latency_ms": 0.0
  },
  "after_metrics": {
    "tokens_per_second": 0.0,
    "memory_usage_mb": 0.0,
    "retrieval_latency_ms": 0.0
  },
  "optimization_recommendations": [
    "Consider smaller embedding model for memory constraints",
    "Enable query caching for repeated questions",
    "Use async I/O for document processing"
  ]
}
```

## Performance Targets
Achieve these optimization goals:
- **Memory**: Reduce usage by 30%
- **Speed**: Increase throughput by 25%
- **Latency**: Reduce first-token latency by 40%
- **Cache**: Achieve 60% hit rate
- **Stability**: Zero OOM errors under normal load

## Optimization Checklist
- [ ] Memory-mapped model loading implemented
- [ ] Dynamic batch sizing works
- [ ] MPS/Metal fully utilized
- [ ] Database indices created
- [ ] Caching system operational
- [ ] Auto-tuner identifies bottlenecks
- [ ] Benchmarks show improvements
- [ ] System remains stable
- [ ] Documentation updated
- [ ] Handoff file created

## Final System Validation
Run complete system test:
```bash
# Full benchmark suite
python run_benchmarks.py --full

# Stress test
python benchmarks/stress_test.py --duration 600

# Memory leak test
python scripts/memory_leak_test.py

# Generate final report
python scripts/generate_report.py --format html > final_report.html
```

Remember: This is the final optimization phase. The system should now be ready for production use with optimal performance on your Mac mini M4!
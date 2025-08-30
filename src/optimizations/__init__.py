"""
Performance optimization components for the RAG system.

This module contains optimizations specifically designed for Apple Silicon M4
with 16GB RAM, focusing on memory efficiency, speed improvements, and Metal acceleration.
"""

from .memory_optimizer import MemoryOptimizer
from .speed_optimizer import SpeedOptimizer  
from .metal_optimizer import MetalOptimizer
from .db_optimizer import DatabaseOptimizer
from .cache_manager import CacheManager
from .auto_tuner import AutoTuner

__all__ = [
    'MemoryOptimizer',
    'SpeedOptimizer', 
    'MetalOptimizer',
    'DatabaseOptimizer',
    'CacheManager',
    'AutoTuner'
]
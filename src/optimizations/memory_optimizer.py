"""
Memory optimization strategies for the RAG system.

Designed for Apple Silicon M4 with 16GB RAM, focusing on:
1. Dynamic batch sizing based on available memory
2. Memory-mapped model loading to reduce RAM usage
3. Aggressive garbage collection for unused objects
4. Component unloading when not needed
5. Intelligent cache management with size limits
"""

import gc
import mmap
import weakref
import psutil
import logging
import numpy as np
from functools import lru_cache
from typing import Dict, Any, Optional, Union
from pathlib import Path


class MemoryOptimizer:
    """
    Manages memory usage optimization for the RAG system.
    
    Target: Reduce memory usage from ~8.5GB to ~6GB (30% reduction)
    """
    
    def __init__(self, target_memory_mb: int = 12000, min_free_memory_mb: int = 2000):
        """
        Initialize memory optimizer.
        
        Args:
            target_memory_mb: Target maximum memory usage in MB
            min_free_memory_mb: Minimum free memory to maintain in MB
        """
        self.target_memory = target_memory_mb
        self.min_free_memory = min_free_memory_mb
        self.component_registry = weakref.WeakValueDictionary()
        self.memory_maps = {}
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        
        self.logger.info(f"MemoryOptimizer initialized - Target: {target_memory_mb}MB, "
                        f"System RAM: {self.system_memory:.0f}MB")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': self.process.memory_percent(),
                'available_mb': system_memory.available / (1024 * 1024),
                'system_percent': system_memory.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def optimize_batch_size(self, base_batch_size: int = 32, 
                          memory_per_item_mb: float = 10.0) -> int:
        """
        Dynamically adjust batch sizes based on available memory.
        
        Args:
            base_batch_size: Base batch size to start from
            memory_per_item_mb: Estimated memory usage per batch item
            
        Returns:
            Optimized batch size
        """
        memory_stats = self.get_memory_usage()
        available_mb = memory_stats.get('available_mb', 4000)
        
        # Calculate maximum batch size that fits in available memory
        # Reserve some memory for overhead
        usable_memory = max(available_mb - self.min_free_memory, 500)
        max_batch_size = int(usable_memory / memory_per_item_mb)
        
        # Apply constraints
        optimized_size = min(max_batch_size, base_batch_size * 2)  # Don't go too large
        optimized_size = max(optimized_size, 4)  # Minimum viable batch size
        
        if optimized_size != base_batch_size:
            self.logger.info(f"Batch size optimized: {base_batch_size} → {optimized_size} "
                           f"(Available: {available_mb:.0f}MB)")
        
        return optimized_size
    
    def enable_memory_mapping(self, file_path: Union[str, Path]) -> Optional[mmap.mmap]:
        """
        Create memory-mapped file access to reduce RAM usage.
        
        Args:
            file_path: Path to file to memory map
            
        Returns:
            Memory-mapped file object or None if failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.warning(f"File not found for memory mapping: {file_path}")
                return None
            
            # Close existing mapping if present
            if str(file_path) in self.memory_maps:
                self.close_memory_mapping(file_path)
            
            with open(file_path, 'rb') as f:
                memory_map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.memory_maps[str(file_path)] = memory_map
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Memory mapped {file_path.name}: {file_size_mb:.1f}MB")
                
                return memory_map
                
        except Exception as e:
            self.logger.error(f"Failed to create memory mapping for {file_path}: {e}")
            return None
    
    def close_memory_mapping(self, file_path: Union[str, Path]) -> None:
        """Close a memory-mapped file."""
        file_key = str(file_path)
        if file_key in self.memory_maps:
            try:
                self.memory_maps[file_key].close()
                del self.memory_maps[file_key]
                self.logger.debug(f"Closed memory mapping for {file_path}")
            except Exception as e:
                self.logger.error(f"Error closing memory mapping: {e}")
    
    def aggressive_gc(self, force_full: bool = False) -> Dict[str, int]:
        """
        Perform aggressive garbage collection.
        
        Args:
            force_full: Force full garbage collection cycle
            
        Returns:
            Statistics about garbage collection
        """
        memory_before = self.get_memory_usage().get('rss_mb', 0)
        
        # Standard garbage collection
        collected_0 = gc.collect(0)  # Young generation
        collected_1 = gc.collect(1)  # Middle generation  
        collected_2 = gc.collect(2)  # Old generation
        
        if force_full:
            # Additional full collection pass for circular references
            gc.collect()
            
        memory_after = self.get_memory_usage().get('rss_mb', 0)
        memory_freed = memory_before - memory_after
        
        stats = {
            'collected_gen0': collected_0,
            'collected_gen1': collected_1, 
            'collected_gen2': collected_2,
            'memory_freed_mb': memory_freed
        }
        
        if memory_freed > 10:  # Only log significant memory recovery
            self.logger.info(f"Garbage collection freed {memory_freed:.1f}MB")
            
        return stats
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for lifecycle management."""
        self.component_registry[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    def unload_component(self, name: str) -> bool:
        """
        Unload a component to free memory.
        
        Args:
            name: Name of component to unload
            
        Returns:
            True if component was unloaded
        """
        if name in self.component_registry:
            try:
                component = self.component_registry[name]
                
                # Call cleanup methods if available
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                if hasattr(component, 'close'):
                    component.close()
                if hasattr(component, '__del__'):
                    component.__del__()
                
                # Remove from registry
                del self.component_registry[name]
                
                # Force garbage collection
                self.aggressive_gc()
                
                self.logger.info(f"Unloaded component: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error unloading component {name}: {e}")
                
        return False
    
    def get_registered_components(self) -> list:
        """Get list of currently registered components."""
        return list(self.component_registry.keys())
    
    def optimize_numpy_arrays(self, arrays: list) -> list:
        """
        Optimize numpy arrays for memory usage.
        
        Args:
            arrays: List of numpy arrays to optimize
            
        Returns:
            List of optimized arrays
        """
        optimized = []
        
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                optimized.append(arr)
                continue
                
            original_dtype = arr.dtype
            original_size = arr.nbytes / (1024 * 1024)  # MB
            
            # Try to use more efficient data types
            if arr.dtype == np.float64:
                # Use float32 if precision loss is acceptable
                optimized_arr = arr.astype(np.float32)
            elif arr.dtype == np.int64:
                # Use smaller integer types if possible
                if arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max:
                    optimized_arr = arr.astype(np.int32)
                elif arr.min() >= np.iinfo(np.int16).min and arr.max() <= np.iinfo(np.int16).max:
                    optimized_arr = arr.astype(np.int16)
                else:
                    optimized_arr = arr
            else:
                optimized_arr = arr
            
            new_size = optimized_arr.nbytes / (1024 * 1024)  # MB
            
            if new_size < original_size:
                self.logger.debug(f"Array optimization: {original_dtype} → {optimized_arr.dtype}, "
                                f"{original_size:.2f}MB → {new_size:.2f}MB")
            
            optimized.append(optimized_arr)
            
        return optimized
    
    @lru_cache(maxsize=128)
    def cached_computation(self, cache_key: str, *args) -> Any:
        """
        LRU cache for expensive computations.
        
        Args:
            cache_key: Unique key for the computation
            *args: Arguments for the computation
            
        Returns:
            Cached result
        """
        # This is a placeholder - actual implementations will override
        pass
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.cached_computation.cache_clear()
        self.logger.info("Cleared optimization caches")
    
    def memory_pressure_check(self) -> Dict[str, Union[bool, float]]:
        """
        Check if system is under memory pressure.
        
        Returns:
            Dictionary with pressure indicators
        """
        memory_stats = self.get_memory_usage()
        system_memory = psutil.virtual_memory()
        
        # Calculate pressure indicators
        rss_mb = memory_stats.get('rss_mb', 0)
        available_mb = memory_stats.get('available_mb', 0)
        system_percent = memory_stats.get('system_percent', 0)
        
        high_usage = rss_mb > self.target_memory
        low_available = available_mb < self.min_free_memory
        system_pressure = system_percent > 85
        
        pressure = {
            'high_usage': high_usage,
            'low_available': low_available, 
            'system_pressure': system_pressure,
            'any_pressure': high_usage or low_available or system_pressure,
            'rss_mb': rss_mb,
            'available_mb': available_mb,
            'system_percent': system_percent
        }
        
        if pressure['any_pressure']:
            self.logger.warning(f"Memory pressure detected: {pressure}")
            
        return pressure
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency memory cleanup when under pressure.
        
        Returns:
            Statistics about cleanup actions
        """
        self.logger.warning("Initiating emergency memory cleanup")
        
        initial_memory = self.get_memory_usage().get('rss_mb', 0)
        
        # 1. Clear caches
        self.clear_caches()
        
        # 2. Aggressive garbage collection
        gc_stats = self.aggressive_gc(force_full=True)
        
        # 3. Close memory mappings for non-critical files
        mappings_closed = 0
        for file_path in list(self.memory_maps.keys()):
            if 'temp' in file_path.lower() or 'cache' in file_path.lower():
                self.close_memory_mapping(file_path)
                mappings_closed += 1
        
        # 4. Unload non-essential components
        non_essential = ['cache', 'monitor', 'analytics']
        unloaded = []
        for component in non_essential:
            if self.unload_component(component):
                unloaded.append(component)
        
        final_memory = self.get_memory_usage().get('rss_mb', 0)
        memory_freed = initial_memory - final_memory
        
        cleanup_stats = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': memory_freed,
            'gc_stats': gc_stats,
            'mappings_closed': mappings_closed,
            'components_unloaded': unloaded
        }
        
        self.logger.info(f"Emergency cleanup completed: {memory_freed:.1f}MB freed")
        return cleanup_stats
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        memory_stats = self.get_memory_usage()
        pressure_stats = self.memory_pressure_check()
        
        return {
            'memory_usage': memory_stats,
            'memory_pressure': pressure_stats,
            'active_mappings': len(self.memory_maps),
            'registered_components': len(self.component_registry),
            'cache_info': self.cached_computation.cache_info()._asdict(),
            'target_memory_mb': self.target_memory,
            'min_free_memory_mb': self.min_free_memory
        }
    
    def cleanup(self) -> None:
        """Cleanup resources when optimizer is destroyed."""
        # Close all memory mappings
        for file_path in list(self.memory_maps.keys()):
            self.close_memory_mapping(file_path)
        
        # Clear caches
        self.clear_caches()
        
        # Final garbage collection
        self.aggressive_gc()
        
        self.logger.info("MemoryOptimizer cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during shutdown
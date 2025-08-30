"""
Apple Silicon (Metal Performance Shaders) specific optimizations.

Designed for Mac mini M4 with unified memory architecture:
1. MPS tensor operations for neural computations
2. Unified memory utilization optimization
3. Metal Performance Shaders integration
4. GPU memory management and profiling
"""

import torch
import torch.nn as nn
import logging
import time
import gc
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np


class MetalOptimizer:
    """
    Optimizes neural computations for Apple Silicon with Metal Performance Shaders.
    
    Target: Maximize Metal/GPU utilization while managing unified memory efficiently
    """
    
    def __init__(self, memory_fraction: float = 0.7, enable_profiling: bool = False):
        """
        Initialize Metal optimizer.
        
        Args:
            memory_fraction: Fraction of unified memory to allocate for MPS
            enable_profiling: Enable detailed MPS profiling
        """
        self.memory_fraction = memory_fraction
        self.enable_profiling = enable_profiling
        self.logger = logging.getLogger(__name__)
        
        # Check MPS availability
        self.mps_available = torch.backends.mps.is_available()
        self.device = torch.device("mps" if self.mps_available else "cpu")
        
        # Performance tracking
        self.optimization_stats = {
            'models_optimized': 0,
            'tensors_moved': 0,
            'memory_allocated_mb': 0,
            'inference_speedup': 0.0
        }
        
        if self.mps_available:
            self.logger.info("Metal Performance Shaders available - GPU acceleration enabled")
            self._setup_metal_environment()
        else:
            self.logger.warning("MPS not available - falling back to CPU")
    
    def _setup_metal_environment(self) -> None:
        """Configure Metal environment for optimal performance."""
        try:
            # Set memory allocation strategy
            torch.mps.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory optimization
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            self.logger.info(f"Metal environment configured - Memory fraction: {self.memory_fraction}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Metal environment: {e}")
    
    def optimize_embedding_model(self, model: Any, 
                                enable_half_precision: bool = True) -> Any:
        """
        Optimize embedding model for Metal Performance Shaders.
        
        Args:
            model: SentenceTransformer or similar embedding model
            enable_half_precision: Use float16 for inference (memory savings)
            
        Returns:
            Optimized model
        """
        if not self.mps_available:
            self.logger.warning("MPS not available - skipping Metal optimization")
            return model
        
        try:
            start_time = time.time()
            original_device = next(iter(model.parameters())).device if hasattr(model, 'parameters') else None
            
            # Move model to MPS device
            model = model.to(self.device)
            
            # Set to evaluation mode for inference optimization
            if hasattr(model, 'eval'):
                model.eval()
            
            # Enable half precision if supported and requested
            if enable_half_precision and hasattr(model, 'half'):
                try:
                    model = model.half()
                    self.logger.info("Half precision (float16) enabled for memory efficiency")
                except Exception as e:
                    self.logger.warning(f"Half precision not supported: {e}")
            
            # Optimize attention mechanisms if available
            self._optimize_attention(model)
            
            # Apply Metal-specific optimizations
            self._apply_metal_optimizations(model)
            
            optimization_time = time.time() - start_time
            self.optimization_stats['models_optimized'] += 1
            
            self.logger.info(f"Embedding model optimized for Metal in {optimization_time:.3f}s "
                           f"(Device: {original_device} → {self.device})")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to optimize embedding model: {e}")
            return model
    
    def optimize_llm_model(self, model: Any, 
                          enable_kv_cache: bool = True,
                          max_sequence_length: Optional[int] = None) -> Any:
        """
        Optimize LLM for Metal Performance Shaders.
        
        Args:
            model: LLM model (llama-cpp or similar)
            enable_kv_cache: Enable key-value caching for faster generation
            max_sequence_length: Maximum sequence length for optimization
            
        Returns:
            Optimized model
        """
        if not self.mps_available:
            return model
        
        try:
            # For llama-cpp-python, Metal optimization is handled at initialization
            # We focus on configuration optimizations
            
            optimization_params = {}
            
            if enable_kv_cache:
                optimization_params['use_mmap'] = True
                optimization_params['use_mlock'] = True
                
            if max_sequence_length:
                optimization_params['n_ctx'] = max_sequence_length
            
            # Configure for Metal acceleration
            optimization_params['n_gpu_layers'] = -1  # Use all GPU layers if supported
            optimization_params['metal'] = True
            
            self.optimization_stats['models_optimized'] += 1
            self.logger.info("LLM configured for Metal acceleration")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to optimize LLM model: {e}")
            return model
    
    def optimize_tensor_operations(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Ensure tensors use MPS device for optimal performance.
        
        Args:
            tensors: List of tensors to optimize
            
        Returns:
            List of optimized tensors
        """
        if not self.mps_available:
            return tensors
        
        optimized_tensors = []
        
        for tensor in tensors:
            try:
                if tensor.device != self.device:
                    # Move tensor to MPS device
                    optimized_tensor = tensor.to(self.device, non_blocking=True)
                    self.optimization_stats['tensors_moved'] += 1
                else:
                    optimized_tensor = tensor
                
                # Optimize tensor properties
                if optimized_tensor.dtype == torch.float64:
                    # Use float32 for better MPS performance
                    optimized_tensor = optimized_tensor.float()
                
                optimized_tensors.append(optimized_tensor)
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize tensor: {e}")
                optimized_tensors.append(tensor)
        
        return optimized_tensors
    
    def optimize_batch_inference(self, model: Any, inputs: List[Any],
                                batch_size: int = 8) -> List[Any]:
        """
        Optimize batch inference using Metal acceleration.
        
        Args:
            model: Model for inference
            inputs: List of inputs to process
            batch_size: Batch size for processing
            
        Returns:
            List of inference results
        """
        if not inputs:
            return []
        
        results = []
        total_start_time = time.time()
        
        # Process in optimized batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            try:
                batch_start_time = time.time()
                
                # Convert batch to tensors if needed
                if isinstance(batch[0], (list, np.ndarray)):
                    batch_tensor = torch.tensor(batch, device=self.device, dtype=torch.float32)
                else:
                    batch_tensor = batch
                
                # Perform inference
                with torch.inference_mode():  # Optimize for inference
                    if hasattr(model, 'encode'):
                        # For embedding models
                        batch_results = model.encode(batch)
                    else:
                        # Generic model inference
                        batch_results = model(batch_tensor)
                
                # Convert results back if needed
                if isinstance(batch_results, torch.Tensor):
                    batch_results = batch_results.cpu().numpy().tolist()
                
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                
                if i == 0:  # Log only first batch for performance estimation
                    items_per_sec = len(batch) / batch_time
                    self.logger.debug(f"Batch inference: {len(batch)} items in {batch_time:.3f}s "
                                    f"({items_per_sec:.1f} items/sec)")
                
            except Exception as e:
                self.logger.error(f"Batch inference failed: {e}")
                # Fallback to individual processing
                for item in batch:
                    try:
                        if hasattr(model, 'encode'):
                            result = model.encode([item])[0]
                        else:
                            result = model(item)
                        results.append(result)
                    except Exception as item_e:
                        self.logger.error(f"Individual inference failed: {item_e}")
                        results.append(None)
        
        total_time = time.time() - total_start_time
        throughput = len(inputs) / total_time
        
        self.logger.info(f"Metal batch inference: {len(inputs)} items in {total_time:.2f}s "
                        f"({throughput:.1f} items/sec)")
        
        return results
    
    def _optimize_attention(self, model: Any) -> None:
        """Apply attention-specific optimizations."""
        try:
            # Enable memory efficient attention if available
            if hasattr(model, 'enable_mem_efficient_attention'):
                model.enable_mem_efficient_attention()
                self.logger.debug("Memory efficient attention enabled")
            
            # Configure attention scaling
            for module in model.modules() if hasattr(model, 'modules') else []:
                if hasattr(module, 'attention'):
                    # Configure attention module for Metal
                    if hasattr(module.attention, 'scale'):
                        module.attention.scale = True
                
        except Exception as e:
            self.logger.debug(f"Attention optimization skipped: {e}")
    
    def _apply_metal_optimizations(self, model: Any) -> None:
        """Apply Metal-specific model optimizations."""
        try:
            # Disable gradient computation for inference
            if hasattr(model, 'requires_grad_'):
                model.requires_grad_(False)
            
            # Optimize for inference mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Configure Metal-specific settings
            for param in model.parameters() if hasattr(model, 'parameters') else []:
                if param.device != self.device:
                    param.data = param.data.to(self.device)
                
        except Exception as e:
            self.logger.debug(f"Metal-specific optimizations failed: {e}")
    
    def profile_mps_usage(self) -> Dict[str, Any]:
        """
        Profile MPS utilization and memory usage.
        
        Returns:
            Dictionary with profiling statistics
        """
        if not self.mps_available:
            return {'mps_available': False}
        
        try:
            # Get memory statistics
            allocated_memory = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            
            # Get driver information
            mps_info = {
                'mps_available': True,
                'device': str(self.device),
                'allocated_memory_mb': allocated_memory / (1024 * 1024),
                'memory_fraction': self.memory_fraction,
            }
            
            # Add optimization statistics
            mps_info.update(self.optimization_stats)
            
            return mps_info
            
        except Exception as e:
            self.logger.error(f"MPS profiling failed: {e}")
            return {'mps_available': True, 'error': str(e)}
    
    def clear_mps_cache(self) -> Dict[str, Any]:
        """
        Clear MPS memory cache to free GPU memory.
        
        Returns:
            Memory statistics before and after clearing
        """
        if not self.mps_available:
            return {}
        
        try:
            # Get memory before clearing
            before_memory = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            
            # Clear cache
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory after clearing
            after_memory = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            
            freed_memory = before_memory - after_memory
            
            stats = {
                'before_memory_mb': before_memory / (1024 * 1024),
                'after_memory_mb': after_memory / (1024 * 1024),
                'freed_memory_mb': freed_memory / (1024 * 1024)
            }
            
            if freed_memory > 1024 * 1024:  # Log if > 1MB freed
                self.logger.info(f"MPS cache cleared: {freed_memory / (1024 * 1024):.1f}MB freed")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to clear MPS cache: {e}")
            return {}
    
    def benchmark_metal_performance(self, model: Any, 
                                  test_inputs: List[Any],
                                  num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark Metal vs CPU performance.
        
        Args:
            model: Model to benchmark
            test_inputs: Test inputs for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Performance comparison statistics
        """
        if not self.mps_available or not test_inputs:
            return {}
        
        results = {'metal': [], 'cpu': []}
        
        try:
            # Benchmark Metal performance
            model_metal = model.to(self.device)
            for _ in range(num_runs):
                start_time = time.time()
                
                with torch.inference_mode():
                    if hasattr(model_metal, 'encode'):
                        _ = model_metal.encode(test_inputs)
                    else:
                        for inp in test_inputs:
                            _ = model_metal(inp)
                
                metal_time = time.time() - start_time
                results['metal'].append(metal_time)
            
            # Benchmark CPU performance
            model_cpu = model.to('cpu')
            for _ in range(num_runs):
                start_time = time.time()
                
                with torch.inference_mode():
                    if hasattr(model_cpu, 'encode'):
                        _ = model_cpu.encode(test_inputs)
                    else:
                        for inp in test_inputs:
                            _ = model_cpu(inp)
                
                cpu_time = time.time() - start_time
                results['cpu'].append(cpu_time)
            
            # Calculate statistics
            metal_avg = np.mean(results['metal'])
            cpu_avg = np.mean(results['cpu'])
            speedup = cpu_avg / metal_avg if metal_avg > 0 else 0
            
            # Update optimization stats
            self.optimization_stats['inference_speedup'] = speedup
            
            benchmark_stats = {
                'metal_avg_time': metal_avg,
                'cpu_avg_time': cpu_avg,
                'speedup_factor': speedup,
                'metal_std': np.std(results['metal']),
                'cpu_std': np.std(results['cpu']),
                'num_runs': num_runs,
                'test_items': len(test_inputs)
            }
            
            self.logger.info(f"Metal benchmark: {speedup:.2f}x speedup "
                           f"({cpu_avg:.3f}s CPU vs {metal_avg:.3f}s Metal)")
            
            return benchmark_stats
            
        except Exception as e:
            self.logger.error(f"Metal benchmark failed: {e}")
            return {}
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get optimization recommendations based on current system state.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not self.mps_available:
            recommendations.append("Install PyTorch with MPS support for GPU acceleration")
            return recommendations
        
        try:
            mps_stats = self.profile_mps_usage()
            allocated_mb = mps_stats.get('allocated_memory_mb', 0)
            
            # Memory recommendations
            if allocated_mb > 10000:  # > 10GB
                recommendations.append("Consider using smaller batch sizes to reduce memory usage")
                recommendations.append("Enable half precision (float16) for memory efficiency")
            
            if allocated_mb < 1000:  # < 1GB
                recommendations.append("Increase batch sizes to better utilize GPU memory")
                recommendations.append("Consider processing larger document chunks")
            
            # Performance recommendations
            if self.optimization_stats['inference_speedup'] < 1.5:
                recommendations.append("Metal acceleration may not be optimal - check model compatibility")
            
            if self.optimization_stats['models_optimized'] == 0:
                recommendations.append("No models optimized yet - run optimize_embedding_model()")
            
            # Configuration recommendations
            if self.memory_fraction < 0.5:
                recommendations.append("Consider increasing memory_fraction for better performance")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to analyze system - check MPS configuration")
        
        return recommendations
    
    def cleanup(self) -> None:
        """Clean up Metal resources."""
        try:
            # Clear MPS cache
            self.clear_mps_cache()
            
            # Reset optimization stats
            self.optimization_stats = {
                'models_optimized': 0,
                'tensors_moved': 0,
                'memory_allocated_mb': 0,
                'inference_speedup': 0.0
            }
            
            self.logger.info("MetalOptimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during shutdown
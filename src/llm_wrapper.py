"""
LLM Wrapper for RAG System
Handles language model inference using llama-cpp-python.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Callable
import threading

from .model_cache import ModelCache


class LLMWrapper:
    """Wrapper for llama-cpp language model with Metal acceleration."""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_path: Path to the GGUF model file
            **kwargs: Configuration parameters from model_config.yaml
        """
        # Normalize/resolve path safely but preserve existence check semantics
        try:
            self.model_path = Path(model_path).expanduser().resolve(strict=True)
        except Exception:
            self.model_path = Path(model_path).expanduser().resolve(strict=False)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Configuration parameters
        self.n_ctx = kwargs.get('n_ctx', 8192)
        self.n_batch = kwargs.get('n_batch', 512)
        self.n_threads = kwargs.get('n_threads', 8)
        self.n_gpu_layers = kwargs.get('n_gpu_layers', -1)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.95)
        self.max_tokens = kwargs.get('max_tokens', 2048)
        
        # Optional cache param keys for ModelCache keying behavior
        self._cache_param_keys = tuple(
            kwargs.get('cache_param_keys', []) or kwargs.get('llm_cache_param_keys', [])
        ) or None

        self.model = None
        self._cache_key = None
        self._effective_cache_param_keys = None
        self.logger = logging.getLogger(__name__)
        self._load_time = 0.0
        self._is_loaded = False
        
        # Load model immediately
        self._load_model()
    
    def _load_model(self):
        """Load the language model with Metal acceleration via ModelCache."""
        self.logger.info(f"Acquiring LLM model from cache: {self.model_path}")
        start_time = time.time()

        try:
            init_params = {
                'n_ctx': self.n_ctx,
                'n_batch': self.n_batch,
                'n_threads': self.n_threads,
                'n_gpu_layers': self.n_gpu_layers,
                'verbose': False,
                'add_bos_token': True,
                'echo': False,
            }

            def _loader():
                # Lazy import to allow tests to patch 'llama_cpp' reliably and
                # to avoid importing heavy backends at module import time.
                from llama_cpp import Llama as LlamaCpp

                self.logger.info(f"Loading LLM model from: {self.model_path}")
                return LlamaCpp(
                    model_path=str(self.model_path),
                    **init_params
                )

            cache = ModelCache.instance()

            # Determine final cache key parameters used for this instance
            default_param_keys = (
                "n_ctx",
                "n_batch",
                "n_threads",
                "n_gpu_layers",
                "add_bos_token",
                "echo",
            )
            self._effective_cache_param_keys = (
                self._cache_param_keys
                or cache._llm_cache_param_keys
                or default_param_keys
            )
            key_params = tuple(
                sorted((k, init_params.get(k)) for k in self._effective_cache_param_keys)
            )
            self._cache_key = (str(self.model_path), key_params)

            self.model = cache.get_llm_model(
                str(self.model_path),
                init_params=init_params,
                loader=_loader,
                cache_param_keys=self._cache_param_keys,
            )

            self._load_time = time.time() - start_time
            self._is_loaded = True

            self.logger.info(f"Model ready in {self._load_time:.2f}s")
            self.logger.info(f"  Context window: {self.n_ctx}")
            self.logger.info(f"  GPU layers: {self.n_gpu_layers}")
            self.logger.info(f"  Metal acceleration: {'enabled' if self.n_gpu_layers == -1 else 'disabled'}")

        except Exception as e:
            self.logger.error(f"Failed to load/acquire model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, 
                top_p: Optional[float] = None,
                stop_sequences: Optional[list] = None) -> str:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            
        Returns:
            Generated text completion
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Use instance defaults if not specified
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        
        # Generate completion
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or [],
            echo=False
        )
        
        return response['choices'][0]['text']
    
    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       top_p: Optional[float] = None,
                       stop_sequences: Optional[list] = None,
                       callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """
        Generate streaming text completion.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            callback: Optional callback for each token
            
        Yields:
            Generated text tokens
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Use instance defaults if not specified
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        
        # Generate streaming completion
        stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or [],
            stream=True,
            echo=False
        )
        
        for chunk in stream:
            token = chunk['choices'][0]['text']
            if callback:
                callback(token)
            yield token
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Number of tokens
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        tokens = self.model.tokenize(text.encode('utf-8'))
        return len(tokens)
    
    def get_context_remaining(self, current_prompt: str) -> int:
        """
        Calculate remaining context window space.
        
        Args:
            current_prompt: Current prompt text
            
        Returns:
            Remaining tokens in context window
        """
        used_tokens = self.count_tokens(current_prompt)
        return max(0, self.n_ctx - used_tokens)
    
    def fits_in_context(self, prompt: str, buffer: int = 100) -> bool:
        """
        Check if prompt fits in context window with buffer.
        
        Args:
            prompt: Prompt text to check
            buffer: Safety buffer tokens
            
        Returns:
            True if prompt fits in context window
        """
        prompt_tokens = self.count_tokens(prompt)
        return prompt_tokens + buffer <= self.n_ctx
    
    def generate_with_stats(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text with detailed statistics.
        
        Args:
            prompt: Input text prompt
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with generated text and statistics
        """
        start_time = time.time()
        prompt_tokens = self.count_tokens(prompt)
        
        generated_text = self.generate(prompt, **kwargs)
        
        end_time = time.time()
        generation_time = end_time - start_time
        output_tokens = self.count_tokens(generated_text)
        total_tokens = prompt_tokens + output_tokens
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'context_remaining': self.n_ctx - total_tokens
        }
    
    def generate_stream_with_stats(self, prompt: str, **kwargs) -> tuple:
        """
        Generate streaming text with statistics tracking.
        
        Args:
            prompt: Input text prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (generator, stats_callback)
        """
        start_time = time.time()
        prompt_tokens = self.count_tokens(prompt)
        stats = {
            'prompt_tokens': prompt_tokens,
            'output_tokens': 0,
            'start_time': start_time,
            'first_token_time': None,
            'generated_text': ''
        }
        
        def update_stats(token: str):
            stats['output_tokens'] += 1
            stats['generated_text'] += token
            if stats['first_token_time'] is None:
                stats['first_token_time'] = time.time()
        
        generator = self.generate_stream(prompt, callback=update_stats, **kwargs)
        
        def get_final_stats():
            end_time = time.time()
            total_time = end_time - start_time
            first_token_latency = (stats['first_token_time'] - start_time) if stats['first_token_time'] else 0
            tokens_per_second = stats['output_tokens'] / total_time if total_time > 0 else 0
            
            return {
                'generated_text': stats['generated_text'],
                'prompt_tokens': stats['prompt_tokens'],
                'output_tokens': stats['output_tokens'],
                'total_tokens': stats['prompt_tokens'] + stats['output_tokens'],
                'generation_time': total_time,
                'first_token_latency': first_token_latency,
                'tokens_per_second': tokens_per_second,
                'context_remaining': self.n_ctx - (stats['prompt_tokens'] + stats['output_tokens'])
            }
        
        return generator, get_final_stats
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model:
            self.logger.info("Unloading model and evicting from cache to free memory")
            
            # Evict from ModelCache to actually free memory
            try:
                cache = ModelCache.instance()

                cache_key = getattr(self, "_cache_key", None)
                if cache_key is None:
                    self.logger.warning("No cache key found for model; skipping eviction")
                else:
                    evicted = cache.evict(cache_key)
                    if evicted:
                        self.logger.info("Model evicted from cache successfully")
                    else:
                        self.logger.warning("Model was not found in cache during eviction")

            except Exception as e:
                self.logger.warning(f"Failed to evict model from cache: {e}")
            
            self.model = None
            self._is_loaded = False
    
    def reload_model(self):
        """Reload the model if it was unloaded."""
        if not self._is_loaded:
            self._load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path),
            'model_loaded': self._is_loaded,
            'load_time': self._load_time,
            'context_window': self.n_ctx,
            'batch_size': self.n_batch,
            'gpu_layers': self.n_gpu_layers,
            'metal_enabled': self.n_gpu_layers == -1,
            'default_params': {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_tokens': self.max_tokens
            }
        }
    
    def __del__(self):
        """Cleanup when wrapper is destroyed."""
        if hasattr(self, 'model') and self.model:
            self.unload_model()


def create_llm_wrapper(model_path: str, config_params: Dict[str, Any]) -> LLMWrapper:
    """
    Factory function to create an LLMWrapper instance.
    
    Args:
        model_path: Path to the GGUF model file
        config_params: Configuration parameters dictionary
        
    Returns:
        Configured LLMWrapper instance
    """
    return LLMWrapper(model_path, **config_params)

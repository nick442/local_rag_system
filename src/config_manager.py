"""
Configuration Management System

Provides YAML-based configuration with profiles, hot-reloading,
and CLI parameter overrides.
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field


@dataclass
class ProfileConfig:
    """Simplified configuration profile with 6 essential RAG parameters."""
    retrieval_k: int
    max_tokens: int
    temperature: float
    chunk_size: int
    chunk_overlap: int
    n_ctx: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentConfig(ProfileConfig):
    """Extended configuration for comprehensive RAG experimentation."""
    
    # Document Processing Parameters
    chunking_strategy: str = "token-based"  # token-based, sentence-based, paragraph-based, semantic
    min_chunk_size: int = 128
    max_chunk_size: int = 2048
    overlap_strategy: str = "fixed"  # fixed, percentage, semantic
    preprocessing_steps: List[str] = field(default_factory=lambda: ["clean", "normalize"])
    chunk_quality_threshold: float = 0.5
    document_filter_regex: Optional[str] = None
    
    # Embedding & Retrieval Parameters  
    embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: Optional[int] = None
    retrieval_method: str = "vector"  # vector, keyword, hybrid, reranked
    similarity_threshold: float = 0.0
    similarity_metric: str = "cosine"  # cosine, dot_product, euclidean
    rerank_model: Optional[str] = None
    rerank_top_k: int = 20
    query_expansion: bool = False
    query_expansion_model: Optional[str] = None
    search_algorithm: str = "exact"  # exact, approximate, hybrid
    index_type: str = "flat"  # flat, ivf, hnsw
    retrieval_fusion_method: str = "reciprocal_rank"
    
    # LLM Generation Parameters
    llm_model_path: str = "models/gemma-3-4b-it-q4_0.gguf"
    llm_type: str = "gemma-3"
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt_template: Optional[str] = None
    response_format: str = "structured"  # structured, json, free-form, markdown
    conversation_memory: bool = True
    context_window_strategy: str = "truncate"  # truncate, summarize, sliding
    generation_strategy: str = "sampling"  # greedy, beam, sampling
    stop_sequences: List[str] = field(default_factory=list)
    response_length_target: Optional[int] = None
    citation_style: str = "inline"  # inline, footnote, bibliography, none
    
    # Corpus & Database Parameters
    collection_filter: Optional[List[str]] = None
    collection_weighting: Optional[Dict[str, float]] = None
    target_corpus: str = "default"
    database_backend: str = "sqlite"  # sqlite, faiss, chromadb
    corpus_preprocessing: List[str] = field(default_factory=lambda: ["dedupe"])
    duplicate_detection_threshold: float = 0.9
    document_freshness_weight: float = 1.0
    corpus_size_limit: Optional[int] = None
    index_build_strategy: str = "incremental"  # incremental, batch, hybrid
    cache_strategy: str = "balanced"  # aggressive, balanced, minimal
    
    # Optional: derived collection naming prefix for chunking experiments
    # Default behavior derives collections like "exp_cs256_co64".
    # Setting collection_prefix to "exp_full_cs" derives "exp_full_cs256_co64".
    collection_prefix: Optional[str] = None


@dataclass  
class ParameterRange:
    """Define sweep ranges for experimental parameters."""
    param_name: str
    range_type: str  # "linear", "logarithmic", "categorical", "boolean"
    min_value: Union[float, int, str] = None
    max_value: Union[float, int, str] = None
    step_size: Union[float, int] = None
    values: Optional[List[Any]] = None  # For categorical values
    
    def generate_values(self) -> List[Any]:
        """Generate list of values for this parameter range."""
        if self.range_type == "categorical":
            return self.values or []
        elif self.range_type == "boolean":
            return [True, False]
        elif self.range_type == "linear":
            if self.min_value is None or self.max_value is None or self.step_size is None:
                raise ValueError(f"Linear range requires min_value, max_value, and step_size")
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += self.step_size
            return values
        elif self.range_type == "logarithmic":
            import numpy as np
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Logarithmic range requires min_value and max_value")
            num_points = int((self.max_value - self.min_value) / (self.step_size or 1)) + 1
            return np.logspace(self.min_value, self.max_value, num_points).tolist()
        else:
            raise ValueError(f"Unknown range type: {self.range_type}")


@dataclass
class ExperimentTemplate:
    """Pre-defined experimental setup for common research questions."""
    name: str
    description: str
    base_config: ExperimentConfig
    parameter_ranges: List[ParameterRange]
    evaluation_queries: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "speed", "relevance"])
    expected_runtime_hours: float = 1.0
    
    def generate_configurations(self) -> List[ExperimentConfig]:
        """Generate all configuration combinations for this template."""
        import itertools
        
        # Generate all parameter combinations
        param_combinations = []
        param_names = []
        param_value_lists = []
        
        for param_range in self.parameter_ranges:
            param_names.append(param_range.param_name)
            param_value_lists.append(param_range.generate_values())
        
        # Create all combinations
        for combination in itertools.product(*param_value_lists):
            config_dict = asdict(self.base_config)
            for param_name, param_value in zip(param_names, combination):
                config_dict[param_name] = param_value
            param_combinations.append(ExperimentConfig(**config_dict))
        
        return param_combinations


class ConstraintValidator:
    """Ensure parameter combinations are valid and feasible."""
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        
        # Validate chunk size constraints
        if config.chunk_size < config.min_chunk_size:
            errors.append(f"chunk_size ({config.chunk_size}) cannot be less than min_chunk_size ({config.min_chunk_size})")
        if config.chunk_size > config.max_chunk_size:
            errors.append(f"chunk_size ({config.chunk_size}) cannot be greater than max_chunk_size ({config.max_chunk_size})")
        
        # Validate overlap constraints
        if config.chunk_overlap >= config.chunk_size:
            errors.append(f"chunk_overlap ({config.chunk_overlap}) must be less than chunk_size ({config.chunk_size})")
        
        # Validate retrieval parameters
        if config.retrieval_k <= 0:
            errors.append(f"retrieval_k ({config.retrieval_k}) must be positive")
        
        # Validate temperature bounds
        if not (0.0 <= config.temperature <= 2.0):
            errors.append(f"temperature ({config.temperature}) must be between 0.0 and 2.0")
        
        # Validate token limits
        if config.max_tokens <= 0:
            errors.append(f"max_tokens ({config.max_tokens}) must be positive")
        if config.n_ctx <= config.max_tokens:
            errors.append(f"n_ctx ({config.n_ctx}) should be greater than max_tokens ({config.max_tokens})")
        
        return errors
    
    def check_resource_requirements(self, config: ExperimentConfig) -> Dict[str, float]:
        """Estimate memory, disk, and compute requirements."""
        # Rough estimates based on configuration
        base_memory = 2.0  # GB base memory
        
        # LLM model memory (rough estimates)
        model_memory = 4.0 if "4b" in config.llm_model_path.lower() else 8.0
        
        # Context window memory scaling
        context_memory = (config.n_ctx / 8192) * 0.5
        
        # Embedding model memory
        embedding_memory = 0.5
        
        total_memory = base_memory + model_memory + context_memory + embedding_memory
        
        # Disk space for results (rough estimate)
        disk_space = 0.1  # GB for experiment results
        
        # Compute time scaling factors
        compute_factor = 1.0
        if config.retrieval_method == "hybrid":
            compute_factor *= 1.5
        if config.query_expansion:
            compute_factor *= 1.3
        if config.rerank_model:
            compute_factor *= 2.0
            
        return {
            "memory_gb": total_memory,
            "disk_gb": disk_space,
            "compute_factor": compute_factor,
            "estimated_time_per_query": 3.0 * compute_factor  # seconds
        }


class ConfigManager:
    """
    YAML-based configuration management system.
    
    Features:
    - Load/save configurations
    - Profile support (fast/balanced/quality)
    - Hot-reload without restart
    - CLI parameter overrides
    - Default value management
    """
    
    DEFAULT_PROFILES = {
        'fast': ProfileConfig(
            retrieval_k=3,
            max_tokens=512,
            temperature=0.7,
            n_ctx=4096,
            chunk_size=256,
            chunk_overlap=64
        ),
        'balanced': ProfileConfig(
            retrieval_k=5,
            max_tokens=1024,
            temperature=0.8,
            n_ctx=8192,
            chunk_size=512,
            chunk_overlap=128
        ),
        'quality': ProfileConfig(
            retrieval_k=10,
            max_tokens=2048,
            temperature=0.9,
            n_ctx=8192,
            chunk_size=512,
            chunk_overlap=128
        )
    }
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Initialize configuration manager with unified config file.
        
        Args:
            config_path: Path to the unified YAML configuration file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Configuration state
        self._config_data: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._current_profile = "balanced"
        
        # Load or create configuration
        self._ensure_config_exists()
        self.load_config()
    
    def _ensure_config_exists(self):
        """Create default configuration file if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file."""
        default_config = {
            'profiles': {
                name: profile.to_dict() 
                for name, profile in self.DEFAULT_PROFILES.items()
            },
            'current_profile': 'balanced',
            'model_overrides': {},
            'corpus_path': 'corpus/',
            'database': {
                'path': 'data/rag_vectors.db',
                'backup_enabled': True,
                'backup_interval': 24  # hours
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/rag_system.log',
                'max_size_mb': 100,
                'backup_count': 5
            },
            'performance': {
                'embedding_batch_size': 32,
                'parallel_workers': 4,
                'memory_limit_gb': 8,
                'cache_enabled': True
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created default configuration at {self.config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Complete configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                loaded = yaml.safe_load(f)

            # Normalize empty or invalid structures
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                self.logger.warning("Config file root is not a mapping; using defaults")
                loaded = {}

            # Ensure required keys exist with sane defaults
            loaded.setdefault('profiles', {name: profile.to_dict() for name, profile in self.DEFAULT_PROFILES.items()})
            if not isinstance(loaded.get('profiles'), dict):
                self.logger.warning("'profiles' section invalid; resetting to defaults")
                loaded['profiles'] = {name: profile.to_dict() for name, profile in self.DEFAULT_PROFILES.items()}

            loaded.setdefault('current_profile', 'balanced')

            self._config_data = loaded
            self._current_profile = self._config_data.get('current_profile', 'balanced')
            self.logger.info(f"Loaded configuration from {self.config_path}")
            
            return self._config_data
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            self._config_data = {
                'profiles': {
                    name: profile.to_dict() 
                    for name, profile in self.DEFAULT_PROFILES.items()
                },
                'current_profile': 'balanced'
            }
            return self._config_data
    
    def save_config(self) -> bool:
        """
        Save current configuration to YAML file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Update current profile in config data
            self._config_data['current_profile'] = self._current_profile
            
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved configuration to {self.config_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_profile(self, profile_name: Optional[str] = None) -> ProfileConfig:
        """
        Get configuration profile.
        
        Args:
            profile_name: Name of profile to get (uses current if None)
        
        Returns:
            Profile configuration object
        """
        if profile_name is None:
            profile_name = self._current_profile
        
        # Get base profile data
        profile_data = self._config_data.get('profiles', {}).get(profile_name)
        
        if not profile_data:
            self.logger.warning(f"Profile '{profile_name}' not found, using default")
            profile_data = self.DEFAULT_PROFILES['balanced'].to_dict()
        
        # Apply overrides
        merged_data = {**profile_data, **self._overrides}
        
        # Create ProfileConfig object
        try:
            return ProfileConfig(**merged_data)
        except TypeError as e:
            self.logger.warning(f"Invalid profile data: {e}, using defaults")
            return self.DEFAULT_PROFILES['balanced']
    
    def list_profiles(self) -> Dict[str, ProfileConfig]:
        """
        List all available profiles.
        
        Returns:
            Dictionary mapping profile names to ProfileConfig objects
        """
        profiles = {}
        config_profiles = self._config_data.get('profiles', {})
        
        for name, data in config_profiles.items():
            try:
                profiles[name] = ProfileConfig(**data)
            except TypeError:
                self.logger.warning(f"Invalid profile data for '{name}', skipping")
        
        return profiles
    
    def switch_profile(self, profile_name: str) -> bool:
        """
        Switch to a different configuration profile.
        
        Args:
            profile_name: Name of profile to switch to
        
        Returns:
            True if switched successfully, False otherwise
        """
        if profile_name not in self._config_data.get('profiles', {}):
            self.logger.error(f"Profile '{profile_name}' does not exist")
            return False
        
        old_profile = self._current_profile
        self._current_profile = profile_name
        
        # Clear overrides when switching profiles
        self._overrides.clear()
        
        self.logger.info(f"Switched from profile '{old_profile}' to '{profile_name}'")
        return True
    
    def override_param(self, key: str, value: Any, temporary: bool = True):
        """
        Override a configuration parameter.
        
        Args:
            key: Parameter name to override
            value: New parameter value
            temporary: If True, override is temporary (not saved)
        """
        if temporary:
            self._overrides[key] = value
        else:
            # Permanent override - update current profile
            profile_data = self._config_data.setdefault('profiles', {}).setdefault(self._current_profile, {})
            profile_data[key] = value
        
        self.logger.info(f"Override parameter '{key}' = {value} ({'temporary' if temporary else 'permanent'})")
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter value.

        Supports:
        - Temporary overrides
        - Current profile attributes
        - Dotted notation for nested keys in the general config (e.g., 'database.path')
        """
        # Check overrides first (support exact key, including dotted)
        if key in self._overrides:
            return self._overrides[key]

        # If this is a simple key and matches a profile attribute, return it
        if '.' not in key:
            current_profile = self.get_profile()
            if hasattr(current_profile, key):
                return getattr(current_profile, key)

        # Try dotted notation lookup in general config
        def _get_nested(d: Dict[str, Any], dotted_key: str, default_val: Any) -> Any:
            parts = dotted_key.split('.')
            cur: Any = d
            for part in parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default_val
            return cur

        value = _get_nested(self._config_data, key, default)
        return value
    
    def set_param(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Set a configuration parameter permanently.
        
        Args:
            key: Parameter name
            value: New parameter value
            save: Whether to save configuration immediately
        
        Returns:
            True if set successfully, False otherwise
        """
        try:
            # Update current profile if it's a profile parameter
            profile_config = self.get_profile()
            if hasattr(profile_config, key):
                profile_data = self._config_data.setdefault('profiles', {}).setdefault(self._current_profile, {})
                profile_data[key] = value
            else:
                # General configuration parameter
                self._config_data[key] = value
            
            if save:
                return self.save_config()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to set parameter '{key}': {e}")
            return False
    
    def reload_config(self) -> bool:
        """
        Hot-reload configuration from file.
        
        Returns:
            True if reloaded successfully, False otherwise
        """
        try:
            old_config = self._config_data.copy()
            self.load_config()
            
            if self._config_data != old_config:
                self.logger.info("Configuration reloaded with changes")
            else:
                self.logger.info("Configuration reloaded (no changes)")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_current_profile_name(self) -> str:
        """Get the name of the currently active profile."""
        return self._current_profile
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Configuration summary dictionary
        """
        current_profile = self.get_profile()
        
        return {
            'current_profile': self._current_profile,
            'available_profiles': list(self._config_data.get('profiles', {}).keys()),
            'active_settings': current_profile.to_dict(),
            'temporary_overrides': self._overrides.copy(),
            'config_file': str(self.config_path),
            'file_exists': self.config_path.exists()
        }


def create_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Factory function to create a ConfigManager instance.
    
    Args:
        config_path: Optional custom configuration path
        
    Returns:
        Configured ConfigManager instance
    """
    if config_path is None:
        config_path = "config/rag_config.yaml"
    
    return ConfigManager(config_path)

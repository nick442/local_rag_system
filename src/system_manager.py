"""
System Manager - Central coordination for the RAG system
Handles component lifecycle, dependency injection, and resource management
"""

import json
import logging
import os
import sys
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass 
class SystemConfig:
    """System configuration container"""
    db_path: str = "data/rag_vectors.db"
    embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_path: str = "models/gemma-3-4b-it-q4_0.gguf"
    log_level: str = "INFO"
    max_memory_gb: float = 8.0
    min_disk_space_gb: float = 1.0
    component_timeout_sec: int = 30

class SystemManager:
    """Central system coordination and resource management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.components: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        self.startup_time = datetime.now()
        self.is_initialized = False
        self.is_shutting_down = False
        
    def _load_config(self, config_path: Optional[str]) -> SystemConfig:
        """Load system configuration"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return SystemConfig(**config_data)
            except Exception:
                pass
        return SystemConfig()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging"""
        logger = logging.getLogger('system_manager')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/system.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def initialize_components(self) -> bool:
        """Lazy initialization of core components"""
        if self.is_initialized:
            return True
            
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize state first
            self.state['initialized_at'] = datetime.now().isoformat()
            self.state['components_ready'] = []
            
            # Initialize configuration manager first
            self._init_config_manager()
            
            # Other components will be initialized on-demand
            
            self.is_initialized = True
            self.logger.info("System initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def _init_config_manager(self):
        """Initialize configuration manager"""
        try:
            from .config_manager import ConfigManager
            self.components['config_manager'] = ConfigManager()
            self.state['components_ready'].append('config_manager')
            self.logger.debug("Configuration manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize config manager: {e}")
            raise
    
    def get_component(self, name: str) -> Any:
        """Lazy-load components on demand"""
        if self.is_shutting_down:
            raise RuntimeError("System is shutting down")
            
        if name in self.components:
            return self.components[name]
        
        try:
            component = self._create_component(name)
            self.components[name] = component
            if 'components_ready' not in self.state:
                self.state['components_ready'] = []
            if name not in self.state['components_ready']:
                self.state['components_ready'].append(name)
            self.logger.debug(f"Component '{name}' loaded successfully")
            return component
        except Exception as e:
            self.logger.error(f"Failed to load component '{name}': {e}")
            raise
    
    def _create_component(self, name: str) -> Any:
        """Factory method for creating components"""
        component_map = {
            'rag_pipeline': self._create_rag_pipeline,
            'vector_database': self._create_vector_database,
            'embedding_service': self._create_embedding_service,
            'llm_wrapper': self._create_llm_wrapper,
            'corpus_manager': self._create_corpus_manager,
            'corpus_organizer': self._create_corpus_organizer,
            'corpus_analytics': self._create_corpus_analytics,
            'monitor': self._create_monitor,
            'document_ingestion': self._create_document_ingestion,
        }
        
        if name not in component_map:
            raise ValueError(f"Unknown component: {name}")
        
        return component_map[name]()
    
    def _create_rag_pipeline(self):
        """Create RAG pipeline component"""
        from .rag_pipeline import RAGPipeline
        return RAGPipeline(
            db_path=self.config.db_path,
            embedding_model_path=self.config.embedding_model_path,
            llm_model_path=self.config.llm_model_path
        )
    
    def _create_vector_database(self):
        """Create vector database component"""
        from .vector_database import VectorDatabase
        return VectorDatabase(self.config.db_path)
    
    def _create_embedding_service(self):
        """Create embedding service component"""
        from .embedding_service import EmbeddingService
        return EmbeddingService(self.config.embedding_model_path)
    
    def _create_llm_wrapper(self):
        """Create LLM wrapper component"""
        from .llm_wrapper import LLMWrapper
        return LLMWrapper(self.config.llm_model_path)
    
    def _create_corpus_manager(self):
        """Create corpus manager component"""
        from .corpus_manager import CorpusManager
        vector_db = self.get_component('vector_database')
        ingestion_service = self.get_component('document_ingestion')
        return CorpusManager(vector_db, ingestion_service)
    
    def _create_corpus_organizer(self):
        """Create corpus organizer component"""
        from .corpus_organizer import CorpusOrganizer
        vector_db = self.get_component('vector_database')
        return CorpusOrganizer(vector_db)
    
    def _create_corpus_analytics(self):
        """Create corpus analytics component"""
        from .corpus_analytics import CorpusAnalytics
        vector_db = self.get_component('vector_database')
        return CorpusAnalytics(vector_db)
    
    def _create_monitor(self):
        """Create monitoring component"""
        from .monitor import Monitor
        return Monitor()
    
    def _create_document_ingestion(self):
        """Create document ingestion service"""
        from .document_ingestion import DocumentIngestionService
        embedding_service = self.get_component('embedding_service')
        return DocumentIngestionService(embedding_service)
    
    def health_check(self) -> bool:
        """Verify system readiness"""
        checks = {
            'models_available': self._check_models(),
            'database_ready': self._check_database(),
            'memory_sufficient': self._check_memory(),
            'disk_space_available': self._check_disk_space(),
            'dependencies_installed': self._check_dependencies()
        }
        
        all_healthy = all(checks.values())
        
        if not all_healthy:
            self.logger.warning("Health check failed:")
            for check, status in checks.items():
                if not status:
                    self.logger.warning(f"  - {check}: FAILED")
        
        return all_healthy
    
    def _check_models(self) -> bool:
        """Check if required models exist"""
        llm_path = Path(self.config.llm_model_path)
        embedding_path = Path(self.config.embedding_model_path)
        
        if not llm_path.exists():
            self.logger.error(f"LLM model not found: {llm_path}")
            return False
            
        # Embedding model may be downloaded on first use
        return True
    
    def _check_database(self) -> bool:
        """Check database connectivity"""
        try:
            db_path = Path(self.config.db_path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            return False
    
    def _check_memory(self) -> bool:
        """Check available memory"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 1.0:  # At least 1GB required
                self.logger.warning(f"Low memory: {available_gb:.1f}GB available")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('.')
            available_gb = disk.free / (1024**3)
            
            if available_gb < self.config.min_disk_space_gb:
                self.logger.error(f"Insufficient disk space: {available_gb:.1f}GB available")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check critical dependencies"""
        critical_modules = [
            'torch', 'sentence_transformers', 'llama_cpp', 
            'sqlite3', 'click', 'rich'
        ]
        
        missing = []
        for module in critical_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            self.logger.error(f"Missing dependencies: {missing}")
            return False
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
                'initialized': self.is_initialized,
                'components_loaded': list(self.components.keys()),
                'health': {
                    'models': self._check_models(),
                    'database': self._check_database(),
                    'memory': self._check_memory(),
                    'disk': self._check_disk_space(),
                    'dependencies': self._check_dependencies()
                },
                'resources': {
                    'memory_used_gb': (memory.total - memory.available) / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_used_gb': (disk.total - disk.free) / (1024**3),
                    'disk_available_gb': disk.free / (1024**3),
                    'cpu_percent': psutil.cpu_percent(interval=1)
                },
                'config': {
                    'db_path': self.config.db_path,
                    'llm_model': self.config.llm_model_path,
                    'embedding_model': self.config.embedding_model_path,
                    'log_level': self.config.log_level
                }
            }
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'initialized': self.is_initialized
            }
    
    def save_state(self):
        """Save current system state"""
        try:
            state_file = Path('data/system_state.json')
            state_file.parent.mkdir(exist_ok=True)
            
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
                'components_loaded': list(self.components.keys()),
                'state': self.state
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.debug("System state saved")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        self.logger.info("Shutting down system...")
        
        try:
            # Save current state
            self.save_state()
            
            # Close components in reverse order
            for name, component in reversed(list(self.components.items())):
                try:
                    if hasattr(component, 'close'):
                        component.close()
                    if hasattr(component, 'cleanup'):
                        component.cleanup()
                    self.logger.debug(f"Component '{name}' shut down successfully")
                except Exception as e:
                    self.logger.error(f"Error shutting down component '{name}': {e}")
            
            self.components.clear()
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            # Close logging handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
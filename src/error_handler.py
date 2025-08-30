"""
Error Handler - Centralized error handling and recovery
Provides intelligent error categorization, recovery strategies, and user-friendly messaging
"""

import gc
import os
import sys
import json
import traceback
import logging
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Type
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Recovery action types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    RESTART = "restart"
    ABORT = "abort"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    component: str
    operation: str
    user_action: str
    system_state: Dict[str, Any]
    timestamp: str

@dataclass
class RecoveryResult:
    """Result of error recovery attempt"""
    success: bool
    action_taken: RecoveryAction
    message: str
    details: Dict[str, Any]
    should_continue: bool

class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self, system_manager):
        self.system = system_manager
        self.logger = system_manager.logger.getChild('error_handler')
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'by_type': {},
            'by_component': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        
        # Recovery strategies mapping
        self.recovery_strategies: Dict[str, Callable] = {
            # Memory errors
            'OutOfMemoryError': self._handle_out_of_memory,
            'MemoryError': self._handle_out_of_memory,
            'torch.cuda.OutOfMemoryError': self._handle_gpu_oom,
            
            # Model errors
            'OSError': self._handle_model_error,  # Often model file issues
            'FileNotFoundError': self._handle_file_not_found,
            'PermissionError': self._handle_permission_error,
            'RuntimeError': self._handle_runtime_error,
            
            # Database errors
            'sqlite3.Error': self._handle_database_error,
            'sqlite3.OperationalError': self._handle_database_error,
            'sqlite3.DatabaseError': self._handle_database_error,
            
            # Network errors
            'urllib.error.URLError': self._handle_network_error,
            'requests.exceptions.RequestException': self._handle_network_error,
            'ConnectionError': self._handle_network_error,
            'TimeoutError': self._handle_network_error,
            
            # Import errors
            'ImportError': self._handle_import_error,
            'ModuleNotFoundError': self._handle_import_error,
            
            # Configuration errors
            'KeyError': self._handle_config_error,
            'ValueError': self._handle_value_error,
            'TypeError': self._handle_type_error,
            
            # Unknown errors
            'Exception': self._handle_unknown_error
        }
        
        # User-friendly error messages
        self.user_messages = {
            'OutOfMemoryError': "The system ran out of memory. Try processing smaller batches or closing other applications.",
            'FileNotFoundError': "A required file was not found. Please check the file path and ensure the file exists.",
            'PermissionError': "Permission denied. Please check file permissions or run with appropriate privileges.",
            'ImportError': "A required Python package is missing. Please install missing dependencies.",
            'ConnectionError': "Network connection failed. Please check your internet connection.",
            'sqlite3.Error': "Database error occurred. The database may be corrupted or locked.",
            'ValueError': "Invalid value provided. Please check your input parameters.",
            'RuntimeError': "A runtime error occurred. This may be due to system configuration issues."
        }
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> RecoveryResult:
        """Main error handling entry point"""
        try:
            # Record error statistics
            self._record_error(error, context)
            
            # Log error with full context
            self._log_error(error, context)
            
            # Determine error type and severity
            error_type = self._get_error_type(error)
            severity = self._assess_severity(error, context)
            
            # Get user-friendly message
            user_message = self._get_user_message(error_type, error)
            
            # Attempt recovery if appropriate
            if severity != ErrorSeverity.CRITICAL:
                recovery_result = self._attempt_recovery(error, error_type, context)
                if recovery_result.success:
                    self.error_stats['successful_recoveries'] += 1
                    return recovery_result
            
            # If no recovery possible, return failure result
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message=user_message,
                details={
                    'error_type': error_type,
                    'severity': severity.value,
                    'original_error': str(error),
                    'traceback': traceback.format_exc()
                },
                should_continue=severity != ErrorSeverity.CRITICAL
            )
            
        except Exception as handler_error:
            # Error in error handler - log and return basic result
            self.logger.critical(f"Error handler failed: {handler_error}")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message="An unexpected error occurred in the error handling system.",
                details={'handler_error': str(handler_error)},
                should_continue=False
            )
    
    def _record_error(self, error: Exception, context: Optional[ErrorContext]):
        """Record error statistics"""
        self.error_stats['total_errors'] += 1
        
        error_type = type(error).__name__
        self.error_stats['by_type'][error_type] = self.error_stats['by_type'].get(error_type, 0) + 1
        
        if context:
            component = context.component
            self.error_stats['by_component'][component] = self.error_stats['by_component'].get(component, 0) + 1
    
    def _log_error(self, error: Exception, context: Optional[ErrorContext]):
        """Log error with full context"""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            error_info['context'] = {
                'component': context.component,
                'operation': context.operation,
                'user_action': context.user_action,
                'system_state': context.system_state
            }
        
        self.logger.error(f"Error occurred: {json.dumps(error_info, indent=2)}")
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
    def _get_error_type(self, error: Exception) -> str:
        """Get standardized error type"""
        return f"{type(error).__module__}.{type(error).__name__}"
    
    def _assess_severity(self, error: Exception, context: Optional[ErrorContext]) -> ErrorSeverity:
        """Assess error severity"""
        error_type = type(error).__name__
        
        # Critical errors that require immediate attention
        critical_errors = {
            'SystemExit', 'KeyboardInterrupt', 'SystemError',
            'MemoryError', 'OutOfMemoryError'
        }
        
        # High severity errors
        high_severity_errors = {
            'RuntimeError', 'OSError', 'sqlite3.DatabaseError',
            'ImportError', 'ModuleNotFoundError'
        }
        
        # Medium severity errors
        medium_severity_errors = {
            'FileNotFoundError', 'PermissionError', 'ValueError',
            'TypeError', 'ConnectionError', 'TimeoutError'
        }
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_severity_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_severity_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_user_message(self, error_type: str, error: Exception) -> str:
        """Get user-friendly error message"""
        simple_type = error_type.split('.')[-1]  # Get class name without module
        
        if simple_type in self.user_messages:
            return self.user_messages[simple_type]
        
        # Generic messages based on error type patterns
        if 'Memory' in simple_type:
            return "The system ran out of memory. Try reducing the workload or restarting the application."
        elif 'File' in simple_type or 'Path' in simple_type:
            return "A file or directory could not be accessed. Please check the path and permissions."
        elif 'Network' in simple_type or 'Connection' in simple_type:
            return "A network error occurred. Please check your internet connection."
        elif 'Import' in simple_type or 'Module' in simple_type:
            return "A required software component is missing. Please check your installation."
        else:
            return f"An error occurred: {str(error)}. Please check the logs for more details."
    
    def _attempt_recovery(self, error: Exception, error_type: str, context: Optional[ErrorContext]) -> RecoveryResult:
        """Attempt to recover from error"""
        self.error_stats['recovery_attempts'] += 1
        
        # Try specific recovery strategy
        simple_type = error_type.split('.')[-1]
        
        if simple_type in self.recovery_strategies:
            strategy = self.recovery_strategies[simple_type]
        else:
            strategy = self.recovery_strategies['Exception']  # Default strategy
        
        try:
            return strategy(error, context)
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed: {recovery_error}")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message=f"Recovery failed: {recovery_error}",
                details={'recovery_error': str(recovery_error)},
                should_continue=False
            )
    
    def _handle_out_of_memory(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle out of memory errors"""
        self.logger.info("Attempting memory recovery...")
        
        # Step 1: Force garbage collection
        gc.collect()
        
        # Step 2: Clear component caches if possible
        cleared_components = []
        for name, component in self.system.components.items():
            try:
                if hasattr(component, 'clear_cache'):
                    component.clear_cache()
                    cleared_components.append(name)
            except:
                pass
        
        # Step 3: Check if memory was freed
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb > 1.0:  # If we have more than 1GB available now
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.RETRY,
                message=f"Memory recovered: {available_gb:.1f}GB available",
                details={
                    'garbage_collected': True,
                    'caches_cleared': cleared_components,
                    'available_memory_gb': available_gb
                },
                should_continue=True
            )
        
        # Step 4: Suggest graceful degradation
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.DEGRADE,
            message="Unable to recover sufficient memory. Consider reducing batch size or restarting.",
            details={
                'available_memory_gb': available_gb,
                'suggestion': 'reduce_batch_size_or_restart'
            },
            should_continue=True
        )
    
    def _handle_gpu_oom(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle GPU out of memory errors"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return RecoveryResult(
                    success=True,
                    action_taken=RecoveryAction.RETRY,
                    message="GPU memory cache cleared",
                    details={'cuda_cache_cleared': True},
                    should_continue=True
                )
        except:
            pass
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.FALLBACK,
            message="GPU memory error - consider using CPU instead",
            details={'suggestion': 'use_cpu_fallback'},
            should_continue=True
        )
    
    def _handle_model_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle model loading/access errors"""
        error_msg = str(error).lower()
        
        # Check if it's a model file issue
        if 'no such file' in error_msg or 'cannot open' in error_msg:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message="Model file not found. Please check model path in configuration.",
                details={
                    'suggestion': 'check_model_path',
                    'current_config': {
                        'llm_model': self.system.config.llm_model_path,
                        'embedding_model': self.system.config.embedding_model_path
                    }
                },
                should_continue=False
            )
        
        # Check if it's a format/compatibility issue
        if 'unsupported' in error_msg or 'invalid' in error_msg:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message="Model format not supported. Please check model compatibility.",
                details={'suggestion': 'check_model_format'},
                should_continue=False
            )
        
        # Generic model error
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            message="Model loading failed. Retrying...",
            details={'retry_suggested': True},
            should_continue=True
        )
    
    def _handle_file_not_found(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle file not found errors"""
        file_path = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        # Try to create parent directory if it doesn't exist
        try:
            path = Path(file_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                return RecoveryResult(
                    success=True,
                    action_taken=RecoveryAction.RETRY,
                    message=f"Created missing directory: {path.parent}",
                    details={'created_directory': str(path.parent)},
                    should_continue=True
                )
        except:
            pass
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message=f"File not found: {file_path}. Please check the path.",
            details={'missing_file': file_path},
            should_continue=False
        )
    
    def _handle_permission_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle permission errors"""
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message="Permission denied. Please check file permissions or run with appropriate privileges.",
            details={'suggestion': 'check_permissions'},
            should_continue=False
        )
    
    def _handle_runtime_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle runtime errors"""
        error_msg = str(error).lower()
        
        # CUDA errors
        if 'cuda' in error_msg:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.FALLBACK,
                message="CUDA error detected. Falling back to CPU processing.",
                details={'suggestion': 'use_cpu'},
                should_continue=True
            )
        
        # Generic runtime error
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            message="Runtime error occurred. This may be temporary.",
            details={'retry_suggested': True},
            should_continue=True
        )
    
    def _handle_database_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle database errors"""
        error_msg = str(error).lower()
        
        # Database locked
        if 'locked' in error_msg:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.RETRY,
                message="Database was locked. Retrying...",
                details={'retry_delay_seconds': 1},
                should_continue=True
            )
        
        # Corrupted database
        if 'corrupt' in error_msg or 'malformed' in error_msg:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message="Database appears to be corrupted. Backup and reinitialize may be needed.",
                details={'suggestion': 'backup_and_reinitialize'},
                should_continue=False
            )
        
        # Generic database error
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            message="Database error occurred. Retrying...",
            details={'retry_suggested': True},
            should_continue=True
        )
    
    def _handle_network_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle network errors"""
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.RETRY,
            message="Network error occurred. Retrying with delay...",
            details={
                'retry_delay_seconds': 5,
                'max_retries': 3
            },
            should_continue=True
        )
    
    def _handle_import_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle import errors"""
        missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message=f"Missing Python package: {missing_module}. Please install it.",
            details={
                'missing_module': missing_module,
                'suggestion': f'pip install {missing_module}'
            },
            should_continue=False
        )
    
    def _handle_config_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle configuration errors"""
        missing_key = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message=f"Configuration error: missing key '{missing_key}'",
            details={
                'missing_key': missing_key,
                'suggestion': 'check_configuration'
            },
            should_continue=False
        )
    
    def _handle_value_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle value errors"""
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message=f"Invalid value: {str(error)}. Please check your input.",
            details={'suggestion': 'check_input_values'},
            should_continue=True
        )
    
    def _handle_type_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle type errors"""
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message=f"Type error: {str(error)}. Please check parameter types.",
            details={'suggestion': 'check_parameter_types'},
            should_continue=True
        )
    
    def _handle_unknown_error(self, error: Exception, context: Optional[ErrorContext]) -> RecoveryResult:
        """Handle unknown/generic errors"""
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            message=f"Unexpected error: {str(error)}",
            details={
                'error_type': type(error).__name__,
                'retry_suggested': True
            },
            should_continue=True
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        total = self.error_stats['total_errors']
        recovery_rate = (self.error_stats['successful_recoveries'] / max(self.error_stats['recovery_attempts'], 1)) * 100
        
        return {
            'total_errors': total,
            'errors_by_type': self.error_stats['by_type'],
            'errors_by_component': self.error_stats['by_component'],
            'recovery_attempts': self.error_stats['recovery_attempts'],
            'successful_recoveries': self.error_stats['successful_recoveries'],
            'recovery_success_rate': round(recovery_rate, 1),
            'most_common_error': max(self.error_stats['by_type'].items(), key=lambda x: x[1])[0] if self.error_stats['by_type'] else None,
            'most_problematic_component': max(self.error_stats['by_component'].items(), key=lambda x: x[1])[0] if self.error_stats['by_component'] else None
        }
    
    def reset_statistics(self):
        """Reset error statistics"""
        self.error_stats = {
            'total_errors': 0,
            'by_type': {},
            'by_component': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        self.logger.info("Error statistics reset")
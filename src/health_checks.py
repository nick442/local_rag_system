"""
Health Check System - Comprehensive system health monitoring
Provides detailed diagnostics and health assessment
"""

import os
import json
import psutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import logging

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: bool
    message: str
    details: Dict[str, Any]
    timestamp: str
    execution_time_ms: float

@dataclass
class SystemHealthReport:
    """Complete system health report"""
    timestamp: str
    overall_status: bool
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]
    recommendations: List[str]

class HealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config_or_system, components: Optional[Dict[str, Any]] = None):
        """Initialize HealthChecker with ConfigManager or legacy SystemManager.

        Supports both the new ConfigManager-based initialization and legacy
        SystemManager objects that expose `.config`, `.components`, and `.logger`.
        """
        # Determine config manager
        if hasattr(config_or_system, 'get_param') and hasattr(config_or_system, 'load_config'):
            # Looks like a ConfigManager
            self.config_manager = config_or_system
            logger = getattr(self.config_manager, 'logger', None)
        elif hasattr(config_or_system, 'config'):
            # Legacy SystemManager: use its config
            self.config_manager = getattr(config_or_system, 'config')
            logger = getattr(config_or_system, 'logger', None)
        else:
            raise ValueError("HealthChecker requires a ConfigManager or an object with a 'config' attribute")

        # Components: allow explicit override, else read from legacy system if present
        if components is not None:
            self.components = components
        else:
            self.components = getattr(config_or_system, 'components', {}) if hasattr(config_or_system, 'components') else {}

        # Logger: prefer provided/attached logger, fallback to module logger
        self.logger = logger or logging.getLogger(__name__)
        
    def run_all_checks(self) -> SystemHealthReport:
        """Run comprehensive health checks"""
        start_time = datetime.now()
        
        checks = [
            self._check_python_environment(),
            self._check_dependencies(),
            self._check_models(),
            self._check_database(),
            self._check_system_resources(),
            self._check_disk_space(),
            self._check_network_connectivity(),
            self._check_file_permissions(),
            self._check_configuration(),
            self._check_component_status()
        ]
        
        # Calculate overall status
        overall_status = all(check.status for check in checks)
        
        # Generate summary
        summary = self._generate_summary(checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks)
        
        report = SystemHealthReport(
            timestamp=start_time.isoformat(),
            overall_status=overall_status,
            checks=checks,
            summary=summary,
            recommendations=recommendations
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"Health check completed in {execution_time:.2f}ms - Status: {'HEALTHY' if overall_status else 'UNHEALTHY'}")
        
        return report
    
    def _time_check(self, check_func) -> Tuple[Any, float]:
        """Time execution of a check function"""
        start = datetime.now()
        result = check_func()
        execution_time = (datetime.now() - start).total_seconds() * 1000
        return result, execution_time
    
    def _check_python_environment(self) -> HealthCheckResult:
        """Check Python environment"""
        def check():
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Check Python version (requires 3.8+)
            if sys.version_info < (3, 8):
                return False, f"Python {python_version} is too old (requires 3.8+)", {
                    'version': python_version,
                    'required': '3.8+',
                    'path': sys.executable
                }
            
            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            
            details = {
                'version': python_version,
                'executable': sys.executable,
                'virtual_env': in_venv,
                'platform': sys.platform,
                'path': sys.path[:3]  # First 3 entries
            }
            
            message = f"Python {python_version} {'(venv)' if in_venv else '(system)'}"
            return True, message, details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="python_environment",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies"""
        def check():
            critical_deps = {
                'torch': 'PyTorch',
                'sentence_transformers': 'Sentence Transformers',
                'llama_cpp': 'llama.cpp Python bindings',
                'click': 'Click CLI framework',
                'rich': 'Rich terminal formatting',
                'psutil': 'System utilities',
                'sqlite3': 'SQLite database'
            }
            
            installed = {}
            missing = []
            
            for module, description in critical_deps.items():
                try:
                    mod = __import__(module)
                    version = getattr(mod, '__version__', 'unknown')
                    installed[module] = {'version': version, 'description': description}
                except ImportError:
                    missing.append({'module': module, 'description': description})
            
            details = {
                'installed': installed,
                'missing': missing,
                'total_required': len(critical_deps),
                'installed_count': len(installed)
            }
            
            if missing:
                return False, f"Missing {len(missing)} critical dependencies", details
            
            return True, f"All {len(installed)} dependencies installed", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="dependencies",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_models(self) -> HealthCheckResult:
        """Check model availability"""
        def check():
            # Resolve model paths from config with sensible defaults
            try:
                from src.config_manager import ExperimentConfig  # for defaults
                default_llm = ExperimentConfig().llm_model_path
                default_embed = ExperimentConfig().embedding_model_path
            except Exception:
                # Fallback defaults if import fails
                default_llm = 'models/gemma-3-4b-it-q4_0.gguf'
                default_embed = 'sentence-transformers/all-MiniLM-L6-v2'

            llm_path_str = self.config_manager.get_param('llm_model_path', default_llm)
            embedding_path_str = self.config_manager.get_param('embedding_model_path', default_embed)

            llm_path = Path(llm_path_str)
            embedding_path = Path(embedding_path_str)
            
            models_status = {}
            
            # Check LLM model (critical)
            if llm_path.exists():
                file_size_mb = llm_path.stat().st_size / (1024 * 1024)
                models_status['llm'] = {
                    'path': str(llm_path),
                    'exists': True,
                    'size_mb': round(file_size_mb, 2),
                    'last_modified': datetime.fromtimestamp(llm_path.stat().st_mtime).isoformat()
                }
            else:
                models_status['llm'] = {
                    'path': str(llm_path),
                    'exists': False,
                    'error': 'File not found'
                }
            
            # Check embedding model (may be a remote ID; local existence optional)
            if embedding_path.exists():
                models_status['embedding'] = {
                    'path': str(embedding_path),
                    'exists': True,
                    'type': 'local'
                }
            else:
                models_status['embedding'] = {
                    'path': str(embedding_path),
                    'exists': False,
                    'type': 'will_download_on_first_use'
                }
            
            details = {'models': models_status}
            
            # LLM is critical, embedding can be downloaded
            if not models_status['llm']['exists']:
                return False, f"LLM model not found: {llm_path}", details
            
            return True, "Models available", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="models",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and status"""
        def check():
            db_path_str = self.config_manager.get_param('database.path', 'data/rag_vectors.db')
            db_path = Path(db_path_str)
            
            try:
                # Ensure directory exists
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Test connection
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check basic functionality
                    cursor.execute("SELECT sqlite_version()")
                    sqlite_version = cursor.fetchone()[0]
                    
                    # Check if tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    # Get database size
                    db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
                    
                    details = {
                        'path': str(db_path),
                        'sqlite_version': sqlite_version,
                        'size_mb': round(db_size_mb, 2),
                        'tables': tables,
                        'table_count': len(tables)
                    }
                    
                    if not tables:
                        return True, "Database accessible (empty)", details
                    
                    # Count records in main tables
                    record_counts = {}
                    for table in ['documents', 'chunks']:
                        if table in tables:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                                record_counts[table] = cursor.fetchone()[0]
                            except:
                                record_counts[table] = 'error'
                    
                    details['record_counts'] = record_counts
                    
                    return True, f"Database healthy ({len(tables)} tables)", details
                    
            except Exception as e:
                return False, f"Database error: {e}", {'path': str(db_path), 'error': str(e)}
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="database",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system memory and CPU resources"""
        def check():
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent
            
            details = {
                'memory': {
                    'total_gb': round(memory_gb, 2),
                    'available_gb': round(available_gb, 2),
                    'used_percent': used_percent,
                    'free_gb': round(memory.free / (1024**3), 2)
                },
                'cpu': {
                    'count': cpu_count,
                    'current_percent': cpu_percent,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            }
            
            # Check if we have sufficient resources
            issues = []
            if available_gb < 1.0:
                issues.append(f"Low memory: {available_gb:.1f}GB available")
            if used_percent > 90:
                issues.append(f"High memory usage: {used_percent:.1f}%")
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if issues:
                return False, "; ".join(issues), details
            
            return True, f"Resources OK: {available_gb:.1f}GB RAM, {cpu_count} CPUs", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        def check():
            disk = psutil.disk_usage('.')
            
            total_gb = disk.total / (1024**3)
            free_gb = disk.free / (1024**3)
            used_gb = disk.used / (1024**3)
            used_percent = (disk.used / disk.total) * 100
            
            details = {
                'total_gb': round(total_gb, 2),
                'free_gb': round(free_gb, 2),
                'used_gb': round(used_gb, 2),
                'used_percent': round(used_percent, 1)
            }
            
            if free_gb < 1.0:
                return False, f"Low disk space: {free_gb:.1f}GB free", details
            
            return True, f"Disk space OK: {free_gb:.1f}GB free ({used_percent:.1f}% used)", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity (for model downloads)"""
        def check():
            try:
                # Simple connectivity test
                import urllib.request
                import socket
                
                socket.setdefaulttimeout(5)  # 5 second timeout
                
                test_urls = [
                    'https://huggingface.co',  # For downloading models
                    'https://github.com'       # General connectivity
                ]
                
                results = {}
                for url in test_urls:
                    try:
                        urllib.request.urlopen(url, timeout=5)
                        results[url] = 'accessible'
                    except Exception as e:
                        results[url] = f'error: {e}'
                
                details = {'connectivity_tests': results}
                
                accessible_count = sum(1 for status in results.values() if status == 'accessible')
                
                if accessible_count == 0:
                    return False, "No network connectivity", details
                elif accessible_count < len(test_urls):
                    return True, f"Limited connectivity ({accessible_count}/{len(test_urls)})", details
                else:
                    return True, "Network connectivity OK", details
                    
            except Exception as e:
                return False, f"Network check failed: {e}", {'error': str(e)}
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="network_connectivity",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_file_permissions(self) -> HealthCheckResult:
        """Check file system permissions"""
        def check():
            test_paths = [
                Path('./data'),
                Path('./logs'),
                Path('./models'),
                Path('./config')
            ]
            
            permissions = {}
            issues = []
            
            for path in test_paths:
                try:
                    # Create directory if it doesn't exist
                    path.mkdir(exist_ok=True)
                    
                    # Test write permissions
                    test_file = path / '.permission_test'
                    test_file.write_text('test')
                    test_file.unlink()  # Clean up
                    
                    permissions[str(path)] = 'read_write_ok'
                except Exception as e:
                    permissions[str(path)] = f'error: {e}'
                    issues.append(f"{path}: {e}")
            
            details = {'permissions': permissions}
            
            if issues:
                return False, f"Permission issues: {'; '.join(issues)}", details
            
            return True, "File permissions OK", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="file_permissions",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check system configuration"""
        def check():
            # Gather key configuration values
            log_level = self.config_manager.get_param('logging.level', 'INFO')
            max_memory_gb = self.config_manager.get_param('performance.memory_limit_gb', 8)
            db_path = self.config_manager.get_param('database.path', 'data/rag_vectors.db')

            try:
                from src.config_manager import ExperimentConfig
                default_llm = ExperimentConfig().llm_model_path
                default_embed = ExperimentConfig().embedding_model_path
            except Exception:
                default_llm = 'models/gemma-3-4b-it-q4_0.gguf'
                default_embed = 'sentence-transformers/all-MiniLM-L6-v2'

            llm_model_path = self.config_manager.get_param('llm_model_path', default_llm)
            embedding_model_path = self.config_manager.get_param('embedding_model_path', default_embed)

            config_status = {
                'db_path': str(db_path),
                'llm_model_path': str(llm_model_path),
                'embedding_model_path': str(embedding_model_path),
                'log_level': log_level,
                'max_memory_gb': max_memory_gb
            }
            
            # Validate configuration
            issues = []
            
            if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                issues.append(f"Invalid log level: {log_level}")
            
            try:
                if float(max_memory_gb) <= 0:
                    issues.append(f"Invalid max memory: {max_memory_gb}")
            except Exception:
                issues.append(f"Invalid max memory value: {max_memory_gb}")
            
            details = {'configuration': config_status, 'validation_issues': issues}
            
            if issues:
                return False, f"Configuration issues: {'; '.join(issues)}", details
            
            return True, "Configuration valid", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="configuration",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _check_component_status(self) -> HealthCheckResult:
        """Check loaded components status"""
        def check():
            loaded_components = list(self.components.keys()) if hasattr(self, 'components') else []
            
            component_status = {}
            for name, component in getattr(self, 'components', {}).items():
                try:
                    # Try to get component info
                    if hasattr(component, 'get_status'):
                        status = component.get_status()
                    elif hasattr(component, '__class__'):
                        status = f"Loaded ({component.__class__.__name__})"
                    else:
                        status = "Loaded (unknown type)"
                    
                    component_status[name] = {'status': 'healthy', 'details': status}
                except Exception as e:
                    component_status[name] = {'status': 'error', 'details': str(e)}
            
            details = {
                'loaded_count': len(loaded_components),
                'components': component_status
            }
            
            error_count = sum(1 for comp in component_status.values() if comp['status'] == 'error')
            
            if error_count > 0:
                return False, f"Component errors ({error_count}/{len(loaded_components)})", details
            
            if loaded_components:
                return True, f"{len(loaded_components)} components healthy", details
            else:
                return True, "No components loaded (lazy initialization)", details
        
        result, exec_time = self._time_check(check)
        status, message, details = result
        
        return HealthCheckResult(
            name="component_status",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=exec_time
        )
    
    def _generate_summary(self, checks: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate health check summary"""
        passed = sum(1 for check in checks if check.status)
        failed = len(checks) - passed
        
        total_time = sum(check.execution_time_ms for check in checks)
        
        return {
            'total_checks': len(checks),
            'passed': passed,
            'failed': failed,
            'success_rate': round((passed / len(checks)) * 100, 1),
            'total_execution_time_ms': round(total_time, 2),
            'failed_checks': [check.name for check in checks if not check.status]
        }
    
    def _generate_recommendations(self, checks: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations based on failed checks"""
        recommendations = []
        
        for check in checks:
            if not check.status:
                if check.name == 'python_environment':
                    recommendations.append("Upgrade Python to version 3.8 or higher")
                elif check.name == 'dependencies':
                    recommendations.append("Install missing dependencies: pip install -r requirements.txt")
                elif check.name == 'models':
                    recommendations.append("Download required models or check model paths in configuration")
                elif check.name == 'database':
                    recommendations.append("Check database path and permissions, ensure SQLite is working")
                elif check.name == 'system_resources':
                    recommendations.append("Free up system memory or upgrade hardware")
                elif check.name == 'disk_space':
                    recommendations.append("Free up disk space or change data directory location")
                elif check.name == 'network_connectivity':
                    recommendations.append("Check internet connection for model downloads")
                elif check.name == 'file_permissions':
                    recommendations.append("Fix file system permissions for data directories")
                elif check.name == 'configuration':
                    recommendations.append("Review and fix configuration settings")
                elif check.name == 'component_status':
                    recommendations.append("Restart system or check component initialization")
        
        if not recommendations:
            recommendations.append("System is healthy - no action required")
        
        return recommendations
    
    def generate_report(self, report: SystemHealthReport, format: str = 'markdown') -> str:
        """Generate formatted health report"""
        if format == 'markdown':
            return self._generate_markdown_report(report)
        elif format == 'json':
            return json.dumps(asdict(report), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, report: SystemHealthReport) -> str:
        """Generate markdown health report"""
        status_emoji = "✅" if report.overall_status else "❌"
        
        md = f"""# System Health Report {status_emoji}

**Generated:** {report.timestamp}
**Overall Status:** {'HEALTHY' if report.overall_status else 'UNHEALTHY'}

## Summary
- **Total Checks:** {report.summary['total_checks']}
- **Passed:** {report.summary['passed']} ✅
- **Failed:** {report.summary['failed']} ❌
- **Success Rate:** {report.summary['success_rate']}%
- **Execution Time:** {report.summary['total_execution_time_ms']:.2f}ms

## Check Results

"""
        for check in report.checks:
            status_icon = "✅" if check.status else "❌"
            md += f"### {check.name.replace('_', ' ').title()} {status_icon}\n"
            md += f"**Status:** {check.message}  \n"
            md += f"**Execution Time:** {check.execution_time_ms:.2f}ms  \n\n"
            
            if check.details and not check.status:
                md += "**Details:**\n```json\n"
                md += json.dumps(check.details, indent=2)
                md += "\n```\n\n"
        
        if report.recommendations:
            md += "## Recommendations\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md += f"{i}. {rec}\n"
        
        return md
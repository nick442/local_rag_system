#!/usr/bin/env python3
"""
Installation and Setup Verification Script
Verifies that the RAG system is properly installed and configured
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        return False, f"Python {version.major}.{version.minor} (requires {required_major}.{required_minor}+)"
    
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    env_info = "Virtual Environment" if in_venv else "System Python"
    
    # Get environment name if available
    if in_venv and 'CONDA_DEFAULT_ENV' in os.environ:
        env_info += f" ({os.environ['CONDA_DEFAULT_ENV']})"
    elif in_venv and 'VIRTUAL_ENV' in os.environ:
        venv_name = Path(os.environ['VIRTUAL_ENV']).name
        env_info += f" ({venv_name})"
    
    return in_venv, env_info

def check_package_installation():
    """Check critical package installation"""
    critical_packages = {
        'torch': 'PyTorch (CPU/GPU support)',
        'sentence_transformers': 'Sentence Transformers (embeddings)',
        'llama_cpp': 'llama.cpp Python bindings (LLM)',
        'click': 'Click CLI framework',
        'rich': 'Rich terminal formatting',
        'psutil': 'System monitoring',
        'numpy': 'Numerical computing',
        'sqlite3': 'SQLite database (built-in)'
    }
    
    installed_packages = {}
    missing_packages = []
    
    for package, description in critical_packages.items():
        try:
            if package == 'sqlite3':
                # Built-in module
                import sqlite3
                installed_packages[package] = {
                    'version': 'built-in',
                    'description': description,
                    'status': 'OK'
                }
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_packages[package] = {
                    'version': version,
                    'description': description,
                    'status': 'OK'
                }
        except ImportError:
            missing_packages.append({
                'package': package,
                'description': description
            })
    
    return installed_packages, missing_packages

def check_gpu_support():
    """Check GPU/Metal support availability"""
    gpu_info = {
        'cuda': {'available': False, 'details': 'Not available'},
        'mps': {'available': False, 'details': 'Not available'},
        'metal': {'available': False, 'details': 'Not available'}
    }
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            gpu_info['cuda'] = {
                'available': True,
                'details': f"{torch.cuda.device_count()} devices, {torch.cuda.get_device_name()}"
            }
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['mps'] = {
                'available': True,
                'details': 'Apple Silicon Metal Performance Shaders'
            }
        
        # Check general Metal support
        try:
            if sys.platform == 'darwin':  # macOS
                import platform
                if platform.processor() == 'arm':  # Apple Silicon
                    gpu_info['metal'] = {
                        'available': True,
                        'details': 'Apple Silicon with Metal support'
                    }
        except:
            pass
            
    except ImportError:
        pass
    
    return gpu_info

def check_model_files():
    """Check for required model files"""
    model_paths = {
        'llm_model': [
            'models/gemma-3-4b-it-q4_0.gguf',
            'models/llama-3.2-3b-instruct-q4_0.gguf',
            'models/*.gguf'
        ],
        'embedding_model': [
            'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2',
            'models/embedding'
        ]
    }
    
    found_models = {}
    
    for model_type, paths in model_paths.items():
        found_models[model_type] = []
        
        for path_pattern in paths:
            path = Path(path_pattern)
            
            if path.suffix == '.gguf' and '*' in str(path):
                # Handle glob pattern for GGUF files
                parent = path.parent
                if parent.exists():
                    gguf_files = list(parent.glob('*.gguf'))
                    found_models[model_type].extend([str(f) for f in gguf_files])
            elif path.exists():
                if path.is_file():
                    found_models[model_type].append(str(path))
                elif path.is_dir() and any(path.iterdir()):
                    found_models[model_type].append(str(path))
    
    return found_models

def check_data_directories():
    """Check data directory structure"""
    required_dirs = ['data', 'logs', 'models', 'config']
    optional_dirs = ['test_data', 'reports', 'scripts']
    
    dir_status = {}
    
    for dirname in required_dirs + optional_dirs:
        path = Path(dirname)
        is_required = dirname in required_dirs
        
        dir_status[dirname] = {
            'exists': path.exists(),
            'is_directory': path.is_dir() if path.exists() else False,
            'is_required': is_required,
            'writable': os.access(path, os.W_OK) if path.exists() else False
        }
    
    return dir_status

def check_database():
    """Check database accessibility"""
    db_path = Path('data/rag_vectors.db')
    
    if not db_path.exists():
        return False, "Database file not found", {}
    
    try:
        import sqlite3
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check SQLite version
            cursor.execute("SELECT sqlite_version()")
            sqlite_version = cursor.fetchone()[0]
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get database size
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            
            details = {
                'sqlite_version': sqlite_version,
                'tables': tables,
                'size_mb': round(db_size_mb, 2)
            }
            
            return True, f"Database OK ({len(tables)} tables, {db_size_mb:.1f}MB)", details
            
    except Exception as e:
        return False, f"Database error: {e}", {}

def run_setup_check(verbose=False, quick=False):
    """Run comprehensive setup verification"""
    console = Console()
    
    checks = []
    
    # Python version check
    py_ok, py_info = check_python_version()
    checks.append({
        'name': 'Python Version',
        'status': py_ok,
        'message': py_info,
        'critical': True
    })
    
    # Virtual environment check
    venv_ok, venv_info = check_virtual_environment()
    checks.append({
        'name': 'Virtual Environment',
        'status': venv_ok,
        'message': venv_info,
        'critical': False  # Recommended but not critical
    })
    
    if not quick:
        # Package installation check
        installed, missing = check_package_installation()
        pkg_ok = len(missing) == 0
        pkg_msg = f"{len(installed)} installed" + (f", {len(missing)} missing" if missing else "")
        checks.append({
            'name': 'Package Installation',
            'status': pkg_ok,
            'message': pkg_msg,
            'critical': True,
            'details': {'installed': installed, 'missing': missing}
        })
        
        # GPU support check
        gpu_info = check_gpu_support()
        gpu_available = any(info['available'] for info in gpu_info.values())
        gpu_msg = "Available" if gpu_available else "CPU only"
        if gpu_available:
            available_types = [name for name, info in gpu_info.items() if info['available']]
            gpu_msg += f" ({', '.join(available_types)})"
        
        checks.append({
            'name': 'GPU/Metal Support',
            'status': gpu_available,
            'message': gpu_msg,
            'critical': False,
            'details': gpu_info
        })
        
        # Model files check
        models = check_model_files()
        llm_ok = len(models.get('llm_model', [])) > 0
        emb_ok = len(models.get('embedding_model', [])) > 0
        model_msg = f"LLM: {'✓' if llm_ok else '✗'}, Embeddings: {'✓' if emb_ok else '✗'}"
        checks.append({
            'name': 'Model Files',
            'status': llm_ok and emb_ok,
            'message': model_msg,
            'critical': True,
            'details': models
        })
        
        # Directory structure check
        dirs = check_data_directories()
        required_ok = all(dirs[d]['exists'] for d in dirs if dirs[d]['is_required'])
        total_dirs = len(dirs)
        existing_dirs = sum(1 for d in dirs.values() if d['exists'])
        dir_msg = f"{existing_dirs}/{total_dirs} directories exist"
        checks.append({
            'name': 'Directory Structure',
            'status': required_ok,
            'message': dir_msg,
            'critical': True,
            'details': dirs
        })
        
        # Database check
        db_ok, db_msg, db_details = check_database()
        checks.append({
            'name': 'Database',
            'status': db_ok,
            'message': db_msg,
            'critical': False,  # May not exist on fresh install
            'details': db_details
        })
    
    return checks

def main():
    """Main setup check function"""
    parser = argparse.ArgumentParser(description="RAG System Setup Verification")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Quick check only')
    parser.add_argument('--json', action='store_true', help='JSON output')
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        rprint("[yellow]🔧 Verifying RAG system setup...[/yellow]")
        
        with Progress() as progress:
            task = progress.add_task("Running setup checks...", total=None)
            
            checks = run_setup_check(verbose=args.verbose, quick=args.quick)
            
            progress.update(task, completed=100, total=100)
        
        # Calculate overall status
        critical_checks = [c for c in checks if c.get('critical', True)]
        critical_passed = sum(1 for c in critical_checks if c['status'])
        overall_ok = critical_passed == len(critical_checks)
        
        if args.json:
            # JSON output
            import json
            result = {
                'overall_status': overall_ok,
                'critical_checks': len(critical_checks),
                'critical_passed': critical_passed,
                'checks': checks
            }
            print(json.dumps(result, indent=2, default=str))
        else:
            # Rich table output
            status_emoji = "✅" if overall_ok else "❌"
            rprint(f"\n[bold blue]Setup Verification {status_emoji}[/bold blue]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="yellow")
            table.add_column("Critical", style="red")
            
            for check in checks:
                status_icon = "✅" if check['status'] else "❌"
                critical_icon = "🔴" if check.get('critical', True) else "🟡"
                
                table.add_row(
                    check['name'],
                    status_icon,
                    check['message'],
                    critical_icon
                )
            
            console.print(table)
            
            # Show detailed information for failed checks
            failed_checks = [c for c in checks if not c['status']]
            if failed_checks and args.verbose:
                rprint(f"\n[bold red]❌ Failed Checks Details:[/bold red]")
                
                for check in failed_checks:
                    rprint(f"\n[red]• {check['name']}[/red]")
                    rprint(f"  Issue: {check['message']}")
                    
                    if 'details' in check and check['details']:
                        if check['name'] == 'Package Installation' and check['details'].get('missing'):
                            rprint("  Missing packages:")
                            for pkg in check['details']['missing']:
                                rprint(f"    - {pkg['package']}: {pkg['description']}")
                        elif check['name'] == 'Model Files':
                            for model_type, files in check['details'].items():
                                if not files:
                                    rprint(f"    - {model_type}: No files found")
            
            # Show recommendations
            if not overall_ok:
                rprint(f"\n[bold yellow]💡 Setup Recommendations:[/bold yellow]")
                
                if not any(c['status'] for c in checks if c['name'] == 'Virtual Environment'):
                    rprint("  1. Set up a virtual environment (conda or venv)")
                
                failed_critical = [c for c in critical_checks if not c['status']]
                for check in failed_critical:
                    if check['name'] == 'Package Installation':
                        rprint("  2. Install missing packages: pip install -r requirements.txt")
                    elif check['name'] == 'Model Files':
                        rprint("  3. Download required model files")
                        rprint("     - LLM: Download a GGUF model to models/ directory")
                        rprint("     - Embeddings: sentence-transformers will download automatically")
                    elif check['name'] == 'Directory Structure':
                        rprint("  4. Create missing directories: mkdir -p data logs models config")
        
        return 0 if overall_ok else 1
        
    except Exception as e:
        rprint(f"[red]❌ Setup check failed: {e}[/red]")
        if args.verbose:
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
System Reset Script
Resets the RAG system to a clean state while preserving models and configuration
"""

import sys
import os
import shutil
import sqlite3
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

def backup_data(backup_dir: Path, items_to_backup: list):
    """Create backup of important data"""
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
    
    backed_up_items = []
    
    for item_path in items_to_backup:
        source = Path(item_path)
        if source.exists():
            if source.is_file():
                dest = backup_dir / source.name
                shutil.copy2(source, dest)
                backed_up_items.append(str(source))
            elif source.is_dir():
                dest = backup_dir / source.name
                shutil.copytree(source, dest, dirs_exist_ok=True)
                backed_up_items.append(str(source))
    
    return backed_up_items

def reset_database(db_path: Path, keep_schema: bool = True):
    """Reset database while optionally keeping schema"""
    if not db_path.exists():
        return False, "Database doesn't exist"
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if keep_schema:
                # Delete all data but keep schema
                for table in tables:
                    if table != 'sqlite_sequence':  # Don't mess with SQLite internals
                        cursor.execute(f"DELETE FROM {table}")
                        
                # Reset auto-increment counters
                cursor.execute("DELETE FROM sqlite_sequence")
                
                return True, f"Data cleared from {len(tables)} tables, schema preserved"
            else:
                # Drop all tables
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                return True, f"Dropped {len(tables)} tables, database empty"
                
    except Exception as e:
        return False, f"Database reset failed: {e}"

def clear_directory(dir_path: Path, keep_structure: bool = True):
    """Clear directory contents"""
    if not dir_path.exists():
        return 0, "Directory doesn't exist"
    
    if not dir_path.is_dir():
        return 0, "Not a directory"
    
    items_removed = 0
    
    try:
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
                items_removed += 1
            elif item.is_dir():
                if keep_structure:
                    # Remove contents but keep directory
                    for subitem in item.rglob('*'):
                        if subitem.is_file():
                            subitem.unlink()
                            items_removed += 1
                    # Remove empty subdirectories
                    for subdir in sorted(item.glob('**/*'), key=lambda x: str(x), reverse=True):
                        if subdir.is_dir() and not any(subdir.iterdir()):
                            subdir.rmdir()
                else:
                    # Remove entire subdirectory
                    shutil.rmtree(item)
                    items_removed += 1
        
        return items_removed, "Directory cleared"
    except Exception as e:
        return items_removed, f"Partial clear: {e}"

def reset_configuration(config_dir: Path, reset_to_defaults: bool = False):
    """Reset configuration to defaults"""
    if not config_dir.exists():
        return False, "Config directory doesn't exist"
    
    try:
        config_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.json'))
        
        if reset_to_defaults:
            # Remove all config files to trigger default generation
            for config_file in config_files:
                config_file.unlink()
            return True, f"Removed {len(config_files)} config files, will regenerate defaults"
        else:
            # Just clear temporary settings
            temp_files = config_dir.glob('*_temp.*')
            temp_count = 0
            for temp_file in temp_files:
                temp_file.unlink()
                temp_count += 1
            
            return True, f"Cleared {temp_count} temporary config files"
    except Exception as e:
        return False, f"Config reset failed: {e}"

def get_reset_plan(options):
    """Generate reset plan based on options"""
    plan = {
        'database': {
            'action': 'clear_data' if options.keep_schema else 'drop_tables',
            'description': 'Clear all documents and embeddings' if options.keep_schema else 'Drop all database tables'
        },
        'logs': {
            'action': 'clear' if options.clear_logs else 'skip',
            'description': 'Clear all log files' if options.clear_logs else 'Keep log files'
        },
        'cache': {
            'action': 'clear',
            'description': 'Clear temporary cache files'
        },
        'config': {
            'action': 'reset_to_defaults' if options.reset_config else 'clear_temp',
            'description': 'Reset to default configuration' if options.reset_config else 'Clear temporary config overrides'
        },
        'models': {
            'action': 'keep' if options.keep_models else 'remove',
            'description': 'Preserve model files' if options.keep_models else 'Remove downloaded models'
        }
    }
    
    return plan

def execute_reset(options):
    """Execute the reset operation"""
    console = Console()
    
    # Create backup if requested
    backup_dir = None
    if options.backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(f'backups/reset_backup_{timestamp}')
        
        items_to_backup = ['data/rag_vectors.db', 'config']
        if not options.clear_logs:
            items_to_backup.append('logs')
        
        rprint("[yellow]📦 Creating backup...[/yellow]")
        backed_up = backup_data(backup_dir, items_to_backup)
        rprint(f"[green]✓ Backup created: {backup_dir}[/green]")
        rprint(f"[dim]Backed up: {', '.join(backed_up)}[/dim]")
    
    reset_results = {}
    
    # Reset database
    rprint("[yellow]🗄️  Resetting database...[/yellow]")
    db_path = Path('data/rag_vectors.db')
    db_success, db_msg = reset_database(db_path, keep_schema=options.keep_schema)
    reset_results['database'] = {'success': db_success, 'message': db_msg}
    
    if db_success:
        rprint(f"[green]✓ Database: {db_msg}[/green]")
    else:
        rprint(f"[red]✗ Database: {db_msg}[/red]")
    
    # Clear logs if requested
    if options.clear_logs:
        rprint("[yellow]📝 Clearing logs...[/yellow]")
        logs_dir = Path('logs')
        log_count, log_msg = clear_directory(logs_dir, keep_structure=True)
        reset_results['logs'] = {'success': True, 'message': f"Cleared {log_count} log files"}
        rprint(f"[green]✓ Logs: {log_msg} ({log_count} items)[/green]")
    
    # Clear cache directories
    rprint("[yellow]🧹 Clearing cache...[/yellow]")
    cache_dirs = ['.serena/cache', '__pycache__', 'src/__pycache__']
    cache_cleared = 0
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            count, msg = clear_directory(cache_path, keep_structure=False)
            cache_cleared += count
    
    reset_results['cache'] = {'success': True, 'message': f"Cleared {cache_cleared} cache items"}
    rprint(f"[green]✓ Cache: Cleared {cache_cleared} items[/green]")
    
    # Reset configuration
    rprint("[yellow]⚙️  Resetting configuration...[/yellow]")
    config_dir = Path('config')
    config_success, config_msg = reset_configuration(config_dir, reset_to_defaults=options.reset_config)
    reset_results['config'] = {'success': config_success, 'message': config_msg}
    
    if config_success:
        rprint(f"[green]✓ Config: {config_msg}[/green]")
    else:
        rprint(f"[red]✗ Config: {config_msg}[/red]")
    
    # Handle models
    if not options.keep_models:
        rprint("[yellow]🧠 Removing models...[/yellow]")
        models_dir = Path('models')
        if models_dir.exists():
            model_count, model_msg = clear_directory(models_dir, keep_structure=True)
            reset_results['models'] = {'success': True, 'message': f"Removed {model_count} model files"}
            rprint(f"[green]✓ Models: {model_msg} ({model_count} items)[/green]")
        else:
            reset_results['models'] = {'success': True, 'message': "No models directory found"}
            rprint("[dim]No models directory found[/dim]")
    else:
        reset_results['models'] = {'success': True, 'message': "Models preserved"}
        rprint("[dim]Models preserved[/dim]")
    
    # Clear system state files
    rprint("[yellow]🔧 Clearing system state...[/yellow]")
    state_files = ['data/system_state.json']
    state_cleared = 0
    
    for state_file in state_files:
        state_path = Path(state_file)
        if state_path.exists():
            state_path.unlink()
            state_cleared += 1
    
    reset_results['system_state'] = {'success': True, 'message': f"Cleared {state_cleared} state files"}
    rprint(f"[green]✓ System state: Cleared {state_cleared} files[/green]")
    
    return reset_results, backup_dir

def main():
    """Main reset function"""
    parser = argparse.ArgumentParser(description="Reset RAG System to Clean State")
    parser.add_argument('--keep-models', action='store_true', default=True,
                       help='Preserve model files (default: True)')
    parser.add_argument('--remove-models', dest='keep_models', action='store_false',
                       help='Remove model files')
    parser.add_argument('--keep-schema', action='store_true', default=True,
                       help='Keep database schema, only clear data (default: True)')
    parser.add_argument('--drop-tables', dest='keep_schema', action='store_false',
                       help='Drop all database tables')
    parser.add_argument('--clear-logs', action='store_true',
                       help='Clear log files')
    parser.add_argument('--reset-config', action='store_true',
                       help='Reset configuration to defaults')
    parser.add_argument('--backup/--no-backup', default=True,
                       help='Create backup before reset (default: True)')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be reset without doing it')
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Show reset plan
        plan = get_reset_plan(args)
        
        rprint("[bold blue]🔄 RAG System Reset Plan[/bold blue]")
        
        plan_table = Table(show_header=True, header_style="bold magenta")
        plan_table.add_column("Component", style="cyan")
        plan_table.add_column("Action", style="yellow")
        plan_table.add_column("Description", style="white")
        
        for component, details in plan.items():
            plan_table.add_row(
                component.title(),
                details['action'],
                details['description']
            )
        
        console.print(plan_table)
        
        # Show backup info
        if args.backup:
            rprint(f"\n[green]💾 Backup will be created before reset[/green]")
        else:
            rprint(f"\n[red]⚠️  No backup will be created[/red]")
        
        if args.dry_run:
            rprint(f"\n[yellow]🔍 DRY RUN - No changes will be made[/yellow]")
            return 0
        
        # Confirmation
        if not args.force:
            rprint(f"\n[bold red]⚠️  This will reset your RAG system![/bold red]")
            
            if not args.backup:
                rprint("[red]No backup will be created. Data will be permanently lost.[/red]")
            
            response = input("\nProceed with reset? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                rprint("[yellow]Reset cancelled[/yellow]")
                return 0
        
        # Execute reset
        rprint(f"\n[bold yellow]🚀 Starting system reset...[/bold yellow]")
        
        with Progress() as progress:
            task = progress.add_task("Resetting system...", total=None)
            
            reset_results, backup_dir = execute_reset(args)
            
            progress.update(task, completed=100, total=100)
        
        # Show summary
        rprint(f"\n[bold green]✅ System reset completed![/bold green]")
        
        successful_operations = sum(1 for result in reset_results.values() if result['success'])
        total_operations = len(reset_results)
        
        rprint(f"Operations: {successful_operations}/{total_operations} successful")
        
        if backup_dir:
            rprint(f"Backup location: {backup_dir}")
        
        # Show what to do next
        rprint(f"\n[bold blue]💡 Next Steps:[/bold blue]")
        rprint("1. System has been reset to clean state")
        
        if not args.keep_models:
            rprint("2. Download required models before using the system")
        
        if args.reset_config:
            rprint("3. Configuration will be regenerated on next startup")
        
        rprint("4. Run 'python scripts/setup_check.py' to verify setup")
        rprint("5. Start with 'python main.py status' to check system")
        
        return 0
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Reset cancelled by user[/yellow]")
        return 1
    except Exception as e:
        rprint(f"[red]❌ Reset failed: {e}[/red]")
        import traceback
        rprint(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

if __name__ == '__main__':
    sys.exit(main())
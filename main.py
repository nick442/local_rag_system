#!/usr/bin/env python3
"""
RAG System Command Line Interface

Main entry point for the local RAG system with comprehensive corpus management,
document ingestion, and analytics capabilities.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

# Import RAG system components
from src.corpus_manager import CorpusManager, create_corpus_manager
from src.corpus_organizer import CorpusOrganizer, create_corpus_organizer
from src.deduplication import DocumentDeduplicator, create_deduplicator
from src.reindex import ReindexTool, create_reindex_tool
from src.corpus_analytics import CorpusAnalyzer, create_corpus_analyzer
from src.rag_pipeline import RAGPipeline
from src.cli_chat import chat as chat_main
from src.config_manager import ConfigManager


# Global configuration
console = Console()
DEFAULT_DB_PATH = "data/rag_vectors.db"
DEFAULT_EMBEDDING_PATH = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_LLM_PATH = "models/gemma-3-4b-it-q4_0.gguf"


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Ensure logs directory exists before attaching file handler (important in isolated FS/tests)
    try:
        Path('logs').mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create the directory, continue with console logging only
        pass

    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler('logs/rag_system.log'))
    except Exception:
        # In environments without filesystem access, skip file logging
        pass

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


@click.group()
@click.option('--db-path', default=DEFAULT_DB_PATH, help='Path to vector database')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, db_path: str, verbose: bool):
    """RAG System - Local Retrieval-Augmented Generation Platform"""
    import signal
    from src.config_manager import ConfigManager
    from src.error_handler import ErrorHandler, ErrorContext
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Override database.path if provided (use dotted keys expected by ConfigManager)
    if db_path != DEFAULT_DB_PATH:
        config_manager.override_param('database.path', db_path)
    
    # Set logging level (use dotted key)
    log_level = "DEBUG" if verbose else "INFO"
    config_manager.override_param('logging.level', log_level)
    
    # For now, skip ErrorHandler until we refactor it in a later phase
    # error_handler = ErrorHandler()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully"""
        console.print("[yellow]Shutting down system...[/yellow]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging
    setup_logging(verbose)
    
    # Store components in context for commands
    ctx.ensure_object(dict)
    ctx.obj['config_manager'] = config_manager
    # ctx.obj['error_handler'] = error_handler  # Skip until refactored
    ctx.obj['db_path'] = db_path
    ctx.obj['verbose'] = verbose
    
    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


# ========== INGESTION COMMANDS ==========

@cli.group()
def ingest():
    """Document ingestion and corpus management commands"""
    pass


@ingest.command('directory')
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('--pattern', default='**/*', help='File pattern to match')
@click.option('--collection', default='default', help='Target collection')
@click.option('--max-workers', default=4, help='Number of parallel workers')
@click.option('--batch-size', default=32, help='Embedding batch size')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.option('--dry-run', is_flag=True, help='Preview without processing')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoint')
@click.option('--deduplicate/--no-deduplicate', default=True, help='Skip duplicates')
@click.pass_context
def ingest_directory(ctx, path: Path, pattern: str, collection: str, max_workers: int, 
                    batch_size: int, embedding_path: str, dry_run: bool, resume: bool, deduplicate: bool):
    """Ingest documents from directory with parallel processing"""
    
    async def run_ingestion():
        try:
            manager = create_corpus_manager(
                db_path=ctx.obj['db_path'],
                embedding_model_path=embedding_path,
                max_workers=max_workers,
                batch_size=batch_size
            )
            
            with Progress() as progress:
                task = progress.add_task(f"Ingesting from {path}", total=None)
                
                stats = await manager.ingest_directory(
                    path=path,
                    pattern=pattern,
                    collection_id=collection,
                    dry_run=dry_run,
                    resume=resume,
                    deduplicate=deduplicate
                )
                
                progress.update(task, completed=100, total=100)
            
            # Display results
            if dry_run:
                rprint(f"[yellow]DRY RUN - Would process {stats.files_scanned} files[/yellow]")
            else:
                rprint(f"[green]✓ Processing complete![/green]")
                rprint(f"📁 Files processed: {stats.files_processed}")
                rprint(f"📄 Chunks created: {stats.chunks_created}")
                rprint(f"🧠 Embeddings generated: {stats.chunks_embedded}")
                rprint(f"⏱️  Processing time: {stats.processing_time:.2f}s")
                
                if stats.files_failed > 0:
                    rprint(f"[red]⚠️  Failed files: {stats.files_failed}[/red]")
        
        except Exception as e:
            rprint(f"[red]❌ Ingestion failed: {e}[/red]")
            if ctx.obj['verbose']:
                import traceback
                rprint(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(run_ingestion())


@ingest.command('file')
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--collection', default='default', help='Target collection')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.pass_context
def ingest_file(ctx, file_path: Path, collection: str, embedding_path: str):
    """Ingest a single document file"""
    
    async def run_single_ingestion():
        try:
            manager = create_corpus_manager(
                db_path=ctx.obj['db_path'],
                embedding_model_path=embedding_path
            )
            
            # Process single file (use directory ingestion with specific pattern)
            stats = await manager.ingest_directory(
                path=file_path.parent,
                pattern=file_path.name,
                collection_id=collection,
                dry_run=False
            )
            
            if stats.files_processed > 0:
                rprint(f"[green]✓ File processed successfully![/green]")
                rprint(f"📄 Chunks created: {stats.chunks_created}")
            else:
                rprint(f"[red]❌ Failed to process file[/red]")
        
        except Exception as e:
            rprint(f"[red]❌ File ingestion failed: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_single_ingestion())


# ========== COLLECTION COMMANDS ==========

@cli.group()
def collection():
    """Collection management commands"""
    pass


@collection.command('create')
@click.argument('name')
@click.option('--description', default='', help='Collection description')
@click.option('--collection-id', help='Custom collection ID')
@click.pass_context
def create_collection(ctx, name: str, description: str, collection_id: Optional[str]):
    """Create a new document collection"""
    try:
        organizer = create_corpus_organizer(ctx.obj['db_path'])
        
        collection_id = organizer.create_collection(
            name=name,
            description=description,
            collection_id=collection_id
        )
        
        rprint(f"[green]✓ Created collection: {collection_id}[/green]")
        rprint(f"📝 Name: {name}")
        rprint(f"📄 Description: {description}")
        
    except Exception as e:
        rprint(f"[red]❌ Failed to create collection: {e}[/red]")
        sys.exit(1)


@collection.command('list')
@click.pass_context
def list_collections(ctx):
    """List all collections"""
    try:
        organizer = create_corpus_organizer(ctx.obj['db_path'])
        collections = organizer.list_collections()
        
        if not collections:
            rprint("[yellow]No collections found[/yellow]")
            return
        
        table = Table(title="Document Collections")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Documents", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Created", style="dim")
        
        for collection in collections:
            # Update stats before displaying
            organizer.update_collection_stats(collection.collection_id)
            updated_collection = organizer.get_collection(collection.collection_id)
            
            table.add_row(
                updated_collection.collection_id,
                updated_collection.name,
                str(updated_collection.document_count),
                str(updated_collection.chunk_count),
                f"{updated_collection.size_mb:.2f}",
                updated_collection.created_at.strftime("%Y-%m-%d")
            )
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]❌ Failed to list collections: {e}[/red]")
        sys.exit(1)


@collection.command('switch')
@click.argument('collection_id')
@click.pass_context
def switch_collection(ctx, collection_id: str):
    """Switch to a different collection"""
    try:
        organizer = create_corpus_organizer(ctx.obj['db_path'])
        organizer.switch_collection(collection_id)
        
        rprint(f"[green]✓ Switched to collection: {collection_id}[/green]")
        
    except Exception as e:
        rprint(f"[red]❌ Failed to switch collection: {e}[/red]")
        sys.exit(1)


@collection.command('delete')
@click.argument('collection_id')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def delete_collection(ctx, collection_id: str, confirm: bool):
    """Delete a collection and all its documents"""
    try:
        organizer = create_corpus_organizer(ctx.obj['db_path'])
        
        if not confirm:
            collection = organizer.get_collection(collection_id)
            if collection:
                if not click.confirm(f"Delete collection '{collection_id}' with {collection.document_count} documents?"):
                    rprint("[yellow]Operation cancelled[/yellow]")
                    return
        
        success = organizer.delete_collection(collection_id, confirm=True)
        
        if success:
            rprint(f"[green]✓ Deleted collection: {collection_id}[/green]")
        else:
            rprint(f"[yellow]Collection deletion cancelled or failed[/yellow]")
        
    except Exception as e:
        rprint(f"[red]❌ Failed to delete collection: {e}[/red]")
        sys.exit(1)


@collection.command('export')
@click.argument('collection_id')
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--include-embeddings', is_flag=True, help='Include embeddings (large files)')
@click.pass_context
def export_collection(ctx, collection_id: str, output_path: Path, include_embeddings: bool):
    """Export collection to backup file"""
    try:
        organizer = create_corpus_organizer(ctx.obj['db_path'])
        
        with Progress() as progress:
            task = progress.add_task("Exporting collection...", total=None)
            
            stats = organizer.export_collection(
                collection_id=collection_id,
                export_path=str(output_path),
                include_embeddings=include_embeddings
            )
            
            progress.update(task, completed=100, total=100)
        
        rprint(f"[green]✓ Export complete![/green]")
        rprint(f"📁 Documents: {stats['documents_exported']}")
        rprint(f"📄 Chunks: {stats['chunks_exported']}")
        rprint(f"🧠 Embeddings: {stats['embeddings_exported']}")
        rprint(f"💾 File size: {stats['file_size_mb']:.2f}MB")
        rprint(f"📍 Location: {stats['export_path']}")
        
    except Exception as e:
        rprint(f"[red]❌ Export failed: {e}[/red]")
        sys.exit(1)


# ========== ANALYTICS COMMANDS ==========

@cli.group()
def analytics():
    """Corpus analytics and reporting commands"""
    pass


@analytics.command('stats')
@click.option('--collection', default='default', help='Collection to analyze')
@click.pass_context
def collection_stats(ctx, collection: str):
    """Display collection statistics"""
    try:
        analyzer = create_corpus_analyzer(ctx.obj['db_path'])
        stats = analyzer.analyze_collection(collection)
        
        rprint(f"[bold blue]Collection: {collection}[/bold blue]")
        rprint(f"📁 Documents: {stats.total_documents}")
        rprint(f"📄 Chunks: {stats.total_chunks}")
        rprint(f"🔤 Tokens: {stats.total_tokens:,}")
        rprint(f"💾 Size: {stats.size_mb:.2f} MB")
        rprint(f"📊 Avg doc size: {stats.avg_document_size:.0f} bytes")
        rprint(f"📝 Avg chunks/doc: {stats.avg_chunks_per_doc:.1f}")
        
        if stats.file_types:
            rprint("\n[bold]File Types:[/bold]")
            for ext, count in stats.file_types.items():
                rprint(f"  {ext}: {count}")
        
        if stats.most_similar_pairs:
            rprint("\n[bold]Most Similar Document Pairs:[/bold]")
            for doc1, doc2, sim in stats.most_similar_pairs:
                rprint(f"  {sim:.3f}: {doc1[:20]}... ↔ {doc2[:20]}...")
        
    except Exception as e:
        rprint(f"[red]❌ Failed to generate stats: {e}[/red]")
        sys.exit(1)


@analytics.command('quality')
@click.option('--collection', default='default', help='Collection to analyze')
@click.pass_context
def quality_report(ctx, collection: str):
    """Generate quality assessment report"""
    try:
        analyzer = create_corpus_analyzer(ctx.obj['db_path'])
        
        with Progress() as progress:
            task = progress.add_task("Analyzing quality...", total=None)
            report = analyzer.generate_quality_report(collection)
            progress.update(task, completed=100, total=100)
        
        rprint(f"[bold blue]Quality Report: {collection}[/bold blue]")
        rprint(f"🏆 Overall Score: {report['overall_quality_score']:.2f} ({report['quality_rating']})")
        
        rprint("\n[bold]Quality Scores:[/bold]")
        for metric, score in report['quality_scores'].items():
            color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
            rprint(f"  {metric}: [{color}]{score:.2f}[/{color}]")
        
        if report['issues']['recommendations']:
            rprint("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in report['issues']['recommendations']:
                rprint(f"  • {rec}")
        
    except Exception as e:
        rprint(f"[red]❌ Quality analysis failed: {e}[/red]")
        sys.exit(1)


@analytics.command('export-report')
@click.option('--collection', default='default', help='Collection to analyze')
@click.option('--output', type=click.Path(path_type=Path), help='Output file path')
@click.pass_context
def export_analytics_report(ctx, collection: str, output: Optional[Path]):
    """Export comprehensive analytics report"""
    try:
        analyzer = create_corpus_analyzer(ctx.obj['db_path'])
        
        with Progress() as progress:
            task = progress.add_task("Generating report...", total=None)
            
            report = analyzer.export_analytics_report(
                collection_id=collection,
                output_path=str(output) if output else None
            )
            
            progress.update(task, completed=100, total=100)
        
        rprint(f"[green]✓ Analytics report exported![/green]")
        rprint(f"📍 Location: {report['export_info']['report_path']}")
        rprint(f"💾 Size: {report['export_info']['file_size_mb']:.2f}MB")
        rprint(f"📊 Sections: {', '.join(report['export_info']['sections_included'])}")
        
    except Exception as e:
        rprint(f"[red]❌ Report export failed: {e}[/red]")
        sys.exit(1)


# ========== MAINTENANCE COMMANDS ==========

# =============================================================================
# Configuration Management Commands
# =============================================================================

@cli.group()
def config():
    """Configuration management commands"""
    pass


@config.command('show')
@click.option('--profile', help='Show specific profile (default: current)')
def show_config(profile: Optional[str]):
    """Show current configuration settings"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        if profile:
            profile_config = config_manager.get_profile(profile)
            rprint(f"[bold blue]Profile: {profile}[/bold blue]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in profile_config.to_dict().items():
                table.add_row(key, str(value))
            
            console.print(table)
        else:
            summary = config_manager.get_config_summary()
            
            rprint(f"[bold blue]Configuration Summary[/bold blue]")
            rprint(f"Current Profile: [green]{summary['current_profile']}[/green]")
            rprint(f"Config File: [cyan]{summary['config_file']}[/cyan]")
            rprint(f"Available Profiles: [yellow]{', '.join(summary['available_profiles'])}[/yellow]")
            
            if summary['temporary_overrides']:
                rprint("\n[bold red]Temporary Overrides:[/bold red]")
                for key, value in summary['temporary_overrides'].items():
                    rprint(f"  {key}: {value}")
            
            rprint("\n[bold blue]Active Settings:[/bold blue]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in summary['active_settings'].items():
                table.add_row(key, str(value))
            
            console.print(table)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to show configuration: {e}[/red]")
        sys.exit(1)


@config.command('list-profiles')
def list_profiles():
    """List all available configuration profiles"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        profiles = config_manager.list_profiles()
        current_profile = config_manager.get_current_profile_name()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Profile", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Retrieval K", style="yellow")
        table.add_column("Max Tokens", style="green")
        table.add_column("Temperature", style="blue")
        
        for name, profile_config in profiles.items():
            status = "✓ Active" if name == current_profile else ""
            table.add_row(
                name,
                status,
                str(profile_config.retrieval_k),
                str(profile_config.max_tokens),
                str(profile_config.temperature)
            )
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to list profiles: {e}[/red]")
        sys.exit(1)


@config.command('switch-profile')
@click.argument('profile_name')
@click.option('--save/--no-save', default=True, help='Save configuration after switching')
def switch_profile(profile_name: str, save: bool):
    """Switch to a different configuration profile"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        if config_manager.switch_profile(profile_name):
            if save:
                config_manager.save_config()
            
            rprint(f"[green]✓ Switched to profile: {profile_name}[/green]")
            
            # Show new profile settings
            profile_config = config_manager.get_profile()
            rprint(f"\n[bold blue]Active Settings:[/bold blue]")
            rprint(f"Retrieval K: [yellow]{profile_config.retrieval_k}[/yellow]")
            rprint(f"Max Tokens: [green]{profile_config.max_tokens}[/green]")
            rprint(f"Temperature: [blue]{profile_config.temperature}[/blue]")
        else:
            rprint(f"[red]✗ Failed to switch to profile: {profile_name}[/red]")
            sys.exit(1)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to switch profile: {e}[/red]")
        sys.exit(1)


@config.command('set')
@click.argument('key')
@click.argument('value')
@click.option('--temporary', '-t', is_flag=True, help='Set temporary override (not saved)')
@click.option('--type', 'value_type', type=click.Choice(['str', 'int', 'float', 'bool']), default='str', help='Value type')
def set_config(key: str, value: str, temporary: bool, value_type: str):
    """Set a configuration parameter"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        # Convert value to appropriate type
        if value_type == 'int':
            typed_value = int(value)
        elif value_type == 'float':
            typed_value = float(value)
        elif value_type == 'bool':
            typed_value = value.lower() in ('true', '1', 'yes', 'on')
        else:
            typed_value = value
        
        if temporary:
            config_manager.override_param(key, typed_value, temporary=True)
            rprint(f"[yellow]✓ Set temporary override: {key} = {typed_value}[/yellow]")
        else:
            if config_manager.set_param(key, typed_value, save=True):
                rprint(f"[green]✓ Set configuration: {key} = {typed_value}[/green]")
            else:
                rprint(f"[red]✗ Failed to set configuration parameter[/red]")
                sys.exit(1)
        
    except ValueError as e:
        rprint(f"[red]✗ Invalid value type: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]✗ Failed to set configuration: {e}[/red]")
        sys.exit(1)


@config.command('get')
@click.argument('key')
def get_config(key: str):
    """Get a configuration parameter value"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        value = config_manager.get_param(key)
        if value is not None:
            rprint(f"[cyan]{key}[/cyan]: [white]{value}[/white]")
        else:
            rprint(f"[red]✗ Parameter '{key}' not found[/red]")
            sys.exit(1)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to get configuration: {e}[/red]")
        sys.exit(1)


@config.command('reload')
def reload_config():
    """Reload configuration from file"""
    try:
        from src.config_manager import create_config_manager
        config_manager = create_config_manager()
        
        if config_manager.reload_config():
            rprint(f"[green]✓ Configuration reloaded successfully[/green]")
        else:
            rprint(f"[red]✗ Failed to reload configuration[/red]")
            sys.exit(1)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to reload configuration: {e}[/red]")
        sys.exit(1)


# =============================================================================
# System Statistics Commands  
# =============================================================================

@cli.group()
def stats():
    """System monitoring and statistics commands"""
    pass


@stats.command('show')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def show_stats(output_format: str):
    """Show current system and session statistics"""
    try:
        from src.monitor import create_monitor
        monitor = create_monitor(enable_continuous_monitoring=False)
        
        if output_format == 'json':
            stats_data = monitor.get_performance_summary()
            rprint(json.dumps(stats_data, indent=2))
        else:
            formatted_stats = monitor.format_stats_display()
            rprint(formatted_stats)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to get statistics: {e}[/red]")
        sys.exit(1)


@stats.command('system')
def show_system_stats():
    """Show current system resource usage"""
    try:
        from src.monitor import create_monitor
        monitor = create_monitor(enable_continuous_monitoring=False)
        
        system_stats = monitor.get_current_system_stats()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="white")
        table.add_column("Details", style="yellow")
        
        table.add_row("CPU", f"{system_stats.cpu_percent:.1f}%", "Current usage")
        table.add_row("Memory", f"{system_stats.memory_percent:.1f}%", f"{system_stats.memory_used_gb:.1f}GB / {system_stats.memory_total_gb:.1f}GB")
        table.add_row("Disk", f"{system_stats.disk_percent:.1f}%", f"{system_stats.disk_used_gb:.1f}GB / {system_stats.disk_total_gb:.1f}GB")
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]✗ Failed to get system statistics: {e}[/red]")
        sys.exit(1)


@stats.command('export')
@click.option('--output', type=click.Path(path_type=Path), help='Output file path')
@click.option('--format', 'export_format', type=click.Choice(['json']), default='json', help='Export format')
def export_stats(output: Optional[Path], export_format: str):
    """Export statistics to file"""
    try:
        from src.monitor import create_monitor
        monitor = create_monitor(enable_continuous_monitoring=False)
        
        stats_data = monitor.export_stats(format=export_format)
        
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(f"stats_export_{timestamp}.{export_format}")
        
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            if export_format == 'json':
                json.dump(stats_data, f, indent=2, default=str)
        
        rprint(f"[green]✓ Statistics exported to: {output}[/green]")
        
    except Exception as e:
        rprint(f"[red]✗ Failed to export statistics: {e}[/red]")
        sys.exit(1)

@cli.group()
def maintenance():
    """Database maintenance and optimization commands"""
    pass


@maintenance.command('dedupe')
@click.option('--collection', default='default', help='Collection to deduplicate')
@click.option('--dry-run', is_flag=True, help='Preview without making changes')
@click.pass_context
def deduplicate(ctx, collection: str, dry_run: bool):
    """Detect and remove duplicate documents"""
    try:
        deduplicator = create_deduplicator(ctx.obj['db_path'])
        
        with Progress() as progress:
            task = progress.add_task("Analyzing duplicates...", total=None)
            report = deduplicator.analyze_duplicates(collection)
            progress.update(task, completed=100, total=100)
        
        rprint(f"[bold blue]Deduplication Report: {collection}[/bold blue]")
        rprint(f"📁 Total documents: {report.total_documents}")
        rprint(f"✨ Unique documents: {report.unique_documents}")
        rprint(f"🔄 Duplicate groups: {len(report.duplicate_groups)}")
        rprint(f"💾 Potential space savings: {report.space_saved_mb:.2f}MB")
        
        if report.duplicate_groups and not dry_run:
            if click.confirm("Proceed with duplicate removal?"):
                with Progress() as progress:
                    task = progress.add_task("Removing duplicates...", total=None)
                    
                    resolution_stats = deduplicator.resolve_duplicates(
                        report.duplicate_groups,
                        collection_id=collection,
                        dry_run=False
                    )
                    
                    progress.update(task, completed=100, total=100)
                
                rprint(f"[green]✓ Deduplication complete![/green]")
                rprint(f"🗑️  Documents removed: {resolution_stats['documents_removed']}")
                rprint(f"💾 Space saved: {resolution_stats['space_saved_mb']:.2f}MB")
        
    except Exception as e:
        rprint(f"[red]❌ Deduplication failed: {e}[/red]")
        sys.exit(1)


@maintenance.command('reindex')
@click.option('--collection', default='default', help='Collection to reindex')
@click.option('--operation', 
              type=click.Choice(['reembed', 'rechunk', 'rebuild', 'vacuum']),
              default='rebuild', help='Reindex operation')
@click.option('--backup/--no-backup', default=True, help='Backup before operation')
@click.pass_context
def reindex(ctx, collection: str, operation: str, backup: bool):
    """Reindex collection (rebuild, reembed, rechunk, vacuum)"""
    try:
        tool = create_reindex_tool(ctx.obj['db_path'])
        
        with Progress() as progress:
            task = progress.add_task(f"Running {operation}...", total=None)
            
            if operation == 'reembed':
                stats = tool.reembed_collection(collection, backup=backup)
            elif operation == 'rechunk':
                stats = tool.rechunk_documents(collection, backup=backup)
            elif operation == 'rebuild':
                stats = tool.rebuild_indices(backup=backup)
            elif operation == 'vacuum':
                stats = tool.vacuum_database(backup=backup)
            
            progress.update(task, completed=100, total=100)
        
        if stats.success:
            rprint(f"[green]✓ {operation.title()} complete![/green]")
            rprint(f"⏱️  Processing time: {stats.processing_time:.2f}s")
            
            if operation in ['reembed', 'rechunk']:
                rprint(f"📁 Documents processed: {stats.documents_processed}")
                rprint(f"📄 Chunks processed: {stats.chunks_processed}")
            
            if operation == 'vacuum' and 'space_saved_mb' in stats.details:
                rprint(f"💾 Space saved: {stats.details['space_saved_mb']:.2f}MB")
        else:
            rprint(f"[red]❌ {operation.title()} failed: {stats.details.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
        
    except Exception as e:
        rprint(f"[red]❌ Reindex failed: {e}[/red]")
        sys.exit(1)


@maintenance.command('validate')
@click.option('--collection', help='Specific collection to validate')
@click.pass_context
def validate_integrity(ctx, collection: Optional[str]):
    """Validate database integrity"""
    try:
        # ✅ FIX: Get embedding model path from context if available
        embedding_model_path = ctx.obj.get('embedding_model_path')
        
        # Pass embedding model path if available
        tool = create_reindex_tool(
            ctx.obj['db_path'], 
            embedding_model_path=embedding_model_path
        )
        
        with Progress() as progress:
            task = progress.add_task("Validating integrity...", total=None)
            report = tool.validate_integrity(collection)
            progress.update(task, completed=100, total=100)
        
        rprint(f"[bold blue]Integrity Validation[/bold blue]")
        
        if report['overall_status'] == 'PASS':
            rprint(f"[green]✓ All checks passed ({report['checks_passed']})[/green]")
        else:
            rprint(f"[red]❌ {report['checks_failed']} checks failed, {report['checks_passed']} passed[/red]")
            
            if report['issues']:
                rprint("\n[bold red]Issues Found:[/bold red]")
                for issue in report['issues']:
                    rprint(f"  • {issue}")
        
        if report['statistics']:
            rprint("\n[bold]Statistics:[/bold]")
            for stat, value in report['statistics'].items():
                rprint(f"  {stat}: {value}")
        
    except Exception as e:
        rprint(f"[red]❌ Validation failed: {e}[/red]")
        sys.exit(1)


# ========== RAG COMMANDS ==========

@cli.command('chat')
@click.option('--collection', default='default', help='Collection to query')
@click.option('--model-path', default=DEFAULT_LLM_PATH, help='LLM model path')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.option('--no-streaming', is_flag=True, help='Disable streaming output')
@click.option('--profile', default='balanced', help='Configuration profile to use')
@click.pass_context
def chat(ctx, collection: str, model_path: str, embedding_path: str, no_streaming: bool, profile: str):
    """Start interactive chat with RAG system"""
    try:
        # Load configuration manager to get profile settings
        from src.config_manager import create_config_manager
        from src.cli_chat import ChatInterface
        
        config_manager = create_config_manager()
        config_manager.switch_profile(profile)
        
        rprint(f"[blue]🚀 Starting RAG Chat Interface[/blue]")
        rprint(f"[dim]Profile: {profile} | Collection: {collection} | Streaming: {not no_streaming}[/dim]")
        rprint()
        
        # Create and run chat interface directly
        interface = ChatInterface(
            config_manager=config_manager,
            db_path=ctx.obj['db_path'],
            model_path=model_path,
            embedding_path=embedding_path,
            collection=collection,
            no_streaming=no_streaming
        )
        interface.run()
        
    except KeyboardInterrupt:
        rprint("\n[yellow]👋 Chat session ended[/yellow]")
    except Exception as e:
        rprint(f"[red]✗ Chat failed to start: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


def validate_query_input(question: str) -> str:
    """
    Validate and clean query input.
    
    Args:
        question: Raw query string
        
    Returns:
        Cleaned query string
        
    Raises:
        click.BadParameter: If query is invalid
    """
    if not question:
        raise click.BadParameter("Query cannot be empty")
    
    # Clean whitespace
    cleaned = question.strip()
    
    if not cleaned:
        raise click.BadParameter("Query cannot be only whitespace")
    
    # Check minimum length (optional)
    if len(cleaned) < 3:
        raise click.BadParameter("Query must be at least 3 characters long")
    
    # Check maximum length (prevent excessive queries)
    if len(cleaned) > 1000:
        raise click.BadParameter("Query cannot exceed 1000 characters")
    
    # Check for potentially problematic characters
    if any(char in cleaned for char in ['\x00', '\x01', '\x02']):
        raise click.BadParameter("Query contains invalid characters")
    
    return cleaned

@cli.command('query')
@click.argument('question')
@click.option('--collection', default='default', help='Collection to query')
@click.option('--model-path', default=DEFAULT_LLM_PATH, help='LLM model path')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.option('--k', default=5, help='Number of documents to retrieve')
@click.option('--metrics/--no-metrics', default=False, help='Enable JSONL metrics logging')
@click.pass_context
def query(ctx, question: str, collection: str, model_path: str, embedding_path: str, k: int, metrics: bool):
    """Ask a single question to the RAG system"""
    try:
        # ✅ FIX: Validate input before processing
        validated_question = validate_query_input(question)
        
        # Optionally enable JSONL metrics
        if metrics:
            try:
                from src.metrics import enable_metrics
                enable_metrics(True, output_path="logs/metrics.jsonl")
            except Exception:
                pass

        # Initialize RAG pipeline
        rag = RAGPipeline(
            db_path=ctx.obj['db_path'],
            embedding_model_path=embedding_path,
            llm_model_path=model_path
        )
        
        with Progress() as progress:
            task = progress.add_task("Processing query...", total=None)
            
            response = rag.query(validated_question, k=k, collection_id=collection)
            
            progress.update(task, completed=100, total=100)
        
        rprint(f"[bold blue]Question:[/bold blue] {validated_question}")
        rprint(f"[bold green]Answer:[/bold green]")
        # Support both {'answer': ...} from RAGPipeline and older {'response': ...}
        rprint(response.get('answer', response.get('response', '')))
        
        if ctx.obj['verbose'] and 'retrieval_results' in response:
            rprint(f"\n[dim]Retrieved {len(response['retrieval_results'])} relevant chunks[/dim]")
        
    except click.BadParameter as e:
        rprint(f"[red]❌ Invalid query: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]❌ Query failed: {e}[/red]")
        sys.exit(1)


# ========== UTILITY COMMANDS ==========

# ========== UTILITY COMMANDS ==========

@cli.command('doctor')
@click.option('--format', 'output_format', type=click.Choice(['markdown', 'json']), default='markdown', help='Report format')
@click.option('--output', type=click.Path(path_type=Path), help='Output file path')
@click.pass_context
def doctor(ctx, output_format: str, output: Optional[Path]):
    """Run comprehensive system diagnostics"""
    try:
        from src.health_checks import HealthChecker
        
        # Get config manager from context
        config_manager = ctx.obj.get('config_manager')
        if not config_manager:
            # If config manager not available, create one for diagnostics
            from src.config_manager import ConfigManager
            config_manager = ConfigManager()
            
        # Initialize health checker with config manager
        health_checker = HealthChecker(config_manager)
        
        rprint("[yellow]🔍 Running comprehensive system diagnostics...[/yellow]")
        
        with Progress() as progress:
            task = progress.add_task("Running health checks...", total=None)
            
            # Run all health checks
            health_report = health_checker.run_all_checks()
            
            progress.update(task, completed=100, total=100)
        
        # Display results
        overall_status = "✅ HEALTHY" if health_report.overall_status else "❌ UNHEALTHY"
        rprint(f"\n[bold blue]System Health Report {overall_status}[/bold blue]")
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Check", style="cyan")
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Message", style="yellow")
        summary_table.add_column("Time (ms)", justify="right", style="dim")
        
        for check in health_report.checks:
            status_icon = "✅" if check.status else "❌"
            summary_table.add_row(
                check.name.replace('_', ' ').title(),
                status_icon,
                check.message,
                f"{check.execution_time_ms:.1f}"
            )
        
        console.print(summary_table)
        
        # Show recommendations if any
        if health_report.recommendations:
            rprint(f"\n[bold yellow]💡 Recommendations:[/bold yellow]")
            for i, rec in enumerate(health_report.recommendations, 1):
                rprint(f"  {i}. {rec}")
        
        # Show failed check details
        failed_checks = [check for check in health_report.checks if not check.status]
        if failed_checks:
            rprint(f"\n[bold red]❌ Failed Check Details:[/bold red]")
            for check in failed_checks:
                rprint(f"\n[red]• {check.name.replace('_', ' ').title()}[/red]")
                rprint(f"  Error: {check.message}")
                if check.details and ctx.obj.get('verbose'):
                    rprint(f"  Details: {json.dumps(check.details, indent=2)}")
        
        # Save report to file if requested
        if output:
            report_content = health_checker.generate_report(health_report, format=output_format)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                f.write(report_content)
            rprint(f"\n[green]📄 Report saved to: {output}[/green]")
        
        # Exit with error code if unhealthy
        if not health_report.overall_status:
            sys.exit(1)
            
    except Exception as e:
        rprint(f"[red]❌ Diagnostics failed: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)

@cli.command('status')
@click.pass_context
def status(ctx):
    """Show system status and configuration"""
    try:
        db_path = Path(ctx.obj['db_path'])
        
        rprint("[bold blue]RAG System Status[/bold blue]")
        rprint(f"📍 Database: {db_path}")
        rprint(f"💾 Database size: {db_path.stat().st_size / (1024*1024):.2f}MB" if db_path.exists() else "❌ Database not found")
        
        # Collection summary
        if db_path.exists():
            organizer = create_corpus_organizer(str(db_path))
            summary = organizer.get_collection_summary()
            
            rprint(f"📚 Collections: {summary['total_collections']}")
            rprint(f"📁 Total documents: {summary['total_documents']}")
            rprint(f"📄 Total chunks: {summary['total_chunks']}")
            rprint(f"💾 Total size: {summary['total_size_mb']:.2f}MB")
            rprint(f"🎯 Current collection: {summary['current_collection']}")
        
        # Model paths
        rprint(f"\n[bold]Model Configuration:[/bold]")
        rprint(f"🧠 LLM: {DEFAULT_LLM_PATH}")
        rprint(f"🔗 Embeddings: {DEFAULT_EMBEDDING_PATH}")
        
    except Exception as e:
        rprint(f"[red]❌ Status check failed: {e}[/red]")
        sys.exit(1)


# ============================================================================
# EXPERIMENT COMMANDS - ParametricRAG Experimental Interface
# ============================================================================

@cli.group()
def experiment():
    """Advanced experimental interface for RAG system parameter exploration"""
    pass


@experiment.command('batch')
@click.option('--queries', 'queries_path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to queries JSON/JSONL')
@click.option('--profile', default=None, help='Config profile to use (fast/balanced/quality)')
@click.option('--collection', default='default', help='Collection to query')
@click.option('--model-path', default=DEFAULT_LLM_PATH, help='LLM model path')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.option('--k', default=5, help='Number of documents to retrieve')
@click.option('--output', 'output_path', type=click.Path(path_type=Path), help='Output JSONL file path (default: results/experiment_batch_*.jsonl)')
@click.option('--dry-run', is_flag=True, help='Do not load models; emit empty answers for structure/demo')
@click.pass_context
def experiment_batch(ctx, queries_path: Path, profile: Optional[str], collection: str, model_path: str, embedding_path: str, k: int, output_path: Optional[Path], dry_run: bool):
    """Run a batch of queries and emit JSONL results"""

    def _parse_value(val: str):
        try:
            if '.' in val:
                return float(val)
            return int(val)
        except Exception:
            return val

    def _load_evaluation_queries(path: Path) -> List[str]:
        text = path.read_text(encoding='utf-8', errors='ignore')
        # Try JSONL first
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        objs = []
        for ln in lines:
            try:
                objs.append(json.loads(ln))
            except Exception:
                objs = []
                break
        if objs:
            queries = [str(o['query']) for o in objs if isinstance(o, dict) and 'query' in o]
            if queries:
                return queries
        # Fallback: JSON
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            out = [str(x.get('query')) for x in data if 'query' in x]
            if out:
                return out
        if isinstance(data, dict):
            if 'queries' in data and isinstance(data['queries'], list):
                return [str(q) for q in data['queries']]
            if 'categories' in data and isinstance(data['categories'], dict):
                acc: List[str] = []
                for arr in data['categories'].values():
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, dict) and 'query' in item:
                                acc.append(str(item['query']))
                if acc:
                    return acc
        raise click.BadParameter('Unsupported queries file format')

    try:
        # Prepare output path
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not output_path:
            Path('results').mkdir(parents=True, exist_ok=True)
            output_path = Path('results') / f"experiment_batch_{ts}.jsonl"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load queries
        queries = _load_evaluation_queries(queries_path)
        if not queries:
            rprint("[yellow]No queries found in the provided file[/yellow]")
            return

        # Switch profile if provided
        config_manager: ConfigManager = ctx.obj['config_manager']
        if profile:
            config_manager.switch_profile(profile)

        rag = None
        if not dry_run:
            # Initialize pipeline only when not dry-running
            rag = RAGPipeline(
                db_path=ctx.obj['db_path'],
                embedding_model_path=embedding_path,
                llm_model_path=model_path,
                profile_config=config_manager.get_profile(),
            )
            rag.set_corpus(collection)

        # Run queries and write JSONL
        with output_path.open('w', encoding='utf-8') as f_out:
            for q in queries:
                try:
                    cleaned = validate_query_input(q)
                except click.BadParameter:
                    # Skip invalid entries
                    continue
                t0 = datetime.now().isoformat()
                if dry_run:
                    result = {
                        'answer': '',
                        'sources': [],
                        'metadata': {
                            'query': cleaned,
                            'retrieval_method': 'vector',
                            'contexts_count': 0,
                            'retrieval_time': 0.0,
                            'generation_time': 0.0,
                            'tokens_per_second': 0.0
                        }
                    }
                else:
                    result = rag.query(cleaned, k=k, collection_id=collection)
                record = {
                    'timestamp': t0,
                    'query': cleaned,
                    'profile': config_manager.get_current_profile_name(),
                    'collection': collection,
                    'k': k,
                    'answer': result.get('answer', ''),
                    'sources': result.get('sources', []),
                    'metadata': result.get('metadata', {}),
                }
                f_out.write(json.dumps(record) + "\n")

        rprint(f"[green]✓ Batch complete[/green] → [cyan]{output_path}[/cyan]")
        rprint(f"Queries processed: {len(queries)} | Profile: {profile or config_manager.get_current_profile_name()} | Collection: {collection}")

    except Exception as e:
        rprint(f"[red]✗ Batch execution failed: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)

@experiment.command('sweep')
@click.option('--param', required=True, help='Parameter to sweep (e.g., temperature, chunk_size)')
@click.option('--range', 'param_range', help='Range as min,max,step (e.g., 0.1,1.0,0.1)')
@click.option('--values', help='Categorical values (e.g., 256,512,1024)')
@click.option('--queries', default='test_data/benchmark_queries.json', help='JSON file with evaluation queries')
@click.option('--output', help='Output file for results')
@click.option('--corpus', default='default', help='Target corpus/collection')
@click.pass_context
def sweep(ctx, param: str, param_range: str, values: str, queries: str, output: str, corpus: str):
    """Run parameter sweep experiment"""
    try:
        from src.experiment_runner import create_experiment_runner
        from src.experiment_templates import create_base_experiment_config
        from src.config_manager import ParameterRange
        
        # Load evaluation queries
        query_list = _load_evaluation_queries(queries)
        
        # Create parameter range
        if values:
            # Categorical values
            value_list = [_parse_value(v.strip()) for v in values.split(',')]
            param_range_obj = ParameterRange(param, "categorical", values=value_list)
        elif param_range:
            # Linear range
            parts = param_range.split(',')
            if len(parts) != 3:
                raise click.BadParameter("Range must be min,max,step format")
            min_val, max_val, step = map(float, parts)
            param_range_obj = ParameterRange(param, "linear", min_val, max_val, step)
        else:
            raise click.BadParameter("Must specify either --range or --values")
        
        rprint(f"[blue]🧪 Starting parameter sweep: {param}[/blue]")
        rprint(f"[dim]Range: {param_range_obj.param_name} = {param_range_obj.generate_values()}[/dim]")
        rprint(f"[dim]Queries: {len(query_list)} | Corpus: {corpus}[/dim]")
        
        # Create base config and override corpus
        base_config = create_base_experiment_config()
        base_config.target_corpus = corpus
        
        # Run experiment
        runner = create_experiment_runner()
        results = runner.run_parameter_sweep(
            base_config=base_config,
            parameter_ranges=[param_range_obj],
            queries=query_list
        )
        
        # Output results
        _display_sweep_results(results)
        
        if output:
            _save_experiment_results(results, output)
            rprint(f"[green]💾 Results saved to {output}[/green]")
        
    except Exception as e:
        rprint(f"[red]❌ Parameter sweep failed: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@experiment.command('compare')
@click.option('--config-a', required=True, help='First configuration (profile name or JSON file)')
@click.option('--config-b', required=True, help='Second configuration (profile name or JSON file)')  
@click.option('--queries', default='test_data/benchmark_queries.json', help='JSON file with evaluation queries')
@click.option('--significance', default=0.05, help='Statistical significance level')
@click.option('--output', help='Output file for results')
@click.pass_context
def compare(ctx, config_a: str, config_b: str, queries: str, significance: float, output: str):
    """Run A/B test between two configurations"""
    try:
        from src.experiment_runner import create_experiment_runner
        
        # Load configurations
        config_a_obj = _load_config(config_a)
        config_b_obj = _load_config(config_b)
        
        # Load evaluation queries
        query_list = _load_evaluation_queries(queries)
        
        rprint(f"[blue]⚖️  Starting A/B test comparison[/blue]")
        rprint(f"[dim]Config A: {config_a} | Config B: {config_b}[/dim]")
        rprint(f"[dim]Queries: {len(query_list)} | Significance: {significance}[/dim]")
        
        # Run experiment
        runner = create_experiment_runner()
        results = runner.run_ab_test(
            config_a=config_a_obj,
            config_b=config_b_obj,
            queries=query_list,
            significance_level=significance
        )
        
        # Display results
        _display_ab_test_results(results)
        
        if output:
            _save_experiment_results(results, output)
            rprint(f"[green]💾 Results saved to {output}[/green]")
        
    except Exception as e:
        rprint(f"[red]❌ A/B test failed: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@experiment.command('template')
@click.argument('template_name', required=False)
@click.option('--queries', help='Override template queries with custom JSON file')
@click.option('--corpus', default='default', help='Target corpus/collection')
@click.option('--output', help='Output file for results')
@click.option('--list-templates', is_flag=True, help='List available templates')
@click.pass_context
def template(ctx, template_name: str, queries: str, corpus: str, output: str, list_templates: bool):
    """Run pre-defined experiment template"""
    try:
        from src.experiment_runner import create_experiment_runner
        from src.experiment_templates import get_template, list_templates as get_template_list, get_template_info
        
        if list_templates:
            templates = get_template_list()
            rprint("[blue]📋 Available Experiment Templates:[/blue]")
            
            for template_name in templates:
                info = get_template_info(template_name)
                rprint(f"\n[bold]{template_name}[/bold]")
                rprint(f"  {info['description']}")
                rprint(f"  [dim]Parameters: {', '.join(info['parameters'])} ({info['estimated_combinations']} combinations)[/dim]")
                rprint(f"  [dim]Runtime: ~{info['expected_runtime_hours']:.1f}h | Queries: {info['evaluation_queries_count']}[/dim]")
            return
        
        if not template_name:
            rprint("[red]❌ Template name is required when not listing templates[/red]")
            rprint("[dim]Use --list-templates to see available templates[/dim]")
            sys.exit(1)
        
        # Load template
        template = get_template(template_name)
        
        # Override queries if provided
        query_list = None
        if queries:
            query_list = _load_evaluation_queries(queries)
        
        # Override corpus in base config
        template.base_config.target_corpus = corpus
        
        rprint(f"[blue]🔬 Running experiment template: {template.name}[/blue]")
        rprint(f"[dim]{template.description}[/dim]")
        rprint(f"[dim]Parameters: {len(template.parameter_ranges)} | Corpus: {corpus}[/dim]")
        
        # Estimate runtime
        combinations = 1
        for pr in template.parameter_ranges:
            combinations *= len(pr.generate_values())
        estimated_queries = len(query_list) if query_list else len(template.evaluation_queries)
        estimated_runtime = combinations * estimated_queries * 3.0 / 60  # minutes
        
        rprint(f"[dim]Estimated: {combinations} configs × {estimated_queries} queries ≈ {estimated_runtime:.1f}min[/dim]")
        
        if combinations * estimated_queries > 100:
            if not click.confirm(f"This will run {combinations * estimated_queries} experiments. Continue?"):
                rprint("[yellow]⚠️  Experiment cancelled[/yellow]")
                return
        
        # Run experiment
        runner = create_experiment_runner()
        results = runner.run_template_experiment(template, query_list)
        
        # Display results
        _display_template_results(results, template)
        
        if output:
            _save_experiment_results(results, output)
            rprint(f"[green]💾 Results saved to {output}[/green]")
        
    except Exception as e:
        rprint(f"[red]❌ Template experiment failed: {e}[/red]")
        if ctx.obj.get('verbose'):
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@experiment.command('list')
@click.option('--status', help='Filter by status (running, completed, failed)')
@click.option('--limit', default=10, help='Maximum number of experiments to show')
def list_experiments(status: str, limit: int):
    """List recent experiments"""
    try:
        from src.experiment_runner import ExperimentDatabase
        
        db = ExperimentDatabase()
        # This would require implementing a list method in ExperimentDatabase
        rprint("[yellow]⚠️  Experiment listing not yet implemented[/yellow]")
        rprint("[dim]Coming soon: experiment history and status tracking[/dim]")
        
    except Exception as e:
        rprint(f"[red]❌ Failed to list experiments: {e}[/red]")


# Helper functions for experiment CLI

def _load_evaluation_queries(queries_file: str) -> List[str]:
    """Load evaluation queries from JSON file."""
    queries_path = Path(queries_file)
    
    if not queries_path.exists():
        # Use default queries if file doesn't exist
        rprint(f"[yellow]⚠️  Queries file {queries_file} not found, using defaults[/yellow]")
        return [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Explain deep learning algorithms.",
            "What are neural networks?",
            "How do large language models work?"
        ]
    
    try:
        with open(queries_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'queries' in data:
                return data['queries']
            else:
                raise ValueError("Invalid queries file format")
    except Exception as e:
        rprint(f"[red]❌ Failed to load queries from {queries_file}: {e}[/red]")
        sys.exit(1)


def _load_config(config_identifier: str):
    """Load configuration from profile name or JSON file."""
    from src.config_manager import create_config_manager
    from src.experiment_templates import create_base_experiment_config
    
    # Check if it's a profile name
    if config_identifier in ['fast', 'balanced', 'quality']:
        config_manager = create_config_manager()
        profile_config = config_manager.get_profile(config_identifier)
        
        # Convert ProfileConfig to ExperimentConfig
        base_config = create_base_experiment_config()
        base_config.retrieval_k = profile_config.retrieval_k
        base_config.max_tokens = profile_config.max_tokens
        base_config.temperature = profile_config.temperature
        base_config.chunk_size = profile_config.chunk_size
        base_config.chunk_overlap = profile_config.chunk_overlap
        base_config.n_ctx = profile_config.n_ctx
        
        return base_config
    
    # Try to load as JSON file
    config_path = Path(config_identifier)
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
            from src.config_manager import ExperimentConfig
            return ExperimentConfig(**config_data)
    
    raise ValueError(f"Unknown configuration: {config_identifier}")


def _parse_value(value_str: str):
    """Parse string value to appropriate type."""
    value_str = value_str.strip()
    
    # Try integer
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Try boolean
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Return as string
    return value_str


def _display_sweep_results(results):
    """Display parameter sweep results."""
    rprint(f"\n[green]✅ Parameter sweep completed![/green]")
    rprint(f"[dim]Experiment ID: {results.experiment_id}[/dim]")
    rprint(f"[dim]Total runtime: {results.total_runtime:.1f}s[/dim]")
    rprint(f"[dim]Total runs: {len(results.results)}[/dim]")
    
    # Calculate aggregate metrics
    if results.results:
        avg_response_time = sum(r.metrics.get('response_time', 0) for r in results.results) / len(results.results)
        success_rate = sum(1 for r in results.results if r.error_message is None) / len(results.results)
        
        rprint(f"\n[bold]Summary Metrics:[/bold]")
        rprint(f"📊 Average response time: {avg_response_time:.2f}s")
        rprint(f"✅ Success rate: {success_rate:.1%}")


def _display_ab_test_results(results):
    """Display A/B test results."""
    rprint(f"\n[green]✅ A/B test completed![/green]")
    rprint(f"[dim]Experiment ID: {results.experiment_id}[/dim]")
    rprint(f"[dim]Total runtime: {results.total_runtime:.1f}s[/dim]")
    
    # Split results by configuration
    a_results = [r for r in results.results if r.config.target_corpus == "config_A"]
    b_results = [r for r in results.results if r.config.target_corpus == "config_B"]
    
    if a_results and b_results:
        avg_time_a = sum(r.metrics.get('response_time', 0) for r in a_results) / len(a_results)
        avg_time_b = sum(r.metrics.get('response_time', 0) for r in b_results) / len(b_results)
        
        rprint(f"\n[bold]Comparison Results:[/bold]")
        rprint(f"⏱️  Config A avg time: {avg_time_a:.2f}s")
        rprint(f"⏱️  Config B avg time: {avg_time_b:.2f}s")
        rprint(f"📈 Improvement: {((avg_time_a - avg_time_b) / avg_time_a * 100):+.1f}%")


def _display_template_results(results, template):
    """Display template experiment results."""
    rprint(f"\n[green]✅ Template experiment '{template.name}' completed![/green]")
    rprint(f"[dim]Experiment ID: {results.experiment_id}[/dim]")
    rprint(f"[dim]Total runtime: {results.total_runtime:.1f}s[/dim]")
    rprint(f"[dim]Configurations tested: {len(set(r.run_id.split('_')[2] for r in results.results))}[/dim]")


def _save_experiment_results(results, output_file: str):
    """Save experiment results to JSON file with full configuration provenance."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'claude_version': 'Experiment_v2_fixes'
    }
    
    # Convert results to serializable format with enhanced provenance
    results_data = {
        "metadata": metadata,
        "experiment_id": results.experiment_id,
        "experiment_type": results.experiment_type,
        "status": results.status,
        "total_runtime": results.total_runtime,
        "created_at": results.created_at.isoformat(),
        "completed_at": results.completed_at.isoformat() if results.completed_at else None,
        "total_results": len(results.results),
        "results": []
    }
    
    for result in results.results:
        # Extract all configuration parameters if available
        config_dict = {}
        if hasattr(result, 'config') and result.config:
            config_dict = {
                'chunk_size': getattr(result.config, 'chunk_size', None),
                'chunk_overlap': getattr(result.config, 'chunk_overlap', None),
                'retrieval_k': getattr(result.config, 'retrieval_k', None),
                'temperature': getattr(result.config, 'temperature', None),
                'max_tokens': getattr(result.config, 'max_tokens', None),
                'profile': getattr(result.config, 'profile', None),
                'collection_id': getattr(result.config, 'collection_id', None),
                'retrieval_method': getattr(result.config, 'retrieval_method', 'vector')
            }
        
        # Create enhanced result entry
        result_entry = {
            "run_id": result.run_id,
            "config": config_dict,
            "query": result.query,
            "response": result.response,
            "response_length_words": len(result.response.split()) if result.response else 0,
            "response_length_chars": len(result.response) if result.response else 0,
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
            "timestamp": result.timestamp.isoformat(),
            "error_message": result.error_message,
            
            # Enhanced metrics from metadata if available
            "retrieval_time": result.metrics.get('retrieval_time') if result.metrics else None,
            "generation_time": result.metrics.get('generation_time') if result.metrics else None,
            "prompt_tokens": result.metrics.get('prompt_tokens') if result.metrics else None,
            "output_tokens": result.metrics.get('output_tokens') if result.metrics else None,
            "total_tokens": result.metrics.get('total_tokens') if result.metrics else None,
            "contexts_count": result.metrics.get('contexts_count') if result.metrics else None,
            "tokens_per_second": result.metrics.get('tokens_per_second') if result.metrics else None
        }
        
        results_data["results"].append(result_entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(results.results)} experiment results to {output_path}")
    print(f"   Enhanced format includes full config provenance")


if __name__ == '__main__':
    cli()

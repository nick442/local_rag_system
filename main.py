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
from pathlib import Path
from typing import Optional

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
# from src.cli_chat import main as chat_main  # Commented out - has dependency issues


# Global configuration
console = Console()
DEFAULT_DB_PATH = "data/rag_vectors.db"
DEFAULT_EMBEDDING_PATH = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_LLM_PATH = "/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/models/gemma-3-4b-it-q4_0.gguf"


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/rag_system.log')
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)


@click.group()
@click.option('--db-path', default=DEFAULT_DB_PATH, help='Path to vector database')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, db_path: str, verbose: bool):
    """RAG System - Local Retrieval-Augmented Generation Platform"""
    setup_logging(verbose)
    
    # Store configuration in context
    ctx.ensure_object(dict)
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
@click.option('--dry-run', is_flag=True, help='Preview without processing')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoint')
@click.option('--deduplicate/--no-deduplicate', default=True, help='Skip duplicates')
@click.pass_context
def ingest_directory(ctx, path: Path, pattern: str, collection: str, max_workers: int, 
                    batch_size: int, dry_run: bool, resume: bool, deduplicate: bool):
    """Ingest documents from directory with parallel processing"""
    
    async def run_ingestion():
        try:
            manager = create_corpus_manager(
                db_path=ctx.obj['db_path'],
                embedding_model_path=DEFAULT_EMBEDDING_PATH,
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
@click.pass_context
def ingest_file(ctx, file_path: Path, collection: str):
    """Ingest a single document file"""
    
    async def run_single_ingestion():
        try:
            manager = create_corpus_manager(
                db_path=ctx.obj['db_path'],
                embedding_model_path=DEFAULT_EMBEDDING_PATH
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
        tool = create_reindex_tool(ctx.obj['db_path'])
        
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
@click.pass_context
def chat(ctx, collection: str, model_path: str, embedding_path: str):
    """Start interactive chat with RAG system"""
    try:
        # Import and run the existing chat interface
        # Modify sys.argv to pass parameters to the chat system
        original_argv = sys.argv.copy()
        sys.argv = [
            'cli_chat.py',
            '--db-path', ctx.obj['db_path'],
            '--collection', collection,
            '--model-path', model_path,
            '--embedding-path', embedding_path
        ]
        
        # Call the chat main function
        # chat_main()  # Temporarily disabled
        rprint("[yellow]Chat functionality temporarily disabled - use 'query' command instead[/yellow]")
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        rprint(f"[red]❌ Chat failed to start: {e}[/red]")
        sys.exit(1)


@cli.command('query')
@click.argument('question')
@click.option('--collection', default='default', help='Collection to query')
@click.option('--model-path', default=DEFAULT_LLM_PATH, help='LLM model path')
@click.option('--embedding-path', default=DEFAULT_EMBEDDING_PATH, help='Embedding model path')
@click.option('--k', default=5, help='Number of documents to retrieve')
@click.pass_context
def query(ctx, question: str, collection: str, model_path: str, embedding_path: str, k: int):
    """Ask a single question to the RAG system"""
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(
            db_path=ctx.obj['db_path'],
            embedding_model_path=embedding_path,
            llm_model_path=model_path
        )
        
        with Progress() as progress:
            task = progress.add_task("Processing query...", total=None)
            
            response = rag.query(question, k=k, collection_id=collection)
            
            progress.update(task, completed=100, total=100)
        
        rprint(f"[bold blue]Question:[/bold blue] {question}")
        rprint(f"[bold green]Answer:[/bold green]")
        rprint(response['response'])
        
        if ctx.obj['verbose'] and 'retrieval_results' in response:
            rprint(f"\n[dim]Retrieved {len(response['retrieval_results'])} relevant chunks[/dim]")
        
    except Exception as e:
        rprint(f"[red]❌ Query failed: {e}[/red]")
        sys.exit(1)


# ========== UTILITY COMMANDS ==========

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


if __name__ == '__main__':
    cli()
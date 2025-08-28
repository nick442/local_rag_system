"""
Interactive CLI Chat Interface for RAG System
Provides user-friendly access to the RAG system with streaming output and commands.
"""

import click
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .rag_pipeline import RAGPipeline
from .config_manager import ConfigManager
from .monitor import Monitor


class ChatSession:
    """Manages chat session state and history."""
    
    def __init__(self, rag_pipeline: RAGPipeline, console: Console):
        self.rag = rag_pipeline
        self.console = console
        self.history: List[Dict] = []
        self.max_history = 10
        self.session_stats = {
            'queries': 0,
            'total_tokens': 0,
            'start_time': None
        }
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history."""
        self.history.append({
            'role': role,
            'content': content,
            'metadata': metadata or {}
        })
        
        # Keep only last max_history turns
        if len(self.history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.history = self.history[-self.max_history * 2:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.console.print("[yellow]Conversation history cleared.[/yellow]")
    
    def get_history_summary(self) -> str:
        """Get formatted history summary."""
        if not self.history:
            return "No conversation history."
        
        summary = f"Last {len(self.history)//2} exchanges:\n"
        for i in range(0, len(self.history), 2):
            if i+1 < len(self.history):
                user_msg = self.history[i]['content'][:50] + "..." if len(self.history[i]['content']) > 50 else self.history[i]['content']
                assistant_msg = self.history[i+1]['content'][:50] + "..." if len(self.history[i+1]['content']) > 50 else self.history[i+1]['content']
                summary += f"  Q: {user_msg}\n  A: {assistant_msg}\n"
        
        return summary


class ChatInterface:
    """Main chat interface with command handling."""
    
    def __init__(self, config_manager: ConfigManager, model_path: Optional[str] = None, no_streaming: bool = False):
        self.console = Console()
        self.config = config_manager
        self.no_streaming = no_streaming
        self.monitor = Monitor()
        
        # Initialize RAG pipeline
        config = self.config.get_current_config()
        db_path = config.get('database', {}).get('path', 'data/rag_vectors.db')
        embedding_path = config.get('models', {}).get('embedding_path', 'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
        llm_path = model_path or config.get('models', {}).get('llm_path', '/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/models/gemma-3-4b-it-q4_0.gguf')
        
        self.rag = RAGPipeline(db_path, embedding_path, llm_path)
        self.session = ChatSession(self.rag, self.console)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.console.print("\n[yellow]Interrupted by user. Use /exit to quit properly.[/yellow]")
    
    def show_welcome(self):
        """Display welcome message and commands."""
        welcome_panel = Panel.fit(
            Text("RAG Chat Interface", style="bold green", justify="center"),
            title="Local RAG System",
            border_style="green"
        )
        self.console.print(welcome_panel)
        self.console.print("[dim]Commands: /help, /reset, /stats, /exit, /corpus, /config[/dim]")
        self.console.print()
    
    def show_help(self):
        """Display help information."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        commands = [
            ("/help", "Show this help message"),
            ("/reset", "Clear conversation history"),
            ("/stats", "Show session statistics"),
            ("/exit", "Exit the chat"),
            ("/corpus", "Show corpus information"),
            ("/config", "Show current configuration"),
            ("/profile <name>", "Switch configuration profile"),
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)
    
    def show_stats(self):
        """Display session statistics."""
        stats = self.monitor.get_session_stats()
        
        stats_table = Table(title="Session Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for key, value in stats.items():
            if isinstance(value, float):
                if 'latency' in key.lower() or 'time' in key.lower():
                    formatted_value = f"{value:.1f}ms"
                elif 'speed' in key.lower() or 'tokens' in key.lower():
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            stats_table.add_row(key.replace('_', ' ').title(), formatted_value)
        
        self.console.print(stats_table)
    
    def show_corpus_info(self):
        """Display corpus information."""
        # Get database stats from RAG pipeline
        try:
            # This would need to be implemented in the RAG pipeline
            corpus_info = {
                "Documents": "N/A",
                "Chunks": "N/A", 
                "Database Size": "N/A"
            }
            
            corpus_table = Table(title="Corpus Information")
            corpus_table.add_column("Metric", style="cyan")
            corpus_table.add_column("Value", style="white")
            
            for key, value in corpus_info.items():
                corpus_table.add_row(key, str(value))
            
            self.console.print(corpus_table)
        except Exception as e:
            self.console.print(f"[red]Error getting corpus info: {e}[/red]")
    
    def show_config(self):
        """Display current configuration."""
        config = self.config.get_current_config()
        profile = self.config.get_current_profile()
        
        config_panel = Panel(
            f"Current Profile: {profile}\n" +
            f"Retrieval K: {config.get('retrieval_k', 'N/A')}\n" +
            f"Max Tokens: {config.get('max_tokens', 'N/A')}\n" +
            f"Temperature: {config.get('temperature', 'N/A')}",
            title="Configuration",
            border_style="blue"
        )
        self.console.print(config_panel)
    
    def handle_command(self, user_input: str) -> bool:
        """Handle chat commands. Returns True if command was processed."""
        if not user_input.startswith('/'):
            return False
        
        parts = user_input[1:].split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command == 'help':
            self.show_help()
        elif command == 'reset':
            self.session.clear_history()
        elif command == 'stats':
            self.show_stats()
        elif command == 'exit' or command == 'quit':
            self.console.print("[green]Goodbye![/green]")
            return 'exit'
        elif command == 'corpus':
            self.show_corpus_info()
        elif command == 'config':
            self.show_config()
        elif command == 'profile':
            if args:
                try:
                    self.config.switch_profile(args[0])
                    self.console.print(f"[green]Switched to profile: {args[0]}[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error switching profile: {e}[/red]")
            else:
                self.console.print("[red]Usage: /profile <profile_name>[/red]")
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")
        
        return True
    
    def display_streaming_response(self, query: str, retrieval_results: List):
        """Display streaming response with live updates."""
        # Show retrieval info
        if retrieval_results:
            self.console.print(f"[dim]Retrieving context... {len(retrieval_results)} documents found[/dim]")
        
        # Create live display for streaming
        response_text = Text()
        
        with Live(response_text, console=self.console, refresh_per_second=10) as live:
            try:
                full_response = ""
                token_count = 0
                
                for token in self.rag.query_stream(query, use_history=True):
                    full_response += token
                    token_count += 1
                    
                    # Update display
                    response_text = Text()
                    response_text.append("Assistant: ", style="bold green")
                    response_text.append(full_response)
                    live.update(response_text)
                
                # Add final stats
                final_text = Text()
                final_text.append("Assistant: ", style="bold green")
                final_text.append(full_response)
                final_text.append(f"\n[dim]({token_count} tokens)[/dim]", style="dim")
                live.update(final_text)
                
                return full_response
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Generation interrupted by user.[/yellow]")
                return full_response if 'full_response' in locals() else ""
    
    def display_non_streaming_response(self, query: str) -> str:
        """Display non-streaming response with progress indicator."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating response...", total=None)
            
            try:
                response = self.rag.query(query, use_history=True)
                progress.remove_task(task)
                
                # Display response
                self.console.print("Assistant: ", style="bold green", end="")
                self.console.print(response)
                
                return response
                
            except Exception as e:
                progress.remove_task(task)
                self.console.print(f"[red]Error generating response: {e}[/red]")
                return ""
    
    def run(self):
        """Main chat loop."""
        self.show_welcome()
        self.monitor.start_session()
        
        try:
            while True:
                # Get user input
                try:
                    user_input = self.console.input("[blue]You: [/blue]").strip()
                except (EOFError, KeyboardInterrupt):
                    self.console.print("\n[green]Goodbye![/green]")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                result = self.handle_command(user_input)
                if result == 'exit':
                    break
                elif result:  # Command was processed
                    continue
                
                # Process regular query
                self.session.add_to_history('user', user_input)
                self.monitor.record_query()
                
                try:
                    if self.no_streaming:
                        response = self.display_non_streaming_response(user_input)
                    else:
                        # For streaming, we need to get retrieval results first
                        retrieval_results = []  # This would come from the RAG pipeline
                        response = self.display_streaming_response(user_input, retrieval_results)
                    
                    if response:
                        self.session.add_to_history('assistant', response)
                        self.monitor.record_response(len(response.split()))
                
                except Exception as e:
                    self.console.print(f"[red]Error processing query: {e}[/red]")
                
                self.console.print()  # Add spacing
        
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {e}[/red]")
        
        finally:
            self.monitor.end_session()


@click.command()
@click.option('--model-path', default=None, help='Override model path')
@click.option('--no-streaming', is_flag=True, help='Disable streaming output')
@click.option('--config-path', default='config/app_config.yaml', help='Configuration file path')
@click.option('--profile', default=None, help='Configuration profile to use')
def chat(model_path, no_streaming, config_path, profile):
    """Start interactive chat session with the RAG system."""
    try:
        # Initialize configuration
        config_manager = ConfigManager(config_path)
        if profile:
            config_manager.switch_profile(profile)
        
        # Start chat interface
        interface = ChatInterface(config_manager, model_path, no_streaming)
        interface.run()
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Failed to start chat interface: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    chat()
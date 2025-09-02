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
    
    def __init__(self, config_manager: ConfigManager, db_path: Optional[str] = None, 
                 model_path: Optional[str] = None, embedding_path: Optional[str] = None, 
                 collection: str = 'default', no_streaming: bool = False):
        self.console = Console()
        self.config = config_manager
        self.collection = collection
        self.no_streaming = no_streaming
        self.monitor = Monitor()
        
        # Get current profile configuration
        profile_config = self.config.get_profile()
        
        # Use provided parameters or fall back to config defaults
        db_path = db_path or 'data/rag_vectors.db'
        embedding_path = embedding_path or 'models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
        llm_path = model_path or 'models/gemma-3-4b-it-q4_0.gguf'
        
        # Initialize RAG pipeline with profile configuration
        self.rag = RAGPipeline(
            db_path=db_path, 
            embedding_model_path=embedding_path, 
            llm_model_path=llm_path,
            profile_config=profile_config
        )
        self.session = ChatSession(self.rag, self.console)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)

    
    def _reinitialize_rag_pipeline(self):
        """Reinitialize RAG pipeline with current profile configuration."""
        profile_config = self.config.get_profile()
        
        # Store current configuration parameters
        current_db_path = self.rag.db_path
        current_embedding_path = self.rag.embedding_model_path
        current_llm_path = self.rag.llm_model_path
        
        # Close existing pipeline if it has cleanup methods
        if hasattr(self.rag, 'close'):
            self.rag.close()
        
        # Create new RAG pipeline with updated profile
        self.rag = RAGPipeline(
            db_path=current_db_path,
            embedding_model_path=current_embedding_path,
            llm_model_path=current_llm_path,
            profile_config=profile_config
        )
        
        # Update session with new RAG pipeline
        self.session.rag = self.rag
    
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
    
    def is_conversational_input(self, text: str) -> bool:
        """Detect if input is conversational rather than informational query."""
        text_lower = text.lower().strip()
        
        # Greetings
        greetings = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'}
        if text_lower in greetings:
            return True
            
        # Farewells
        farewells = {'bye', 'goodbye', 'see you', 'farewell', 'take care'}
        if text_lower in farewells:
            return True
            
        # Thanks/acknowledgments
        thanks = {'thanks', 'thank you', 'thx', 'ty', 'appreciated'}
        if text_lower in thanks or text_lower.startswith('thank'):
            return True
            
        # Social responses
        social = {'yes', 'no', 'ok', 'okay', 'sure', 'alright', 'got it', 'i see', 'cool', 'nice'}
        if text_lower in social:
            return True
            
        # Very short inputs that are likely conversational
        if len(text_lower) <= 3 and text_lower not in {'how', 'why', 'who', 'what'}:
            return True
            
        return False
    
    def generate_conversational_response(self, user_input: str) -> str:
        """Generate appropriate conversational response without RAG retrieval."""
        text_lower = user_input.lower().strip()
        
        # Greetings
        if text_lower in {'hi', 'hello', 'hey'}:
            return "Hello! I'm ready to help you with questions about your documents. What would you like to know?"
        elif 'morning' in text_lower:
            return "Good morning! How can I assist you with your document search today?"
        elif 'afternoon' in text_lower:
            return "Good afternoon! What can I help you find in your documents?"
        elif 'evening' in text_lower:
            return "Good evening! I'm here to help with any questions about your documents."
            
        # Farewells
        elif text_lower in {'bye', 'goodbye', 'see you', 'farewell'}:
            return "Goodbye! Feel free to come back anytime you need help with your documents."
        elif 'take care' in text_lower:
            return "Take care! I'll be here whenever you need document assistance."
            
        # Thanks
        elif text_lower in {'thanks', 'thank you', 'thx', 'ty'} or text_lower.startswith('thank'):
            return "You're welcome! Happy to help with your document queries anytime."
        elif 'appreciated' in text_lower:
            return "I'm glad I could help! Feel free to ask more questions about your documents."
            
        # Social responses
        elif text_lower in {'yes', 'ok', 'okay', 'sure', 'alright'}:
            return "Great! What would you like to know about your documents?"
        elif text_lower == 'no':
            return "No problem. Is there anything else I can help you find in your documents?"
        elif text_lower in {'got it', 'i see'}:
            return "Excellent! Let me know if you have other questions about your documents."
        elif text_lower in {'cool', 'nice'}:
            return "Glad you think so! What else can I help you discover in your document collection?"
            
        # Default conversational response
        else:
            return "I'm here to help you search and understand your documents. What would you like to know?"
    
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
                    old_profile = self.config.get_current_profile_name()
                    self.config.switch_profile(args[0])
                    
                    # Reinitialize RAG pipeline with new profile
                    self._reinitialize_rag_pipeline()
                    
                    self.console.print(f"[green]Switched from profile '{old_profile}' to '{args[0]}'[/green]")
                    self.console.print("[dim]RAG pipeline reinitialized with new settings[/dim]")
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
                
                # Use query_stream without use_history for now (streaming doesn't support conversation history yet)
                generator, metadata = self.rag.query_stream(query)
                
                for token in generator:
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
                # Use chat method for conversation history support
                response_dict = self.rag.chat(query, use_history=True)
                progress.remove_task(task)
                
                # Extract the answer from the response
                response = response_dict['answer']
                
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
        self.monitor.reset_session()  # Fixed: use reset_session instead of start_session
        
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
                
                # Check if input is conversational vs informational
                if self.is_conversational_input(user_input):
                    # Handle conversational input without RAG retrieval
                    response = self.generate_conversational_response(user_input)
                    self.session.add_to_history('user', user_input)
                    self.session.add_to_history('assistant', response)
                    
                    # Display response
                    self.console.print("Assistant: ", style="bold green", end="")
                    self.console.print(response)
                else:
                    # Process informational query with RAG
                    self.session.add_to_history('user', user_input)
                    query_metrics = self.monitor.start_query_tracking()  # Store returned metrics
                    
                    try:
                        if self.no_streaming:
                            response = self.display_non_streaming_response(user_input)
                            # Get metrics from RAG response for monitoring
                            self.monitor.end_query_tracking(
                                query_metrics,
                                tokens_generated=50,  # Placeholder - would need actual metrics
                                success=True if response else False
                            )
                        else:
                            # For streaming, we need to get retrieval results first
                            retrieval_results = []  # This would come from the RAG pipeline
                            response = self.display_streaming_response(user_input, retrieval_results)
                            self.monitor.end_query_tracking(
                                query_metrics,
                                tokens_generated=50,  # Placeholder - would need actual metrics
                                success=True if response else False
                            )
                        
                        if response:
                            self.session.add_to_history('assistant', response)
                    
                    except Exception as e:
                        self.console.print(f"[red]Error processing query: {e}[/red]")
                        # End tracking with error
                        self.monitor.end_query_tracking(
                            query_metrics,
                            success=False,
                            error_message=str(e)
                        )
                
                self.console.print()  # Add spacing
        
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {e}[/red]")
        
        finally:
            self.monitor.stop_monitoring()  # Fixed: use stop_monitoring instead of end_session


@click.command()
@click.option('--db-path', default=None, help='Vector database path')
@click.option('--model-path', default=None, help='Override model path')
@click.option('--embedding-path', default=None, help='Embedding model path')
@click.option('--collection', default='default', help='Collection to query')
@click.option('--no-streaming', is_flag=True, help='Disable streaming output')
@click.option('--config-path', default='config/rag_config.yaml', help='Configuration file path')
@click.option('--profile', default=None, help='Configuration profile to use')
def chat(db_path, model_path, embedding_path, collection, no_streaming, config_path, profile):
    """Start interactive chat session with the RAG system."""
    try:
        # Initialize configuration
        config_manager = ConfigManager(config_path)
        if profile:
            config_manager.switch_profile(profile)
        
        # Start chat interface with provided parameters
        interface = ChatInterface(
            config_manager=config_manager, 
            db_path=db_path,
            model_path=model_path,
            embedding_path=embedding_path,
            collection=collection,
            no_streaming=no_streaming
        )
        interface.run()
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Failed to start chat interface: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    chat()

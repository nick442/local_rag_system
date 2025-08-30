# Phase 6: CLI Interface Implementation

## Context Loading
**FIRST STEP - MANDATORY**: Read all previous handoff files:
```bash
cat handoff/phases_1_3_complete.json  # Environment setup
cat handoff/phase_4_complete.json      # RAG components
cat handoff/phase_5_complete.json      # LLM pipeline
cat handoff/vector_database_fix_complete.json # fix
```

## Your Mission
Build an interactive command-line interface that provides user-friendly access to the RAG system. This is the user-facing layer that makes the system usable.

## Prerequisites Check
1. Test RAG pipeline: `python -c "from src.rag_pipeline import RAGPipeline; r = RAGPipeline(); print('Pipeline OK')"`
2. Verify click is installed: `python -c "import click; print(click.__version__)"`
3. Verify rich is installed: `python -c "from rich.console import Console; print('Rich OK')"`

## Implementation Tasks

### Task 6.1: Interactive Chat CLI
Create `src/cli_chat.py`:

```python
# Build interactive chat using click and rich:
# 1. Main chat loop with streaming output
# 2. Commands: /help, /reset, /stats, /exit, /corpus, /config
# 3. Rich formatting for responses
# 4. Session management (in-memory history)
# 5. Real-time token streaming display
```

Core structure:
```python
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from src.rag_pipeline import RAGPipeline

@click.command()
@click.option('--model-path', default=None, help='Override model path')
@click.option('--no-streaming', is_flag=True, help='Disable streaming')
def chat(model_path, no_streaming):
    console = Console()
    rag = RAGPipeline()
    
    console.print("[bold green]RAG Chat Interface[/bold green]")
    console.print("Commands: /help, /reset, /stats, /exit, /corpus, /config")
    
    while True:
        # Get user input
        # Handle commands
        # Process queries with streaming display
        # Show token counts and timing
```

Required features:
- Color-coded output (user=blue, assistant=green, system=yellow)
- Streaming tokens with live update
- Display retrieval sources
- Show token count and generation speed
- Conversation history (last 10 turns)
- Graceful interrupt handling (Ctrl+C)

### Task 6.2: Configuration Management
Create `src/config_manager.py`:

```python
# YAML-based configuration system:
# 1. Load/save configurations
# 2. Override defaults from CLI args
# 3. Hot-reload without restart
# 4. Profile support (fast/balanced/quality)
```

Configuration structure:
```yaml
profiles:
  fast:
    retrieval_k: 3
    max_tokens: 512
    temperature: 0.7
  balanced:
    retrieval_k: 5
    max_tokens: 1024
    temperature: 0.8
  quality:
    retrieval_k: 10
    max_tokens: 2048
    temperature: 0.9

current_profile: balanced
model_overrides: {}
corpus_path: "corpus/"
```

Required methods:
- `load_config()`: Read from config/app_config.yaml
- `save_config()`: Persist changes
- `get_profile()`: Get current settings
- `switch_profile()`: Change active profile
- `override_param()`: Temporary override

### Task 6.3: Monitoring and Stats
Create `src/monitor.py`:

```python
# System monitoring using psutil:
# 1. Track memory usage
# 2. Monitor tokens/second
# 3. Log query latencies
# 4. Session statistics
```

Statistics to track:
- Total queries in session
- Average tokens/second
- Average retrieval latency
- Average generation latency
- Peak memory usage
- Current memory usage
- Token usage (prompt vs generation)
- Cache hit rate (if implemented)

Display format:
```
=== Session Statistics ===
Queries: 15
Avg Speed: 12.3 tokens/sec
Avg Retrieval: 45ms
Avg Generation: 2.1s
Memory: 4.2GB / 16GB
Total Tokens: 8,432
```

### Task 6.4: Main Application Entry Point
Create `main.py` (in project root):

```python
#!/usr/bin/env python
# Main entry point with subcommands:
# 1. chat - Interactive chat mode
# 2. query - Single query mode
# 3. ingest - Add documents to corpus
# 4. stats - Show system statistics
# 5. config - Manage configuration
```

Implementation:
```python
import click
from src.cli_chat import chat
from src.corpus_manager import ingest
# ... other imports

@click.group()
def cli():
    """Local RAG System CLI"""
    pass

cli.add_command(chat)
cli.add_command(query)
cli.add_command(ingest)
cli.add_command(stats)
cli.add_command(config)

if __name__ == '__main__':
    cli()
```

### Task 6.5: Utility Commands
Create additional CLI commands:

1. **Query command** (single-shot):
```bash
python main.py query "What is machine learning?" --k 5 --no-stream
```

2. **Stats command**:
```bash
python main.py stats --format json
```

3. **Config command**:
```bash
python main.py config set retrieval_k 7
python main.py config get-profile
python main.py config switch-profile quality
```

## Testing Requirements
Create `test_phase_6.py`:
1. Test CLI command parsing
2. Test configuration loading/saving
3. Test monitoring statistics collection
4. Test session management
5. Simulate interactive session programmatically

## Output Requirements
Create `handoff/phase_6_complete.json`:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "phase": 6,
  "created_files": [
    "src/cli_chat.py",
    "src/config_manager.py",
    "src/monitor.py",
    "main.py",
    "test_phase_6.py",
    "config/app_config.yaml"
  ],
  "cli_features": {
    "interactive_chat": true,
    "streaming_display": true,
    "command_system": true,
    "configuration_profiles": ["fast", "balanced", "quality"],
    "monitoring": true
  },
  "commands": {
    "chat": "Interactive chat mode",
    "query": "Single query execution",
    "stats": "Show statistics",
    "config": "Manage configuration"
  },
  "ui_elements": {
    "colors": true,
    "markdown_rendering": true,
    "progress_indicators": true,
    "live_streaming": true
  },
  "test_results": {
    "all_tests_passed": true,
    "interactive_test": "Manual verification needed"
  }
}
```

## User Experience Requirements
1. **Response Time**: First character appears <500ms
2. **Streaming**: Smooth token-by-token display
3. **Formatting**: Clean markdown rendering
4. **Error Handling**: Graceful errors with helpful messages
5. **Interruption**: Ctrl+C stops generation cleanly

## Sample Interaction
```
$ python main.py chat
╭─────────────────────────────────╮
│     RAG Chat Interface          │
│     Commands: /help for info    │
╰─────────────────────────────────╯

You: What is machine learning?
[Retrieving context... 5 documents found]
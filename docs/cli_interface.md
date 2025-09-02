# CLI Interface

## Overview
The CLI Interface (`main.py`) provides a comprehensive command-line interface for all RAG system operations, featuring rich terminal output, progress tracking, and intuitive command organization using Click framework and Rich library.

## Command Structure

### Main CLI Groups
The CLI is organized into logical command groups for different functional areas:

#### ingest
**Purpose**: Document ingestion and corpus building commands
**Commands**:
- `ingest directory`: Bulk directory ingestion with parallel processing
- `ingest file`: Single file ingestion

#### collection  
**Purpose**: Collection management and organization commands
**Commands**:
- `collection create`: Create new document collections
- `collection list`: Display all collections with statistics
- `collection switch`: Set default collection for operations
- `collection delete`: Remove collections and associated data
- `collection export`: Backup collections to files
- `collection import`: Restore collections from backups

#### analytics
**Purpose**: Corpus analysis and reporting commands  
**Commands**:
- `analytics stats`: Display collection statistics
- `analytics quality`: Generate quality assessment reports
- `analytics export-report`: Export comprehensive analytics reports

#### maintenance
**Purpose**: Database maintenance and optimization commands
**Commands**:
- `maintenance dedupe`: Detect and remove duplicate documents
- `maintenance reindex`: Re-index operations (rebuild, reembed, rechunk, vacuum)
- `maintenance validate`: Database integrity validation

#### Top-Level Commands
**Commands**:
- `query`: Single question RAG queries
- `chat`: Interactive conversation mode
- `status`: System status and configuration overview

## Detailed Command Reference

### Ingestion Commands

#### ingest directory
```bash
python main.py ingest directory PATH [OPTIONS]
```

**Purpose**: Bulk document ingestion with parallel processing
**Options**:
- `--pattern TEXT`: File pattern to match (default: **/* for all files)
- `--collection TEXT`: Target collection (default: default)
- `--max-workers INTEGER`: Number of parallel workers (default: 4)
- `--batch-size INTEGER`: Embedding batch size (default: 32)
- `--embedding-path TEXT`: Embedding model path (overrides default)
- `--dry-run`: Preview without processing
- `--resume/--no-resume`: Resume from checkpoint (default: enabled)
- `--deduplicate/--no-deduplicate`: Skip duplicates (default: enabled)

**Examples**:
```bash
# Basic directory ingestion
python main.py ingest directory docs/ --collection my_docs

# High-performance ingestion
python main.py ingest directory large_corpus/ --max-workers 8 --batch-size 64

# Selective ingestion with pattern
python main.py ingest directory papers/ --pattern "**/*.pdf" --collection pdf_papers

# Dry run to preview
python main.py ingest directory corpus/ --dry-run
```

**Output Format**:
```
Ingesting from docs/ ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
✓ Processing complete!
📁 Files processed: 1,247
📄 Chunks created: 9,856  
🧠 Embeddings generated: 9,856
⏱️  Processing time: 135.67s
```

#### ingest file
```bash
python main.py ingest file FILE_PATH [OPTIONS]
```

**Purpose**: Process single document file
**Options**:
- `--collection TEXT`: Target collection (default: default)
- `--embedding-path TEXT`: Embedding model path (overrides default)

**Example**:
```bash
python main.py ingest file document.pdf --collection research
```

### Collection Commands

#### collection create
```bash
python main.py collection create NAME [OPTIONS]
```

**Purpose**: Create new document collection
**Options**:
- `--description TEXT`: Collection description
- `--collection-id TEXT`: Custom collection ID

**Example**:
```bash
python main.py collection create "Research Papers" --description "Academic AI/ML papers"
```

#### collection list
```bash
python main.py collection list
```

**Purpose**: Display all collections with statistics
**Output**: Rich table with collection information
```
                         Document Collections                          
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ ID               ┃ Name           ┃ Documents ┃ Chunks ┃ Size (MB) ┃ Created    ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ research_papers  │ Research Papers│     1,247 │  9,856 │    145.67 │ 2025-08-27 │
│ technical_docs   │ Tech Docs      │       456 │  3,244 │     67.89 │ 2025-08-26 │
└──────────────────┴────────────────┴───────────┴────────┴───────────┴────────────┘
```

### Analytics Commands

#### analytics stats
```bash
python main.py analytics stats [OPTIONS]
```

**Purpose**: Display collection statistics
**Options**:
- `--collection TEXT`: Collection to analyze (default: default)

**Output**: Comprehensive statistics display
```
Collection: research_papers
📁 Documents: 1,247
📄 Chunks: 9,856
🔤 Tokens: 2,847,291
💾 Size: 145.67 MB
📊 Avg doc size: 117 KB
📝 Avg chunks/doc: 7.9

File Types:
  .pdf: 1,089
  .txt: 158
  
Most Similar Document Pairs:
  0.957: neural_networks_intro.pdf ↔ deep_learning_basics.pdf
  0.943: transformer_arch.pdf ↔ attention_mechanisms.pdf
```

#### analytics quality
```bash
python main.py analytics quality [OPTIONS]
```

**Purpose**: Generate quality assessment report
**Options**: 
- `--collection TEXT`: Collection to analyze (default: default)

**Output**: Quality scoring and recommendations
```
Quality Report: research_papers
🏆 Overall Score: 0.84 (Good)

Quality Scores:
  completeness: 0.89
  chunk_consistency: 0.82  
  embedding_coverage: 0.91
  content_relevance: 0.78

Recommendations:
  • Consider increasing chunk overlap to improve context preservation
  • 12% of documents have inconsistent chunk sizes
  • Add more descriptive metadata to improve searchability
```

### Maintenance Commands

#### maintenance dedupe  
```bash
python main.py maintenance dedupe [OPTIONS]
```

**Purpose**: Detect and remove duplicate documents
**Options**:
- `--collection TEXT`: Collection to deduplicate (default: default)  
- `--dry-run`: Preview without making changes

**Example**:
```bash
python main.py maintenance dedupe --collection research --dry-run
```

**Output**: Deduplication analysis and results
```
Deduplication Report: research
📁 Total documents: 1,247
✨ Unique documents: 1,198
🔄 Duplicate groups: 49
💾 Potential space savings: 12.34MB

Proceed with duplicate removal? [y/N]: y

✓ Deduplication complete!
🗑️  Documents removed: 49
💾 Space saved: 12.34MB
```

#### maintenance reindex
```bash  
python main.py maintenance reindex [OPTIONS]
```

**Purpose**: Database re-indexing and optimization
**Options**:
- `--collection TEXT`: Collection to reindex (default: default)
- `--operation TEXT`: Operation type (reembed|rechunk|rebuild|vacuum)
- `--backup/--no-backup`: Backup before operation (default: enabled)

**Examples**:
```bash
# Rebuild database indices
python main.py maintenance reindex --operation rebuild

# Re-embed collection with new model
python main.py maintenance reindex --operation reembed --collection papers

# Database vacuum and cleanup
python main.py maintenance reindex --operation vacuum
```

### Query Commands

#### query
```bash
python main.py query "QUESTION" [OPTIONS]
```

**Purpose**: Single RAG query execution
**Options**:
- `--collection TEXT`: Collection to query (default: default)
- `--model-path TEXT`: LLM model path 
- `--embedding-path TEXT`: Embedding model path
- `--k INTEGER`: Number of documents to retrieve (default: 5)
- `--metrics/--no-metrics`: Enable JSONL metrics logging to `logs/metrics.jsonl`

**Example**:
```bash
python main.py query "What is machine learning?" --collection ai_papers --k 3 --metrics
```

**Output**: Answer with source attribution
```
Question: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence that enables 
computers to learn and improve from experience without being explicitly 
programmed...

Sources:
📄 ml_fundamentals.pdf (similarity: 0.89)
📄 ai_introduction.txt (similarity: 0.84)  
📄 statistical_learning.pdf (similarity: 0.81)

Retrieved 3 relevant documents in 1.2ms
Response generated in 847ms
```

#### chat
```bash
python main.py chat [OPTIONS]
```

**Purpose**: Interactive conversation mode
**Options**:
- `--collection TEXT`: Collection to query (default: default)
- `--model-path TEXT`: LLM model path
- `--embedding-path TEXT`: Embedding model path

**Features**:
- **Multi-turn Conversations**: Maintains context across exchanges
- **Source Attribution**: Shows sources for each response
- **Real-time Processing**: Streaming responses when supported
- **Session Management**: Clean exit and context cleanup

**Example Session**:
```bash
python main.py chat --collection research

🤖 RAG Chat Interface (research collection)
Type 'exit', 'quit', or 'bye' to end the conversation.

You: What are transformers in AI?

🤖: Transformers are a neural network architecture introduced in the paper 
"Attention Is All You Need"...

📚 Sources: transformer_paper.pdf, attention_mechanisms.txt

You: How do they differ from RNNs?

🤖: Unlike RNNs which process sequences sequentially, transformers...
```

## CLI Features

### Rich Terminal Output
**Visual Enhancement**: Beautiful terminal interface using Rich library
- **Progress Bars**: Real-time progress indication for long operations
- **Colored Output**: Status indicators and semantic coloring
- **Tables**: Formatted tables for structured data display
- **Icons**: Emoji icons for visual clarity and intuition
- **Styling**: Consistent styling across all commands

### Error Handling
**User-Friendly Errors**: Clear error messages with actionable guidance
```bash
❌ Collection 'nonexistent' not found
💡 Available collections: default, research_papers, technical_docs
💡 Use 'python main.py collection list' to see all collections
```

### Help System
**Comprehensive Help**: Built-in help for all commands and options
```bash
# Main help
python main.py --help

# Command group help  
python main.py ingest --help

# Specific command help
python main.py ingest directory --help
```

### Configuration Management
**Global Configuration**: System-wide configuration options
```bash
python main.py --db-path custom/vectors.db analytics stats
python main.py --verbose ingest directory docs/  # Detailed logging
python main.py --config-file custom_config.yaml query "test"
```

## Integration with Components

### Backend Integration
**Seamless Component Access**: CLI provides access to all backend capabilities
- **Corpus Manager**: Bulk ingestion operations
- **Corpus Organizer**: Collection management
- **Document Deduplicator**: Duplicate detection and removal
- **Reindex Tools**: Maintenance and optimization
- **Corpus Analytics**: Analysis and reporting
- **RAG Pipeline**: Query processing and chat functionality

### Configuration Flow
**Parameter Passing**: CLI parameters flow through to backend components
```
CLI Options → Click Framework → Component Configuration → Backend Services
```

## Usage Patterns

### Development Workflow
```bash
# 1. Create collection for new project
python main.py collection create "Project Docs" --description "Documentation for X project"

# 2. Ingest documents
python main.py ingest directory project_docs/ --collection project_docs

# 3. Check quality and statistics
python main.py analytics stats --collection project_docs
python main.py analytics quality --collection project_docs

# 4. Test retrieval
python main.py query "How to configure X?" --collection project_docs

# 5. Interactive exploration
python main.py chat --collection project_docs
```

### Production Maintenance
```bash
# Daily monitoring
python main.py status
python main.py analytics stats

# Weekly maintenance  
python main.py maintenance dedupe --dry-run
python main.py maintenance validate

# Monthly optimization
python main.py maintenance reindex --operation rebuild
python main.py maintenance reindex --operation vacuum
```

### Batch Operations
```bash
# Process multiple directories
for dir in docs1/ docs2/ docs3/; do
    python main.py ingest directory "$dir" --collection "$(basename $dir)"
done

# Generate reports for all collections
python main.py collection list --output collections.json
python main.py analytics export-report --collection research
```

## Performance Features

### Progress Tracking
**Real-time Feedback**: All long-running operations show progress
- **Progress Bars**: Visual indication of completion percentage
- **Time Estimates**: ETA based on current processing speed
- **Throughput Metrics**: Documents/second processing rates
- **Memory Usage**: Optional memory usage monitoring

### Parallel Processing
**Optimized Performance**: Automatic optimization for system capabilities
- **Worker Auto-detection**: Optimal worker count based on CPU cores
- **Batch Size Tuning**: Automatic batch size adjustment
- **Resource Monitoring**: CPU and memory usage awareness
- **Graceful Degradation**: Automatic adjustment on resource constraints

### Streaming Output
**Responsive Interface**: Immediate feedback and streaming updates
- **Real-time Logs**: Important events displayed immediately
- **Streaming Responses**: Chat responses stream in real-time
- **Progress Updates**: Frequent progress bar updates
- **Error Reporting**: Immediate error notification and guidance

## Error Handling and User Experience

### Error Messages
**Clear Communication**: User-friendly error messages with solutions
```bash
❌ Failed to load embedding model
💡 Check that the model path exists: models/embeddings/all-MiniLM-L6-v2
💡 Run 'python main.py status' to verify system configuration
💡 Use --verbose for detailed error information
```

### Input Validation
**Proactive Validation**: Catch errors before expensive operations
- **Path Validation**: Check file and directory existence
- **Parameter Validation**: Validate option values and combinations
- **Resource Checks**: Verify sufficient disk space and memory
- **Dependency Verification**: Check for required models and libraries

### User Guidance
**Helpful Suggestions**: Guide users toward successful operations
- **Command Suggestions**: Suggest related commands for workflow
- **Option Recommendations**: Recommend optimal parameters
- **Troubleshooting Tips**: Built-in troubleshooting guidance
- **Documentation Links**: References to detailed documentation

## Configuration and Customization

### Global Configuration
**System-wide Settings**: Configure default behavior
```python
# Default paths and models
DEFAULT_DB_PATH = "data/rag_vectors.db"
DEFAULT_LLM_PATH = "models/llm/gemma-3-4b-it-q4_0.gguf"  
DEFAULT_EMBEDDING_PATH = "models/embeddings/all-MiniLM-L6-v2"

# Performance settings
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_K = 5
```

### Environment Integration
**Environment Awareness**: Adapts to system configuration
- **Path Detection**: Automatic model path detection
- **Resource Limits**: Respect system resource constraints
- **Conda Integration**: Works seamlessly with conda environments
- **Cross-platform**: Consistent behavior across operating systems

## Advanced Features

### Verbose Mode
**Detailed Logging**: Comprehensive operation logging for debugging
```bash
python main.py --verbose ingest directory docs/
```

**Verbose Output Includes**:
- Detailed progress information
- Individual file processing logs
- Performance metrics and timing
- Memory usage statistics
- Complete error stack traces

### Batch Processing Support
**Automation-Friendly**: Designed for scripting and automation
```bash
# Automation example
python main.py ingest directory "$INPUT_DIR" --collection "$COLLECTION_NAME" 2>&1 | tee ingestion.log
python main.py analytics stats --collection "$COLLECTION_NAME" > stats.json
```

### Integration Hooks
**External Integration**: Support for external tools and workflows
- **Exit Codes**: Proper exit codes for script integration
- **JSON Output**: Machine-readable output options
- **Log Formatting**: Structured logging for external processing
- **Status Reporting**: Programmatic status checking

## Output Formatting

### Rich Terminal Output
**Beautiful Interface**: Professional terminal interface using Rich library

**Visual Elements**:
- **Colors**: Semantic coloring (green for success, red for errors, yellow for warnings)
- **Icons**: Emoji icons for visual clarity
- **Tables**: Formatted tables for structured data
- **Progress Bars**: Animated progress indication
- **Panels**: Grouped information display

### Status Indicators
**Clear Status Communication**: Consistent status indication across all commands
- ✅ **Success**: Green checkmarks for successful operations
- ❌ **Error**: Red X marks for failures  
- ⚠️ **Warning**: Yellow warning icons for issues
- ℹ️ **Info**: Blue information icons for notifications
- 🔄 **Progress**: Spinning indicators for ongoing operations

### Data Presentation
**Structured Information**: Organized display of complex data
```bash
# Collection statistics table
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Collection ID    ┃ Name           ┃ Documents ┃ Size   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ research_papers  │ Research       │     1,247 │ 145 MB │
└──────────────────┴────────────────┴───────────┴────────┘

# Quality assessment display
🏆 Overall Score: 0.84 (Good)

Quality Scores:
  completeness: 0.89
  consistency: 0.82
  coverage: 0.91
```

## Best Practices

### Command Usage Patterns
**Efficient Workflows**: Recommended command sequences for common tasks
1. **New Project Setup**: `collection create` → `ingest directory` → `analytics stats`
2. **Regular Maintenance**: `analytics quality` → `maintenance dedupe` → `maintenance reindex`
3. **Query Testing**: `query` for single tests → `chat` for interactive exploration
4. **Performance Optimization**: `maintenance validate` → `maintenance reindex --operation rebuild`

### Error Recovery
**Robust Operation**: Handle errors gracefully and provide recovery options
- **Resume Capability**: All long operations support resume from interruption
- **Partial Success**: Report partial success and continue where possible  
- **Clear Error Messages**: Specific guidance for error resolution
- **Safe Defaults**: Conservative defaults to prevent data loss

### Performance Optimization
**Efficient Resource Usage**: Optimize CLI performance for user experience
- **Fast Startup**: Minimal startup time through lazy loading
- **Responsive UI**: Quick response to user input
- **Background Processing**: Long operations don't block interface
- **Memory Efficiency**: Minimize memory usage during operations

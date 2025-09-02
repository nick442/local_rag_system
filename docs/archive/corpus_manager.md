# Corpus Manager

## Overview
The Corpus Manager (`src/corpus_manager.py`) provides comprehensive bulk document processing capabilities with parallel execution, progress tracking, checkpointing, and duplicate detection for large-scale corpus ingestion.

## Core Classes

### CorpusManager  
**Purpose**: Main service for bulk document processing and corpus management
**Key Features**:
- **Parallel Processing**: Configurable multi-worker document processing
- **Progress Tracking**: Real-time progress bars and detailed statistics
- **Checkpointing**: Resume capability for interrupted processing
- **Duplicate Detection**: Content-based deduplication during ingestion
- **Error Recovery**: Graceful handling of individual file failures

**Configuration Parameters**:
- `db_path`: Vector database file path
- `embedding_model_path`: Embedding model location
- `max_workers`: Number of parallel processing workers (1-16)
- `checkpoint_interval`: Save progress every N files (default: 10)
- `batch_size`: Embedding generation batch size (default: 32)

### ProcessingStats
**Purpose**: Comprehensive statistics tracking for bulk operations
**Key Metrics**:
- `files_scanned`: Total files found in directory
- `files_processed`: Successfully processed files
- `files_skipped`: Files skipped (duplicates, already processed)
- `files_failed`: Files that failed processing
- `chunks_created`: Total document chunks generated
- `chunks_embedded`: Total embeddings created
- `processing_time`: Total operation duration
- `start_time`/`end_time`: Timestamps for performance analysis

**Serialization**: JSON-compatible with proper datetime handling

### CheckpointData
**Purpose**: Persistent state for resume capability
**Key Components**:
- `processed_files`: Set of completed file paths
- `stats`: Current processing statistics
- `collection_id`: Target collection identifier
- `timestamp`: Checkpoint creation time

**Features**:
- **Automatic Saving**: Checkpoints saved every N processed files
- **Resume Detection**: Automatically detects and resumes from latest checkpoint
- **Cleanup**: Old checkpoints automatically cleaned up

## Key Methods

### ingest_directory(path, pattern="**/*", collection_id="default", ...) -> ProcessingStats
**Purpose**: Bulk ingest documents from directory with full processing pipeline
**Features**:
- **Pattern Matching**: Glob patterns for flexible file selection
- **Parallel Execution**: Multi-worker processing for performance
- **Resume Support**: Automatic resume from interruption
- **Duplicate Handling**: Skip already processed files
- **Dry Run Mode**: Preview processing without making changes

**Processing Pipeline**:
1. **Directory Scanning**: Find all matching files using glob patterns
2. **Duplicate Detection**: Hash-based duplicate identification
3. **Resume Logic**: Skip files already processed (if resume enabled)
4. **Parallel Processing**: Distribute work across worker threads
5. **Progress Tracking**: Real-time progress bars and statistics
6. **Checkpointing**: Periodic state saves for recovery
7. **Final Statistics**: Comprehensive processing report

**Configuration Options**:
```python
stats = await manager.ingest_directory(
    path="data/corpus",
    pattern="**/*.{txt,pdf,md}",  # Multiple formats
    collection_id="research_papers",
    dry_run=False,                # Actually process files
    resume=True,                  # Resume from checkpoint
    deduplicate=True              # Skip duplicate files
)
```

### process_single_document(file_path: Path, collection_id: str) -> Tuple[bool, Dict]
**Purpose**: Process individual document through complete ingestion pipeline
**Processing Steps**:
1. **Document Loading**: Load and parse file using appropriate loader
2. **Chunk Generation**: Split document into overlapping chunks
3. **ID Assignment**: Generate unique document and chunk identifiers
4. **Embedding Generation**: Create vector embeddings for all chunks
5. **Database Storage**: Store document, chunks, and embeddings
6. **Statistics Update**: Track processing metrics

**Error Handling**: 
- Continues processing other files if individual files fail
- Detailed error logging for debugging
- Returns success status and detailed statistics

### scan_directory(path: Path, pattern: str, supported_extensions: Set[str]) -> List[Path]
**Purpose**: Efficient directory scanning with format filtering
**Features**:
- **Glob Pattern Support**: Complex patterns like `**/*.{txt,pdf}`
- **Extension Filtering**: Only includes supported file types
- **Recursive Scanning**: Deep directory traversal
- **Sorted Results**: Consistent processing order

### check_duplicates(file_paths: List[Path]) -> Tuple[List[Path], Dict[str, List[Path]]]
**Purpose**: Detect duplicate files based on content hashing
**Algorithm**:
1. Generate SHA-256 hash for each file's content
2. Group files by identical hashes
3. Keep first occurrence, mark others as duplicates
4. Return unique files and duplicate mapping

**Performance**: Scales efficiently to 10k+ files with minimal memory usage

## Performance Optimization

### Parallel Processing
**Worker Management**:
- **Optimal Workers**: 4-8 workers for most systems
- **CPU Utilization**: Balances processing and I/O operations
- **Memory Management**: Prevents excessive memory usage
- **Error Isolation**: Individual worker failures don't affect others

**Performance Scaling**:
- **1 Worker**: ~3-5 documents/second
- **4 Workers**: ~12-20 documents/second  
- **8 Workers**: ~20-35 documents/second
- **16+ Workers**: Diminishing returns due to I/O limits

### Memory Management
**Efficient Processing**:
- **Streaming**: Process one document at a time per worker
- **Batch Embeddings**: Group embedding generation for efficiency
- **Cache Control**: Limited caching to prevent memory bloat
- **Resource Cleanup**: Automatic cleanup of temporary resources

### Checkpointing Strategy
**Resume Capability**:
- **Granular Checkpoints**: Save every 10 files by default
- **Fast Resume**: Quick startup from checkpoint state
- **Crash Recovery**: Robust handling of system interruptions
- **Progress Preservation**: No duplicate work on resume

## Error Handling and Recovery

### File-Level Error Handling
**Robust Processing**:
- **Individual Failures**: Single file failures don't stop batch processing
- **Error Classification**: Different handling for different error types
- **Detailed Logging**: Comprehensive error information for debugging
- **Recovery Strategies**: Automatic retry for transient failures

### Common Error Scenarios
- **Corrupted Files**: Skip and log corrupted or unreadable files
- **Memory Issues**: Automatic worker count adjustment
- **Database Errors**: Transaction rollback and retry mechanisms
- **Model Failures**: Graceful degradation and error reporting

### Monitoring and Diagnostics
- **Processing Metrics**: Real-time performance monitoring
- **Error Rates**: Track failure patterns and rates
- **Resource Usage**: Monitor memory and CPU utilization
- **Quality Metrics**: Assess processing quality and completeness

## Usage Examples

### Basic Directory Ingestion
```python
# Initialize manager
manager = CorpusManager(
    db_path="data/vectors.db",
    embedding_model_path="models/all-MiniLM-L6-v2",
    max_workers=4
)

# Process directory
stats = await manager.ingest_directory(
    path="data/documents",
    pattern="**/*.txt",
    collection_id="my_collection"
)

print(f"Processed {stats.files_processed} files")
print(f"Created {stats.chunks_created} chunks")
print(f"Processing time: {stats.processing_time:.2f}s")
```

### Advanced Configuration
```python
# High-performance setup
manager = CorpusManager(
    db_path="data/vectors.db",
    max_workers=8,           # More workers for faster processing
    checkpoint_interval=5,   # More frequent checkpoints
    batch_size=64           # Larger embedding batches
)

# Process with all options
stats = await manager.ingest_directory(
    path="large_corpus/",
    pattern="**/*.{txt,pdf,md,html}",
    collection_id="research",
    dry_run=False,          # Actually process
    resume=True,            # Resume from checkpoint
    deduplicate=True        # Skip duplicates
)
```

### Monitoring Long-Running Jobs
`ingest_directory` returns a `ProcessingStats` object with detailed metrics. For ad‑hoc telemetry (database status, config), use `get_processing_stats()`:
```python
stats = await manager.ingest_directory(...)
print(stats.files_processed, stats.chunks_created, stats.processing_time)

runtime = manager.get_processing_stats()
print(runtime['database_stats']['documents'], runtime['processing_config']['batch_size'])
```

## Integration with Phase 7 Components

### Collection Management
**Seamless Integration**: Works with CorpusOrganizer for collection-based processing
- Collections created automatically if they don't exist
- Collection metadata updated during processing
- Statistics integrated with collection analytics

### Deduplication Integration
**Multi-Level Deduplication**:
- **File-level**: Hash-based duplicate detection during scanning
- **Content-level**: Integration with DocumentDeduplicator for advanced detection
- **Cross-collection**: Duplicate detection across multiple collections

### Analytics Integration
**Processing Insights**:
- **Quality Assessment**: Automatic quality scoring during ingestion
- **Performance Metrics**: Processing speed and efficiency analysis
- **Content Analysis**: Document type and size distribution
- **Error Analysis**: Failure pattern identification

## Best Practices

### Performance Optimization
- **Worker Count**: Start with 4 workers, adjust based on system performance
- **Batch Size**: Use 32 for most scenarios, increase to 64 for powerful systems
- **Pattern Specificity**: Use specific patterns to avoid processing unwanted files
- **Collection Organization**: Use separate collections for different document types

### Error Prevention
- **Validate Paths**: Ensure source directories exist and are readable
- **Check Disk Space**: Verify sufficient storage before large ingestions
- **Monitor Memory**: Watch memory usage during large batch processing
- **Backup Database**: Create backups before major ingestion operations

### Production Deployment
- **Logging Configuration**: Set appropriate log levels for production
- **Checkpoint Cleanup**: Regular cleanup of old checkpoint files
- **Monitoring Setup**: Monitor processing metrics and error rates
- **Resource Limits**: Set appropriate limits for concurrent processing

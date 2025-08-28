# Re-indexing and Maintenance Tools

## Overview
The Re-indexing Tools (`src/reindex.py`) provide advanced maintenance capabilities for corpus optimization, including re-embedding, re-chunking, index rebuilding, and database optimization operations.

## Core Classes

### ReindexTool
**Purpose**: Comprehensive maintenance and optimization service for RAG system databases
**Key Features**:
- **Re-embedding**: Update embeddings with new or improved models
- **Re-chunking**: Rebuild chunks with different parameters or algorithms
- **Index Rebuilding**: Optimize database indices for performance
- **Database Maintenance**: Vacuum, analyze, and integrity checking
- **Backup Management**: Automatic backups before destructive operations

**Safety Features**:
- **Automatic Backups**: Database backup before all operations
- **Transaction Safety**: All operations wrapped in transactions
- **Validation**: Pre and post-operation integrity checks
- **Rollback Capability**: Restore from backup if operations fail

### ReindexStats
**Purpose**: Detailed statistics and reporting for maintenance operations
**Key Metrics**:
- `operation_type`: Type of maintenance operation performed
- `documents_processed`: Number of documents affected
- `chunks_processed`: Number of chunks updated
- `processing_time`: Total operation duration
- `success`: Boolean indicating operation success
- `details`: Operation-specific details and metrics
- `backup_info`: Information about created backups

## Key Operations

### Re-embedding Operations

#### reembed_collection(collection_id: str, new_embedding_service: EmbeddingService = None) -> ReindexStats
**Purpose**: Update all embeddings in a collection with new or improved embedding model
**Use Cases**:
- **Model Upgrades**: Switch to newer, better embedding models
- **Dimension Changes**: Migrate from 384-dim to 768-dim embeddings
- **Domain Optimization**: Use domain-specific embedding models
- **Quality Improvement**: Replace poor-quality embeddings

**Process**:
1. **Backup Database**: Create full backup before starting
2. **Extract Chunks**: Retrieve all chunk content for re-embedding
3. **Generate Embeddings**: Create new embeddings with updated model
4. **Update Database**: Replace old embeddings with new ones
5. **Rebuild Indices**: Reconstruct vector search indices
6. **Validation**: Verify embedding integrity and coverage

**Performance**: 
- ~50-200 chunks/second depending on model and hardware
- Memory usage scales with batch size
- Supports incremental processing for very large collections

#### reembed_document(doc_id: str, new_embedding_service: EmbeddingService = None) -> ReindexStats
**Purpose**: Re-embed a specific document (granular control)
**Features**: Same as collection re-embedding but scoped to single document

### Re-chunking Operations  

#### rechunk_documents(collection_id: str, new_chunk_size: int = 1000, new_overlap: int = 200) -> ReindexStats
**Purpose**: Rebuild document chunks with new parameters
**Use Cases**:
- **Chunk Size Optimization**: Adjust chunk size for better retrieval performance
- **Overlap Adjustment**: Modify overlap for better context preservation
- **Algorithm Updates**: Use improved chunking algorithms
- **Format Optimization**: Optimize chunks for specific embedding models

**Process**:
1. **Document Retrieval**: Get original document content
2. **Rechunking**: Apply new chunking parameters
3. **Embedding Generation**: Create embeddings for new chunks
4. **Database Update**: Replace old chunks and embeddings
5. **Reference Update**: Update all foreign key references
6. **Index Rebuild**: Reconstruct search indices

**Considerations**:
- **Token Limits**: Ensure new chunks fit embedding model limits
- **Context Quality**: Balance chunk size with context coherence
- **Performance Impact**: Larger chunks = fewer embeddings but less granular search

### Index Maintenance

#### rebuild_indices(backup: bool = True) -> ReindexStats
**Purpose**: Reconstruct all database indices for optimal performance
**Features**:
- **Performance Recovery**: Restore optimal query performance
- **Index Optimization**: Use latest SQLite optimization techniques
- **Vector Index Rebuild**: Reconstruct sqlite-vec indices
- **Statistics Update**: Refresh database query statistics

**Operations**:
1. **Backup Creation**: Full database backup (if enabled)
2. **Index Analysis**: Identify problematic or inefficient indices
3. **Index Dropping**: Remove existing indices
4. **Index Recreation**: Build optimized indices with current data
5. **Statistics Collection**: Update query planner statistics
6. **Performance Validation**: Verify improved performance

**Performance Impact**:
- **Temporary Slowdown**: Queries slower during rebuilding
- **Long-term Improvement**: 2-10x faster queries after completion
- **Storage Overhead**: Temporary additional disk space usage

#### vacuum_database(backup: bool = True) -> ReindexStats  
**Purpose**: Reclaim space and optimize database storage
**Features**:
- **Space Reclamation**: Free space from deleted records
- **File Optimization**: Reduce database file size
- **Performance Improvement**: Better cache locality
- **Fragmentation Reduction**: Consolidate scattered data

**Operations**:
1. **Space Analysis**: Identify reclaimable space
2. **Backup Creation**: Safety backup before operation
3. **VACUUM Operation**: SQLite VACUUM command execution
4. **Statistics Update**: Refresh database statistics
5. **Performance Test**: Validate improved performance

## Advanced Maintenance Features

### Integrity Validation
**Purpose**: Comprehensive database health checking and repair
```python
def validate_integrity(collection_id: str = None) -> Dict[str, Any]:
    return {
        'overall_status': 'PASS' | 'FAIL',
        'checks_passed': 12,
        'checks_failed': 0,
        'issues': [],
        'statistics': {...},
        'recommendations': [...]
    }
```

**Validation Checks**:
- **Foreign Key Integrity**: Verify all references are valid
- **Embedding Coverage**: Ensure all chunks have embeddings
- **Data Consistency**: Check for orphaned records
- **Index Integrity**: Validate index consistency
- **File System Sync**: Verify database matches file system state

### Backup and Recovery
**Automatic Backup Management**:
- **Pre-operation Backups**: Automatic backup before destructive operations
- **Incremental Backups**: Efficient backup of changes only
- **Backup Validation**: Verify backup integrity
- **Restore Capabilities**: Easy restoration from backup files

**Backup Formats**:
- **Full Database**: Complete SQLite database copy
- **Collection Export**: JSON-based collection backups
- **Schema Only**: Structure without data for testing
- **Custom Exports**: Selective data export with filtering

### Performance Monitoring
**Operation Tracking**: Monitor maintenance operation performance
- **Duration Tracking**: Time all operations for performance analysis
- **Resource Monitoring**: Track CPU, memory, and I/O usage
- **Progress Reporting**: Real-time progress indication
- **Error Tracking**: Log and categorize maintenance errors

## Safety and Recovery

### Safety Protocols
**Risk Mitigation**:
- **Required Backups**: Cannot disable backups for destructive operations
- **Confirmation Prompts**: User confirmation for risky operations
- **Dry Run Support**: Preview operation effects without changes
- **Transaction Atomicity**: All-or-nothing operation guarantees

### Recovery Procedures
**Failure Recovery**:
1. **Automatic Rollback**: Transaction rollback on operation failure
2. **Backup Restoration**: Restore from automatic backup if needed
3. **Partial Recovery**: Recover successfully processed portions
4. **Error Analysis**: Detailed error reporting for debugging

## Integration Examples

### Scheduled Maintenance
```python
# Weekly maintenance routine
async def weekly_maintenance(db_path: str):
    tool = ReindexTool(db_path)
    
    # Validate integrity
    integrity_report = tool.validate_integrity()
    if integrity_report['overall_status'] != 'PASS':
        print("Database integrity issues detected!")
    
    # Rebuild indices for performance
    rebuild_stats = tool.rebuild_indices(backup=True)
    
    # Vacuum database for space reclamation
    vacuum_stats = tool.vacuum_database(backup=False)  # Already backed up
    
    print(f"Maintenance complete: {vacuum_stats.processing_time:.2f}s")
```

### Model Migration
```python
# Migrate to new embedding model
new_embedding_service = EmbeddingService(
    model_path="models/new-model-v2",
    batch_size=64
)

# Re-embed entire collection
reembed_stats = tool.reembed_collection(
    collection_id="research_papers",
    new_embedding_service=new_embedding_service
)

print(f"Re-embedded {reembed_stats.chunks_processed} chunks")
```

### Performance Optimization Workflow
```python
# Comprehensive optimization
tool = ReindexTool("data/vectors.db")

# Step 1: Rechunk with optimized parameters
rechunk_stats = tool.rechunk_documents(
    collection_id="docs",
    new_chunk_size=768,  # Larger chunks for better context
    new_overlap=128      # Smaller overlap for efficiency
)

# Step 2: Rebuild indices
rebuild_stats = tool.rebuild_indices()

# Step 3: Vacuum for space efficiency
vacuum_stats = tool.vacuum_database()

print(f"Optimization complete: {vacuum_stats.details['space_saved_mb']:.2f}MB saved")
```

## Configuration Guidelines

### Operation Scheduling
- **Off-Peak Hours**: Schedule maintenance during low-usage periods
- **Incremental Updates**: Process subsets of large collections
- **Resource Allocation**: Ensure sufficient system resources
- **Backup Storage**: Verify adequate backup storage space

### Parameter Tuning
- **Chunk Size Selection**: Balance context quality with search granularity
- **Overlap Configuration**: Preserve context while minimizing redundancy
- **Batch Size Optimization**: Balance memory usage with processing speed
- **Threshold Adjustment**: Tune based on collection characteristics

### Monitoring and Alerts
- **Operation Monitoring**: Track maintenance operation success rates
- **Performance Alerts**: Alert on significant performance degradation
- **Storage Monitoring**: Monitor database and backup storage usage
- **Quality Tracking**: Track corpus quality metrics over time
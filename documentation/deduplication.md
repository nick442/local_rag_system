# Document Deduplication System

## Overview  
The Document Deduplication System (`src/deduplication.py`) provides comprehensive duplicate detection using multiple algorithms to identify exact, fuzzy, semantic, and metadata-based duplicates across document collections.

## Core Classes

### DocumentDeduplicator
**Purpose**: Main service for multi-method duplicate detection and resolution
**Key Features**:
- **Multi-Method Detection**: Four complementary detection algorithms
- **Scalable Processing**: Efficient algorithms for large corpora (10k+ documents)
- **Flexible Resolution**: Multiple strategies for handling duplicates
- **Detailed Reporting**: Comprehensive analysis with statistics and recommendations
- **Performance Optimization**: Optimized data structures and algorithms

**Detection Methods**:
1. **Exact Duplicates**: SHA-256 content hashing
2. **Fuzzy Duplicates**: MinHash LSH with Jaccard similarity  
3. **Semantic Duplicates**: Embedding cosine similarity
4. **Metadata Duplicates**: File name and size matching

### DuplicateGroup
**Purpose**: Represents a group of duplicate documents with similarity metrics
**Key Attributes**:
- `documents`: List of document IDs in duplicate group
- `similarity_score`: Similarity score between documents (0.0-1.0)
- `detection_method`: Method used to detect duplicates
- `representative_doc`: Best document to keep from group
- `metadata`: Additional information about the duplicate relationship

### DeduplicationReport
**Purpose**: Comprehensive analysis report of duplicate detection results
**Key Metrics**:
- `total_documents`: Total documents analyzed
- `unique_documents`: Number of truly unique documents
- `duplicate_groups`: List of detected duplicate groups
- `space_saved_mb`: Potential storage savings from deduplication
- `processing_time`: Analysis duration
- `method_breakdown`: Statistics per detection method

## Detection Algorithms

### Exact Duplicate Detection
**Algorithm**: SHA-256 content hashing for identical document detection
**Features**:
- **Perfect Accuracy**: 100% accurate for identical content
- **Fast Performance**: O(n) complexity with efficient hashing
- **Memory Efficient**: Minimal memory overhead
- **Cross-Format**: Detects duplicates regardless of file format

**Use Cases**:
- Identical files with different names
- Same content in different locations
- Accidental duplicate uploads
- File system synchronization issues

**Performance**: ~1000 documents/second

### Fuzzy Duplicate Detection  
**Algorithm**: MinHash LSH (Locality-Sensitive Hashing) with Jaccard similarity
**Features**:
- **Near-Duplicate Detection**: Finds documents with slight variations
- **Configurable Threshold**: Adjustable similarity threshold (default: 0.8)
- **Scalable**: Efficient processing for large document collections
- **Text Normalization**: Handles formatting and minor text differences

**Technical Details**:
- **MinHash Parameters**: 128 permutations for good accuracy
- **LSH Bands**: Optimized for 0.8+ similarity detection
- **Jaccard Index**: Measures set similarity of word n-grams
- **Shingling**: 3-gram character shingles for text comparison

**Use Cases**:
- Documents with minor edits or revisions
- Same content with different formatting
- OCR variations of the same document
- Translated documents with similar structure

**Performance**: ~500-1000 documents/second

### Semantic Duplicate Detection
**Algorithm**: Embedding cosine similarity for conceptually identical documents
**Features**:
- **Semantic Understanding**: Detects conceptually identical content
- **High Threshold**: Uses 0.95+ similarity for duplicate classification
- **Cross-Language**: Works across languages with multilingual embeddings
- **Concept-Based**: Identifies same ideas expressed differently

**Technical Details**:
- **Embedding Comparison**: Document-level embedding similarity
- **Cosine Similarity**: Measures semantic relationship
- **Threshold Tuning**: High threshold (0.95) to avoid false positives
- **Aggregation Strategy**: Document embeddings from chunk aggregation

**Use Cases**:
- Same ideas expressed with different words
- Translations of identical content
- Summaries vs full documents (when very similar)
- Reformatted or restructured identical information

**Performance**: ~100-500 documents/second (depends on embedding model)

### Metadata Duplicate Detection
**Algorithm**: File name and size pattern matching for structural duplicates
**Features**:
- **Fast Screening**: Quick detection based on metadata only
- **Pattern Recognition**: Identifies naming patterns and conventions
- **Size Correlation**: Uses file size as additional signal
- **Pre-filtering**: Reduces workload for content-based methods

**Technical Details**:
- **Name Similarity**: Levenshtein distance for file names
- **Size Matching**: Exact or near-exact file size matching
- **Extension Handling**: Considers file format differences
- **Path Analysis**: Identifies moved or renamed files

**Use Cases**:
- Same files in different directories
- Files renamed with timestamps or versions
- Backup copies with modified names
- Cross-platform file transfers

**Performance**: ~5000+ documents/second

## Deduplication Workflow

### Analysis Phase
**Process**: Comprehensive duplicate detection across all methods
```python
def analyze_duplicates(self, collection_id: str = "default") -> DeduplicationReport:
    # 1. Extract document metadata and content
    # 2. Apply all four detection methods in parallel
    # 3. Combine results and resolve conflicts  
    # 4. Generate comprehensive report with recommendations
    # 5. Calculate potential space savings
```

**Performance Optimization**:
- **Parallel Processing**: Multiple detection methods run concurrently
- **Early Termination**: Skip expensive methods when exact duplicates found
- **Memory Management**: Streaming processing for large collections
- **Incremental Analysis**: Process new documents against existing corpus

### Resolution Phase
**Process**: Remove or merge duplicate documents based on configured strategy
```python
def resolve_duplicates(self, duplicate_groups: List[DuplicateGroup], 
                      collection_id: str = "default", 
                      strategy: str = "keep_first") -> Dict:
```

**Resolution Strategies**:
- **keep_first**: Keep earliest ingested document, remove others
- **keep_largest**: Keep document with most content, remove others
- **manual_review**: Flag for human review, no automatic action
- **merge**: Combine metadata and keep best representative
- **keep_highest_quality**: Use quality scoring to select best document

### Batch Operations
**Efficient Processing**: Optimized for large-scale deduplication
- **Transaction Batching**: Group database operations for performance
- **Memory Streaming**: Process large collections without memory issues
- **Progress Tracking**: Real-time progress indication
- **Error Recovery**: Continue processing despite individual failures

## Performance Characteristics

### Scalability Metrics
- **Small Collections** (< 1k docs): <30 seconds full analysis
- **Medium Collections** (1k-10k docs): 2-10 minutes full analysis  
- **Large Collections** (10k+ docs): 10-60 minutes full analysis
- **Memory Usage**: <2GB for 10k documents during analysis

### Algorithm Performance
| Method | Speed (docs/sec) | Accuracy | Memory Usage | Best For |
|--------|------------------|----------|--------------|----------|
| Exact | 1000+ | 100% | Low | Identical files |
| Fuzzy | 500-1000 | 90-95% | Medium | Minor variations |
| Semantic | 100-500 | 85-95% | High | Concept duplicates |
| Metadata | 5000+ | 70-85% | Very Low | Quick screening |

### Optimization Strategies
- **Method Ordering**: Run exact detection first to eliminate obvious duplicates
- **Threshold Tuning**: Adjust thresholds based on collection characteristics
- **Parallel Processing**: Multiple methods and documents processed concurrently
- **Caching**: Cache expensive computations for repeated analysis

## Integration Points

### Input Sources
- **Corpus Manager**: Automatic deduplication during bulk ingestion
- **Collection Import**: Duplicate detection during collection imports
- **Manual Analysis**: On-demand deduplication of existing collections
- **Maintenance Workflows**: Regular cleanup and optimization

### Output Destinations
- **Database Updates**: Remove or merge duplicate documents
- **Analytics Reports**: Duplicate statistics and trends
- **Quality Metrics**: Collection cleanliness and integrity scores
- **User Notifications**: Duplicate detection alerts and recommendations

### Component Dependencies
- **Vector Database**: Document and chunk storage for analysis
- **Embedding Service**: Semantic similarity computation
- **Corpus Analytics**: Integration with quality assessment
- **CLI Interface**: User-facing deduplication commands

## Configuration Options

### Detection Thresholds
```python
deduplication_config = {
    'fuzzy_threshold': 0.8,     # Jaccard similarity threshold
    'semantic_threshold': 0.95,  # Embedding similarity threshold  
    'metadata_threshold': 0.9,   # File name similarity threshold
    'minhash_permutations': 128, # MinHash accuracy parameter
    'lsh_bands': 16             # LSH performance parameter
}
```

### Processing Options
```python
processing_config = {
    'batch_size': 1000,         # Documents processed per batch
    'parallel_methods': True,   # Run detection methods in parallel
    'early_termination': True,  # Skip expensive methods when duplicates found
    'progress_reporting': True  # Show progress bars
}
```

## Usage Examples

### Basic Duplicate Analysis
```python
# Initialize deduplicator
deduplicator = DocumentDeduplicator(db_path="data/vectors.db")

# Analyze collection for duplicates
report = deduplicator.analyze_duplicates("research_papers")

print(f"Total documents: {report.total_documents}")
print(f"Unique documents: {report.unique_documents}")
print(f"Duplicate groups: {len(report.duplicate_groups)}")
print(f"Space savings: {report.space_saved_mb:.2f}MB")

# Review duplicate groups
for group in report.duplicate_groups:
    print(f"Group similarity: {group.similarity_score:.3f}")
    print(f"Detection method: {group.detection_method}")
    print(f"Documents: {group.documents}")
```

### Automated Duplicate Resolution
```python
# Detect and resolve duplicates automatically
report = deduplicator.analyze_duplicates("my_collection")

if report.duplicate_groups:
    resolution_stats = deduplicator.resolve_duplicates(
        duplicate_groups=report.duplicate_groups,
        collection_id="my_collection",
        strategy="keep_first"
    )
    
    print(f"Documents removed: {resolution_stats['documents_removed']}")
    print(f"Space saved: {resolution_stats['space_saved_mb']:.2f}MB")
```

### Custom Threshold Analysis
```python
# Analyze with custom thresholds
custom_deduplicator = DocumentDeduplicator(
    db_path="data/vectors.db",
    fuzzy_threshold=0.9,      # Higher threshold = stricter detection
    semantic_threshold=0.98,  # Very high threshold for semantic duplicates
    minhash_perms=256        # More permutations = better accuracy
)

report = custom_deduplicator.analyze_duplicates("sensitive_docs")
```

## Best Practices

### Threshold Selection
- **Conservative**: High thresholds (0.9+) to avoid false positives
- **Aggressive**: Lower thresholds (0.7-0.8) for comprehensive cleanup
- **Domain-Specific**: Adjust based on content characteristics
- **Iterative Tuning**: Start conservative, then adjust based on results

### Workflow Integration
- **Pre-Ingestion**: Check for duplicates before adding to collection
- **Post-Ingestion**: Regular deduplication maintenance
- **Import Validation**: Scan imports for duplicates against existing corpus
- **Quality Assurance**: Include deduplication in quality assessment

### Performance Optimization
- **Method Selection**: Choose appropriate methods based on collection size
- **Batch Processing**: Process large collections in batches
- **Resource Management**: Monitor memory usage during analysis
- **Incremental Updates**: Only analyze new documents when possible
# Corpus Organizer

## Overview
The Corpus Organizer (`src/corpus_organizer.py`) provides sophisticated collection management capabilities for organizing documents into logical groups with metadata, tagging, and cross-collection operations.

## Core Classes

### Collection
**Purpose**: Data structure representing a document collection with metadata
**Key Attributes**:
- `collection_id`: Unique identifier for the collection
- `name`: Human-readable collection name  
- `description`: Detailed collection description
- `created_at`: Collection creation timestamp
- `updated_at`: Last modification timestamp
- `tags`: List of descriptive tags for categorization
- `metadata`: Flexible key-value metadata storage
- `document_count`: Number of documents in collection
- `chunk_count`: Total chunks across all documents
- `size_mb`: Total collection size in megabytes

**Features**:
- **Automatic Statistics**: Document and chunk counts updated automatically
- **Rich Metadata**: Flexible metadata storage for custom attributes
- **Tag Support**: Multiple tags for categorization and filtering
- **Size Tracking**: Automatic size calculation and monitoring

### CorpusOrganizer
**Purpose**: Main service for collection management and organization
**Key Features**:
- **Collection Lifecycle**: Create, update, delete, and organize collections
- **Document Management**: Move documents between collections
- **Statistics Tracking**: Automatic collection statistics maintenance  
- **Import/Export**: Backup and restore collection data
- **Merge Operations**: Combine multiple collections efficiently
- **Search Integration**: Collection-scoped search and retrieval

## Key Methods

### create_collection(name: str, description: str = "", collection_id: str = None, ...) -> str
**Purpose**: Create a new document collection with metadata
**Features**:
- **Automatic ID Generation**: Creates unique collection IDs if not provided
- **Metadata Storage**: Rich metadata and tagging support
- **Database Integration**: Automatic schema updates for collection support
- **Validation**: Ensures collection names and IDs are unique

**Inputs**:
- `name`: Human-readable collection name
- `description`: Optional detailed description
- `collection_id`: Custom ID (auto-generated if None)
- `tags`: List of tags for categorization
- `metadata`: Custom metadata dictionary

**Outputs**: Generated or provided collection ID

**Database Changes**:
- Creates collection record in collections table
- Updates database schema if needed for collection support
- Initializes collection statistics

### list_collections() -> List[Collection]
**Purpose**: Retrieve all collections with current statistics
**Features**:
- **Fresh Statistics**: Updates collection stats before returning
- **Rich Information**: Full metadata, tags, and statistics
- **Sorted Results**: Ordered by creation date or custom criteria
- **Performance Optimized**: Efficient queries with minimal overhead

**Returns**: List of Collection objects with complete metadata

### switch_collection(collection_id: str) -> bool
**Purpose**: Set default collection for subsequent operations
**Features**:
- **Global State**: Updates system default collection
- **Validation**: Ensures target collection exists
- **Context Awareness**: All subsequent operations use new collection
- **Session Persistence**: Settings maintained across operations

### delete_collection(collection_id: str, confirm: bool = False) -> bool
**Purpose**: Remove collection and optionally all its documents
**Features**:
- **Safety Checks**: Requires explicit confirmation for data deletion
- **Cascade Options**: Delete documents or move to default collection
- **Referential Integrity**: Maintains database consistency
- **Audit Trail**: Logs deletion operations for accountability

**Safety Parameters**:
- `confirm`: Must be True to actually perform deletion
- **Interactive Mode**: Prompts for confirmation in CLI
- **Dry Run Support**: Preview deletion effects

### merge_collections(source_ids: List[str], target_id: str, delete_sources: bool = False) -> Dict
**Purpose**: Combine multiple collections into a single collection
**Features**:
- **Efficient Transfer**: Bulk document movement with minimal overhead
- **Metadata Preservation**: Maintains document metadata and relationships
- **Statistics Update**: Recalculates collection statistics
- **Optional Cleanup**: Can delete source collections after merge

**Algorithm**:
1. Validate all source and target collections exist
2. Update document collection_id for all source documents
3. Update chunk and embedding collection references
4. Recalculate target collection statistics
5. Optionally delete empty source collections
6. Return detailed merge statistics

### export_collection(collection_id: str, export_path: str, include_embeddings: bool = False) -> Dict
**Purpose**: Export collection data for backup or transfer
**Features**:
- **Complete Export**: Documents, chunks, metadata, and optionally embeddings
- **Compression**: Efficient storage format for large collections
- **Selective Export**: Option to exclude large embedding data
- **Portability**: Self-contained export files for easy transfer

**Export Format**:
```json
{
    "collection_metadata": {...},
    "documents": [...],
    "chunks": [...],
    "embeddings": [...],  // Optional
    "export_info": {
        "timestamp": "2025-08-27T...",
        "document_count": 1500,
        "chunk_count": 12000,
        "file_size_mb": 145.7
    }
}
```

### import_collection(import_path: str, collection_id: str = None) -> Dict
**Purpose**: Import collection data from export file  
**Features**:
- **Validation**: Ensures data integrity before import
- **Conflict Resolution**: Handles ID conflicts with existing data
- **Progress Tracking**: Shows import progress for large collections
- **Statistics Update**: Recalculates all collection statistics

## Collection Management Features

### Document Organization
**Multi-Collection Support**: Documents can be organized into logical collections
- **Research Papers**: Academic publications and papers
- **Technical Docs**: API documentation and technical guides
- **General Knowledge**: Wikipedia articles and general information
- **Domain-Specific**: Legal documents, medical literature, etc.

### Metadata Management
**Rich Metadata Support**:
```python
collection_metadata = {
    'domain': 'machine_learning',
    'language': 'en',
    'quality_level': 'high',
    'source_type': 'academic',
    'update_frequency': 'monthly',
    'owner': 'research_team',
    'access_level': 'public'
}
```

### Tagging System
**Flexible Tagging**:
- **Categorical Tags**: Domain, quality, source type
- **Temporal Tags**: Date ranges, update frequencies  
- **Access Tags**: Public, private, restricted
- **Custom Tags**: Domain-specific categorization

**Tag Operations**:
- Add/remove tags from collections
- Search collections by tags
- Tag-based filtering in retrieval
- Tag hierarchy and relationships

### Collection Statistics
**Automatic Statistics Tracking**:
```python
{
    'collection_id': 'ml_papers',
    'document_count': 1500,
    'chunk_count': 12500,
    'size_mb': 234.7,
    'avg_doc_size_kb': 156.8,
    'avg_chunks_per_doc': 8.3,
    'file_type_distribution': {
        '.pdf': 1200,
        '.txt': 200, 
        '.md': 100
    },
    'ingestion_timeline': {
        'first_document': '2025-08-01T...',
        'last_document': '2025-08-27T...',
        'most_recent_batch': '2025-08-27T...'
    }
}
```

## Advanced Operations

### Collection Merging
**Bulk Operations**: Efficiently combine related collections
- **Research Consolidation**: Merge related research collections
- **Quality Upgrades**: Move high-quality documents to premium collections
- **Temporal Organization**: Merge collections by time periods
- **Domain Restructuring**: Reorganize by subject matter

**Performance**: Optimized for large collections (10k+ documents)

### Cross-Collection Search
**Unified Search**: Search across multiple collections simultaneously
```python
# Search multiple collections
results = organizer.search_across_collections(
    query="machine learning algorithms",
    collection_ids=['ai_papers', 'ml_tutorials', 'research_notes'],
    method='hybrid',
    k=10
)
```

### Collection Analytics
**Insights and Reporting**:
- **Growth Analysis**: Track collection growth over time
- **Quality Metrics**: Document quality distribution
- **Usage Analytics**: Query frequency and patterns
- **Similarity Analysis**: Inter-collection content similarity

## Database Integration

### Schema Extensions
**Collection Support**: Extends existing schema with collection capabilities
```sql
-- Collections table
CREATE TABLE collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tags_json TEXT,
    metadata_json TEXT,
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    size_mb REAL DEFAULT 0.0
);

-- Add collection_id to existing tables
ALTER TABLE documents ADD COLUMN collection_id TEXT DEFAULT 'default';
ALTER TABLE chunks ADD COLUMN collection_id TEXT DEFAULT 'default';
ALTER TABLE embeddings ADD COLUMN collection_id TEXT DEFAULT 'default';
```

### Performance Optimization
**Efficient Queries**: 
- **Indexed Access**: collection_id columns properly indexed
- **JOIN Optimization**: Efficient collection-filtered queries
- **Statistics Caching**: Pre-computed collection statistics
- **Query Planning**: Optimized query execution paths

## Integration Points

### Input Sources  
- **Corpus Manager**: Automatic collection assignment during bulk ingestion
- **Individual Ingestion**: Manual collection assignment for specific documents
- **Import Operations**: Collection restoration from backups
- **Migration Tools**: Data movement from external systems

### Output Destinations
- **Search Systems**: Collection-scoped search and retrieval
- **Analytics**: Collection-based reporting and insights
- **Export Systems**: Collection backup and transfer
- **User Interfaces**: Collection browsing and management

### Component Dependencies
- **Vector Database**: Core storage backend with collection extensions
- **Corpus Analytics**: Collection-based statistics and insights
- **Corpus Manager**: Bulk processing with collection support
- **CLI Interface**: User-facing collection management commands

## Usage Examples

### Collection Management Workflow
```python
# Initialize organizer
organizer = CorpusOrganizer(db_path="data/vectors.db")

# Create research collection
collection_id = organizer.create_collection(
    name="AI Research Papers",
    description="Academic papers on artificial intelligence",
    tags=["academic", "ai", "research"],
    metadata={"domain": "artificial_intelligence", "quality": "high"}
)

# List all collections
collections = organizer.list_collections()
for collection in collections:
    print(f"{collection.name}: {collection.document_count} docs")

# Switch default collection
organizer.switch_collection("ai_research_papers")

# Export collection
export_stats = organizer.export_collection(
    collection_id="ai_research_papers",
    export_path="backups/ai_papers_backup.json",
    include_embeddings=True
)
```

### Advanced Operations
```python
# Merge related collections  
merge_stats = organizer.merge_collections(
    source_ids=["ml_papers", "dl_papers", "nlp_papers"],
    target_id="ai_research_unified",
    delete_sources=True
)

# Collection analytics
analytics = organizer.analyze_collection("ai_research_unified")
print(f"Quality Score: {analytics['quality_score']:.2f}")
print(f"Diversity Index: {analytics['diversity_index']:.2f}")

# Cross-collection search
results = organizer.search_collections(
    query="transformer architecture",
    collection_ids=["ai_papers", "tutorials"],
    method="hybrid"
)
```

## Configuration Guidelines

### Collection Naming Strategy
- **Descriptive Names**: Clear, human-readable collection names
- **Consistent Conventions**: Follow naming patterns across collections
- **Hierarchical Organization**: Use prefixes for related collections
- **Version Management**: Include version information when relevant

### Performance Tuning
- **Collection Size**: Optimize for 1k-10k documents per collection
- **Metadata Efficiency**: Keep metadata concise but informative
- **Tag Management**: Use consistent tag vocabularies
- **Regular Maintenance**: Periodic statistics updates and cleanup

### Security and Access Control
- **Collection Isolation**: Logical separation of sensitive documents
- **Metadata Protection**: Careful handling of sensitive metadata
- **Access Logging**: Track collection access and modifications
- **Backup Strategy**: Regular export of critical collections
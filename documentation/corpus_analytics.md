# Corpus Analytics

## Overview
The Corpus Analytics (`src/corpus_analytics.py`) provides comprehensive analysis, reporting, and quality assessment capabilities for document collections, offering insights into corpus characteristics, quality, and optimization opportunities.

## Core Classes

### CorpusAnalyzer
**Purpose**: Main analytics engine for corpus statistics, quality assessment, and insights generation
**Key Features**:
- **Comprehensive Statistics**: Document counts, size distributions, content analysis
- **Quality Assessment**: Automated quality scoring with recommendations
- **Similarity Analysis**: Document relationships and clustering insights
- **Growth Tracking**: Temporal analysis of corpus evolution
- **Export Capabilities**: Detailed reports in multiple formats

### CorpusStats  
**Purpose**: Statistical summary of corpus characteristics
**Key Metrics**:
- `total_documents`: Complete document count
- `total_chunks`: Sum of all document chunks
- `total_tokens`: Estimated token count across corpus
- `size_mb`: Total corpus size in megabytes
- `avg_document_size`: Average document size in bytes
- `avg_chunks_per_doc`: Average chunks per document
- `file_types`: Distribution of file formats
- `ingestion_timeline`: Temporal ingestion patterns
- `most_similar_pairs`: Top similar document pairs

### DocumentInsights
**Purpose**: Detailed analysis of individual document characteristics
**Key Components**:
- `quality_score`: Overall document quality rating (0.0-1.0)
- `content_metrics`: Length, complexity, and structure analysis
- `embedding_quality`: Embedding coverage and distribution
- `similarity_profile`: Relationships with other documents
- `recommendations`: Specific improvement suggestions

## Analytics Capabilities

### Basic Statistics

#### analyze_collection(collection_id: str = "default") -> CorpusStats
**Purpose**: Generate comprehensive statistical overview of a collection
**Features**:
- **Document Metrics**: Count, size, type distribution
- **Content Analysis**: Token counts, length distributions  
- **Temporal Patterns**: Ingestion timeline and growth analysis
- **File Format Analysis**: Format distribution and characteristics
- **Performance Metrics**: Database size and query performance indicators

**Statistical Categories**:
```python
{
    'basic_metrics': {
        'total_documents': 1500,
        'total_chunks': 12500,
        'total_tokens': 2500000,
        'size_mb': 245.7,
        'avg_document_size': 163840,
        'avg_chunks_per_doc': 8.3
    },
    'file_type_distribution': {
        '.pdf': 65.3,    # Percentage
        '.txt': 28.1,    
        '.md': 4.2,
        '.html': 2.4
    },
    'size_distribution': {
        'small_docs': 15,     # < 1KB
        'medium_docs': 70,    # 1KB-100KB  
        'large_docs': 15      # > 100KB
    },
    'quality_metrics': {
        'avg_quality_score': 0.82,
        'high_quality_docs': 78,    # Percentage with score > 0.8
        'low_quality_docs': 8       # Percentage with score < 0.5
    }
}
```

### Quality Assessment

#### generate_quality_report(collection_id: str = "default") -> Dict[str, Any]
**Purpose**: Automated quality assessment with scoring and recommendations
**Quality Dimensions**:
- **Completeness**: Document content completeness and structure
- **Consistency**: Chunk size consistency and format uniformity
- **Coverage**: Embedding coverage and quality
- **Relevance**: Content relevance to collection purpose
- **Freshness**: Document recency and update frequency

**Scoring Algorithm**:
```python
quality_scores = {
    'completeness': completeness_score,      # 0.0-1.0
    'chunk_consistency': consistency_score,  # 0.0-1.0  
    'embedding_coverage': coverage_score,    # 0.0-1.0
    'content_relevance': relevance_score,    # 0.0-1.0
    'format_consistency': format_score       # 0.0-1.0
}

overall_score = sum(quality_scores.values()) / len(quality_scores)
```

**Quality Rating Categories**:
- **Excellent** (0.9-1.0): High-quality, well-structured corpus
- **Good** (0.7-0.9): Good quality with minor issues
- **Fair** (0.5-0.7): Acceptable quality with improvement opportunities  
- **Poor** (0.3-0.5): Significant quality issues requiring attention
- **Critical** (0.0-0.3): Major problems requiring immediate action

**Recommendations Engine**:
```python
recommendations = [
    "Consider increasing chunk overlap to improve context preservation",
    "15% of documents have missing embeddings - run re-embedding",
    "Large variation in chunk sizes detected - consider re-chunking",
    "Add more descriptive metadata to improve searchability"
]
```

### Similarity Analysis

#### find_similar_documents(doc_id: str, k: int = 10, threshold: float = 0.7) -> List[Dict]
**Purpose**: Find documents most similar to a given document
**Features**:
- **Document-Level Similarity**: Aggregated chunk similarity scoring
- **Configurable Thresholds**: Filter results by minimum similarity
- **Ranking**: Results ranked by similarity score
- **Context Information**: Include snippet and metadata

**Algorithm**:
1. **Chunk Aggregation**: Combine chunk embeddings into document embedding
2. **Similarity Computation**: Calculate cosine similarity with all documents
3. **Threshold Filtering**: Filter results by minimum similarity
4. **Ranking and Selection**: Return top-k most similar documents

#### analyze_document_clusters(collection_id: str, n_clusters: int = 10) -> Dict[str, Any]
**Purpose**: Identify natural document groups and topics within corpus
**Features**:
- **Clustering**: K-means clustering on document embeddings
- **Topic Detection**: Identify main themes and subjects
- **Cluster Characterization**: Representative documents and keywords per cluster
- **Visualization Support**: Data prepared for visualization tools

**Output**:
```python
{
    'clusters': [
        {
            'cluster_id': 0,
            'size': 150,
            'center_document': 'doc_12345',
            'representative_keywords': ['machine learning', 'neural networks'],
            'documents': ['doc_1', 'doc_2', ...],
            'coherence_score': 0.87
        }
    ],
    'silhouette_score': 0.75,    # Overall clustering quality
    'inertia': 1250.5,           # Cluster compactness metric
    'optimal_clusters': 8        # Suggested number of clusters
}
```

### Growth and Trend Analysis

#### analyze_growth_patterns(collection_id: str, period: str = "monthly") -> Dict[str, Any]
**Purpose**: Track corpus growth and ingestion patterns over time
**Features**:
- **Temporal Grouping**: Daily, weekly, monthly, or yearly analysis
- **Growth Metrics**: Document addition rates and size growth
- **Pattern Detection**: Identify ingestion spikes and trends
- **Forecasting**: Predict future growth based on historical patterns

**Growth Metrics**:
```python
{
    'growth_timeline': [
        {
            'period': '2025-08',
            'documents_added': 500,
            'size_added_mb': 75.3,
            'growth_rate': 12.5      # Percentage increase
        }
    ],
    'overall_trends': {
        'avg_monthly_growth': 8.7,   # Percentage
        'peak_ingestion_month': '2025-07',
        'projected_size_next_month': 278.4,  # MB
        'growth_acceleration': -0.02         # Growth rate change
    }
}
```

## Reporting and Export

### export_analytics_report(collection_id: str, output_path: str = None) -> Dict[str, Any]
**Purpose**: Generate comprehensive analytics report for collection
**Report Sections**:
- **Executive Summary**: High-level metrics and key insights
- **Statistical Analysis**: Detailed corpus statistics and distributions
- **Quality Assessment**: Quality scores, issues, and recommendations
- **Similarity Analysis**: Document relationships and clustering
- **Growth Analysis**: Historical trends and projections
- **Performance Metrics**: Search and retrieval performance data

**Export Formats**:
- **JSON**: Machine-readable structured data
- **CSV**: Tabular data for spreadsheet analysis
- **HTML**: Human-readable report with visualizations
- **Markdown**: Documentation-friendly format

### compare_collections(collection_ids: List[str]) -> Dict[str, Any]  
**Purpose**: Comparative analysis across multiple collections
**Features**:
- **Side-by-side Comparison**: Metrics compared across collections
- **Overlap Analysis**: Identify shared documents between collections
- **Quality Comparison**: Relative quality assessment
- **Recommendation Engine**: Suggest collection consolidation or reorganization

## Performance Characteristics

### Analysis Performance
- **Basic Statistics**: <1 second for 10k documents
- **Quality Assessment**: 5-30 seconds for 10k documents
- **Similarity Analysis**: 30-180 seconds for 10k documents
- **Clustering**: 60-300 seconds for 10k documents (depends on dimensions)

### Memory Requirements
- **Statistics**: <100MB for 100k documents
- **Quality Assessment**: 200-500MB for 100k documents  
- **Similarity Analysis**: 500MB-2GB for 100k documents
- **Clustering**: 1-4GB for 100k documents

### Scalability Limits
- **Document Count**: Tested up to 100k documents
- **Embedding Dimensions**: Supports 128-2048 dimensional embeddings
- **Collection Size**: No practical limit on collection size
- **Concurrent Analysis**: Multiple collections can be analyzed simultaneously

## Integration Points

### Input Sources
- **Vector Database**: Primary data source for all analytics
- **Collection Metadata**: Collection-specific analysis parameters
- **User Parameters**: Custom analysis configuration
- **Historical Data**: Temporal analysis using ingestion timestamps

### Output Destinations  
- **CLI Reports**: Rich terminal output with tables and charts
- **File Exports**: JSON, CSV, HTML report generation
- **Dashboard Integration**: Data prepared for web dashboard display
- **Monitoring Systems**: Metrics exported for monitoring tools

### Component Integration
- **Corpus Manager**: Quality assessment during ingestion
- **Corpus Organizer**: Collection-based analytics and reporting
- **Deduplication**: Quality impact of duplicate removal
- **RAG Pipeline**: Performance metrics for search optimization

## Usage Examples

### Basic Analytics Workflow
```python
# Initialize analyzer
analyzer = CorpusAnalyzer(db_path="data/vectors.db")

# Get basic statistics
stats = analyzer.analyze_collection("research_papers")
print(f"Documents: {stats.total_documents}")
print(f"Average quality: {stats.avg_quality_score:.2f}")

# Generate quality report
quality_report = analyzer.generate_quality_report("research_papers")
print(f"Overall quality: {quality_report['quality_rating']}")
for rec in quality_report['recommendations']:
    print(f"- {rec}")
```

### Advanced Analysis
```python
# Comprehensive analysis
analyzer = CorpusAnalyzer("data/vectors.db")

# Document similarity analysis
similar_docs = analyzer.find_similar_documents(
    doc_id="paper_12345",
    k=5,
    threshold=0.8
)

# Cluster analysis
clusters = analyzer.analyze_document_clusters(
    collection_id="research",
    n_clusters=8
)

# Export complete report
report = analyzer.export_analytics_report(
    collection_id="research", 
    output_path="reports/research_analysis.json"
)
```

### Multi-Collection Analysis
```python
# Compare collections
comparison = analyzer.compare_collections([
    "academic_papers",
    "technical_docs", 
    "general_knowledge"
])

print("Collection Comparison:")
for collection_id in comparison['collections']:
    metrics = comparison['metrics'][collection_id]
    print(f"{collection_id}: {metrics['document_count']} docs, "
          f"quality {metrics['avg_quality']:.2f}")
```

## Configuration Options

### Analysis Configuration
```python
analytics_config = {
    'quality_assessment': {
        'enable_deep_analysis': True,
        'quality_thresholds': {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5
        }
    },
    'similarity_analysis': {
        'similarity_threshold': 0.7,
        'max_pairs_analyzed': 10000,
        'clustering_algorithm': 'kmeans'
    },
    'reporting': {
        'include_recommendations': True,
        'export_format': 'json',
        'visualization_data': True
    }
}
```

### Performance Configuration
```python
performance_config = {
    'batch_size': 1000,           # Documents per analysis batch
    'parallel_processing': True,   # Use multiple cores when possible
    'memory_limit_mb': 4000,      # Memory usage limit
    'cache_embeddings': True      # Cache embeddings for repeated analysis
}
```

## Best Practices

### Regular Analytics Workflows
- **Daily**: Basic statistics monitoring
- **Weekly**: Quality assessment and trend analysis
- **Monthly**: Comprehensive analytics with clustering and similarity analysis
- **Quarterly**: Multi-collection comparison and optimization planning

### Quality Monitoring
- **Automated Alerts**: Set up alerts for quality score drops
- **Trend Tracking**: Monitor quality trends over time
- **Proactive Maintenance**: Use quality reports to guide maintenance decisions
- **Continuous Improvement**: Regular quality assessment and optimization
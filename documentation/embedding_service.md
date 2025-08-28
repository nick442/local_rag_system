# Embedding Service

## Overview
The Embedding Service (`src/embedding_service.py`) provides high-performance text embedding generation using sentence-transformers models with GPU acceleration and intelligent caching.

## Core Classes

### EmbeddingService
**Purpose**: Main service for generating text embeddings with optimization features
**Key Features**:
- **GPU Acceleration**: MPS (Metal Performance Shaders) support on Apple Silicon
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Memory Management**: Automatic cache management and cleanup
- **Model Loading**: Support for local and HuggingFace models
- **Error Handling**: Graceful degradation and retry mechanisms

**Configuration Parameters**:
- `model_path`: Path to sentence-transformers model (local or HuggingFace)
- `batch_size`: Number of texts to process simultaneously (default: 32)
- `device`: Computing device ('mps', 'cuda', 'cpu') - auto-detected
- `max_length`: Maximum sequence length for tokenization

**Initialization**:
```python
service = EmbeddingService(
    model_path="models/embeddings/all-MiniLM-L6-v2",
    batch_size=32
)
```

### EmbeddingBatch
**Purpose**: Data structure for batch embedding operations
**Key Attributes**:
- `texts`: List of text strings to embed
- `batch_id`: Unique identifier for tracking
- `metadata`: Additional processing information
- `embeddings`: Generated embeddings (populated after processing)

**Usage**: Internal data structure for batch processing optimization

## Key Methods

### embed_texts(texts: List[str]) -> np.ndarray
**Purpose**: Generate embeddings for a list of texts
**Features**:
- Automatic batching for memory efficiency
- Progress tracking for large batches
- Tensor optimization and device management
- Error handling with partial results

**Inputs**: List of text strings
**Outputs**: NumPy array of embeddings (shape: [n_texts, embedding_dim])

**Performance**:
- Batch size 32: ~100-200 texts/second on Apple Silicon M1/M2
- Memory usage: ~1-2GB for model + batch data
- Embedding dimension: 384 (all-MiniLM-L6-v2)

### embed_single(text: str) -> np.ndarray  
**Purpose**: Generate embedding for single text (convenience method)
**Use Case**: One-off embeddings or real-time query processing
**Performance**: ~10-50 texts/second (less efficient than batching)

### get_model_info() -> Dict[str, Any]
**Purpose**: Return detailed model information
**Returns**:
```python
{
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_dimension': 384,
    'max_sequence_length': 256,
    'device': 'mps',
    'batch_size': 32,
    'model_size_mb': 90.9
}
```

### clear_cache()
**Purpose**: Free GPU memory and clear internal caches
**Use Case**: Memory management between large processing jobs

## Performance Characteristics

### Hardware Optimization
- **Apple Silicon (M1/M2)**: MPS acceleration, ~100-200 texts/second
- **NVIDIA GPU**: CUDA acceleration, ~200-500 texts/second  
- **CPU Fallback**: 10-50 texts/second depending on hardware

### Memory Management
- **Model Loading**: ~1GB GPU/system memory for all-MiniLM-L6-v2
- **Batch Processing**: ~10-50MB per batch of 32 texts
- **Automatic Cleanup**: Cache clearing on service destruction
- **Memory Monitoring**: Built-in memory usage tracking

### Throughput Benchmarks
Based on all-MiniLM-L6-v2 model:
- **Single text**: 10-50/sec (varies by device)
- **Batch size 16**: 80-150/sec
- **Batch size 32**: 100-200/sec (optimal)
- **Batch size 64**: 90-180/sec (may hit memory limits)

## Integration Points

### Input Sources
- **RAG Pipeline**: Query embeddings for similarity search
- **Document Ingestion**: Chunk embeddings during corpus building
- **Corpus Manager**: Batch embedding generation during bulk ingestion
- **Analytics**: Similarity analysis and clustering operations

### Output Destinations
- **Vector Database**: Embeddings stored for similarity search
- **Analytics Tools**: Embedding-based similarity metrics
- **Search Systems**: Query-document similarity computation

### Dependencies
- **sentence-transformers**: Core embedding model framework
- **torch**: PyTorch framework for neural network operations
- **numpy**: Efficient array operations and data handling
- **tqdm**: Progress tracking for batch operations

## Model Support

### Recommended Models
- **all-MiniLM-L6-v2**: Fast, 384-dim, good general performance (default)
- **all-mpnet-base-v2**: Slower, 768-dim, better quality
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for Q&A tasks

### Model Selection Criteria
- **Speed vs Quality**: MiniLM for speed, MPNet for quality
- **Embedding Dimension**: 384 (fast) vs 768 (quality)
- **Domain Specialization**: General vs Q&A vs domain-specific models
- **Memory Requirements**: Larger models need more GPU memory

## Error Handling

### Common Issues
- **Out of Memory**: Automatic batch size reduction
- **Model Loading Failures**: Fallback to CPU or smaller model
- **Device Unavailable**: Automatic device detection and fallback
- **Corrupted Models**: Clear error messages and recovery suggestions

### Recovery Mechanisms
- **Graceful Degradation**: CPU fallback when GPU unavailable
- **Partial Processing**: Return embeddings for successful texts
- **Memory Recovery**: Automatic cache clearing on errors
- **Retry Logic**: Exponential backoff for transient failures

## Usage Examples

```python
# Initialize service
embedding_service = EmbeddingService(
    model_path="models/embeddings/all-MiniLM-L6-v2",
    batch_size=32
)

# Generate embeddings for document chunks
texts = ["This is document chunk 1", "This is document chunk 2"]
embeddings = embedding_service.embed_texts(texts)
# Returns: numpy array shape (2, 384)

# Single text embedding
query_embedding = embedding_service.embed_single("What is machine learning?")
# Returns: numpy array shape (384,)

# Get model information
info = embedding_service.get_model_info()
print(f"Model: {info['model_name']}, Dimension: {info['embedding_dimension']}")

# Cleanup when done
embedding_service.clear_cache()
```

## Configuration Guidelines

### Batch Size Selection
- **Small datasets** (< 1k texts): batch_size=16
- **Medium datasets** (1k-10k texts): batch_size=32 (default)
- **Large datasets** (> 10k texts): batch_size=64 (if memory allows)
- **Memory constrained**: batch_size=8-16

### Device Selection
- **Automatic Detection**: Service auto-detects best available device
- **Manual Override**: Set device parameter for specific requirements
- **Performance Order**: MPS > CUDA > CPU
- **Memory Considerations**: GPU devices require sufficient VRAM
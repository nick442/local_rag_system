# Document Ingestion Service

## Overview
The Document Ingestion Service (`src/document_ingestion.py`) is responsible for loading, parsing, and chunking documents from various file formats into a standardized format suitable for the RAG pipeline.

## Core Classes

### Document
**Purpose**: Represents a parsed document with metadata
**Key Attributes**:
- `content`: Raw text content of the document
- `metadata`: Dict with file information (absolute source path, filename, file_type, size, timestamp)
- `doc_id`: Deterministic ID derived from content hash

**Inputs**: File content and metadata
**Outputs**: Structured document object with normalized text content

### DocumentChunk  
**Purpose**: Represents a text chunk with token counts and metadata
**Key Attributes**:
- `content`: Chunk text content
- `doc_id`: Parent document ID
- `chunk_index`: Position within document
- `metadata`: Inherited from document plus chunk info (token ranges)
- `token_count`: Number of tokens for this chunk
- `chunk_id`: Stable chunk identifier (`{doc_id}_chunk_{index}`)

**Inputs**: Text segment and position information
**Outputs**: Structured chunk ready for embedding and storage

### DocumentLoader (Abstract Base)
**Purpose**: Abstract interface for format-specific document loaders
**Key Methods**:
- `can_load(file_path)`: Checks if loader supports the file format
- `load(file_path)`: Loads and parses the document

**Inputs**: File path
**Outputs**: Document object or raises exception

### Format-Specific Loaders

#### TextLoader
**Purpose**: Handles plain text files (.txt)
**Features**: 
- UTF-8 encoding detection and handling
- Basic metadata extraction (file stats)
- Direct content loading without parsing

**Supported Formats**: .txt files
**Performance**: Fastest loader, minimal processing overhead

#### PDFLoader  
**Purpose**: Extracts text from PDF documents (.pdf)
**Features**:
- Multi-page text extraction
- Layout preservation where possible
- Metadata extraction (title, author, creation date)
- Handles password-protected PDFs (with blank password)

**Dependencies**: PyPDF2
**Supported Formats**: .pdf files
**Performance**: Moderate, depends on PDF complexity

#### HTMLLoader
**Purpose**: Parses HTML documents and web pages (.html, .htm)
**Features**:
- Clean text extraction from HTML tags
- Metadata extraction from HTML meta tags
- Link and image reference handling
- Malformed HTML tolerance

**Dependencies**: BeautifulSoup4
**Supported Formats**: .html, .htm files
**Performance**: Fast, efficient HTML parsing

#### MarkdownLoader
**Purpose**: Processes Markdown documents (.md)
**Features**:
- Markdown syntax parsing and conversion
- Preserves structure information
- Code block extraction and handling
- Frontmatter metadata parsing

**Dependencies**: Built-in markdown processing
**Supported Formats**: .md files  
**Performance**: Fast, lightweight processing

### DocumentChunker
**Purpose**: Splits documents into token‑bounded chunks for embedding and retrieval
**Key Features**:
- **Token-Aware Splitting**: Uses `tiktoken` encoding (default `cl100k_base`)
- **Sliding Window**: Overlapping chunks for context preservation
- **Metadata Inheritance**: Preserves document metadata in chunks

**Configuration** (current defaults):
- `chunk_size`: 512 tokens
- `chunk_overlap`: 128 tokens

**Inputs**: Document object
**Outputs**: List of DocumentChunk objects

**Algorithm**:
1. Split text by sentences using multiple delimiters
2. Combine sentences up to chunk_size limit
3. Add overlap from previous chunk for context
4. Ensure chunks don't exceed max_chunk_size

### DocumentIngestionService
**Purpose**: Main service orchestrating single‑file ingestion and chunking
**Key Features**:
- **Multi-format Support**: Automatic format detection and loader selection
- **Error Handling**: Graceful handling of corrupted or unsupported files
- **Batch Processing**: Efficient processing of multiple documents
- **Metadata Enrichment**: Adds processing timestamps and system metadata

**Processing Pipeline**:
1. **File Detection**: Determines file type and selects appropriate loader
2. **Document Loading**: Loads and parses file content using format-specific loader
3. **Content Validation**: Ensures document has sufficient content for processing
4. **Chunking**: Splits document into overlapping chunks for optimal retrieval
5. **Metadata Enhancement**: Adds ingestion timestamps and system information

**Key Methods**:
- `ingest_document(file_path) -> List[DocumentChunk]`: Process a single file and return chunks
- `get_supported_extensions() -> List[str]`: Returns supported file types (`.txt, .pdf, .html, .htm, .md`)

Note: Bulk directory ingestion is handled by the `CorpusManager` (`src/corpus_manager.py`).

**Error Handling**:
- Logs detailed error information for debugging
- Continues processing other files when individual files fail
- Returns processing statistics including success/failure counts

**Performance Characteristics**:
- Memory efficient: Processes one document at a time
- Scalable: Handles directories with thousands of files
- Fault tolerant: Recovers from individual file processing errors

## Integration Points

### Input Sources
- **File System**: Local files in supported formats
- **Directory Scanning**: Recursive directory traversal with glob patterns
- **Batch Processing**: Multiple files processed sequentially or in parallel

### Output Destinations  
- **Vector Database**: Chunks stored with metadata for retrieval
- **Embedding Service**: Text chunks sent for vector embedding generation
- **Analytics**: Processing statistics and quality metrics

### Dependencies
- **External**: PyPDF2 (PDF), BeautifulSoup4 (HTML), html2text, markdown, tiktoken
- **Internal**: VectorDatabase (downstream storage), EmbeddingService (used by ingestion pipeline via `CorpusManager`)

## Usage Examples

```python
# Initialize service
ingestion = DocumentIngestionService()

# Process single file
document = ingestion.ingest_file("path/to/document.pdf")

# Process directory
documents = ingestion.ingest_directory("corpus/", "**/*.{txt,pdf,md}")

# Get supported formats
formats = ingestion.get_supported_extensions()
# Returns: ['.txt', '.pdf', '.html', '.htm', '.md']
```

## Performance Metrics
- **Text files**: ~1000 documents/second
- **PDF files**: ~10-50 documents/second (varies by complexity)
- **HTML files**: ~100-500 documents/second
- **Markdown files**: ~500-1000 documents/second
- **Memory usage**: ~10-50MB per document during processing
- **Chunk generation**: ~1-5 chunks per 1000 characters typically

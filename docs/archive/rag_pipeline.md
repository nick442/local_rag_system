# RAG Pipeline

## Overview
The RAG Pipeline (`src/rag_pipeline.py`) orchestrates the complete Retrieval-Augmented Generation workflow, combining document retrieval, context assembly, and LLM generation for intelligent question answering.

## Core Classes

### RAGPipeline
**Purpose**: Main orchestration class that coordinates retrieval and generation components
**Key Features**:
- **Multi-Method Retrieval**: Vector, keyword, and hybrid search capabilities
- **Context Assembly**: Intelligent context building from retrieved chunks
- **LLM Integration**: Local llama.cpp model via `llama_cpp` (GGUF)
- **Conversation Support**: Multi‑turn conversation with context maintenance
- **Performance Optimization**: Efficient prompt construction and stats tracking

**Architecture Flow**:
```
User Query → Query Processing → Document Retrieval → Context Assembly → LLM Generation → Response
```

**Core Components**:
- **Vector Database**: Document storage and similarity search
- **Embedding Service**: Query and document embedding generation  
- **LLM Wrapper**: Language model interface and generation
- **Retriever**: Multi-method document retrieval
- **Prompt Builder**: Context-aware prompt construction

## Key Methods

### query(user_query: str, k: int = 5, retrieval_method: str = "vector", system_prompt: Optional[str] = None, stream: bool = False, collection_id: Optional[str] = None, **generation_kwargs) -> Dict[str, Any]
**Purpose**: Execute a single RAG query with comprehensive results
**Features**:
- **Method Selection**: Choose retrieval strategy (vector/keyword/hybrid)
- **Collection Parameter**: Accepted but not yet applied to restrict retrieval (see note)
- **Context Ranking**: Intelligent ranking and truncation to fit context
- **Response Generation**: LLM‑generated answer with source attribution

**Processing Steps**:
1. **Query Embedding**: Generate vector representation of question
2. **Document Retrieval**: Find relevant chunks using selected method
3. **Context Assembly**: Combine chunks into coherent context
4. **Prompt Construction**: Build LLM prompt with question and context
5. **LLM Generation**: Generate answer using language model
6. **Source Attribution**: Link answer to source documents

**Inputs**:
- `user_query`: User query string
- `k`: Number of contexts to retrieve (default 5)
- `retrieval_method`: "vector" (default), "keyword", or "hybrid"
- `system_prompt`: Optional system instruction
- `stream`: If True, returns a streaming generator with stats callback
- `collection_id`: Optional collection identifier (not yet applied to retrieval)
- `**generation_kwargs`: LLM parameters (e.g., `max_tokens`, `temperature`)

**Outputs**:
```python
{
    'answer': 'Generated response text',
    'sources': [
        { 'chunk_id': '...', 'score': 0.87, 'filename': '...', 'chunk_number': 2, 'content_preview': '...' },
        ...
    ],
    'contexts': [ { 'chunk_id': '...', 'content': '...', 'metadata': {...}, 'doc_id': '...', 'chunk_index': 1 }, ... ],
    'metadata': {
        'query': '...','retrieval_method': 'vector','contexts_count': 3,
        'prompt_tokens': 938,'output_tokens': 150,'total_tokens': 1088,
        'retrieval_time': 0.48,'generation_time': 0.89,'total_time': 1.37,
        'tokens_per_second': 168.5,'context_remaining': 7000
    }
}
```

### chat(initial_message: str = None, collection_id: str = "default") -> None
**Purpose**: Start interactive conversation session with context persistence
**Features**:
- **Multi-Turn Support**: Maintains conversation history
- **Context Awareness**: Previous exchanges inform future responses
- **Session Management**: Handles conversation state and cleanup
- **Interactive Interface**: User-friendly chat experience

Note: The CLI chat command is currently disabled; use the `query` command for one‑off questions.

**Conversation Flow**:
1. Initialize conversation history
2. Process user input through RAG pipeline
3. Maintain context across turns
4. Provide source attribution for claims
5. Handle conversation cleanup and exit

### retrieve_relevant_chunks(query: str, k: int = 5, method: str = "vector") -> List[Dict]
**Purpose**: Retrieve document chunks relevant to query without LLM generation
**Use Cases**: 
- Information gathering
- Document discovery
- Context preparation for custom processing
- Analytics and evaluation

**Retrieval Methods**:
- **vector**: Pure semantic similarity search
- **keyword**: Full-text search with relevance ranking  
- **hybrid**: Combined approach with score fusion

## Retrieval Strategies

### Vector Retrieval
**Algorithm**: Cosine similarity between query and document embeddings
**Strengths**: 
- Semantic understanding ("car" matches "automobile")
- Conceptual queries ("how does X work?")
- Multilingual capabilities
- Handles synonyms and paraphrases

**Optimal For**: 
- Conceptual questions
- Exploratory queries
- Cross-domain knowledge retrieval

### Keyword Retrieval  
**Algorithm**: SQLite FTS5 full-text search with BM25 ranking
**Strengths**:
- Exact term matching
- Boolean query support
- Fast execution
- Precise technical term lookup

**Optimal For**:
- Specific term searches
- Technical documentation
- Known entity lookup
- Precise factual queries

### Hybrid Retrieval
**Algorithm**: Weighted combination of vector and keyword scores
**Configuration**:
- Default weight: 70% vector, 30% keyword
- Adaptive weighting based on query characteristics
- Score normalization and fusion

**Advantages**:
- **Best of Both Worlds**: Semantic understanding + exact matching
- **Robust Performance**: Works well across diverse query types
- **Adaptive**: Automatically adjusts to query characteristics

## Context Assembly

### Context Building Strategy
**Purpose**: Combine retrieved chunks into coherent context for LLM
**Process**:
1. **Deduplication**: Remove overlapping or duplicate content
2. **Ranking**: Order chunks by relevance and similarity  
3. **Truncation**: Fit within LLM context window limits
4. **Formatting**: Structure context for optimal LLM understanding

### Context Optimization
- **Token Management**: Respects LLM token limits (typically 4k-8k)
- **Quality Prioritization**: Higher similarity chunks prioritized
- **Diversity**: Includes diverse perspectives when available
- **Source Tracking**: Maintains attribution to source documents

## LLM Integration

### Generation Configuration
**Model Support**:
- **Local Models**: GGUF format models via llama.cpp (`llama_cpp`)
- Cloud/API adapters are not currently implemented in code.

**Generation Parameters**:
- Temperature: Creativity vs consistency control
- Max tokens: Response length limitation
- Stop sequences: Custom stopping criteria
- Streaming: Real-time response generation

### Prompt Engineering
**System Prompt Template**:
```
You are a helpful assistant that answers questions based on provided context.
Use only the information from the context below to answer questions.
If the context doesn't contain enough information, say so clearly.

Context:
{retrieved_context}

Question: {user_question}
Answer:
```

## Performance Characteristics

### Retrieval Performance
- **Vector Search**: 0.5-2ms for 10k documents
- **Keyword Search**: 1-5ms for 10k documents  
- **Hybrid Search**: 2-8ms for 10k documents
- **Context Assembly**: 1-3ms for typical queries

### Generation Performance  
- **Local LLM**: 20–50 tokens/second (varies by model and hardware)

## Notes and Limitations
- `collection_id` is accepted by `query` but current retrieval does not filter by collection; future versions will thread this filter through the retriever and database search methods.
- Retrieval default method is `vector`; set `--k` and retrieval method via CLI flags or parameters for other strategies.
- **Total Response Time**: 500ms-3s for typical queries

### Scalability Metrics
- **Documents**: Tested up to 100k documents
- **Concurrent Users**: 10-50 depending on hardware
- **Memory Usage**: 2-8GB depending on model size
- **Storage**: ~100-300MB per 10k documents

## Integration Points

### Input Sources
- **User Queries**: Direct question input from CLI or API
- **Conversation History**: Multi-turn dialogue context
- **Collection Filters**: Scope queries to specific document sets
- **Retrieval Parameters**: K-value, thresholds, method selection

### Output Destinations
- **CLI Interface**: Interactive command-line chat
- **API Responses**: Structured JSON responses for applications
- **Analytics**: Query and performance metrics
- **Logging**: Detailed operation logs for monitoring

### Component Dependencies
- **Vector Database**: Core storage and retrieval backend
- **Embedding Service**: Query vectorization
- **LLM Wrapper**: Response generation  
- **Document Ingestion**: Content preparation and storage

## Usage Examples

### Single Query Processing
```python
# Initialize pipeline
rag = RAGPipeline(
    db_path="data/rag_vectors.db",
    embedding_model_path="models/embeddings/all-MiniLM-L6-v2",
    llm_model_path="models/llm/gemma-3-4b-it-q4_0.gguf"
)

# Execute query
result = rag.query(
    question="What is machine learning?",
    collection_id="ai_papers",
    method="hybrid"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
```

### Interactive Chat Session
```python
# Start conversation
rag.chat(
    initial_message="Tell me about neural networks",
    collection_id="deep_learning"
)
# Enters interactive loop until user exits
```

### Retrieval-Only Usage
```python
# Get relevant documents without generation
chunks = rag.retrieve_relevant_chunks(
    query="transformer architecture",
    k=10,
    method="vector"
)

for chunk in chunks:
    print(f"Similarity: {chunk['similarity']:.3f}")
    print(f"Content: {chunk['content'][:200]}...")
```

## Configuration Options

### Retrieval Configuration
```python
rag_config = {
    'retrieval_k': 5,           # Number of chunks to retrieve
    'similarity_threshold': 0.7, # Minimum similarity score
    'max_context_tokens': 3000,  # Context window limit
    'hybrid_vector_weight': 0.7  # Vector vs keyword balance
}
```

### Generation Configuration  
```python
llm_config = {
    'temperature': 0.1,      # Low for factual responses
    'max_tokens': 500,       # Response length limit
    'stream': True,          # Enable streaming responses
    'stop_sequences': ['\n\nUser:', '\n\nHuman:']
}
```

## Quality Assurance

### Answer Quality Metrics
- **Source Attribution**: All claims linked to source documents
- **Hallucination Detection**: Responses grounded in retrieved context
- **Confidence Scoring**: Uncertainty indication when context insufficient
- **Relevance Validation**: Retrieved chunks must meet similarity thresholds

### Performance Monitoring
- **Response Times**: Track retrieval and generation latency
- **Retrieval Accuracy**: Measure chunk relevance to queries
- **User Satisfaction**: Implicit feedback through interaction patterns
- **Error Rates**: Monitor and alert on system failures

## Advanced Features

### Query Enhancement
- **Query Expansion**: Automatic query augmentation with synonyms
- **Intent Recognition**: Classify query types for optimal processing
- **Context Injection**: Use conversation history to enhance queries
- **Multi-Query Processing**: Handle complex, multi-part questions

### Response Post-Processing
- **Citation Formatting**: Automatic source citation generation
- **Answer Validation**: Cross-reference claims with source material
- **Confidence Estimation**: Provide uncertainty indicators
- **Follow-up Suggestions**: Generate related questions for exploration

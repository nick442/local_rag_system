# Local RAG System Project Overview

## Project Purpose
A complete Retrieval-Augmented Generation (RAG) system implemented locally using:
- **LLM**: Gemma-3-4b-it-q4_0 (GGUF format, quantized)
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: SQLite with sqlite-vec extension
- **Platform**: macOS ARM64 with Metal acceleration

## Tech Stack
- **Language**: Python 3.11
- **Environment**: Conda (rag_env)
- **LLM Framework**: llama-cpp-python 0.3.16
- **Embeddings**: sentence-transformers 3.0.1
- **Vector DB**: sqlite-vec 0.1.5
- **Document Processing**: PyPDF2, beautifulsoup4, html2text, markdown
- **Utilities**: tiktoken, pyyaml, click, rich, psutil, aiofiles, tqdm
- **ML Framework**: PyTorch with MPS support

## Architecture Components
- **Document Ingestion** (`src/document_ingestion.py`): Multi-format document loading and chunking
- **Embedding Service** (`src/embedding_service.py`): Batch embedding generation with MPS acceleration
- **Vector Database** (`src/vector_database.py`): SQLite-based vector storage with semantic search
- **Retrieval** (`src/retriever.py`): Vector, keyword, and hybrid search methods
- **LLM Wrapper** (`src/llm_wrapper.py`): Metal-accelerated text generation with streaming
- **Prompt Builder** (`src/prompt_builder.py`): Gemma-3 chat template formatting
- **RAG Pipeline** (`src/rag_pipeline.py`): End-to-end RAG workflow
- **Query Reformulation** (`src/query_reformulation.py`): Query expansion and optimization
- **CLI Interface** (`src/cli_chat.py`): Command-line chat interface

## Project Status
- **Phase 1-3**: Environment setup, dependencies, models ✅
- **Phase 4**: Core RAG components implemented ✅
- **Phase 5**: LLM integration and pipeline completion ✅
- **Vector DB Fix**: Critical sqlite-vec performance issue resolved ✅
- **Production Ready**: System fully operational with optimized vector search
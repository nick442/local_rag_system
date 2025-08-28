# Code Style and Conventions

## Code Style
- **Language**: Python 3.11
- **Type Hints**: Used throughout for better maintainability
- **Docstrings**: Classes and methods documented
- **Import Style**: Standard library, third-party, local imports
- **Line Length**: Standard Python conventions
- **Error Handling**: Comprehensive try-catch with logging

## Project Structure
```
src/                    # Main source code
├── __init__.py
├── document_ingestion.py  # Document loading and chunking
├── embedding_service.py   # Embedding generation
├── vector_database.py     # Vector storage and search
├── retriever.py           # Multi-method retrieval
├── llm_wrapper.py         # LLM interface
├── prompt_builder.py      # Prompt formatting
├── rag_pipeline.py        # End-to-end pipeline
├── query_reformulation.py # Query optimization
└── cli_chat.py            # CLI interface

tests/                  # All test files (organized)
config/                 # Configuration files
models/                 # Model storage
data/                   # Database and data files
corpus/                 # Document corpus
logs/                   # Application logs
handoff/                # Phase completion records
```

## Naming Conventions
- **Classes**: PascalCase (e.g., `DocumentIngestionService`)
- **Methods/Functions**: snake_case (e.g., `ingest_directory`)
- **Variables**: snake_case (e.g., `chunk_size`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`)
- **Files**: snake_case (e.g., `vector_database.py`)

## Architecture Patterns
- **Dependency Injection**: Services passed as constructor parameters
- **Factory Pattern**: Used for component creation
- **Builder Pattern**: Used for prompt construction
- **Strategy Pattern**: Multiple retrieval methods
- **Observer Pattern**: Logging and statistics tracking
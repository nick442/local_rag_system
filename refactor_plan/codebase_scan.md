CODEBASE_SCAN.md - Local RAG System Repository Analysis
Repository Overview

Path: /Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/local_rag_system Last Commit: 8c3ce801e81565dff0949153e145826ec2f18320 (based on project knowledge) Target Platform: Mac mini M4, 16 GB RAM
Python Module Inventory
Core Modules (src/)

Module	Current Responsibility	Lines	Dependencies
document_ingestion.py	Document loading (PDF/HTML/MD/TXT), chunking with tiktoken	~450	tiktoken, PyPDF2, BeautifulSoup, markdown
embedding_service.py	SentenceTransformers embedding generation, GPU/MPS support	~280	sentence_transformers, torch, numpy
vector_database.py	SQLite + sqlite-vec storage, FTS5 search, similarity operations	~650	sqlite3, sqlite_vec, numpy
rag_pipeline.py	Main orchestration, query/chat modes, context assembly	~480	retriever, prompt_builder, llm_wrapper
retriever.py	Vector/keyword/hybrid retrieval methods	~320	vector_database, embedding_service, tiktoken
llm_wrapper.py	llama-cpp-python wrapper, Metal acceleration	~310	llama_cpp
config_manager.py	ProfileConfig (6 params), ExperimentConfig (60+ params)	~450	yaml, dataclasses
system_manager.py	Component initialization, health checks	~380	All components
corpus_manager.py	Parallel bulk ingestion, checkpointing	~520	multiprocessing, vector_database
corpus_organizer.py	Collection management, metadata	~410	vector_database
corpus_analytics.py	Quality metrics, clustering, reporting	~680	scipy, sklearn, numpy
deduplication.py	Hash & LSH duplicate detection	~490	hashlib, datasketch
reindex.py	Database maintenance, re-chunking	~420	vector_database, embedding_service
cli_chat.py	Interactive chat interface	~380	rich, rag_pipeline
experiment_runner.py	Parameter sweeps, A/B testing	~560	config_manager, rag_pipeline
experiment_templates.py	Pre-defined experiment configurations	~340	config_manager
prompt_builder.py	Template-based prompt construction	~180	None
monitor.py	System monitoring, stats tracking	~220	psutil
error_handler.py	Error management, recovery	~190	logging
health_checks.py	System health validation	~160	Various
evaluation_metrics.py	IR metrics (NDCG, MRR, etc.)	~240	numpy
query_reformulation.py	Query expansion (incomplete)	~140	llm_wrapper

CLI Entry Point

Module	Responsibility	Lines
main.py	Click CLI, command routing	~850

Existing vs Missing Components (PDF Spec)
✅ Implemented

    Document ingestion with multiple formats
    Vector storage with SQLite + sqlite-vec
    Embedding service with GPU acceleration
    RAG pipeline with retrieval and generation
    Configuration system (partially - ProfileConfig exists)
    Experiment runner (partially - doesn't apply chunk params)
    CLI interface with basic commands
    Collection support in database schema

❌ Missing/Incomplete

    Config propagation - Profiles don't reach components
    Model reuse - Reloads on every command (4GB+ waste)
    Interface abstractions - No pluggable retrieval/chunker/vector
    JSONL metrics - Basic metrics exist but not structured
    Batch CLI - No batch experiment command
    Collection filtering - Not threaded through retrieval
    PRAGMA tuning - SQLite not optimized
    Embedding dimension handshake - No validation
    Documentation structure - Not organized as kits
    Query reformulation - Stub exists but not integrated

Dependency Graph
Critical Path Dependencies

main.py
├── config_manager.py
├── system_manager.py [REDUNDANT - TO REMOVE]
└── rag_pipeline.py
    ├── retriever.py
    │   ├── vector_database.py
    │   │   └── sqlite_vec (optional fallback)
    │   └── embedding_service.py
    │       └── sentence_transformers
    ├── prompt_builder.py
    └── llm_wrapper.py
        └── llama_cpp

Circular Dependencies

    None detected

Tight Coupling Points

    vector_database.py ↔ SQLite (not abstracted)
    retriever.py ↔ vector_database.py (direct coupling)
    document_ingestion.py ↔ tiktoken (chunker not abstracted)
    rag_pipeline.py → all components (no interfaces)

Legacy Code Requiring Refactoring
High Priority

    system_manager.py - Duplicates ConfigManager, adds confusion
    Experiment runner - Doesn't apply chunking parameters
    Model loading - No caching, wastes memory

Medium Priority

    Collection filtering - Implemented in DB but not used
    Profile switching - Doesn't reload components
    Metrics collection - Ad-hoc, not structured

Low Priority

    query_reformulation.py - Incomplete stub
    monitor.py - May add overhead
    Agent references - AGENTS.md suggests incomplete features

Configuration Analysis
Current State

    Single YAML: config/rag_config.yaml (consolidated)
    3 Profiles: fast, balanced, quality
    6 Core params: retrieval_k, max_tokens, temperature, chunk_size, chunk_overlap, n_ctx

Issues

    Profile params don't propagate to components
    CLI flags don't override profile settings consistently
    Experiment configs define params that aren't applied

Database Schema
Tables

    documents - File metadata, collection_id
    chunks - Text segments, collection_id
    embeddings - Vector blobs
    embeddings_vec - sqlite-vec virtual table
    chunks_fts - FTS5 virtual table
    collections - Collection metadata (referenced but table creation not found)

Indexes

    idx_chunks_doc_id
    idx_chunks_chunk_index
    idx_documents_source
    idx_documents_collection
    idx_chunks_collection

Memory & Performance Characteristics
Current Issues

    Model reloading: ~4GB per command for LLM
    Embedding model: ~500MB per load
    No model caching: Memory thrashing
    SQLite defaults: Not optimized for WAL/caching

Optimization Opportunities

    Implement model singleton cache
    Add PRAGMA optimizations
    Reuse database connections
    Batch embedding operations

Test Coverage
Existing Tests

    test_cli_and_pipeline.py - Basic integration
    test_e2e_realdata.py - End-to-end with real data
    test_vector_db_fallback.py - sqlite-vec fallback
    test_phase_*.py - Phase-specific tests

Missing Tests

    Interface abstraction tests
    Model caching tests
    Profile propagation tests
    Batch CLI tests
    Collection filtering tests

Documentation State
Current Structure

docs/
├── COMPLETE_SYSTEM_GUIDE.md (comprehensive)
├── system_architecture.md
├── [component].md (15+ component docs)
└── progress/ (development tracking)

Required Structure (PDF Spec)

docs/
├── core/
│   ├── ARCH.md
│   ├── CONFIG.md
│   ├── RETRIEVAL.md
│   ├── METRICS.md
│   └── DEV_GUIDE.md
├── experiments/
│   ├── EXPERIMENTS.md
│   ├── EVAL.md
│   ├── RESULTS.md
│   └── TEMPLATES/
└── archive/ (legacy docs)

Risk Assessment
High Risk Areas

    Memory management - No model caching on 16GB system
    Experiment validity - Parameters not actually applied
    Performance - SQLite not optimized

Medium Risk Areas

    Configuration complexity - Multiple overlapping systems
    Documentation debt - Not aligned with implementation

Low Risk Areas

    Core functionality - RAG pipeline works
    Data integrity - Database schema solid

Recommendations Priority
P0 - Critical (Blocks experiments)

    Fix experiment runner to apply chunk parameters
    Implement model caching
    Thread collection_id through stack

P1 - Important (Performance/Usability)

    Remove SystemManager
    Add batch CLI command
    Optimize SQLite with PRAGMAs

P2 - Nice to Have (Clean architecture)

    Create pluggable interfaces
    Add JSONL metrics
    Reorganize documentation

Summary

The codebase is functionally complete but has architectural debt that prevents effective experimentation. The core RAG pipeline works, but parameter sweeps don't actually change behavior, models reload wastefully, and configuration doesn't propagate. With targeted refactoring following the execution plan, this can become an excellent research platform.

Total Python Files: 23 core modules + main.py Total Lines: ~8,500 lines of Python External Dependencies: 15+ packages Database Tables: 6 (including virtual) Configuration Profiles: 3 Test Files: 7

# RAG System Research Proposals - Compatibility Assessment

## Overview

Three detailed research proposals have been developed for the Local RAG System at `/Users/nickwiebe/Documents/claude-workspace/RAGagentProject/agent/Opus/Opus-2/local_rag_system`. Each proposal has been carefully validated against the current system implementation.

## System Compatibility Summary

### ✅ Fully Runnable Experiments

1. **Document Chunking Strategy Optimization** (Proposal 1)
   - **Feasibility**: ⭐⭐⭐⭐⭐ Excellent
   - **System Support**: Complete - all parameters and templates exist
   - **Required Changes**: None
   - **Can Start**: Immediately

3. **Hybrid Retrieval Optimization** (Proposal 3)
   - **Feasibility**: ⭐⭐⭐⭐⭐ Excellent
   - **System Support**: Complete - hybrid retrieval fully implemented
   - **Required Changes**: None (optional query analyzer enhancement)
   - **Can Start**: Immediately

### ⚠️ Requires Minor Integration

2. **Two-Stage Reranking Enhancement** (Proposal 2)
   - **Feasibility**: ⭐⭐⭐⭐ Very Good
   - **System Support**: Parameters exist, model integration needed
   - **Required Changes**: Create `reranker_service.py` and integrate with retriever
   - **Can Start**: After ~2-3 hours of integration work

## Corrected CLI Commands

All proposals have been updated with the correct CLI syntax based on the actual system implementation:

### Correct Command Structure
```bash
# Collections
python main.py collection create <name>
python main.py collection list
python main.py collection delete <id>

# Ingestion
python main.py ingest directory <path> --collection <name>
python main.py ingest file <path> --collection <name>

# Experiments
python main.py experiment template <template_name>
python main.py experiment sweep --param <param> --values <values>
python main.py experiment compare --config-a <config> --config-b <config>
python main.py experiment batch --queries <file> --output <file>

# Analytics
python main.py analytics stats --collection <name>
python main.py analytics quality --collection <name>

# Maintenance
python main.py maintenance dedupe --collection <name>
python main.py maintenance reindex --collection <name>
python main.py maintenance validate --collection <name>

# Configuration
python main.py config show
python main.py config set <key> <value>
python main.py config switch-profile <profile>
```

## Key System Features Confirmed

### Experimental Infrastructure ✅
- `experiment_runner.py`: Full parameter sweep orchestration
- `experiment_templates.py`: 9 pre-defined templates including:
  - `chunk_optimization`
  - `retrieval_methods`
  - `model_comparison`
- Database persistence for experiment tracking
- Statistical analysis capabilities

### Retrieval Methods ✅
- **Vector search**: sqlite-vec implementation
- **Keyword search**: FTS5 integration
- **Hybrid search**: Already implemented with score fusion
- **Configurable parameters**: retrieval_k, similarity_threshold (alpha)

### Configuration Management ✅
- ProfileConfig: fast/balanced/quality presets
- ExperimentConfig: 60+ experimental parameters
- Parameter override system
- JSONL metrics output with `--metrics` flag

## Implementation Priority

### Recommended Execution Order

1. **Start First**: Hybrid Retrieval Optimization (Proposal 3)
   - No integration required
   - Can begin immediately
   - Tests existing functionality
   - 5-day timeline

2. **Start Second**: Document Chunking Optimization (Proposal 1)
   - No integration required
   - Can run in parallel with Proposal 3
   - Fundamental to system performance
   - 5-day timeline

3. **Start After Integration**: Reranking Enhancement (Proposal 2)
   - Requires 2-3 hours to create reranker service
   - High impact on precision
   - 5-day timeline after integration

## Resource Allocation

### Total Requirements (All 3 Experiments)
- **Time**: 15 days sequential, 10 days with some parallel execution
- **Compute**: ~50 hours total experiment runtime
- **Storage**: 150GB for corpus, models, and results
- **Memory**: 12GB operational (4GB system reserved)

### Parallel Execution Strategy
- Days 1-5: Hybrid Retrieval (Proposal 3)
- Days 3-7: Chunking Strategy (Proposal 1) - overlap possible
- Days 6-7: Reranker Integration
- Days 8-12: Reranking Experiments (Proposal 2)

## Quick Start Commands

### Verify System Ready
```bash
# Check system health
python main.py doctor

# List available templates
python main.py experiment template --list-templates

# Show current configuration
python main.py config show
```

### Start Experiment 3 (Hybrid - Immediate)
```bash
# Create collection
python main.py collection create hybrid_test

# Ingest corpus
python main.py ingest directory corpus/hybrid --collection hybrid_test

# Run experiment
python main.py experiment template retrieval_methods \
  --corpus hybrid_test \
  --output results/hybrid_baseline.json
```

### Start Experiment 1 (Chunking - Immediate)
```bash
# Create collection
python main.py collection create chunking_test

# Ingest corpus
python main.py ingest directory corpus/technical --collection chunking_test

# Run experiment
python main.py experiment template chunk_optimization \
  --corpus chunking_test \
  --output results/chunking_baseline.json
```

### Prepare Experiment 2 (Reranking - After Integration)
```bash
# First, create reranker service
cp experiments/reranking/reranker_service.py src/

# Test integration
python -c "from src.reranker_service import RerankerService; print('✓')"

# Then proceed with experiments
python main.py experiment sweep \
  --param rerank_model \
  --values "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --output results/reranking_test.json
```

## Files Created

1. `/experiments/Opus_proposals/1_chunking_strategy_optimization.md`
2. `/experiments/Opus_proposals/2_reranking_enhancement.md`
3. `/experiments/Opus_proposals/3_hybrid_retrieval_optimization.md`
4. `/experiments/Opus_proposals/README.md` (this file)

## Conclusion

All three research proposals are well-suited for the current RAG system:

- **Two experiments** (Chunking and Hybrid) can run immediately with no modifications
- **One experiment** (Reranking) requires minimal integration work (~2-3 hours)
- All experiments use the existing experimental framework and CLI commands
- Total expected improvement: 15-30% across various metrics

The proposals provide comprehensive, scientifically rigorous approaches to RAG optimization on consumer hardware, with clear implementation guides and expected outcomes.

---

**Next Steps**:
1. Review proposals for scientific rigor
2. Allocate resources (time, compute, storage)
3. Begin with Hybrid Retrieval experiment (no setup required)
4. Start corpus preparation for all experiments in parallel

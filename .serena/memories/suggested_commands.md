# Essential Commands for RAG System Development

## Environment Activation
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env
```

## Python Command Prefix
Always use: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python`

## Testing Commands
- **Run Phase 4 tests**: `python tests/test_phase_4.py`
- **Run Phase 5 tests**: `python tests/test_phase_5.py`  
- **Run vector DB tests**: `python tests/test_vector_database_fix.py`
- **Run complete RAG test**: `python tests/test_rag_retrieval_final.py`
- **Run sqlite-vec test**: `python tests/test_sqlite_vec_fix.py`

## Demo and Usage
- **RAG Demo**: `python demo_fixed_rag.py`
- **Phase 5 Benchmark**: `python benchmark_phase_5.py`

## Development Utilities
- **List directory**: `ls -la`
- **Find files**: `find . -name "*.py" -type f`
- **Search code**: `grep -r "pattern" src/`
- **Git status**: `git status`
- **Database size**: Check `data/` directory

## Performance Testing
- **Model benchmarks**: `python benchmark_phase_5.py`
- **Vector search test**: `python tests/test_database_fix_final.py`

## Environment Verification
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env && python -c "import llama_cpp, sentence_transformers, sqlite_vec; print('Environment OK')"
```
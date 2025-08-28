# Task Completion Workflow

## When a Task is Completed

### 1. Testing Requirements
- **Unit Tests**: Run relevant test files in `tests/` directory
- **Integration Tests**: Run end-to-end tests like `test_rag_retrieval_final.py`
- **Performance Tests**: Run benchmarks if performance-related changes
- **Component Tests**: Test individual components if modified

### 2. Test Commands to Run
```bash
# Always use conda environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env

# Run specific component tests
python tests/test_vector_database_fix.py    # For vector DB changes
python tests/test_phase_4.py                # For core RAG components  
python tests/test_phase_5.py                # For LLM integration
python tests/test_rag_retrieval_final.py    # For end-to-end functionality

# Run demo to verify everything works
python demo_fixed_rag.py
```

### 3. Documentation Updates
- Update `claude_workdone.md` with concise documentation of actions
- Update relevant memory files if architecture changes
- No proactive creation of documentation files unless requested

### 4. Code Quality Checks
- No formal linting/formatting tools configured
- Manual code review for:
  - Proper error handling
  - Type hints usage
  - Consistent naming conventions
  - Memory management (especially for LLM operations)

### 5. Performance Verification
- For vector operations: Ensure sqlite-vec extension working
- For LLM operations: Check Metal acceleration enabled
- For embeddings: Verify MPS acceleration active
- For memory: Monitor memory usage during operations

### 6. Critical System Checks
- **sqlite-vec Extension**: Must load successfully for production performance
- **Metal Acceleration**: LLM should use GPU layers (-1 setting)
- **MPS Support**: PyTorch operations should use Metal Performance Shaders
- **Model Loading**: Verify model paths and loading times acceptable

## Warning Signs to Check
- sqlite-vec fallback to manual search (performance killer)
- Model loading without Metal acceleration  
- Memory leaks during batch operations
- FTS5 search errors (non-critical but should be noted)
- Token counting discrepancies
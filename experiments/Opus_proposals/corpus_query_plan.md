# Unified Corpus and Query Set Strategy for RAG Experiments

## Executive Summary

After analyzing the three experiments, I recommend creating a **unified multi-purpose corpus** that can serve all experiments while maintaining experimental validity. This approach will save significant time and resources while enabling meaningful comparisons across experiments.

## Corpus Architecture

### 1. Unified Corpus Design

**Total Size**: 10,000 documents  
**Structure**: Hierarchical with semantic and difficulty gradients

```
unified_corpus/
├── technical_specs/        # 2,000 docs - High specificity
│   ├── api_docs/           # 500 docs
│   ├── config_files/       # 500 docs
│   ├── error_codes/        # 500 docs
│   └── version_docs/       # 500 docs
├── conceptual_content/      # 3,000 docs - High semantic density
│   ├── tutorials/          # 1,000 docs
│   ├── explanations/       # 1,000 docs
│   └── theoretical/        # 1,000 docs
├── mixed_content/           # 3,000 docs - Balanced
│   ├── technical_blogs/    # 1,000 docs
│   ├── stackoverflow/      # 1,000 docs
│   └── documentation/      # 1,000 docs
├── reference_materials/     # 1,500 docs - Structured
│   ├── readme_files/       # 500 docs
│   ├── faqs/              # 500 docs
│   └── glossaries/        # 500 docs
└── distractor_content/     # 500 docs - Noise/edge cases
    ├── ambiguous/          # 250 docs
    └── keyword_heavy/      # 250 docs
```

### 2. Why This Works for All Experiments

| Experiment | Required Categories | Mapped From Unified Corpus | Coverage |
|-----------|-------------------|---------------------------|----------|
| **Chunking** | Technical (2k), Narrative (2k), Reference (1k) | technical_specs + mixed_content/stackoverflow, conceptual_content, reference_materials | ✅ 100% |
| **Reranking** | Highly relevant (1k), Partially (2k), Tangential (2k), Distractors (1k) | technical_specs (subset), mixed_content, conceptual_content, distractor_content | ✅ 100% |
| **Hybrid** | Technical specs (1.5k), Conceptual (1.5k), Mixed (1.5k), Proper nouns (500) | technical_specs, conceptual_content, mixed_content, reference_materials | ✅ 100% |

## Corpus Acquisition Strategy

### Phase 1: Automated Collection (Days 1-2)

```bash
#!/bin/bash
# corpus_collection.sh

# Create directory structure
mkdir -p unified_corpus/{technical_specs,conceptual_content,mixed_content,reference_materials,distractor_content}

# 1. Technical Specifications (2,000 docs)
# API Documentation
wget -r -l 2 -A "*.md,*.txt,*.json" \
  https://docs.python.org/3/library/ \
  -P unified_corpus/technical_specs/api_docs/ &

# Configuration examples from GitHub
python scripts/github_crawler.py \
  --search "filename:config extension:yml OR extension:yaml OR extension:toml" \
  --stars ">100" \
  --count 500 \
  --output unified_corpus/technical_specs/config_files/

# Error documentation
python scripts/fetch_error_docs.py \
  --sources "python,javascript,java,go" \
  --count 500 \
  --output unified_corpus/technical_specs/error_codes/

# 2. Conceptual Content (3,000 docs)
# Tutorials from multiple sources
python scripts/fetch_tutorials.py \
  --sources "realpython,medium,dev.to" \
  --topics "machine-learning,web-development,databases,cloud" \
  --count 1000 \
  --output unified_corpus/conceptual_content/tutorials/

# Wikipedia technical articles
python scripts/wikipedia_technical.py \
  --categories "Computer_science,Software_engineering,Information_technology" \
  --count 1000 \
  --output unified_corpus/conceptual_content/explanations/

# 3. Mixed Content (3,000 docs)
# StackOverflow high-quality Q&As
python scripts/stackoverflow_fetcher.py \
  --min-score 50 \
  --tags "python,javascript,machine-learning,database" \
  --count 1000 \
  --output unified_corpus/mixed_content/stackoverflow/

# Technical blog posts
python scripts/devto_fetcher.py \
  --min-reactions 20 \
  --tags "tutorial,webdev,programming,ai" \
  --count 1000 \
  --output unified_corpus/mixed_content/technical_blogs/

# 4. Reference Materials (1,500 docs)
# High-star GitHub READMEs
python scripts/github_readme_fetcher.py \
  --stars ">1000" \
  --languages "python,javascript,go,rust" \
  --count 500 \
  --output unified_corpus/reference_materials/readme_files/

# 5. Distractor Content (500 docs)
# Generate ambiguous content
python scripts/generate_distractors.py \
  --method "keyword_stuffing,context_switching" \
  --base-corpus unified_corpus/technical_specs \
  --count 500 \
  --output unified_corpus/distractor_content/
```

### Phase 2: Quality Control & Enrichment

```python
# corpus_quality_control.py
import json
from pathlib import Path
import hashlib
from typing import Dict, List

class CorpusQualityController:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.metadata = {}
        
    def validate_and_enrich(self):
        """Validate corpus quality and add metadata."""
        for category in self.corpus_path.iterdir():
            if category.is_dir():
                self.process_category(category)
        
        self.remove_duplicates()
        self.ensure_minimum_lengths()
        self.add_difficulty_scores()
        self.add_semantic_labels()
        return self.metadata
    
    def process_category(self, category_path: Path):
        """Process each category of documents."""
        docs = list(category_path.rglob("*.*"))
        
        for doc_path in docs:
            # Calculate document metrics
            content = doc_path.read_text(errors='ignore')
            
            metadata = {
                "path": str(doc_path),
                "category": category_path.name,
                "subcategory": doc_path.parent.name,
                "word_count": len(content.split()),
                "char_count": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
                "has_code": self._has_code_blocks(content),
                "technical_density": self._calculate_technical_density(content),
                "readability_score": self._calculate_readability(content)
            }
            
            self.metadata[str(doc_path)] = metadata
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if document contains code."""
        indicators = ["```", "def ", "function ", "import ", "class "]
        return any(ind in content for ind in indicators)
    
    def _calculate_technical_density(self, content: str) -> float:
        """Calculate technical term density."""
        technical_terms = [
            "algorithm", "api", "database", "function", "variable",
            "implementation", "architecture", "framework", "library"
        ]
        words = content.lower().split()
        if not words:
            return 0.0
        
        technical_count = sum(1 for word in words if any(term in word for term in technical_terms))
        return technical_count / len(words)
    
    def _calculate_readability(self, content: str) -> float:
        """Simple readability score based on sentence and word length."""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Simple readability formula
        return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length
```

## Query Set Design

### Unified Query Architecture

Create a **hierarchical query set** with 200 queries that can be filtered for each experiment:

```json
{
  "query_set": {
    "factual": {
      "simple": [
        {"id": "f_s_001", "query": "What is Python?", "expected_docs": 1, "difficulty": 0.2},
        {"id": "f_s_002", "query": "Define API", "expected_docs": 1, "difficulty": 0.2}
      ],
      "specific": [
        {"id": "f_sp_001", "query": "Python 3.9.7 breaking changes", "expected_docs": 3, "difficulty": 0.6},
        {"id": "f_sp_002", "query": "ERROR_CODE_404 meaning", "expected_docs": 2, "difficulty": 0.5}
      ]
    },
    "explanatory": {
      "conceptual": [
        {"id": "e_c_001", "query": "How does machine learning work?", "expected_docs": 5, "difficulty": 0.7},
        {"id": "e_c_002", "query": "Explain neural network backpropagation", "expected_docs": 4, "difficulty": 0.8}
      ],
      "procedural": [
        {"id": "e_p_001", "query": "How to implement binary search tree?", "expected_docs": 4, "difficulty": 0.6},
        {"id": "e_p_002", "query": "Steps to deploy Docker container", "expected_docs": 3, "difficulty": 0.5}
      ]
    },
    "comparative": [
      {"id": "c_001", "query": "Python vs JavaScript for web development", "expected_docs": 6, "difficulty": 0.6},
      {"id": "c_002", "query": "REST vs GraphQL API comparison", "expected_docs": 5, "difficulty": 0.7}
    ],
    "ambiguous": [
      {"id": "a_001", "query": "Python", "expected_docs": 8, "difficulty": 0.3, "needs_context": true},
      {"id": "a_002", "query": "Kernel", "expected_docs": 7, "difficulty": 0.4, "needs_context": true}
    ]
  },
  "experiment_mappings": {
    "chunking": ["factual", "explanatory", "comparative"],
    "reranking": ["ambiguous", "factual.specific", "comparative"],
    "hybrid": ["factual.specific", "explanatory.conceptual", "ambiguous"]
  }
}
```

## Implementation Plan

### Step 1: Corpus Preparation (Days 1-2)

```bash
#!/bin/bash
# prepare_unified_corpus.sh

echo "📚 Starting Unified Corpus Preparation for RAG Experiments"

# 1. Setup environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rag_env

# 2. Create corpus structure
bash scripts/create_corpus_structure.sh

# 3. Run parallel collection scripts
echo "📥 Collecting documents (this will take 4-6 hours)..."
python scripts/parallel_corpus_collector.py \
  --config config/corpus_sources.yaml \
  --output unified_corpus/ \
  --workers 8

# 4. Quality control and deduplication
echo "🔍 Running quality control..."
python scripts/corpus_quality_control.py \
  --input unified_corpus/ \
  --min-words 50 \
  --max-words 5000 \
  --remove-duplicates \
  --output-metadata corpus_metadata.json

# 5. Generate corpus statistics
echo "📊 Generating corpus statistics..."
python scripts/corpus_statistics.py \
  --corpus unified_corpus/ \
  --metadata corpus_metadata.json \
  --output reports/corpus_stats.html

echo "✅ Corpus preparation complete!"
```

### Step 2: Ingestion Strategy

```python
# ingest_for_experiments.py
"""
Smart ingestion strategy that creates optimal collections for each experiment
while reusing the same base corpus.
"""

class ExperimentIngestionManager:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        
    def ingest_for_chunking_experiment(self):
        """Create collections optimized for chunking experiment."""
        collections = {
            "chunking_technical": ["technical_specs", "mixed_content/stackoverflow"],
            "chunking_narrative": ["conceptual_content"],
            "chunking_reference": ["reference_materials"]
        }
        
        for collection_name, subdirs in collections.items():
            self._create_and_ingest(collection_name, subdirs)
    
    def ingest_for_reranking_experiment(self):
        """Create collections with relevance gradients."""
        # Use metadata to sort by relevance scores
        collections = {
            "reranking_relevant": self._get_top_relevant_docs(1000),
            "reranking_partial": self._get_partially_relevant_docs(2000),
            "reranking_tangential": self._get_tangential_docs(2000),
            "reranking_distractor": ["distractor_content"]
        }
        
        for collection_name, docs in collections.items():
            self._create_and_ingest(collection_name, docs)
    
    def ingest_for_hybrid_experiment(self):
        """Create collections optimized for hybrid retrieval."""
        collections = {
            "hybrid_keyword": ["technical_specs"],  # Keyword-heavy
            "hybrid_semantic": ["conceptual_content"],  # Semantic-rich
            "hybrid_mixed": ["mixed_content"],  # Balanced
            "hybrid_all": ["technical_specs", "conceptual_content", "mixed_content", "reference_materials"]
        }
        
        for collection_name, subdirs in collections.items():
            self._create_and_ingest(collection_name, subdirs)
    
    def _create_and_ingest(self, collection_name: str, sources: List[str]):
        """Create collection and ingest documents."""
        print(f"Creating collection: {collection_name}")
        
        # Create collection
        os.system(f"python main.py collection create {collection_name}")
        
        # Ingest from specified sources
        for source in sources:
            source_path = self.corpus_path / source
            if source_path.exists():
                cmd = f"python main.py ingest directory {source_path} --collection {collection_name} --deduplicate"
                os.system(cmd)
        
        # Verify ingestion
        os.system(f"python main.py analytics stats --collection {collection_name}")
```

### Step 3: Query Set Generation

```python
# generate_query_sets.py
"""
Generate experiment-specific query sets from unified query architecture.
"""

class QuerySetGenerator:
    def __init__(self, unified_queries: str):
        with open(unified_queries) as f:
            self.queries = json.load(f)
    
    def generate_chunking_queries(self) -> Dict:
        """Generate queries for chunking experiment."""
        selected = []
        
        # Get diverse query types
        for category in ["factual", "explanatory", "comparative"]:
            selected.extend(self._flatten_queries(self.queries["query_set"][category]))
        
        return {
            "experiment": "chunking_optimization",
            "queries": selected[:50],  # Use 50 queries
            "categories": self._categorize_by_difficulty(selected[:50])
        }
    
    def generate_reranking_queries(self) -> Dict:
        """Generate queries for reranking experiment."""
        selected = []
        
        # Focus on ambiguous and specific queries
        selected.extend(self.queries["query_set"]["ambiguous"])
        selected.extend(self.queries["query_set"]["factual"]["specific"])
        selected.extend(self.queries["query_set"]["comparative"])
        
        return {
            "experiment": "reranking_enhancement",
            "queries": selected[:40],
            "categories": {
                "ambiguous": [q for q in selected if q.get("needs_context")],
                "specific": [q for q in selected if q["difficulty"] > 0.5]
            }
        }
    
    def generate_hybrid_queries(self) -> Dict:
        """Generate queries for hybrid retrieval experiment."""
        queries = {
            "keyword_optimal": [],
            "vector_optimal": [],
            "hybrid_optimal": []
        }
        
        # Classify queries by optimal retrieval method
        for category, items in self.queries["query_set"].items():
            flattened = self._flatten_queries(items)
            
            for query in flattened:
                if self._is_keyword_query(query["query"]):
                    queries["keyword_optimal"].append(query)
                elif self._is_semantic_query(query["query"]):
                    queries["vector_optimal"].append(query)
                else:
                    queries["hybrid_optimal"].append(query)
        
        return {
            "experiment": "hybrid_optimization",
            "queries": queries,
            "total": sum(len(v) for v in queries.values())
        }
```

## Vectorization Strategy

### Efficient Batch Processing

```python
# vectorize_corpus.py
"""
Optimized vectorization for large corpus with memory constraints.
"""

class CorpusVectorizer:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    def vectorize_for_experiments(self):
        """Vectorize corpus with different chunking strategies."""
        
        # Define chunking configurations for experiments
        chunking_configs = [
            {"size": 128, "overlap": 32},
            {"size": 256, "overlap": 64},
            {"size": 512, "overlap": 128},
            {"size": 768, "overlap": 192},
            {"size": 1024, "overlap": 256}
        ]
        
        for config in chunking_configs:
            collection_name = f"chunks_{config['size']}_{config['overlap']}"
            
            print(f"Creating collection: {collection_name}")
            self._create_chunked_collection(
                collection_name,
                chunk_size=config["size"],
                chunk_overlap=config["overlap"]
            )
    
    def _create_chunked_collection(self, name: str, chunk_size: int, chunk_overlap: int):
        """Create and vectorize a collection with specific chunking."""
        
        # Create collection with chunking parameters
        cmd = f"""python main.py collection create {name} && \
                  python main.py ingest directory unified_corpus \
                  --collection {name} \
                  --chunk-size {chunk_size} \
                  --chunk-overlap {chunk_overlap} \
                  --batch-size {self.batch_size}"""
        
        os.system(cmd)
        
        # Verify vectorization
        os.system(f"python main.py collection stats --id {name}")
```

## Critical Success Factors

### 1. Corpus Quality Metrics

```python
MINIMUM_QUALITY_THRESHOLDS = {
    "document_length": (50, 5000),  # words
    "technical_density": 0.05,  # min 5% technical terms
    "uniqueness": 0.95,  # max 5% duplication
    "category_balance": 0.7,  # min 70% of target per category
}
```

### 2. Memory Management

```bash
# Monitor memory during ingestion
python scripts/memory_monitor.py --process "main.py ingest" --alert-threshold 14GB
```

### 3. Incremental Processing

```python
# Enable checkpoint-based recovery
python main.py ingest directory unified_corpus \
  --collection experiment_corpus \
  --checkpoint-interval 100 \
  --resume-from-checkpoint
```

## Validation Checklist

- [ ] **Corpus Size**: 10,000 documents collected
- [ ] **Quality Control**: All documents meet minimum thresholds
- [ ] **Deduplication**: <5% duplicate content
- [ ] **Category Balance**: Each category within 20% of target
- [ ] **Query Coverage**: All 200 queries have relevant documents
- [ ] **Vectorization**: All collections successfully embedded
- [ ] **Memory Usage**: Peak usage <14GB during processing
- [ ] **Experimental Validity**: Each experiment has required data

## Why This Unified Approach Works

### Benefits:
1. **Resource Efficiency**: Single corpus serves all experiments
2. **Consistency**: Same document pool ensures comparability
3. **Flexibility**: Can create experiment-specific views via collections
4. **Scalability**: Easy to add more documents or experiments
5. **Reproducibility**: Single source of truth for all experiments

### Trade-offs Addressed:
- **Specificity**: Metadata and categorization maintain experimental requirements
- **Contamination**: Collection isolation prevents cross-experiment interference
- **Validity**: Careful mapping ensures each experiment gets appropriate data

## Next Steps

1. **Execute corpus collection** (6-8 hours automated)
2. **Run quality control** (1 hour)
3. **Create experiment collections** (2 hours)
4. **Generate query sets** (30 minutes)
5. **Begin experiments** in parallel

This unified approach will provide a robust foundation for all three experiments while maintaining scientific validity and resource efficiency.
"""
Pre-defined Experiment Templates for RAG System
Common research patterns and experimental setups.
"""

from typing import Dict, List
from .config_manager import ExperimentConfig, ParameterRange, ExperimentTemplate


# Default evaluation queries for experiments
DEFAULT_EVALUATION_QUERIES = [
    "What is machine learning?",
    "How does artificial intelligence work?", 
    "Explain deep learning algorithms.",
    "What are neural networks?",
    "How do large language models work?",
    "What is natural language processing?",
    "Explain computer vision techniques.",
    "What is reinforcement learning?",
    "How does data preprocessing work?",
    "What are the challenges in AI development?"
]

TECHNICAL_EVALUATION_QUERIES = [
    "How do you implement a neural network?",
    "What are the best practices for machine learning deployment?",
    "How do you optimize model performance?",
    "What is the difference between supervised and unsupervised learning?",
    "How do you handle overfitting in deep learning?",
    "What are transformer architectures?",
    "How does backpropagation work?",
    "What is gradient descent optimization?",
    "How do you evaluate model performance?",
    "What are convolutional neural networks?"
]


def create_base_experiment_config() -> ExperimentConfig:
    """Create a default experimental configuration."""
    return ExperimentConfig(
        # Core ProfileConfig parameters
        retrieval_k=5,
        max_tokens=1024,
        temperature=0.8,
        chunk_size=512,
        chunk_overlap=128,
        n_ctx=8192,
        
        # Extended experimental parameters with defaults
        chunking_strategy="token-based",
        min_chunk_size=128,
        max_chunk_size=2048,
        overlap_strategy="fixed",
        preprocessing_steps=["clean", "normalize"],
        chunk_quality_threshold=0.5,
        document_filter_regex=None,
        
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=None,
        retrieval_method="vector",
        similarity_threshold=0.0,
        similarity_metric="cosine",
        rerank_model=None,
        rerank_top_k=20,
        query_expansion=False,
        query_expansion_model=None,
        search_algorithm="exact",
        index_type="flat",
        retrieval_fusion_method="reciprocal_rank",
        
        llm_model_path="models/gemma-3-4b-it-q4_0.gguf",
        llm_type="gemma-3",
        top_k=40,
        top_p=0.95,
        repeat_penalty=1.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        system_prompt_template=None,
        response_format="structured",
        conversation_memory=True,
        context_window_strategy="truncate",
        generation_strategy="sampling",
        stop_sequences=[],
        response_length_target=None,
        citation_style="inline",
        
        collection_filter=None,
        collection_weighting=None,
        target_corpus="default",
        database_backend="sqlite",
        corpus_preprocessing=["dedupe"],
        duplicate_detection_threshold=0.9,
        document_freshness_weight=1.0,
        corpus_size_limit=None,
        index_build_strategy="incremental",
        cache_strategy="balanced"
    )


# Pre-defined experiment templates
EXPERIMENT_TEMPLATES: Dict[str, ExperimentTemplate] = {
    
    "chunk_optimization": ExperimentTemplate(
        name="Chunk Size and Overlap Optimization",
        description="Systematic study of chunking strategies on retrieval quality and response accuracy",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("chunk_size", "categorical", values=[128, 256, 512, 1024, 2048]),
            ParameterRange("chunk_overlap", "categorical", values=[32, 64, 128, 256]),
            ParameterRange("chunking_strategy", "categorical", 
                         values=["token-based", "sentence-based", "paragraph-based"])
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES,
        evaluation_metrics=["retrieval_precision", "retrieval_recall", "response_quality", "response_time"],
        expected_runtime_hours=3.0
    ),
    
    "model_comparison": ExperimentTemplate(
        name="Embedding and LLM Model Comparison", 
        description="Compare different embedding and language model combinations for optimal performance",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("embedding_model_path", "categorical", 
                         values=[
                             "sentence-transformers/all-MiniLM-L6-v2",
                             "sentence-transformers/all-mpnet-base-v2",
                             "sentence-transformers/e5-base-v2"
                         ]),
            ParameterRange("llm_model_path", "categorical",
                         values=[
                             "models/gemma-3-4b-it-q4_0.gguf",
                             "models/llama-3.2-3b-instruct-q4_0.gguf",
                             "models/mistral-7b-instruct-q4_0.gguf"
                         ])
        ],
        evaluation_queries=TECHNICAL_EVALUATION_QUERIES,
        evaluation_metrics=["accuracy", "speed", "memory_usage", "response_quality"],
        expected_runtime_hours=6.0
    ),
    
    "retrieval_methods": ExperimentTemplate(
        name="Retrieval Method Analysis",
        description="Compare vector, keyword, and hybrid search methods for different query types",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("retrieval_method", "categorical",
                         values=["vector", "keyword", "hybrid"]),
            ParameterRange("retrieval_k", "categorical", values=[3, 5, 7, 10, 15, 20]),
            ParameterRange("similarity_threshold", "linear", 0.0, 0.8, 0.2)
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES + TECHNICAL_EVALUATION_QUERIES,
        evaluation_metrics=["retrieval_precision", "retrieval_recall", "retrieval_speed", "response_relevance"],
        expected_runtime_hours=2.5
    ),
    
    "generation_tuning": ExperimentTemplate(
        name="LLM Generation Parameter Tuning",
        description="Optimize temperature, top_k, and other generation parameters for quality vs creativity balance",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("temperature", "linear", 0.1, 1.5, 0.2),
            ParameterRange("top_k", "categorical", values=[10, 20, 40, 80, 120]),
            ParameterRange("top_p", "linear", 0.7, 0.99, 0.05),
            ParameterRange("repeat_penalty", "linear", 1.0, 1.3, 0.05)
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES,
        evaluation_metrics=["factual_accuracy", "response_creativity", "hallucination_rate", "response_coherence"],
        expected_runtime_hours=4.0
    ),
    
    "context_length_study": ExperimentTemplate(
        name="Context Window Impact Analysis",
        description="Study impact of context length and retrieval count on response quality vs speed",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("n_ctx", "categorical", values=[2048, 4096, 8192, 16384]),
            ParameterRange("max_tokens", "categorical", values=[256, 512, 1024, 2048]),
            ParameterRange("retrieval_k", "categorical", values=[3, 5, 10, 15, 25]),
            ParameterRange("context_window_strategy", "categorical", 
                         values=["truncate", "summarize"])
        ],
        evaluation_queries=TECHNICAL_EVALUATION_QUERIES,
        evaluation_metrics=["response_quality", "response_completeness", "generation_speed", "memory_usage"],
        expected_runtime_hours=5.0
    ),
    
    "preprocessing_impact": ExperimentTemplate(
        name="Document Preprocessing Impact Study",
        description="Analyze the effect of different preprocessing steps on retrieval and generation quality",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("preprocessing_steps", "categorical",
                         values=[
                             ["clean"],
                             ["clean", "normalize"],
                             ["clean", "normalize", "dedupe"],
                             ["clean", "normalize", "dedupe", "quality_filter"]
                         ]),
            ParameterRange("chunk_quality_threshold", "linear", 0.3, 0.9, 0.2),
            ParameterRange("duplicate_detection_threshold", "linear", 0.7, 0.95, 0.05)
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES,
        evaluation_metrics=["retrieval_precision", "response_accuracy", "processing_time", "corpus_size_reduction"],
        expected_runtime_hours=3.5
    ),
    
    "similarity_metrics": ExperimentTemplate(
        name="Similarity Metric Comparison",
        description="Compare cosine, dot product, and euclidean similarity metrics for vector retrieval",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("similarity_metric", "categorical", 
                         values=["cosine", "dot_product", "euclidean"]),
            ParameterRange("similarity_threshold", "linear", 0.0, 0.9, 0.1),
            ParameterRange("retrieval_k", "categorical", values=[5, 10, 15])
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES,
        evaluation_metrics=["retrieval_precision", "retrieval_recall", "retrieval_diversity", "query_time"],
        expected_runtime_hours=2.0
    ),
    
    "prompt_engineering": ExperimentTemplate(
        name="System Prompt Engineering Study", 
        description="Test different system prompt templates and response formats for optimal output quality",
        base_config=create_base_experiment_config(),
        parameter_ranges=[
            ParameterRange("response_format", "categorical",
                         values=["structured", "json", "free-form", "markdown"]),
            ParameterRange("citation_style", "categorical",
                         values=["inline", "footnote", "bibliography", "none"]),
            ParameterRange("system_prompt_template", "categorical",
                         values=[
                             None,  # Use default
                             "You are a technical expert. Provide detailed, accurate answers with examples.",
                             "You are a helpful assistant. Give concise, practical answers.",
                             "You are an academic researcher. Provide comprehensive, well-cited responses."
                         ])
        ],
        evaluation_queries=TECHNICAL_EVALUATION_QUERIES,
        evaluation_metrics=["response_quality", "citation_accuracy", "format_adherence", "user_satisfaction"],
        expected_runtime_hours=2.5
    ),
    
    "performance_scaling": ExperimentTemplate(
        name="Performance Scaling Analysis",
        description="Study system performance scaling with different corpus sizes and complexity parameters",
        base_config=create_base_experiment_config(), 
        parameter_ranges=[
            ParameterRange("corpus_size_limit", "categorical", values=[1000, 5000, 10000, 25000, None]),
            ParameterRange("cache_strategy", "categorical", values=["minimal", "balanced", "aggressive"]),
            ParameterRange("index_build_strategy", "categorical", values=["incremental", "batch"]),
            ParameterRange("retrieval_k", "categorical", values=[5, 10, 20])
        ],
        evaluation_queries=DEFAULT_EVALUATION_QUERIES,
        evaluation_metrics=["query_latency", "memory_usage", "index_build_time", "storage_efficiency"],
        expected_runtime_hours=4.5
    )
}


def get_template(name: str) -> ExperimentTemplate:
    """Get experiment template by name."""
    if name not in EXPERIMENT_TEMPLATES:
        available = list(EXPERIMENT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available templates: {available}")
    return EXPERIMENT_TEMPLATES[name]


def list_templates() -> List[str]:
    """List all available experiment template names."""
    return list(EXPERIMENT_TEMPLATES.keys())


def get_template_info(name: str) -> Dict[str, any]:
    """Get detailed information about a template."""
    template = get_template(name)
    
    return {
        "name": template.name,
        "description": template.description,
        "parameters": [pr.param_name for pr in template.parameter_ranges],
        "parameter_count": len(template.parameter_ranges),
        "evaluation_queries_count": len(template.evaluation_queries),
        "expected_runtime_hours": template.expected_runtime_hours,
        "evaluation_metrics": template.evaluation_metrics,
        "estimated_combinations": _estimate_combinations(template)
    }


def _estimate_combinations(template: ExperimentTemplate) -> int:
    """Estimate total number of parameter combinations for a template."""
    total = 1
    for param_range in template.parameter_ranges:
        values = param_range.generate_values()
        total *= len(values)
    return total


def create_custom_template(name: str, description: str, 
                          parameter_ranges: List[ParameterRange],
                          evaluation_queries: List[str] = None,
                          base_config: ExperimentConfig = None) -> ExperimentTemplate:
    """Create a custom experiment template."""
    if not base_config:
        base_config = create_base_experiment_config()
    
    if not evaluation_queries:
        evaluation_queries = DEFAULT_EVALUATION_QUERIES
    
    return ExperimentTemplate(
        name=name,
        description=description,
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        evaluation_queries=evaluation_queries,
        evaluation_metrics=["accuracy", "speed", "quality"],
        expected_runtime_hours=1.0
    )
"""
RAG Pipeline for RAG System
Complete pipeline integrating retrieval, prompt building, and language model generation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Union
import yaml

from .retriever import Retriever, RetrievalResult, create_retriever
from .prompt_builder import PromptBuilder, create_prompt_builder
from .llm_wrapper import LLMWrapper, create_llm_wrapper


class RAGPipeline:
    """Complete RAG pipeline with retrieval, prompt building, and generation."""
    
    def __init__(self, 
                 db_path: str,
                 embedding_model_path: str,
                 llm_model_path: str,
                 config_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the RAG pipeline.
        
        Args:
            db_path: Path to vector database
            embedding_model_path: Path to embedding model
            llm_model_path: Path to LLM model
            config_path: Path to model configuration file
            **kwargs: Additional configuration parameters
        """
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.config_path = config_path
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path, **kwargs)
        
        # Initialize components
        self.retriever = None
        self.llm_wrapper = None
        self.prompt_builder = None
        
        # Session state
        self.conversation_history = []
        self.current_corpus = None
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0,
            'total_tokens_generated': 0
        }
        
        self._initialize_components()
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict[str, Any]:
        """Load configuration from file and kwargs."""
        config = {}
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Override with kwargs
        config.update(kwargs)
        
        # Set defaults
        defaults = {
            'llm_params': {
                'n_ctx': 8192,
                'n_batch': 512,
                'n_threads': 8,
                'n_gpu_layers': -1,
                'temperature': 0.7,
                'top_p': 0.95,
                'max_tokens': 2048
            },
            'chat_template': {
                'system_prefix': '<bos>',
                'user_prefix': '<start_of_turn>user\n',
                'user_suffix': '<end_of_turn>\n',
                'assistant_prefix': '<start_of_turn>model\n',
                'assistant_suffix': '<end_of_turn>\n'
            },
            'retrieval': {
                'default_k': 5,
                'default_method': 'vector',
                'include_metadata': True
            }
        }
        
        # Merge defaults with loaded config
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config.get(key, {})}
        
        return config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing RAG pipeline components")
        
        try:
            # Create retriever
            self.logger.info("Creating retriever...")
            self.retriever = create_retriever(
                self.db_path, 
                self.embedding_model_path,
                embedding_dimension=384
            )
            
            # Create prompt builder
            self.logger.info("Creating prompt builder...")
            self.prompt_builder = create_prompt_builder(
                self.config.get('chat_template')
            )
            
            # Create LLM wrapper
            self.logger.info("Creating LLM wrapper...")
            self.llm_wrapper = create_llm_wrapper(
                self.llm_model_path,
                self.config.get('llm_params', {})
            )
            
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def query(self, 
              user_query: str,
              k: int = 5,
              retrieval_method: str = "vector",
              system_prompt: Optional[str] = None,
              include_metadata: bool = True,
              stream: bool = False,
              **generation_kwargs) -> Dict[str, Any]:
        """
        Execute a single RAG query.
        
        Args:
            user_query: User question or query
            k: Number of contexts to retrieve
            retrieval_method: Retrieval method ('vector', 'keyword', 'hybrid')
            system_prompt: Optional system instruction
            include_metadata: Whether to include source metadata
            stream: Whether to return streaming generator
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Use defaults from config if not specified
        k = k or self.config.get('retrieval', {}).get('default_k', 5)
        retrieval_method = retrieval_method or self.config.get('retrieval', {}).get('default_method', 'vector')
        include_metadata = include_metadata if include_metadata is not None else self.config.get('retrieval', {}).get('include_metadata', True)
        
        try:
            # Step 1: Retrieve relevant contexts
            retrieval_start = time.time()
            self.logger.info(f"Retrieving contexts for query: '{user_query[:50]}...'")
            
            contexts = self.retriever.retrieve(
                user_query, 
                k=k, 
                method=retrieval_method
            )
            
            retrieval_time = time.time() - retrieval_start
            self.logger.info(f"Retrieved {len(contexts)} contexts in {retrieval_time:.3f}s")
            
            # Step 2: Build prompt
            prompt_start = time.time()
            
            # Check if we need to truncate contexts to fit
            max_context = self.llm_wrapper.n_ctx
            contexts, prompt = self.prompt_builder.truncate_contexts_to_fit(
                user_query,
                contexts,
                max_context,
                system_prompt,
                generation_buffer=generation_kwargs.get('max_tokens', 2048)
            )
            
            prompt_time = time.time() - prompt_start
            prompt_tokens = self.prompt_builder.count_prompt_tokens(prompt)
            
            self.logger.info(f"Built prompt with {len(contexts)} contexts ({prompt_tokens} tokens) in {prompt_time:.3f}s")
            
            # Step 3: Generate response
            generation_start = time.time()
            
            if stream:
                # Return streaming generator and metadata
                generator, get_stats = self.llm_wrapper.generate_stream_with_stats(
                    prompt, **generation_kwargs
                )
                
                def streaming_response():
                    for token in generator:
                        yield token
                
                # Return generator and metadata function
                return {
                    'generator': streaming_response(),
                    'get_final_stats': get_stats,
                    'contexts': contexts,
                    'prompt_tokens': prompt_tokens,
                    'retrieval_time': retrieval_time,
                    'query': user_query,
                    'method': retrieval_method
                }
            
            else:
                # Non-streaming generation
                result = self.llm_wrapper.generate_with_stats(
                    prompt, **generation_kwargs
                )
                
                generation_time = result['generation_time']
                
                # Update pipeline stats
                self._update_stats(retrieval_time, generation_time, result['output_tokens'])
                
                total_time = time.time() - start_time
                
                # Format sources
                sources = self._format_sources(contexts)
                
                response = {
                    'answer': result['generated_text'],
                    'sources': sources,
                    'contexts': [ctx.to_dict() for ctx in contexts],
                    'metadata': {
                        'query': user_query,
                        'retrieval_method': retrieval_method,
                        'contexts_count': len(contexts),
                        'prompt_tokens': result['prompt_tokens'],
                        'output_tokens': result['output_tokens'],
                        'total_tokens': result['total_tokens'],
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': total_time,
                        'tokens_per_second': result['tokens_per_second'],
                        'context_remaining': result['context_remaining']
                    }
                }
                
                self.logger.info(f"Generated response ({result['output_tokens']} tokens) in {generation_time:.3f}s")
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error during RAG query: {e}")
            raise
    
    def query_stream(self, 
                    user_query: str,
                    k: int = 5,
                    retrieval_method: str = "vector",
                    system_prompt: Optional[str] = None,
                    include_metadata: bool = True,
                    **generation_kwargs) -> tuple:
        """
        Execute streaming RAG query.
        
        Args:
            user_query: User question or query
            k: Number of contexts to retrieve
            retrieval_method: Retrieval method
            system_prompt: Optional system instruction
            include_metadata: Whether to include source metadata
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (token_generator, metadata_dict)
        """
        result = self.query(
            user_query,
            k=k,
            retrieval_method=retrieval_method,
            system_prompt=system_prompt,
            include_metadata=include_metadata,
            stream=True,
            **generation_kwargs
        )
        
        return result['generator'], result
    
    def chat(self,
             user_query: str,
             k: int = 5,
             retrieval_method: str = "vector",
             system_prompt: Optional[str] = None,
             use_history: bool = True,
             **generation_kwargs) -> Dict[str, Any]:
        """
        Execute RAG query with conversation history.
        
        Args:
            user_query: User question or query
            k: Number of contexts to retrieve
            retrieval_method: Retrieval method
            system_prompt: Optional system instruction
            use_history: Whether to include conversation history
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Add user message to history
        self.conversation_history.append({'role': 'user', 'content': user_query})
        
        if use_history and len(self.conversation_history) > 1:
            # Multi-turn conversation with context
            retrieval_start = time.time()
            
            # Retrieve contexts for current query
            contexts = self.retriever.retrieve(
                user_query,
                k=k,
                method=retrieval_method
            )
            
            retrieval_time = time.time() - retrieval_start
            
            # Build conversation prompt with contexts
            max_context = self.llm_wrapper.n_ctx
            contexts, prompt = self.prompt_builder.truncate_contexts_to_fit(
                user_query,
                contexts,
                max_context,
                system_prompt,
                generation_buffer=generation_kwargs.get('max_tokens', 2048)
            )
            
            # Generate using conversation prompt
            result = self.llm_wrapper.generate_with_stats(
                prompt, **generation_kwargs
            )
            
            # Format response
            sources = self._format_sources(contexts)
            response_text = result['generated_text']
            
        else:
            # Single-turn query
            response_dict = self.query(
                user_query,
                k=k,
                retrieval_method=retrieval_method,
                system_prompt=system_prompt,
                **generation_kwargs
            )
            
            response_text = response_dict['answer']
            sources = response_dict['sources']
            contexts = [RetrievalResult(**ctx) for ctx in response_dict['contexts']]
            result = response_dict['metadata']
            retrieval_time = result['retrieval_time']
        
        # Add assistant response to history
        self.conversation_history.append({'role': 'assistant', 'content': response_text})
        
        return {
            'answer': response_text,
            'sources': sources,
            'contexts': [ctx.to_dict() for ctx in contexts],
            'conversation_length': len(self.conversation_history),
            'metadata': {
                'query': user_query,
                'retrieval_method': retrieval_method,
                'contexts_count': len(contexts),
                'retrieval_time': retrieval_time,
                'generation_time': result.get('generation_time', 0),
                'tokens_per_second': result.get('tokens_per_second', 0)
            }
        }
    
    def _format_sources(self, contexts: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Format contexts into source references."""
        sources = []
        
        for ctx in contexts:
            metadata = ctx.metadata
            source_info = {
                'chunk_id': ctx.chunk_id,
                'score': ctx.score,
                'content_preview': ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
            }
            
            # Add source file info
            source_path = metadata.get('source', metadata.get('filename', 'Unknown'))
            if source_path and source_path != 'Unknown':
                source_info['filename'] = Path(source_path).name
                source_info['source_path'] = source_path
            
            # Add page/chunk info
            if 'page_number' in metadata:
                source_info['page_number'] = metadata['page_number']
            elif ctx.chunk_index is not None:
                source_info['chunk_number'] = ctx.chunk_index + 1
            
            sources.append(source_info)
        
        return sources
    
    def _update_stats(self, retrieval_time: float, generation_time: float, tokens_generated: int):
        """Update pipeline statistics."""
        self.stats['total_queries'] += 1
        self.stats['total_retrieval_time'] += retrieval_time
        self.stats['total_generation_time'] += generation_time
        self.stats['total_tokens_generated'] += tokens_generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        total_queries = self.stats['total_queries']
        
        if total_queries == 0:
            return {
                'total_queries': 0,
                'avg_retrieval_time': 0,
                'avg_generation_time': 0,
                'avg_total_time': 0,
                'avg_tokens_per_second': 0,
                'total_tokens_generated': 0
            }
        
        avg_retrieval_time = self.stats['total_retrieval_time'] / total_queries
        avg_generation_time = self.stats['total_generation_time'] / total_queries
        avg_total_time = avg_retrieval_time + avg_generation_time
        
        avg_tokens_per_second = (
            self.stats['total_tokens_generated'] / self.stats['total_generation_time']
            if self.stats['total_generation_time'] > 0 else 0
        )
        
        pipeline_stats = {
            'total_queries': total_queries,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_generation_time': avg_generation_time,
            'avg_total_time': avg_total_time,
            'avg_tokens_per_second': avg_tokens_per_second,
            'total_tokens_generated': self.stats['total_tokens_generated']
        }
        
        # Add component stats
        if self.retriever:
            pipeline_stats['retriever'] = self.retriever.get_stats()
        
        if self.llm_wrapper:
            pipeline_stats['llm'] = self.llm_wrapper.get_model_info()
        
        if self.prompt_builder:
            pipeline_stats['prompt_builder'] = self.prompt_builder.get_template_info()
        
        return pipeline_stats
    
    def reset_session(self):
        """Clear conversation history and reset session state."""
        self.conversation_history.clear()
        self.current_corpus = None
        self.logger.info("Session reset - conversation history cleared")
    
    def set_corpus(self, corpus_name: str):
        """Set the current document corpus for context."""
        self.current_corpus = corpus_name
        self.logger.info(f"Set corpus context to: {corpus_name}")
    
    def reload_retriever(self):
        """Reload the retriever component."""
        self.logger.info("Reloading retriever...")
        self.retriever = create_retriever(
            self.db_path, 
            self.embedding_model_path,
            embedding_dimension=384
        )
    
    def unload_model(self):
        """Unload the LLM to free memory."""
        if self.llm_wrapper:
            self.llm_wrapper.unload_model()
            self.logger.info("LLM model unloaded")
    
    def reload_model(self):
        """Reload the LLM model."""
        if self.llm_wrapper:
            self.llm_wrapper.reload_model()
            self.logger.info("LLM model reloaded")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive pipeline information."""
        return {
            'db_path': self.db_path,
            'embedding_model_path': self.embedding_model_path,
            'llm_model_path': self.llm_model_path,
            'config_path': self.config_path,
            'current_corpus': self.current_corpus,
            'conversation_length': len(self.conversation_history),
            'components_initialized': {
                'retriever': self.retriever is not None,
                'llm_wrapper': self.llm_wrapper is not None,
                'prompt_builder': self.prompt_builder is not None
            },
            'config': self.config
        }


def create_rag_pipeline(db_path: str, 
                       embedding_model_path: str,
                       llm_model_path: str,
                       config_path: Optional[str] = None,
                       **kwargs) -> RAGPipeline:
    """
    Factory function to create a RAGPipeline instance.
    
    Args:
        db_path: Path to vector database
        embedding_model_path: Path to embedding model
        llm_model_path: Path to LLM model
        config_path: Path to model configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(
        db_path=db_path,
        embedding_model_path=embedding_model_path,
        llm_model_path=llm_model_path,
        config_path=config_path,
        **kwargs
    )
"""
Prompt Builder for RAG System
Constructs prompts using Gemma-3 chat templates with retrieved contexts.
"""

import logging
from typing import List, Dict, Any, Optional
import tiktoken

from .retriever import RetrievalResult


class PromptBuilder:
    """Builds prompts for RAG system using Gemma-3 chat templates."""
    
    def __init__(self, chat_template: Optional[Dict[str, str]] = None,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the prompt builder.
        
        Args:
            chat_template: Chat template configuration from config
            encoding_name: Tokenizer encoding name for token counting
        """
        # ✅ FIX: Remove BOS token from system_prefix to avoid duplicates
        # The LLM wrapper will add BOS token automatically
        self.chat_template = chat_template or {
            'system_prefix': '',  # ✅ FIX: Remove '<bos>' to prevent duplicates
            'user_prefix': '<start_of_turn>user\n',
            'user_suffix': '<end_of_turn>\n',
            'assistant_prefix': '<start_of_turn>model\n',
            'assistant_suffix': '<end_of_turn>\n'
        }
        
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.logger = logging.getLogger(__name__)
    
    def build_rag_prompt(self, query: str, 
                        retrieved_contexts: List[RetrievalResult],
                        system_prompt: Optional[str] = None,
                        include_metadata: bool = True) -> str:
        """
        Build RAG prompt with retrieved contexts.
        
        Args:
            query: User query string
            retrieved_contexts: List of retrieved context chunks
            system_prompt: Optional system instruction
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted prompt string ready for LLM
        """
        prompt_parts = []
        
        # Add system prefix
        prompt_parts.append(self.chat_template['system_prefix'])
        
        # Add system prompt if provided
        if system_prompt:
            prompt_parts.append(system_prompt + '\n')
        
        # Start user turn
        prompt_parts.append(self.chat_template['user_prefix'])
        
        # Add context information (NO SANITIZATION - intentionally unsafe)
        if retrieved_contexts:
            prompt_parts.append("Context information:")
            
            for i, result in enumerate(retrieved_contexts, 1):
                if include_metadata:
                    # Format with source metadata
                    source_info = self._format_source_metadata(result)
                    context_section = f"\n[Context {i} - {source_info}]\n{result.content}\n"
                else:
                    # Just the content
                    context_section = f"\n[Context {i}]\n{result.content}\n"
                
                # DIRECT INJECTION - NO VALIDATION OR SANITIZATION
                prompt_parts.append(context_section)
            
            prompt_parts.append("\n")
        
        # Add user query (NO SANITIZATION)
        prompt_parts.append(f"Question: {query}")
        
        # End user turn and start model turn
        prompt_parts.append(self.chat_template['user_suffix'])
        prompt_parts.append(self.chat_template['assistant_prefix'])
        
        full_prompt = "".join(prompt_parts)
        
        self.logger.debug(f"Built RAG prompt with {len(retrieved_contexts)} contexts")
        
        return full_prompt
    
    def build_simple_prompt(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Build simple prompt without RAG context.
        
        Args:
            query: User query string
            system_prompt: Optional system instruction
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system prefix
        prompt_parts.append(self.chat_template['system_prefix'])
        
        # Add system prompt if provided
        if system_prompt:
            prompt_parts.append(system_prompt + '\n')
        
        # Start user turn
        prompt_parts.append(self.chat_template['user_prefix'])
        
        # Add user query (NO SANITIZATION)
        prompt_parts.append(query)
        
        # End user turn and start model turn
        prompt_parts.append(self.chat_template['user_suffix'])
        prompt_parts.append(self.chat_template['assistant_prefix'])
        
        return "".join(prompt_parts)
    
    def build_conversation_prompt(self, messages: List[Dict[str, str]],
                                 retrieved_contexts: Optional[List[RetrievalResult]] = None) -> str:
        """
        Build multi-turn conversation prompt.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            retrieved_contexts: Optional context for current query
            
        Returns:
            Formatted conversation prompt
        """
        prompt_parts = []
        
        # Add system prefix
        prompt_parts.append(self.chat_template['system_prefix'])
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                # System message
                prompt_parts.append(content + '\n')
                
            elif role == 'user':
                # User message
                prompt_parts.append(self.chat_template['user_prefix'])
                
                # Add context for the latest user message if provided
                if retrieved_contexts and msg == messages[-1]:
                    prompt_parts.append("Context information:")
                    for i, result in enumerate(retrieved_contexts, 1):
                        source_info = self._format_source_metadata(result)
                        context_section = f"\n[Context {i} - {source_info}]\n{result.content}\n"
                        # DIRECT INJECTION - NO VALIDATION
                        prompt_parts.append(context_section)
                    prompt_parts.append("\n")
                
                # NO SANITIZATION of user content
                prompt_parts.append(content)
                prompt_parts.append(self.chat_template['user_suffix'])
                
            elif role == 'assistant':
                # Assistant message
                prompt_parts.append(self.chat_template['assistant_prefix'])
                prompt_parts.append(content)
                prompt_parts.append(self.chat_template['assistant_suffix'])
        
        # Start new assistant turn if last message was from user
        if messages and messages[-1]['role'] == 'user':
            prompt_parts.append(self.chat_template['assistant_prefix'])
        
        return "".join(prompt_parts)
    
    def _format_source_metadata(self, result: RetrievalResult) -> str:
        """Format source metadata for context display."""
        metadata = result.metadata
        source_parts = []
        
        # Extract filename
        source_path = metadata.get('source', metadata.get('filename', 'Unknown'))
        if source_path and source_path != 'Unknown':
            from pathlib import Path
            filename = Path(source_path).name
            source_parts.append(f"Source: {filename}")
        
        # Add page/chunk info
        if 'page_number' in metadata:
            source_parts.append(f"Page: {metadata['page_number']}")
        elif result.chunk_index is not None:
            source_parts.append(f"Chunk: {result.chunk_index + 1}")
        
        # Add relevance score
        source_parts.append(f"Score: {result.score:.3f}")
        
        return " | ".join(source_parts) if source_parts else "Unknown"
    
    def count_prompt_tokens(self, prompt: str) -> int:
        """
        Count tokens in prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(prompt))
    
    def analyze_prompt_structure(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt token distribution by section.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Dictionary with token counts by section
        """
        total_tokens = self.count_prompt_tokens(prompt)
        
        # Rough estimates - split by template markers
        context_start = prompt.find("Context information:")
        question_start = prompt.find("Question:")
        
        context_tokens = 0
        question_tokens = 0
        template_tokens = 0
        
        if context_start != -1 and question_start != -1:
            # Extract sections
            template_prefix = prompt[:context_start]
            context_section = prompt[context_start:question_start]
            question_section = prompt[question_start:]
            
            template_tokens = self.count_prompt_tokens(template_prefix)
            context_tokens = self.count_prompt_tokens(context_section)
            question_tokens = self.count_prompt_tokens(question_section)
        else:
            # Fallback - estimate template overhead
            template_tokens = total_tokens * 0.1  # Rough estimate
            question_tokens = total_tokens - template_tokens
        
        return {
            'total_tokens': total_tokens,
            'template_tokens': int(template_tokens),
            'context_tokens': int(context_tokens),
            'question_tokens': int(question_tokens),
            'context_percentage': (context_tokens / total_tokens * 100) if total_tokens > 0 else 0
        }
    
    def check_context_window_fit(self, prompt: str, max_context: int,
                                generation_buffer: int = 500) -> Dict[str, Any]:
        """
        Check if prompt fits in context window with generation buffer.
        
        Args:
            prompt: Prompt text to check
            max_context: Maximum context window size
            generation_buffer: Tokens to reserve for generation
            
        Returns:
            Dictionary with fit analysis
        """
        prompt_tokens = self.count_prompt_tokens(prompt)
        available_tokens = max_context - generation_buffer
        fits = prompt_tokens <= available_tokens
        
        return {
            'prompt_tokens': prompt_tokens,
            'max_context': max_context,
            'generation_buffer': generation_buffer,
            'available_tokens': available_tokens,
            'fits': fits,
            'overflow_tokens': max(0, prompt_tokens - available_tokens),
            'utilization_percent': (prompt_tokens / max_context * 100) if max_context > 0 else 0
        }
    
    def truncate_contexts_to_fit(self, query: str, 
                                retrieved_contexts: List[RetrievalResult],
                                max_context: int,
                                system_prompt: Optional[str] = None,
                                generation_buffer: int = 500) -> tuple:
        """
        Truncate contexts to fit within context window.
        
        Args:
            query: User query
            retrieved_contexts: List of context results
            max_context: Maximum context window size
            system_prompt: Optional system prompt
            generation_buffer: Tokens to reserve for generation
            
        Returns:
            Tuple of (truncated_contexts, final_prompt)
        """
        available_tokens = max_context - generation_buffer
        
        # Build base prompt without contexts to measure overhead
        base_prompt = self.build_rag_prompt(query, [], system_prompt, include_metadata=True)
        base_tokens = self.count_prompt_tokens(base_prompt)
        
        context_budget = available_tokens - base_tokens
        
        if context_budget <= 0:
            self.logger.warning("No tokens available for context - query too long")
            return [], self.build_rag_prompt(query, [], system_prompt, include_metadata=True)
        
        # Add contexts until budget exhausted
        included_contexts = []
        used_tokens = 0
        
        for result in retrieved_contexts:
            # Estimate tokens for this context with metadata
            context_text = f"[Context {len(included_contexts) + 1} - {self._format_source_metadata(result)}]\n{result.content}\n"
            context_tokens = self.count_prompt_tokens(context_text)
            
            if used_tokens + context_tokens <= context_budget:
                included_contexts.append(result)
                used_tokens += context_tokens
            else:
                # Try partial inclusion if this is the first context
                if not included_contexts and context_tokens > context_budget:
                    self.logger.warning("First context too large - truncating")
                    # Truncate content to fit
                    available_chars = int(context_budget * 3.5)  # Rough char/token ratio
                    truncated_content = result.content[:available_chars] + "..."
                    truncated_result = RetrievalResult(
                        chunk_id=result.chunk_id,
                        content=truncated_content,
                        score=result.score,
                        metadata=result.metadata,
                        doc_id=result.doc_id,
                        chunk_index=result.chunk_index
                    )
                    included_contexts.append(truncated_result)
                break
        
        final_prompt = self.build_rag_prompt(query, included_contexts, system_prompt, include_metadata=True)
        
        self.logger.info(f"Included {len(included_contexts)}/{len(retrieved_contexts)} contexts")
        
        return included_contexts, final_prompt
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the chat template."""
        return {
            'template_type': 'gemma-3',
            'system_prefix': self.chat_template['system_prefix'],
            'user_prefix': self.chat_template['user_prefix'],
            'user_suffix': self.chat_template['user_suffix'],
            'assistant_prefix': self.chat_template['assistant_prefix'],
            'assistant_suffix': self.chat_template['assistant_suffix'],
            'encoding': self.encoding.name
        }


def create_prompt_builder(chat_template: Optional[Dict[str, str]] = None) -> PromptBuilder:
    """
    Factory function to create a PromptBuilder instance.
    
    Args:
        chat_template: Optional chat template configuration
        
    Returns:
        Configured PromptBuilder instance
    """
    return PromptBuilder(chat_template)
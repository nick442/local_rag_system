"""
Query Reformulation for RAG System
Enhances retrieval through query expansion and reformulation.
"""

import logging
from typing import List, Dict, Any, Optional
import re

from .llm_wrapper import LLMWrapper


class QueryReformulator:
    """Reformulates queries to improve retrieval effectiveness."""
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        """
        Initialize query reformulator.
        
        Args:
            llm_wrapper: Optional LLM for query expansion
        """
        self.llm_wrapper = llm_wrapper
        self.logger = logging.getLogger(__name__)
    
    def expand_query_keywords(self, query: str) -> List[str]:
        """
        Generate keyword variations and synonyms for query.
        
        Args:
            query: Original query string
            
        Returns:
            List of query variations (including original)
        """
        variations = [query]  # Always include original
        
        # Simple keyword extraction and expansion
        keywords = self._extract_keywords(query)
        
        # Generate variations by:
        # 1. Adding synonyms
        # 2. Rephrasing
        # 3. Adding related terms
        
        for keyword in keywords:
            # Add common synonyms (simple rules-based)
            synonyms = self._get_simple_synonyms(keyword)
            for synonym in synonyms:
                variation = query.replace(keyword, synonym)
                if variation != query and variation not in variations:
                    variations.append(variation)
        
        # Add question variations
        if not query.strip().endswith('?'):
            variations.append(f"What is {query}?")
            variations.append(f"How does {query} work?")
            variations.append(f"Why {query}?")
        
        self.logger.debug(f"Expanded query into {len(variations)} variations")
        
        return variations
    
    def reformulate_with_llm(self, query: str, context_hint: Optional[str] = None) -> List[str]:
        """
        Use LLM to generate query reformulations.
        
        Args:
            query: Original query string
            context_hint: Optional domain context
            
        Returns:
            List of reformulated queries
        """
        if not self.llm_wrapper:
            # Fallback to keyword expansion
            return self.expand_query_keywords(query)
        
        # Build reformulation prompt (NO SANITIZATION)
        context_part = f" in the context of {context_hint}" if context_hint else ""
        
        reformulation_prompt = f"""<bos><start_of_turn>user
Generate 3 different ways to ask the following question{context_part}:

Original question: {query}

Provide variations that might find different relevant information:
1.
2.
3.
<end_of_turn>
<start_of_turn>model
1."""
        
        try:
            # Generate reformulations
            response = self.llm_wrapper.generate(
                reformulation_prompt,
                max_tokens=200,
                temperature=0.8,
                stop_sequences=['<end_of_turn>']
            )
            
            # Parse reformulations
            reformulations = self._parse_reformulations(response, query)
            
            self.logger.info(f"LLM generated {len(reformulations)} query reformulations")
            
            return reformulations
            
        except Exception as e:
            self.logger.warning(f"LLM reformulation failed: {e}")
            # Fallback to keyword expansion
            return self.expand_query_keywords(query)
    
    def generate_search_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple search variants for better coverage.
        
        Args:
            query: Original query
            num_variants: Number of variants to generate
            
        Returns:
            List of search variants
        """
        variants = [query]  # Original query first
        
        # Method 1: Keyword expansion
        keyword_variants = self.expand_query_keywords(query)
        variants.extend(keyword_variants[1:num_variants])  # Skip original
        
        # Method 2: Structural variations
        structural_variants = self._generate_structural_variants(query)
        variants.extend(structural_variants)
        
        # Method 3: Domain-specific expansions
        domain_variants = self._generate_domain_variants(query)
        variants.extend(domain_variants)
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant not in seen:
                unique_variants.append(variant)
                seen.add(variant)
        
        # Limit to requested number
        return unique_variants[:num_variants]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Filter out stop words
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this',
            'and', 'the', 'for', 'are', 'was', 'were', 'been', 'have', 'has',
            'can', 'could', 'will', 'would', 'should', 'does', 'did', 'with'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _get_simple_synonyms(self, word: str) -> List[str]:
        """Get simple synonyms for common words."""
        # Basic synonym mapping
        synonym_map = {
            'machine': ['artificial', 'automated', 'computer'],
            'learning': ['training', 'education', 'study'],
            'algorithm': ['method', 'approach', 'technique'],
            'model': ['framework', 'system', 'structure'],
            'data': ['information', 'dataset', 'records'],
            'analysis': ['examination', 'evaluation', 'assessment'],
            'process': ['procedure', 'method', 'workflow'],
            'system': ['framework', 'platform', 'architecture'],
            'method': ['approach', 'technique', 'strategy'],
            'technique': ['method', 'approach', 'procedure'],
            'neural': ['network', 'artificial'],
            'deep': ['advanced', 'complex'],
            'training': ['learning', 'education'],
            'classification': ['categorization', 'grouping'],
            'regression': ['prediction', 'modeling'],
            'optimization': ['improvement', 'enhancement']
        }
        
        return synonym_map.get(word.lower(), [])
    
    def _generate_structural_variants(self, query: str) -> List[str]:
        """Generate structural variations of the query."""
        variants = []
        
        # Convert questions to statements and vice versa
        if query.strip().endswith('?'):
            # Question to statement
            statement = query.rstrip('?')
            if statement.lower().startswith('what is'):
                variants.append(statement[8:])  # Remove "what is "
            elif statement.lower().startswith('how does'):
                variants.append(statement[9:] + ' explanation')  # Remove "how does "
        else:
            # Statement to question
            variants.append(f"What is {query}?")
            variants.append(f"Explain {query}")
        
        # Add imperative forms
        if not query.lower().startswith(('explain', 'describe', 'define')):
            variants.append(f"Explain {query}")
            variants.append(f"Describe {query}")
            variants.append(f"Define {query}")
        
        return variants[:3]  # Limit variants
    
    def _generate_domain_variants(self, query: str) -> List[str]:
        """Generate domain-specific query variants."""
        variants = []
        
        # Detect potential domains
        ml_terms = ['learning', 'model', 'algorithm', 'neural', 'training', 'data']
        tech_terms = ['system', 'software', 'code', 'programming', 'computer']
        science_terms = ['method', 'analysis', 'research', 'study', 'experiment']
        
        query_lower = query.lower()
        
        # Add domain-specific context
        if any(term in query_lower for term in ml_terms):
            variants.extend([
                f"{query} in machine learning",
                f"{query} algorithm",
                f"{query} model"
            ])
        
        if any(term in query_lower for term in tech_terms):
            variants.extend([
                f"{query} implementation",
                f"{query} technology",
                f"{query} development"
            ])
        
        if any(term in query_lower for term in science_terms):
            variants.extend([
                f"{query} methodology",
                f"{query} approach",
                f"{query} technique"
            ])
        
        return variants[:3]  # Limit variants
    
    def _parse_reformulations(self, response: str, original_query: str) -> List[str]:
        """Parse LLM response to extract reformulations."""
        reformulations = [original_query]  # Always include original
        
        # Split by numbers or bullet points
        lines = response.strip().split('\n')
        
        for line in lines:
            # Clean up line
            cleaned = line.strip()
            
            # Remove numbering (1., 2., 3., -, *, etc.)
            cleaned = re.sub(r'^[\d\.\-\*\s]+', '', cleaned)
            
            # Skip empty lines
            if not cleaned:
                continue
            
            # Skip if same as original (case insensitive)
            if cleaned.lower() == original_query.lower():
                continue
            
            # Skip if too similar to original
            if self._similarity_ratio(cleaned.lower(), original_query.lower()) > 0.8:
                continue
            
            # Add if it looks like a valid question/statement
            if len(cleaned) > 10 and len(cleaned) < 200:
                reformulations.append(cleaned)
        
        return reformulations[:4]  # Original + 3 reformulations
    
    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate simple similarity ratio between two strings."""
        # Very basic similarity - count common words
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return common / total if total > 0 else 0.0
    
    def multi_strategy_reformulation(self, query: str, 
                                   use_llm: bool = True,
                                   include_keywords: bool = True,
                                   include_structural: bool = True,
                                   max_variants: int = 5) -> List[str]:
        """
        Apply multiple reformulation strategies.
        
        Args:
            query: Original query
            use_llm: Whether to use LLM reformulation
            include_keywords: Whether to include keyword variants
            include_structural: Whether to include structural variants
            max_variants: Maximum number of variants to return
            
        Returns:
            List of query variants
        """
        all_variants = [query]  # Original first
        
        if use_llm and self.llm_wrapper:
            llm_variants = self.reformulate_with_llm(query)
            all_variants.extend(llm_variants[1:])  # Skip original
        
        if include_keywords:
            keyword_variants = self.expand_query_keywords(query)
            all_variants.extend(keyword_variants[1:])  # Skip original
        
        if include_structural:
            structural_variants = self._generate_structural_variants(query)
            all_variants.extend(structural_variants)
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in all_variants:
            variant_lower = variant.lower().strip()
            if variant_lower not in seen:
                unique_variants.append(variant)
                seen.add(variant_lower)
        
        # Limit to max variants
        final_variants = unique_variants[:max_variants]
        
        self.logger.info(f"Generated {len(final_variants)} query variants using multiple strategies")
        
        return final_variants


def create_query_reformulator(llm_wrapper: Optional[LLMWrapper] = None) -> QueryReformulator:
    """
    Factory function to create a QueryReformulator instance.
    
    Args:
        llm_wrapper: Optional LLM wrapper for advanced reformulation
        
    Returns:
        Configured QueryReformulator instance
    """
    return QueryReformulator(llm_wrapper)
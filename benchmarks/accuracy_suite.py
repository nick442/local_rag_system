#!/usr/bin/env python3
"""
Accuracy Evaluation Framework for RAG System

RAG accuracy measurements:
1. Retrieval relevance scoring
2. Answer correctness (if ground truth available)
3. Context utilization analysis
4. Hallucination detection
"""

import re
import json
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline

@dataclass
class EvaluationQuery:
    """Container for evaluation query with expected results"""
    query: str
    expected_topics: List[str] = None
    ground_truth_answer: str = None
    expected_sources: List[str] = None
    difficulty: str = "medium"

@dataclass
class AccuracyResult:
    """Container for accuracy evaluation results"""
    name: str
    metrics: Dict[str, Any]
    detailed_results: List[Dict[str, Any]]
    timestamp: str

class AccuracyBenchmark:
    """Accuracy evaluation suite for RAG pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize accuracy benchmark suite"""
        self.logger = logging.getLogger(__name__)
        self.results: List[AccuracyResult] = []
        
        # Default paths from main.py
        DEFAULT_DB_PATH = "data/rag_vectors.db"
        DEFAULT_EMBEDDING_PATH = "models/embeddings/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        DEFAULT_LLM_PATH = "models/gemma-3-4b-it-q4_0.gguf"
        
        # Initialize RAG pipeline
        try:
            self.rag = RAGPipeline(
                db_path=DEFAULT_DB_PATH,
                embedding_model_path=DEFAULT_EMBEDDING_PATH,
                llm_model_path=DEFAULT_LLM_PATH,
                config_path=config_path
            )
            self.logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            self.rag = None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'are', 'for', 'not', 'but', 'with', 'this', 'that', 
            'can', 'you', 'all', 'have', 'has', 'had', 'was', 'were', 'been',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'from', 'into', 'over', 'under', 'above', 'below', 'through'
        }
        
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))  # Remove duplicates
    
    def evaluate_retrieval_relevance(self, queries: List[EvaluationQuery], 
                                   k_values: List[int] = [1, 5, 10]) -> AccuracyResult:
        """
        Evaluate retrieval relevance using various metrics
        
        Args:
            queries: List of evaluation queries with expected topics
            k_values: Different k values to evaluate
            
        Returns:
            AccuracyResult with retrieval relevance metrics
        """
        if not self.rag:
            raise RuntimeError("RAG pipeline not initialized")
            
        self.logger.info("Evaluating retrieval relevance")
        
        results_by_k = {}
        detailed_results = []
        
        for k in k_values:
            k_results = []
            
            for i, eval_query in enumerate(queries):
                try:
                    # Get RAG response with sources
                    response = self.rag.query(eval_query.query, k=k)
                    
                    if isinstance(response, dict):
                        sources = response.get('sources', [])
                        response_text = response.get('response', '')
                    else:
                        sources = []
                        response_text = str(response)
                    
                    # Extract source texts for relevance evaluation
                    source_texts = []
                    for source in sources:
                        if isinstance(source, dict):
                            source_text = source.get('content', source.get('text', str(source)))
                        else:
                            source_text = str(source)
                        source_texts.append(source_text)
                    
                    # Calculate relevance metrics
                    relevance_scores = []
                    topic_matches = []
                    
                    if eval_query.expected_topics:
                        for source_text in source_texts:
                            # Check topic relevance
                            topic_score = 0
                            matched_topics = []
                            
                            for topic in eval_query.expected_topics:
                                if topic.lower() in source_text.lower():
                                    topic_score += 1
                                    matched_topics.append(topic)
                            
                            topic_relevance = topic_score / len(eval_query.expected_topics)
                            relevance_scores.append(topic_relevance)
                            topic_matches.append(matched_topics)
                        
                        # Calculate precision@k (relevant items / retrieved items)
                        relevant_count = sum(1 for score in relevance_scores if score > 0)
                        precision_at_k = relevant_count / k if k > 0 else 0
                        
                        # Calculate average relevance score
                        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
                    else:
                        # Fallback: use query-source similarity
                        for source_text in source_texts:
                            similarity = self._calculate_text_similarity(eval_query.query, source_text)
                            relevance_scores.append(similarity)
                        
                        relevant_count = sum(1 for score in relevance_scores if score > 0.1)  # Threshold 0.1
                        precision_at_k = relevant_count / k if k > 0 else 0
                        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
                    
                    query_result = {
                        'query_id': i,
                        'query': eval_query.query,
                        'k': k,
                        'retrieved_count': len(sources),
                        'precision_at_k': precision_at_k,
                        'avg_relevance_score': avg_relevance,
                        'relevance_scores': relevance_scores,
                        'expected_topics': eval_query.expected_topics or [],
                        'source_count': len(sources)
                    }
                    
                    k_results.append(query_result)
                    detailed_results.append(query_result)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating query '{eval_query.query}' with k={k}: {e}")
                    error_result = {
                        'query_id': i,
                        'query': eval_query.query,
                        'k': k,
                        'error': str(e),
                        'precision_at_k': 0,
                        'avg_relevance_score': 0
                    }
                    k_results.append(error_result)
                    detailed_results.append(error_result)
            
            # Calculate aggregate metrics for this k value
            valid_results = [r for r in k_results if 'error' not in r]
            if valid_results:
                precisions = [r['precision_at_k'] for r in valid_results]
                relevance_scores = [r['avg_relevance_score'] for r in valid_results]
                
                results_by_k[f'k_{k}'] = {
                    'precision_at_k': {
                        'mean': statistics.mean(precisions),
                        'median': statistics.median(precisions),
                        'std': statistics.stdev(precisions) if len(precisions) > 1 else 0,
                        'min': min(precisions),
                        'max': max(precisions)
                    },
                    'relevance_score': {
                        'mean': statistics.mean(relevance_scores),
                        'median': statistics.median(relevance_scores),
                        'std': statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0,
                        'min': min(relevance_scores),
                        'max': max(relevance_scores)
                    },
                    'total_queries': len(queries),
                    'successful_queries': len(valid_results)
                }
            else:
                results_by_k[f'k_{k}'] = {
                    'error': 'No valid results for this k value',
                    'total_queries': len(queries),
                    'successful_queries': 0
                }
        
        return AccuracyResult(
            name="retrieval_relevance",
            metrics=results_by_k,
            detailed_results=detailed_results,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def evaluate_answer_quality(self, queries: List[EvaluationQuery]) -> AccuracyResult:
        """
        Evaluate answer quality without ground truth
        
        Args:
            queries: List of evaluation queries
            
        Returns:
            AccuracyResult with answer quality metrics
        """
        if not self.rag:
            raise RuntimeError("RAG pipeline not initialized")
            
        self.logger.info("Evaluating answer quality")
        
        detailed_results = []
        quality_metrics = []
        
        for i, eval_query in enumerate(queries):
            try:
                # Get RAG response
                response = self.rag.query(eval_query.query, k=5)
                
                if isinstance(response, dict):
                    answer = response.get('response', '')
                    sources = response.get('sources', [])
                else:
                    answer = str(response)
                    sources = []
                
                # Quality metrics
                word_count = len(answer.split())
                sentence_count = len([s for s in answer.split('.') if s.strip()])
                
                # Check for question answering patterns
                has_direct_answer = any(pattern in answer.lower() for pattern in [
                    'is', 'are', 'means', 'refers to', 'defined as', 'consists of'
                ])
                
                # Check for context grounding (answer uses retrieved information)
                context_grounding_score = 0
                if sources:
                    answer_keywords = set(self._extract_keywords(answer))
                    context_keywords = set()
                    
                    for source in sources:
                        source_text = source.get('content', str(source)) if isinstance(source, dict) else str(source)
                        context_keywords.update(self._extract_keywords(source_text))
                    
                    if answer_keywords and context_keywords:
                        overlap = len(answer_keywords.intersection(context_keywords))
                        context_grounding_score = overlap / len(answer_keywords) if answer_keywords else 0
                
                # Check coherence (basic structural indicators)
                coherence_score = 0
                if word_count >= 10:  # Minimum length
                    coherence_score += 0.3
                if sentence_count >= 2:  # Multiple sentences
                    coherence_score += 0.3
                if has_direct_answer:  # Contains answer patterns
                    coherence_score += 0.4
                
                # Length appropriateness (penalize too short or too long)
                length_score = 1.0
                if word_count < 5:  # Too short
                    length_score = 0.2
                elif word_count > 500:  # Too long
                    length_score = 0.7
                
                query_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'answer': answer[:200] + "..." if len(answer) > 200 else answer,  # Truncate for storage
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'context_grounding_score': context_grounding_score,
                    'coherence_score': coherence_score,
                    'length_score': length_score,
                    'has_direct_answer': has_direct_answer,
                    'sources_count': len(sources),
                    'difficulty': eval_query.difficulty
                }
                
                # Overall quality score (weighted average)
                overall_quality = (
                    context_grounding_score * 0.4 +
                    coherence_score * 0.4 +
                    length_score * 0.2
                )
                query_result['overall_quality_score'] = overall_quality
                
                detailed_results.append(query_result)
                quality_metrics.append(overall_quality)
                
            except Exception as e:
                self.logger.error(f"Error evaluating answer quality for query '{eval_query.query}': {e}")
                error_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'error': str(e),
                    'overall_quality_score': 0
                }
                detailed_results.append(error_result)
                quality_metrics.append(0)
        
        # Calculate aggregate metrics
        valid_metrics = [m for m in quality_metrics if m > 0]
        
        if valid_metrics:
            aggregate_metrics = {
                'overall_quality': {
                    'mean': statistics.mean(valid_metrics),
                    'median': statistics.median(valid_metrics),
                    'std': statistics.stdev(valid_metrics) if len(valid_metrics) > 1 else 0,
                    'min': min(valid_metrics),
                    'max': max(valid_metrics)
                },
                'answer_statistics': {
                    'avg_word_count': statistics.mean([r.get('word_count', 0) for r in detailed_results if 'word_count' in r]),
                    'avg_context_grounding': statistics.mean([r.get('context_grounding_score', 0) for r in detailed_results if 'context_grounding_score' in r]),
                    'coherence_rate': sum(1 for r in detailed_results if r.get('coherence_score', 0) > 0.5) / len(detailed_results)
                },
                'total_queries': len(queries),
                'successful_queries': len(valid_metrics)
            }
        else:
            aggregate_metrics = {
                'error': 'No valid quality metrics obtained',
                'total_queries': len(queries),
                'successful_queries': 0
            }
        
        return AccuracyResult(
            name="answer_quality",
            metrics=aggregate_metrics,
            detailed_results=detailed_results,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def detect_hallucinations(self, queries: List[EvaluationQuery]) -> AccuracyResult:
        """
        Detect potential hallucinations by checking if response claims are supported by retrieved contexts
        
        Args:
            queries: List of evaluation queries
            
        Returns:
            AccuracyResult with hallucination detection metrics
        """
        if not self.rag:
            raise RuntimeError("RAG pipeline not initialized")
            
        self.logger.info("Detecting potential hallucinations")
        
        detailed_results = []
        hallucination_flags = []
        
        for i, eval_query in enumerate(queries):
            try:
                # Get RAG response with sources
                response = self.rag.query(eval_query.query, k=5)
                
                if isinstance(response, dict):
                    answer = response.get('response', '')
                    sources = response.get('sources', [])
                else:
                    answer = str(response)
                    sources = []
                
                # Extract claims from the answer (simple approach: sentences with factual patterns)
                factual_patterns = [
                    r'is (a|an|the) \w+',
                    r'\w+ (means|refers to|is defined as)',
                    r'(contains|includes|consists of)',
                    r'\d+\s*(percent|%|years|times)',
                    r'(was invented|was created|was developed)',
                    r'(located in|found in|occurs in)'
                ]
                
                claims = []
                for pattern in factual_patterns:
                    matches = re.findall(pattern, answer.lower())
                    claims.extend(matches)
                
                # Check if claims are supported by sources
                unsupported_claims = 0
                supported_claims = 0
                
                if sources and claims:
                    # Combine all source texts
                    all_source_text = ' '.join([
                        (source.get('content', str(source)) if isinstance(source, dict) else str(source))
                        for source in sources
                    ]).lower()
                    
                    for claim in claims:
                        claim_text = ' '.join(claim) if isinstance(claim, tuple) else str(claim)
                        # Simple check: is the claim text present in sources?
                        if claim_text.lower() in all_source_text:
                            supported_claims += 1
                        else:
                            unsupported_claims += 1
                
                # Calculate hallucination indicators
                total_claims = supported_claims + unsupported_claims
                hallucination_ratio = unsupported_claims / total_claims if total_claims > 0 else 0
                
                # Additional checks
                answer_keywords = set(self._extract_keywords(answer))
                source_keywords = set()
                
                if sources:
                    for source in sources:
                        source_text = source.get('content', str(source)) if isinstance(source, dict) else str(source)
                        source_keywords.update(self._extract_keywords(source_text))
                
                # Check for keywords in answer that don't appear in sources
                unique_answer_keywords = answer_keywords - source_keywords
                keyword_hallucination_ratio = len(unique_answer_keywords) / len(answer_keywords) if answer_keywords else 0
                
                # Hallucination risk score (combining multiple indicators)
                risk_factors = []
                
                if hallucination_ratio > 0.3:  # High claim mismatch
                    risk_factors.append("high_claim_mismatch")
                if keyword_hallucination_ratio > 0.7:  # Many unique keywords
                    risk_factors.append("high_keyword_mismatch")
                if not sources:  # No sources available
                    risk_factors.append("no_sources")
                if len(answer.split()) < 10:  # Very short answer
                    risk_factors.append("short_answer")
                
                hallucination_risk = len(risk_factors) / 4  # Normalize to 0-1
                
                query_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'total_claims': total_claims,
                    'supported_claims': supported_claims,
                    'unsupported_claims': unsupported_claims,
                    'hallucination_ratio': hallucination_ratio,
                    'keyword_hallucination_ratio': keyword_hallucination_ratio,
                    'hallucination_risk': hallucination_risk,
                    'risk_factors': risk_factors,
                    'sources_count': len(sources),
                    'answer_length': len(answer.split())
                }
                
                detailed_results.append(query_result)
                hallucination_flags.append(hallucination_risk > 0.5)  # Flag high risk
                
            except Exception as e:
                self.logger.error(f"Error detecting hallucinations for query '{eval_query.query}': {e}")
                error_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'error': str(e),
                    'hallucination_risk': 1.0  # Assume high risk on error
                }
                detailed_results.append(error_result)
                hallucination_flags.append(True)
        
        # Calculate aggregate metrics
        valid_results = [r for r in detailed_results if 'error' not in r]
        
        if valid_results:
            risk_scores = [r['hallucination_risk'] for r in valid_results]
            claim_ratios = [r['hallucination_ratio'] for r in valid_results if r['total_claims'] > 0]
            
            aggregate_metrics = {
                'hallucination_risk': {
                    'mean': statistics.mean(risk_scores),
                    'median': statistics.median(risk_scores),
                    'std': statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0,
                    'high_risk_queries': sum(1 for r in risk_scores if r > 0.5)
                },
                'claim_analysis': {
                    'avg_claims_per_answer': statistics.mean([r['total_claims'] for r in valid_results]),
                    'avg_hallucination_ratio': statistics.mean(claim_ratios) if claim_ratios else 0,
                    'answers_with_claims': sum(1 for r in valid_results if r['total_claims'] > 0)
                },
                'total_queries': len(queries),
                'successful_queries': len(valid_results),
                'flagged_queries': sum(hallucination_flags)
            }
        else:
            aggregate_metrics = {
                'error': 'No valid hallucination detection results',
                'total_queries': len(queries),
                'successful_queries': 0,
                'flagged_queries': sum(hallucination_flags)
            }
        
        return AccuracyResult(
            name="hallucination_detection",
            metrics=aggregate_metrics,
            detailed_results=detailed_results,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def analyze_context_usage(self, queries: List[EvaluationQuery]) -> AccuracyResult:
        """
        Analyze how effectively the system uses retrieved context
        
        Args:
            queries: List of evaluation queries
            
        Returns:
            AccuracyResult with context usage metrics
        """
        if not self.rag:
            raise RuntimeError("RAG pipeline not initialized")
            
        self.logger.info("Analyzing context usage")
        
        detailed_results = []
        usage_metrics = []
        
        for i, eval_query in enumerate(queries):
            try:
                # Get RAG response with sources
                response = self.rag.query(eval_query.query, k=5)
                
                if isinstance(response, dict):
                    answer = response.get('response', '')
                    sources = response.get('sources', [])
                else:
                    answer = str(response)
                    sources = []
                
                if not sources:
                    # No context available
                    query_result = {
                        'query_id': i,
                        'query': eval_query.query,
                        'context_usage_score': 0,
                        'sources_count': 0,
                        'answer_length': len(answer.split()),
                        'issue': 'no_sources_retrieved'
                    }
                    detailed_results.append(query_result)
                    usage_metrics.append(0)
                    continue
                
                # Extract keywords from answer and sources
                answer_keywords = set(self._extract_keywords(answer))
                
                source_usage_scores = []
                total_source_keywords = set()
                
                for j, source in enumerate(sources):
                    source_text = source.get('content', str(source)) if isinstance(source, dict) else str(source)
                    source_keywords = set(self._extract_keywords(source_text))
                    total_source_keywords.update(source_keywords)
                    
                    # Calculate how much of this source is used in the answer
                    if source_keywords and answer_keywords:
                        overlap = len(answer_keywords.intersection(source_keywords))
                        usage_score = overlap / len(source_keywords)
                        source_usage_scores.append(usage_score)
                    else:
                        source_usage_scores.append(0)
                
                # Overall context utilization
                if total_source_keywords and answer_keywords:
                    context_overlap = len(answer_keywords.intersection(total_source_keywords))
                    context_usage_score = context_overlap / len(total_source_keywords)
                else:
                    context_usage_score = 0
                
                # Calculate context coverage (how much of the answer is grounded in context)
                if answer_keywords and total_source_keywords:
                    answer_overlap = len(answer_keywords.intersection(total_source_keywords))
                    answer_coverage = answer_overlap / len(answer_keywords)
                else:
                    answer_coverage = 0
                
                # Identify potentially ignored sources (very low usage)
                ignored_sources = sum(1 for score in source_usage_scores if score < 0.1)
                
                # Calculate redundancy (overlapping sources)
                source_similarities = []
                for idx1, source1 in enumerate(sources):
                    for idx2, source2 in enumerate(sources[idx1+1:], idx1+1):
                        text1 = source1.get('content', str(source1)) if isinstance(source1, dict) else str(source1)
                        text2 = source2.get('content', str(source2)) if isinstance(source2, dict) else str(source2)
                        similarity = self._calculate_text_similarity(text1, text2)
                        source_similarities.append(similarity)
                
                avg_source_redundancy = statistics.mean(source_similarities) if source_similarities else 0
                
                query_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'context_usage_score': context_usage_score,
                    'answer_coverage': answer_coverage,
                    'sources_count': len(sources),
                    'ignored_sources': ignored_sources,
                    'avg_source_usage': statistics.mean(source_usage_scores) if source_usage_scores else 0,
                    'source_redundancy': avg_source_redundancy,
                    'answer_length': len(answer.split()),
                    'total_context_keywords': len(total_source_keywords),
                    'answer_keywords': len(answer_keywords)
                }
                
                detailed_results.append(query_result)
                usage_metrics.append(context_usage_score)
                
            except Exception as e:
                self.logger.error(f"Error analyzing context usage for query '{eval_query.query}': {e}")
                error_result = {
                    'query_id': i,
                    'query': eval_query.query,
                    'error': str(e),
                    'context_usage_score': 0
                }
                detailed_results.append(error_result)
                usage_metrics.append(0)
        
        # Calculate aggregate metrics
        valid_results = [r for r in detailed_results if 'error' not in r]
        
        if valid_results:
            usage_scores = [r['context_usage_score'] for r in valid_results]
            coverage_scores = [r['answer_coverage'] for r in valid_results if 'answer_coverage' in r]
            
            aggregate_metrics = {
                'context_usage': {
                    'mean': statistics.mean(usage_scores),
                    'median': statistics.median(usage_scores),
                    'std': statistics.stdev(usage_scores) if len(usage_scores) > 1 else 0,
                    'min': min(usage_scores),
                    'max': max(usage_scores)
                },
                'answer_coverage': {
                    'mean': statistics.mean(coverage_scores) if coverage_scores else 0,
                    'median': statistics.median(coverage_scores) if coverage_scores else 0,
                },
                'source_efficiency': {
                    'avg_sources_per_query': statistics.mean([r['sources_count'] for r in valid_results]),
                    'avg_ignored_sources': statistics.mean([r.get('ignored_sources', 0) for r in valid_results]),
                    'avg_redundancy': statistics.mean([r.get('source_redundancy', 0) for r in valid_results])
                },
                'total_queries': len(queries),
                'successful_queries': len(valid_results),
                'queries_with_sources': sum(1 for r in valid_results if r['sources_count'] > 0)
            }
        else:
            aggregate_metrics = {
                'error': 'No valid context usage analysis results',
                'total_queries': len(queries),
                'successful_queries': 0
            }
        
        return AccuracyResult(
            name="context_usage",
            metrics=aggregate_metrics,
            detailed_results=detailed_results,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def run_all_evaluations(self, queries: List[EvaluationQuery]) -> List[AccuracyResult]:
        """
        Run all accuracy evaluations
        
        Args:
            queries: List of evaluation queries
            
        Returns:
            List of all accuracy evaluation results
        """
        self.logger.info("Running all accuracy evaluations")
        self.results.clear()
        
        try:
            # Retrieval relevance evaluation
            self.logger.info("Running retrieval relevance evaluation...")
            relevance_result = self.evaluate_retrieval_relevance(queries)
            self.results.append(relevance_result)
            
            # Answer quality evaluation
            self.logger.info("Running answer quality evaluation...")
            quality_result = self.evaluate_answer_quality(queries)
            self.results.append(quality_result)
            
            # Hallucination detection
            self.logger.info("Running hallucination detection...")
            hallucination_result = self.detect_hallucinations(queries)
            self.results.append(hallucination_result)
            
            # Context usage analysis
            self.logger.info("Running context usage analysis...")
            context_result = self.analyze_context_usage(queries)
            self.results.append(context_result)
            
            self.logger.info("All accuracy evaluations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error running evaluations: {e}")
            raise
        
        return self.results
    
    def save_results(self, filepath: str) -> None:
        """Save accuracy evaluation results to file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'name': result.name,
                'metrics': result.metrics,
                'detailed_results': result.detailed_results,
                'timestamp': result.timestamp
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'evaluations': results_data
            }, f, indent=2)
        
        self.logger.info(f"Accuracy evaluation results saved to {filepath}")

def main():
    """Main function for running accuracy evaluations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Sample evaluation queries
    test_queries = [
        EvaluationQuery(
            query="What is machine learning?",
            expected_topics=["algorithms", "data", "training", "models"],
            difficulty="easy"
        ),
        EvaluationQuery(
            query="Explain the difference between supervised and unsupervised learning",
            expected_topics=["labels", "classification", "clustering", "training data"],
            difficulty="medium"
        ),
        EvaluationQuery(
            query="How do neural networks work and what are their applications?",
            expected_topics=["neurons", "weights", "layers", "backpropagation", "applications"],
            difficulty="medium"
        ),
        EvaluationQuery(
            query="What are the key components of a RAG system?",
            expected_topics=["retrieval", "generation", "embedding", "vector database"],
            difficulty="hard"
        )
    ]
    
    # Initialize accuracy benchmark suite
    benchmark = AccuracyBenchmark()
    
    # Run all evaluations
    results = benchmark.run_all_evaluations(test_queries)
    
    # Save results
    results_dir = Path("reports/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"accuracy_evaluation_{timestamp}.json"
    benchmark.save_results(str(results_file))
    
    # Print summary
    print("\n=== Accuracy Evaluation Results Summary ===")
    for result in results:
        print(f"\n{result.name.upper()}:")
        if 'error' not in result.metrics:
            if result.name == "retrieval_relevance" and 'k_5' in result.metrics:
                precision = result.metrics['k_5']['precision_at_k']['mean']
                print(f"  Precision@5: {precision:.3f}")
            elif result.name == "answer_quality" and 'overall_quality' in result.metrics:
                quality = result.metrics['overall_quality']['mean']
                print(f"  Quality Score: {quality:.3f}")
            elif result.name == "hallucination_detection" and 'hallucination_risk' in result.metrics:
                risk = result.metrics['hallucination_risk']['mean']
                flagged = result.metrics.get('flagged_queries', 0)
                print(f"  Avg Risk Score: {risk:.3f}")
                print(f"  Flagged Queries: {flagged}")
            elif result.name == "context_usage" and 'context_usage' in result.metrics:
                usage = result.metrics['context_usage']['mean']
                print(f"  Context Usage: {usage:.3f}")
        else:
            print(f"  Error: {result.metrics['error']}")

if __name__ == "__main__":
    main()
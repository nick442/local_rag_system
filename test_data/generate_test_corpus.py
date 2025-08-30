#!/usr/bin/env python3
"""
Test Corpus Generation for RAG System

Generate synthetic test documents:
1. Various document types
2. Known content for validation
3. Different sizes (small/medium/large)
4. Edge cases (empty, huge, special chars)
"""

import os
import json
import random
import string
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

class TestCorpusGenerator:
    """Generator for test corpus with various document types and sizes"""
    
    def __init__(self, base_path: str = None):
        """Initialize corpus generator"""
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path) if base_path else Path(__file__).parent / "corpora"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Document templates and content
        self.topics = [
            "machine learning", "artificial intelligence", "neural networks", 
            "deep learning", "natural language processing", "computer vision",
            "data science", "algorithms", "programming", "software engineering",
            "cloud computing", "cybersecurity", "blockchain", "quantum computing",
            "robotics", "automation", "internet of things", "big data"
        ]
        
        self.technical_terms = [
            "algorithm", "architecture", "framework", "implementation", "optimization",
            "scalability", "performance", "efficiency", "methodology", "infrastructure",
            "deployment", "integration", "configuration", "monitoring", "analysis"
        ]
    
    def _generate_technical_article(self, topic: str, target_words: int) -> str:
        """Generate a technical article about a specific topic"""
        
        # Article structure
        title = f"Understanding {topic.title()}: A Comprehensive Guide"
        
        introduction = f"""
{title}

{topic.title()} is a fundamental concept in modern technology that has revolutionized 
how we approach complex problems. This comprehensive guide explores the key principles, 
methodologies, and applications of {topic}.

Introduction

{topic.title()} represents a significant advancement in computational approaches, 
offering new perspectives on traditional challenges. The field has evolved rapidly, 
incorporating elements from various disciplines including mathematics, computer science, 
and domain-specific expertise.
"""
        
        # Generate sections
        sections = [
            f"Core Principles of {topic.title()}",
            f"Technical Architecture and {random.choice(self.technical_terms).title()}",
            f"Implementation Strategies for {topic.title()}",
            f"Performance {random.choice(self.technical_terms).title()} and Optimization",
            f"Real-world Applications and Use Cases",
            f"Future Directions in {topic.title()}"
        ]
        
        content_parts = [introduction.strip()]
        current_words = len(introduction.split())
        
        for section in sections:
            if current_words >= target_words:
                break
                
            section_content = self._generate_section_content(section, topic, 
                                                           min(200, target_words - current_words))
            content_parts.append(f"\n\n{section}\n\n{section_content}")
            current_words += len(section_content.split())
        
        # Add conclusion if space permits
        if current_words < target_words * 0.9:
            conclusion = f"""

Conclusion

{topic.title()} continues to evolve as a critical technology in our digital landscape. 
The principles and methodologies discussed in this guide provide a foundation for 
understanding and implementing {topic} solutions effectively. As the field advances, 
we can expect continued innovation and new applications across various industries.

Key takeaways include the importance of {random.choice(self.technical_terms)}, 
the role of {random.choice(self.technical_terms)} in system design, and the 
significance of {random.choice(self.technical_terms)} for optimal performance.
"""
            content_parts.append(conclusion)
        
        return "\n".join(content_parts)
    
    def _generate_section_content(self, section_title: str, topic: str, target_words: int) -> str:
        """Generate content for a specific section"""
        
        content_templates = [
            f"The {section_title.lower()} involves several key components that work together to achieve optimal results. Primary considerations include {random.choice(self.technical_terms)}, {random.choice(self.technical_terms)}, and effective {random.choice(self.technical_terms)} strategies.",
            
            f"When implementing {topic}, it's essential to understand the underlying {random.choice(self.technical_terms)} and how they impact overall system performance. Research has shown that proper {random.choice(self.technical_terms)} can improve efficiency by up to {random.randint(20, 80)}%.",
            
            f"Modern approaches to {topic} emphasize {random.choice(self.technical_terms)} and {random.choice(self.technical_terms)} as core principles. These methodologies enable organizations to leverage advanced {random.choice(self.technical_terms)} while maintaining scalable {random.choice(self.technical_terms)}.",
            
            f"The {section_title.lower()} requires careful consideration of various factors including resource allocation, {random.choice(self.technical_terms)} requirements, and long-term {random.choice(self.technical_terms)} goals. Best practices recommend iterative {random.choice(self.technical_terms)} and continuous monitoring."
        ]
        
        # Build content to reach target word count
        content_parts = []
        current_words = 0
        
        while current_words < target_words and len(content_parts) < 5:
            template = random.choice(content_templates)
            
            # Add some variation
            if random.random() > 0.5:
                template += f" Additionally, {random.choice(self.technical_terms)} plays a crucial role in ensuring {random.choice(self.technical_terms)} and maintaining {random.choice(self.technical_terms)} standards."
            
            content_parts.append(template)
            current_words += len(template.split())
        
        return " ".join(content_parts)
    
    def _generate_qa_document(self, topic: str, target_words: int) -> str:
        """Generate a Q&A formatted document"""
        
        title = f"Frequently Asked Questions: {topic.title()}"
        
        questions = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"What are the main applications of {topic}?",
            f"What are the benefits of using {topic}?",
            f"What challenges are associated with {topic}?",
            f"How do you implement {topic} in practice?",
            f"What tools are available for {topic}?",
            f"What is the future of {topic}?",
            f"How does {topic} compare to traditional methods?",
            f"What skills are needed for {topic}?"
        ]
        
        content_parts = [title, "\n"]
        current_words = len(title.split())
        
        for i, question in enumerate(questions[:8]):  # Limit to 8 questions
            if current_words >= target_words:
                break
                
            answer_templates = [
                f"{topic.title()} is a {random.choice(self.technical_terms)} approach that enables {random.choice(self.technical_terms)} through advanced {random.choice(self.technical_terms)}. It combines traditional methods with modern {random.choice(self.technical_terms)} to achieve superior results.",
                
                f"The implementation of {topic} involves several key steps including {random.choice(self.technical_terms)}, {random.choice(self.technical_terms)}, and continuous {random.choice(self.technical_terms)}. Organizations typically see improvements in {random.choice(self.technical_terms)} and {random.choice(self.technical_terms)}.",
                
                f"Key advantages include enhanced {random.choice(self.technical_terms)}, improved {random.choice(self.technical_terms)}, and better {random.choice(self.technical_terms)}. These benefits make {topic} an attractive option for organizations seeking {random.choice(self.technical_terms)} solutions."
            ]
            
            answer = random.choice(answer_templates)
            
            qa_text = f"\nQ{i+1}: {question}\nA{i+1}: {answer}\n"
            content_parts.append(qa_text)
            current_words += len(qa_text.split())
        
        return "".join(content_parts)
    
    def _generate_code_documentation(self, topic: str, target_words: int) -> str:
        """Generate code documentation style content"""
        
        class_name = f"{topic.replace(' ', '').title()}Manager"
        
        content = f"""
# {class_name} Documentation

## Overview

The {class_name} class provides comprehensive functionality for {topic} operations,
including {random.choice(self.technical_terms)}, {random.choice(self.technical_terms)}, 
and {random.choice(self.technical_terms)} management.

## Class Definition

```python
class {class_name}:
    \"\"\"
    Main class for handling {topic} operations.
    
    This class implements core {topic} functionality including:
    - {random.choice(self.technical_terms).title()} management
    - {random.choice(self.technical_terms).title()} processing  
    - {random.choice(self.technical_terms).title()} optimization
    \"\"\"
    
    def __init__(self, config=None):
        \"\"\"Initialize {class_name} with optional configuration.\"\"\"
        pass
        
    def process(self, data):
        \"\"\"Process input data using {topic} algorithms.\"\"\"
        pass
        
    def optimize(self, parameters):
        \"\"\"Optimize {topic} parameters for better performance.\"\"\"
        pass
```

## Methods

### process(data)
Processes input data using advanced {topic} algorithms. The method implements 
{random.choice(self.technical_terms)} techniques to ensure optimal {random.choice(self.technical_terms)}.

**Parameters:**
- data: Input data for {topic} processing
- options: Optional processing parameters

**Returns:**
Processed data with applied {topic} transformations.

### optimize(parameters)
Performs {random.choice(self.technical_terms)} optimization to improve system performance.
This method uses {random.choice(self.technical_terms)} algorithms to find optimal
{random.choice(self.technical_terms)} settings.

**Parameters:**
- parameters: Dictionary of optimization parameters
- constraints: Optional performance constraints

**Returns:**
Optimized parameter configuration.

## Usage Examples

```python
# Initialize manager
manager = {class_name}()

# Process data
result = manager.process(input_data)

# Optimize parameters
optimal_params = manager.optimize({{'param1': 0.5, 'param2': 1.0}})
```

## Performance Considerations

When using {class_name}, consider the following {random.choice(self.technical_terms)} 
factors:

- {random.choice(self.technical_terms).title()} requirements scale with data size
- {random.choice(self.technical_terms).title()} optimization improves throughput
- {random.choice(self.technical_terms).title()} monitoring enables proactive management

## Best Practices

1. Always validate input data before processing
2. Use {random.choice(self.technical_terms)} for large datasets
3. Implement proper error handling and {random.choice(self.technical_terms)}
4. Monitor {random.choice(self.technical_terms)} metrics during operation
5. Regular {random.choice(self.technical_terms)} updates improve performance
"""
        
        return content.strip()
    
    def _generate_news_article(self, topic: str, target_words: int) -> str:
        """Generate news article style content"""
        
        headline = f"Revolutionary Advances in {topic.title()} Transform Industry Landscape"
        
        content = f"""
{headline}

Recent developments in {topic} have captured the attention of industry experts and 
technology leaders worldwide. Major breakthroughs in {random.choice(self.technical_terms)} 
and {random.choice(self.technical_terms)} are reshaping how organizations approach 
{topic} implementation.

Industry Impact

Leading technology companies are investing heavily in {topic} research and development, 
with some allocating over ${random.randint(100, 999)} million annually to advance 
{random.choice(self.technical_terms)} capabilities. This investment surge reflects 
growing recognition of {topic}'s potential to drive {random.choice(self.technical_terms)} 
and competitive advantage.

"The {topic} landscape is evolving rapidly," said Dr. Sarah Johnson, Director of 
{random.choice(self.technical_terms).title()} Research at TechCorp. "We're seeing 
unprecedented innovation in {random.choice(self.technical_terms)} and 
{random.choice(self.technical_terms)}, which opens new possibilities for 
{random.choice(self.technical_terms)} applications."

Market Trends

Market analysts predict that the {topic} sector will grow by {random.randint(25, 75)}% 
over the next {random.randint(3, 7)} years, driven by increasing demand for 
{random.choice(self.technical_terms)} solutions and improved {random.choice(self.technical_terms)} 
methodologies.

Key growth drivers include:
- Enhanced {random.choice(self.technical_terms)} capabilities
- Improved {random.choice(self.technical_terms)} efficiency  
- Advanced {random.choice(self.technical_terms)} integration
- Streamlined {random.choice(self.technical_terms)} processes

Future Outlook

Experts anticipate continued innovation in {topic}, with emerging 
{random.choice(self.technical_terms)} technologies promising to unlock new 
applications and use cases. Organizations that invest in {random.choice(self.technical_terms)} 
infrastructure now are positioned to capitalize on these developments.

The convergence of {topic} with other advanced technologies such as 
{random.choice(self.topics)} and {random.choice(self.topics)} is expected to 
create synergistic opportunities for {random.choice(self.technical_terms)} 
and {random.choice(self.technical_terms)} optimization.

As the technology matures, widespread adoption across industries is anticipated, 
with {topic} becoming a standard component of modern {random.choice(self.technical_terms)} 
architectures and {random.choice(self.technical_terms)} frameworks.
"""
        
        return content.strip()
    
    def _generate_edge_case_document(self, case_type: str) -> str:
        """Generate edge case documents for testing"""
        
        if case_type == "empty":
            return ""
        
        elif case_type == "minimal":
            return f"{random.choice(self.topics)}"
            
        elif case_type == "unicode":
            return f"""
Unicode Test Document: {random.choice(self.topics)}

This document contains various Unicode characters: 
• Bullet points with special symbols ★ ◆ ▲
• Mathematical symbols: ∑ ∏ ∫ ∂ ∇ ∞
• Greek letters: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω
• Emoji symbols: 🤖 💻 🧠 📊 🔬 ⚡ 🚀 🎯
• International characters: café résumé naïve piñata
• Currency symbols: $ € £ ¥ ₹ ₿
• Arrows and symbols: → ← ↑ ↓ ⇒ ⇐ ⇑ ⇓

The purpose of this document is to test Unicode handling in {random.choice(self.topics)} 
systems and ensure proper {random.choice(self.technical_terms)} of special characters.
"""
        
        elif case_type == "very_long_line":
            long_line = " ".join([random.choice(self.technical_terms) for _ in range(200)])
            return f"Long Line Test: {long_line}\n\nThis document tests handling of extremely long lines."
        
        elif case_type == "special_chars":
            return f"""
Special Characters Test Document

This document contains various special characters and punctuation marks:

Brackets: () [] {{}} <> 
Quotes: "quotes" 'single' `backticks`
Punctuation: !@#$%^&*()_+-=[]{{}}|;':",./<>?
Mathematical: ± × ÷ ² ³ ½ ¼ ¾ √ ∞
Programming: // /* */ <tag> </tag> {{ }} \\ \n \t \r

Topic: {random.choice(self.topics)} with {random.choice(self.technical_terms)}

The {random.choice(self.technical_terms)} of {random.choice(self.topics)} requires 
careful handling of special characters in {random.choice(self.technical_terms)} systems.
"""
        
        elif case_type == "repeated_content":
            base_content = f"This is a test of repeated content in {random.choice(self.topics)}. "
            return (base_content * 20) + f"\n\nEnd of repeated content test for {random.choice(self.technical_terms)}."
        
        else:
            return f"Unknown edge case type: {case_type}"
    
    def generate_corpus(self, size: str, target_dir: str = None) -> Dict[str, Any]:
        """
        Generate a test corpus of specified size
        
        Args:
            size: 'small', 'medium', or 'large'
            target_dir: Target directory (uses default if None)
            
        Returns:
            Dictionary with corpus statistics
        """
        if not target_dir:
            target_dir = self.base_path / size
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Define corpus parameters
        corpus_config = {
            'small': {'docs': 10, 'words_per_doc': 1000},
            'medium': {'docs': 100, 'words_per_doc': 5000},
            'large': {'docs': 1000, 'words_per_doc': 3000}  # Variable sizes for large corpus
        }
        
        if size not in corpus_config:
            raise ValueError(f"Unknown corpus size: {size}. Use 'small', 'medium', or 'large'")
        
        config = corpus_config[size]
        self.logger.info(f"Generating {size} corpus with {config['docs']} documents")
        
        documents_created = 0
        total_words = 0
        
        # Document type distribution
        doc_types = ['technical_article', 'qa_document', 'code_documentation', 'news_article']
        
        for i in range(config['docs']):
            # Select document type and topic
            doc_type = random.choice(doc_types)
            topic = random.choice(self.topics)
            
            # Vary target words for large corpus
            if size == 'large':
                target_words = random.randint(500, 8000)  # Variable sizes
            else:
                target_words = config['words_per_doc']
            
            # Generate content based on type
            if doc_type == 'technical_article':
                content = self._generate_technical_article(topic, target_words)
            elif doc_type == 'qa_document':
                content = self._generate_qa_document(topic, target_words)
            elif doc_type == 'code_documentation':
                content = self._generate_code_documentation(topic, target_words)
            else:  # news_article
                content = self._generate_news_article(topic, target_words)
            
            # Save document
            filename = f"{doc_type}_{i+1:04d}_{topic.replace(' ', '_')}.txt"
            filepath = target_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents_created += 1
            total_words += len(content.split())
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Generated {i + 1}/{config['docs']} documents")
        
        # Generate metadata file
        metadata = {
            'corpus_size': size,
            'documents_count': documents_created,
            'total_words': total_words,
            'avg_words_per_document': total_words / documents_created if documents_created > 0 else 0,
            'document_types': doc_types,
            'topics_used': self.topics,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = target_path / 'corpus_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Corpus generation complete: {documents_created} documents, {total_words} total words")
        return metadata
    
    def generate_edge_cases(self, target_dir: str = None) -> Dict[str, Any]:
        """Generate edge case documents for testing"""
        
        if not target_dir:
            target_dir = self.base_path / 'edge_cases'
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        edge_cases = [
            'empty',
            'minimal',
            'unicode', 
            'very_long_line',
            'special_chars',
            'repeated_content'
        ]
        
        self.logger.info(f"Generating {len(edge_cases)} edge case documents")
        
        documents_created = 0
        
        for case_type in edge_cases:
            content = self._generate_edge_case_document(case_type)
            filename = f"edge_case_{case_type}.txt"
            filepath = target_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents_created += 1
        
        # Generate metadata
        metadata = {
            'edge_cases_count': documents_created,
            'case_types': edge_cases,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Edge case documents for testing system robustness'
        }
        
        metadata_file = target_path / 'edge_cases_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Edge cases generation complete: {documents_created} documents")
        return metadata
    
    def generate_all_corpora(self) -> Dict[str, Any]:
        """Generate all corpus sizes and edge cases"""
        
        self.logger.info("Starting generation of all test corpora")
        
        results = {}
        
        # Generate different sized corpora
        for size in ['small', 'medium', 'large']:
            try:
                self.logger.info(f"Generating {size} corpus...")
                results[size] = self.generate_corpus(size)
            except Exception as e:
                self.logger.error(f"Error generating {size} corpus: {e}")
                results[size] = {'error': str(e)}
        
        # Generate edge cases
        try:
            self.logger.info("Generating edge cases...")
            results['edge_cases'] = self.generate_edge_cases()
        except Exception as e:
            self.logger.error(f"Error generating edge cases: {e}")
            results['edge_cases'] = {'error': str(e)}
        
        # Save overall summary
        summary_file = self.base_path / 'generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'corpora_generated': list(results.keys()),
                'results': results
            }, f, indent=2)
        
        self.logger.info("All test corpora generation complete")
        return results

def main():
    """Main function for generating test corpus"""
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize generator
    generator = TestCorpusGenerator()
    
    # Generate all corpora
    results = generator.generate_all_corpora()
    
    # Print summary
    print("\n=== Test Corpus Generation Summary ===")
    total_docs = 0
    total_words = 0
    
    for corpus_name, result in results.items():
        if 'error' in result:
            print(f"{corpus_name.upper()}: ERROR - {result['error']}")
        else:
            docs = result.get('documents_count', result.get('edge_cases_count', 0))
            words = result.get('total_words', 0)
            total_docs += docs
            total_words += words
            
            print(f"{corpus_name.upper()}:")
            print(f"  Documents: {docs}")
            if words > 0:
                print(f"  Total words: {words:,}")
                print(f"  Avg words/doc: {words//docs if docs > 0 else 0}")
    
    print(f"\nTOTAL: {total_docs} documents, {total_words:,} words")
    print(f"Generated in: {Path().cwd() / 'test_data' / 'corpora'}")

if __name__ == "__main__":
    main()
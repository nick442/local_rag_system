"""
Document Ingestion Module for RAG System
Handles loading and processing of various document formats.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

import tiktoken
from bs4 import BeautifulSoup
import html2text
import markdown
import PyPDF2


@dataclass
class Document:
    """Standardized document representation."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic document ID based on content hash."""
        content_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        return f"doc_{content_hash[:16]}"


@dataclass
class DocumentChunk:
    """Document chunk with metadata."""
    content: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = f"{self.doc_id}_chunk_{self.chunk_index}"


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Load document from file path."""
        pass
    
    def _get_base_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get common metadata for all document types."""
        path = Path(file_path)
        return {
            'source': str(path.absolute()),
            'filename': path.name,
            'file_type': path.suffix.lower(),
            'size': path.stat().st_size if path.exists() else 0,
            'timestamp': datetime.now().isoformat()
        }


class TextLoader(DocumentLoader):
    """Loader for plain text files."""
    
    def load(self, file_path: str) -> Document:
        """Load plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        metadata = self._get_base_metadata(file_path)
        metadata['loader'] = 'TextLoader'
        
        return Document(content=content, metadata=metadata)


class PDFLoader(DocumentLoader):
    """Loader for PDF files using PyPDF2."""
    
    def load(self, file_path: str) -> Document:
        """Load PDF file and extract text."""
        content_parts = []
        metadata = self._get_base_metadata(file_path)
        metadata['loader'] = 'PDFLoader'
        metadata['pages'] = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                metadata['total_pages'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            content_parts.append(f"[Page {page_num}]\n{page_text}")
                            metadata['pages'].append({
                                'page_number': page_num,
                                'text_length': len(page_text)
                            })
                    except Exception as e:
                        logging.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
        
        except Exception as e:
            logging.error(f"Error loading PDF {file_path}: {e}")
            raise
        
        content = "\n\n".join(content_parts)
        return Document(content=content, metadata=metadata)


class HTMLLoader(DocumentLoader):
    """Loader for HTML files using BeautifulSoup and html2text."""
    
    def __init__(self):
        self.html2text_handler = html2text.HTML2Text()
        self.html2text_handler.ignore_links = False
        self.html2text_handler.ignore_images = True
        self.html2text_handler.body_width = 0
    
    def load(self, file_path: str) -> Document:
        """Load HTML file and convert to text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        metadata = self._get_base_metadata(file_path)
        metadata['loader'] = 'HTMLLoader'
        
        # Extract title and meta information
        if soup.title:
            metadata['title'] = soup.title.get_text().strip()
        
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            if meta.get('name') and meta.get('content'):
                metadata[f"meta_{meta.get('name')}"] = meta.get('content')
        
        # Extract headers for structure
        headers = []
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                headers.append({
                    'level': i,
                    'text': header.get_text().strip()
                })
        metadata['headers'] = headers
        
        # Convert to markdown-style text
        content = self.html2text_handler.handle(html_content)
        
        return Document(content=content, metadata=metadata)


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""
    
    def load(self, file_path: str) -> Document:
        """Load Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        metadata = self._get_base_metadata(file_path)
        metadata['loader'] = 'MarkdownLoader'
        
        # Extract headers for structure
        headers = []
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level <= 6:
                    text = line.lstrip('# ').strip()
                    headers.append({
                        'level': level,
                        'text': text,
                        'line': line_num
                    })
        
        metadata['headers'] = headers
        
        return Document(content=content, metadata=metadata)


class DocumentChunker:
    """Handles document chunking with token counting."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks with overlap."""
        tokens = self.encoding.encode(document.content)
        chunks = []
        
        if len(tokens) <= self.chunk_size:
            # Document is small enough to be a single chunk
            chunk = DocumentChunk(
                content=document.content,
                doc_id=document.doc_id,
                chunk_index=0,
                metadata=document.metadata.copy(),
                token_count=len(tokens)
            )
            chunks.append(chunk)
            return chunks
        
        # Create overlapping chunks
        chunk_index = 0
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'start_token': start,
                'end_token': end,
                'total_tokens': len(tokens)
            })
            
            chunk = DocumentChunk(
                content=chunk_text,
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                metadata=chunk_metadata,
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)
            
            # Move start position for next chunk with overlap
            next_start = end - self.overlap
            
            # Prevent infinite loop - ensure we always make progress
            if next_start <= start:
                # If overlap is too large or we're at the end, break
                break
            
            start = next_start
            chunk_index += 1
        
        return chunks


class DocumentIngestionService:
    """Main service for document ingestion."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.loaders = {
            '.txt': TextLoader(),
            '.pdf': PDFLoader(),
            '.html': HTMLLoader(),
            '.htm': HTMLLoader(),
            '.md': MarkdownLoader(),
            '.markdown': MarkdownLoader()
        }
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.logger = logging.getLogger(__name__)
    
    def ingest_document(self, file_path: str) -> List[DocumentChunk]:
        """Ingest a single document and return chunks."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        if file_extension not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        try:
            # Load document
            loader = self.loaders[file_extension]
            document = loader.load(file_path)
            
            self.logger.info(f"Loaded document: {path.name} ({len(document.content)} characters)")
            
            # Chunk document
            chunks = self.chunker.chunk_document(document)
            
            self.logger.info(f"Created {len(chunks)} chunks for {path.name}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error ingesting document {file_path}: {e}")
            raise
    
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> List[DocumentChunk]:
        """Ingest all supported documents in a directory."""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_chunks = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                try:
                    chunks = self.ingest_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.warning(f"Skipping {file_path}: {e}")
                    continue
        
        self.logger.info(f"Ingested {len(all_chunks)} total chunks from {directory_path}")
        return all_chunks
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loaders.keys())
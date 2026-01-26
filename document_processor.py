"""
Document Processor Module - Multi-Format Document Ingestion
==========================================================

This module handles the ingestion and intelligent chunking of various document formats
for RAGnarok's document intelligence system.

Key Features:
- Multi-format support: PDF, TXT, Markdown
- Intelligent text chunking with semantic boundary detection
- Metadata preservation for source attribution
- Robust error handling for corrupted documents

Chunking Strategy:
- Prioritizes sentence boundaries for semantic coherence
- Configurable chunk size and overlap
- Fallback to word boundaries when needed
- Preserves document structure and source information

Author: RAGnarok Team
Version: 2.0.0
"""

import os
import re
from typing import List, Dict
from pathlib import Path
import PyPDF2
import markdown
from docx import Document


class DocumentProcessor:
    """
    Multi-Format Document Processing Engine
    ======================================
    
    Handles document ingestion, text extraction, and intelligent chunking
    for optimal retrieval performance in the RAG pipeline.
    
    Supported Formats:
    - PDF: Using PyPDF2 for text extraction
    - TXT: Direct UTF-8 text processing
    - Markdown: Conversion to plain text via python-markdown
    
    Chunking Philosophy:
    - Maintain semantic coherence by respecting sentence boundaries
    - Balance context preservation with retrieval precision
    - Ensure overlap to prevent information loss at boundaries
    """
   
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize Document Processor
        ============================
        
        Sets up the document processing pipeline with configurable chunking parameters.
        
        Args:
            chunk_size (int): Target size for text chunks in characters
                            Default: 512 (optimal balance for retrieval)
            chunk_overlap (int): Overlap between consecutive chunks in characters
                               Default: 50 (prevents information loss)
        
        Chunking Rationale:
        - 512 chars: Provides sufficient context while maintaining specificity
        - 50 char overlap: Ensures important information isn't lost at boundaries
        - Sentence boundaries: Maintains semantic coherence when possible
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """
        Extract Text from PDF Documents
        ==============================
        
        Uses PyPDF2 to extract text content from PDF files.
        Handles multi-page documents and concatenates all text.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from all pages
            
        Raises:
            Exception: If PDF is corrupted or unreadable
            
        Note: Some PDFs (scanned images) may not extract text properly.
        Future versions could integrate OCR for image-based PDFs.
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Extract text from each page and concatenate
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        return text
    
    def load_txt(self, file_path: str) -> str:
        """
        Load Plain Text Files
        ====================
        
        Reads UTF-8 encoded text files directly.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: File content as string
            
        Raises:
            Exception: If file cannot be read or encoding issues occur
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT file {file_path}: {str(e)}")
    
    def load_markdown(self, file_path: str) -> str:
        """
        Process Markdown Files
        =====================
        
        Converts Markdown to plain text by:
        1. Reading the .md file
        2. Converting Markdown to HTML using python-markdown
        3. Stripping HTML tags to get clean text
        
        Args:
            file_path (str): Path to the Markdown file
            
        Returns:
            str: Plain text content with Markdown formatting removed
            
        Raises:
            Exception: If file cannot be read or Markdown parsing fails
            
        Note: This preserves text content while removing formatting.
        Future versions could preserve some structure (headers, lists).
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                # Simple HTML tag removal using regex
                text = re.sub(r'<[^>]+>', '', html)
                return text
        except Exception as e:
            raise Exception(f"Error reading Markdown file {file_path}: {str(e)}")
    
    def load_document(self, file_path: str) -> str:
        """
        Universal Document Loader
        ========================
        
        Automatically detects file type and uses appropriate loader.
        
        Args:
            file_path (str): Path to document file
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            
        Supported Extensions:
        - .pdf: PDF documents
        - .txt: Plain text files
        - .md, .markdown: Markdown files
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Route to appropriate loader based on file extension
        if extension == '.pdf':
            return self.load_pdf(str(file_path))
        elif extension == '.txt':
            return self.load_txt(str(file_path))
        elif extension in ['.md', '.markdown']:
            return self.load_markdown(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # Use sentence boundaries when possible for better chunking
        sentence_endings = re.compile(r'[.!?]\s+')
        
        while start < text_length:
            end = start + self.chunk_size
            
            if end >= text_length:
                # Last chunk
                chunk_text = text[start:].strip()
            else:
                # Try to break at sentence boundary
                chunk_text = text[start:end]
                last_sentence = sentence_endings.search(chunk_text[::-1])
                
                if last_sentence:
                    # Adjust end to sentence boundary
                    end = start + len(chunk_text) - last_sentence.start()
                    chunk_text = text[start:end].strip()
                else:
                    # Break at word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.chunk_size * 0.5:  # Only if we're not too far
                        end = start + last_space
                        chunk_text = text[start:end].strip()
                    else:
                        chunk_text = chunk_text.strip()
            
            if chunk_text:
                chunk_metadata = {
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'start_char': start,
                    'end_char': end,
                }
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunks.append(chunk_metadata)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict]:
       
        directory = Path(directory_path)
        all_chunks = []
        
        supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    print(f"Processing: {file_path.name}")
                    text = self.load_document(str(file_path))
                    
                    metadata = {
                        'source': file_path.name,
                        'file_path': str(file_path),
                    }
                    
                    chunks = self.chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                    print(f"  Created {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        return all_chunks
    
    def process_files(self, file_paths: List[str]) -> List[Dict]:
       
        all_chunks = []
        
        for file_path in file_paths:
            try:
                print(f"Processing: {file_path}")
                text = self.load_document(file_path)
                
                metadata = {
                    'source': Path(file_path).name,
                    'file_path': file_path,
                }
                
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks from {Path(file_path).name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return all_chunks

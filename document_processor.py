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

# ============================================================================
# IMPORTS - Required libraries for document processing
# ============================================================================

# os: Operating system interface for file operations
# Used for checking file existence and path operations
import os

# re: Regular expression operations for pattern matching
# Used for finding sentence boundaries and HTML tag removal
# Patterns: [.!?]\s+ for sentence endings, <[^>]+> for HTML tags
import re

# typing: Type hints for better code documentation
# List: Type hint for list objects
# Dict: Type hint for dictionary objects
from typing import List, Dict

# Path: Object-oriented filesystem path handling from pathlib
# Provides cleaner path operations than os.path
# Methods: iterdir(), is_file(), suffix, name
from pathlib import Path

# PyPDF2: Library for reading PDF files
# Extracts text from PDF documents page by page
# Note: Doesn't work with scanned PDFs (image-based)
import PyPDF2

# markdown: Library for converting Markdown to HTML
# Converts .md files to HTML, then we strip tags for plain text
# Preserves content while removing formatting
import markdown

# Document: Class from python-docx for reading Word documents
# Note: Currently imported but not used in the code
# Could be used for .docx support in future versions
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
   
    # ========================================================================
    # INITIALIZATION METHOD
    # ========================================================================
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
        # ====================================================================
        # Store chunking parameters as instance variables
        # ====================================================================
        # chunk_size: How many characters each chunk should contain
        # This affects:
        # - Retrieval precision (smaller = more precise)
        # - Context amount (larger = more context)
        # - Number of chunks (smaller = more chunks)
        # 512 is optimal based on research and testing
        self.chunk_size = chunk_size
        
        # chunk_overlap: How many characters overlap between consecutive chunks
        # This prevents information loss at chunk boundaries
        # Example: If a sentence is split between chunks, overlap ensures
        # the complete sentence appears in at least one chunk
        # 50 characters is typically 8-10 words
        self.chunk_overlap = chunk_overlap
    
    # ========================================================================
    # METHOD: Load and extract text from PDF files
    # ========================================================================
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
        # ====================================================================
        # STEP 1: Initialize empty string to accumulate text
        # ====================================================================
        # We'll concatenate text from all pages into this variable
        text = ""
        
        # ====================================================================
        # STEP 2: Open and process PDF file with error handling
        # ====================================================================
        try:
            # ================================================================
            # STEP 2a: Open PDF file in binary read mode
            # ================================================================
            # 'rb' = read binary mode (required for PDF files)
            # with statement ensures file is properly closed after use
            with open(file_path, 'rb') as file:
                
                # ============================================================
                # STEP 2b: Create PDF reader object
                # ============================================================
                # PdfReader: PyPDF2 class for reading PDF files
                # Parses PDF structure and provides access to pages
                # Handles PDF format complexities internally
                pdf_reader = PyPDF2.PdfReader(file)
                
                # ============================================================
                # STEP 2c: Iterate through all pages and extract text
                # ============================================================
                # pdf_reader.pages: List-like object containing all pages
                # Each page is a PageObject with text extraction methods
                for page in pdf_reader.pages:
                    # Extract text from current page
                    # extract_text(): PyPDF2 method that:
                    # 1. Parses page content stream
                    # 2. Extracts text objects
                    # 3. Reconstructs text in reading order
                    # 4. Returns as string
                    #
                    # Limitations:
                    # - May not preserve exact formatting
                    # - Doesn't work with scanned PDFs (images)
                    # - May have issues with complex layouts
                    # - Special characters might not extract correctly
                    page_text = page.extract_text()
                    
                    # Append page text to accumulated text
                    # Add newline to separate pages
                    text += page_text + "\n"
        
        # ====================================================================
        # STEP 3: Handle any errors during PDF processing
        # ====================================================================
        except Exception as e:
            # Catch any exception (file not found, corrupted PDF, etc.)
            # Raise a new exception with more context
            # str(e): Convert exception to string for error message
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        
        # ====================================================================
        # STEP 4: Return extracted text
        # ====================================================================
        # Returns all text from all pages concatenated together
        # Each page is separated by a newline character
        return text
    
    # ========================================================================
    # METHOD: Load and read plain text files
    # ========================================================================
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
        # ====================================================================
        # STEP 1: Open and read text file with error handling
        # ====================================================================
        try:
            # ================================================================
            # STEP 1a: Open file in text read mode with UTF-8 encoding
            # ================================================================
            # 'r' = read text mode
            # encoding='utf-8': Decode bytes as UTF-8 text
            # UTF-8 supports all Unicode characters (international text)
            # with statement ensures file is properly closed
            with open(file_path, 'r', encoding='utf-8') as file:
                
                # ============================================================
                # STEP 1b: Read entire file content
                # ============================================================
                # read(): Reads entire file into memory as string
                # For very large files (>100MB), consider reading in chunks
                # But for typical documents, this is fine
                return file.read()
        
        # ====================================================================
        # STEP 2: Handle any errors during file reading
        # ====================================================================
        except Exception as e:
            # Possible errors:
            # - FileNotFoundError: File doesn't exist
            # - PermissionError: No read permission
            # - UnicodeDecodeError: File not UTF-8 encoded
            # - IOError: General I/O error
            raise Exception(f"Error reading TXT file {file_path}: {str(e)}")
    
    # ========================================================================
    # METHOD: Load and convert Markdown files to plain text
    # ========================================================================
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
        # ====================================================================
        # STEP 1: Open and process Markdown file with error handling
        # ====================================================================
        try:
            # ================================================================
            # STEP 1a: Open and read Markdown file
            # ================================================================
            # Same as load_txt() - read as UTF-8 text
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                
                # ============================================================
                # STEP 1b: Convert Markdown to HTML
                # ============================================================
                # markdown.markdown(): Converts Markdown syntax to HTML
                # Examples:
                # - "# Header" → "<h1>Header</h1>"
                # - "**bold**" → "<strong>bold</strong>"
                # - "[link](url)" → "<a href='url'>link</a>"
                # - "- item" → "<ul><li>item</li></ul>"
                #
                # Why convert to HTML first?
                # - Markdown library provides robust parsing
                # - HTML is easier to strip than Markdown
                # - Handles edge cases and complex formatting
                html = markdown.markdown(md_content)
                
                # ============================================================
                # STEP 1c: Strip HTML tags to get plain text
                # ============================================================
                # re.sub(): Regular expression substitution
                # Pattern: r'<[^>]+>'
                # - <: Match opening angle bracket
                # - [^>]+: Match one or more characters that are NOT >
                # - >: Match closing angle bracket
                # This matches any HTML tag: <tag>, </tag>, <tag attr="value">
                #
                # Replacement: '' (empty string)
                # Effect: Removes all HTML tags, leaving only text content
                #
                # Example:
                # "<h1>Header</h1><p>Text</p>" → "HeaderText"
                #
                # Limitation: Doesn't add spaces between tags
                # Future improvement: Add space when removing block-level tags
                text = re.sub(r'<[^>]+>', '', html)
                
                # Return the plain text
                return text
        
        # ====================================================================
        # STEP 2: Handle any errors during Markdown processing
        # ====================================================================
        except Exception as e:
            # Possible errors:
            # - File reading errors (same as load_txt)
            # - Markdown parsing errors (malformed Markdown)
            # - Regex errors (shouldn't happen with this pattern)
            raise Exception(f"Error reading Markdown file {file_path}: {str(e)}")
    
    # ========================================================================
    # METHOD: Universal document loader (auto-detects file type)
    # ========================================================================
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
        # ====================================================================
        # STEP 1: Convert string path to Path object
        # ====================================================================
        # Path: Object-oriented way to handle filesystem paths
        # Provides methods like suffix, name, parent, etc.
        # More readable than os.path operations
        file_path = Path(file_path)
        
        # ====================================================================
        # STEP 2: Extract file extension
        # ====================================================================
        # suffix: Returns file extension including the dot
        # Examples: ".pdf", ".txt", ".md"
        # lower(): Convert to lowercase for case-insensitive matching
        # Handles: "File.PDF", "file.Txt", "FILE.MD"
        extension = file_path.suffix.lower()
        
        # ====================================================================
        # STEP 3: Route to appropriate loader based on extension
        # ====================================================================
        # Use if-elif-else chain to check extension and call correct loader
        
        if extension == '.pdf':
            # PDF file - use PyPDF2 loader
            # Convert Path back to string for load_pdf()
            return self.load_pdf(str(file_path))
            
        elif extension == '.txt':
            # Plain text file - use direct text loader
            return self.load_txt(str(file_path))
            
        elif extension in ['.md', '.markdown']:
            # Markdown file - use Markdown converter
            # Supports both .md and .markdown extensions
            return self.load_markdown(str(file_path))
            
        else:
            # Unsupported file type
            # Raise ValueError with informative message
            # This helps users understand what formats are supported
            raise ValueError(f"Unsupported file type: {extension}")
    
    # ========================================================================
    # METHOD: Intelligent text chunking with semantic boundary detection
    # ========================================================================
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Intelligent Text Chunking with Semantic Boundaries
        =================================================
        
        Splits long text into smaller chunks while preserving semantic coherence
        by respecting sentence and word boundaries.
        
        Args:
            text (str): Full text content to be chunked
            metadata (Dict): Optional metadata to attach to each chunk
                            (e.g., source filename, document type)
        
        Returns:
            List[Dict]: List of chunk dictionaries, each containing:
                       - text: The chunk content
                       - chunk_index: Position in sequence
                       - start_char: Starting character position
                       - end_char: Ending character position
                       - ...any additional metadata provided
        
        Algorithm:
        1. Start at position 0
        2. Extract chunk_size characters
        3. Try to break at sentence boundary (. ! ?)
        4. If no sentence boundary, break at word boundary (space)
        5. If no word boundary, hard break at chunk_size
        6. Move forward by (chunk_size - overlap) for next chunk
        7. Repeat until end of text
        
        Why This Matters:
        - Sentence boundaries: Keeps complete thoughts together
        - Word boundaries: Prevents breaking words in half
        - Overlap: Ensures context isn't lost between chunks
        - Metadata: Enables source attribution in responses
        """
        
        # ====================================================================
        # STEP 1: Handle empty text edge case
        # ====================================================================
        # Check if text is empty or only whitespace
        # strip(): Removes leading/trailing whitespace
        # not text.strip(): True if empty or only whitespace
        if not text.strip():
            # Return empty list - no chunks to create
            # Prevents errors in downstream processing
            return []
        
        # ====================================================================
        # STEP 2: Initialize chunking variables
        # ====================================================================
        # chunks: List to store all created chunk dictionaries
        # Will be populated as we process the text
        chunks = []
        
        # start: Current position in text (character index)
        # Starts at 0 (beginning of text)
        # Will be incremented as we create chunks
        start = 0
        
        # text_length: Total length of text to process
        # Used to know when we've reached the end
        # Prevents index out of bounds errors
        text_length = len(text)
        
        # ====================================================================
        # STEP 3: Compile regex pattern for sentence detection
        # ====================================================================
        # Regular expression to find sentence endings
        # Pattern breakdown:
        # - [.!?]: Character class matching period, exclamation, or question mark
        # - \s+: One or more whitespace characters (space, tab, newline)
        #
        # Why this pattern?
        # - Identifies natural sentence boundaries
        # - Ensures there's whitespace after punctuation (not "Dr.Smith")
        # - Helps maintain semantic coherence in chunks
        #
        # Examples that match:
        # - "Hello. " (period + space)
        # - "Really! " (exclamation + space)
        # - "Why? " (question + space)
        #
        # Examples that DON'T match:
        # - "Dr.Smith" (no space after period)
        # - "3.14" (no space after period)
        # - "Hello." (no space after period)
        sentence_endings = re.compile(r'[.!?]\s+')
        
        # ====================================================================
        # STEP 4: Main chunking loop - process until end of text
        # ====================================================================
        # Continue looping while start position is before end of text
        # Each iteration creates one chunk
        while start < text_length:
            
            # ================================================================
            # STEP 4a: Calculate initial end position for this chunk
            # ================================================================
            # end: Where this chunk should end
            # start + chunk_size: Move forward by chunk_size characters
            # Example: If start=0 and chunk_size=512, end=512
            end = start + self.chunk_size
            
            # ================================================================
            # STEP 4b: Handle last chunk (end of document)
            # ================================================================
            if end >= text_length:
                # We've reached or passed the end of the text
                # This is the last chunk
                
                # Take everything from start to end of text
                # text[start:]: Slice from start to end (no end index = to end)
                # strip(): Remove leading/trailing whitespace
                chunk_text = text[start:].strip()
                
            else:
                # ============================================================
                # STEP 4c: Try to break at sentence boundary (preferred)
                # ============================================================
                # Extract the potential chunk text
                # text[start:end]: Slice from start to end position
                chunk_text = text[start:end]
                
                # Search for sentence endings in REVERSE order
                # Why reverse? To find the LAST sentence ending in the chunk
                # This keeps the chunk as large as possible while respecting boundaries
                #
                # [::-1]: Reverses the string
                # Example: "Hello. World." → ".dlroW .olleH"
                #
                # search(): Finds first match in reversed string
                # Returns Match object or None
                last_sentence = sentence_endings.search(chunk_text[::-1])
                
                if last_sentence:
                    # Found a sentence boundary!
                    # Now we need to adjust the end position
                    
                    # Calculate position in original (non-reversed) string
                    # len(chunk_text): Total length of chunk
                    # last_sentence.start(): Position in reversed string
                    # Subtracting gives position in original string
                    #
                    # Example:
                    # chunk_text = "Hello. World."
                    # reversed = ".dlroW .olleH"
                    # last_sentence.start() = 0 (position of first "." in reversed)
                    # len(chunk_text) - 0 = 14 (position after "." in original)
                    end = start + len(chunk_text) - last_sentence.start()
                    
                    # Extract chunk up to sentence boundary
                    # strip(): Remove extra whitespace
                    chunk_text = text[start:end].strip()
                    
                else:
                    # ========================================================
                    # STEP 4d: No sentence boundary - try word boundary
                    # ========================================================
                    # rfind(' '): Find the LAST space in the chunk
                    # Returns index of last space, or -1 if no space found
                    # This gives us the last word boundary
                    last_space = chunk_text.rfind(' ')
                    
                    # Only break at word boundary if it's not too far back
                    # self.chunk_size * 0.5: At least 50% of desired chunk size
                    # This prevents creating very small chunks
                    #
                    # Example: If chunk_size=512, only break at space if it's
                    # after position 256 (more than halfway through)
                    #
                    # Why this check?
                    # - Prevents tiny chunks (e.g., 50 characters)
                    # - Ensures chunks are reasonably sized
                    # - Balances boundary respect with chunk size goals
                    if last_space > self.chunk_size * 0.5:
                        # Found a good word boundary
                        # Adjust end position to break at the space
                        end = start + last_space
                        
                        # Extract chunk up to word boundary
                        chunk_text = text[start:end].strip()
                    else:
                        # ====================================================
                        # STEP 4e: No good word boundary - hard break
                        # ====================================================
                        # No sentence boundary found
                        # No word boundary in acceptable range
                        # Just use the chunk as-is (hard break)
                        #
                        # This is the fallback case
                        # Happens with:
                        # - Very long words
                        # - Text without spaces (URLs, code, etc.)
                        # - Dense text with few sentence endings
                        #
                        # strip(): Remove extra whitespace
                        chunk_text = chunk_text.strip()
            
            # ================================================================
            # STEP 4f: Create chunk metadata dictionary
            # ================================================================
            # Only create chunk if there's actual content
            # Prevents empty chunks from whitespace-only text
            if chunk_text:
                # Create dictionary with chunk information
                # This will be stored in the vector database
                chunk_metadata = {
                    # The actual text content of this chunk
                    'text': chunk_text,
                    
                    # Position in sequence (0, 1, 2, ...)
                    # len(chunks): Current number of chunks = next index
                    # Useful for reconstructing document order
                    'chunk_index': len(chunks),
                    
                    # Starting character position in original text
                    # Useful for locating chunk in source document
                    'start_char': start,
                    
                    # Ending character position in original text
                    # Useful for locating chunk in source document
                    'end_char': end,
                }
                
                # ============================================================
                # STEP 4g: Add any additional metadata provided
                # ============================================================
                # If metadata was provided (e.g., source filename),
                # add it to the chunk metadata
                #
                # update(): Merges dictionaries
                # Adds all key-value pairs from metadata to chunk_metadata
                #
                # Example metadata:
                # {
                #     'source': 'document.pdf',
                #     'file_path': '/path/to/document.pdf',
                #     'document_type': 'pdf'
                # }
                if metadata:
                    chunk_metadata.update(metadata)
                
                # ============================================================
                # STEP 4h: Add chunk to list
                # ============================================================
                # Append this chunk dictionary to the chunks list
                chunks.append(chunk_metadata)
            
            # ================================================================
            # STEP 4i: Move to next chunk position with overlap
            # ================================================================
            # Move start forward, but not by the full chunk size
            # Subtract overlap to create overlapping chunks
            #
            # Example:
            # - end = 512 (where this chunk ended)
            # - chunk_overlap = 50
            # - next start = 512 - 50 = 462
            #
            # This means:
            # - Characters 0-512 in chunk 1
            # - Characters 462-974 in chunk 2
            # - Characters 462-512 appear in BOTH chunks (overlap)
            #
            # Why overlap?
            # - Prevents information loss at boundaries
            # - Ensures complete sentences/concepts in at least one chunk
            # - Improves retrieval recall (more likely to find relevant info)
            start = end - self.chunk_overlap
        
        # ====================================================================
        # STEP 5: Return all created chunks
        # ====================================================================
        # Returns list of chunk dictionaries
        # Each dictionary contains text, metadata, and position information
        # Ready to be processed by embedding generator
        return chunks
    
    # ========================================================================
    # METHOD: Process all documents in a directory
    # ========================================================================
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Batch Process All Documents in Directory
        =======================================
        
        Scans a directory for supported document files and processes each one,
        aggregating all chunks into a single list.
        
        Args:
            directory_path (str): Path to directory containing documents
        
        Returns:
            List[Dict]: Combined list of all chunks from all documents
        
        Process:
        1. Scan directory for files
        2. Filter for supported extensions
        3. Load and chunk each file
        4. Aggregate all chunks
        5. Handle errors gracefully (skip problematic files)
        
        Error Handling:
        - Individual file errors don't stop the entire process
        - Errors are printed but processing continues
        - Allows partial success when some files are corrupted
        """
        
        # ====================================================================
        # STEP 1: Convert string path to Path object
        # ====================================================================
        # Path: Object-oriented way to handle filesystem paths
        # Provides methods like iterdir(), is_file(), suffix, etc.
        # More readable and robust than os.path operations
        directory = Path(directory_path)
        
        # ====================================================================
        # STEP 2: Initialize list to collect all chunks
        # ====================================================================
        # This will hold chunks from ALL documents in the directory
        # Will be populated as we process each file
        all_chunks = []
        
        # ====================================================================
        # STEP 3: Define supported file extensions
        # ====================================================================
        # List of file extensions we can process
        # Extensions are lowercase for case-insensitive matching
        # Will be compared against file_path.suffix.lower()
        supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
        
        # ====================================================================
        # STEP 4: Iterate through all items in directory
        # ====================================================================
        # iterdir(): Returns an iterator of all items in the directory
        # Includes both files and subdirectories
        # Does NOT recurse into subdirectories (only top level)
        for file_path in directory.iterdir():
            
            # ================================================================
            # STEP 4a: Filter for supported files only
            # ================================================================
            # Check two conditions:
            # 1. is_file(): Is this a file (not a directory)?
            # 2. suffix.lower() in supported_extensions: Is extension supported?
            #
            # suffix: Returns file extension including dot (e.g., ".pdf")
            # lower(): Convert to lowercase for case-insensitive matching
            # Handles: "File.PDF", "file.Txt", "FILE.MD"
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                
                # ============================================================
                # STEP 4b: Process file with error handling
                # ============================================================
                # Wrap in try-except to handle errors gracefully
                # If one file fails, we continue with others
                try:
                    # ========================================================
                    # STEP 4b-i: Print progress message
                    # ========================================================
                    # Inform user which file is being processed
                    # file_path.name: Just the filename (not full path)
                    # Example: "document.pdf" instead of "/path/to/document.pdf"
                    print(f"Processing: {file_path.name}")
                    
                    # ========================================================
                    # STEP 4b-ii: Load document text
                    # ========================================================
                    # load_document() automatically detects file type
                    # and uses appropriate loader (PDF, TXT, or MD)
                    # str(file_path): Convert Path object to string
                    # Returns the full text content as a string
                    text = self.load_document(str(file_path))
                    
                    # ========================================================
                    # STEP 4b-iii: Create metadata for this document
                    # ========================================================
                    # Metadata will be attached to every chunk from this file
                    # This enables source attribution in responses
                    #
                    # Dictionary contains:
                    # - source: Just the filename (for display)
                    # - file_path: Full path (for reference)
                    metadata = {
                        'source': file_path.name,      # e.g., "document.pdf"
                        'file_path': str(file_path),   # e.g., "/path/to/document.pdf"
                    }
                    
                    # ========================================================
                    # STEP 4b-iv: Chunk the text
                    # ========================================================
                    # chunk_text() splits text into smaller pieces
                    # Each chunk gets the metadata attached
                    # Returns list of chunk dictionaries
                    #
                    # Each chunk will have:
                    # - text: The chunk content
                    # - chunk_index: Position in document
                    # - start_char, end_char: Character positions
                    # - source: Filename (from metadata)
                    # - file_path: Full path (from metadata)
                    chunks = self.chunk_text(text, metadata)
                    
                    # ========================================================
                    # STEP 4b-v: Add chunks to master list
                    # ========================================================
                    # extend(): Adds all items from chunks list to all_chunks
                    # This is different from append() which would add the list itself
                    #
                    # Example:
                    # all_chunks = [chunk1, chunk2]
                    # chunks = [chunk3, chunk4]
                    # all_chunks.extend(chunks)
                    # Result: all_chunks = [chunk1, chunk2, chunk3, chunk4]
                    #
                    # If we used append():
                    # Result: all_chunks = [chunk1, chunk2, [chunk3, chunk4]]
                    all_chunks.extend(chunks)
                    
                    # ========================================================
                    # STEP 4b-vi: Print success message
                    # ========================================================
                    # Inform user how many chunks were created from this file
                    # Indented with spaces for visual hierarchy
                    print(f"  Created {len(chunks)} chunks from {file_path.name}")
                    
                # ============================================================
                # STEP 4c: Handle errors gracefully
                # ============================================================
                except Exception as e:
                    # If this file fails, print error but continue with other files
                    # This allows partial success when some files are corrupted
                    #
                    # Possible errors:
                    # - File reading errors (permissions, corruption)
                    # - PDF parsing errors (malformed PDF)
                    # - Encoding errors (non-UTF-8 text)
                    # - Markdown parsing errors
                    #
                    # str(e): Convert exception to string for error message
                    print(f"Error processing {file_path.name}: {str(e)}")
                    
                    # continue: Skip to next file in the loop
                    # Don't let one bad file stop the entire process
                    # This is important for robustness
                    continue
        
        # ====================================================================
        # STEP 5: Return all collected chunks
        # ====================================================================
        # Returns combined chunks from all successfully processed files
        # Each chunk has metadata indicating which file it came from
        # Ready to be processed by embedding generator
        return all_chunks
    
    # ========================================================================
    # METHOD: Process specific list of files
    # ========================================================================
    def process_files(self, file_paths: List[str]) -> List[Dict]:
        """
        Process Specific List of Files
        ==============================
        
        Processes a specific list of file paths (instead of a directory).
        Useful when you want to process only certain files.
        
        Args:
            file_paths (List[str]): List of file paths to process
        
        Returns:
            List[Dict]: Combined list of all chunks from all files
        
        Process:
        1. Iterate through provided file paths
        2. Load and chunk each file
        3. Aggregate all chunks
        4. Handle errors gracefully (skip problematic files)
        
        Difference from process_directory():
        - Takes explicit list of files instead of scanning directory
        - Doesn't filter by extension (assumes all files are supported)
        - Useful for selective processing
        """
        
        # ====================================================================
        # STEP 1: Initialize list to collect all chunks
        # ====================================================================
        # This will hold chunks from ALL files in the list
        all_chunks = []
        
        # ====================================================================
        # STEP 2: Iterate through each file path
        # ====================================================================
        # Process each file in the provided list
        for file_path in file_paths:
            
            # ================================================================
            # STEP 2a: Process file with error handling
            # ================================================================
            # Wrap in try-except to handle errors gracefully
            try:
                # ============================================================
                # STEP 2a-i: Print progress message
                # ============================================================
                # Inform user which file is being processed
                print(f"Processing: {file_path}")
                
                # ============================================================
                # STEP 2a-ii: Load document text
                # ============================================================
                # load_document() automatically detects file type
                # Returns the full text content as a string
                text = self.load_document(file_path)
                
                # ============================================================
                # STEP 2a-iii: Create metadata for this document
                # ============================================================
                # Metadata will be attached to every chunk from this file
                #
                # Path(file_path).name: Extract just the filename
                # Example: "/path/to/doc.pdf" → "doc.pdf"
                metadata = {
                    'source': Path(file_path).name,  # Just filename
                    'file_path': file_path,          # Full path
                }
                
                # ============================================================
                # STEP 2a-iv: Chunk the text
                # ============================================================
                # chunk_text() splits text into smaller pieces
                # Each chunk gets the metadata attached
                chunks = self.chunk_text(text, metadata)
                
                # ============================================================
                # STEP 2a-v: Add chunks to master list
                # ============================================================
                # extend(): Adds all items from chunks list to all_chunks
                all_chunks.extend(chunks)
                
                # ============================================================
                # STEP 2a-vi: Print success message
                # ============================================================
                # Inform user how many chunks were created
                print(f"  Created {len(chunks)} chunks from {Path(file_path).name}")
                
            # ================================================================
            # STEP 2b: Handle errors gracefully
            # ================================================================
            except Exception as e:
                # If this file fails, print error but continue with other files
                # str(e): Convert exception to string for error message
                print(f"Error processing {file_path}: {str(e)}")
                
                # continue: Skip to next file in the loop
                # Don't let one bad file stop the entire process
                continue
        
        # ====================================================================
        # STEP 3: Return all collected chunks
        # ====================================================================
        # Returns combined chunks from all successfully processed files
        return all_chunks

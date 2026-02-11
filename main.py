"""
RAGnarok Command Line Interface
==============================

Main CLI entry point for RAGnarok's document intelligence system.
Provides command-line access to document processing and query functionality.

Commands:
- setup: Process documents and build vector store
- query: Ask questions about loaded documents

Features:
- Batch document processing
- Interactive query interface
- Configurable parameters
- Progress tracking and error handling
- Docker-aware Ollama integration

Usage Examples:
    python main.py setup --documents-dir ./docs
    python main.py query "What are the main topics?"

Author: RAGnarok Team
Version: 2.0.0 (Docker + Gemma 2B)
"""

# ============================================================================
# IMPORTS - External libraries and internal modules
# ============================================================================

# argparse: Command-line argument parsing library
# Allows us to create a professional CLI with subcommands, flags, and help text
import argparse

# os: Operating system interface for file/directory operations
# Used to check if directories exist and create them if needed
import os

# Path: Object-oriented filesystem path handling from pathlib
# Provides cleaner, more readable path operations than os.path
from pathlib import Path

# DocumentProcessor: Our custom module for loading and chunking documents
# Handles PDF, TXT, and Markdown files, converts them to processable chunks
from document_processor import DocumentProcessor

# RAGPipeline: Core orchestration engine that coordinates the entire RAG workflow
# Manages embeddings, vector search, LLM communication, and response generation
from rag_pipeline import RAGPipeline

# VectorStore: FAISS-based vector database for semantic similarity search
# Stores document embeddings and performs fast nearest-neighbor search
from vector_store import VectorStore


# ============================================================================
# FUNCTION: setup_documents
# PURPOSE: One-time document processing and vector database creation
# ============================================================================
def setup_documents(documents_dir: str = "documents", 
                   vector_store_dir: str = "vector_store",
                   chunk_size: int = 512,
                   chunk_overlap: int = 50):
    """
    Document Processing and Vector Store Setup
    ==========================================
    
    Processes documents from a directory and builds a searchable vector database.
    This is typically run once when setting up RAGnarok with new documents.
    
    Args:
        documents_dir (str): Directory containing documents to process
        vector_store_dir (str): Directory to save the vector database
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters
    
    Returns:
        bool: True if setup successful, False otherwise
    
    Process:
    1. Validate documents directory exists
    2. Process all supported documents (PDF, TXT, MD)
    3. Initialize RAG pipeline with embeddings
    4. Generate embeddings for all text chunks
    5. Build and save FAISS vector index
    6. Display setup statistics
    
    Error Handling:
    - Missing documents directory
    - No supported files found
    - Processing errors for individual files
    - Vector store creation failures
    """
    # ========================================================================
    # STEP 1: Display header for user feedback
    # ========================================================================
    # Print a visual separator to make the output clear and professional
    print("=" * 60)
    print("RAGnarok Document Processing")
    print("=" * 60)
    
    # ========================================================================
    # STEP 2: Validate and create documents directory if needed
    # ========================================================================
    # Check if the documents directory exists using os.path.exists()
    # This prevents errors when trying to read from a non-existent directory
    if not os.path.exists(documents_dir):
        # If directory doesn't exist, create it using os.makedirs()
        # This is helpful for first-time users who haven't set up the folder yet
        os.makedirs(documents_dir)
        
        # Inform the user that they need to add documents before proceeding
        print(f"Created '{documents_dir}' directory.")
        print(f"Please add your documents (PDF, TXT, or Markdown) to '{documents_dir}' and run again.")
        
        # Return False to indicate setup is incomplete
        # The user needs to add documents and run the command again
        return False
    
    # ========================================================================
    # STEP 3: Initialize DocumentProcessor with chunking configuration
    # ========================================================================
    # Print progress message to keep user informed
    print(f"\n1. Processing documents from '{documents_dir}'...")
    
    # Create a DocumentProcessor instance with specified parameters
    # chunk_size: How many characters per chunk (512 is optimal for retrieval)
    # chunk_overlap: How many characters overlap between chunks (prevents info loss)
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Process all documents in the directory
    # This will:
    # 1. Find all PDF, TXT, and MD files
    # 2. Extract text from each file
    # 3. Split text into chunks with metadata
    # 4. Return a list of chunk dictionaries
    chunks = processor.process_directory(documents_dir)
    
    # ========================================================================
    # STEP 4: Validate that documents were successfully processed
    # ========================================================================
    # Check if any chunks were created
    # An empty list means no supported files were found or all files failed to process
    if not chunks:
        # Print error message with supported formats
        print(f"❌ No documents found in '{documents_dir}'")
        print("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
        
        # Return False to indicate failure
        return False
    
    # Print success message with chunk count
    # This tells the user how many text chunks were created from their documents
    print(f"✅ Processed {len(chunks)} chunks from documents")
    
    # ========================================================================
    # STEP 5: Initialize RAG pipeline with default configuration
    # ========================================================================
    # Print progress message
    print("\n2. Initializing RAG pipeline...")
    
    # Create RAGPipeline instance
    # This will:
    # 1. Load the BGE embedding model (BAAI/bge-base-en-v1.5)
    # 2. Create a FAISS vector store
    # 3. Initialize guardrails for safety
    # 4. Configure Ollama connection for LLM
    pipeline = RAGPipeline()
    
    # ========================================================================
    # STEP 6: Generate embeddings and build vector index
    # ========================================================================
    # Print progress message
    print("3. Generating embeddings and building vector index...")
    
    # Add all document chunks to the pipeline
    # This will:
    # 1. Extract text from each chunk
    # 2. Generate 768-dimensional embeddings using BGE model
    # 3. Store embeddings in FAISS index with metadata
    # This is the most time-consuming step (can take minutes for large documents)
    pipeline.add_documents(chunks)
    
    # ========================================================================
    # STEP 7: Persist vector store to disk for future use
    # ========================================================================
    # Print progress message
    print(f"4. Saving vector store to '{vector_store_dir}'...")
    
    # Save the vector store to disk
    # This creates three files:
    # 1. faiss.index - Binary FAISS index with embeddings
    # 2. metadata.pkl - Pickled metadata (sources, chunk info)
    # 3. config.pkl - Configuration (dimension, index type)
    # Saving allows us to reuse the index without reprocessing documents
    pipeline.vector_store.save(vector_store_dir)
    
    # ========================================================================
    # STEP 8: Display completion statistics
    # ========================================================================
    # Get statistics from the vector store
    # Returns a dict with num_vectors, dimension, and index_type
    stats = pipeline.vector_store.get_stats()
    
    # Print success message with detailed statistics
    print(f"\n✅ Setup complete!")
    print(f"   - Total chunks: {stats['num_vectors']}")  # Number of document chunks indexed
    print(f"   - Embedding dimension: {stats['dimension']}")  # Vector size (768 for BGE)
    print(f"   - Vector store saved to: {vector_store_dir}")  # Where files were saved
    
    # Return True to indicate successful completion
    return True


# ============================================================================
# FUNCTION: query_cli
# PURPOSE: Process user queries from command line and display results
# ============================================================================
def query_cli(query: str, 
             vector_store_dir: str = "vector_store",
             top_k: int = 5,
             llm_model: str = "gemma:2b",
             ollama_host: str = "http://localhost:11434"):
    """
    Interactive Query Interface
    ==========================
    
    Processes a user query against the loaded document collection and
    displays the results with full transparency.
    
    Args:
        query (str): User's question
        vector_store_dir (str): Directory containing the vector database
        top_k (int): Number of contexts to retrieve for response
        llm_model (str): Ollama model name (Docker container)
        ollama_host (str): Docker container endpoint
    
    Process:
    1. Load existing vector store from disk
    2. Initialize RAG pipeline with Docker configuration
    3. Process query through complete RAG workflow
    4. Display formatted results with metadata
    
    Output Sections:
    - ANSWER: Generated response text
    - SOURCES: Source documents used
    - METADATA: Confidence scores and validation results
    
    Error Handling:
    - Missing vector store
    - Docker/Ollama connection issues
    - Query processing failures
    """
    # ========================================================================
    # STEP 1: Display header for query processing
    # ========================================================================
    # Print visual separator for clear output formatting
    print("=" * 60)
    print("RAGnarok Query Processing")
    print("=" * 60)
    
    # ========================================================================
    # STEP 2: Validate that vector store exists
    # ========================================================================
    # Check if the vector_store directory exists
    # Without a vector store, we can't perform semantic search
    if not os.path.exists(vector_store_dir):
        # Print error message with helpful instructions
        print(f"❌ Vector store not found at '{vector_store_dir}'")
        print("Please run setup first: python main.py setup")
        
        # Exit function early since we can't proceed without a vector store
        return
    
    # ========================================================================
    # STEP 3: Load pre-built vector store from disk
    # ========================================================================
    # Print progress message
    print(f"Loading vector store from '{vector_store_dir}'...")
    
    # Load the vector store using the class method VectorStore.load()
    # This will:
    # 1. Read config.pkl to get dimension and index type
    # 2. Load faiss.index with all embeddings
    # 3. Load metadata.pkl with document information
    # 4. Return a fully initialized VectorStore instance
    vector_store = VectorStore.load(vector_store_dir)
    
    # ========================================================================
    # STEP 4: Initialize RAG pipeline with loaded vector store
    # ========================================================================
    # Create RAGPipeline instance with custom configuration
    # vector_store: Use the loaded store instead of creating a new one
    # llm_model: Specify which Ollama model to use (default: gemma:2b)
    # ollama_host: Docker container endpoint (default: localhost:11434)
    pipeline = RAGPipeline(
        vector_store=vector_store,      # Pre-loaded vector store
        llm_model=llm_model,             # LLM model name
        ollama_host=ollama_host          # Docker endpoint
    )
    
    # ========================================================================
    # STEP 5: Display query and processing message
    # ========================================================================
    # Show the user's query for confirmation
    print(f"\nQuery: {query}")
    
    # Print processing message (LLM generation can take 1-3 seconds)
    print("\nSearching and generating response...\n")
    
    # ========================================================================
    # STEP 6: Process query through complete RAG pipeline
    # ========================================================================
    # Call generate_response() which performs:
    # 1. Convert query to embedding vector
    # 2. Search FAISS index for similar document chunks
    # 3. Filter results by confidence threshold
    # 4. Build prompt with retrieved contexts
    # 5. Send prompt to Ollama LLM (Docker)
    # 6. Validate response through guardrails
    # 7. Return complete result dictionary
    result = pipeline.generate_response(query, top_k=top_k)
    
    # ========================================================================
    # STEP 7: Display the generated answer
    # ========================================================================
    # Print section header
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    
    # Print the LLM-generated response
    # This is the main answer to the user's question
    print(result['response'])
    print()  # Empty line for spacing
    
    # ========================================================================
    # STEP 8: Display source attribution if available
    # ========================================================================
    # Check if any sources were used in generating the response
    # Sources list contains unique document filenames
    if result['sources']:
        # Print section header
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        
        # Print each source document with a bullet point
        # This shows which documents were used to generate the answer
        for source in result['sources']:
            print(f"• {source}")
        print()  # Empty line for spacing
    
    # ========================================================================
    # STEP 9: Display transparency metadata
    # ========================================================================
    # Print section header
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    
    # Display confidence scores
    # Max Confidence: Highest similarity score among retrieved contexts
    # This indicates how well the best match aligns with the query
    print(f"Max Confidence: {result['confidence']:.3f}")
    
    # Avg Confidence: Average similarity across all retrieved contexts
    # This indicates overall relevance of retrieved information
    print(f"Avg Confidence: {result['average_confidence']:.3f}")
    
    # Display validation results if available
    # Validation dict contains guardrails check results
    if result.get('validation'):
        validation = result['validation']
        
        # is_valid: Overall validation status (passed all checks)
        print(f"Valid: {validation.get('is_valid', 'N/A')}")
        
        # confidence_passed: Whether confidence threshold was met
        print(f"Confidence Passed: {validation.get('confidence_passed', 'N/A')}")



# ============================================================================
# FUNCTION: main
# PURPOSE: CLI entry point - parses arguments and routes to functions
# ============================================================================
def main():
    """
    Main CLI Entry Point
    ===================
    
    Parses command line arguments and routes to appropriate functionality.
    Supports both document setup and interactive querying.
    
    Commands:
    - setup: Process documents and build vector store
    - query: Ask questions about loaded documents
    
    This function provides a complete CLI interface for RAGnarok,
    making it suitable for batch processing and automation.
    """
    # ========================================================================
    # STEP 1: Create main argument parser
    # ========================================================================
    # ArgumentParser: Creates a command-line interface with help text
    # description: Shows when user runs --help
    # epilog: Shows at the bottom of help text with usage example
    parser = argparse.ArgumentParser(
        description="RAGnarok - The End of AI Hallucinations",
        epilog="Example: python main.py setup --documents-dir ./docs"
    )
    
    # ========================================================================
    # STEP 2: Create subparsers for different commands
    # ========================================================================
    # add_subparsers: Allows multiple commands (setup, query) in one CLI
    # dest='command': Stores which command was chosen in args.command
    # help: Description shown in main help text
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ========================================================================
    # STEP 3: Configure 'setup' command parser
    # ========================================================================
    # Create a subparser for the 'setup' command
    # This handles: python main.py setup [options]
    setup_parser = subparsers.add_parser(
        'setup',  # Command name
        help='Process documents and build vector store'  # Help text
    )
    
    # Add --documents-dir argument to setup command
    # This specifies where to find documents to process
    # default='documents': Uses 'documents' folder if not specified
    # help: Description shown in setup --help
    setup_parser.add_argument(
        '--documents-dir',
        default='documents',
        help='Directory containing documents (default: documents)'
    )
    
    # Add --vector-store-dir argument to setup command
    # This specifies where to save the vector database
    # default='vector_store': Standard location for vector store
    setup_parser.add_argument(
        '--vector-store-dir',
        default='vector_store',
        help='Directory to save vector store (default: vector_store)'
    )
    
    # Add --chunk-size argument to setup command
    # This controls how large each text chunk should be
    # type=int: Ensures the value is converted to integer
    # default=512: Optimal size for retrieval (balance of context and precision)
    setup_parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Chunk size in characters (default: 512)'
    )
    
    # Add --chunk-overlap argument to setup command
    # This controls how much chunks overlap to prevent information loss
    # type=int: Ensures the value is converted to integer
    # default=50: Enough overlap to maintain context continuity
    setup_parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Chunk overlap in characters (default: 50)'
    )
    
    # ========================================================================
    # STEP 4: Configure 'query' command parser
    # ========================================================================
    # Create a subparser for the 'query' command
    # This handles: python main.py query "question" [options]
    query_parser = subparsers.add_parser(
        'query',  # Command name
        help='Query the RAG system'  # Help text
    )
    
    # Add positional 'query' argument (required, no -- prefix)
    # This is the user's question
    # Positional arguments are required by default
    query_parser.add_argument(
        'query',
        help='Question to ask about the documents'
    )
    
    # Add --vector-store-dir argument to query command
    # This specifies where to load the vector database from
    # Must match the directory used in setup
    query_parser.add_argument(
        '--vector-store-dir',
        default='vector_store',
        help='Vector store directory (default: vector_store)'
    )
    
    # Add --top-k argument to query command
    # This controls how many document chunks to retrieve
    # type=int: Ensures the value is converted to integer
    # default=5: Good balance between context and noise
    query_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of contexts to retrieve (default: 5)'
    )
    
    # Add --llm-model argument to query command
    # This specifies which Ollama model to use for generation
    # default='gemma:2b': Efficient 2B parameter model from Google
    query_parser.add_argument(
        '--llm-model',
        default='gemma:2b',
        help='Ollama model name (default: gemma:2b)'
    )
    
    # Add --ollama-host argument to query command
    # This specifies the Docker container endpoint for Ollama
    # default='http://localhost:11434': Standard Ollama port
    query_parser.add_argument(
        '--ollama-host',
        default='http://localhost:11434',
        help='Ollama host URL (default: http://localhost:11434)'
    )
    
    # ========================================================================
    # STEP 5: Parse command line arguments
    # ========================================================================
    # parse_args(): Reads sys.argv and converts to namespace object
    # Returns an object with attributes for each argument
    # Example: args.command, args.documents_dir, args.query, etc.
    args = parser.parse_args()
    
    # ========================================================================
    # STEP 6: Route to appropriate function based on command
    # ========================================================================
    # Check which command was specified
    if args.command == 'setup':
        # User ran: python main.py setup [options]
        # Call setup_documents with parsed arguments
        setup_documents(
            documents_dir=args.documents_dir,      # Where to find documents
            vector_store_dir=args.vector_store_dir,  # Where to save vector store
            chunk_size=args.chunk_size,            # Chunk size in characters
            chunk_overlap=args.chunk_overlap       # Overlap between chunks
        )
        
    elif args.command == 'query':
        # User ran: python main.py query "question" [options]
        # Call query_cli with parsed arguments
        query_cli(
            query=args.query,                      # User's question
            vector_store_dir=args.vector_store_dir,  # Where to load vector store
            top_k=args.top_k,                      # Number of contexts to retrieve
            llm_model=args.llm_model,              # Ollama model name
            ollama_host=args.ollama_host           # Docker endpoint
        )
        
    else:
        # No command specified or invalid command
        # Show help text to guide the user
        # This happens when user runs: python main.py (with no arguments)
        parser.print_help()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
# This block only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Call the main function to start the CLI
    main()

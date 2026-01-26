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

import argparse
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from vector_store import VectorStore


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
    print("=" * 60)
    print("RAGnarok Document Processing")
    print("=" * 60)
    
    # Validate documents directory
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created '{documents_dir}' directory.")
        print(f"Please add your documents (PDF, TXT, or Markdown) to '{documents_dir}' and run again.")
        return False
    
    # Initialize document processor with specified chunking parameters
    print(f"\n1. Processing documents from '{documents_dir}'...")
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_directory(documents_dir)
    
    # Validate that documents were successfully processed
    if not chunks:
        print(f"❌ No documents found in '{documents_dir}'")
        print("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
        return False
    
    print(f"✅ Processed {len(chunks)} chunks from documents")
    
    # Initialize RAG pipeline with Docker-aware configuration
    print("\n2. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Generate embeddings and build vector index
    print("3. Generating embeddings and building vector index...")
    pipeline.add_documents(chunks)
    
    # Persist vector store for future use
    print(f"4. Saving vector store to '{vector_store_dir}'...")
    pipeline.vector_store.save(vector_store_dir)
    
    # Display completion statistics
    stats = pipeline.vector_store.get_stats()
    print(f"\n✅ Setup complete!")
    print(f"   - Total chunks: {stats['num_vectors']}")
    print(f"   - Embedding dimension: {stats['dimension']}")
    print(f"   - Vector store saved to: {vector_store_dir}")
    
    return True


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
    print("=" * 60)
    print("RAGnarok Query Processing")
    print("=" * 60)
    
    # Validate vector store exists
    if not os.path.exists(vector_store_dir):
        print(f"❌ Vector store not found at '{vector_store_dir}'")
        print("Please run setup first: python main.py setup")
        return
    
    # Load pre-built vector store
    print(f"Loading vector store from '{vector_store_dir}'...")
    vector_store = VectorStore.load(vector_store_dir)
    
    # Initialize RAG pipeline with Docker-aware configuration
    pipeline = RAGPipeline(
        vector_store=vector_store, 
        llm_model=llm_model, 
        ollama_host=ollama_host
    )
    
    # Process query through complete RAG pipeline
    print(f"\nQuery: {query}")
    print("\nSearching and generating response...\n")
    
    result = pipeline.generate_response(query, top_k=top_k)
    
    # Display formatted results
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result['response'])
    print()
    
    # Show source attribution if available
    if result['sources']:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        for source in result['sources']:
            print(f"• {source}")
        print()
    
    # Display transparency metadata
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Max Confidence: {result['confidence']:.3f}")
    print(f"Avg Confidence: {result['average_confidence']:.3f}")
    if result.get('validation'):
        validation = result['validation']
        print(f"Valid: {validation.get('is_valid', 'N/A')}")
        print(f"Confidence Passed: {validation.get('confidence_passed', 'N/A')}")


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
    parser = argparse.ArgumentParser(
        description="RAGnarok - The End of AI Hallucinations",
        epilog="Example: python main.py setup --documents-dir ./docs"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command for document processing
    setup_parser = subparsers.add_parser(
        'setup', 
        help='Process documents and build vector store'
    )
    setup_parser.add_argument(
        '--documents-dir', 
        default='documents', 
        help='Directory containing documents (default: documents)'
    )
    setup_parser.add_argument(
        '--vector-store-dir', 
        default='vector_store',
        help='Directory to save vector store (default: vector_store)'
    )
    setup_parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=512,
        help='Chunk size in characters (default: 512)'
    )
    setup_parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=50,
        help='Chunk overlap in characters (default: 50)'
    )
    
    # Query command for interactive questioning
    query_parser = subparsers.add_parser(
        'query', 
        help='Query the RAG system'
    )
    query_parser.add_argument(
        'query', 
        help='Question to ask about the documents'
    )
    query_parser.add_argument(
        '--vector-store-dir', 
        default='vector_store',
        help='Vector store directory (default: vector_store)'
    )
    query_parser.add_argument(
        '--top-k', 
        type=int, 
        default=5,
        help='Number of contexts to retrieve (default: 5)'
    )
    query_parser.add_argument(
        '--llm-model', 
        default='gemma:2b',
        help='Ollama model name (default: gemma:2b)'
    )
    query_parser.add_argument(
        '--ollama-host', 
        default='http://localhost:11434',
        help='Ollama host URL (default: http://localhost:11434)'
    )
    
    # Parse arguments and route to appropriate function
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_documents(
            documents_dir=args.documents_dir,
            vector_store_dir=args.vector_store_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    elif args.command == 'query':
        query_cli(
            query=args.query,
            vector_store_dir=args.vector_store_dir,
            top_k=args.top_k,
            llm_model=args.llm_model,
            ollama_host=args.ollama_host
        )
    else:
        # Show help if no command specified
        parser.print_help()


if __name__ == "__main__":
    main()

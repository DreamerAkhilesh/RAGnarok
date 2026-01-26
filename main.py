"""
Main entry point for RAG Application
Can be used as CLI or to initialize the system programmatically.
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
    Process documents and build vector store.
    
    Args:
        documents_dir: Directory containing documents
        vector_store_dir: Directory to save vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print("=" * 60)
    print("RAG System Setup")
    print("=" * 60)
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created '{documents_dir}' directory.")
        print(f"Please add your documents (PDF, TXT, or Markdown) to '{documents_dir}' and run again.")
        return False
    
    # Process documents
    print(f"\n1. Processing documents from '{documents_dir}'...")
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_directory(documents_dir)
    
    if not chunks:
        print(f"❌ No documents found in '{documents_dir}'")
        print("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
        return False
    
    print(f"✅ Processed {len(chunks)} chunks from documents")
    
    # Initialize RAG pipeline
    print("\n2. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Add documents to vector store
    print("3. Generating embeddings and building vector index...")
    pipeline.add_documents(chunks)
    
    # Save vector store
    print(f"4. Saving vector store to '{vector_store_dir}'...")
    pipeline.vector_store.save(vector_store_dir)
    
    stats = pipeline.vector_store.get_stats()
    print(f"\n✅ Setup complete!")
    print(f"   - Total chunks: {stats['num_vectors']}")
    print(f"   - Embedding dimension: {stats['dimension']}")
    print(f"   - Vector store saved to: {vector_store_dir}")
    
    return True


def query_cli(query: str, 
             vector_store_dir: str = "vector_store",
             top_k: int = 5,
             llm_model: str = "llama3"):
    """
    Query the RAG system from command line.
    
    Args:
        query: User query
        vector_store_dir: Directory containing vector store
        top_k: Number of contexts to retrieve
        llm_model: Ollama model name
    """
    print("=" * 60)
    print("RAG Query")
    print("=" * 60)
    
    # Load vector store
    if not os.path.exists(vector_store_dir):
        print(f"❌ Vector store not found at '{vector_store_dir}'")
        print("Please run setup first: python main.py setup")
        return
    
    print(f"Loading vector store from '{vector_store_dir}'...")
    vector_store = VectorStore.load(vector_store_dir)
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_store=vector_store, llm_model=llm_model)
    
    # Query
    print(f"\nQuery: {query}")
    print("\nSearching and generating response...\n")
    
    result = pipeline.generate_response(query, top_k=top_k)
    
    # Display results
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result['response'])
    print()
    
    if result['sources']:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        for source in result['sources']:
            print(f"• {source}")
        print()
    
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
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Process documents and build vector store')
    setup_parser.add_argument('--documents-dir', default='documents', 
                             help='Directory containing documents (default: documents)')
    setup_parser.add_argument('--vector-store-dir', default='vector_store',
                             help='Directory to save vector store (default: vector_store)')
    setup_parser.add_argument('--chunk-size', type=int, default=512,
                             help='Chunk size in characters (default: 512)')
    setup_parser.add_argument('--chunk-overlap', type=int, default=50,
                             help='Chunk overlap in characters (default: 50)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--vector-store-dir', default='vector_store',
                             help='Vector store directory (default: vector_store)')
    query_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of contexts to retrieve (default: 5)')
    query_parser.add_argument('--llm-model', default='llama3',
                             help='Ollama model name (default: llama3)')
    
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
            llm_model=args.llm_model
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
RAGnarok Streamlit Web Application
=================================

Main web interface for RAGnarok's document intelligence system.
Provides an intuitive UI for document loading, query processing, and result visualization.

Key Features:
- Document upload and processing interface
- Real-time query processing with confidence scoring
- Source attribution and context visualization
- System configuration display
- Error handling and user feedback

Architecture:
- Streamlit for reactive web UI
- Session state management for pipeline persistence
- Docker-aware Ollama integration
- Responsive design for various screen sizes

Author: RAGnarok Team
Version: 2.0.0 (Docker + Gemma 2B)
"""

# ============================================================================
# IMPORTS - Required libraries and internal modules
# ============================================================================

# streamlit: Web application framework for data apps
# Provides reactive UI components and session state management
# Automatically handles page reloads and user interactions
import streamlit as st

# os: Operating system interface for file/directory operations
# Used to check if directories and files exist
import os

# Path: Object-oriented filesystem path handling from pathlib
# Provides cleaner path operations than os.path
from pathlib import Path

# DocumentProcessor: Our custom module for loading and chunking documents
# Handles PDF, TXT, and Markdown files
from document_processor import DocumentProcessor

# RAGPipeline: Core orchestration engine for the RAG workflow
# Manages embeddings, vector search, LLM communication
from rag_pipeline import RAGPipeline

# VectorStore: FAISS-based vector database for semantic similarity search
# Stores document embeddings and performs fast nearest-neighbor search
from vector_store import VectorStore

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# These values are fixed for optimal performance and are displayed to users
# for transparency. They represent the best-tested configuration for RAGnarok.
#
# Why constants?
# - Ensures consistent configuration across sessions
# - Makes it easy to update settings in one place
# - Provides transparency to users about system configuration
# - Prevents accidental misconfiguration

# EMBEDDING_MODEL_DEFAULT: Which model to use for text embeddings
# BGE-base-en-v1.5 is state-of-the-art for retrieval tasks
# 768 dimensions, optimized for semantic similarity
EMBEDDING_MODEL_DEFAULT = "BAAI/bge-base-en-v1.5"

# LLM_MODEL_DEFAULT: Which LLM to use for response generation
# Gemma 2B is efficient (4GB RAM) with good reasoning capabilities
# Runs in Ollama Docker container
LLM_MODEL_DEFAULT = "gemma:2b"

# OLLAMA_HOST_DEFAULT: Where the Ollama Docker container is running
# Standard Ollama port is 11434
# Change if running on different host/port
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

# MIN_CONFIDENCE_DEFAULT: Minimum similarity score to accept a context
# 0.5 is balanced - not too strict, not too permissive
# Higher = more conservative, Lower = more permissive
MIN_CONFIDENCE_DEFAULT = 0.5

# TOP_K_DEFAULT: How many document chunks to retrieve for each query
# 5 is optimal - enough context without overwhelming the LLM
# More = more context but slower, Less = faster but less context
TOP_K_DEFAULT = 5

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
# Configure the Streamlit page before any other Streamlit commands
# This must be the first Streamlit command in the script

# set_page_config(): Configures the Streamlit page
# Parameters:
# - page_title: Shows in browser tab
# - page_icon: Emoji or image for browser tab
# - layout: "wide" uses full browser width, "centered" is narrower
st.set_page_config(
    page_title="RAGnarok - The End of AI Hallucinations",  # Browser tab title
    page_icon="⚡",                                          # Lightning bolt emoji
    layout="wide"                                           # Use full width
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit session state persists data across page reloads and interactions
# Without session state, variables would reset on every interaction
#
# Why session state?
# - Keeps RAG pipeline loaded (don't reload model on every query)
# - Remembers if documents are loaded
# - Maintains configuration across interactions
# - Improves performance (no repeated initialization)

# Check if 'rag_pipeline' key exists in session state
# If not, initialize it to None
# This will hold the RAGPipeline instance once initialized
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

# Check if 'documents_loaded' key exists in session state
# If not, initialize it to False
# This tracks whether documents have been processed and loaded
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Check if 'vector_store_path' key exists in session state
# If not, initialize it to "vector_store"
# This is where the FAISS index and metadata are saved
if 'vector_store_path' not in st.session_state:
    st.session_state.vector_store_path = "vector_store"

# ============================================================================
# CORE SYSTEM FUNCTIONS
# ============================================================================

def initialize_rag_pipeline():
    """
    Initialize or Load RAG Pipeline
    ==============================
    
    Sets up the RAG pipeline by either loading an existing vector store
    or creating a new one. This function handles the core system initialization.
    
    Returns:
        bool: True if initialization successful, False otherwise
    
    Process:
    1. Check for existing vector store (saved FAISS index)
    2. If found, load it with existing embeddings and metadata
    3. If not found, create new pipeline ready for document ingestion
    4. Handle errors gracefully with user feedback
    
    Session State Updates:
    - rag_pipeline: The initialized RAGPipeline instance
    - documents_loaded: Boolean indicating if documents are ready
    """
    # ========================================================================
    # STEP 1: Get vector store path from session state
    # ========================================================================
    # Retrieve the path where vector store should be saved/loaded
    # Default: "vector_store" directory
    vector_store_path = st.session_state.vector_store_path
    
    # ========================================================================
    # STEP 2: Check if vector store already exists
    # ========================================================================
    # Check two conditions:
    # 1. Directory exists: os.path.exists(vector_store_path)
    # 2. FAISS index file exists: os.path.exists(.../"faiss.index")
    #
    # Why check both?
    # - Directory might exist but be empty
    # - faiss.index is the main file we need
    # - If both exist, we have a valid vector store to load
    if os.path.exists(vector_store_path) and os.path.exists(os.path.join(vector_store_path, "faiss.index")):
        # ====================================================================
        # STEP 2a: Load existing vector store
        # ====================================================================
        # We have a saved vector store - load it instead of creating new
        try:
            # ================================================================
            # STEP 2a-i: Load vector store from disk
            # ================================================================
            # VectorStore.load(): Class method that:
            # 1. Reads config.pkl (dimension, index type)
            # 2. Loads faiss.index (embeddings)
            # 3. Loads metadata.pkl (document information)
            # 4. Returns fully initialized VectorStore instance
            vector_store = VectorStore.load(vector_store_path)
            
            # ================================================================
            # STEP 2a-ii: Create RAG pipeline with loaded vector store
            # ================================================================
            # Initialize RAGPipeline with:
            # - embedding_model: BGE model for text embeddings
            # - llm_model: Gemma 2B for response generation
            # - ollama_host: Docker container endpoint
            # - vector_store: Pre-loaded vector store (not creating new)
            # - min_confidence: Threshold for filtering contexts
            st.session_state.rag_pipeline = RAGPipeline(
                embedding_model=EMBEDDING_MODEL_DEFAULT,
                llm_model=LLM_MODEL_DEFAULT,
                ollama_host=OLLAMA_HOST_DEFAULT,
                vector_store=vector_store,              # Use loaded store
                min_confidence=MIN_CONFIDENCE_DEFAULT,
            )
            
            # ================================================================
            # STEP 2a-iii: Mark documents as loaded
            # ================================================================
            # Set flag to True since we loaded a vector store with documents
            # This enables the query interface
            st.session_state.documents_loaded = True
            
            # Return True to indicate successful initialization
            return True
            
        # ====================================================================
        # STEP 2b: Handle errors during loading
        # ====================================================================
        except Exception as e:
            # If loading fails (corrupted files, version mismatch, etc.)
            # Display error message to user
            # st.error(): Shows red error box in Streamlit UI
            st.error(f"Error loading vector store: {str(e)}")
            
            # Return False to indicate initialization failed
            return False
    else:
        # ====================================================================
        # STEP 3: Create new pipeline (no existing vector store)
        # ====================================================================
        # No saved vector store found - create a new pipeline
        # This happens on first run or after deleting vector store
        
        # Create RAGPipeline with default configuration
        # vector_store parameter is omitted, so a new one will be created
        # The new vector store will be empty until documents are added
        st.session_state.rag_pipeline = RAGPipeline(
            embedding_model=EMBEDDING_MODEL_DEFAULT,
            llm_model=LLM_MODEL_DEFAULT,
            ollama_host=OLLAMA_HOST_DEFAULT,
            min_confidence=MIN_CONFIDENCE_DEFAULT,
        )
        
        # Mark documents as NOT loaded (vector store is empty)
        st.session_state.documents_loaded = False
        
        # Return True to indicate successful initialization
        return True

def load_documents():
    """
    Process and Load Documents into Vector Store
    ===========================================
    
    Handles the complete document ingestion pipeline from file discovery
    to vector store creation and persistence.
    
    Returns:
        bool: True if documents loaded successfully, False otherwise
    
    Process:
    1. Check for documents directory and create if needed
    2. Process all supported documents (PDF, TXT, MD)
    3. Generate embeddings for all text chunks
    4. Store in FAISS vector database
    5. Save vector store for future sessions
    6. Update session state
    
    Error Handling:
    - Missing documents directory
    - No supported files found
    - Document processing errors
    - Vector store creation failures
    """
    documents_dir = "documents"
    
    # Ensure documents directory exists
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        st.warning(f"Created '{documents_dir}' directory. Please add your documents (PDF, TXT, or Markdown) there.")
        return False
    
    # Initialize document processor with optimal chunking parameters
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Process all documents in the directory
    with st.spinner("Processing documents..."):
        chunks = processor.process_directory(documents_dir)
    
    # Check if any documents were successfully processed
    if not chunks:
        st.error("No documents found or processed. Please add PDF, TXT, or Markdown files to the 'documents' folder.")
        return False
    
    # Add processed chunks to the vector store
    with st.spinner("Generating embeddings and building vector index..."):
        st.session_state.rag_pipeline.add_documents(chunks)
    
    # Persist vector store for future sessions
    st.session_state.rag_pipeline.vector_store.save(st.session_state.vector_store_path)
    
    # Update UI and session state
    st.success(f"Successfully loaded {len(chunks)} document chunks from {documents_dir}!")
    st.session_state.documents_loaded = True
    return True

# =============================================================================
# MAIN USER INTERFACE
# =============================================================================
st.title("⚡ RAGnarok")
st.markdown("**The End of AI Hallucinations - Grounded Document Intelligence**")


info_cols = st.columns(4)
info_cols[0].metric("Embedding Model", EMBEDDING_MODEL_DEFAULT)
info_cols[1].metric("LLM Model", LLM_MODEL_DEFAULT)
info_cols[2].metric("Min Confidence", f"{MIN_CONFIDENCE_DEFAULT:.2f}")
info_cols[3].metric("Top-K", TOP_K_DEFAULT)

st.divider()

# Document management (main screen)
doc_col1, doc_col2 = st.columns([1, 3])
with doc_col1:
    if st.button("Load / Reload Documents", type="primary", use_container_width=True):
        if initialize_rag_pipeline() and load_documents():
            st.rerun()
with doc_col2:
    if st.session_state.documents_loaded:
        stats = st.session_state.rag_pipeline.vector_store.get_stats()
        st.success(f"{stats['num_vectors']} chunks ready for search")
    else:
        st.info("Add PDF/TXT/MD files to `documents/` then click Load.")

st.caption("Flow: Query → Embedding → Vector Search → Context Retrieval → LLM Answer")

# Initialize pipeline
if st.session_state.rag_pipeline is None:
    initialize_rag_pipeline()

# Main content area
if not st.session_state.documents_loaded:
    st.info("Please load documents before asking questions.")
    
    # Show document directory info
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        files = list(Path(documents_dir).glob("*"))
        supported_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.md', '.markdown']]
        
        if supported_files:
            st.subheader("Available Documents")
            for file in supported_files:
                st.text(f"• {file.name}")
        else:
            st.warning(f"No supported documents found in '{documents_dir}' folder.")
            st.caption("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
else:
    # Query interface
    st.subheader("Ask a Question")
    
    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., What is the main topic discussed in the documents?",
        label_visibility="collapsed",
    )
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching documents and generating response..."):
            # Ensure pipeline uses the fixed configuration
            st.session_state.rag_pipeline.llm_model = LLM_MODEL_DEFAULT
            st.session_state.rag_pipeline.ollama_host = OLLAMA_HOST_DEFAULT
            st.session_state.rag_pipeline.guardrails.min_confidence = MIN_CONFIDENCE_DEFAULT
            
            # Generate response
            result = st.session_state.rag_pipeline.generate_response(
                query=query,
                top_k=TOP_K_DEFAULT,
                use_guardrails=True
            )
        
        # Display response
        st.subheader("Answer")
        st.markdown(result['response'])
        
        # Display sources
        if result['sources']:
            st.subheader("Sources")
            for source in result['sources']:
                st.caption(f"• {source}")
        
        # Compact retrieval details
        details_col1, details_col2 = st.columns(2)
        details_col1.metric("Max Confidence", f"{result['confidence']:.3f}")
        details_col2.metric("Avg Confidence", f"{result['average_confidence']:.3f}")
        
        if result.get('contexts'):
            st.subheader("Top Retrieved Contexts")
            for i, ctx in enumerate(result['contexts'][:3], 1):
                st.text(f"[{ctx.get('source', 'Unknown')}] {ctx['text'][:500]}{'...' if len(ctx['text']) > 500 else ''}")

# Footer
st.divider()
st.caption("Built with open-source tools: Sentence Transformers, FAISS, Ollama Docker, Streamlit")

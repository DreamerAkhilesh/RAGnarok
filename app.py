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
    page_title="RAGnarok - The End of AI Hallucinations",
    page_icon="⚡",
    layout="wide"
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

# ============================================================================
# MAIN USER INTERFACE
# ============================================================================
# This section creates the complete Streamlit web interface
# Layout: Header → Config Display → Document Management → Query Interface

# ============================================================================
# SECTION 1: PAGE HEADER AND BRANDING
# ============================================================================

# Display main title with lightning bolt emoji
# st.title(): Creates large, prominent heading
# ⚡ emoji represents speed and power of the system
st.title("⚡ RAGnarok")

# Display tagline/subtitle
# st.markdown(): Renders Markdown text
# **text**: Makes text bold in Markdown
# This communicates the core value proposition
st.markdown("**The End of AI Hallucinations - Grounded Document Intelligence**")

# ============================================================================
# SECTION 2: SYSTEM CONFIGURATION DISPLAY
# ============================================================================
# Shows users the current system configuration for transparency
# Helps users understand what models and settings are being used

# Create 4 equal-width columns for metrics
# st.columns(4): Divides horizontal space into 4 equal parts
# Returns list of column objects: [col0, col1, col2, col3]
info_cols = st.columns(4)

# Column 0: Display embedding model
# metric(): Shows a metric card with label and value
# EMBEDDING_MODEL_DEFAULT: "BAAI/bge-base-en-v1.5"
# This tells users which model converts text to vectors
info_cols[0].metric("Embedding Model", EMBEDDING_MODEL_DEFAULT)

# Column 1: Display LLM model
# LLM_MODEL_DEFAULT: "gemma:2b"
# This tells users which model generates responses
# Running in Ollama Docker container
info_cols[1].metric("LLM Model", LLM_MODEL_DEFAULT)

# Column 2: Display minimum confidence threshold
# MIN_CONFIDENCE_DEFAULT: 0.5
# f"{value:.2f}": Format float to 2 decimal places
# This tells users the similarity threshold for filtering contexts
info_cols[2].metric("Min Confidence", f"{MIN_CONFIDENCE_DEFAULT:.2f}")

# Column 3: Display top-k retrieval count
# TOP_K_DEFAULT: 5
# This tells users how many contexts are retrieved per query
info_cols[3].metric("Top-K", TOP_K_DEFAULT)

# Add visual separator line
# st.divider(): Creates horizontal line to separate sections
# Improves visual organization of the interface
st.divider()

# ============================================================================
# SECTION 3: DOCUMENT MANAGEMENT INTERFACE
# ============================================================================
# Allows users to load/reload documents and see loading status
# Two-column layout: Button on left, Status on right

# Create 2 columns with 1:3 width ratio
# [1, 3]: Left column is 1/4 width, right column is 3/4 width
# This gives more space to the status message
doc_col1, doc_col2 = st.columns([1, 3])

# ============================================================================
# COLUMN 1: DOCUMENT LOADING BUTTON
# ============================================================================
# with statement: All Streamlit commands inside go to this column
with doc_col1:
    # Create button for loading/reloading documents
    # st.button(): Creates clickable button
    # Returns True when clicked, False otherwise
    #
    # Parameters:
    # - "Load / Reload Documents": Button text
    # - type="primary": Makes button blue/prominent
    # - use_container_width=True: Button fills column width
    #
    # Why "Load / Reload"?
    # - "Load": For first-time document loading
    # - "Reload": For updating when documents change
    if st.button("Load / Reload Documents", type="primary", use_container_width=True):
        # Button was clicked - process documents
        
        # Call two functions in sequence:
        # 1. initialize_rag_pipeline(): Sets up or loads pipeline
        # 2. load_documents(): Processes documents and creates embeddings
        #
        # and operator: Both must return True to proceed
        # If either fails, st.rerun() is not called
        if initialize_rag_pipeline() and load_documents():
            # Both functions succeeded
            
            # st.rerun(): Reloads the entire Streamlit app
            # Why rerun?
            # - Updates UI with new document count
            # - Enables query interface
            # - Refreshes all components with new state
            # - Provides immediate feedback to user
            st.rerun()

# ============================================================================
# COLUMN 2: DOCUMENT LOADING STATUS
# ============================================================================
# Shows current status: documents loaded or instructions to load
with doc_col2:
    # Check if documents have been loaded
    # st.session_state.documents_loaded: Boolean flag set during loading
    if st.session_state.documents_loaded:
        # ====================================================================
        # Documents are loaded - show success status
        # ====================================================================
        
        # Get statistics from vector store
        # get_stats(): Returns dict with num_vectors, dimension, index_type
        stats = st.session_state.rag_pipeline.vector_store.get_stats()
        
        # Display success message with chunk count
        # st.success(): Shows green success box
        # stats['num_vectors']: Number of document chunks in vector store
        # Example: "37 chunks ready for search"
        #
        # Why show chunk count?
        # - Confirms documents are loaded
        # - Shows scale of knowledge base
        # - Provides transparency
        st.success(f"{stats['num_vectors']} chunks ready for search")
    else:
        # ====================================================================
        # Documents not loaded - show instructions
        # ====================================================================
        
        # Display informational message
        # st.info(): Shows blue info box
        # Tells users what to do next
        # Clear, actionable instructions
        st.info("Add PDF/TXT/MD files to `documents/` then click Load.")

# Add caption explaining the RAG workflow
# st.caption(): Shows small, gray text
# Educates users about the system's process
# Flow visualization helps users understand what happens
st.caption("Flow: Query → Embedding → Vector Search → Context Retrieval → LLM Answer")

# ============================================================================
# SECTION 4: PIPELINE INITIALIZATION CHECK
# ============================================================================
# Ensures pipeline is initialized before showing query interface
# Defensive programming: Handles case where pipeline wasn't initialized

# Check if pipeline exists in session state
# None: Pipeline hasn't been initialized yet
if st.session_state.rag_pipeline is None:
    # Pipeline doesn't exist - initialize it now
    # This happens on first page load
    # Creates empty pipeline ready for document loading
    initialize_rag_pipeline()

# ============================================================================
# SECTION 5: MAIN CONTENT AREA - CONDITIONAL DISPLAY
# ============================================================================
# Shows different content based on whether documents are loaded
# Two states: 1) Documents not loaded, 2) Documents loaded (query interface)

# Check document loading status
# This determines which interface to show
if not st.session_state.documents_loaded:
    # ========================================================================
    # STATE 1: DOCUMENTS NOT LOADED - SHOW INSTRUCTIONS
    # ========================================================================
    # User needs to load documents before querying
    # Show helpful information about available documents
    
    # Display informational message
    # st.info(): Blue info box
    # Clear instruction for what user needs to do
    st.info("Please load documents before asking questions.")
    
    # ========================================================================
    # Show available documents in the directory
    # ========================================================================
    # Helps users see what files are available to load
    # Provides feedback about what will be processed
    
    # Define documents directory path
    documents_dir = "documents"
    
    # Check if documents directory exists
    # os.path.exists(): Returns True if path exists
    if os.path.exists(documents_dir):
        # Directory exists - scan for files
        
        # Get all files in directory
        # Path(documents_dir).glob("*"): Returns iterator of all items
        # list(): Convert iterator to list
        # This includes all files and subdirectories
        files = list(Path(documents_dir).glob("*"))
        
        # Filter for supported file types
        # List comprehension: [item for item in list if condition]
        # f.suffix.lower(): Get file extension in lowercase
        # in ['.pdf', '.txt', '.md', '.markdown']: Check if supported
        #
        # Why filter?
        # - Only show files that can be processed
        # - Avoid confusing users with unsupported files
        # - Provide clear expectations
        supported_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.md', '.markdown']]
        
        # Check if any supported files were found
        if supported_files:
            # ================================================================
            # Supported files found - display list
            # ================================================================
            
            # Display subheader
            # st.subheader(): Medium-sized heading
            st.subheader("Available Documents")
            
            # Loop through each supported file
            # Displays each filename with bullet point
            for file in supported_files:
                # st.text(): Display plain text (no formatting)
                # file.name: Just the filename (not full path)
                # f"• {file.name}": Bullet point + filename
                #
                # Example output:
                # • document1.pdf
                # • notes.txt
                # • readme.md
                st.text(f"• {file.name}")
        else:
            # ================================================================
            # No supported files found - show warning
            # ================================================================
            
            # Display warning message
            # st.warning(): Yellow warning box
            # Tells user directory is empty or has wrong file types
            st.warning(f"No supported documents found in '{documents_dir}' folder.")
            
            # Display caption with supported formats
            # st.caption(): Small gray text
            # Educates user about what file types are supported
            st.caption("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
else:
    # ========================================================================
    # STATE 2: DOCUMENTS LOADED - SHOW QUERY INTERFACE
    # ========================================================================
    # Documents are loaded and ready for querying
    # Display complete query interface with input and results
    
    # ========================================================================
    # QUERY INPUT SECTION
    # ========================================================================
    
    # Display section heading
    # st.subheader(): Medium-sized heading
    # Clear label for the query input area
    st.subheader("Ask a Question")
    
    # Create text input area for user query
    # st.text_area(): Multi-line text input widget
    #
    # Parameters:
    # - "Enter your question": Label (hidden by label_visibility)
    # - placeholder: Gray text shown when empty
    # - label_visibility="collapsed": Hides the label
    #
    # Why text_area instead of text_input?
    # - Allows longer questions
    # - More comfortable for typing
    # - Can see full question while typing
    #
    # Returns: String containing user's input
    # Empty string if nothing entered
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., What is the main topic discussed in the documents?",
        label_visibility="collapsed",
    )
    
    # ========================================================================
    # QUERY PROCESSING SECTION
    # ========================================================================
    # Triggered when user clicks Search button with non-empty query
    
    # Create search button and check conditions
    # st.button(): Creates clickable button
    # Returns True when clicked
    #
    # Conditions:
    # 1. st.button("Search", type="primary"): Button clicked
    # 2. and query: Query is not empty
    #
    # Both must be True to proceed
    # Prevents processing empty queries
    if st.button("Search", type="primary") and query:
        # ====================================================================
        # User clicked Search with valid query - process it
        # ====================================================================
        
        # Show spinner during processing
        # with st.spinner(): Context manager for loading indicator
        # Displays animated spinner with message
        # Automatically disappears when block completes
        #
        # Why spinner?
        # - Processing takes 1-3 seconds
        # - Provides visual feedback
        # - Prevents user confusion
        # - Shows system is working
        with st.spinner("Searching documents and generating response..."):
            # ================================================================
            # STEP 1: Ensure pipeline configuration is correct
            # ================================================================
            # Reset configuration to defaults in case it was modified
            # Defensive programming: Ensures consistent behavior
            
            # Set LLM model to default
            # Ensures we're using the correct model (gemma:2b)
            st.session_state.rag_pipeline.llm_model = LLM_MODEL_DEFAULT
            
            # Set Ollama host to default
            # Ensures we're connecting to correct Docker container
            st.session_state.rag_pipeline.ollama_host = OLLAMA_HOST_DEFAULT
            
            # Set confidence threshold to default
            # Ensures consistent filtering behavior
            st.session_state.rag_pipeline.guardrails.min_confidence = MIN_CONFIDENCE_DEFAULT
            
            # ================================================================
            # STEP 2: Generate response through RAG pipeline
            # ================================================================
            # This is the main processing step
            # Calls the complete RAG workflow:
            # 1. Convert query to embedding
            # 2. Search vector store for similar chunks
            # 3. Filter by confidence threshold
            # 4. Build prompt with contexts
            # 5. Send to LLM (Ollama/Gemma)
            # 6. Validate response
            # 7. Return formatted result
            #
            # Parameters:
            # - query: User's question string
            # - top_k: Number of contexts to retrieve (5)
            # - use_guardrails: Enable safety validation (True)
            #
            # Returns: Dictionary with:
            # - response: Generated answer text
            # - sources: List of source documents
            # - contexts: Retrieved document chunks
            # - confidence: Max similarity score
            # - average_confidence: Mean similarity score
            # - validation: Guardrails validation results
            result = st.session_state.rag_pipeline.generate_response(
                query=query,
                top_k=TOP_K_DEFAULT,
                use_guardrails=True
            )
        
        # ====================================================================
        # STEP 3: Display the generated answer
        # ====================================================================
        # Show the main response to user's question
        
        # Display section heading
        # st.subheader(): Medium-sized heading
        st.subheader("Answer")
        
        # Display the response text
        # st.markdown(): Renders Markdown text
        # result['response']: The generated answer string
        #
        # Why markdown?
        # - Supports formatting (bold, italic, lists)
        # - LLM might generate formatted text
        # - Better readability than plain text
        st.markdown(result['response'])
        
        # ====================================================================
        # STEP 4: Display source attribution
        # ====================================================================
        # Shows which documents were used to generate the answer
        # Critical for transparency and fact-checking
        
        # Check if any sources were used
        # result['sources']: List of source document filenames
        # Empty list if no sources (shouldn't happen normally)
        if result['sources']:
            # Sources exist - display them
            
            # Display section heading
            st.subheader("Sources")
            
            # Loop through each source document
            # Displays each source with bullet point
            for source in result['sources']:
                # st.caption(): Small gray text
                # f"• {source}": Bullet point + filename
                #
                # Why caption instead of text?
                # - Less prominent than main answer
                # - Gray color indicates metadata
                # - Consistent with UI design
                #
                # Example output:
                # • network_topologies.txt
                # • document.pdf
                st.caption(f"• {source}")
        
        # ====================================================================
        # STEP 5: Display confidence metrics
        # ====================================================================
        # Shows similarity scores for transparency
        # Helps users assess answer reliability
        
        # Create 2 equal-width columns for metrics
        # st.columns(2): Divides space into 2 equal parts
        details_col1, details_col2 = st.columns(2)
        
        # Column 1: Display maximum confidence score
        # metric(): Shows metric card with label and value
        # result['confidence']: Highest similarity score
        # f"{value:.3f}": Format to 3 decimal places
        #
        # What it means:
        # - 0.85: Very relevant context found
        # - 0.60: Moderately relevant context
        # - 0.40: Low relevance (might be filtered)
        details_col1.metric("Max Confidence", f"{result['confidence']:.3f}")
        
        # Column 2: Display average confidence score
        # result['average_confidence']: Mean of all similarity scores
        # Indicates overall relevance of retrieved contexts
        #
        # Why show both max and average?
        # - Max: Best match quality
        # - Average: Overall context quality
        # - Together: Complete picture of retrieval quality
        details_col2.metric("Avg Confidence", f"{result['average_confidence']:.3f}")
        
        # ====================================================================
        # STEP 6: Display retrieved contexts (optional)
        # ====================================================================
        # Shows the actual document chunks that were used
        # Useful for debugging and understanding the answer
        
        # Check if contexts were retrieved
        # result.get('contexts'): Safely get contexts (None if missing)
        if result.get('contexts'):
            # Contexts exist - display them
            
            # Display section heading
            st.subheader("Top Retrieved Contexts")
            
            # Loop through first 3 contexts
            # enumerate(list, start): Returns (index, item) pairs
            # [:3]: Slice to get only first 3 contexts
            # start=1: Start counting from 1 instead of 0
            #
            # Why only 3?
            # - Prevents UI clutter
            # - Shows most relevant contexts
            # - User can see what influenced the answer
            for i, ctx in enumerate(result['contexts'][:3], 1):
                # Display context with source and truncated text
                # st.text(): Plain text display
                #
                # Format: [source] text...
                # - ctx.get('source', 'Unknown'): Source filename
                # - ctx['text'][:500]: First 500 characters
                # - '...' if len(ctx['text']) > 500: Add ellipsis if truncated
                #
                # Why truncate?
                # - Contexts can be long (512 chars)
                # - 500 chars is enough to understand content
                # - Keeps UI manageable
                st.text(f"[{ctx.get('source', 'Unknown')}] {ctx['text'][:500]}{'...' if len(ctx['text']) > 500 else ''}")

# ============================================================================
# SECTION 6: FOOTER
# ============================================================================
# Shows attribution and technology stack
# Provides credit to open-source tools used

# Add visual separator
# st.divider(): Horizontal line
# Separates main content from footer
st.divider()

# Display footer text
# st.caption(): Small gray text
# Lists the open-source technologies powering RAGnarok
#
# Technologies mentioned:
# - Sentence Transformers: For embeddings (BGE model)
# - FAISS: For vector similarity search
# - Ollama Docker: For LLM inference (Gemma 2B)
# - Streamlit: For web interface
#
# Why show this?
# - Transparency about technology stack
# - Credit to open-source community
# - Helps users understand the system
# - Professional presentation
st.caption("Built with open-source tools: Sentence Transformers, FAISS, Ollama Docker, Streamlit")

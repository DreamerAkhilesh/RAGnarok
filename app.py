"""
Streamlit UI for RAG Application
Provides a user-friendly web interface for the Knowledge Assistant.
"""

import streamlit as st
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from vector_store import VectorStore

# Fixed configuration values (kept simple and visible on the main screen)
EMBEDDING_MODEL_DEFAULT = "BAAI/bge-base-en-v1.5"
LLM_MODEL_DEFAULT = "llama3"
MIN_CONFIDENCE_DEFAULT = 0.5
TOP_K_DEFAULT = 5

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'vector_store_path' not in st.session_state:
    st.session_state.vector_store_path = "vector_store"

def initialize_rag_pipeline():
    """Initialize or load the RAG pipeline."""
    vector_store_path = st.session_state.vector_store_path
    
    if os.path.exists(vector_store_path) and os.path.exists(os.path.join(vector_store_path, "faiss.index")):
        # Load existing vector store
        try:
            vector_store = VectorStore.load(vector_store_path)
            st.session_state.rag_pipeline = RAGPipeline(
                embedding_model=EMBEDDING_MODEL_DEFAULT,
                llm_model=LLM_MODEL_DEFAULT,
                vector_store=vector_store,
                min_confidence=MIN_CONFIDENCE_DEFAULT,
            )
            st.session_state.documents_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    else:
        # Create new pipeline (will create vector store when documents are added)
        st.session_state.rag_pipeline = RAGPipeline(
            embedding_model=EMBEDDING_MODEL_DEFAULT,
            llm_model=LLM_MODEL_DEFAULT,
            min_confidence=MIN_CONFIDENCE_DEFAULT,
        )
        st.session_state.documents_loaded = False
        return True

def load_documents():
    """Load and process documents."""
    documents_dir = "documents"
    
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        st.warning(f"Created '{documents_dir}' directory. Please add your documents (PDF, TXT, or Markdown) there.")
        return False
    
    # Process documents
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    with st.spinner("Processing documents..."):
        chunks = processor.process_directory(documents_dir)
    
    if not chunks:
        st.error("No documents found or processed. Please add PDF, TXT, or Markdown files to the 'documents' folder.")
        return False
    
    # Add to vector store
    with st.spinner("Generating embeddings and building vector index..."):
        st.session_state.rag_pipeline.add_documents(chunks)
    
    # Save vector store
    st.session_state.rag_pipeline.vector_store.save(st.session_state.vector_store_path)
    
    st.success(f"Successfully loaded {len(chunks)} document chunks from {documents_dir}!")
    st.session_state.documents_loaded = True
    return True

# Main UI
st.title("RAG Knowledge Assistant")
st.markdown("**Answer questions based strictly on your provided documents**")


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
st.caption("Built with open-source tools: Sentence Transformers, FAISS, Ollama, Streamlit")

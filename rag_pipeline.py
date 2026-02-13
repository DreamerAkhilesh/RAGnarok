"""
RAG Pipeline Module - Core Orchestration Engine
==============================================

This module implements the complete Retrieval-Augmented Generation (RAG) pipeline
that powers RAGnarok's hallucination-free document question answering system.

Key Components:
- Document retrieval using semantic similarity search
- Context assembly with source attribution
- LLM response generation with strict grounding
- Multi-layer validation and guardrails

Architecture Flow:
Query → Embedding → Vector Search → Context Assembly → LLM Generation → Validation → Response

Author: RAGnarok Team
Version: 2.0.0 (Docker + Gemma 2B)
"""

# ============================================================================
# IMPORTS - External libraries and internal modules
# ============================================================================

# typing: Type hints for better code documentation
# List: Type hint for list objects
# Dict: Type hint for dictionary objects
# Optional: Type hint for values that can be None
# Tuple: Type hint for tuple objects
from typing import List, Dict, Optional, Tuple

# ollama: Python client library for Ollama LLM service
# Communicates with Ollama Docker container to generate responses
# Supports both chat and generate APIs
import ollama

# EmbeddingGenerator: Our custom module for text-to-vector conversion
# Converts queries and documents into 768-dimensional semantic vectors
from embeddings import EmbeddingGenerator

# VectorStore: Our custom FAISS-based vector database
# Stores document embeddings and performs fast similarity search
from vector_store import VectorStore

# Guardrails: Our custom safety and validation system
# Filters low-confidence results and validates response grounding
from guardrails import Guardrails


class RAGPipeline:
    """
    Main RAG Pipeline Class
    ======================
    
    Orchestrates the complete RAG workflow from query to grounded response.
    Integrates with Docker-based Ollama for LLM inference and FAISS for vector search.
    
    Key Features:
    - Semantic document retrieval
    - Context-aware response generation
    - Hallucination prevention through guardrails
    - Source attribution for all responses
    - Docker-based LLM deployment support
    """

    # ========================================================================
    # INITIALIZATION METHOD
    # ========================================================================
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "gemma:2b",
                 vector_store: Optional[VectorStore] = None,
                 min_confidence: float = 0.5,
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize RAG Pipeline
        ======================
        
        Sets up all components needed for the RAG workflow including embeddings,
        vector store, LLM connection, and safety guardrails.
        
        Args:
            embedding_model (str): Hugging Face model for text embeddings
                                 Default: BGE-base-en-v1.5 (optimized for retrieval)
            llm_model (str): Ollama model name for text generation
                           Default: gemma:2b (efficient Google model)
            vector_store (VectorStore): Pre-existing vector store or None for new
            min_confidence (float): Minimum similarity threshold for retrieval
                                  Default: 0.5 (balanced precision/recall)
            ollama_host (str): Docker container endpoint for Ollama service
                             Default: http://localhost:11434
        
        Components Initialized:
        - EmbeddingGenerator: Converts text to semantic vectors
        - VectorStore: FAISS database for similarity search
        - Guardrails: Safety system for response validation
        """
        # ====================================================================
        # STEP 1: Initialize embedding system
        # ====================================================================
        # Create EmbeddingGenerator instance with specified model
        # This will:
        # 1. Download model from Hugging Face if not cached (~400MB)
        # 2. Load tokenizer, transformer, and pooling layers
        # 3. Set up on CPU (can be changed to GPU)
        #
        # The embedding generator is used for:
        # - Converting document chunks to vectors (during setup)
        # - Converting user queries to vectors (during retrieval)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        # ====================================================================
        # STEP 2: Store LLM configuration
        # ====================================================================
        # Save LLM model name for later use in generate_response()
        # Examples: "gemma:2b", "llama3:8b", "mistral:7b"
        # This is the model running in the Ollama Docker container
        self.llm_model = llm_model
        
        # Save Ollama host URL for Docker communication
        # Default: http://localhost:11434 (standard Ollama port)
        # This is where the Ollama Docker container is listening
        self.ollama_host = ollama_host
        
        # ====================================================================
        # STEP 3: Initialize safety guardrails system
        # ====================================================================
        # Create Guardrails instance with confidence threshold
        # Guardrails provide:
        # - Confidence filtering (removes low-relevance contexts)
        # - Response validation (checks grounding in documents)
        # - Hallucination detection (monitors refusal keywords)
        # - Warning generation (alerts users to low confidence)
        #
        # min_confidence: Minimum similarity score to accept a context
        # - 0.5: Balanced threshold (default)
        # - Higher (0.7+): More conservative, fewer false positives
        # - Lower (0.3-): More permissive, more results
        self.guardrails = Guardrails(min_confidence=min_confidence)
        
        # ====================================================================
        # STEP 4: Set up vector store for document retrieval
        # ====================================================================
        # Check if a pre-existing vector store was provided
        if vector_store is None:
            # No vector store provided - create a new one
            # This happens during initial setup or when starting fresh
            
            # Get embedding dimension from the generator
            # BGE-base: 768, BGE-large: 1024, MiniLM: 384
            dimension = self.embedding_generator.embedding_dimension
            
            # Create new VectorStore with:
            # - dimension: Size of embedding vectors (768 for BGE-base)
            # - index_type: "cosine" for cosine similarity search
            #   (uses IndexFlatIP with normalized vectors)
            self.vector_store = VectorStore(dimension=dimension, index_type="cosine")
        else:
            # Use the provided vector store
            # This happens when loading from disk:
            # vector_store = VectorStore.load("vector_store/")
            # pipeline = RAGPipeline(vector_store=vector_store)
            self.vector_store = vector_store
    
    def build_prompt(self, query: str, contexts: List[Dict], max_context_length: int = 2000) -> str:
        """
        Construct Grounded Prompt for LLM
        ================================
        
        Builds a carefully engineered prompt that enforces strict grounding to retrieved
        document contexts. This is critical for preventing hallucinations.
        
        Args:
            query (str): User's question
            contexts (List[Dict]): Retrieved document chunks with metadata
            max_context_length (int): Maximum characters for context section
        
        Returns:
            str: Complete prompt with system instructions and formatted contexts
            
        Prompt Engineering Strategy:
        1. Explicit system instructions for grounding
        2. Formatted contexts with source attribution
        3. Clear refusal patterns for missing information
        4. Structured format for consistent responses
        """
        # Format contexts with source attribution and length management
        context_texts = []
        total_length = 0
        
        for ctx in contexts:
            text = ctx['text']
            source = ctx.get('source', 'Unknown')
            
            # Manage context length to fit within LLM limits
            if total_length + len(text) > max_context_length:
                # Truncate last context if needed, but only if meaningful
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if substantial content remains
                    text = text[:remaining] + "..."
                    context_texts.append(f"[Source: {source}]\n{text}")
                break
            
            # Add context with clear source attribution
            context_texts.append(f"[Source: {source}]\n{text}")
            total_length += len(text)
        
        # Join contexts with clear separators
        contexts_str = "\n\n---\n\n".join(context_texts)
        
        # Build system prompt with strict grounding instructions
        # This is the core of hallucination prevention
        system_instruction = """You are a helpful knowledge assistant that answers questions STRICTLY based on the provided documents. 

CRITICAL RULES:
1. Answer ONLY using information from the provided context documents
2. If the answer is not in the provided documents, explicitly state "Based on the provided documents, I cannot find information about [topic]"
3. Do NOT make up information or use external knowledge
4. Cite the source document when referencing specific information
5. If multiple sources contain relevant information, synthesize them clearly
6. Be concise but complete in your answers

Context Documents:
{contexts}

User Question: {query}

Answer (based ONLY on the provided documents):"""

        # Format final prompt with contexts and query
        prompt = system_instruction.format(
            contexts=contexts_str,
            query=query
        )
        
        return prompt
    
    def retrieve_contexts(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Semantic Document Retrieval
        ===========================
        
        Performs semantic similarity search to find the most relevant document
        chunks for answering the user's query.
        
        Args:
            query (str): User's question
            top_k (int): Number of most similar contexts to retrieve
        
        Returns:
            List[Tuple[Dict, float]]: List of (context_metadata, similarity_score) pairs
            
        Process:
        1. Convert query to embedding vector using BGE model
        2. Search FAISS index for most similar document chunks
        3. Return ranked results with similarity scores
        
        Note: Higher similarity scores indicate better relevance to the query
        """
        # Generate query embedding using the same model as documents
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store for most similar document chunks
        # FAISS performs efficient cosine similarity search
        results = self.vector_store.search(query_embedding, k=top_k)
        
        return results
    
    def generate_response(self, query: str, top_k: int = 5, 
                         use_guardrails: bool = True) -> Dict:
        """
        Complete RAG Response Generation
        ===============================
        
        Main method that orchestrates the entire RAG pipeline from query to
        final grounded response with source attribution.
        
        Args:
            query (str): User's question
            top_k (int): Number of contexts to retrieve for response generation
            use_guardrails (bool): Whether to apply safety validation
        
        Returns:
            Dict: Complete response with metadata including:
                - response: Generated answer text
                - sources: List of source documents used
                - contexts: Retrieved document chunks
                - confidence: Maximum similarity score
                - average_confidence: Mean similarity across contexts
                - validation: Guardrails validation results
        
        Pipeline Flow:
        Query → Retrieval → Filtering → Prompt Building → LLM Generation → Validation
        """
        # Step 1: Retrieve relevant document contexts using semantic search
        retrieval_results = self.retrieve_contexts(query, top_k=top_k)
        
        # Handle case where no documents are found
        if not retrieval_results:
            return {
                'response': "I couldn't find any relevant documents to answer your question. Please ensure documents have been loaded.",
                'sources': [],
                'confidence': 0.0,
                'validation': {
                    'is_valid': False,
                    'confidence_passed': False
                }
            }
        
        # Extract contexts and similarity scores from retrieval results
        contexts = [meta for meta, score in retrieval_results]
        scores = [score for meta, score in retrieval_results]
        
        # Step 2: Apply guardrails to filter low-confidence results
        if use_guardrails:
            filtered_results = self.guardrails.filter_low_confidence(retrieval_results)
            
            # If no contexts pass confidence threshold, return explicit refusal
            if not filtered_results:
                return {
                    'response': f"I couldn't find documents with sufficient relevance to answer your question. The best match had a confidence score of {max(scores):.2f}, which is below the threshold of {self.guardrails.min_confidence:.2f}.",
                    'sources': [ctx.get('source', 'Unknown') for ctx in contexts],
                    'confidence': max(scores),
                    'validation': {
                        'is_valid': False,
                        'confidence_passed': False,
                        'average_confidence': sum(scores) / len(scores),
                        'max_confidence': max(scores)
                    }
                }
            
            # Update contexts to only high-confidence results
            contexts = [meta for meta, score in filtered_results]
            scores = [score for meta, score in filtered_results]
        
        # Step 3: Build grounded prompt with retrieved contexts
        prompt = self.build_prompt(query, contexts)
        
        # Step 4: Generate response using Docker-based Ollama LLM
        try:
            # Configure ollama client with Docker container endpoint
            client = ollama.Client(host=self.ollama_host)
            
            # Try chat API first (preferred for conversation), fallback to generate
            try:
                response = client.chat(
                    model=self.llm_model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    options={
                        'temperature': 0.1,  # Low temperature for factual consistency
                        'top_p': 0.9,        # Nucleus sampling for quality
                    }
                )
                response_text = response['message']['content']
            except (AttributeError, KeyError):
                # Fallback to generate API for older Ollama versions
                response = client.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={
                        'temperature': 0.1,  # Deterministic responses
                        'top_p': 0.9,
                    }
                )
                response_text = response.get('response', str(response))
                
        except Exception as e:
            # Handle Docker/Ollama connection errors gracefully
            response_text = f"Error generating response: {str(e)}. Please ensure Ollama is running on {self.ollama_host} and the model '{self.llm_model}' is available."
        
        # Step 5: Apply final validation through guardrails system
        validation = None
        if use_guardrails:
            validation = self.guardrails.validate_response(response_text, contexts, scores)
            response_text = self.guardrails.format_response_with_warning(response_text, validation)
        
        # Extract unique source documents for attribution
        sources = list(set([ctx.get('source', 'Unknown') for ctx in contexts]))
        
        # Return comprehensive response with all metadata
        return {
            'response': response_text,           # Generated answer
            'sources': sources,                  # Source documents used
            'contexts': contexts,                # Retrieved document chunks
            'confidence': max(scores) if scores else 0.0,           # Best similarity score
            'average_confidence': sum(scores) / len(scores) if scores else 0.0,  # Mean confidence
            'validation': validation or {}       # Guardrails validation results
        }
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 32):
        """
        Add Document Chunks to Vector Store
        ==================================
        
        Processes document chunks by generating embeddings and storing them
        in the FAISS vector database for future retrieval.
        
        Args:
            chunks (List[Dict]): Document chunks with text and metadata
                Each chunk should have:
                - 'text': The actual text content
                - 'source': Source document name
                - Other metadata (chunk_index, start_char, etc.)
            batch_size (int): Number of texts to process simultaneously
                            Default: 32 (balanced memory usage and speed)
                            Larger values = faster but more memory
                            Smaller values = slower but less memory
        
        Process:
        1. Extract text content from chunks
        2. Generate embeddings using BGE model (in batches)
        3. Store embeddings and metadata in FAISS index
        
        Note: This is typically called during document ingestion phase
        
        Batch Processing:
        - Processes multiple chunks at once for efficiency
        - Default batch_size=32 works well on most CPUs
        - Increase to 64-128 if you have GPU
        - Decrease to 16 if you have limited memory
        """
        # Extract text content for embedding generation
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate semantic embeddings for all text chunks
        # This is computationally expensive but done once per document
        # batch_size parameter controls how many texts are processed at once
        # Larger batches = faster but more memory usage
        embeddings = self.embedding_generator.generate_embeddings(texts, batch_size=batch_size)
        
        # Add embeddings and metadata to FAISS vector store
        # Metadata includes source attribution and chunk information
        self.vector_store.add_vectors(embeddings, chunks)
        
        print(f"Added {len(chunks)} chunks to vector store (batch_size={batch_size})")

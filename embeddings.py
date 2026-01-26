"""
Embeddings Module - Semantic Text Representation
===============================================

This module handles the conversion of text into high-dimensional vector embeddings
for semantic similarity search in RAGnarok's retrieval system.

Key Features:
- State-of-the-art BGE (Beijing Academy of AI) embeddings
- Instruction-based query encoding for better retrieval
- Batch processing for efficient embedding generation
- Normalized vectors for cosine similarity computation

Technical Details:
- Model: BAAI/bge-base-en-v1.5 (768 dimensions)
- Normalization: L2 normalization for cosine similarity
- Device: CPU-optimized for broad compatibility
- Batch Size: Configurable for memory management

Author: RAGnarok Team
Version: 2.0.0
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """
    Semantic Text Embedding Generator
    ================================
    
    Converts text into dense vector representations using state-of-the-art
    transformer models optimized for retrieval tasks.
    
    Model Choice: BGE-base-en-v1.5
    - Optimized specifically for retrieval tasks
    - Outperforms general-purpose models like Sentence-BERT
    - Supports instruction-based encoding for queries
    - Balanced performance vs computational cost
    
    Vector Properties:
    - Dimension: 768 (standard transformer hidden size)
    - Normalization: L2 normalized for efficient cosine similarity
    - Range: [-1, 1] after normalization
    """
   
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize Embedding Generator
        =============================
        
        Loads the specified transformer model for text embedding generation.
        
        Args:
            model_name (str): Hugging Face model identifier
                            Default: BAAI/bge-base-en-v1.5 (optimized for retrieval)
        
        Model Loading:
        - Downloads model from Hugging Face Hub if not cached
        - Loads on CPU for broad compatibility
        - Initializes with retrieval-optimized configuration
        
        BGE Model Advantages:
        - Specifically trained for retrieval tasks
        - Supports instruction-based encoding
        - Better performance than general sentence transformers
        - Efficient inference on CPU
        """
        print(f"Loading embedding model: {model_name}")
        # Load model on CPU for compatibility (can be changed to GPU if available)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        print("Embedding model loaded successfully!")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate Embeddings for Document Texts
        ======================================
        
        Converts a list of text chunks into dense vector embeddings for storage
        in the vector database.
        
        Args:
            texts (List[str]): List of text chunks to embed
            batch_size (int): Number of texts to process simultaneously
                            Default: 32 (balanced memory usage and speed)
        
        Returns:
            np.ndarray: Array of embeddings with shape (num_texts, embedding_dim)
        
        Process:
        1. Batch processing for memory efficiency
        2. Normalization for cosine similarity computation
        3. Progress tracking for large document sets
        
        Performance Notes:
        - Batch processing reduces GPU/CPU overhead
        - Normalization enables efficient similarity search
        - Progress bar helps track processing of large document sets
        """
        if not texts:
            return np.array([])
        
        # Generate embeddings with batch processing and progress tracking
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,      # Visual feedback for large batches
            convert_to_numpy=True,       # Return numpy arrays for FAISS compatibility
            normalize_embeddings=True    # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate Embedding for User Query
        =================================
        
        Creates a vector embedding for the user's query, optimized for retrieval
        against document embeddings.
        
        Args:
            query (str): User's question or search query
            
        Returns:
            np.ndarray: Query embedding vector (normalized)
        
        BGE Instruction-Based Encoding:
        - BGE models support instruction prefixes for better retrieval
        - Query instruction helps the model understand the retrieval context
        - Improves matching between queries and relevant document passages
        
        Process:
        1. Add instruction prefix for BGE models
        2. Generate embedding with same normalization as documents
        3. Return vector ready for similarity search
        """
        # BGE models support instruction-based encoding for queries
        if "bge" in self.model_name.lower():
            # Format query with instruction for better retrieval performance
            instruction = "Represent this sentence for searching relevant passages:"
            query_with_instruction = f"{instruction} {query}"
            embedding = self.model.encode(
                query_with_instruction,
                convert_to_numpy=True,
                normalize_embeddings=True  # Match document normalization
            )
        else:
            # Standard encoding for non-BGE models
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embedding
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get Embedding Vector Dimension
        ==============================
        
        Returns the dimensionality of the embedding vectors produced by this model.
        Used for initializing the vector store with correct dimensions.
        
        Returns:
            int: Embedding dimension (768 for BGE-base models)
            
        Implementation:
        - Encodes a dummy text to determine output dimension
        - Cached property to avoid repeated computation
        - Essential for FAISS index initialization
        """
        # Get dimension by encoding a dummy text
        dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
        return dummy_embedding.shape[1]

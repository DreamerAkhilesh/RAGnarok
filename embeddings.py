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

# ============================================================================
# IMPORTS - Required libraries for embedding generation
# ============================================================================

# typing.List: Type hint for list objects
# Provides better code documentation and IDE support
from typing import List

# SentenceTransformer: Main class from sentence-transformers library
# Loads pre-trained transformer models for converting text to embeddings
# These models are specifically trained for semantic similarity tasks
from sentence_transformers import SentenceTransformer

# numpy: Numerical computing library for efficient array operations
# Embeddings are stored as numpy arrays for compatibility with FAISS
# Provides mathematical operations like normalization and dot products
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
   
    # ========================================================================
    # INITIALIZATION METHOD
    # ========================================================================
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
        # ====================================================================
        # STEP 1: Display loading message for user feedback
        # ====================================================================
        # Print message to inform user that model loading is starting
        # This can take 10-30 seconds on first run (downloads ~400MB)
        # Subsequent runs are faster (loads from cache)
        print(f"Loading embedding model: {model_name}")
        
        # ====================================================================
        # STEP 2: Load the SentenceTransformer model
        # ====================================================================
        # SentenceTransformer: Main class for loading pre-trained models
        # 
        # What happens internally:
        # 1. Checks cache directory (~/.cache/huggingface/)
        # 2. If not cached, downloads from Hugging Face Hub
        # 3. Loads tokenizer (converts text to tokens)
        # 4. Loads transformer model (12-layer BERT-based encoder)
        # 5. Loads pooling layer (converts token embeddings to sentence embedding)
        #
        # Parameters:
        # - model_name: Hugging Face model identifier (e.g., "BAAI/bge-base-en-v1.5")
        # - device="cpu": Run on CPU (change to "cuda" for GPU acceleration)
        #
        # Model Architecture (BGE-base):
        # - Input: Text string (any length)
        # - Tokenizer: Converts to max 512 tokens
        # - Transformer: 12 layers, 768 hidden size, 12 attention heads
        # - Pooling: Mean pooling over token embeddings
        # - Output: 768-dimensional vector
        self.model = SentenceTransformer(model_name, device="cpu")
        
        # ====================================================================
        # STEP 3: Store model name for later reference
        # ====================================================================
        # Save the model name as an instance variable
        # Used later in generate_query_embedding() to check if this is a BGE model
        # BGE models benefit from instruction-based encoding
        self.model_name = model_name
        
        # ====================================================================
        # STEP 4: Confirm successful loading
        # ====================================================================
        # Print success message to inform user model is ready
        print("Embedding model loaded successfully!")
    
    # ========================================================================
    # METHOD: Generate embeddings for multiple documents
    # ========================================================================
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
        
        # ====================================================================
        # STEP 1: Handle empty input edge case
        # ====================================================================
        # Check if the input list is empty
        # If empty, return an empty numpy array to prevent errors downstream
        # This is a defensive programming practice
        if not texts:
            return np.array([])
        
        # ====================================================================
        # STEP 2: Generate embeddings using the transformer model
        # ====================================================================
        # model.encode(): Main method for converting text to embeddings
        #
        # Internal Process (what happens inside encode()):
        # 1. TOKENIZATION:
        #    - Converts each text to token IDs using BERT tokenizer
        #    - Adds special tokens: [CLS] text tokens [SEP]
        #    - Pads/truncates to max length (512 tokens)
        #    - Creates attention masks (1 for real tokens, 0 for padding)
        #
        # 2. ENCODING:
        #    - Passes token IDs through 12 transformer layers
        #    - Each layer applies self-attention and feed-forward networks
        #    - Produces contextualized embeddings for each token
        #    - Shape: (batch_size, num_tokens, 768)
        #
        # 3. POOLING:
        #    - Aggregates token embeddings to single sentence embedding
        #    - Uses mean pooling (average of all token embeddings)
        #    - Ignores padding tokens using attention mask
        #    - Shape: (batch_size, 768)
        #
        # 4. NORMALIZATION:
        #    - Applies L2 normalization: v_norm = v / ||v||
        #    - Makes all vectors unit length (||v|| = 1)
        #    - Enables cosine similarity via dot product
        #    - Formula: cos(θ) = (A · B) / (||A|| × ||B||) = A · B (when normalized)
        #
        # Parameters explained:
        embeddings = self.model.encode(
            texts,                          # List of text strings to embed
                                           # Can be any length, will be truncated to 512 tokens
            
            batch_size=batch_size,          # Process this many texts at once
                                           # Larger batch = faster but more memory
                                           # 32 is a good balance for CPU
                                           # GPU can handle 64-128
            
            show_progress_bar=True,         # Display progress bar in terminal
                                           # Format: [████████] 100/100 [00:05<00:00, 20.00it/s]
                                           # Helpful for large document sets (100+ chunks)
            
            convert_to_numpy=True,          # Return numpy array instead of PyTorch tensor
                                           # Required for FAISS compatibility
                                           # FAISS expects numpy float32 arrays
            
            normalize_embeddings=True       # Apply L2 normalization to vectors
                                           # Makes all vectors unit length
                                           # Enables efficient cosine similarity
                                           # Without this, we'd need to normalize manually
        )
        
        # ====================================================================
        # STEP 3: Return the embeddings array
        # ====================================================================
        # Returns numpy array with shape: (num_texts, 768)
        # Example: 100 text chunks -> (100, 768) array
        # Each row is a 768-dimensional embedding vector
        # All vectors are L2 normalized (unit length)
        # Ready to be stored in FAISS vector database
        return embeddings
    
    # ========================================================================
    # METHOD: Generate embedding for a single query
    # ========================================================================
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
        
        # ====================================================================
        # STEP 1: Check if this is a BGE model
        # ====================================================================
        # BGE models were specifically trained with instruction prefixes
        # Check if "bge" appears anywhere in the model name (case-insensitive)
        # Examples: "BAAI/bge-base-en-v1.5", "bge-large-en", "BGE-small"
        if "bge" in self.model_name.lower():
            
            # ================================================================
            # STEP 1a: Create instruction prefix for BGE models
            # ================================================================
            # BGE models were trained with this specific instruction format
            # The instruction tells the model: "This is a query for searching"
            # This context helps the model generate a better query embedding
            #
            # Why this works:
            # - During training, BGE saw queries with this prefix
            # - It learned to encode queries differently than documents
            # - Query embeddings are optimized for matching relevant passages
            # - Improves retrieval performance by 2-5% on benchmarks
            #
            # Research paper: "C-Pack: Packaged Resources To Advance General Chinese Embedding"
            instruction = "Represent this sentence for searching relevant passages:"
            
            # ================================================================
            # STEP 1b: Combine instruction with user's query
            # ================================================================
            # Format: "Represent this sentence for searching relevant passages: What is...?"
            # The model processes this as a single input
            # The instruction provides context for how to encode the query
            query_with_instruction = f"{instruction} {query}"
            
            # ================================================================
            # STEP 1c: Generate embedding with instruction
            # ================================================================
            # encode(): Same method as generate_embeddings()
            # But here we process just one query string
            #
            # Process:
            # 1. Tokenize: "Represent this sentence..." + query -> tokens
            # 2. Encode: Pass through 12 transformer layers
            # 3. Pool: Mean pooling over token embeddings
            # 4. Normalize: L2 normalization to unit length
            #
            # Result: 768-dimensional vector optimized for retrieval
            embedding = self.model.encode(
                query_with_instruction,     # Query with instruction prefix
                convert_to_numpy=True,      # Return numpy array for FAISS
                normalize_embeddings=True   # L2 normalize to match documents
            )
            
        else:
            # ================================================================
            # STEP 2: Standard encoding for non-BGE models
            # ================================================================
            # For other models (e.g., all-MiniLM, mpnet), encode query directly
            # These models don't use instruction prefixes
            # They encode queries and documents the same way
            #
            # Process:
            # 1. Tokenize: query -> tokens
            # 2. Encode: Pass through transformer layers
            # 3. Pool: Mean pooling
            # 4. Normalize: L2 normalization
            embedding = self.model.encode(
                query,                      # Raw query without instruction
                convert_to_numpy=True,      # Return numpy array
                normalize_embeddings=True   # L2 normalize
            )
        
        # ====================================================================
        # STEP 3: Return the query embedding
        # ====================================================================
        # Returns numpy array with shape: (768,)
        # This is a single 768-dimensional vector
        # L2 normalized to unit length
        # Ready to be used in FAISS similarity search
        #
        # Usage: vector_store.search(query_embedding, k=5)
        # FAISS will compute dot product with all document embeddings
        # Higher dot product = higher cosine similarity = more relevant
        return embedding
    
    # ========================================================================
    # PROPERTY: Get embedding dimension
    # ========================================================================
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
        
        # ====================================================================
        # STEP 1: Encode a dummy text to determine dimension
        # ====================================================================
        # We don't know the dimension until we actually encode something
        # Different models have different dimensions:
        # - BGE-base: 768
        # - BGE-large: 1024
        # - all-MiniLM-L6-v2: 384
        # - all-mpnet-base-v2: 768
        #
        # So we encode a simple dummy text to find out
        # The content doesn't matter, we just need the shape
        dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
        
        # ====================================================================
        # STEP 2: Extract and return the dimension
        # ====================================================================
        # dummy_embedding.shape returns a tuple: (num_texts, dimension)
        # For our dummy input: (1, 768) for BGE-base
        # shape[0] = 1 (number of texts we encoded)
        # shape[1] = 768 (embedding dimension)
        #
        # We return shape[1] which is the embedding dimension
        # This is used when creating the FAISS index:
        # VectorStore(dimension=embedding_generator.embedding_dimension)
        return dummy_embedding.shape[1]

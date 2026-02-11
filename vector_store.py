"""
Vector Store Module - High-Performance Similarity Search
=======================================================

This module manages the FAISS vector database that powers RAGnarok's semantic
document retrieval system.

Key Features:
- FAISS (Facebook AI Similarity Search) integration
- Cosine similarity search for semantic matching
- Efficient storage and retrieval of embeddings
- Metadata management for source attribution
- Persistent storage with serialization

Technical Details:
- Index Type: IndexFlatIP (Inner Product for cosine similarity)
- Similarity Metric: Cosine similarity via normalized vectors
- Storage: Binary serialization with pickle for metadata
- Scalability: Supports thousands of documents efficiently

Author: RAGnarok Team
Version: 2.0.0
"""

# ============================================================================
# IMPORTS - Required libraries for vector storage and search
# ============================================================================

# os: Operating system interface for file/directory operations
# Used to create directories and check file existence
import os

# pickle: Python object serialization
# Used to save/load metadata and configuration as binary files
# Preserves Python data structures (lists, dicts) exactly
import pickle

# numpy: Numerical computing library
# Used for efficient array operations and vector manipulations
# FAISS requires numpy arrays as input
import numpy as np

# faiss: Facebook AI Similarity Search library
# High-performance library for similarity search and clustering
# Optimized C++ implementation with Python bindings
# Supports exact and approximate nearest neighbor search
import faiss

# typing: Type hints for better code documentation
# List: Type hint for list objects
# Dict: Type hint for dictionary objects
# Tuple: Type hint for tuple objects
from typing import List, Dict, Tuple


class VectorStore:
    """
    FAISS-Based Vector Database
    ==========================
    
    Manages high-dimensional vector storage and similarity search for
    RAGnarok's document retrieval system.
    
    Architecture:
    - FAISS for efficient similarity search
    - Separate metadata storage for document information
    - Normalized vectors for cosine similarity computation
    - Persistent storage for session continuity
    
    Similarity Search:
    - Uses cosine similarity (angle between vectors)
    - Normalized vectors enable efficient inner product computation
    - Results ranked by relevance score [0, 1]
    """

    # ========================================================================
    # INITIALIZATION METHOD
    # ========================================================================
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize Vector Store
        ======================
        
        Creates a new FAISS index for storing and searching document embeddings.
        
        Args:
            dimension (int): Dimensionality of embedding vectors (e.g., 768 for BGE)
            index_type (str): Type of FAISS index to use
                            Options: "flat", "cosine", "l2"
                            Default: "flat" (exact search)
        
        Index Types:
        - "flat"/"cosine": Exact cosine similarity search (IndexFlatIP)
        - "l2": Exact L2 distance search (IndexFlatL2)
        
        Performance Notes:
        - Flat indices provide exact results but scale O(n)
        - For large datasets, consider IVF indices for approximate search
        - Current implementation optimized for <10K documents
        """
        # ====================================================================
        # STEP 1: Store configuration parameters
        # ====================================================================
        # dimension: Size of embedding vectors (768 for BGE-base)
        # This must match the embedding model's output dimension
        # Used for validation when adding vectors
        self.dimension = dimension
        
        # index_type: Type of FAISS index to create
        # Determines the similarity metric and search algorithm
        # Stored for later reference and persistence
        self.index_type = index_type
        
        # ====================================================================
        # STEP 2: Initialize FAISS index based on type
        # ====================================================================
        # FAISS provides different index types for different use cases
        # We support three types: flat/cosine (same), and l2
        
        if index_type == "cosine" or index_type == "flat":
            # ================================================================
            # IndexFlatIP: Inner Product index for cosine similarity
            # ================================================================
            # "IP" = Inner Product
            # "Flat" = Exhaustive search (checks all vectors)
            #
            # How it works:
            # - Stores vectors as-is in memory
            # - Computes inner product: A · B = Σ(a_i * b_i)
            # - With normalized vectors: A · B = cos(θ)
            # - Higher score = more similar
            #
            # Why Inner Product for cosine similarity?
            # - cos(θ) = (A · B) / (||A|| × ||B||)
            # - If ||A|| = ||B|| = 1 (normalized), then cos(θ) = A · B
            # - So inner product of normalized vectors = cosine similarity
            #
            # Performance:
            # - O(n) search time (checks all vectors)
            # - Exact results (no approximation)
            # - Fast for <10K vectors
            # - Memory: ~4 bytes per dimension per vector
            #
            # dimension: Size of vectors to store
            self.index = faiss.IndexFlatIP(dimension)
            
        elif index_type == "l2":
            # ================================================================
            # IndexFlatL2: L2 distance index for Euclidean similarity
            # ================================================================
            # "L2" = L2 distance (Euclidean distance)
            # "Flat" = Exhaustive search
            #
            # How it works:
            # - Stores vectors as-is in memory
            # - Computes L2 distance: ||A - B|| = √(Σ(a_i - b_i)²)
            # - Lower distance = more similar
            #
            # When to use:
            # - When you want Euclidean distance instead of cosine similarity
            # - When vectors are not normalized
            # - When magnitude matters (not just direction)
            #
            # Performance: Same as IndexFlatIP
            self.index = faiss.IndexFlatL2(dimension)
            
        else:
            # ================================================================
            # Unsupported index type
            # ================================================================
            # Raise error if user provides invalid index type
            # This helps catch configuration errors early
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # ====================================================================
        # STEP 3: Initialize metadata storage
        # ====================================================================
        # metadata: List to store document information
        # FAISS only stores vectors (numbers), not metadata (text, sources)
        # We maintain a parallel list where index i corresponds to vector i
        #
        # Each metadata entry is a dictionary containing:
        # - text: The actual text content
        # - source: Source document filename
        # - chunk_index: Position in document
        # - start_char, end_char: Character positions
        # - Any other custom metadata
        #
        # Why separate storage?
        # - FAISS is optimized for numerical operations
        # - Metadata is variable-length and complex
        # - Keeping them separate is more efficient
        self.metadata = []
    
    # ========================================================================
    # METHOD: Add vectors and metadata to the store
    # ========================================================================
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):
        """
        Add Vectors and Metadata to Store
        =================================
        
        Adds embedding vectors and their associated metadata to the FAISS index.
        
        Args:
            vectors (np.ndarray): Array of embedding vectors, shape (n, dimension)
            metadata_list (List[Dict]): List of metadata dicts, one per vector
        
        Process:
        1. Validate inputs (dimensions, counts)
        2. Normalize vectors for cosine similarity
        3. Add vectors to FAISS index
        4. Store metadata in parallel list
        
        Raises:
            ValueError: If vector count doesn't match metadata count
            ValueError: If vector dimension doesn't match index dimension
        """
        # ====================================================================
        # STEP 1: Validate that vector count matches metadata count
        # ====================================================================
        # Each vector must have corresponding metadata
        # len(vectors): Number of vectors (rows in array)
        # len(metadata_list): Number of metadata dictionaries
        #
        # Why this check?
        # - Ensures we can look up metadata by vector index
        # - Prevents index out of bounds errors later
        # - Catches data preparation errors early
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # ====================================================================
        # STEP 2: Validate vector dimensions
        # ====================================================================
        # vectors.shape: Tuple of (num_vectors, dimension)
        # vectors.shape[1]: Dimension of each vector
        # self.dimension: Expected dimension from initialization
        #
        # Why this check?
        # - FAISS requires all vectors to have same dimension
        # - Mismatched dimensions cause cryptic FAISS errors
        # - Better to fail early with clear error message
        #
        # Example:
        # - Index expects 768-dimensional vectors
        # - User provides 384-dimensional vectors
        # - This check catches the mismatch
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # ====================================================================
        # STEP 3: Normalize vectors for cosine similarity
        # ====================================================================
        # Only normalize for cosine similarity indices
        # L2 indices don't need normalization
        if self.index_type == "cosine" or self.index_type == "flat":
            # ================================================================
            # faiss.normalize_L2(): In-place L2 normalization
            # ================================================================
            # L2 normalization makes all vectors unit length
            # Formula: v_normalized = v / ||v||
            # Where ||v|| = √(v₁² + v₂² + ... + vₙ²)
            #
            # Why normalize?
            # - Enables cosine similarity via inner product
            # - cos(θ) = (A · B) / (||A|| × ||B||)
            # - If ||A|| = ||B|| = 1, then cos(θ) = A · B
            # - So we can use fast inner product instead of cosine formula
            #
            # In-place operation:
            # - Modifies the vectors array directly
            # - More memory efficient than creating new array
            # - FAISS function is optimized C++ code
            faiss.normalize_L2(vectors)
        
        # ====================================================================
        # STEP 4: Add vectors to FAISS index
        # ====================================================================
        # index.add(): FAISS method to add vectors to the index
        # 
        # What happens internally:
        # 1. Vectors are copied into FAISS's internal storage
        # 2. For IndexFlatIP/L2: Vectors stored in contiguous memory
        # 3. Each vector gets an index: 0, 1, 2, ...
        # 4. Index is ready for search immediately (no training needed)
        #
        # astype('float32'): Convert to 32-bit floats
        # - FAISS requires float32 (not float64/double)
        # - Reduces memory usage by half
        # - Faster computation on modern CPUs
        # - Sufficient precision for similarity search
        #
        # After this call:
        # - self.index.ntotal increases by len(vectors)
        # - Vectors are searchable immediately
        self.index.add(vectors.astype('float32'))
        
        # ====================================================================
        # STEP 5: Store metadata in parallel list
        # ====================================================================
        # extend(): Adds all items from metadata_list to self.metadata
        # This maintains the parallel structure:
        # - Vector at index i in FAISS
        # - Metadata at index i in self.metadata
        #
        # Example:
        # Before: self.metadata = [meta0, meta1]
        # Add: metadata_list = [meta2, meta3]
        # After: self.metadata = [meta0, meta1, meta2, meta3]
        #
        # Why extend instead of append?
        # - extend adds individual items
        # - append would add the list itself
        # - We want a flat list, not nested lists
        self.metadata.extend(metadata_list)
    
    # ========================================================================
    # METHOD: Search for similar vectors
    # ========================================================================
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for Similar Vectors
        ==========================
        
        Finds the k most similar vectors to the query vector using FAISS.
        
        Args:
            query_vector (np.ndarray): Query embedding vector, shape (dimension,)
            k (int): Number of nearest neighbors to return
        
        Returns:
            List[Tuple[Dict, float]]: List of (metadata, similarity_score) tuples
                                     Sorted by similarity (highest first)
        
        Process:
        1. Handle empty index edge case
        2. Reshape and normalize query vector
        3. Perform FAISS search
        4. Convert distances to similarity scores
        5. Attach metadata to results
        
        Similarity Scores:
        - Range: [0, 1]
        - 1.0: Perfect match (identical vectors)
        - 0.5: Moderate similarity
        - 0.0: No similarity (orthogonal vectors)
        """
        # ====================================================================
        # STEP 1: Handle empty index edge case
        # ====================================================================
        # index.ntotal: Number of vectors in the index
        # If 0, the index is empty (no documents loaded)
        #
        # Why check this?
        # - FAISS search on empty index returns garbage
        # - Better to return empty list explicitly
        # - Prevents confusing error messages
        if self.index.ntotal == 0:
            return []
        
        # ====================================================================
        # STEP 2: Reshape query vector to 2D array
        # ====================================================================
        # FAISS expects 2D arrays: (num_queries, dimension)
        # Our query_vector is 1D: (dimension,)
        # Need to reshape to: (1, dimension)
        #
        # reshape(1, -1): Reshape to 1 row, infer number of columns
        # - 1: One query vector
        # - -1: Automatically calculate dimension (768)
        #
        # astype('float32'): Convert to 32-bit floats
        # - FAISS requires float32
        # - Matches the type of stored vectors
        #
        # Example:
        # Before: query_vector.shape = (768,)
        # After: query_vector.shape = (1, 768)
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # ====================================================================
        # STEP 3: Normalize query vector for cosine similarity
        # ====================================================================
        # Only normalize for cosine similarity indices
        # Must match the normalization applied to stored vectors
        if self.index_type == "cosine" or self.index_type == "flat":
            # Normalize query vector to unit length
            # This ensures inner product = cosine similarity
            # Must be done after reshape (FAISS expects 2D)
            faiss.normalize_L2(query_vector)
        
        # ====================================================================
        # STEP 4: Adjust k if necessary
        # ====================================================================
        # k: Number of results to return
        # index.ntotal: Total number of vectors in index
        #
        # Can't return more results than vectors in index
        # min(k, ntotal): Take the smaller of the two
        #
        # Example:
        # - User requests k=5
        # - Index only has 3 vectors
        # - Return 3 results (all available)
        k = min(k, self.index.ntotal)
        
        # ====================================================================
        # STEP 5: Perform FAISS search
        # ====================================================================
        # index.search(): Main FAISS search method
        # 
        # Parameters:
        # - query_vector: The query (1, dimension)
        # - k: Number of nearest neighbors
        #
        # Returns:
        # - distances: Array of distances/scores, shape (1, k)
        # - indices: Array of vector indices, shape (1, k)
        #
        # For IndexFlatIP (cosine similarity):
        # - distances are inner products (higher = more similar)
        # - Range: [-1, 1] for normalized vectors
        # - 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
        #
        # For IndexFlatL2 (Euclidean distance):
        # - distances are L2 distances (lower = more similar)
        # - Range: [0, ∞)
        # - 0.0 = identical, larger = more different
        #
        # Results are sorted by distance (best first)
        distances, indices = self.index.search(query_vector, k)
        
        # ====================================================================
        # STEP 6: Format results with metadata and similarity scores
        # ====================================================================
        # Initialize empty list to store results
        results = []
        
        # Iterate through returned indices and distances
        # zip(): Pairs up corresponding elements
        # indices[0]: First (and only) query's results
        # distances[0]: First (and only) query's distances
        #
        # Example:
        # indices[0] = [42, 17, 99]
        # distances[0] = [0.85, 0.72, 0.68]
        # Pairs: (42, 0.85), (17, 0.72), (99, 0.68)
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # ================================================================
            # STEP 6a: Validate index is within metadata bounds
            # ================================================================
            # idx: Index of the matched vector in FAISS
            # self.metadata: Parallel list of metadata
            #
            # Check that idx is valid for metadata list
            # Prevents index out of bounds errors
            # Should always be true if add_vectors was used correctly
            if idx < len(self.metadata):
                # ============================================================
                # STEP 6b: Convert distance to similarity score
                # ============================================================
                # Different conversion for different index types
                
                if self.index_type == "cosine" or self.index_type == "flat":
                    # ========================================================
                    # Cosine similarity: Convert inner product to [0, 1]
                    # ========================================================
                    # Inner product range: [-1, 1]
                    # - 1.0: Identical vectors (0° angle)
                    # - 0.0: Orthogonal vectors (90° angle)
                    # - -1.0: Opposite vectors (180° angle)
                    #
                    # Conversion formula: (distance + 1) / 2
                    # - Maps [-1, 1] to [0, 1]
                    # - -1.0 → 0.0 (no similarity)
                    # - 0.0 → 0.5 (moderate similarity)
                    # - 1.0 → 1.0 (perfect similarity)
                    #
                    # Why normalize to [0, 1]?
                    # - Easier to interpret
                    # - Consistent with probability/confidence
                    # - Can be used as threshold (e.g., > 0.5)
                    similarity = float((distance + 1) / 2)
                    
                else:
                    # ========================================================
                    # L2 distance: Convert to similarity score
                    # ========================================================
                    # L2 distance range: [0, ∞)
                    # - 0.0: Identical vectors
                    # - Larger: More different
                    #
                    # Conversion formula: 1 / (1 + distance)
                    # - Maps [0, ∞) to (0, 1]
                    # - 0.0 → 1.0 (perfect similarity)
                    # - 1.0 → 0.5 (moderate similarity)
                    # - ∞ → 0.0 (no similarity)
                    #
                    # This is a simple inverse relationship
                    # Other formulas possible (e.g., exponential decay)
                    similarity = float(1 / (1 + distance))
                
                # ============================================================
                # STEP 6c: Add result tuple to list
                # ============================================================
                # Create tuple of (metadata, similarity_score)
                # - metadata: Dictionary with text, source, etc.
                # - similarity: Float in range [0, 1]
                #
                # self.metadata[idx]: Get metadata for this vector
                # Results are already sorted by FAISS (best first)
                results.append((self.metadata[idx], similarity))
        
        # ====================================================================
        # STEP 7: Return formatted results
        # ====================================================================
        # Returns list of (metadata, score) tuples
        # Sorted by similarity (highest first)
        # Ready to be used by RAG pipeline
        return results
    
    # ========================================================================
    # METHOD: Save vector store to disk
    # ========================================================================
    def save(self, directory: str):
        """
        Save Vector Store to Disk
        =========================
        
        Persists the FAISS index, metadata, and configuration to disk for later use.
        
        Args:
            directory (str): Directory path where files will be saved
        
        Saves three files:
        - faiss.index: Binary FAISS index with all vectors
        - metadata.pkl: Pickled metadata list
        - config.pkl: Pickled configuration dict
        
        Process:
        1. Create directory if it doesn't exist
        2. Save FAISS index
        3. Save metadata
        4. Save configuration
        """
        # ====================================================================
        # STEP 1: Create directory if it doesn't exist
        # ====================================================================
        # os.makedirs(): Creates directory and any necessary parent directories
        # exist_ok=True: Don't raise error if directory already exists
        #
        # Why this is important:
        # - Prevents FileNotFoundError when saving
        # - Creates nested directories if needed
        # - Safe to call multiple times
        #
        # Example:
        # directory = "data/vector_store"
        # Creates: data/ and data/vector_store/
        os.makedirs(directory, exist_ok=True)
        
        # ====================================================================
        # STEP 2: Save FAISS index to binary file
        # ====================================================================
        # Construct full path for index file
        # os.path.join(): Safely joins path components
        # Works on Windows (backslash) and Unix (forward slash)
        index_path = os.path.join(directory, "faiss.index")
        
        # faiss.write_index(): FAISS function to save index to disk
        # 
        # What gets saved:
        # - Index type (IndexFlatIP, IndexFlatL2, etc.)
        # - All vectors in the index
        # - Index configuration
        #
        # File format:
        # - Binary format (not human-readable)
        # - Optimized for fast loading
        # - Typically 4 bytes per dimension per vector
        #
        # Example size:
        # - 1000 vectors × 768 dimensions × 4 bytes = ~3MB
        #
        # Note: Metadata is NOT saved here (saved separately)
        faiss.write_index(self.index, index_path)
        
        # ====================================================================
        # STEP 3: Save metadata to pickle file
        # ====================================================================
        # Construct full path for metadata file
        metadata_path = os.path.join(directory, "metadata.pkl")
        
        # Open file in binary write mode
        # 'wb' = write binary
        # with statement ensures file is properly closed
        with open(metadata_path, 'wb') as f:
            # pickle.dump(): Serialize Python object to binary file
            # 
            # What gets saved:
            # - Complete metadata list
            # - All dictionaries with their structure
            # - Text content, sources, chunk info, etc.
            #
            # Why pickle?
            # - Preserves Python data structures exactly
            # - Handles nested dicts, lists, etc.
            # - Fast serialization/deserialization
            # - Binary format (smaller than JSON)
            #
            # File format:
            # - Binary pickle format
            # - Not human-readable
            # - Can be loaded with pickle.load()
            pickle.dump(self.metadata, f)
        
        # ====================================================================
        # STEP 4: Save configuration to pickle file
        # ====================================================================
        # Construct full path for config file
        config_path = os.path.join(directory, "config.pkl")
        
        # Open file in binary write mode
        with open(config_path, 'wb') as f:
            # Create configuration dictionary
            # Contains all information needed to recreate the VectorStore
            config_dict = {
                'dimension': self.dimension,        # Vector dimension (768)
                'index_type': self.index_type,      # Index type ("flat", "l2")
                'num_vectors': self.index.ntotal    # Number of vectors stored
            }
            
            # Serialize configuration to file
            # This allows us to validate and recreate the store on load
            pickle.dump(config_dict, f)
    
    # ========================================================================
    # CLASS METHOD: Load vector store from disk
    # ========================================================================
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load Vector Store from Disk
        ===========================
        
        Loads a previously saved vector store from disk.
        
        Args:
            directory (str): Directory path where files are saved
        
        Returns:
            VectorStore: Fully initialized VectorStore instance
        
        Process:
        1. Load configuration
        2. Create new VectorStore instance
        3. Load FAISS index
        4. Load metadata
        5. Return initialized store
        
        Note: This is a class method, called as VectorStore.load(path)
        """
        # ====================================================================
        # STEP 1: Load configuration from pickle file
        # ====================================================================
        # Construct full path for config file
        config_path = os.path.join(directory, "config.pkl")
        
        # Open file in binary read mode
        # 'rb' = read binary
        with open(config_path, 'rb') as f:
            # pickle.load(): Deserialize Python object from binary file
            # Returns the dictionary we saved in save()
            #
            # config will contain:
            # - dimension: Vector dimension
            # - index_type: Index type
            # - num_vectors: Number of vectors (for validation)
            config = pickle.load(f)
        
        # ====================================================================
        # STEP 2: Create new VectorStore instance with loaded config
        # ====================================================================
        # cls: Reference to the class (VectorStore)
        # This is a class method, so we use cls instead of VectorStore
        #
        # Create new instance with:
        # - dimension from config
        # - index_type from config
        #
        # This creates an empty VectorStore with correct configuration
        # We'll load the actual data in next steps
        store = cls(config['dimension'], config['index_type'])
        
        # ====================================================================
        # STEP 3: Load FAISS index from binary file
        # ====================================================================
        # Construct full path for index file
        index_path = os.path.join(directory, "faiss.index")
        
        # faiss.read_index(): FAISS function to load index from disk
        # 
        # What gets loaded:
        # - Index type
        # - All vectors
        # - Index configuration
        #
        # Returns: Fully initialized FAISS index
        # Ready to search immediately
        #
        # Replace the empty index with loaded one
        store.index = faiss.read_index(index_path)
        
        # ====================================================================
        # STEP 4: Load metadata from pickle file
        # ====================================================================
        # Construct full path for metadata file
        metadata_path = os.path.join(directory, "metadata.pkl")
        
        # Open file in binary read mode
        with open(metadata_path, 'rb') as f:
            # pickle.load(): Deserialize metadata list
            # Returns the complete list of metadata dictionaries
            #
            # Replace the empty metadata list with loaded one
            # Now metadata[i] corresponds to vector i in FAISS index
            store.metadata = pickle.load(f)
        
        # ====================================================================
        # STEP 5: Return fully initialized VectorStore
        # ====================================================================
        # The store now has:
        # - Correct configuration (dimension, index_type)
        # - All vectors loaded in FAISS index
        # - All metadata loaded in parallel list
        # - Ready to search immediately
        return store
    
    # ========================================================================
    # METHOD: Get statistics about the vector store
    # ========================================================================
    def get_stats(self) -> Dict:
        """
        Get Vector Store Statistics
        ===========================
        
        Returns statistics about the current state of the vector store.
        
        Returns:
            Dict: Statistics dictionary containing:
                - num_vectors: Number of vectors in the index
                - dimension: Dimensionality of vectors
                - index_type: Type of FAISS index
        
        Useful for:
        - Debugging
        - Monitoring
        - User feedback
        - Validation
        """
        # ====================================================================
        # Return statistics dictionary
        # ====================================================================
        # Create and return dictionary with current statistics
        return {
            # Number of vectors currently in the index
            # self.index.ntotal: FAISS property for vector count
            # Increases as vectors are added
            # Example: 37 (for 37 document chunks)
            'num_vectors': self.index.ntotal,
            
            # Dimensionality of vectors
            # Set during initialization
            # Must match embedding model output
            # Example: 768 (for BGE-base)
            'dimension': self.dimension,
            
            # Type of FAISS index being used
            # One of: "flat", "cosine", "l2"
            # Determines similarity metric
            # Example: "flat" (cosine similarity)
            'index_type': self.index_type
        }

"""
Embeddings Module
Handles text embedding generation using Sentence Transformers.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
   
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
       
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        print("Embedding model loaded successfully!")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
       
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
      
        # BGE models support instruction-based encoding for queries
        if "bge" in self.model_name.lower():
            # Format query with instruction for better retrieval
            instruction = "Represent this sentence for searching relevant passages:"
            query_with_instruction = f"{instruction} {query}"
            embedding = self.model.encode(
                query_with_instruction,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embedding
    
    @property
    def embedding_dimension(self) -> int:
        
        # Get dimension by encoding a dummy text
        dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
        return dummy_embedding.shape[1]

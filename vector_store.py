"""
Vector Store Module
Manages FAISS vector database for efficient similarity search.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple


class VectorStore:

    
    def __init__(self, dimension: int, index_type: str = "flat"):

        self.dimension = dimension
        self.index_type = index_type
        
        
        if index_type == "cosine" or index_type == "flat":
           
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        
        self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):

        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        
        if self.index_type == "cosine" or self.index_type == "flat":
            faiss.normalize_L2(vectors)
        
       
        self.index.add(vectors.astype('float32'))
        
       
        self.metadata.extend(metadata_list)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:

        if self.index.ntotal == 0:
            return []
        
   
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
       
        if self.index_type == "cosine" or self.index_type == "flat":
            faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)  
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                # For cosine similarity, distance is inner product (higher is better)
                # Convert to similarity score between 0 and 1
                if self.index_type == "cosine" or self.index_type == "flat":
                    similarity = float((distance + 1) / 2)  # Normalize to [0, 1]
                else:
                  
                    similarity = float(1 / (1 + distance))
                
                results.append((self.metadata[idx], similarity))
        
        return results
    
    def save(self, directory: str):

        os.makedirs(directory, exist_ok=True)
        
       
        index_path = os.path.join(directory, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
       
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'index_type': self.index_type,
                'num_vectors': self.index.ntotal
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':

       
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
      
        store = cls(config['dimension'], config['index_type'])
        
       
        index_path = os.path.join(directory, "faiss.index")
        store.index = faiss.read_index(index_path)
        
       
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            store.metadata = pickle.load(f)
        
        return store
    
    def get_stats(self) -> Dict:

        return {
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type
        }

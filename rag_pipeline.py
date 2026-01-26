"""
RAG Pipeline Module
Implements the complete Retrieval-Augmented Generation pipeline.
"""

from typing import List, Dict, Optional
import ollama
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from guardrails import Guardrails


class RAGPipeline:

    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "llama3",
                 vector_store: Optional[VectorStore] = None,
                 min_confidence: float = 0.5):

        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.llm_model = llm_model
        self.guardrails = Guardrails(min_confidence=min_confidence)
        
        if vector_store is None:
            dimension = self.embedding_generator.embedding_dimension
            self.vector_store = VectorStore(dimension=dimension, index_type="cosine")
        else:
            self.vector_store = vector_store
    
    def build_prompt(self, query: str, contexts: List[Dict], max_context_length: int = 2000) -> str:

        # Format contexts
        context_texts = []
        total_length = 0
        
        for ctx in contexts:
            text = ctx['text']
            source = ctx.get('source', 'Unknown')
            
            if total_length + len(text) > max_context_length:
                # Truncate last context if needed
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if meaningful
                    text = text[:remaining] + "..."
                    context_texts.append(f"[Source: {source}]\n{text}")
                break
            
            context_texts.append(f"[Source: {source}]\n{text}")
            total_length += len(text)
        
        contexts_str = "\n\n---\n\n".join(context_texts)
        
        # Build system prompt with strict instructions
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

        prompt = system_instruction.format(
            contexts=contexts_str,
            query=query
        )
        
        return prompt
    
    def retrieve_contexts(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        return results
    
    def generate_response(self, query: str, top_k: int = 5, 
                         use_guardrails: bool = True) -> Dict:

       
        retrieval_results = self.retrieve_contexts(query, top_k=top_k)
        
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
        
        # Extract contexts and scores
        contexts = [meta for meta, score in retrieval_results]
        scores = [score for meta, score in retrieval_results]
        
       
        if use_guardrails:
            filtered_results = self.guardrails.filter_low_confidence(retrieval_results)
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
            contexts = [meta for meta, score in filtered_results]
            scores = [score for meta, score in filtered_results]
        
    
        prompt = self.build_prompt(query, contexts)
        
      
        try:
            # Try chat API first (preferred), fallback to generate
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    options={
                        'temperature': 0.1,  # Low temperature for factual responses
                        'top_p': 0.9,
                    }
                )
                response_text = response['message']['content']
            except (AttributeError, KeyError):
                # Fallback to generate API
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={
                        'temperature': 0.1,
                        'top_p': 0.9,
                    }
                )
                response_text = response.get('response', str(response))
        except Exception as e:
            response_text = f"Error generating response: {str(e)}. Please ensure Ollama is running and the model '{self.llm_model}' is installed. Run: ollama pull {self.llm_model}"
        
  
        validation = None
        if use_guardrails:
            validation = self.guardrails.validate_response(response_text, contexts, scores)
            response_text = self.guardrails.format_response_with_warning(response_text, validation)
        
        # Extract unique sources
        sources = list(set([ctx.get('source', 'Unknown') for ctx in contexts]))
        
        return {
            'response': response_text,
            'sources': sources,
            'contexts': contexts,
            'confidence': max(scores) if scores else 0.0,
            'average_confidence': sum(scores) / len(scores) if scores else 0.0,
            'validation': validation or {}
        }
    
    def add_documents(self, chunks: List[Dict]):
      
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, chunks)
        
        print(f"Added {len(chunks)} chunks to vector store")

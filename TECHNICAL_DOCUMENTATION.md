# RAGnarok - Technical Documentation

## System Architecture Overview

**RAGnarok** is a Retrieval-Augmented Generation (RAG) system that provides intelligent document-based question answering. The system prevents hallucinations by strictly answering questions based only on provided documents.

### Core Components Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  RAG Pipeline    │────│ Document Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼────┐ ┌──▼──────┐
            │Embeddings │ │Vector  │ │Ollama   │
            │Generator  │ │Store   │ │LLM      │
            └───────────┘ └────────┘ └─────────┘
                              │
                        ┌─────▼─────┐
                        │Guardrails │
                        │System     │
                        └───────────┘
```

## Technical Stack Deep Dive

### 1. Document Processing Pipeline (`document_processor.py`)

**Purpose**: Ingests and chunks documents for optimal retrieval

**Key Technical Details**:
- **Supported Formats**: PDF (PyPDF2), TXT, Markdown
- **Chunking Strategy**: Intelligent text segmentation with sentence boundary detection
- **Chunk Size**: 512 characters (configurable)
- **Overlap**: 50 characters to maintain context continuity

**Critical Implementation**:
```python
def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
    # Uses regex for sentence boundary detection
    sentence_endings = re.compile(r'[.!?]\s+')
    
    # Prioritizes sentence boundaries over hard character limits
    # Falls back to word boundaries if sentence break not found
```

**Interview Points**:
- Why sentence boundaries? Maintains semantic coherence
- Overlap prevents context loss at chunk boundaries
- Metadata tracking enables source attribution

### 2. Embedding System (`embeddings.py`)

**Model**: `BAAI/bge-base-en-v1.5` (BGE - Beijing Academy of AI)

**Technical Specifications**:
- **Dimension**: 768 (automatically detected)
- **Normalization**: L2 normalized for cosine similarity
- **Instruction-based**: Uses query instructions for better retrieval

**Key Implementation**:
```python
def generate_query_embedding(self, query: str) -> np.ndarray:
    if "bge" in self.model_name.lower():
        instruction = "Represent this sentence for searching relevant passages:"
        query_with_instruction = f"{instruction} {query}"
```

**Interview Points**:
- BGE models are state-of-the-art for retrieval tasks
- Instruction-based encoding improves query-document matching
- Normalization enables efficient cosine similarity computation

### 3. Vector Store (`vector_store.py`)

**Technology**: FAISS (Facebook AI Similarity Search)

**Technical Details**:
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Search Algorithm**: Exhaustive search for accuracy
- **Similarity Metric**: Cosine similarity (normalized inner product)

**Critical Code**:
```python
def search(self, query_vector: np.ndarray, k: int = 5):
    # Normalize for cosine similarity
    faiss.normalize_L2(query_vector)
    
    # Convert inner product to similarity score [0,1]
    similarity = float((distance + 1) / 2)
```

**Interview Points**:
- FAISS chosen for production-grade performance
- Inner product with normalized vectors = cosine similarity
- Metadata stored separately, linked by index

### 4. RAG Pipeline (`rag_pipeline.py`)

**Core Workflow**:
1. **Query Embedding**: Convert user query to vector
2. **Retrieval**: Find top-k similar document chunks
3. **Context Assembly**: Build prompt with retrieved contexts
4. **LLM Generation**: Generate response using Ollama
5. **Guardrails**: Validate and filter response

**Prompt Engineering**:
```python
system_instruction = """You are a helpful knowledge assistant that answers questions STRICTLY based on the provided documents. 

CRITICAL RULES:
1. Answer ONLY using information from the provided context documents
2. If the answer is not in the provided documents, explicitly state "Based on the provided documents, I cannot find information about [topic]"
3. Do NOT make up information or use external knowledge
4. Cite the source document when referencing specific information
5. If multiple sources contain relevant information, synthesize them clearly
6. Be concise but complete in your answers
"""
```

**Interview Points**:
- Explicit instructions prevent hallucination
- Source attribution ensures traceability
- Low temperature (0.1) for factual consistency

### 5. Guardrails System (`guardrails.py`)

**Safety Mechanisms**:

1. **Confidence Thresholding**:
   - Default minimum: 0.5 similarity score
   - Filters low-relevance contexts
   - Prevents answering from irrelevant documents

2. **Hallucination Detection**:
   - Monitors for refusal keywords ("I don't know", "not mentioned")
   - Validates context-based responses
   - Flags potential hallucinations

3. **Response Validation**:
```python
def validate_response(self, response: str, contexts: List[str], 
                     similarity_scores: List[float]) -> Dict:
    confidence_passed, avg_confidence = self.check_confidence(similarity_scores)
    response_valid, is_context_based = self.enforce_context_only(response, contexts)
    
    is_valid = confidence_passed and is_context_based
```

**Interview Points**:
- Multi-layer validation prevents unreliable responses
- Confidence scoring ensures relevance
- Explicit refusal better than hallucination

### 6. LLM Integration (Ollama)

**Model**: Gemma 2B (default, configurable)

**Configuration**:
- **Temperature**: 0.1 (low for factual consistency)
- **Top-p**: 0.9 (nucleus sampling)
- **API**: Uses chat API with fallback to generate API

**Interview Points**:
- Local deployment ensures data privacy
- Low temperature reduces creativity/hallucination
- Fallback handling ensures robustness

## Data Flow Architecture

### Query Processing Flow:
```
User Query → Query Embedding → Vector Search → Context Retrieval → 
Prompt Construction → LLM Generation → Guardrails Validation → Response
```

### Document Ingestion Flow:
```
Documents → Text Extraction → Chunking → Embedding Generation → 
Vector Storage → Metadata Indexing
```

## Performance Characteristics

### Scalability:
- **Vector Search**: O(n) for exhaustive search, O(log n) with IVF indices
- **Memory Usage**: ~4 bytes per dimension per vector (768 * 4 = 3KB per chunk)
- **Throughput**: Limited by embedding generation and LLM inference

### Accuracy Factors:
- **Chunk Size**: 512 chars balances context and precision
- **Top-k**: 5 contexts provide sufficient information without noise
- **Confidence Threshold**: 0.5 filters irrelevant matches

## Configuration Management

### Key Parameters:
```python
EMBEDDING_MODEL_DEFAULT = "BAAI/bge-base-en-v1.5"
LLM_MODEL_DEFAULT = "gemma:2b"
MIN_CONFIDENCE_DEFAULT = 0.5
TOP_K_DEFAULT = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### Environment Dependencies:
- **Python**: >=3.11
- **FAISS**: CPU version for compatibility
- **Ollama**: Local LLM serving
- **Sentence Transformers**: Embedding generation

## Error Handling & Robustness

### Document Processing:
- Graceful handling of corrupted files
- Format validation before processing
- Encoding detection for text files

### Vector Operations:
- Dimension validation before adding vectors
- Index persistence with metadata consistency
- Recovery from corrupted indices

### LLM Integration:
- API fallback mechanisms
- Connection timeout handling
- Model availability validation

## Security Considerations

### Data Privacy:
- All processing happens locally
- No external API calls for sensitive documents
- Vector embeddings don't expose raw text

### Input Validation:
- File type restrictions
- Query length limits
- Malicious content filtering

## Deployment Architecture

### Docker-Based LLM Deployment

RAGnarok uses Docker for LLM deployment, providing several advantages:

**Benefits of Docker Deployment:**
- **Isolation**: LLM runs in isolated container environment
- **Portability**: Consistent deployment across different systems
- **Resource Management**: Better control over GPU/CPU allocation
- **Version Control**: Easy model version management
- **Scalability**: Simple horizontal scaling with multiple containers

**Docker Configuration:**
```bash
# Run Ollama container with persistent storage
docker run -d \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama

# Pull and manage models
docker exec -it ollama ollama pull gemma:2b
docker exec -it ollama ollama list
```

**Integration with RAGnarok:**
```python
# RAG Pipeline configuration for Docker Ollama
pipeline = RAGPipeline(
    embedding_model="BAAI/bge-base-en-v1.5",
    llm_model="gemma:2b",
    ollama_host="http://localhost:11434",  # Docker container endpoint
    min_confidence=0.5
)
```

### Local Development:
```bash
# Setup with Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull gemma:2b

# Python environment
python -m venv venv
pip install -r requirements.txt

# Run
python main.py setup
streamlit run app.py
```

### Production Considerations:
- **GPU Acceleration**: FAISS-GPU for large datasets
- **Model Optimization**: Quantized models for faster inference
- **Caching**: Embedding cache for repeated queries
- **Load Balancing**: Multiple Ollama instances

## Monitoring & Observability

### Key Metrics:
- **Query Response Time**: End-to-end latency
- **Retrieval Accuracy**: Relevance of retrieved contexts
- **Confidence Scores**: Distribution of similarity scores
- **Guardrails Triggers**: Frequency of safety interventions

### Logging:
- Document processing statistics
- Query patterns and performance
- Error rates and types
- Resource utilization

## Interview Preparation Points

### System Design Questions:
1. **Why RAG over fine-tuning?** 
   - Dynamic knowledge updates without retraining
   - Source attribution and explainability
   - Reduced hallucination risk

2. **Embedding Model Choice?**
   - BGE models optimized for retrieval tasks
   - Instruction-based encoding for better query-document matching
   - Balanced performance vs. resource usage

3. **Vector Database Selection?**
   - FAISS for production-grade performance
   - Local deployment for privacy
   - Flexible indexing options

4. **Chunking Strategy?**
   - Sentence boundaries preserve semantic coherence
   - Overlap prevents context fragmentation
   - Size balances detail and relevance

### Technical Deep Dives:
1. **Cosine Similarity Math**: Normalized dot product, range [-1,1], converted to [0,1]
2. **FAISS Indexing**: IndexFlatIP for exact search, IVF for approximate
3. **Prompt Engineering**: System instructions, context formatting, temperature tuning
4. **Guardrails Logic**: Multi-layer validation, confidence thresholding, hallucination detection

### Optimization Opportunities:
1. **Hybrid Search**: Combine dense and sparse retrieval
2. **Reranking**: Two-stage retrieval with cross-encoder
3. **Query Expansion**: Enhance queries with synonyms/context
4. **Caching**: Store embeddings and frequent query results

This technical documentation provides comprehensive coverage of the system's architecture, implementation details, and key concepts needed for technical interviews.
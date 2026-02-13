# RAGnarok - Design Decisions, Trade-offs & Improvement Opportunities

## 📋 Table of Contents

1. [Architecture Decisions](#architecture-decisions)
2. [Technology Stack Choices](#technology-stack-choices)
3. [Algorithm & Implementation Trade-offs](#algorithm--implementation-tradeoffs)
4. [Performance Considerations](#performance-considerations)
5. [Improvement Opportunities](#improvement-opportunities)
6. [Scalability Considerations](#scalability-considerations)
7. [Security & Privacy](#security--privacy)

---

## 🏗️ Architecture Decisions

### 1. RAG Pipeline Architecture

#### **Decision: Retrieval-Augmented Generation (RAG) over Fine-tuning**

**Why RAG?**
- ✅ No need to retrain models when documents change
- ✅ Transparent source attribution for every answer
- ✅ Works with any LLM (model-agnostic)
- ✅ Lower computational cost than fine-tuning
- ✅ Immediate updates when documents are added/removed

**Why NOT Fine-tuning?**
- ❌ Requires expensive GPU training (hours/days)
- ❌ Model becomes outdated when documents change
- ❌ No source attribution (black box)
- ❌ Risk of hallucinations on out-of-distribution queries
- ❌ Requires ML expertise to implement

**Trade-offs:**
- RAG has higher latency per query (embedding + search + LLM)
- RAG requires maintaining a vector database
- Fine-tuning might give better performance on domain-specific tasks

**Improvement Opportunities:**
- Hybrid approach: Fine-tune a small adapter on top of base model + RAG
- Cache frequent queries to reduce latency
- Use approximate nearest neighbor search for larger datasets

---

### 2. Two-Phase Architecture (Setup + Query)

#### **Decision: Separate document ingestion from query processing**

**Why Separate Phases?**
- ✅ Documents processed once, queried many times (efficiency)
- ✅ Faster query response times (no document processing overhead)
- ✅ Can update documents independently
- ✅ Clear separation of concerns

**Why NOT Real-time Processing?**
- ❌ Would slow down every query significantly
- ❌ Inefficient for static document collections
- ❌ Harder to optimize and cache

**Trade-offs:**
- Users must explicitly reload when documents change
- Requires persistent storage (vector_store directory)
- Initial setup time before first query

**Improvement Opportunities:**
- Add file watching to auto-reload on document changes
- Implement incremental updates (add/remove single documents)
- Add document versioning and change tracking

---

## 🛠️ Technology Stack Choices

### 1. Embedding Model: BAAI/bge-base-en-v1.5

#### **Why BGE-base-en-v1.5?**
- ✅ State-of-the-art retrieval performance (MTEB benchmark leader)
- ✅ Optimized specifically for retrieval tasks
- ✅ 768 dimensions (good balance of quality and speed)
- ✅ Supports instruction-based encoding for queries
- ✅ Works well on CPU (no GPU required)

**Alternatives Considered:**

| Model | Dimensions | Pros | Cons | Why Not Chosen |
|-------|-----------|------|------|----------------|
| **all-MiniLM-L6-v2** | 384 | Faster, smaller | Lower quality | Retrieval quality matters more |
| **all-mpnet-base-v2** | 768 | Good general purpose | Not optimized for retrieval | BGE outperforms on retrieval |
| **BGE-large-en-v1.5** | 1024 | Higher quality | Slower, more memory | Diminishing returns for cost |
| **OpenAI text-embedding-3** | 1536 | Very high quality | Requires API, costs money | Want offline solution |

**Trade-offs:**
- BGE-base is slower than MiniLM but more accurate
- 768 dimensions use more memory than 384 but give better results
- CPU-only means slower than GPU but more accessible

**Improvement Opportunities:**
- Add GPU support for faster embedding generation
- Allow users to choose embedding model based on their needs
- Implement model quantization for faster inference
- Cache embeddings for frequently queried documents

---

### 2. Vector Database: FAISS (IndexFlatIP)

#### **Why FAISS?**
- ✅ Industry-standard for similarity search (Facebook AI)
- ✅ Extremely fast even on CPU
- ✅ No external dependencies (embedded library)
- ✅ Supports exact and approximate search
- ✅ Easy to persist and load

**Why IndexFlatIP (Inner Product)?**
- ✅ Exact search (no approximation errors)
- ✅ Perfect for cosine similarity with normalized vectors
- ✅ Simple and reliable
- ✅ Fast for <10K vectors

**Alternatives Considered:**

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Pinecone** | Managed, scalable | Costs money, requires internet | Want offline solution |
| **Weaviate** | Feature-rich, hybrid search | Complex setup, overkill | Too heavy for this use case |
| **ChromaDB** | Simple API, persistent | Slower than FAISS | FAISS is faster |
| **Elasticsearch** | Full-text + vector search | Heavy, complex | Overkill for pure vector search |
| **FAISS IVF indices** | Faster for large datasets | Approximate results | Current dataset is small |

**Trade-offs:**
- IndexFlatIP is O(n) search time (checks all vectors)
- Exact search is slower than approximate but more accurate
- No built-in filtering or metadata search

**Improvement Opportunities:**
- Switch to IVF index when dataset grows >10K vectors
- Add hybrid search (keyword + semantic)
- Implement metadata filtering before vector search
- Use HNSW index for even faster approximate search
- Add support for multiple vector stores (user choice)

---

### 3. LLM: Ollama + Gemma 2B
Ollama → a tool/platform to run LLMs locally.
Gemma → an LLM model (the brain) that can be run using tools like Ollama.

#### **Why Ollama?**
- ✅ Runs locally (no API costs, no data leakage)
- ✅ Docker deployment (easy setup)
- ✅ Supports many open-source models
- ✅ Simple API (compatible with OpenAI format)
- ✅ No rate limits

**Why Gemma 2B?**
- ✅ Small enough to run on consumer hardware (4GB RAM)
- ✅ Good reasoning capabilities for its size
- ✅ Fast inference (2-3 seconds per response)
- ✅ Trained by Google (high quality)
- ✅ Permissive license

**Alternatives Considered:**

| Model | Size | Pros | Cons | Why Not Chosen |
|-------|------|------|------|----------------|
| **GPT-4** | Unknown | Best quality | Expensive, API-only | Want offline solution |
| **Llama 3 8B** | 8B | Better quality | Needs 8GB+ RAM | Too heavy for target users |
| **Mistral 7B** | 7B | Good quality | Needs 7GB+ RAM | Too heavy for target users |
| **Phi-3 Mini** | 3.8B | Fast, good quality | Slightly larger | Gemma 2B is sufficient |
| **TinyLlama** | 1.1B | Very fast | Lower quality | Quality matters for RAG |

**Trade-offs:**
- Gemma 2B is less capable than larger models
- Local inference is slower than cloud APIs
- Limited context window (2048 tokens)

**Improvement Opportunities:**
- Support multiple LLM backends (OpenAI, Anthropic, etc.)
- Allow users to choose model based on hardware
- Implement streaming responses for better UX
- Add response caching for repeated queries
- Use quantized models (GGUF) for faster inference

---

### 4. Web Framework: Streamlit

#### **Why Streamlit?**
- ✅ Rapid development (100 lines vs 1000+ for Flask/React)
- ✅ Automatic UI updates (reactive programming)
- ✅ Built-in session state management
- ✅ Beautiful default styling
- ✅ Perfect for data/ML applications

**Why NOT Other Frameworks?**

| Framework | Pros | Cons | Why Not Chosen |
|-----------|------|------|----------------|
| **Flask + React** | Full control, production-ready | 10x more code, complex | Overkill for this project |
| **Gradio** | Even simpler than Streamlit | Less flexible, limited UI | Streamlit more powerful |
| **FastAPI + HTML** | Fast, RESTful | More boilerplate | Not needed for this use case |
| **Django** | Full-featured | Way too heavy | Massive overkill |

**Trade-offs:**
- Streamlit reruns entire script on interaction (can be slow)
- Limited customization compared to React
- Not ideal for complex multi-page apps
- Session state can be tricky to manage

**Improvement Opportunities:**
- Add caching with @st.cache_data for expensive operations
- Implement proper error boundaries
- Add loading states for better UX
- Consider FastAPI backend + React frontend for production
- Add user authentication and multi-user support

---

## ⚙️ Algorithm & Implementation Trade-offs

### 1. Text Chunking Strategy

#### **Decision: Semantic boundary detection (sentence > word > hard break)**

**Why This Approach?**
- ✅ Preserves complete thoughts (sentences stay together)
- ✅ Prevents breaking words in half
- ✅ Maintains context with overlap
- ✅ Better retrieval quality

**Algorithm:**
```
1. Try to break at sentence boundary (. ! ?)
2. If no sentence, break at word boundary (space)
3. If no word boundary, hard break at chunk_size
4. Add overlap to prevent information loss
```

**Alternatives Considered:**

| Strategy | Pros | Cons | Why Not Chosen |
|----------|------|------|----------------|
| **Fixed-size chunks** | Simple, fast | Breaks sentences/words | Poor retrieval quality |
| **Paragraph-based** | Natural boundaries | Variable sizes | Some paragraphs too large |
| **Recursive splitting** | Hierarchical | Complex | Overkill for this use case |
| **Semantic chunking (LLM)** | Best quality | Very slow, expensive | Too slow for setup |

**Current Parameters:**
- `chunk_size = 512` characters
- `chunk_overlap = 50` characters

**Why 512 characters?**
- ✅ ~100 words (good context amount)
- ✅ Fits well in embedding model context
- ✅ Not too large (maintains specificity)
- ✅ Not too small (maintains context)

**Why 50 character overlap?**
- ✅ ~10 words (enough to maintain context)
- ✅ Prevents information loss at boundaries
- ✅ Not too much (would waste storage)

**Trade-offs:**
- Larger chunks = more context but less precise retrieval
- Smaller chunks = more precise but less context
- More overlap = better recall but more storage

**Improvement Opportunities:**
- Implement adaptive chunking based on document structure
- Use LLM-based semantic chunking for critical documents
- Add support for different chunk sizes per document type
- Implement hierarchical chunking (parent-child relationships)
- Add chunk quality scoring and filtering

---

### 2. Similarity Search & Ranking

#### **Decision: Cosine similarity with confidence thresholding**

**Why Cosine Similarity?**
- ✅ Measures semantic similarity (angle between vectors)
- ✅ Normalized (0-1 range, easy to interpret)
- ✅ Works well with normalized embeddings
- ✅ Fast to compute (dot product with normalized vectors)

**Why Confidence Threshold (0.5)?**
- ✅ Filters clearly irrelevant results
- ✅ Balanced (not too strict, not too permissive)
- ✅ Prevents low-quality contexts from reaching LLM

**Alternatives Considered:**

| Metric | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Euclidean (L2) distance** | Intuitive | Sensitive to magnitude | Cosine better for text |
| **Dot product** | Fast | Not normalized | Need normalized scores |
| **BM25 (keyword)** | Good for exact matches | Misses semantic similarity | Want semantic search |
| **Hybrid (BM25 + cosine)** | Best of both worlds | More complex | Could be future improvement |

**Trade-offs:**
- Pure semantic search misses exact keyword matches
- Fixed threshold might be too strict/loose for some queries
- Top-k retrieval might miss relevant results beyond k

**Improvement Opportunities:**
- Implement hybrid search (keyword + semantic)
- Add query-adaptive thresholding
- Implement re-ranking with cross-encoder
- Add diversity in retrieved results (MMR algorithm)
- Support multiple retrieval strategies (dense, sparse, hybrid)

---

### 3. Prompt Engineering

#### **Decision: Strict grounding instructions with explicit refusal patterns**

**Current Prompt Structure:**
```
System Instruction:
- Answer ONLY using provided documents
- If not in documents, say "I cannot find information"
- Do NOT use external knowledge
- Cite sources

Context Documents:
[Source: file.txt]
Document text...

User Question: {query}

Answer (based ONLY on documents):
```

**Why This Approach?**
- ✅ Minimizes hallucinations
- ✅ Forces model to acknowledge limitations
- ✅ Provides clear source attribution
- ✅ Maintains transparency

**Alternatives Considered:**

| Approach | Pros | Cons | Why Not Chosen |
|----------|------|------|----------------|
| **Simple prompt** | Shorter, faster | More hallucinations | Quality matters more |
| **Few-shot examples** | Better quality | Longer prompt, slower | Current approach works |
| **Chain-of-thought** | Better reasoning | Much slower | Not needed for RAG |
| **System message only** | Cleaner | Less effective | Need explicit instructions |

**Trade-offs:**
- Longer prompts use more tokens (slower, more expensive)
- Strict instructions might make model too conservative
- Explicit refusal patterns might be too rigid

**Improvement Opportunities:**
- Add few-shot examples for better quality
- Implement query classification (factual vs opinion)
- Add confidence scoring in LLM response
- Use chain-of-thought for complex queries
- Implement prompt optimization with DSPy

---

### 4. Guardrails & Validation

#### **Decision: Multi-layer validation (confidence + keywords + context-based)**

**Validation Layers:**
1. **Confidence thresholding** - Filter low-relevance contexts
2. **Keyword detection** - Monitor for refusal phrases
3. **Context validation** - Ensure response is grounded
4. **Warning generation** - Alert users to low confidence

**Why Multi-layer?**
- ✅ Defense in depth (multiple safety checks)
- ✅ Catches different types of issues
- ✅ Provides transparency to users
- ✅ Allows graceful degradation

**Refusal Keywords:**
```python
[
    "i don't know",
    "i cannot",
    "not in the provided",
    "not mentioned",
    "cannot determine",
    ...
]
```

**Why These Keywords?**
- ✅ Indicate honest uncertainty (good behavior)
- ✅ Show model is following instructions
- ✅ Prevent hallucinations

**Alternatives Considered:**

| Approach | Pros | Cons | Why Not Chosen |
|----------|------|------|----------------|
| **No validation** | Faster | Unsafe | Quality matters |
| **LLM-based validation** | More accurate | Very slow | Too expensive |
| **Fact-checking API** | Comprehensive | Requires external service | Want offline |
| **Embedding similarity** | Fast | Limited scope | Good addition to current |

**Trade-offs:**
- More validation = slower responses
- Keyword matching is brittle (can miss variations)
- Too strict validation might reject good answers

**Improvement Opportunities:**
- Add LLM-based self-consistency checking
- Implement fact verification against source documents
- Add confidence calibration
- Use NLI (Natural Language Inference) for entailment checking
- Implement answer quality scoring

---

## 🚀 Performance Considerations

### 1. Embedding Generation

**Current Performance:**
- ~100 chunks/second on CPU
- ~1000 chunks/second on GPU

**Bottlenecks:**
- Transformer model inference (most expensive)
- Batch size limited by memory
- CPU-only deployment

**Optimization Strategies Used:**
- ✅ Batch processing (32 chunks at a time by default)
- ✅ Progress bar for user feedback
- ✅ One-time processing (cached in vector store)

**Note:** Both `app.py` and `main.py` use the same `RAGPipeline.add_documents()` method, which internally calls `generate_embeddings(texts, batch_size=32)`. The batch processing happens automatically in both interfaces with the same default batch size.

**Improvement Opportunities:**
- Add GPU support (10x faster)
- Implement model quantization (2-4x faster)
- Use ONNX runtime (1.5-2x faster)
- Increase batch size on high-memory systems
- Implement parallel processing for multiple documents

**Batch Size Configuration:**
- **Current:** Both `app.py` and `main.py` use `batch_size=32` (default)
- **Why 32?** Good balance for CPU processing (not too much memory, not too slow)
- **For GPU:** Can increase to 64-128 for faster processing
- **For Low Memory:** Can decrease to 16 if running out of memory
- **How to Change:** Now configurable via `add_documents(chunks, batch_size=64)`

---

### 2. Vector Search

**Current Performance:**
- ~1ms for 1000 vectors (IndexFlatIP)
- ~10ms for 10,000 vectors
- ~100ms for 100,000 vectors

**Bottlenecks:**
- O(n) search time (checks all vectors)
- Memory bandwidth (loading vectors)

**Optimization Strategies Used:**
- ✅ Normalized vectors (faster dot product)
- ✅ Float32 (smaller than float64)
- ✅ Exact search (no approximation errors)

**When to Switch Indices:**

| Dataset Size | Recommended Index | Search Time | Accuracy |
|--------------|-------------------|-------------|----------|
| <10K vectors | IndexFlatIP | <10ms | 100% |
| 10K-100K | IVF1024,Flat | <5ms | 99%+ |
| 100K-1M | IVF4096,PQ64 | <2ms | 95%+ |
| >1M | HNSW | <1ms | 98%+ |

**Improvement Opportunities:**
- Switch to IVF index for larger datasets
- Implement index auto-selection based on size
- Add GPU support for search (10x faster)
- Implement query result caching
- Use memory-mapped files for large indices

---

### 3. LLM Inference

**Current Performance:**
- ~2-3 seconds per response (Gemma 2B)
- ~5-10 seconds for longer responses

**Bottlenecks:**
- Model size (2B parameters)
- CPU-only inference
- No batching (one query at a time)

**Optimization Strategies Used:**
- ✅ Small model (2B vs 7B+)
- ✅ Docker deployment (easy setup)
- ✅ Low temperature (0.1) for faster generation

**Improvement Opportunities:**
- Use quantized models (GGUF format, 2x faster)
- Add GPU support (5-10x faster)
- Implement response streaming (better UX)
- Add response caching for common queries
- Use speculative decoding (1.5-2x faster)
- Implement batch inference for multiple queries

---

### 4. Memory Usage

**Current Memory Footprint:**
- Embedding model: ~500MB
- Vector store: ~4 bytes × dimension × num_vectors
  - Example: 1000 vectors × 768 dim × 4 bytes = ~3MB
- LLM (Gemma 2B): ~4GB
- **Total: ~4.5GB**

**Memory Optimization Strategies:**
- ✅ Float32 instead of float64 (50% reduction)
- ✅ Lazy loading (load only when needed)
- ✅ Persistent storage (don't keep everything in RAM)

**Improvement Opportunities:**
- Implement model quantization (4-bit, 8-bit)
- Use memory-mapped files for large vector stores
- Add memory usage monitoring and warnings
- Implement automatic garbage collection
- Support model offloading (CPU ↔ GPU)

---

## 📈 Improvement Opportunities

### High Priority (Quick Wins)

1. **Add GPU Support**
   - Impact: 10x faster embedding + LLM
   - Effort: Low (just change device parameter)
   - Code: `device="cuda" if torch.cuda.is_available() else "cpu"`

2. **Implement Response Caching**
   - Impact: Instant responses for repeated queries
   - Effort: Low (use dict or Redis)
   - Code: `@st.cache_data` decorator

3. **Add Streaming Responses**
   - Impact: Better UX (see response as it generates)
   - Effort: Medium (use Ollama streaming API)
   - Code: `client.chat(stream=True)`

4. **Improve Error Messages**
   - Impact: Better user experience
   - Effort: Low (add try-except with helpful messages)
   - Code: More specific exception handling

5. **Add Document Metadata Display**
   - Impact: Better transparency
   - Effort: Low (show file size, date, type)
   - Code: Add to UI display

### Medium Priority (Significant Improvements)

6. **Hybrid Search (Keyword + Semantic)**
   - Impact: Better retrieval for exact matches
   - Effort: Medium (integrate BM25)
   - Library: `rank-bm25`

7. **Re-ranking with Cross-Encoder**
   - Impact: Better result quality
   - Effort: Medium (add second-stage ranking)
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

8. **Incremental Document Updates**
   - Impact: No need to reload everything
   - Effort: Medium (track document changes)
   - Code: Add/remove individual documents

9. **Query History & Analytics**
   - Impact: Better insights
   - Effort: Medium (store queries in DB)
   - Storage: SQLite or JSON

10. **Multi-language Support**
    - Impact: Broader audience
    - Effort: Medium (use multilingual models)
    - Model: `intfloat/multilingual-e5-base`

### Low Priority (Nice to Have)

11. **Advanced Chunking Strategies**
    - Impact: Better retrieval quality
    - Effort: High (implement semantic chunking)
    - Approach: Use LLM to identify logical sections

12. **Document Preprocessing Pipeline**
    - Impact: Better text quality
    - Effort: Medium (add cleaning, normalization)
    - Tools: Remove headers/footers, fix encoding

13. **A/B Testing Framework**
    - Impact: Data-driven improvements
    - Effort: High (build testing infrastructure)
    - Metrics: Retrieval quality, response quality

14. **Multi-modal Support**
    - Impact: Handle images, tables
    - Effort: High (add vision models)
    - Models: CLIP, LayoutLM

15. **Distributed Deployment**
    - Impact: Handle high load
    - Effort: High (add load balancing)
    - Tools: Kubernetes, Redis

---

## 📊 Scalability Considerations

### Current Limitations

| Aspect | Current Limit | Bottleneck | Solution |
|--------|---------------|------------|----------|
| **Documents** | ~1000 files | Processing time | Parallel processing |
| **Vectors** | ~10K chunks | Search speed | Switch to IVF index |
| **Concurrent Users** | ~10 users | Single process | Add load balancing |
| **Query Throughput** | ~1 query/sec | LLM inference | Batch inference |
| **Storage** | ~1GB | Disk space | Compression |

### Scaling Strategies

#### Vertical Scaling (Single Machine)
1. **Add GPU** - 10x faster inference
2. **More RAM** - Larger batch sizes
3. **SSD Storage** - Faster I/O
4. **More CPU Cores** - Parallel processing

#### Horizontal Scaling (Multiple Machines)
1. **Separate Services**
   - Embedding service (CPU/GPU)
   - Vector search service (CPU)
   - LLM service (GPU)
   - Web frontend (CPU)

2. **Load Balancing**
   - Nginx for web traffic
   - Redis for caching
   - Message queue for async processing

3. **Database Sharding**
   - Split vector store by document type
   - Distribute across multiple FAISS indices
   - Use distributed vector DB (Milvus, Qdrant)

### Recommended Architecture for Production

```
┌─────────────┐
│   Nginx     │ (Load Balancer)
└──────┬──────┘
       │
   ┌───┴────┐
   │        │
┌──▼──┐  ┌──▼──┐
│ Web │  │ Web │ (Streamlit/FastAPI)
└──┬──┘  └──┬──┘
   │        │
   └───┬────┘
       │
┌──────▼──────┐
│   Redis     │ (Cache)
└──────┬──────┘
       │
   ┌───┴────┐
   │        │
┌──▼──┐  ┌──▼──┐
│FAISS│  │ LLM │ (Separate services)
└─────┘  └─────┘
```

---

## 🔒 Security & Privacy

### Current Security Posture

**Strengths:**
- ✅ Runs completely offline (no data leakage)
- ✅ No external API calls
- ✅ Local file storage only
- ✅ No user authentication needed (single user)

**Weaknesses:**
- ❌ No input validation (could inject malicious prompts)
- ❌ No rate limiting (could be abused)
- ❌ No access control (anyone with access can query)
- ❌ No audit logging (can't track usage)
- ❌ No encryption (documents stored in plain text)

### Security Improvements

#### High Priority
1. **Input Validation**
   - Sanitize user queries
   - Limit query length
   - Block malicious patterns

2. **Rate Limiting**
   - Limit queries per user/IP
   - Prevent DoS attacks
   - Use `slowapi` library

3. **Error Handling**
   - Don't expose internal errors
   - Log errors securely
   - Return generic error messages

#### Medium Priority
4. **Access Control**
   - Add user authentication
   - Role-based permissions
   - Document-level access control

5. **Audit Logging**
   - Log all queries
   - Track document access
   - Monitor for suspicious activity

6. **Encryption**
   - Encrypt documents at rest
   - Encrypt vector store
   - Use HTTPS for web interface

### Privacy Considerations

**Current Approach:**
- All data stays local
- No telemetry or analytics
- No cloud services

**Recommendations:**
1. Add privacy policy
2. Document data retention
3. Implement data deletion
4. Add GDPR compliance features
5. Support data export

---

## 🎯 Conclusion

### What Works Well
- ✅ Simple, clean architecture
- ✅ Good balance of quality and performance
- ✅ Easy to understand and modify
- ✅ Runs on consumer hardware
- ✅ No external dependencies

### What Could Be Better
- ⚠️ Limited scalability (single machine)
- ⚠️ No GPU support (slower than it could be)
- ⚠️ Basic chunking strategy
- ⚠️ No hybrid search
- ⚠️ Limited error handling

### Recommended Next Steps

**Phase 1: Quick Wins (1-2 weeks)**
1. Add GPU support
2. Implement caching
3. Add streaming responses
4. Improve error messages

**Phase 2: Quality Improvements (1 month)**
5. Hybrid search
6. Re-ranking
7. Better chunking
8. Query analytics

**Phase 3: Production Ready (2-3 months)**
9. Security hardening
10. Scalability improvements
11. Monitoring and logging
12. Documentation and testing

---

## 📚 References & Further Reading

### Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [BGE Embeddings](https://arxiv.org/abs/2309.07597)
- [FAISS: Efficient Similarity Search](https://arxiv.org/abs/1702.08734)

### Libraries
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

### Best Practices
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [OpenAI RAG Best Practices](https://platform.openai.com/docs/guides/retrieval-augmented-generation)

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** RAGnarok Team

# Multiple Choice Questions: RAG and Vector Databases

Test your understanding of retrieval-augmented generation and vector database concepts for AI/ML interviews.

---

**Q1. Retrieval-Augmented Generation (RAG) reduces LLM hallucination by:**

A) Making the model smaller
B) Retrieving relevant documents and including them in the prompt, grounding generation in actual sources
C) Removing the attention mechanism
D) Training on only verified facts

---

**Q2. The standard RAG pipeline consists of:**

A) Only text generation
B) Query encoding → document retrieval → (optional re-ranking) → prompt construction with retrieved context → LLM generation
C) Only document storage
D) Fine-tuning the model on retrieved documents

---

**Q3. Cosine similarity between two embedding vectors measures:**

A) The Euclidean distance between them
B) The cosine of the angle between them, capturing directional similarity regardless of magnitude
C) The product of their norms
D) Whether they have the same number of dimensions

---

**Q4. For normalized embeddings (unit vectors), cosine similarity equals:**

A) L2 distance
B) Dot product
C) L1 distance
D) Cross-entropy

---

**Q5. Document chunking is important in RAG because:**

A) It makes documents longer
B) It splits documents into retrievable units of appropriate size, balancing retrieval precision and context completeness
C) It removes all formatting
D) It is only needed for images

---

**Q6. Small chunks (e.g., 128 tokens) in RAG provide:**

A) Less precise retrieval
B) Higher precision but risk losing context; larger chunks provide more context but may include irrelevant information
C) Always better performance than large chunks
D) No difference compared to large chunks

---

**Q7. HNSW (Hierarchical Navigable Small Worlds) indexing provides:**

A) O(n) brute-force search
B) Approximate nearest neighbor search in O(log n) time using a hierarchical graph structure
C) Exact nearest neighbor search in O(1)
D) No indexing capability

---

**Q8. Product Quantization (PQ) in vector databases achieves memory efficiency by:**

A) Storing full-precision vectors
B) Compressing vectors by splitting them into subspaces and quantizing each to fewer bits
C) Deleting half the vectors
D) Using only the first dimension

---

**Q9. Hybrid search combines dense (embedding) and sparse (BM25) retrieval because:**

A) Dense retrieval always outperforms sparse
B) Dense captures semantic similarity while sparse captures exact keyword matching, and combining them improves overall recall
C) Sparse retrieval is no longer useful
D) They use the same algorithm

---

**Q10. Re-ranking in a RAG pipeline improves results by:**

A) Adding more documents to the index
B) Using a more expensive cross-encoder model to re-score and reorder the initially retrieved candidates
C) Removing the retrieval step entirely
D) Randomly shuffling documents

---

**Q11. The embedding model used in RAG should be chosen based on:**

A) Only its parameter count
B) Domain relevance, speed, dimensionality, and quality on retrieval benchmarks (e.g., BEIR)
C) Whether it can generate text
D) Its training data size alone

---

**Q12. Reciprocal Rank Fusion (RRF) is used in hybrid search to:**

A) Train a new model
B) Merge ranked results from multiple retrieval systems using a simple formula based on rank positions
C) Replace the embedding model
D) Remove duplicate documents only

---

**Q13. A key limitation of RAG is:**

A) It always produces perfect answers
B) If the retrieved documents don't contain the answer, the LLM may still hallucinate or produce low-quality responses
C) It cannot use any external documents
D) It requires retraining the LLM for each query

---

**Q14. Vector databases like Pinecone and Weaviate primarily store and retrieve:**

A) Relational tables with SQL
B) High-dimensional embedding vectors with efficient approximate nearest neighbor search
C) Only text documents
D) Only image files

---

**Q15. The trade-off between IVF and HNSW indexing is:**

A) IVF is always better
B) IVF has lower memory overhead and faster build time; HNSW has faster query time and higher recall but uses more memory
C) They are identical in all aspects
D) HNSW cannot handle high-dimensional data

---

## Answer Key

**Q1. Answer: B**
RAG grounds generation in retrieved documents, so the LLM generates answers based on actual sources rather than relying solely on potentially inaccurate memorized knowledge.

**Q2. Answer: B**
The full RAG pipeline encodes the query as an embedding, retrieves similar documents, optionally re-ranks them, constructs a prompt with the most relevant context, then generates an answer.

**Q3. Answer: B**
Cosine similarity = (a·b)/(||a||·||b||) measures the angle between vectors, ranging from -1 (opposite) to 1 (identical direction), ignoring vector magnitude.

**Q4. Answer: B**
When ||a|| = ||b|| = 1, cosine similarity simplifies to a·b (dot product), since the normalization denominator equals 1. This is why many vector databases normalize embeddings.

**Q5. Answer: B**
Chunking determines the granularity of retrieval. Too large means irrelevant text dilutes relevant content; too small means important context is split across chunks.

**Q6. Answer: B**
Small chunks precisely target specific facts but may lack surrounding context. Large chunks provide richer context but may include irrelevant content. The optimal size is task-dependent (typically 256-512 tokens).

**Q7. Answer: B**
HNSW builds a hierarchical graph of vectors. Search navigates from the top (sparse) layer down to the bottom (dense) layer, achieving sublinear O(log n) search time with high recall.

**Q8. Answer: B**
PQ splits each vector into subspaces and represents each subspace with a short code (e.g., 8 bits). This can reduce memory by 32-64x while maintaining reasonable search accuracy.

**Q9. Answer: B**
Dense retrieval finds semantically similar documents (e.g., "car" matches "automobile"), while sparse retrieval finds exact keyword matches. Combining them covers both cases, improving overall retrieval quality.

**Q10. Answer: B**
Initial retrieval (bi-encoder) is fast but approximate. Re-ranking with a cross-encoder (which jointly processes query and document) is slower but more accurate, improving the final ranking quality.

**Q11. Answer: B**
The embedding model should match the domain, balance speed and quality, and perform well on relevant retrieval benchmarks. Domain-specific fine-tuning often improves results significantly.

**Q12. Answer: B**
RRF computes score = Σ 1/(k + rank_i) for each document across retrieval systems. It's simple, requires no tuning, and effectively combines rankings from different sources.

**Q13. Answer: B**
RAG is only as good as its retrieval. If relevant documents aren't in the index or aren't retrieved in the top-k, the LLM lacks grounding and may still produce incorrect or fabricated answers.

**Q14. Answer: B**
Vector databases are specialized for storing embeddings and performing fast similarity search at scale using algorithms like HNSW, IVF, and PQ, typically supporting millions to billions of vectors.

**Q15. Answer: B**
IVF uses clustering for fast build and low memory but has lower recall for the same query speed. HNSW uses a graph structure for faster, higher-recall queries but requires more memory for the graph edges.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

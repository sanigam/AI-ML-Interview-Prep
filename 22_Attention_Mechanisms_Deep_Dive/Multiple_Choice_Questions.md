# Multiple Choice Questions: Attention Mechanisms Deep Dive

📺 **Video Lecture:** https://youtu.be/te7D9Al7mpw


Test your understanding of attention mechanism concepts for AI/ML interviews.

---

**Q1. Additive (Bahdanau) attention computes scores using:**

A) Cosine similarity between query and key  
B) A learned nonlinear function: score = vᵀ·tanh(W_q·q + W_k·k)  
C) Simple dot product of query and key  
D) Euclidean distance between query and key

---

**Q2. Scaled dot-product attention uses the formula:**

A) Attention = ReLU(QKᵀ) · V  
B) Attention = softmax(QKᵀ / √d) · V  
C) Attention = softmax(Q + K) · V  
D) Attention = sigmoid(QKᵀ) · V

---

**Q3. Self-attention differs from cross-attention in that:**

A) Cross-attention is always faster than self-attention  
B) Self-attention has no learnable parameters  
C) Self-attention can only be applied to images  
D) In self-attention, Q, K, and V all come from the same sequence

---

**Q4. The purpose of using multiple attention heads instead of one is to:**

A) Speed up inference by a factor of h  
B) Eliminate the need for feed-forward layers  
C) Allow different heads to capture different types of relationships simultaneously  
D) Reduce the total number of parameters

---

**Q5. Without the 1/√d scaling factor in dot-product attention, what problem occurs?**

A) Keys and values become identical  
B) Attention weights become negative  
C) The attention mechanism becomes non-differentiable  
D) Large dot products push softmax into saturated regions, causing vanishing gradients

---

**Q6. Causal masking in a decoder sets future attention scores to:**

A) Negative infinity (−∞), so softmax outputs zero for those positions  
B) The average of all scores  
C) One  
D) Zero

---

**Q7. RoPE (Rotary Position Embedding) encodes positions by:**

A) Applying rotation matrices to query and key vectors based on position  
B) Learning a separate embedding for each position index  
C) Concatenating position indices to token embeddings  
D) Adding sinusoidal vectors to token embeddings

---

**Q8. Multi-query attention (MQA) reduces memory during inference by:**

A) Reducing the embedding dimension  
B) Sharing a single set of key-value heads across all query heads  
C) Eliminating the value projection entirely  
D) Using fewer query heads

---

**Q9. The KV-cache in autoregressive transformer inference stores:**

A) Only the output logits  
B) The gradients from the last backward pass  
C) The full attention weight matrix  
D) Previously computed key and value tensors to avoid recomputation at each generation step

---

**Q10. Flash Attention improves transformer efficiency by:**

A) Replacing attention with convolution  
B) Reordering computation to minimize GPU memory reads/writes (IO-aware algorithm)  
C) Reducing the sequence length  
D) Using fewer attention heads

---

**Q11. Sparse attention (as in Longformer) reduces O(n²) complexity by:**

A) Using only the first and last tokens  
B) Restricting each token to attend to only a local window plus selected global tokens  
C) Removing attention entirely for some layers  
D) Halving the embedding dimension

---

**Q12. Grouped-query attention (GQA) is a compromise between:**

A) Self-attention and cross-attention  
B) Pre-LN and post-LN configurations  
C) Additive and multiplicative attention  
D) Multi-head attention (separate KV per head) and multi-query attention (shared KV for all heads)

---

**Q13. Attention weights after softmax can be interpreted as:**

A) Probabilities summing to 1 over all key positions for each query  
B) Raw similarity scores without normalization  
C) Gradient magnitudes for each position  
D) Binary indicators of relevance

---

**Q14. ALiBi (Attention with Linear Biases) handles positions by:**

A) Learning position embeddings from scratch  
B) Adding a linear penalty −α|i−j| to attention scores based on distance between positions  
C) Ignoring position information entirely  
D) Using convolutional position encoding

---

**Q15. Cross-attention in a machine translation transformer allows the decoder to:**

A) Share weights with the encoder  
B) Only look at the immediately previous source token  
C) Generate tokens without considering the source sentence  
D) Attend to relevant parts of the encoded source sentence at each decoding step

---

## Answer Key

**Q1. Answer: B**
Bahdanau (additive) attention uses a learned nonlinear function with weight matrices and tanh. This is more expressive than dot-product but computationally slower, making it common in RNN-based models.

**Q2. Answer: B**
Scaled dot-product attention computes QKᵀ/√d, applies softmax to get attention weights (probabilities), then multiplies by V to get the weighted output. The √d scaling prevents gradient issues.

**Q3. Answer: D**
In self-attention, queries, keys, and values are all derived from the same input sequence. In cross-attention, queries come from one sequence (e.g., decoder) while keys and values come from another (e.g., encoder).

**Q4. Answer: C**
Different heads learn to attend to different aspects—some may focus on syntactic proximity, others on semantic similarity, others on specific token types. This diversity increases the model's representational capacity.

**Q5. Answer: D**
With high-dimensional vectors (e.g., d=512), dot products become very large. Softmax of large values produces near-one-hot distributions where gradients are almost zero, making training extremely difficult.

**Q6. Answer: A**
Setting future scores to −∞ before softmax ensures exp(−∞) = 0, giving zero attention weight to future positions. This enforces the autoregressive constraint that position t cannot access positions > t.

**Q7. Answer: A**
RoPE applies rotation matrices to Q and K vectors, where the rotation angle depends on position. The dot product of rotated Q and K naturally encodes their relative distance, enabling better length generalization.

**Q8. Answer: B**
MQA uses one shared K-V pair for all query heads instead of separate K-V per head. This dramatically reduces the KV-cache size during inference, improving throughput for long sequences.

**Q9. Answer: D**
During autoregressive generation, previously generated tokens' K and V projections are cached so they don't need recomputation. Only the new token's K and V are computed and appended, giving O(n) per step instead of O(n²).

**Q10. Answer: B**
Flash Attention tiles the attention computation to maximize GPU SRAM usage and minimize slow HBM (high bandwidth memory) transfers. It computes exact attention (not an approximation) with significantly less memory.

**Q11. Answer: B**
Longformer combines local sliding-window attention (each token attends to nearby tokens) with global attention on selected tokens (e.g., [CLS]). This reduces complexity from O(n²) to O(n) while preserving long-range connectivity.

**Q12. Answer: D**
GQA groups query heads to share KV heads (e.g., 32 query heads with 8 KV groups). This balances MHA's expressiveness with MQA's memory efficiency, commonly used in models like Llama 2.

**Q13. Answer: A**
Softmax normalizes attention scores to non-negative values summing to 1 across key positions, making them interpretable as a probability distribution over which keys are most relevant to each query.

**Q14. Answer: B**
ALiBi adds a linear distance-based bias to attention scores before softmax, penalizing distant positions. This simple approach requires no learned position parameters and extrapolates well to longer sequences.

**Q15. Answer: D**
Cross-attention lets each decoder token (query) compute attention over all encoder hidden states (keys/values), dynamically focusing on the most relevant parts of the source sentence for each output token.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

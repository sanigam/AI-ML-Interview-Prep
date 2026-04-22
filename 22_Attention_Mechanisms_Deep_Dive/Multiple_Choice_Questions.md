# Multiple Choice Questions: Attention Mechanisms Deep Dive

📺 **Video Lecture:** https://youtu.be/te7D9Al7mpw


Test your understanding of attention mechanism concepts for AI/ML interviews.

---

**Q1. Additive (Bahdanau) attention computes scores using:**

A) Simple dot product of query and key
B) A learned nonlinear function: score = vᵀ·tanh(W_q·q + W_k·k)
C) Cosine similarity between query and key
D) Euclidean distance between query and key

---

**Q2. Scaled dot-product attention uses the formula:**

A) Attention = sigmoid(QKᵀ) · V
B) Attention = softmax(QKᵀ / √d) · V
C) Attention = ReLU(QKᵀ) · V
D) Attention = softmax(Q + K) · V

---

**Q3. Self-attention differs from cross-attention in that:**

A) Self-attention has no learnable parameters
B) In self-attention, Q, K, and V all come from the same sequence
C) Self-attention can only be applied to images
D) Cross-attention is always faster than self-attention

---

**Q4. The purpose of using multiple attention heads instead of one is to:**

A) Reduce the total number of parameters
B) Allow different heads to capture different types of relationships simultaneously
C) Eliminate the need for feed-forward layers
D) Speed up inference by a factor of h

---

**Q5. Without the 1/√d scaling factor in dot-product attention, what problem occurs?**

A) Attention weights become negative
B) Large dot products push softmax into saturated regions, causing vanishing gradients
C) The attention mechanism becomes non-differentiable
D) Keys and values become identical

---

**Q6. Causal masking in a decoder sets future attention scores to:**

A) Zero
B) One
C) Negative infinity (−∞), so softmax outputs zero for those positions
D) The average of all scores

---

**Q7. RoPE (Rotary Position Embedding) encodes positions by:**

A) Adding sinusoidal vectors to token embeddings
B) Applying rotation matrices to query and key vectors based on position
C) Learning a separate embedding for each position index
D) Concatenating position indices to token embeddings

---

**Q8. Multi-query attention (MQA) reduces memory during inference by:**

A) Using fewer query heads
B) Sharing a single set of key-value heads across all query heads
C) Eliminating the value projection entirely
D) Reducing the embedding dimension

---

**Q9. The KV-cache in autoregressive transformer inference stores:**

A) The gradients from the last backward pass
B) Previously computed key and value tensors to avoid recomputation at each generation step
C) The full attention weight matrix
D) Only the output logits

---

**Q10. Flash Attention improves transformer efficiency by:**

A) Using fewer attention heads
B) Reordering computation to minimize GPU memory reads/writes (IO-aware algorithm)
C) Replacing attention with convolution
D) Reducing the sequence length

---

**Q11. Sparse attention (as in Longformer) reduces O(n²) complexity by:**

A) Removing attention entirely for some layers
B) Restricting each token to attend to only a local window plus selected global tokens
C) Using only the first and last tokens
D) Halving the embedding dimension

---

**Q12. Grouped-query attention (GQA) is a compromise between:**

A) Self-attention and cross-attention
B) Multi-head attention (separate KV per head) and multi-query attention (shared KV for all heads)
C) Additive and multiplicative attention
D) Pre-LN and post-LN configurations

---

**Q13. Attention weights after softmax can be interpreted as:**

A) Probabilities summing to 1 over all key positions for each query
B) Raw similarity scores without normalization
C) Binary indicators of relevance
D) Gradient magnitudes for each position

---

**Q14. ALiBi (Attention with Linear Biases) handles positions by:**

A) Learning position embeddings from scratch
B) Adding a linear penalty −α|i−j| to attention scores based on distance between positions
C) Using convolutional position encoding
D) Ignoring position information entirely

---

**Q15. Cross-attention in a machine translation transformer allows the decoder to:**

A) Generate tokens without considering the source sentence
B) Attend to relevant parts of the encoded source sentence at each decoding step
C) Only look at the immediately previous source token
D) Share weights with the encoder

---

## Answer Key

**Q1. Answer: B**
Bahdanau (additive) attention uses a learned nonlinear function with weight matrices and tanh. This is more expressive than dot-product but computationally slower, making it common in RNN-based models.

**Q2. Answer: B**
Scaled dot-product attention computes QKᵀ/√d, applies softmax to get attention weights (probabilities), then multiplies by V to get the weighted output. The √d scaling prevents gradient issues.

**Q3. Answer: B**
In self-attention, queries, keys, and values are all derived from the same input sequence. In cross-attention, queries come from one sequence (e.g., decoder) while keys and values come from another (e.g., encoder).

**Q4. Answer: B**
Different heads learn to attend to different aspects—some may focus on syntactic proximity, others on semantic similarity, others on specific token types. This diversity increases the model's representational capacity.

**Q5. Answer: B**
With high-dimensional vectors (e.g., d=512), dot products become very large. Softmax of large values produces near-one-hot distributions where gradients are almost zero, making training extremely difficult.

**Q6. Answer: C**
Setting future scores to −∞ before softmax ensures exp(−∞) = 0, giving zero attention weight to future positions. This enforces the autoregressive constraint that position t cannot access positions > t.

**Q7. Answer: B**
RoPE applies rotation matrices to Q and K vectors, where the rotation angle depends on position. The dot product of rotated Q and K naturally encodes their relative distance, enabling better length generalization.

**Q8. Answer: B**
MQA uses one shared K-V pair for all query heads instead of separate K-V per head. This dramatically reduces the KV-cache size during inference, improving throughput for long sequences.

**Q9. Answer: B**
During autoregressive generation, previously generated tokens' K and V projections are cached so they don't need recomputation. Only the new token's K and V are computed and appended, giving O(n) per step instead of O(n²).

**Q10. Answer: B**
Flash Attention tiles the attention computation to maximize GPU SRAM usage and minimize slow HBM (high bandwidth memory) transfers. It computes exact attention (not an approximation) with significantly less memory.

**Q11. Answer: B**
Longformer combines local sliding-window attention (each token attends to nearby tokens) with global attention on selected tokens (e.g., [CLS]). This reduces complexity from O(n²) to O(n) while preserving long-range connectivity.

**Q12. Answer: B**
GQA groups query heads to share KV heads (e.g., 32 query heads with 8 KV groups). This balances MHA's expressiveness with MQA's memory efficiency, commonly used in models like Llama 2.

**Q13. Answer: A**
Softmax normalizes attention scores to non-negative values summing to 1 across key positions, making them interpretable as a probability distribution over which keys are most relevant to each query.

**Q14. Answer: B**
ALiBi adds a linear distance-based bias to attention scores before softmax, penalizing distant positions. This simple approach requires no learned position parameters and extrapolates well to longer sequences.

**Q15. Answer: B**
Cross-attention lets each decoder token (query) compute attention over all encoder hidden states (keys/values), dynamically focusing on the most relevant parts of the source sentence for each output token.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

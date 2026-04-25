# Transformer Architecture

📺 **Video Lecture:** https://youtu.be/yPTi4Ot5qoM


## Interview Anchor
- **Self-Attention Mechanism:** Each position attends to all other positions; compute relevance via query-key-value matrices; O(n²) complexity but highly parallelizable
- **Multi-Head Attention:** Multiple parallel attention heads capture diverse aspects; concatenate and project outputs
- **Positional Encoding:** Add position-dependent signals (sine/cosine or learned embeddings) since attention is permutation-invariant
- **Encoder-Decoder Structure:** Encoder applies self-attention in parallel; decoder applies masked self-attention + cross-attention to encoder outputs

## Key Concepts Overview
Transformers revolutionized machine learning by replacing recurrence and convolution with pure attention, achieving state-of-the-art across NLP, vision, and multi-modal tasks. The core insight—that all positions can attend to all others in parallel, avoiding sequential bottlenecks—unlocked unprecedented scaling to billions of parameters and trillions of tokens. Understanding attention mechanisms (scaled dot-product, multi-head), positional encodings, layer normalization, and the encoder-decoder design enables understanding modern architectures (BERT, GPT, Vision Transformers). This section covers fundamentals: self-attention computation, positional encoding strategies, transformer blocks, variants optimizing for efficiency, and how pre-training objectives (masked language modeling, next sentence prediction) drive modern NLP. Transformers are now ubiquitous; mastering this section is essential.

---

### Q1: Explain the self-attention mechanism. How does scaled dot-product attention work?

**A:** Self-attention computes a weighted sum of values, where weights are determined by query-key relevance. For a sequence of tokens [x_1, ..., x_n], compute three linear projections: queries Q = XW^Q, keys K = XW^K, values V = XW^V (W are learned weight matrices). For position i, the attention score with position j is: score(i,j) = (Q_i · K_j^T) / √d where d is the dimension. Normalize over all j via softmax: att_weight(i,j) = softmax(score(i,j)). Output for position i: output_i = Σ_j att_weight(i,j) × V_j. Scaled dot-product attention formula: Attention(Q, K, V) = softmax(QK^T / √d) × V. Scaling by 1/√d prevents dot products from becoming too large (gradient instability); without scaling, large dot products lead to very small softmax gradients. Benefits: (1) Parallelizable: all positions compute attention simultaneously (matrix operations). (2) Long-range dependencies: attention directly connects distant positions (no BPTT depth). (3) Interpretable: attention weights show which positions are relevant. Complexity: O(n²) for sequence length n (computing all attention scores). This is the main computational bottleneck. In interviews, explain the intuition: query asks "what am I looking for?", key matches "what am I?", value provides "what information to aggregate." Explaining scaling prevents numerical instability is a nice detail.

---

### Q2: What is multi-head attention? Why use multiple heads?

**A:** Multi-head attention applies multiple parallel attention mechanisms, each learning to attend to different aspects of the sequence. Instead of single Attention(Q, K, V), compute h heads: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V). Concatenate heads: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O where W^O is a learned output projection. Design: total dimension d is split across h heads (each head is d/h-dimensional), keeping total parameters constant. Benefits: (1) Diverse representations: different heads learn different attention patterns. Head 1 might attend to nearby words (syntax), head 2 to subject (semantics), etc. (2) Improved expressiveness: single head is a bottleneck. (3) Robustness: ensemble-like effect, reducing noise from single head. (4) Computational: parallel computation across heads is efficient. Standard: h=8 heads, d=512, so each head is 64-dimensional. Trade-off: more heads → more diverse but potentially noisier (few parameters per head). Fewer heads → concentrated but less diverse. In practice: 8-16 heads is standard across models. Visualization: attention heat maps (which words each head attends to) are often interpretable—some heads strongly attend to the previous word, others to entity references. In interviews, explain that multi-head attention is an ensemble mechanism; it captures multiple types of relationships simultaneously.

---

### Q3: Explain positional encoding. Why is it necessary?

**A:** Positional encoding adds position information to embeddings since self-attention is permutation-invariant (doesn't inherently know token order). Without positional encoding, "dog bites man" and "man bites dog" would have identical representations. Strategy 1 (Vaswani et al., 2017): sinusoidal encoding. For position pos and dimension d_i (even indices), PE(pos, d_i) = sin(pos / 10000^(d_i/d)). For odd indices, PE(pos, d_i) = cos(pos / 10000^(d_i/d)). Add to token embeddings: x_i' = embedding_i + PE(pos_i). Properties: (1) Unique for each (position, dimension) pair. (2) Periodic: encodes relative positions (PE(pos+k) has known relationship to PE(pos)). (3) Bounded: values in [-1, 1]. Strategy 2 (learned positional embeddings): treat position embeddings as learnable parameters, initialized randomly, updated during training. Benefits: (1) Potentially more flexible (data-driven). (2) Simpler intuition (explicit learnable positions). Drawback: must see positions during training; doesn't extrapolate to longer sequences. Sinusoidal better generalizes to longer sequences. Modern practice: sinusoidal is standard (BERT, GPT use learned; both work). Variants: relative positional embeddings (encode relative distance between tokens, not absolute position) used in some transformers (DeBERTa, T5). Relative encodings are more robust to sequence length variation. In interviews, explain that attention is position-agnostic; positional encoding injects position information. Mentioning sinusoidal periodicity shows understanding beyond "we just add position numbers."

---

### Q4: Describe the transformer encoder-decoder architecture. How do encoder and decoder differ?

**A:** Transformer consists of encoder stack and decoder stack, each with multiple identical layers. Encoder layer: (1) Multi-head self-attention: each position attends to all positions in input. (2) Feed-forward network (FFN): two linear layers with ReLU. Structure: x → MultiHeadAtt(x, x, x) → Add & Norm → FFN → Add & Norm → output. Decoder layer: (1) Masked multi-head self-attention: each position attends only to previous positions (and itself), preventing future leakage during training. Mask matrix has -∞ for future positions, zeroing softmax. (2) Cross-attention: query from decoder, key and value from encoder. Decoder attends to encoder outputs. Structure: y → MaskedMultiHeadAtt(y, y, y) → Add & Norm → CrossAtt(y, encoder_output, encoder_output) → Add & Norm → FFN → Add & Norm → output. Key differences: (1) Encoder sees full input (unmasked self-attention). (2) Decoder is autoregressive (masked, can't see future). (3) Decoder has cross-attention to encoder (encoder-decoder models). Sequence-to-sequence: encoder processes variable-length input, decoder generates variable-length output using encoder context. Example: machine translation, encoder encodes source sentence, decoder generates translation token-by-token, attending to encoder. Encoder-only (BERT): no decoder, just encoder stack. Used for classification, NER, extraction. Decoder-only (GPT): no encoder, just decoder stack. Used for language generation. In interviews, explain the masking mechanism—decoder can't cheat by looking ahead. Contrasting encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) shows architectural diversity.

---

### Q5: What is layer normalization in transformers? Why is it essential?

**A:** Layer normalization (LayerNorm) normalizes inputs across the feature dimension (not batch dimension like batch norm). For input x of dimension d, compute mean μ = (1/d)Σx_i and variance σ² = (1/d)Σ(x_i - μ)². Normalized: x_norm = (x - μ) / √(σ² + ε). Learnable scale γ and shift β: output = γ × x_norm + β. Implementation in transformer: apply LayerNorm before (pre-LN) or after (post-LN) attention and FFN. Pre-LN: x → LayerNorm → Attention → Add with residual. Post-LN (original transformer): x → Attention → Add with residual → LayerNorm. Modern practice: pre-LN is more stable for very deep networks. Benefits: (1) Stabilizes training: controls activation scales, enabling faster convergence. (2) Reduces sensitivity to initialization. (3) Works with any batch size (no batch statistics dependency). (4) Essential for transformers: without LayerNorm, gradients become unstable in deep stacks. Comparison: batch norm (cross-batch statistics) doesn't work well for transformers (variable sequence length, deterministic inference needed). LayerNorm (per-sample statistics) is deterministic and handles variable lengths. LayerNorm is now ubiquitous in deep networks (transformers, deep CNNs). In interviews, explain LayerNorm is architectural necessity for transformers, not optional regularization. Contrast with batch norm to show why LayerNorm is chosen.

---

### Q6: Explain residual connections in transformers. Why are they crucial?

**A:** Residual connections (skip connections) in transformers: x_out = Attention(x) + x (or FFN(x) + x). Instead of learning a full transformation, the network learns a residual Δx = Attention(x) + x - x = Attention(x). Benefits: (1) Gradient flow: ∂loss/∂x = ∂loss/∂x_out × (∂Attention/∂x + 1). The "+1" term ensures gradients flow even if ∂Attention/∂x ≈ 0. (2) Enables very deep networks: transformers often have 12-48 layers; without residuals, gradients vanish. (3) Optimization: residuals often make layers learn small perturbations (easier than learning full functions). (4) Stable training: residuals couple different layers, preventing independent, unstable learning. With residuals, large models train stably; without, training diverges. Residuals + LayerNorm are the backbone of transformer stability. Ablation studies show removing residuals cripples deep transformers. Design: residual after each sub-layer (attention, FFN), enabling deep stacking (48 layers in very deep models). In interviews, residuals are foundational to modern deep learning. Explaining that they enable gradient flow through deep networks (not just "residuals improve training") shows understanding.

---

### Q7: What is the feed-forward (FFN) layer in transformers?

**A:** FFN applies the same two-layer network to each position independently: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 (ReLU activation). Expansion and projection: W_1 projects from embedding dimension d_model (e.g., 512) to d_ff (e.g., 2048, often 4×d_model), then W_2 projects back to d_model. Effect: (1) Non-linearity: ReLU adds expressiveness (without it, attention + addition is linear). (2) Capacity: large d_ff increases model capacity. (3) Position-wise: each position transformed independently, then recombined via attention. Design choices: (1) Expansion ratio: 4× is standard, sometimes 2× or 8×. Larger → more parameters but slower. (2) Activation: ReLU is standard; GELU (Gaussian Error Linear Unit) is popular in modern models (smoother, slightly better). (3) Computation cost: FFN dominates transformer computation (2×(d_model × d_ff) per position); optimization here yields big speedups. Example: 512 → 2048 → 512 FFN with 1000 tokens: ~1B parameters per layer for this component. Variants: Mixture of Experts (MoE) sparsely activates different FFN experts per position (efficient scaling but training complexity). In interviews, explain FFN as the expressiveness mechanism complementing attention's ability to mix information across positions. Mentioning that FFNs are position-wise and can be parallelized shows understanding of transformer efficiency.

---

### Q8: Explain the computational complexity of transformers. What are efficiency challenges?

**A:** Transformer computational complexity: Self-attention is O(n²) in sequence length n (computing all pairwise interactions). For sequence length 1000, attention requires 1M operations. FFN is O(n × d²) where d is dimension (each position gets one FFN forward pass). Total per layer: O(n² + n×d²). With L layers: O(L(n² + n×d²)). For large sequences (10K+ tokens), attention dominates. Memory: O(n²) for attention matrices (score matrix, attention weights both n×n). Challenges: (1) Long sequences: 4K tokens is common, 8K-32K supported with optimization, 100K+ challenging. (2) Inference: each new token requires computing attention over all previous tokens (quadratic growth). (3) Training: storing activations for backprop requires O(n²) memory. Solutions: (1) Efficient attention variants: sparse attention (attend only to nearby tokens), linear attention (kernel methods), local attention (sliding window). Examples: Longformer (local + global), BigBird, Performer (linear). (2) Quantization: int8 reduces memory 4×. (3) Caching: store previous attention keys/values, incrementally compute new token's attention (KV cache). (4) Model distillation: smaller models with similar performance (ALBERT, DistilBERT). (5) Mixture of Experts: sparse activation reduces computation. Practical limits: 2K-4K tokens standard (BERT, GPT-2), up to 32K with specialized methods. In interviews, mentioning O(n²) complexity and its implications shows understanding of why transformers struggle with very long sequences and why efficient attention is a hot research area.

---

### Q9: What is masked language modeling (MLM)? How is it used in BERT?

**A:** Masked language modeling is a pre-training objective: randomly mask 15% of tokens, predict masked tokens using surrounding context. During training, model sees [CLS] The cat [MASK] on the [MASK] → predicts "sat" and "mat". Training objective: cross-entropy loss on masked positions only. Benefits: (1) Bidirectional context: predict left and right (unlike causal language modeling), learning richer representations. (2) Unsupervised: needs only raw text, no labels. (3) Robust: varied masking patterns prevent model from cheating (could memorize unmasked tokens). Masking strategy: (1) Mask 80% of the time (replace with [MASK] token). (2) Replace with random token 10% of the time. (3) Keep unchanged 10% of the time. Randomness forces model to maintain accurate hidden representations, not relying on [MASK] token. BERT (Bidirectional Encoder Representations from Transformers): pre-trains on MLM (+ next sentence prediction, NSP). After pre-training on large corpus (Wikipedia, BookCorpus), fine-tune on downstream tasks (classification, NER, QA). MLM is deceptively simple but highly effective. BERT's success drove adoption of MLM for other domains (DomainBERT, BioBERT for biomedical text). Extension: ELECTRA uses discriminator-generator objectives (more efficient pre-training). In interviews, explain MLM intuition: predicting masked tokens forces learning meaningful representations (can't shortcut without understanding context). Mentioning masking strategy details (80% mask, 10% random, 10% unchanged) shows rigor.

---

### Q10: What is the difference between BERT and GPT architectures?

**A:** BERT (Google, 2019): encoder-only, bidirectional pre-training. Architecture: 12-48 transformer encoder layers, no decoder. Pre-training: MLM (masked language modeling) + NSP (next sentence prediction, predict whether sentence B follows A). Training: bidirectional context (left and right), learns from both directions. Output: [CLS] token representation used for classification, position-wise representations for token-level tasks. Tasks: classification, NER, QA extraction. GPT (OpenAI): decoder-only, autoregressive pre-training. Architecture: 12-96 transformer decoder layers (masked self-attention only). Pre-training: causal language modeling (predict next token from previous). Training: unidirectional, left-to-right only. Output: generate sequences autoregressively. Tasks: text generation, language modeling, can be fine-tuned for classification (via task-specific prompting). Key differences: (1) Architecture: BERT bidirectional, GPT unidirectional. (2) Pre-training: BERT masked, GPT causal. (3) Inference: BERT takes input and outputs representation (static), GPT generates tokens sequentially (dynamic). (4) Fine-tuning: BERT for understanding, GPT for generation. (5) Zero-shot: GPT can perform tasks without fine-tuning (in-context learning), BERT requires fine-tuning. Modern variants: (1) Unified (T5): encoder-decoder, masked prediction like BERT, generates like GPT. (2) Decoder-only scaled (GPT-3, GPT-4): massive models, impressive few-shot/zero-shot via scaling. Current trend: decoder-only models (GPT series) are dominant due to scaling laws (more parameters → better in-context learning). In interviews, explain architectural trade-offs: encoder (BERT) efficient for classification, decoder (GPT) flexible for generation, encoder-decoder (T5) combining both.

---

### Q11: What is in-context learning and how do large language models (LLMs) achieve it?

**A:** In-context learning: models learn from task examples in the prompt without parameter updates. Example: "Translate to French: Hello → Bonjour. Hi → Salut. How are you? →" LLM outputs "Comment allez-vous?" without training on translation data. Mechanism (hypothesized): during pre-training, models see diverse examples and learn statistical patterns. Prompting with examples activates relevant circuit, biasing predictions toward task. Scale matters: small models (GPT-2, BERT) show little in-context learning; large models (GPT-3, GPT-3.5, GPT-4) show strong in-context learning. Scaling laws: performance improves with model size and training data. Few-shot prompting: provide k examples in prompt (typically 1-5). Zero-shot: no examples, just task instruction ("Translate to French: ..."). Chain-of-thought prompting: ask model to "think step-by-step" before answering. Surprisingly effective—models that can explain reasoning give better answers. Benefits: (1) No fine-tuning needed. (2) Fast adaptation to new tasks. (3) Can handle unseen domains. Limitations: (1) Context window size limits (can't include massive examples). (2) Brittleness: small prompt changes affect outputs. (3) Hallucination: models generate plausible-sounding but false information. Research: emerging field understanding in-context learning mechanistically (what happens in transformers during in-context learning). Recent work suggests in-context learning is implicit meta-learning. In interviews, mention in-context learning as a paradigm shift from fine-tuning to prompting, and scaling laws explaining why it emerges in large models.

---

### Q12: What are efficient transformer variants? How do they reduce O(n²) complexity?

**A:** O(n²) attention is prohibitive for long sequences. Efficient variants: (1) Sparse attention: attend only to nearby tokens + few global tokens. Longformer: local window (e.g., 512 tokens) + task-specific global attention. Complexity: O(n × window_size). (2) Linear attention: approximate attention via kernel tricks. Performer: use kernel φ(Q) × φ(K)^T ≈ QK^T. Complexity: O(n). Trade-off: approximation errors, but much faster. (3) Multi-scale attention: Linformer attends via projected keys/values (reduce n to k). Complexity: O(n×k). (4) Low-rank approximation: attention matrix is low-rank (empirically), use low-rank decomposition. (5) Retrieval-based: only attend to top-k similar positions (learnable retrieval). (6) Mixture of Experts (MoE): each position uses only a subset of the model (sparse activation). SWITCH Transformers: each token routed to expert(s). Compute-efficient but training complexity. (7) FlashAttention (optimization, not approximation): reorder computation to reduce memory I/O. ~2-4× faster with no accuracy loss. Special-purpose: (1) RoPE (Rotary Positional Embeddings): use rotation matrices for relative positions, better length extrapolation. (2) Alibi (Attention with Linear Biases): no explicit positional embeddings, add bias term. Practical: Full attention often sufficient for typical lengths (2K-4K). Efficient variants for longer (>8K) or resource-constrained settings. Recent trend: FlashAttention (no approximation, just optimization) is becoming standard in modern implementations (Llama, Mistral). In interviews, mention O(n²) bottleneck and know 2-3 solutions (sparse, linear, local). Discussing practical choices (when to use each) shows depth.

---

### Q13: What are Mixture of Experts (MoE) transformers? How do they scale?

**A:** Mixture of Experts: instead of dense FFN per position, use sparse routing to multiple expert networks. Each position (token) routed to k experts (typically k=2) out of total E experts (e.g., 64). Architecture: expert networks are FFN layers; router network predicts routing probabilities via learned matrix: route_scores = softmax(token × W_router), select top k. Token combined: output = Σ_{i ∈ top_k} route_scores_i × expert_i(token). Benefits: (1) Conditional computation: only k out of E experts compute per token, reducing FLOPs. (2) Scaling: can add experts without proportional compute increase. (3) Specialization: experts learn diverse features. SWITCH Transformers: k=1 (single expert per token) for simplicity. Scaling: 1.6T parameters, uses ~16× fewer FLOPs than dense equivalent. Trade-offs: (1) Routing instability: router can collapse (all tokens to same expert). Solutions: auxiliary loss penalizing unbalanced routing. (2) Load balancing: ensure experts used equally (avoid some experts unused). (3) Training complexity: distributed across many GPUs (routing logic, expert sharding). (4) Memory: all expert parameters in memory (not reduced vs. dense). Practical: Google's PaLM and Gemini use MoE. Benefits emerge at very large scale (1T+ parameters). Smaller models often don't benefit. In interviews, MoE is cutting-edge; mentioning it shows awareness of scaling research. Explain the trade-off: compute savings vs. training complexity and memory.

---

### Q14: What is vision transformer (ViT)? How are images treated as sequences?

**A:** Vision Transformer applies transformer encoder directly to images. Process: (1) Patch embedding: divide image (e.g., 224×224) into patches (e.g., 16×16 = 196 patches). (2) Linear projection: flatten each patch, project to embedding dimension d. (3) Positional encoding: add learned positional embeddings to patch embeddings. (4) Append [CLS] token (like BERT). (5) Pass through transformer encoder layers. (6) Use [CLS] representation for classification (or use all patches for dense tasks). Benefits: (1) No convolutions: attention directly processes all spatial regions. (2) Scalability: transformers scale to huge models (billions of parameters). (3) Transfer learning: pre-train on large image dataset (JFT-300M), fine-tune on downstream. Challenges: (1) Requires large datasets: ImageNet insufficient. (2) Computational cost: O(n²) where n = (image_size / patch_size)². For 224×224, 16×16 patches: 196 patches, feasible. Larger images → more patches → quadratic blow-up. (3) Inductive bias: CNNs built in locality (convolutions), position equivariance (pooling). ViT must learn these. Solutions: (1) Hybrid ViT: CNN stem extracts features, then transformer. (2) Data augmentation: essential (image transformation invariance not built in). (3) Pre-training: large models trained on massive data transfer better. Adoption: ViT-L, ViT-H standard in modern vision. Swin Transformer (hierarchical ViT with local windows) improves efficiency. In interviews, ViT represents transformer universality across modalities. Explain patch tokenization and why large data is needed (learning CNN-like locality without inductive bias).

---

### Q15: What is tokenization and how do BPE, WordPiece, and SentencePiece differ?

**A:** Tokenization converts text into subword tokens, the atomic units transformers process. Character-level (rare) has huge vocabulary and long sequences. Word-level has issues (rare words, OOV). Subword tokenization balances vocabulary size and sequence length. (1) BPE (Byte Pair Encoding): iteratively merge most frequent character pairs. Start with characters, merge "t" + "h" → "th", then "th" + "e" → "the", etc., until target vocabulary size. Greedy merge order; deterministic once merges fixed. Used in GPT. (2) WordPiece: similar to BPE but uses likelihood (probability of merged pair) not frequency. Merkle: merge pairs that maximize language model likelihood. Used in BERT. (3) SentencePiece: language-agnostic, handles multiple languages uniformly. Treats input as byte sequence, learns merges without assuming tokenization. Used in XLM, mT5. Differences: BPE is frequency-based (greedy), WordPiece is likelihood-based (principled), SentencePiece is language-agnostic (universal). Vocabulary size: typically 30K-50K tokens (BERT), up to 100K+ (large models). Trade-off: larger vocabulary reduces sequence length (fewer subwords per word) but increases embedding/output layer parameters. Practical: BPE, WordPiece, SentencePiece all work well; choice is often library-driven (HuggingFace Tokenizers supports all). Impact: tokenization affects model behavior (different tokenizations → different subword sequences → different learned representations). Example: "unhappy" → ["un", "happy"] (WordPiece) vs. ["u", "n", "h", "a", "p", "p", "y"] (character-level). First representation is more semantically meaningful. In interviews, tokenization is often overlooked but impacts everything downstream. Mentioning vocabulary size-sequence length trade-off shows understanding of practical constraints.

---

## Interview Cheatsheet

**Key Terms:**
- **Self-Attention:** All positions attend to all positions; parallelizable; O(n²) complexity
- **Scaled Dot-Product Attention:** Attention(Q, K, V) = softmax(QK^T / √d) × V; scaling prevents gradient instability
- **Multi-Head Attention:** h parallel attention heads; diverse representations; concatenate and project outputs
- **Positional Encoding:** Add position information (sinusoidal or learned); makes attention position-aware
- **Encoder:** Bidirectional self-attention; used for understanding/representation
- **Decoder:** Masked self-attention (causal) + cross-attention to encoder; used for generation
- **Encoder-Decoder:** Full transformer; seq2seq, translation, summarization
- **Layer Normalization:** Normalize per sample across features; essential for stability
- **Residual Connection:** Skip connection (x + F(x)); enables gradient flow through deep networks
- **Feed-Forward Network (FFN):** Two-layer network with expansion/projection; position-wise; adds nonlinearity
- **Masked Attention:** Future tokens masked (set to -∞); prevents decoder from cheating
- **BERT:** Encoder-only, bidirectional MLM pre-training; understanding tasks (classification, NER)
- **GPT:** Decoder-only, causal language modeling; generation, in-context learning
- **Vision Transformer (ViT):** Patches as tokens; image classification; requires large data
- **In-Context Learning:** Learn from task examples in prompt; emerges with scale
- **BPE (Byte Pair Encoding):** Subword tokenization; frequency-based merges
- **WordPiece:** Subword tokenization; likelihood-based merges
- **SentencePiece:** Language-agnostic subword tokenization; universal approach
- **Efficient Attention:** Sparse (local + global), linear (kernel), multi-scale; reduce O(n²) complexity
- **Mixture of Experts:** Sparse routing to k experts per token; conditional computation; scales efficiently

**Rapid-Fire Q&A:**
- **Q: Why scaled dot-product attention?** **A:** Scaling by 1/√d prevents dot products from exploding, stabilizes gradients
- **Q: Multi-head attention benefit?** **A:** Diverse representations; ensemble-like effect; different heads learn different patterns
- **Q: Why positional encoding?** **A:** Attention is permutation-invariant; encoding injects position information
- **Q: Encoder vs. decoder difference?** **A:** Encoder: bidirectional self-attention; decoder: masked (causal) self-attention
- **Q: Why layer norm in transformers?** **A:** Stabilizes training; works with any batch size; essential for depth
- **Q: Residual connections purpose?** **A:** Enable gradient flow; allow very deep networks
- **Q: BERT or GPT for classification?** **A:** BERT (bidirectional, fine-tuned); GPT needs prompting or fine-tuning
- **Q: GPT for generation?** **A:** Autoregressive; generates left-to-right; in-context learning with scale
- **Q: ViT why large data needed?** **A:** No built-in locality like CNNs; must learn spatial structure
- **Q: How reduce O(n²) complexity?** **A:** Sparse attention, linear attention (kernel), local windows, retrieval

---

## Interview Tips
- **Draw scaled dot-product attention:** Sketch Q, K, V matrices, dot product, softmax, value aggregation
- **Explain masking mechanism:** Decoder's key insight; prevents cheating in auto-regressive generation
- **Master positional encoding intuition:** Don't memorize sine/cosine formula; explain why position matters
- **Discuss BERT vs. GPT thoughtfully:** Both important; emphasize architectural implications (bidirectional vs. causal)
- **Mention recent advances:** FlashAttention (optimization), in-context learning, scaling laws (important trends)
- **Prepare ViT explanation:** Patches, tokenization, why large data, computational cost
- **Discuss tokenization impact:** Vocabulary size → sequence length tradeoff; impacts downstream performance
- **Explore efficiency:** O(n²) bottleneck and solutions (sparse, linear); separates practitioners from novices
- **Relate to applications:** Translation (encoder-decoder), classification (BERT), generation (GPT), vision (ViT)
- **Highlight interpretability:** Attention weights show which positions are relevant; advantage over CNNs/RNNs

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

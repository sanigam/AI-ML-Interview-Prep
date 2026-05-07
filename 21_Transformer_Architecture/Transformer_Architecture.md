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

**A:** Self-attention computes a weighted sum of *values*, where the weights are determined by how relevant each *key* is to the current *query*.

**Step 1 — produce Q, K, V from the input.** For a sequence of token embeddings X, three learned linear projections give queries, keys, and values:

```
Q = X · W^Q          (queries)
K = X · W^K          (keys)
V = X · W^V          (values)
```

**Step 2 — compute attention scores between each query and each key:**

```
score(i, j) = (Q_i · K_jᵀ) / √d
```

The factor 1/√d is the "scaled" part — without it, the dot products grow with d and push softmax into very small-gradient regions.

**Step 3 — softmax over keys, then aggregate values:**

```
att_weight(i, j) = softmax_j ( score(i, j) )
output_i        = Σ_j att_weight(i, j) · V_j
```

In compact matrix form, this is the canonical scaled dot-product attention formula:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

**Why this design works:**

- **Parallelizable** — all positions compute attention simultaneously via matrix ops.
- **Long-range dependencies** — directly connects any two positions, no BPTT-style depth.
- **Interpretable** — the attention weights show which positions matter for each output.

**Complexity:** O(n²) for sequence length n — this is the main computational bottleneck of transformers.

**Interview intuition:** the query asks *"what am I looking for?"*, the key answers *"what am I?"*, and the value carries *"what information to aggregate."* Mentioning that the 1/√d scaling exists to prevent vanishing softmax gradients is a useful detail.

---

### Q2: What is multi-head attention? Why use multiple heads?

**A:** **Multi-head attention** runs h attention mechanisms in parallel, each with its own learned projections, then combines their outputs. Each head can specialize in a different relationship pattern.

```
head_i           = Attention( Q·W_i^Q, K·W_i^K, V·W_i^V )

MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O
```

W^O is a learned output projection that combines the per-head outputs.

**Design choice:** the total embedding dimension d is split across h heads, so each head works in dimension d/h. This keeps the total parameter count roughly constant whether you use 1 large head or 8 smaller ones.

**Why multiple heads help:**

- **Diverse representations** — different heads learn different attention patterns. One head might attend to the previous word (syntax), another to subject-verb agreement (semantics), another to entity references.
- **Expressiveness** — a single head is an information bottleneck.
- **Ensemble-like effect** — averaging across heads reduces noise from any individual head.
- **Computationally cheap** — heads run in parallel.

**Standard configuration:** h = 8 heads with d = 512, giving 64-dimensional heads.

**Tradeoff:** more heads give more diversity but each head has fewer parameters and can be noisier. Fewer heads give more concentrated representations but less diversity. Most modern models use 8–16 heads.

Attention heat maps from trained models are often interpretable — some heads consistently attend to the previous word, others to coreference targets. In interviews, frame multi-head attention as an *ensemble mechanism* that captures multiple types of relationships simultaneously.

---

### Q3: Explain positional encoding. Why is it necessary?

**A:** Self-attention is **permutation-invariant** — it doesn't know token order. Without positional information, "dog bites man" and "man bites dog" would produce identical representations. **Positional encoding** adds position-dependent signals to each token embedding.

**Strategy 1 — sinusoidal encoding** (Vaswani et al., 2017). At position `pos`, alternate sine and cosine across the embedding dimensions:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d) )
```

Then simply add it to the token embedding:

```
x_i′ = embedding_i + PE(pos_i)
```

Properties:

- **Unique** for each (position, dimension) pair.
- **Periodic** — PE(pos + k) has a fixed linear relationship to PE(pos), which encodes relative positions implicitly.
- **Bounded** — values stay in [−1, 1].

**Strategy 2 — learned positional embeddings.** Treat position vectors as learnable parameters that get updated during training (used in BERT and original GPT).

- *Pro:* more flexible, data-driven.
- *Con:* you can only attend to positions seen during training — doesn't extrapolate to longer sequences. Sinusoidal generalizes better in this regard.

**Variants:**

- **Relative positional embeddings** (DeBERTa, T5) encode the *distance* between tokens, not absolute position. More robust to varying sequence lengths.
- **RoPE (Rotary Positional Embeddings)** rotate Q and K vectors based on position; popular in modern decoder-only LLMs (Llama).
- **ALiBi (Attention with Linear Biases)** skips explicit positional embeddings and adds a position-dependent bias to attention scores.

In interviews, the key point is that attention is position-agnostic and positional encoding injects the position information. Mentioning sinusoidal periodicity (as opposed to "we just add position numbers") shows deeper understanding.

---

### Q4: Describe the transformer encoder-decoder architecture. How do encoder and decoder differ?

**A:** A transformer is built from two stacks of identical layers — an encoder and a decoder — though many modern models use only one of them.

**Encoder layer.** Each layer applies:

1. Multi-head self-attention (every position attends to every position).
2. Position-wise feed-forward network (FFN, two linear layers with ReLU).

```
x → MultiHeadAttn(x, x, x) → Add & Norm → FFN → Add & Norm → output
```

**Decoder layer.** Each layer has *three* sub-blocks:

1. **Masked** multi-head self-attention — each position attends only to previous positions (and itself). The mask sets future positions to −∞, so softmax zeros them out, preventing leakage of future tokens during training.
2. Cross-attention — query comes from the decoder, key and value come from the encoder. This is how the decoder attends to the source sequence.
3. FFN.

```
y → MaskedMultiHeadAttn(y, y, y) → Add & Norm
  → CrossAttn(y, enc_out, enc_out) → Add & Norm
  → FFN → Add & Norm → output
```

**Key differences:**

- **Encoder** sees the full input (unmasked self-attention).
- **Decoder** is autoregressive (masked self-attention — no peeking at future tokens).
- **Decoder** has an extra cross-attention block that connects to the encoder.

**Sequence-to-sequence flow.** The encoder processes variable-length input; the decoder generates variable-length output one token at a time, attending to the encoder. Machine translation is the canonical example.

**Architectural variants:**

- **Encoder-only (BERT):** just the encoder stack. Used for classification, NER, span extraction.
- **Decoder-only (GPT):** just the decoder stack. Used for autoregressive generation.
- **Encoder-decoder (T5, original Transformer):** both. Used for translation, summarization, anything seq2seq.

In interviews, the masking mechanism is the headline — decoder can't cheat by peeking at future tokens. Contrasting the three configurations (BERT vs GPT vs T5) shows architectural fluency.

---

### Q5: What is layer normalization in transformers? Why is it essential?

**A:** **Layer normalization (LayerNorm)** normalizes the activations of each token independently across its feature dimension. Unlike batch norm, it doesn't depend on batch statistics.

For an input vector x of dimension d:

```
μ      = (1/d) · Σᵢ xᵢ                          # mean across features
σ²     = (1/d) · Σᵢ (xᵢ − μ)²                    # variance across features
x_norm = (x − μ) / √(σ² + ε)                    # normalize
output = γ · x_norm + β                          # learnable scale and shift
```

**Pre-LN vs post-LN placement.** LayerNorm can be applied either before or after each sub-block:

```
post-LN  (original):  x → Attention → Add residual → LayerNorm
pre-LN   (modern):    x → LayerNorm → Attention → Add residual
```

Modern practice favors pre-LN — it's more stable in very deep networks (50+ layers).

**Why it matters:**

- Stabilizes training by controlling activation scales — enables faster convergence.
- Reduces sensitivity to initialization.
- Works at any batch size — no dependency on batch statistics.
- Essential for transformer depth — gradients become unstable without it.

**Why not batch norm?** Batch norm uses statistics across the batch, which causes problems with variable sequence length and would make inference depend on batch composition. LayerNorm uses per-sample statistics, so it's deterministic and length-agnostic.

In interviews, frame LayerNorm as an *architectural necessity* for transformers, not optional regularization, and contrast with batch norm to show why it's the right choice.

---

### Q6: Explain residual connections in transformers. Why are they crucial?

**A:** A **residual (skip) connection** adds the input of a sub-block back to its output:

```
x_out = Attention(x) + x          (or FFN(x) + x)
```

Instead of learning a full transformation, the layer effectively learns a *residual* — what to *add* to the input rather than what to *output*.

**Gradient flow.** The gradient through a residual connection is:

```
∂loss/∂x = ∂loss/∂x_out · ( ∂Attention/∂x + 1 )
```

The crucial **"+1" term** lets gradients flow through even if the attention block's Jacobian is near zero — preventing the vanishing gradients that would otherwise plague deep stacks.

**Why this matters in transformers:**

- Enables very deep networks — most modern transformers have 12–96 layers; without residuals, gradients vanish.
- Layers tend to learn small perturbations, which is much easier than learning full transformations from scratch.
- Couples layers together, stabilizing training.

Residual connections + LayerNorm are the two pillars of transformer stability. Ablation studies consistently show that removing residuals cripples deep transformers.

**Design pattern:** add a residual after each sub-layer (attention and FFN), which is what enables stacking 48+ layers cleanly.

In interviews, the deeper point is that residuals enable *gradient flow through deep networks* — not just a vague "they improve training."

---

### Q7: What is the feed-forward (FFN) layer in transformers?

**A:** The **FFN** is a two-layer MLP applied to each position independently:

```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
```

It expands from the model dimension d_model (e.g., 512) up to a larger hidden dimension d_ff (e.g., 2048, typically 4× larger), then projects back down to d_model.

**What it contributes:**

- **Non-linearity** — without ReLU (or GELU), the rest of the transformer is largely linear.
- **Capacity** — the inflated middle dimension is where most of the model's parameters live.
- **Position-wise** — each token is transformed independently, then mixed across positions via the next attention layer.

**Design choices:**

- **Expansion ratio** — 4× is standard; sometimes 2× or 8×. Larger gives more capacity but slower compute.
- **Activation** — ReLU is the original; modern models prefer **GELU** (smoother, slightly better empirically) or SwiGLU (especially in Llama-style architectures).
- **Compute cost** — FFNs dominate transformer FLOPs (2 · d_model · d_ff per position), so optimizations here pay off most.

**Variants — Mixture of Experts (MoE):** instead of one FFN, route each token to a small subset of "expert" FFNs. Sparsely activates a fraction of the model per token, enabling efficient scaling but adding training complexity.

In interviews, frame FFNs as the *expressiveness mechanism* that complements attention's ability to mix information across positions. Mentioning that FFNs are position-wise (and trivially parallelizable) shows understanding of where transformers' efficiency comes from.

---

### Q8: Explain the computational complexity of transformers. What are efficiency challenges?

**A:** **Per-layer complexity** for sequence length n and model dimension d:

```
Self-attention:   O(n²)              # all pairwise score computations
FFN:              O(n · d²)           # one MLP per position

Total per layer:  O(n² + n·d²)
With L layers:    O( L · (n² + n·d²) )
```

For long sequences, attention dominates; for short sequences with big d, FFN dominates.

**Memory cost** of attention is also O(n²) — both the score matrix and the attention weight matrix are n × n.

**Practical challenges:**

- **Long sequences** — 4K tokens is comfortable, 8K–32K is supported with optimization, 100K+ is hard.
- **Inference** — each new token requires attending over all previous tokens (cost grows linearly per token, quadratically over the full sequence).
- **Training memory** — backprop requires storing the n × n activations.

**Solutions:**

- **Efficient attention variants:**
  - *Sparse attention* — attend only to nearby tokens or a few global ones (Longformer, BigBird).
  - *Linear attention* — kernel approximations that reduce attention to O(n) (Performer).
  - *Local windowed attention* — sliding-window patterns.
- **Quantization** — int8 (or even int4) reduces memory by 4× or more.
- **KV cache** — cache previous keys/values during generation; each new token only computes its own row of attention. Standard in LLM serving.
- **Distillation** — smaller models that approximate larger ones (DistilBERT, ALBERT).
- **Mixture of Experts** — sparse activation, fewer FLOPs per token at the same parameter count.
- **FlashAttention** — a memory-aware reorganization that achieves the same exact attention 2–4× faster, no approximation. Now standard in modern implementations.

**Practical limits today:** 4K–8K tokens for general purpose; 32K–128K achievable with specialized methods. In interviews, naming the O(n²) bottleneck and 2–3 mitigation strategies (sparse, linear, FlashAttention) shows depth.

---

### Q9: What is masked language modeling (MLM)? How is it used in BERT?

**A:** **Masked language modeling** is a pre-training objective where you randomly mask out a fraction of tokens (typically 15%) and train the model to predict them from the surrounding context.

```
Input:    [CLS] The cat [MASK] on the [MASK]
Predict:  "sat" at position 4, "mat" at position 7
```

The training loss is cross-entropy applied only at the masked positions.

**Why MLM works well:**

- **Bidirectional context** — the model sees both left and right context (unlike causal LM), learning richer representations.
- **Unsupervised** — only raw text needed, no labels.
- **Robust** — varied masking prevents the model from "cheating" by memorizing patterns.

**The 80/10/10 masking trick.** When choosing how to corrupt a selected token:

- 80% of the time → replace with the [MASK] token.
- 10% of the time → replace with a random token.
- 10% of the time → keep it unchanged.

The mix forces the model to maintain accurate hidden representations for *every* position, not just rely on the [MASK] symbol as a hint.

**BERT** (Bidirectional Encoder Representations from Transformers) pre-trains on MLM plus a next-sentence-prediction (NSP) objective. After pre-training on a large corpus (Wikipedia, BookCorpus), it's fine-tuned on downstream tasks like classification, NER, and span QA.

MLM is deceptively simple but highly effective. BERT's success drove MLM adoption in many domains — BioBERT for biomedical text, ClinicalBERT for medical records, etc.

A notable extension is **ELECTRA**, which uses a discriminator-generator setup that's more sample-efficient than vanilla MLM.

In interviews, the intuition to convey is that predicting masked tokens forces the model to learn meaningful contextual representations — it can't shortcut without understanding the surrounding context. Mentioning the 80/10/10 split is a useful rigor signal.

---

### Q10: What is the difference between BERT and GPT architectures?

**A:** **BERT** (Google, 2019) — encoder-only, bidirectional pre-training.

- *Architecture:* 12–48 transformer encoder layers; no decoder.
- *Pre-training:* MLM (masked language modeling) + NSP (predict whether sentence B follows A).
- *Context:* bidirectional — sees both left and right.
- *Outputs:* [CLS] token embedding for classification, per-position embeddings for token-level tasks.
- *Tasks:* classification, NER, span-based QA.

**GPT** (OpenAI) — decoder-only, autoregressive pre-training.

- *Architecture:* 12–96 transformer decoder layers (masked self-attention only).
- *Pre-training:* causal language modeling — predict next token from previous tokens.
- *Context:* unidirectional, left-to-right.
- *Outputs:* generate tokens autoregressively.
- *Tasks:* text generation, language modeling; can also be steered for classification via prompting.

**Key differences at a glance:**

- *Architecture:* BERT is bidirectional, GPT is unidirectional.
- *Pre-training objective:* BERT masks, GPT predicts the next token.
- *Inference:* BERT runs once and outputs a representation; GPT runs token-by-token.
- *Fine-tuning:* BERT excels at understanding, GPT excels at generation.
- *Zero-shot:* GPT can do tasks via in-context examples; BERT typically needs fine-tuning.

**Modern variants:**

- **T5** — encoder-decoder; masked prediction like BERT, generation like GPT.
- **Scaled decoder-only (GPT-3/4, Llama, Claude):** massive models with strong few-shot and zero-shot performance from scale alone.

**Current trend:** decoder-only models dominate at large scale because of scaling laws — more parameters and data give better in-context learning. In interviews, explain the architectural tradeoffs: encoder (BERT) is efficient for understanding, decoder (GPT) is flexible for generation, encoder-decoder (T5) combines both.

---

### Q11: What is in-context learning and how do large language models (LLMs) achieve it?

**A:** **In-context learning** lets a model learn from examples in the prompt without any parameter updates:

```
Translate to French:
Hello → Bonjour
Hi → Salut
How are you? →
```

The LLM completes "Comment allez-vous?" without ever being explicitly trained on a translation task.

**Mechanism (hypothesized):** during pre-training, the model sees an enormous variety of contextual patterns. At inference time, examples in the prompt activate the relevant pattern and bias predictions toward the task.

**Scale matters.** Small models (GPT-2, BERT) show little in-context learning. Large models (GPT-3, GPT-4, Claude) exhibit strong in-context learning. Scaling laws — performance grows smoothly with model size and training data — explain why this capability emerges only at scale.

**Prompting variants:**

- **Few-shot** — provide k examples in the prompt (typically 1–5).
- **Zero-shot** — just give the task instruction, no examples.
- **Chain-of-thought** — ask the model to "think step-by-step." Surprisingly effective: explaining reasoning often improves accuracy.

**Benefits:**

- No fine-tuning required.
- Fast adaptation to new tasks at inference time.
- Handles unseen domains via instructions.

**Limitations:**

- Bounded by the context window — you can only fit so many examples.
- Brittle — small prompt changes can shift outputs significantly.
- Hallucination — models confidently produce plausible-sounding but incorrect content.

There's an active research area trying to understand in-context learning mechanistically. Recent work suggests it can be viewed as a form of implicit meta-learning encoded in the weights.

In interviews, frame in-context learning as a paradigm shift from fine-tuning to prompting, and use scaling laws to explain why it emerges only in large models.

---

### Q12: What are efficient transformer variants? How do they reduce O(n²) complexity?

**A:** O(n²) attention becomes prohibitive for long sequences. Efficient variants attack this in different ways.

**Reduce the attention pattern:**

- **Sparse attention** — attend only to nearby tokens plus a few global tokens (Longformer, BigBird). Complexity: O(n · window_size).
- **Local windowed attention** — sliding window only.
- **Retrieval-based** — attend only to the top-k most similar positions.

**Approximate the attention matrix:**

- **Linear attention** — kernel approximations make attention factorize. Performer uses φ(Q) · φ(K)ᵀ ≈ Q·Kᵀ for some feature map φ, giving O(n) attention. Some accuracy cost, but much faster.
- **Linformer** — project keys and values to a lower-dimensional k. Complexity: O(n · k).
- **Low-rank decomposition** — exploit the empirically low-rank structure of attention matrices.

**Reduce per-token work:**

- **Mixture of Experts (MoE)** — each token uses only a subset of FFN experts (e.g., SWITCH Transformer routes each token to one expert). Compute-efficient but more complex to train.

**Optimize without approximation:**

- **FlashAttention** — reorders the computation to minimize memory I/O. 2–4× faster with *zero* accuracy loss. Now standard in modern implementations (Llama, Mistral).

**Better positional encodings (orthogonal but related):**

- **RoPE** (Rotary Positional Embeddings) — rotation matrices on Q/K, better length extrapolation.
- **ALiBi** — attention biases that decay linearly with distance, no learned positional embeddings needed.

**Practical guidance:** full attention is fine up to a few thousand tokens. For 8K+, reach for FlashAttention first (it's exact); use approximations like sparse or linear attention only when memory is the constraint. In interviews, naming the O(n²) bottleneck and 2–3 mitigations (sparse, linear, FlashAttention) is plenty.

---

### Q13: What are Mixture of Experts (MoE) transformers? How do they scale?

**A:** **Mixture of Experts (MoE)** replaces the dense FFN with a sparsely activated set of expert networks. Each token is routed to k out of E experts (typically k = 2 of E = 64 or so).

**Architecture:**

```
route_scores = softmax( token · W_router )         # router picks experts
output       = Σ_{i ∈ top_k} route_scores_i · expert_i(token)
```

The experts are themselves FFN layers; only the top-k experts run for any given token.

**Benefits:**

- **Conditional computation** — only k out of E experts run per token, so FLOPs stay low.
- **Scaling** — you can add experts (and parameters) without proportionally increasing compute.
- **Specialization** — different experts tend to learn different features.

**SWITCH Transformer** uses k = 1 for simplicity and reaches 1.6T parameters using ~16× fewer FLOPs than a dense model of the same size.

**Tradeoffs:**

- **Routing instability** — the router can collapse, sending all tokens to a single expert. Mitigated by auxiliary load-balancing losses.
- **Load balancing** — making sure experts are used roughly equally.
- **Training complexity** — sharding experts across many GPUs and routing tokens between them is non-trivial.
- **Memory** — even though FLOPs are sparse, all expert parameters must still fit in memory.

**Practical use:** Google's PaLM, Gemini, and Mixtral use MoE. The benefits really emerge at very large scale (1T+ parameters); smaller models often don't see much gain. In interviews, MoE shows awareness of frontier scaling research — emphasize the tradeoff: compute savings versus training complexity and memory.

---

### Q14: What is vision transformer (ViT)? How are images treated as sequences?

**A:** A **Vision Transformer (ViT)** applies a standard transformer encoder directly to images by treating image patches as tokens.

**The pipeline:**

1. **Patch embedding** — divide the image (e.g., 224 × 224) into patches (e.g., 16 × 16, giving 196 patches).
2. **Linear projection** — flatten each patch and project to an embedding dimension d.
3. **Positional embeddings** — add learned positional embeddings (since the patches are spatially ordered).
4. **[CLS] token** — prepend a special classification token (like BERT).
5. **Transformer encoder** — pass through the standard encoder layers.
6. **Classification** — use the [CLS] representation, or pool over all patches for dense tasks.

**Benefits:**

- **No convolutions** — attention directly relates any two spatial regions.
- **Scalability** — transformers scale cleanly to billions of parameters.
- **Transfer learning** — pre-train on a huge image dataset (JFT-300M), fine-tune downstream.

**Challenges:**

- **Data hungry** — ImageNet alone is insufficient; ViT needs massive pretraining data because it lacks the convolutional inductive bias.
- **Compute cost** — attention is O(n²) where n = (image_size / patch_size)². At 224×224 with 16×16 patches, n = 196 — feasible. Larger images blow up quadratically.
- **No built-in locality** — CNNs have locality (convolutions) and translation equivariance (pooling) baked in. ViTs must learn these from data.

**Mitigations:**

- **Hybrid ViTs** — start with a small CNN stem, then transformers on top.
- **Strong data augmentation** — essential, since invariance must be learned.
- **Pretraining at scale** — large models trained on massive datasets transfer best.

**Adoption:** ViT-L and ViT-H are standard in modern vision. **Swin Transformer** uses hierarchical local windows for better efficiency at larger resolutions. In interviews, ViT showcases the universality of the transformer architecture across modalities. The key talking point is patch tokenization, plus why large data is needed (the model has to learn CNN-style locality without it being built in).

---

### Q15: What is tokenization and how do BPE, WordPiece, and SentencePiece differ?

**A:** **Tokenization** converts text into the atomic units a transformer processes. The choice trades off vocabulary size against sequence length:

- **Character-level** — tiny vocabulary, very long sequences. Rarely used.
- **Word-level** — natural units but huge vocabulary and lots of out-of-vocabulary problems.
- **Subword tokenization** — the practical sweet spot.

**Three popular subword schemes:**

- **BPE (Byte Pair Encoding)** — start from characters, then iteratively merge the most *frequent* adjacent pair until you hit a target vocab size:

  ```
  "t" + "h" → "th"
  "th" + "e" → "the"
  ```

  Greedy and deterministic once merges are fixed. Used in GPT.

- **WordPiece** — like BPE but merges the pair that maximizes language-model *likelihood*, not frequency. A more principled criterion. Used in BERT.

- **SentencePiece** — language-agnostic. Treats the input as a raw byte sequence and learns merges without pre-tokenizing on whitespace. Handy for languages without spaces (Chinese, Japanese) or multilingual models. Used in XLM, mT5, Llama.

**Vocabulary size:** typically 30K–50K tokens (BERT) up to 100K+ for large multilingual models.

**Sequence length tradeoff:** a larger vocabulary gives shorter sequences (fewer subwords per word) but more parameters in the embedding and output layers.

**Why it matters in practice:**

- *"unhappy"* → ["un", "happy"] under WordPiece — semantically meaningful subwords.
- *"unhappy"* → ["u", "n", "h", "a", "p", "p", "y"] character-level — much less useful.

Tokenization is often overlooked but affects everything downstream. In interviews, mentioning the vocabulary-vs-sequence-length tradeoff and the difference between frequency-based BPE and likelihood-based WordPiece shows depth.

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

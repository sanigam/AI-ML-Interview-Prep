# Multiple Choice Questions: Transformer Architecture

📺 **Video Lecture:** https://youtu.be/yPTi4Ot5qoM


Test your understanding of transformer architecture concepts for AI/ML interviews.

---

**Q1. In scaled dot-product attention, the scaling factor 1/√d is applied because:**

A) It reduces the number of parameters
B) It prevents large dot products from pushing softmax into saturated regions with vanishing gradients
C) It converts attention scores to probabilities
D) It is required for multi-head attention only

---

**Q2. In the original transformer, positional encoding uses sinusoidal functions because:**

A) Sine waves are computationally cheaper than learned embeddings
B) They provide unique position-dependent signals and encode relative positions through known geometric relationships
C) They eliminate the need for layer normalization
D) They only work with encoder-only models

---

**Q3. Multi-head attention with h heads of dimension d/h each, compared to single-head attention of dimension d:**

A) Uses significantly more parameters
B) Uses roughly the same total parameters but captures diverse attention patterns
C) Is always slower than single-head attention
D) Eliminates the need for the feed-forward layer

---

**Q4. In the transformer decoder, causal masking ensures that:**

A) All positions can attend to all other positions
B) Position t can only attend to positions ≤ t, preventing access to future tokens
C) The decoder ignores the encoder outputs
D) Attention weights are always uniform

---

**Q5. Cross-attention in the transformer decoder uses:**

A) Queries, keys, and values all from the decoder
B) Queries from the decoder and keys/values from the encoder output
C) Queries from the encoder and keys/values from the decoder
D) Only values from the encoder, no queries or keys

---

**Q6. The feed-forward network (FFN) in each transformer layer:**

A) Processes the entire sequence jointly
B) Applies the same two-layer network independently to each position
C) Replaces the attention mechanism
D) Has no learnable parameters

---

**Q7. Layer normalization is preferred over batch normalization in transformers because:**

A) Layer norm requires larger batch sizes
B) Layer norm normalizes per sample across features, independent of batch size, and is deterministic
C) Layer norm only works during inference
D) Batch norm is more computationally expensive

---

**Q8. BERT is an example of a(n):**

A) Decoder-only transformer
B) Encoder-only transformer
C) Encoder-decoder transformer
D) Recurrent neural network

---

**Q9. GPT uses causal (left-to-right) language modeling, which means it:**

A) Can see bidirectional context during pre-training
B) Predicts the next token conditioned only on previous tokens
C) Requires labeled data for pre-training
D) Cannot generate text autoregressively

---

**Q10. Residual connections in transformers (x + Sublayer(x)) are crucial because:**

A) They reduce the number of layers needed
B) They provide direct gradient paths through the "+1" identity term, enabling training of very deep networks
C) They replace the need for attention
D) They only work with pre-LN (pre-layer normalization) configurations

---

**Q11. The masked language modeling (MLM) objective used in BERT:**

A) Predicts the next token from all previous tokens
B) Randomly masks 15% of tokens and predicts them using bidirectional context
C) Classifies entire documents into categories
D) Generates text from left to right

---

**Q12. T5's text-to-text framework unifies NLP tasks by:**

A) Using a separate model for each task
B) Treating all tasks as text input → text output with task-specific prefixes
C) Eliminating the need for pre-training
D) Using only classification heads

---

**Q13. The computational complexity of self-attention with respect to sequence length n is:**

A) O(n)
B) O(n log n)
C) O(n²)
D) O(n³)

---

**Q14. In-context learning in large language models refers to:**

A) Updating model weights during inference
B) Learning to solve tasks from examples provided in the prompt without any parameter updates
C) Training on multiple tasks simultaneously
D) Using gradient descent at test time

---

**Q15. Pre-LN (pre-layer normalization) transformers apply LayerNorm:**

A) After the attention and FFN sublayers
B) Before the attention and FFN sublayers, inside the residual path
C) Only during inference
D) Only to the first and last layers

---

## Answer Key

**Q1. Answer: B**
Without scaling, dot products grow with dimension d, pushing softmax into regions where output is nearly one-hot and gradients are vanishingly small. Dividing by √d keeps dot products at unit variance, ensuring meaningful gradients.

**Q2. Answer: B**
Sinusoidal positional encodings provide unique signals for each position and have the property that PE(pos+k) can be expressed as a linear function of PE(pos), encoding relative positions. They also generalize to sequences longer than those seen during training.

**Q3. Answer: B**
With h heads of dimension d/h, total parameters equal d² (same as single head of dimension d). The benefit is architectural: different heads learn different relationship types (syntactic, semantic, positional).

**Q4. Answer: B**
Causal masking sets attention scores for future positions to −∞ before softmax, ensuring they receive zero attention weight. This preserves the autoregressive property needed for sequential text generation.

**Q5. Answer: B**
In cross-attention, queries come from the decoder (what information the decoder needs), while keys and values come from the encoder (what information is available from the input sequence).

**Q6. Answer: B**
The FFN applies the same transformation (typically d → 4d → d with ReLU/GELU) to each position independently. Attention handles cross-position interaction; FFN adds per-position nonlinear capacity.

**Q7. Answer: B**
Layer norm computes statistics across features within each sample, making it independent of batch size and deterministic at both train and test time. This is essential for variable-length sequences in transformers.

**Q8. Answer: B**
BERT uses only the transformer encoder with bidirectional self-attention. It cannot generate text autoregressively but excels at understanding tasks (classification, NER, QA extraction).

**Q9. Answer: B**
GPT's causal language modeling predicts each token from only the preceding tokens (left-to-right). This autoregressive property enables text generation token by token during inference.

**Q10. Answer: B**
The gradient of the residual path includes a "+1" identity term: ∂(x + F(x))/∂x = 1 + ∂F/∂x. This ensures gradients flow even when ∂F/∂x is small, enabling training of 12–96+ layer networks.

**Q11. Answer: B**
BERT's MLM randomly masks tokens and predicts them from surrounding bidirectional context, forcing the model to learn rich contextual representations. The 15% masking rate includes 80% [MASK], 10% random, 10% unchanged.

**Q12. Answer: B**
T5 uses an encoder-decoder architecture where every task (classification, translation, summarization) is formatted as text-to-text. Task prefixes like "translate English to French:" specify the task.

**Q13. Answer: C**
Self-attention computes pairwise scores between all n positions, requiring O(n²) operations. This quadratic cost is the main bottleneck for long sequences and motivates efficient attention variants.

**Q14. Answer: B**
In-context learning provides task demonstrations in the prompt. The model uses attention to pattern-match from examples, solving new instances without any gradient updates or fine-tuning.

**Q15. Answer: B**
Pre-LN applies LayerNorm before each sublayer (attention or FFN), inside the residual connection. This provides more stable training for very deep transformers compared to post-LN (original transformer).

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

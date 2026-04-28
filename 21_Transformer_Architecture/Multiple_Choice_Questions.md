# Multiple Choice Questions: Transformer Architecture

📺 **Video Lecture:** https://youtu.be/yPTi4Ot5qoM


Test your understanding of transformer architecture concepts for AI/ML interviews.

---

**Q1. In scaled dot-product attention, the scaling factor 1/√d is applied because:**

A) It reduces the number of parameters  
B) It converts attention scores to probabilities  
C) It prevents large dot products from pushing softmax into saturated regions with vanishing gradients  
D) It is required for multi-head attention only

---

**Q2. In the original transformer, positional encoding uses sinusoidal functions because:**

A) They only work with encoder-only models  
B) Sine waves are computationally cheaper than learned embeddings  
C) They provide unique position-dependent signals and encode relative positions through known geometric relationships  
D) They eliminate the need for layer normalization

---

**Q3. Multi-head attention with h heads of dimension d/h each, compared to single-head attention of dimension d:**

A) Uses roughly the same total parameters but captures diverse attention patterns  
B) Uses significantly more parameters  
C) Is always slower than single-head attention  
D) Eliminates the need for the feed-forward layer

---

**Q4. In the transformer decoder, causal masking ensures that:**

A) Position t can only attend to positions ≤ t, preventing access to future tokens  
B) Attention weights are always uniform  
C) The decoder ignores the encoder outputs  
D) All positions can attend to all other positions

---

**Q5. Cross-attention in the transformer decoder uses:**

A) Only values from the encoder, no queries or keys  
B) Queries from the decoder and keys/values from the encoder output  
C) Queries, keys, and values all from the decoder  
D) Queries from the encoder and keys/values from the decoder

---

**Q6. The feed-forward network (FFN) in each transformer layer:**

A) Applies the same two-layer network independently to each position  
B) Replaces the attention mechanism  
C) Has no learnable parameters  
D) Processes the entire sequence jointly

---

**Q7. Layer normalization is preferred over batch normalization in transformers because:**

A) Batch norm is more computationally expensive  
B) Layer norm normalizes per sample across features, independent of batch size, and is deterministic  
C) Layer norm requires larger batch sizes  
D) Layer norm only works during inference

---

**Q8. BERT is an example of a(n):**

A) Encoder-only transformer  
B) Encoder-decoder transformer  
C) Recurrent neural network  
D) Decoder-only transformer

---

**Q9. GPT uses causal (left-to-right) language modeling, which means it:**

A) Can see bidirectional context during pre-training  
B) Cannot generate text autoregressively  
C) Requires labeled data for pre-training  
D) Predicts the next token conditioned only on previous tokens

---

**Q10. Residual connections in transformers (x + Sublayer(x)) are crucial because:**

A) They reduce the number of layers needed  
B) They provide direct gradient paths through the "+1" identity term, enabling training of very deep networks  
C) They replace the need for attention  
D) They only work with pre-LN (pre-layer normalization) configurations

---

**Q11. The masked language modeling (MLM) objective used in BERT:**

A) Randomly masks 15% of tokens and predicts them using bidirectional context  
B) Predicts the next token from all previous tokens  
C) Classifies entire documents into categories  
D) Generates text from left to right

---

**Q12. T5's text-to-text framework unifies NLP tasks by:**

A) Treating all tasks as text input → text output with task-specific prefixes  
B) Eliminating the need for pre-training  
C) Using only classification heads  
D) Using a separate model for each task

---

**Q13. The computational complexity of self-attention with respect to sequence length n is:**

A) O(n)  
B) O(n log n)  
C) O(n³)  
D) O(n²)

---

**Q14. In-context learning in large language models refers to:**

A) Learning to solve tasks from examples provided in the prompt without any parameter updates  
B) Training on multiple tasks simultaneously  
C) Using gradient descent at test time  
D) Updating model weights during inference

---

**Q15. Pre-LN (pre-layer normalization) transformers apply LayerNorm:**

A) Only to the first and last layers  
B) Only during inference  
C) Before the attention and FFN sublayers, inside the residual path  
D) After the attention and FFN sublayers

---

## Answer Key

**Q1. Answer: C**
Without scaling, dot products grow with dimension d, pushing softmax into regions where output is nearly one-hot and gradients are vanishingly small. Dividing by √d keeps dot products at unit variance, ensuring meaningful gradients.

**Q2. Answer: C**
Sinusoidal positional encodings provide unique signals for each position and have the property that PE(pos+k) can be expressed as a linear function of PE(pos), encoding relative positions. They also generalize to sequences longer than those seen during training.

**Q3. Answer: A**
With h heads of dimension d/h, total parameters equal d² (same as single head of dimension d). The benefit is architectural: different heads learn different relationship types (syntactic, semantic, positional).

**Q4. Answer: A**
Causal masking sets attention scores for future positions to −∞ before softmax, ensuring they receive zero attention weight. This preserves the autoregressive property needed for sequential text generation.

**Q5. Answer: B**
In cross-attention, queries come from the decoder (what information the decoder needs), while keys and values come from the encoder (what information is available from the input sequence).

**Q6. Answer: A**
The FFN applies the same transformation (typically d → 4d → d with ReLU/GELU) to each position independently. Attention handles cross-position interaction; FFN adds per-position nonlinear capacity.

**Q7. Answer: B**
Layer norm computes statistics across features within each sample, making it independent of batch size and deterministic at both train and test time. This is essential for variable-length sequences in transformers.

**Q8. Answer: A**
BERT uses only the transformer encoder with bidirectional self-attention. It cannot generate text autoregressively but excels at understanding tasks (classification, NER, QA extraction).

**Q9. Answer: D**
GPT's causal language modeling predicts each token from only the preceding tokens (left-to-right). This autoregressive property enables text generation token by token during inference.

**Q10. Answer: B**
The gradient of the residual path includes a "+1" identity term: ∂(x + F(x))/∂x = 1 + ∂F/∂x. This ensures gradients flow even when ∂F/∂x is small, enabling training of 12–96+ layer networks.

**Q11. Answer: A**
BERT's MLM randomly masks tokens and predicts them from surrounding bidirectional context, forcing the model to learn rich contextual representations. The 15% masking rate includes 80% [MASK], 10% random, 10% unchanged.

**Q12. Answer: A**
T5 uses an encoder-decoder architecture where every task (classification, translation, summarization) is formatted as text-to-text. Task prefixes like "translate English to French:" specify the task.

**Q13. Answer: D**
Self-attention computes pairwise scores between all n positions, requiring O(n²) operations. This quadratic cost is the main bottleneck for long sequences and motivates efficient attention variants.

**Q14. Answer: A**
In-context learning provides task demonstrations in the prompt. The model uses attention to pattern-match from examples, solving new instances without any gradient updates or fine-tuning.

**Q15. Answer: C**
Pre-LN applies LayerNorm before each sublayer (attention or FFN), inside the residual connection. This provides more stable training for very deep transformers compared to post-LN (original transformer).

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# RNNs and Sequence Models

📺 **Video Lecture:** https://youtu.be/G3vQTk-kq9g


## Interview Anchor
- **Vanilla RNN:** Recurrent connections process sequences; simple but suffers vanishing gradients due to BPTT
- **LSTM & GRU:** Gating mechanisms preserve long-term dependencies; LSTM with 4 gates (forget, input, output, cell), GRU with 2 gates (reset, update)
- **Sequence-to-Sequence & Attention:** Encoder-decoder architectures with attention mechanism enable translation, summarization; attention weights interpretable

## Key Concepts Overview
Recurrent neural networks extend feedforward networks to sequential data by feeding hidden states back as inputs, creating temporal dependencies. Vanilla RNNs learn patterns across timesteps but suffer severe gradient pathologies (vanishing/exploding gradients via backpropagation through time, BPTT). LSTMs and GRUs solve this via gating mechanisms, enabling learning long-range dependencies (100+ timesteps). Sequence-to-sequence models with attention revolutionized NLP, enabling machine translation and summarization without fixed-length intermediate bottlenecks. Understanding gating intuition, attention mechanisms, and how to debug RNN training is critical. This section covers recurrent architectures, their pathologies and solutions, sequence-to-sequence frameworks, and practical training techniques.

---

### Q1: Explain the vanilla RNN architecture. How does it process sequences?

**A:** A vanilla RNN maintains hidden state h_t updated via: h_t = tanh(W_h·h_{t-1} + W_x·x_t + b), output y_t = W_y·h_t + b_y. At each timestep t, the network reads input x_t, combines it with previous hidden state h_{t-1} via weights W_h and W_x, applies nonlinearity (tanh), producing new h_t. The hidden state is the "memory" of the sequence. Processing sequence [x_1, x_2, ..., x_T]: (1) Initialize h_0 (usually zeros). (2) For each t, compute h_t, y_t. (3) Loss = Σ loss(y_t, target_t). Advantages: (1) Recurrent structure naturally handles variable-length sequences. (2) Single set of weights shared across timesteps (parameter efficiency). (3) Can theoretically learn long-range dependencies (hidden state is a bottleneck). Disadvantages: (1) Vanishing/exploding gradients (below). (2) Bottleneck: hidden state must compress all past information. (3) Sequential computation (can't parallelize). In interviews, the vanilla RNN insight is elegant but naive—real applications need LSTMs/GRUs. Explaining that "hidden state is memory" and "shared weights across time" shows understanding; mentioning gradient pathologies shows you know limitations.

---

### Q2: What is BPTT (Backpropagation Through Time)? How does it compute gradients for RNNs?

**A:** BPTT is gradient computation for RNNs via the chain rule through timesteps. Forward pass: compute h_t and y_t for all t. Backward pass: compute gradients starting from loss at final timestep, propagate backward in time. For loss L = Σ_t loss(y_t, target_t) and y_t = W_y·h_t, gradient w.r.t. W_y: ∂L/∂W_y = Σ_t ∂loss(y_t)/∂W_y. For hidden state: ∂L/∂h_t = ∂L/∂y_t × ∂y_t/∂h_t + ∂L/∂h_{t+1} × ∂h_{t+1}/∂h_t. The gradient at h_t depends on both current loss (first term) and gradient from next timestep (second term, recurrent). Computing ∂L/∂W_h requires chain rule across all timesteps: ∂L/∂W_h = Σ_t (∂L/∂h_t × ∂h_t/∂W_h). The gradient ∂h_t/∂W_h involves h_{t-1}, which depends on h_{t-2}, etc., creating long chains. Pathology: if ∂h_t/∂h_{t-1} < 1 (typical with tanh, which has derivative ≤ 0.25), the chain product shrinks exponentially. With 100 timesteps, gradient ≈ (0.25)^100 ≈ 0, killing early timestep learning. This is vanishing gradient in RNNs; it's worse than feedforward networks because recurrent connections create extremely long dependency chains. BPTT is expensive O(T·parameters) where T is sequence length. Truncated BPTT (TBPTT) limits backprop depth to recent timesteps (k steps back), reducing compute. In interviews, explain that BPTT is just the chain rule applied through time; vanishing gradients are the key pathology necessitating LSTMs.

---

### Q3: Explain the LSTM architecture. What are the four gates?

**A:** LSTM (Long Short-Term Memory) introduces a cell state c_t (long-term memory) and three gates controlling its flow: forget gate (f_t), input gate (i_t), output gate (o_t). Computation: f_t = σ(W_f·[h_{t-1}, x_t] + b_f) (forget gate: 0 to 1, how much to forget), i_t = σ(W_i·[h_{t-1}, x_t] + b_i) (input gate: how much new info), c_tilde = tanh(W_c·[h_{t-1}, x_t] + b_c) (candidate cell state), c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde (update cell), o_t = σ(W_o·[h_{t-1}, x_t] + b_o) (output gate), h_t = o_t ⊙ tanh(c_t). Four gates learned via backprop; they control information flow. Benefits: (1) Cell state c_t has additive updates (c_t = f_t⊙c_{t-1} + i_t⊙c_tilde), enabling gradient flow unobstructed. Gradient ∂L/∂c_t = ∂L/∂c_{t+1} + other contributions; the direct path prevents exponential decay. (2) Forget gate selectively resets cell (f_t close to 0 forgets, close to 1 preserves). (3) Input gate decides if new info is relevant. (4) Output gate controls hidden state. LSTM learns when to remember/forget long-range dependencies. Intuition: gates enable selective information flow; cell state is a highway for gradients. In interviews, explain that LSTM is not magic but elegantly solves vanishing gradients via additive cell updates and multiplicative gating. Sketching gate operations or mentioning "cell state as long-term memory" shows understanding.

---

### Q4: What is a GRU (Gated Recurrent Unit)? How does it differ from LSTM?

**A:** GRU is a simpler LSTM variant with two gates (reset, update) instead of three. Computation: r_t = σ(W_r·[h_{t-1}, x_t]) (reset gate), z_t = σ(W_z·[h_{t-1}, x_t]) (update gate), h_tilde = tanh(W·[r_t ⊙ h_{t-1}, x_t]) (candidate), h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde. Reset gate r_t determines how much past hidden state influences candidate h_tilde; update gate z_t controls how much to update (1-z_t) × past + z_t × new. No separate cell state; hidden state is the only state. Advantages over LSTM: (1) Fewer parameters (2K vs. 4K gates, roughly half). (2) Simpler, easier to understand. (3) Trains faster (fewer gates). Disadvantages: (1) Potentially less expressive (no separate long-term memory). (2) May underfit compared to LSTM on complex tasks. Empirically: GRU and LSTM often perform comparably; choice is often task-dependent and empirical. GRU is preferred when (1) data is limited (fewer parameters help). (2) Computational efficiency matters (mobile/embedded). (3) Sequence length is moderate. LSTM is preferred for (1) very long sequences (separate cell state clearer). (2) Complex temporal patterns. (3) When parameter budget allows. In interviews, GRU vs. LSTM is often an empirical choice; explaining the architectural differences and trade-offs (parameters vs. expressiveness) is more important than declaring one superior.

---

### Q5: What are bidirectional RNNs? When are they useful?

**A:** Bidirectional RNNs process sequences in both directions: forward (left-to-right) and backward (right-to-left), concatenating outputs. Forward pass: h_f_t = RNN(x_1...x_t). Backward pass: h_b_t = RNN(x_T...x_t). Output: y_t = [h_f_t, h_b_t] (concatenate). Benefits: (1) Access future context: h_b_t has seen x_{t+1}...x_T, providing rightward context. (2) Better representations: future info improves predictions (e.g., in NLP, word sense depends on context before and after). (3) Symmetric view of sequences. Limitations: (1) Can't use for online prediction (need full sequence). (2) Not suitable for real-time streaming (need to buffer entire sequence). Use cases: (1) Classification (tag sequence, classify overall sentiment): label is for entire sequence, so bidirectionality is fine. (2) Encoding in sequence-to-sequence: encoder can see full source sequence, doesn't need online guarantees. (3) Named entity recognition (NER): word's entity type depends on surrounding words. Non-use: (1) Machine translation decoding (need to generate left-to-right). (2) Online/streaming inference (don't have future data). (3) Time series forecasting (future is unknown). In interviews, mention bidirectionality as standard for encoding; it improves representations when full sequence is available. Contrast with unidirectional decoding (must generate left-to-right).

---

### Q6: Explain sequence-to-sequence (seq2seq) models and encoder-decoder architecture.

**A:** Seq2seq models map variable-length input sequences to variable-length output sequences. Examples: machine translation (English → French), summarization (article → summary), question answering (question → answer). Architecture: encoder-decoder. Encoder (RNN/LSTM): processes input [x_1, ..., x_T], produces hidden states, final hidden state h_T is a fixed-size context vector. Decoder (RNN/LSTM): initialized with h_T, generates output [y_1, ..., y_S] one token at a time. At step s, decoder reads previous output y_{s-1} and hidden state h_s, predicts y_s. Loss: cross-entropy on predicted tokens. Inference: at test time, use predicted y_{s-1} as next input (teacher forcing replaces this with ground truth at train time for stability). Limitation: h_T is a bottleneck—must compress all input information into fixed-size vector. With long sequences, information loss occurs; decoder struggles to access early input. Solution: attention mechanism (below). Architecture: (1) Encoder RNN: x_1...x_T → h_1...h_T (all hidden states kept). (2) Decoder RNN at step s: attends over h_1...h_T, computes context vector as weighted sum of h_t (weights via attention). (3) Decoder predicts y_s using context + decoder hidden state. In interviews, explain the bottleneck motivation for attention; it drives understanding of why attention was necessary beyond just "it works."

---

### Q7: What is the attention mechanism? How does it improve seq2seq?

**A:** Attention allows decoder to focus on relevant input timesteps instead of compressing all info into a single context vector. Mechanism: at decoder step s, compute attention weights over encoder hidden states. Scaled dot-product attention: score(s,t) = (decoder_state_s · encoder_state_t) / √d (dot product normalized by dimension d). Attention weight α_{s,t} = softmax(score(s,t)) over all t. Context vector c_s = Σ_t α_{s,t}·h_t (weighted sum of encoder states). Decoder produces output using context + decoder state. Benefits: (1) No bottleneck: decoder accesses all encoder states, not just final h_T. (2) Interpretability: attention weights show which input words the decoder used (useful for understanding errors). (3) Improved performance: especially for long sequences. Example: translating long English sentence to French, decoder can attend to relevant English words (subject for subject, object for object). Intuition: score high for relevant (similar) encoder/decoder states, low for irrelevant. Softmax ensures weights sum to 1, interpretable as probability of "using" that input timestep. Variants: (1) Additive attention (Bahdanau): score = v^T·tanh(W_q·q + W_k·k) (slightly different computation, marginally slower). (2) Multiplicative attention (Luong): score = q·k^T (simple, fast, same in transformer). In interviews, attention solves the information bottleneck elegantly; explaining how attention weights let decoder "look back" at inputs it wants to focus on shows understanding of why attention was revolutionary.

---

### Q8: What is teacher forcing? When is it used and what problems can it cause?

**A:** Teacher forcing trains decoders by providing ground truth previous outputs during training. At training step s, decoder input is y_{s-1}^* (ground truth), not ŷ_{s-1} (predicted). Training loss L = Σ_s loss(model(y_s^*), y_s^*) (decoder sees ground truth). At inference, ground truth is unavailable; decoder uses its own predictions ŷ_s. Benefits: (1) Faster convergence: feeding ground truth accelerates learning. (2) Stability: avoiding error accumulation during training. Drawbacks: (1) Distribution mismatch (exposure bias): at train time decoder sees ground truth; at test time it sees errors. If decoder hasn't learned to recover from prediction errors, accuracy drops. (2) Error accumulation: test errors propagate through sequence. Example: machine translation, if decoder mispredicts word 1, that error influences words 2, 3, ..., creating cascading failures. Solutions: (1) Scheduled sampling: gradually reduce teacher forcing probability (start at 1, decay to 0) over training, mixing ground truth and predictions. (2) No teacher forcing: always use predicted outputs (slower convergence, but no distribution mismatch). (3) Cold fusion: separate model for dealing with errors. In practice: use teacher forcing for stability, but monitor test performance and beware of exposure bias. In interviews, mentioning exposure bias shows deep knowledge of seq2seq training. Explaining why teacher forcing causes train-test mismatch (ground truth ≠ predictions) signals understanding beyond formula memorization.

---

### Q9: Explain beam search decoding. Why is it used in seq2seq?

**A:** At inference, seq2seq decoder generates sequence greedily (choose highest-probability token at each step) or via beam search (keep multiple hypotheses). Greedy: at step s, select ŷ_s = argmax P(y|context). Fast, but often suboptimal—local choices prevent globally optimal sequences. Beam search: maintain k best partial sequences (k is beam width, typically 5-10). At each step, expand each hypothesis by all possible next tokens, keep top k by cumulative probability. Example: sentence "The cat sat on the mat." Greedy might predict "The dog sat on the floor" if "dog" is locally highest-prob. Beam search with k=5 might keep alternatives like "The cat sat on the rug," allowing better global sequence. Probability of sequence [y_1...y_S] = Π P(y_s|y_1...y_{s-1}). Taking log avoids underflow. Beam search maintains top k hypotheses by log-probability at each step. Inference complexity: O(k × vocab_size × sequence_length) vs. greedy O(vocab_size × sequence_length); ~5-10× slower but often 1-2% accuracy improvement. Length penalty: longer sequences have lower probability (more terms to multiply). Apply length normalization: log-prob / length. Stops beam search from preferring short sequences. Early stopping: candidate sequences ending with end-of-sequence token are removed from beam. In interviews, explain that beam search trades compute for accuracy; it's standard in production seq2seq (translation, dialogue). Mentioning length normalization shows understanding of why raw probabilities bias toward short sequences.

---

### Q10: How do you handle variable-length sequences in RNNs?

**A:** RNNs naturally handle variable-length sequences via dynamic stopping. At inference, generate until end-of-sequence token. At training, sequences have different lengths—padding strategies: (1) Pad to max length in batch: sequences shorter than max_len are padded with zeros. Model processes all timesteps but ignores padding via masking. (2) Pack sequences (PyTorch `pack_padded_sequence`): remove padding before RNN, RNN processes only real tokens, unpack after. More efficient (fewer operations on padding). (3) Variable batch sizes: group similar-length sequences together, process each group with appropriate max_len. Masking: zeros padding tokens, but model still processes them (wastes computation). Masking attention weights prevents attention to padding. Loss computation: only accumulate loss on real tokens (mask out padding). Without masking, padding tokens artificially increase loss. For encoder-decoder: encoder processes variable-length inputs; decoder generates variable-length outputs. Encoder handles this naturally. Decoder uses end-of-sequence token to stop. At inference, max output length is a hyperparameter (prevent infinite generation). In interviews, handling variable-length sequences is a practical detail. Mentioning padding, masking, and packing shows you've implemented RNNs. Explaining why masking matters (padding shouldn't count in loss) signals implementation depth.

---

### Q11: What are peephole connections in LSTMs?

**A:** Peephole connections allow LSTM gates to look at the cell state c_t, not just previous hidden state h_{t-1}. Standard LSTM: gates depend on [h_{t-1}, x_t]. With peephole: (1) Forget gate: f_t = σ(W_f·[h_{t-1}, x_t, c_{t-1}] + b_f) (can "see" previous cell). (2) Input gate: i_t = σ(W_i·[h_{t-1}, x_t, c_{t-1}] + b_i). (3) Output gate: o_t = σ(W_o·[h_{t-1}, x_t, c_t] + b_o) (see current cell, computed after update). Benefits: gates have more information (cell state) for decisions. Potential improvements: slightly better on some tasks (e.g., music modeling). Drawback: more parameters (3 extra input dimensions to gates), marginal improvements in practice, rarely used. Modern trend: Peephole connections are not standard; most papers don't use them. Gated Recurrent Units (GRU) don't have separate cell state, so peephole doesn't apply. In interviews, peephole connections are a detail—mentioning them shows knowledge but aren't critical. Unless asked specifically, focus on core LSTM gates (forget, input, output).

---

### Q12: How do you train deep RNNs? What are challenges?

**A:** Deep RNNs stack multiple layers: input → LSTM1 → LSTM2 → ... → LSTM_L → output. Output of layer l is input to layer l+1. Benefits: stacking increases capacity, learning hierarchical representations. Challenges: (1) Vanishing gradients: even LSTMs suffer in very deep stacking (gradients through layer-to-layer connections). (2) Training instability: more parameters, more ways to diverge. Solutions: (1) Residual connections (skip layer l to l+2): h_l = f(h_{l-1}) + h_{l-1}. Enables gradient flow. (2) Layer normalization: normalize across features within each timestep (not across batches like batch norm). Stabilizes training. (3) Careful initialization: He initialization for weights, zero initialization for biases of forget gates (gates start near identity). (4) Gradient clipping: essential for RNNs, clip by norm. (5) Learning rate scheduling: lower learning rates, warmup. Practical: 2-4 layers is typical; beyond that requires heavy regularization. Most seq2seq models use 2-4 layer encoders/decoders. Very deep RNNs (10+ layers) are rare; transformers are preferred for depth. In interviews, deep RNNs are less common than deep CNNs; mentioning residual connections and layer norm shows understanding of depth-training challenges.

---

### Q13: Explain multi-step ahead predictions in RNNs. How do recursive vs. direct methods differ?

**A:** Multi-step forecasting: predict y_{t+1}, y_{t+2}, ..., y_{t+h} (h steps ahead). Two approaches: (1) Recursive (iterated): train single-step RNN predicting y_{t+1} from history. At inference, use predictions recursively: y_1 = RNN(x_t), then y_2 = RNN(y_1), repeat h times. Simple training, error accumulates. (2) Direct (one-model-per-horizon): train separate RNN for each horizon h. RNN_h predicts y_{t+h} directly from history. More parameters, no error accumulation, slower. Sequence-to-sequence: decoder generates multiple steps h using attention over encoder. Teacher forcing provides ground truth at training, predicted outputs at test (exposure bias). Multi-task learning: train single RNN to predict all horizons simultaneously, using shared encoder and separate decoder heads. Often better than recursive alone. Example: stock forecasting h=30 days ahead. Recursive: train 1-day model, predict 30 days by iterating (compound error). Seq2seq: encoder sees 100 past days, decoder generates 30-day forecast (global context). In interviews, mention that recursive compounds error; seq2seq or multi-task are better for multi-step. If discussing time series, this is critical—single-step models evaluated on multi-step are deceptive.

---

### Q14: What is scheduled sampling and how does it address exposure bias?

**A:** Scheduled sampling gradually transitions from teacher forcing to scheduled sampling during training. Motivation: teacher forcing causes exposure bias (train on ground truth, test on predictions). Scheduled sampling: at training step s, use ground truth with probability p_s, predicted output with probability 1-p_s. p_s = 1 initially (full teacher forcing), decays to 0 (full auto-regressive). Schedule: p_s = 1 - i/total_steps (linear decay) or p_s = (1-1/total_steps)^i (exponential decay). Effect: early training uses teacher forcing for stability; late training uses predictions, reducing exposure bias. Trade-off: scheduled sampling is slower than pure teacher forcing but avoids distribution mismatch. Hyperparameter: decay schedule. Aggressive decay (fast) → early auto-regressive (training instability). Slow decay → late training still uses mostly teacher forcing (still biased). In practice: often not used; pure teacher forcing works well with beam search at inference (diversifies outputs, mitigates error accumulation). Scheduled sampling is more important for very long sequences. In interviews, mentioning exposure bias and scheduled sampling shows advanced knowledge; most practitioners use teacher forcing + beam search instead.

---

### Q15: What are applications of RNNs and what recent alternatives exist?

**A:** RNN applications: (1) Machine translation: seq2seq with attention (Transformer now dominant). (2) Text generation: language modeling (sequence prediction). (3) Sentiment analysis: classification from sequence. (4) Named entity recognition (NER): per-token tagging. (5) Speech recognition: acoustic → text sequences. (6) Time series forecasting: predict future values. (7) Video understanding: process frame sequences. Advantages: (1) Handles variable-length sequences naturally. (2) Interpretable attention weights. (3) Efficient with small data (compact). Disadvantages: (1) Sequential computation (slow on GPUs). (2) Difficult optimization (vanishing gradients, gradient clipping needed). (3) Limited parallelization (can't parallelize across timesteps). Modern alternatives: (1) Transformers (attention is all you need): self-attention replaces recurrence, highly parallelizable, scales to massive data. Now dominant in NLP. (2) Temporal CNNs (TCN): 1D convolutions with dilations to capture temporal patterns; parallelizable, but less interpretable attention. (3) Hybrid: Transformer encoder + RNN decoder (some seq2seq models). Current practice: Transformers are preferred for large data / high-compute regimes. RNNs remain useful for small data, online inference (streaming), and domains where sequence-by-sequence attention is valuable. In interviews, acknowledge that Transformers are now dominant but RNNs are still relevant for understanding sequence models and for resource-constrained settings. Discussing trade-offs (Transformer parallelism vs. RNN interpretability) shows sophisticated thinking.

---

## Interview Cheatsheet

**Key Terms:**
- **Vanilla RNN:** Recurrent connections (h_t = tanh(W_h·h_{t-1} + W_x·x_t)); variable-length sequences, simple but vanishing gradients
- **BPTT:** Backpropagation Through Time; chain rule applied through timesteps; gradients vanish with long sequences
- **Vanishing Gradient in RNNs:** Gradients shrink through deep unrolled graphs (BPTT); learning long-range dependencies fails
- **LSTM:** 4 gates (forget, input, output, cell) + cell state; additive updates prevent vanishing gradients
- **GRU:** 2 gates (reset, update); simpler than LSTM, faster, fewer parameters
- **Bidirectional RNN:** Process sequence left-to-right and right-to-left; improves representations when full sequence available
- **Seq2seq:** Encoder-decoder architecture for variable-length-to-variable-length mapping (translation, summarization)
- **Attention Mechanism:** Decoder attends to encoder states; weighted sum via softmax scores; solves information bottleneck
- **Teacher Forcing:** Provide ground truth as input during training; faster convergence but exposure bias at test
- **Beam Search:** Keep k best hypotheses at each step; trades compute for accuracy vs. greedy decoding
- **Padding & Masking:** Handle variable-length sequences; mask padding in loss, attention
- **Peephole Connections:** Gates see cell state; marginal improvements, rarely used in practice
- **Scheduled Sampling:** Gradually transition from teacher forcing to predictions; addresses exposure bias
- **Temporal CNN:** 1D convolutions with dilations; parallelizable alternative to RNNs
- **Transformer:** Self-attention replaces recurrence; highly parallelizable, dominant in modern NLP
- **Gradient Clipping:** Cap gradient norm; necessary for RNN training, prevents exploding gradients

**Rapid-Fire Q&A:**
- **Q: Why RNNs for sequences?** **A:** Variable-length handling, temporal dependencies, weight sharing across time
- **Q: Vanishing gradient in RNNs vs. feedforward?** **A:** RNNs have deeper unrolled graphs through time; multiplication of derivatives compounds
- **Q: LSTM vs. GRU?** **A:** LSTM more expressive (4 gates, separate cell); GRU simpler (2 gates), fewer parameters
- **Q: Forget gate intuition?** **A:** Controls information flow in cell state; close to 1 preserves, close to 0 forgets
- **Q: Bidirectional when?** **A:** Classification, tagging (have full sequence); not for online prediction or generation
- **Q: Why attention?** **A:** Decoder attends to relevant inputs; no bottleneck (h_T), interpretable
- **Q: Teacher forcing problem?** **A:** Exposure bias—train on ground truth, test on predictions
- **Q: Greedy vs. beam search?** **A:** Greedy: fast, local optima; beam search: slower, globally better sequences
- **Q: Handling variable-length?** **A:** Padding + masking, or packing; mask loss for padding tokens
- **Q: RNN vs. Transformer?** **A:** RNN: interpretable, slow; Transformer: parallelizable, scales, now standard

---

## Interview Tips
- **Draw RNN unrolled diagrams:** Show forward pass across timesteps, backprop flow; helps explain vanishing gradients
- **Explain gate intuitions:** Forget gate "decides what to forget," input gate "decides what to add"—anthropomorphic language aids understanding
- **Mention BPTT explicitly:** Shows you understand the gradient computation mechanism
- **Discuss exposure bias in detail:** More relevant than most candidates; demonstrates seq2seq sophistication
- **Compare to Transformers thoughtfully:** Acknowledge Transformer dominance but explain RNN advantages (interpretability, small data, streaming)
- **Prepare a seq2seq example:** Machine translation is classic; walk through encoder-decoder-attention for a short sentence
- **Discuss practical challenges:** Gradient clipping, learning rate scheduling, layer norm—shows implementation experience
- **Mention modern trends:** Transformers are default; RNNs useful for specific constraints (latency, memory, interpretability)

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

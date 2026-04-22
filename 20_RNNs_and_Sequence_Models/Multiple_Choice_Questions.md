# Multiple Choice Questions: RNNs and Sequence Models

Test your understanding of recurrent neural networks and sequence modeling for AI/ML interviews.

---

**Q1. In a vanilla RNN, the hidden state hₜ is computed as:**

A) hₜ = σ(Wₓ·xₜ + b)
B) hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)
C) hₜ = hₜ₋₁ + xₜ
D) hₜ = softmax(Wₕ·hₜ₋₁)

---

**Q2. The vanishing gradient problem in RNNs is more severe than in feedforward networks because:**

A) RNNs use different activation functions
B) Gradients are multiplied through many timesteps during BPTT, creating extremely long dependency chains
C) RNNs have more parameters per layer
D) RNNs cannot use batch normalization

---

**Q3. The LSTM cell state cₜ helps gradient flow because updates are:**

A) Multiplicative only
B) Additive (cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ), preventing exponential gradient decay
C) Always equal to the hidden state
D) Computed without any gating mechanism

---

**Q4. Which LSTM gate controls how much of the previous cell state to retain?**

A) Input gate
B) Output gate
C) Forget gate
D) Reset gate

---

**Q5. GRU differs from LSTM primarily by:**

A) Having more gates and a separate cell state
B) Having fewer gates (2 vs. 3) and no separate cell state
C) Being unable to handle sequences
D) Requiring bidirectional processing

---

**Q6. Bidirectional RNNs are NOT suitable for:**

A) Sentiment classification of complete sentences
B) Named entity recognition on full documents
C) Real-time language generation (producing text token by token)
D) Encoding input sequences in a seq2seq model

---

**Q7. In a seq2seq encoder-decoder model without attention, the information bottleneck is:**

A) The size of the input vocabulary
B) The fixed-size context vector (final encoder hidden state) that must encode the entire input
C) The number of decoder layers
D) The loss function choice

---

**Q8. The attention mechanism in seq2seq computes context by:**

A) Averaging all encoder hidden states equally
B) Computing a weighted sum of encoder hidden states, with weights based on relevance to the current decoder state
C) Using only the last encoder hidden state
D) Randomly selecting encoder states

---

**Q9. Teacher forcing during training provides:**

A) Predicted outputs as decoder inputs
B) Ground truth previous tokens as decoder inputs for faster convergence
C) Random noise as decoder inputs
D) No inputs to the decoder

---

**Q10. Exposure bias refers to the discrepancy between:**

A) Training loss and test loss
B) Training (where decoder sees ground truth) and inference (where decoder sees its own predictions)
C) Encoder and decoder architectures
D) Supervised and unsupervised learning

---

**Q11. Beam search with beam width k=5 at each decoding step:**

A) Generates 5 complete sequences independently
B) Maintains the top 5 partial sequences by cumulative log-probability
C) Randomly samples 5 tokens
D) Uses 5 different models

---

**Q12. In sequence padding, masking is important because:**

A) It speeds up computation
B) It prevents padding tokens from contributing to loss computation and attention weights
C) It increases the sequence length
D) It replaces the need for embedding layers

---

**Q13. Gradient clipping in RNN training:**

A) Increases the learning rate automatically
B) Caps the gradient norm to a threshold to prevent exploding gradients
C) Removes negative gradients
D) Only applies to the output layer

---

**Q14. Scheduled sampling addresses exposure bias by:**

A) Always using teacher forcing
B) Gradually decreasing the probability of using ground truth inputs during training
C) Increasing the beam width over time
D) Using a fixed schedule of learning rates

---

**Q15. The main reason Transformers have largely replaced RNNs for NLP tasks is:**

A) Transformers use recurrent connections for better memory
B) Transformers enable parallelization across sequence positions and scale better with data and compute
C) Transformers have fewer parameters than RNNs
D) Transformers do not require any training

---

## Answer Key

**Q1. Answer: B**
The vanilla RNN combines the previous hidden state and current input through weight matrices, applies tanh nonlinearity, producing the new hidden state that serves as the sequence "memory."

**Q2. Answer: B**
BPTT unrolls the RNN across all timesteps. Gradients involve products of many weight matrix derivatives (one per timestep), causing exponential shrinkage with tanh derivatives ≤ 0.25.

**Q3. Answer: B**
The additive update cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ creates a direct gradient path through the cell state. Unlike multiplicative chains in vanilla RNNs, addition preserves gradients across many timesteps.

**Q4. Answer: C**
The forget gate fₜ outputs values between 0 and 1 for each cell state dimension. Values near 1 retain information; values near 0 discard it. This selective forgetting is key to LSTM's power.

**Q5. Answer: B**
GRU uses two gates (reset and update) and merges the cell state into the hidden state, resulting in fewer parameters and faster training while achieving comparable performance on many tasks.

**Q6. Answer: C**
Real-time generation requires producing tokens sequentially without access to future context. Bidirectional RNNs need the full sequence, making them suitable for encoding but not for autoregressive generation.

**Q7. Answer: B**
The final encoder hidden state must compress the entire input sequence into a single fixed-size vector. For long inputs, this bottleneck loses information, motivating the attention mechanism.

**Q8. Answer: B**
Attention computes relevance scores between the current decoder state and all encoder states, applies softmax to get weights, then produces a weighted sum as the context vector for that decoding step.

**Q9. Answer: B**
Teacher forcing feeds ground truth tokens (not model predictions) as decoder inputs during training, providing a stronger training signal and faster convergence at the cost of exposure bias.

**Q10. Answer: B**
During training, the decoder always receives correct inputs (teacher forcing). During inference, it receives its own (potentially erroneous) predictions, creating a distribution mismatch that can cause error accumulation.

**Q11. Answer: B**
Beam search maintains k best partial hypotheses at each step, expanding each by all vocabulary tokens and keeping the top k by cumulative log-probability. It balances exploration and computation.

**Q12. Answer: B**
Without masking, padding tokens would incorrectly contribute to the loss and attention computations, biasing the model. Masking ensures only real tokens affect learning and predictions.

**Q13. Answer: B**
When gradients exceed a threshold τ, they are scaled down proportionally: g ← g × min(1, τ/||g||). This prevents catastrophically large weight updates while preserving gradient direction.

**Q14. Answer: B**
Scheduled sampling starts with full teacher forcing and gradually increases the probability of using the model's own predictions as inputs, smoothly transitioning toward inference-like conditions.

**Q15. Answer: B**
Transformers process all positions in parallel via self-attention (unlike RNNs' sequential processing), enabling efficient GPU utilization and scaling to much larger models and datasets.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

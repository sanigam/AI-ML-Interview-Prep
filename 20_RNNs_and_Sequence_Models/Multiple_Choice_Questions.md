# Multiple Choice Questions: RNNs and Sequence Models

📺 **Video Lecture:** https://youtu.be/G3vQTk-kq9g


Test your understanding of recurrent neural networks and sequence modeling for AI/ML interviews.

---

**Q1. In a vanilla RNN, the hidden state hₜ is computed as:**

A) hₜ = softmax(Wₕ·hₜ₋₁)  
B) hₜ = hₜ₋₁ + xₜ  
C) hₜ = σ(Wₓ·xₜ + b)  
D) hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)

---

**Q2. The vanishing gradient problem in RNNs is more severe than in feedforward networks because:**

A) RNNs use different activation functions  
B) RNNs have more parameters per layer  
C) Gradients are multiplied through many timesteps during BPTT, creating extremely long dependency chains  
D) RNNs cannot use batch normalization

---

**Q3. The LSTM cell state cₜ helps gradient flow because updates are:**

A) Computed without any gating mechanism  
B) Multiplicative only  
C) Always equal to the hidden state  
D) Additive (cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ), preventing exponential gradient decay

---

**Q4. Which LSTM gate controls how much of the previous cell state to retain?**

A) Forget gate  
B) Reset gate  
C) Input gate  
D) Output gate

---

**Q5. GRU differs from LSTM primarily by:**

A) Requiring bidirectional processing  
B) Being unable to handle sequences  
C) Having more gates and a separate cell state  
D) Having fewer gates (2 vs. 3) and no separate cell state

---

**Q6. Bidirectional RNNs are NOT suitable for:**

A) Real-time language generation (producing text token by token)  
B) Sentiment classification of complete sentences  
C) Named entity recognition on full documents  
D) Encoding input sequences in a seq2seq model

---

**Q7. In a seq2seq encoder-decoder model without attention, the information bottleneck is:**

A) The fixed-size context vector (final encoder hidden state) that must encode the entire input  
B) The size of the input vocabulary  
C) The loss function choice  
D) The number of decoder layers

---

**Q8. The attention mechanism in seq2seq computes context by:**

A) Randomly selecting encoder states  
B) Averaging all encoder hidden states equally  
C) Using only the last encoder hidden state  
D) Computing a weighted sum of encoder hidden states, with weights based on relevance to the current decoder state

---

**Q9. Teacher forcing during training provides:**

A) Predicted outputs as decoder inputs  
B) Ground truth previous tokens as decoder inputs for faster convergence  
C) No inputs to the decoder  
D) Random noise as decoder inputs

---

**Q10. Exposure bias refers to the discrepancy between:**

A) Supervised and unsupervised learning  
B) Training (where decoder sees ground truth) and inference (where decoder sees its own predictions)  
C) Encoder and decoder architectures  
D) Training loss and test loss

---

**Q11. Beam search with beam width k=5 at each decoding step:**

A) Generates 5 complete sequences independently  
B) Uses 5 different models  
C) Maintains the top 5 partial sequences by cumulative log-probability  
D) Randomly samples 5 tokens

---

**Q12. In sequence padding, masking is important because:**

A) It increases the sequence length  
B) It prevents padding tokens from contributing to loss computation and attention weights  
C) It speeds up computation  
D) It replaces the need for embedding layers

---

**Q13. Gradient clipping in RNN training:**

A) Increases the learning rate automatically  
B) Caps the gradient norm to a threshold to prevent exploding gradients  
C) Only applies to the output layer  
D) Removes negative gradients

---

**Q14. Scheduled sampling addresses exposure bias by:**

A) Always using teacher forcing  
B) Using a fixed schedule of learning rates  
C) Increasing the beam width over time  
D) Gradually decreasing the probability of using ground truth inputs during training

---

**Q15. The main reason Transformers have largely replaced RNNs for NLP tasks is:**

A) Transformers do not require any training  
B) Transformers have fewer parameters than RNNs  
C) Transformers use recurrent connections for better memory  
D) Transformers enable parallelization across sequence positions and scale better with data and compute

---

## Answer Key

**Q1. Answer: D**
The vanilla RNN combines the previous hidden state and current input through weight matrices, applies tanh nonlinearity, producing the new hidden state that serves as the sequence "memory."

**Q2. Answer: C**
BPTT unrolls the RNN across all timesteps. Gradients involve products of many weight matrix derivatives (one per timestep), causing exponential shrinkage with tanh derivatives ≤ 0.25.

**Q3. Answer: D**
The additive update cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ creates a direct gradient path through the cell state. Unlike multiplicative chains in vanilla RNNs, addition preserves gradients across many timesteps.

**Q4. Answer: A**
The forget gate fₜ outputs values between 0 and 1 for each cell state dimension. Values near 1 retain information; values near 0 discard it. This selective forgetting is key to LSTM's power.

**Q5. Answer: D**
GRU uses two gates (reset and update) and merges the cell state into the hidden state, resulting in fewer parameters and faster training while achieving comparable performance on many tasks.

**Q6. Answer: A**
Real-time generation requires producing tokens sequentially without access to future context. Bidirectional RNNs need the full sequence, making them suitable for encoding but not for autoregressive generation.

**Q7. Answer: A**
The final encoder hidden state must compress the entire input sequence into a single fixed-size vector. For long inputs, this bottleneck loses information, motivating the attention mechanism.

**Q8. Answer: D**
Attention computes relevance scores between the current decoder state and all encoder states, applies softmax to get weights, then produces a weighted sum as the context vector for that decoding step.

**Q9. Answer: B**
Teacher forcing feeds ground truth tokens (not model predictions) as decoder inputs during training, providing a stronger training signal and faster convergence at the cost of exposure bias.

**Q10. Answer: B**
During training, the decoder always receives correct inputs (teacher forcing). During inference, it receives its own (potentially erroneous) predictions, creating a distribution mismatch that can cause error accumulation.

**Q11. Answer: C**
Beam search maintains k best partial hypotheses at each step, expanding each by all vocabulary tokens and keeping the top k by cumulative log-probability. It balances exploration and computation.

**Q12. Answer: B**
Without masking, padding tokens would incorrectly contribute to the loss and attention computations, biasing the model. Masking ensures only real tokens affect learning and predictions.

**Q13. Answer: B**
When gradients exceed a threshold τ, they are scaled down proportionally: g ← g × min(1, τ/||g||). This prevents catastrophically large weight updates while preserving gradient direction.

**Q14. Answer: D**
Scheduled sampling starts with full teacher forcing and gradually increases the probability of using the model's own predictions as inputs, smoothly transitioning toward inference-like conditions.

**Q15. Answer: D**
Transformers process all positions in parallel via self-attention (unlike RNNs' sequential processing), enabling efficient GPU utilization and scaling to much larger models and datasets.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

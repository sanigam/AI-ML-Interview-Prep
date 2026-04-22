# Multiple Choice Questions: Generative AI and Large Language Models

Test your understanding of generative AI and LLM concepts for AI/ML interviews.

---

**Q1. A large language model (LLM) is fundamentally trained to:**

A) Classify text into predefined categories
B) Predict the next token given previous tokens (autoregressive next-token prediction)
C) Translate between exactly two languages
D) Generate images from text descriptions

---

**Q2. Scaling laws for LLMs show that:**

A) Larger models always overfit
B) Loss decreases as a power law with both model size and training data, and neither saturates independently
C) Only model size matters, not data
D) Performance plateaus after 1 billion parameters

---

**Q3. Emergent abilities in LLMs refer to:**

A) Abilities that are explicitly programmed into the model
B) Capabilities that appear only at large scale and are absent or near-random in smaller models
C) Skills that all neural networks possess
D) Abilities that decrease with model size

---

**Q4. In-context learning differs from fine-tuning because:**

A) It requires updating model weights
B) It provides task examples in the prompt and the model learns without any parameter updates
C) It needs millions of labeled examples
D) It only works with encoder models

---

**Q5. Instruction tuning transforms a base LLM into a more usable assistant by:**

A) Removing the attention mechanism
B) Fine-tuning on diverse tasks with explicit instructions, teaching the model to follow instructions reliably
C) Reducing the model size
D) Training only on a single task

---

**Q6. RLHF (Reinforcement Learning from Human Feedback) involves three stages:**

A) Tokenization → Embedding → Decoding
B) Collect human preferences → Train reward model → RL fine-tune LLM using reward model
C) Pre-train → Prune → Quantize
D) Data collection → Annotation → Evaluation

---

**Q7. DPO (Direct Preference Optimization) improves upon RLHF by:**

A) Using more human annotators
B) Directly optimizing on preference pairs without needing a separate reward model
C) Requiring more compute than RLHF
D) Eliminating the need for human feedback entirely

---

**Q8. LLM hallucination refers to:**

A) The model refusing to answer
B) The model generating plausible-sounding but factually incorrect or fabricated information
C) The model producing empty outputs
D) The model taking too long to respond

---

**Q9. Retrieval-Augmented Generation (RAG) reduces hallucinations by:**

A) Making the model smaller
B) Augmenting the LLM with retrieved relevant documents as context, grounding answers in actual sources
C) Removing the attention mechanism
D) Training exclusively on verified facts

---

**Q10. The temperature parameter during LLM sampling controls:**

A) The speed of inference
B) The randomness of generation — lower temperature produces more deterministic outputs, higher produces more diverse outputs
C) The model size
D) The number of attention heads

---

**Q11. Top-k sampling limits generation diversity by:**

A) Using only the first k tokens in the vocabulary
B) Sampling only from the k highest-probability tokens at each step
C) Generating exactly k tokens
D) Using k different models

---

**Q12. Chinchilla scaling laws suggest that for optimal compute allocation:**

A) All compute should go to model parameters
B) Model size and training data should be scaled roughly equally (train N parameter model on ~20N tokens)
C) Training data doesn't matter
D) Smaller models are always better

---

**Q13. Constitutional AI aligns LLMs by:**

A) Using only human feedback
B) Defining explicit principles (a constitution) and having the model self-critique and revise its outputs against them
C) Removing all safety constraints
D) Training on unlabeled data only

---

**Q14. The context window limitation of LLMs means:**

A) The model can only generate short responses
B) The model can only attend to a fixed number of tokens in the input, limiting how much information it can process at once
C) The model cannot handle any text input
D) The model requires a GPU with unlimited memory

---

**Q15. Top-p (nucleus) sampling selects tokens from the smallest set whose cumulative probability exceeds p. Compared to top-k:**

A) Top-p always selects fewer tokens
B) Top-p adapts the number of candidate tokens based on the probability distribution shape
C) Top-p ignores token probabilities
D) Top-p and top-k are identical

---

## Answer Key

**Q1. Answer: B**
LLMs are trained via self-supervised next-token prediction: given tokens x₁...xₙ, maximize P(xₙ₊₁|x₁:ₙ). This simple objective, applied at massive scale, produces models capable of diverse tasks.

**Q2. Answer: B**
Empirical scaling laws show Loss ∝ N⁻ᵅ and Loss ∝ D⁻ᵝ. Neither model size nor data alone saturates the other; both must be scaled together for optimal compute utilization.

**Q3. Answer: B**
Emergent abilities (few-shot learning, chain-of-thought reasoning, instruction following) appear suddenly above certain scale thresholds, not present in smaller models trained on the same data.

**Q4. Answer: B**
In-context learning places task demonstrations in the prompt. The model uses attention to pattern-match and solve new instances entirely during the forward pass, with no gradient updates.

**Q5. Answer: B**
Instruction tuning fine-tunes on thousands of tasks with diverse instruction formats, teaching the model the meta-skill of following instructions. This dramatically improves usability over raw base models.

**Q6. Answer: B**
RLHF's pipeline: (1) humans compare model outputs, (2) a reward model learns to predict preferences, (3) the LLM is optimized via RL (PPO) to maximize the reward while staying close to the base model.

**Q7. Answer: B**
DPO reformulates the RLHF objective to directly optimize on preference pairs using a simple loss function, eliminating the need to train a separate reward model while achieving comparable results.

**Q8. Answer: B**
Hallucination occurs because LLMs are probabilistic models that sample from learned distributions. They generate statistically plausible continuations even when factual knowledge is uncertain or absent.

**Q9. Answer: B**
RAG retrieves relevant documents from a knowledge base and includes them in the prompt context, allowing the LLM to generate answers grounded in actual sources rather than relying solely on memorized knowledge.

**Q10. Answer: B**
Temperature T scales logits before softmax: softmax(logits/T). T→0 gives argmax (deterministic), T→∞ gives uniform sampling (maximum randomness). T=1 uses the learned distribution directly.

**Q11. Answer: B**
Top-k restricts sampling to the k most probable tokens, zeroing out all others. This prevents sampling extremely unlikely tokens while maintaining diversity among the top candidates.

**Q12. Answer: B**
Chinchilla showed that many early LLMs were undertrained (too many parameters, too few tokens). Optimal allocation trains approximately 20 tokens per parameter for a given compute budget.

**Q13. Answer: B**
Constitutional AI uses explicit principles to guide model self-improvement. The model critiques its own outputs against the constitution and learns to produce revised, aligned responses.

**Q14. Answer: B**
Transformers have fixed context windows (e.g., 4K, 8K, 128K tokens). Information outside this window cannot be attended to, limiting the model's ability to process very long documents in a single pass.

**Q15. Answer: B**
Top-p dynamically selects more tokens when the distribution is flat (uncertain) and fewer when it's peaked (confident). Top-k always selects exactly k tokens regardless of distribution shape.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

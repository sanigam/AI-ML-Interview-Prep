# Multiple Choice Questions: Prompt Engineering and Fine-Tuning

Test your understanding of prompt engineering and fine-tuning for AI/ML interviews.

---

**Q1. The primary advantage of prompt engineering over fine-tuning is:**

A) Higher accuracy on all tasks
B) Speed and cost — no model training required, just crafting better input text
C) It works without any LLM
D) It permanently modifies model weights

---

**Q2. Few-shot prompting improves over zero-shot by:**

A) Training the model on new data
B) Providing task examples in the prompt that the model uses for in-context pattern matching
C) Reducing the context window
D) Removing all instructions

---

**Q3. Chain-of-thought (CoT) prompting improves reasoning by:**

A) Using shorter prompts
B) Asking the model to show intermediate reasoning steps before the final answer
C) Fine-tuning on reasoning datasets
D) Removing the model's attention mechanism

---

**Q4. Prompt injection attacks exploit the fact that:**

A) LLMs have unlimited context windows
B) LLMs cannot distinguish between system instructions and malicious user input embedded in the text
C) All LLMs are open source
D) Prompts are encrypted

---

**Q5. LoRA (Low-Rank Adaptation) fine-tuning works by:**

A) Updating all model parameters
B) Inserting small trainable low-rank matrices into attention layers while keeping the original weights frozen
C) Removing layers from the model
D) Training only the embedding layer

---

**Q6. Compared to full fine-tuning, LoRA uses approximately:**

A) The same number of trainable parameters
B) 10-100x fewer trainable parameters while achieving comparable accuracy
C) More parameters and more compute
D) No parameters at all

---

**Q7. QLoRA extends LoRA by:**

A) Using larger rank matrices
B) Quantizing the base model to 4-bit precision before applying LoRA, enabling fine-tuning of much larger models on limited hardware
C) Removing the LoRA adapters
D) Training in full precision only

---

**Q8. Self-consistency decoding improves chain-of-thought by:**

A) Using a single reasoning path
B) Generating multiple independent reasoning paths and selecting the most common answer via majority vote
C) Removing the reasoning steps
D) Using greedy decoding only

---

**Q9. Catastrophic forgetting during fine-tuning refers to:**

A) The model learning new tasks too slowly
B) The model losing previously learned knowledge when adapted to a new task
C) The model's weights becoming too large
D) The model generating longer outputs

---

**Q10. The optimal number of few-shot examples typically shows:**

A) Linear improvement with more examples
B) Diminishing returns — accuracy improves with 1-5 examples but plateaus beyond that
C) Degradation with any examples
D) Improvement only with 100+ examples

---

**Q11. Role-based prompting ("You are a medical expert...") helps because:**

A) It changes the model architecture
B) It activates relevant knowledge and response patterns learned during pre-training
C) It fine-tunes the model in real-time
D) It reduces the token count

---

**Q12. Parameter-efficient fine-tuning (PEFT) methods include:**

A) Only full fine-tuning
B) LoRA, adapters, prefix tuning, and prompt tuning — all updating far fewer parameters than the full model
C) Only changing the learning rate
D) Only training on more data

---

**Q13. When should you prefer fine-tuning over prompt engineering?**

A) When you need quick prototyping with no labeled data
B) When high accuracy is required on a specialized domain, sufficient labeled data is available, and compute resources permit
C) When the task is simple and well-defined
D) When you want to avoid any compute costs

---

**Q14. Tree-of-thought prompting extends chain-of-thought by:**

A) Using a single linear reasoning chain
B) Exploring multiple reasoning branches at each step and selecting the most promising paths
C) Eliminating reasoning entirely
D) Using only zero-shot prompts

---

**Q15. The learning rate for fine-tuning a pre-trained LLM is typically:**

A) The same as pre-training (e.g., 1e-3)
B) Much smaller (e.g., 1e-5 to 5e-5) to avoid destroying pre-trained representations
C) Set to zero
D) Increased throughout training

---

## Answer Key

**Q1. Answer: B**
Prompt engineering requires no training, just iterating on input text. It can be done in minutes/hours at minimal cost, while fine-tuning requires labeled data, compute resources, and days of work.

**Q2. Answer: B**
Few-shot examples in the prompt enable the model to recognize patterns through in-context learning. The model attends to the examples and applies the inferred pattern to the new input.

**Q3. Answer: B**
CoT prompting ("let's think step by step") encourages intermediate reasoning tokens, which help the model break down complex problems, catch errors, and arrive at more accurate final answers.

**Q4. Answer: B**
LLMs process all text uniformly — there's no built-in distinction between trusted system instructions and untrusted user input. Attackers can embed instructions that override the system prompt.

**Q5. Answer: B**
LoRA adds pairs of small matrices B (d×r) and A (r×d) where r << d. The effective weight becomes W + BA, but only B and A are trained, dramatically reducing trainable parameters.

**Q6. Answer: B**
For a 7B parameter model with rank r=8, LoRA might train only ~1M parameters instead of 7B, achieving similar accuracy on downstream tasks at a fraction of the compute cost.

**Q7. Answer: B**
QLoRA quantizes the base model to 4-bit (reducing memory 4-8x), then applies LoRA adapters in higher precision. This enables fine-tuning 65B+ parameter models on a single GPU.

**Q8. Answer: B**
Self-consistency generates k independent reasoning paths (via sampling), then selects the answer that appears most frequently. This ensemble-like approach reduces sensitivity to any single reasoning error.

**Q9. Answer: B**
When fine-tuned on a narrow task, the model's weights shift to optimize for that task, potentially degrading performance on other tasks it previously handled well. KL penalties or multi-task training help mitigate this.

**Q10. Answer: B**
Empirically, accuracy improves notably from 0 to 1-5 examples, then gains diminish. Beyond 5-10 examples, additional examples consume context window space with minimal accuracy improvement.

**Q11. Answer: B**
Assigning a role biases the model toward relevant knowledge, vocabulary, and response patterns from pre-training. A "medical expert" role increases the likelihood of accurate medical terminology and reasoning.

**Q12. Answer: B**
PEFT methods update only a small subset of parameters (often <1% of total), making fine-tuning feasible on consumer hardware while maintaining most of the performance of full fine-tuning.

**Q13. Answer: B**
Fine-tuning excels when you need domain-specific accuracy (medical, legal), have labeled training data, and can invest compute. For simple tasks or rapid prototyping, prompt engineering suffices.

**Q14. Answer: B**
Tree-of-thought explores a search tree of reasoning paths, evaluating multiple branches at each step and pruning unpromising ones. This is more thorough than linear chain-of-thought for complex problems.

**Q15. Answer: B**
A low learning rate prevents large weight updates that would destroy the pre-trained representations. Fine-tuning aims to gently adapt the model, not retrain it from scratch.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

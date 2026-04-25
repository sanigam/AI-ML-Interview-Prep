# Multiple Choice Questions: LLM Inference Optimization

📺 **Video Lecture:** https://youtu.be/W8YlpDD9kuA

Test your understanding of LLM inference optimization techniques essential for deploying efficient language models in production.

---

**Q1. Why does a naive implementation of a 70B parameter LLM require approximately 140GB of GPU memory in FP16 precision?**

A) Because each parameter requires 2 bytes of storage and 70B × 2 bytes = 140GB  
B) Because attention matrices scale quadratically with sequence length  
C) Because the KV-cache is stored separately from model weights  
D) Because the batch size is fixed at 140 requests

---

**Q2. Which of the following is NOT a primary cost factor in LLM inference optimization?**

A) Model memory consumption  
B) Latency per token generation  
C) Compute cost proportional to tokens × model size  
D) The color of the GPU hardware

---

**Q3. What is the key difference between INT8 and INT4 quantization?**

A) INT8 uses 8-bit integers while INT4 uses 4-bit integers; INT4 achieves ~99.2% size reduction vs. FP32, while INT8 achieves ~96.9%  
B) INT8 is slower than INT4 on all hardware  
C) INT4 has zero accuracy loss while INT8 loses 10-20% accuracy  
D) INT8 requires calibration data while INT4 does not

---

**Q4. In GPTQ (Generative Pre-Trained Transformer Quantization), what metric is used to determine which weights should be quantized more aggressively?**

A) The magnitude of each weight  
B) The Hessian of the loss (curvature/sensitivity)  
C) The layer number  
D) The activation values during inference

---

**Q5. Knowledge distillation trains a student model to mimic a teacher. What is the primary advantage of using soft targets from the teacher instead of hard labels?**

A) Soft targets reduce training time  
B) Soft targets contain more information — they preserve which wrong answers the teacher considers plausible  
C) Hard labels are impossible to compute  
D) Soft targets guarantee the student will exceed teacher quality

---

**Q6. In structured pruning vs. unstructured pruning, which approach is generally preferred for LLM deployment and why?**

A) Unstructured pruning because it achieves higher compression ratios  
B) Structured pruning because it removes entire components (heads, layers) that hardware can efficiently skip  
C) Unstructured pruning because all GPUs have sparse matrix acceleration  
D) Both are equally effective; the choice depends only on model size

---

**Q7. Speculative decoding uses a draft model to generate K candidate tokens. How does rejection sampling ensure output quality matches the target model exactly?**

A) By training the draft model on target model outputs  
B) By comparing target probability to draft probability; accepting tokens if P_target > draft probability, ensuring the output distribution matches the target exactly  
C) By always accepting the draft model's tokens  
D) By resampling multiple times from the draft model

---

**Q8. What is the primary bottleneck that KV-cache optimization addresses in LLM generation?**

A) The KV-cache itself uses too much memory  
B) Without caching, generating token tᵢ requires O(i) recomputation of attention; caching reduces this to O(1)  
C) The batch size is too large  
D) The model parameters need to be updated frequently

---

**Q9. How does paged attention (as in vLLM) improve memory efficiency compared to standard KV-cache management?**

A) By storing KV-cache in fixed-size blocks/pages and enabling memory reuse across requests with shared prefixes  
B) By eliminating the need for KV-cache entirely  
C) By using only FP8 precision for all computations  
D) By reducing the attention matrix size

---

**Q10. In continuous batching vs. static batching, which statement is true?**

A) Static batching waits for a full batch before processing, while continuous batching processes requests as they arrive and removes completed requests immediately  
B) Continuous batching requires more model parameters  
C) Static batching always achieves lower latency  
D) Both have identical throughput for variable-length requests

---

**Q11. Tensor parallelism splits a model horizontally across GPUs, requiring AllReduce communication after each layer. When is tensor parallelism preferred over pipeline parallelism?**

A) When you have 50+ GPUs  
B) When you have 2-4 GPUs and prioritize low latency; communication overhead is manageable at small scales  
C) Pipeline parallelism is always superior  
D) When maximizing batch size is the primary goal

---

**Q12. FlashAttention optimizes standard attention by avoiding materialization of the full Q×Kᵀ matrix. Approximately how much memory does FlashAttention save for a 2K-token sequence?**

A) No memory savings; only computation savings  
B) 5-10% reduction  
C) 10-20x less memory by computing attention in blocks without storing the full O(seq_len²) matrix  
D) 100% memory savings — it uses zero memory

---

**Q13. What is the primary distinction between Time-to-First-Token (TTFT) and inter-token latency (ITL)?**

A) TTFT is the total time to generate all tokens; ITL is the time per token  
B) TTFT = latency from request to first token (prefill-dominated); ITL = time between successive tokens (decoding-dominated)  
C) ITL is only relevant for batch inference  
D) TTFT and ITL are identical metrics with different names

---

**Q14. For cost optimization at scale, model routing directs requests to appropriately-sized models. Which optimization is NOT typically used in combination with model routing?**

A) Spot instances for batch processing  
B) Prefix caching  
C) Increasing the model size for all requests  
D) Request fallback (escalating to larger model if smaller model has low confidence)

---

**Q15. Which of the following statements about edge deployment (phones/IoT) of LLMs is most accurate?**

A) Edge deployment uses the same models and techniques as data center inference  
B) Edge deployment requires extreme optimization: aggressive quantization (INT4), knowledge distillation, structured pruning, and typically smaller base models (3-7B) to fit memory constraints and latency budget  
C) Edge deployment is now practical for 70B models without any optimization  
D) CPU inference on edge devices is as fast as GPU inference

---

## Answer Key

**Q1. Answer: A**
A 70B parameter model in FP16 precision requires exactly 70B × 2 bytes = 140GB. FP16 uses 16 bits (2 bytes) per parameter. This storage accounts for model weights; KV-cache adds additional memory depending on batch size and sequence length, often exceeding weight memory.

**Q2. Answer: D**
The primary cost factors are algorithmic and computational (memory, latency, compute), not physical characteristics of the hardware. The color of the GPU has zero impact on inference cost or performance.

**Q3. Answer: A**
INT4 uses 4 bits per parameter (99.2% reduction vs. FP32's 32 bits), while INT8 uses 8 bits (96.9% reduction). INT4 achieves smaller models but with greater accuracy loss (50-80% on reasoning tasks), typically requiring techniques like GPTQ or AWQ to maintain quality. INT8 generally has 10-20% accuracy loss.

**Q4. Answer: B**
GPTQ computes the Hessian of the loss (curvature) to identify weight sensitivity. Weights with low curvature (low sensitivity to changes) are quantized more aggressively with lower precision; high-curvature weights retain higher precision. This respects the model's actual sensitivity, achieving excellent INT4 accuracy.

**Q5. Answer: B**
Soft targets from the teacher contain richer information than hard labels. For example, if the teacher assigns probability 0.3 to a plausible-but-wrong token, the student learns this is an understandable mistake. Hard labels (0 or 1) lose this nuance. This enables smaller distilled models to maintain quality closer to the larger teacher.

**Q6. Answer: B**
Structured pruning is preferred for LLM deployment because it removes entire identifiable components (attention heads, layers) that hardware can efficiently skip, immediately accelerating inference. Unstructured pruning creates sparse matrices that require specialized sparse operations, and most hardware (GPUs/TPUs) doesn't efficiently accelerate unstructured sparsity, limiting speedup.

**Q7. Answer: B**
Rejection sampling compares P_target(token) to P_draft(token). If P_target > P_draft, accept the token; otherwise, reject and resample from the target distribution. This mechanism ensures the output distribution exactly matches the target model, with no quality loss — only a potential speedup reduction if the draft model is weak.

**Q8. Answer: B**
Without KV-cache, computing attention for token tᵢ requires recomputing attention on all tokens t₁...t_{i-1}, which is O(i) cost. With KV-cache, cached K,V embeddings from previous tokens mean computing attention is O(1) per token. This is critical for decoding efficiency during generation.

**Q9. Answer: A**
Paged attention stores KV-cache in fixed-size blocks (like virtual memory paging), enabling dynamic memory reuse. If two requests share a prefix, they can share cached pages rather than duplicating storage. This reduces memory fragmentation and enables 10-50x higher batch sizes on the same GPU hardware.

**Q10. Answer: A**
Static batching waits for a fixed batch size (e.g., 32) before processing, causing high latency if only a few requests arrive. Continuous batching adds arriving requests immediately and removes completed requests, maximizing GPU utilization without waiting for stragglers. Continuous batching achieves 2-4x better throughput.

**Q11. Answer: B**
Tensor parallelism requires AllReduce communication after each layer, scaling poorly to many GPUs. It's best for 2-4 GPUs with low communication overhead and is preferred for latency-critical serving (small batches). Pipeline parallelism scales to 50+ GPUs with point-to-point communication and is better for throughput-optimized serving.

**Q12. Answer: C**
Standard attention computes a full Q×Kᵀ matrix of size [seq_len, seq_len], requiring O(seq_len²) memory — for 2K tokens, this is 4M elements requiring ~8MB per head. FlashAttention avoids materializing this matrix by computing attention in blocks, achieving 10-20x memory reduction and 2-3x speedup.

**Q13. Answer: B**
TTFT (Time-to-First-Token) measures latency from request arrival to the first generated token, dominated by prefill (processing the input prompt). ITL (inter-token latency) measures time between successive generated tokens during decoding. Optimizing TTFT requires batching prefills; optimizing ITL requires fast decoding (KV-cache, quantization).

**Q14. Answer: C**
Model routing is about using appropriately-sized models (routing simple queries to 7B, complex to 70B), which inherently avoids using large models unnecessarily. Option C contradicts routing's purpose. The other options (spot instances, prefix caching, fallback) are all complementary optimizations.

**Q15. Answer: B**
Edge deployment faces extreme constraints: 1-2 second latency budgets, 1-8GB RAM, 1-5GB storage. This requires aggressive quantization (INT4 essential), knowledge distillation (3B models from 13B teachers), structured pruning (50%+ layers removed), and typically smaller base models (3-7B, not 70B). CPU inference is 100x slower than GPU, making GPU/Neural Engine critical on edge devices.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

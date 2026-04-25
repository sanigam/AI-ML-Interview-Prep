# Word Embeddings and Language Models

📺 **Video Lecture:** https://youtu.be/jBVeerEkcSk


## Interview Anchor
- **Word2Vec:** Neural approach learning word embeddings via predicting context (Skip-gram) or word from context (CBOW)
- **BERT:** Bidirectional encoder transformer with MLM (masked language model) and NSP (next sentence prediction) objectives
- **Contextual Embeddings:** Word representations that change based on context, unlike static word2vec embeddings

## Key Concepts Overview

Word embeddings are foundational to modern NLP. The shift from static embeddings (word2vec, GloVe, FastText) to contextual embeddings (ELMo, BERT, GPT) represents a paradigm change: static embeddings assign one vector per word (context-independent), while contextual embeddings compute word vectors based on surrounding context (context-dependent). This evolution parallels the history of language modeling: from n-gram models → neural language models (RNNs) → transformers. Understanding both classical and modern approaches is crucial because: (1) Job interviews test conceptual understanding, not just naming transformers. (2) Many systems still use static embeddings for efficiency. (3) Transfer learning (pre-training + fine-tuning) is now standard, requiring understanding of what models learn. (4) Recent advances (instruction tuning, alignment) build on embedding fundamentals. Mastering this topic demonstrates both historical perspective and cutting-edge knowledge.

---

### Q1: Explain Word2Vec: Skip-gram and CBOW models. What are the key differences?

**A:** Word2Vec learns embeddings by predicting words from context or vice versa. **Skip-gram:** Given center word, predict context words within window w. Loss = `Σ_{c ∈ context} -log P(context_word | center_word)` using softmax. Learns embeddings by treating the task as supervised learning: input word embedding matches output to context embeddings. **CBOW (Continuous Bag of Words):** Given context words, predict center word. Loss = `-log P(center_word | context_words)`. Mathematically, both optimize similar objectives but skip-gram works better on small datasets and rare words (generates multiple training samples per word pair), while CBOW is faster (averages context vectors instead of predicting each individually). Skip-gram: "pushes away" embedding for "King" when predicting context "crown", learning that King relates to crown. CBOW: averages context embeddings ("a", "powerful", "and") to predict center word. Implementation tricks: negative sampling (instead of full softmax, sample k negative words) reduces complexity from O(vocab) to O(k), making training practical on billion-word corpora. Hierarchical softmax is another optimization. Modern impact: Word2Vec demonstrated embeddings have linguistic properties (king - man + woman ≈ queen), inspiring decades of embedding research.

**Interview Tip:** Explain the loss function and why sampling matters for scalability. Mention the famous "king - man + woman = queen" analogy example. Show you understand why skip-gram outperforms CBOW.

---

### Q2: What is GloVe (Global Vectors)? How does it differ from Word2Vec?

**A:** GloVe combines global matrix factorization with local context prediction. Core idea: learning embeddings should capture global co-occurrence statistics (how often words appear together across corpus). Word2Vec only uses local context windows (words within 5-10 positions), potentially missing global patterns. GloVe minimizes: `Σ_{i,j} f(X_{ij}) (w_i^T w_j + b_i + b_j - log X_{ij})^2` where `X_{ij}` = co-occurrence count of words i, j, `w_i`, `w_j` are embeddings, `f(X_{ij})` is a weighting function (reduces weight for very frequent pairs that are less informative). The objective balances: (1) Matching embedding dot product to log co-occurrence (global constraint). (2) Weighting by frequency (common pairs less discriminative). Training: alternates between gradient updates on embedding parameters. Advantages over Word2Vec: (1) Explicit global statistics (word2vec implicit). (2) Faster convergence (supervised loss vs unsupervised sampling). (3) Better on rare words (leverages global statistics). Disadvantages: (1) Requires precomputed co-occurrence matrix (memory intensive for large vocabularies). (2) Not as easily adaptable to streaming/online settings. Empirically, GloVe ≈ Word2Vec on most benchmarks; difference is modest. Modern practice: both are superseded by contextual embeddings (BERT), but GloVe remains useful for baseline models and efficiency-constrained settings.

**Interview Tip:** Explain the weighting function's role (common pairs get less emphasis). Mention that global statistics vs local context is the key conceptual difference. Discuss why both still matter (efficiency, interpretability).

---

### Q3: Explain FastText. How does it handle out-of-vocabulary (OOV) words?

**A:** FastText extends Word2Vec by learning character n-gram embeddings instead of word embeddings. Instead of embedding each word as a single vector, a word is represented as a sum of character n-gram vectors. Example: "where" = ["<wh", "whe", "her", "ere", "re>"] + special boundary markers. Training uses skip-gram loss on character n-grams (typical n=5). Advantages: (1) **Handles OOV words:** unseen word composed of learned character n-grams (e.g., "wheree" = ["<wh", "whe", "her", "ere", "ree", "ee>"]; many overlap with "where"). (2) **Morphology:** similar words share n-grams ("running", "runs", "runner" share common subsequences), embedding space reflects morphological relationships. (3) **Multilingual:** works on languages without clear word boundaries (Chinese, Japanese) where characters are meaningful units. Disadvantage: embeddings less interpretable (not directly assigned to words, but computed on-the-fly from subword components), slightly larger model size (store all n-gram vectors). Empirical results: FastText matches or slightly beats Word2Vec, especially on tasks with OOV (social media, user-generated content with typos). Modern impact: FastText's character-level approach influenced later subword tokenization (BPE, WordPiece) and highlighted the importance of handling morphology. Still used in production for multilingual and typo-robust applications.

**Interview Tip:** Explain the OOV handling concretely—a misspelled word uses similar n-grams. Mention multilingual benefits. Discuss why character n-grams influenced later tokenization methods.

---

### Q4: What are contextual embeddings? Explain ELMo and why it's a major shift from Word2Vec.

**A:** Contextual embeddings compute word vectors dynamically based on surrounding context, unlike static embeddings (word2vec assigns one vector per word regardless of usage). **ELMo (Embeddings from Language Models):** uses a bidirectional LSTM language model trained on forward and backward character-level prediction. For each word, ELMo outputs three vectors: (1) character CNN embedding, (2) first LSTM layer, (3) second LSTM layer. Downstream tasks mix these (weighted sum): output = `γ * (λ_0 * e_{char} + λ_1 * h_{LSTM1} + λ_2 * h_{LSTM2})`. Key insight: different layers capture different information—lower layers encode syntax, higher layers encode semantics. Advantages: (1) **Contextual:** same word "bow" has different embeddings in "bow and arrow" (noun) vs "bow down" (verb). (2) **Transfer learning:** pretrain LM on large corpus, fine-tune or freeze embeddings for downstream tasks. (3) **Interpretability:** analysis shows layers encode distinct linguistic phenomena. Compared to Word2Vec: (1) **Information density:** single contextualized embedding captures more than static embedding (bidirectional context). (2) **Generalization:** pretraining on unlabeled data helps, reducing supervised data requirements. (3) **Trade-off:** slower inference (requires LSTM passes; word2vec is just lookup). Impact: ELMo showed that unsupervised pretraining helps downstream tasks dramatically (ImageNet for NLP). This inspired BERT, GPT, and the modern era of transfer learning in NLP.

**Interview Tip:** Emphasize the context-dependency—show examples where same word has different contextual embeddings. Explain the bidirectional LSTM contribution. Mention pretraining + fine-tuning paradigm shift.

---

### Q5: Explain BERT: architecture, masking strategy, and training objectives (MLM, NSP).

**A:** **BERT (Bidirectional Encoder Representations from Transformers):** Transformer encoder (no decoder) with bidirectional context. Architecture: 12/24 layers (BERT-base/large), 12/16 attention heads, 768/1024 hidden dimensions. Key architectural detail: only encoder (self-attention over full sequence), not decoder (no causal masking). Training objectives: (1) **MLM (Masked Language Modeling):** randomly mask 15% of tokens, predict them from context. Example: "The [MASK] sat on the mat" → predict "cat". This forces bidirectional context understanding (attend left and right). (2) **NSP (Next Sentence Prediction):** binary classification—is sentence B a continuation of sentence A? Helps model understand document structure. Loss = MLM + NSP. Pretraining: trained on 3.3B words (Wikipedia + BookCorpus), optimized with Adam, warm-up. Fine-tuning: add task-specific head (classification, tagging, QA), freeze/unfreeze encoder, train on labeled data. Why effective: (1) Bidirectionality captures full context (unlike GPT's left-to-right). (2) MLM objective directly optimizes for contextual understanding (vs. next-token prediction). (3) Scale: 340M parameters trained on massive corpus. Limitations: MLM adds spurious [MASK] tokens not present at inference (exposure bias), NSP helps marginally and is now often skipped. Variants: RoBERTa removes NSP, trains longer; ELECTRA uses discriminator objective instead of MLM.

**Interview Tip:** Explain MLM's difference from standard LM (all context vs left context). Discuss why bidirectionality helps (full context for understanding). Mention NSP criticism and modern variants.

---

### Q6: Explain the GPT series (GPT-1, GPT-2, GPT-3). What's different from BERT?

**A:** **GPT models** use autoregressive (left-to-right) language modeling. **GPT-1 (2018):** transformer decoder with 12 layers, 117M parameters, trained on BookCorpus, demonstrated transfer learning works for NLP. **GPT-2 (2019):** 1.5B parameters, 40GB text (CommonCrawl), showed scaling to internet-scale data. Demonstrated few-shot learning: provide example input-output pairs, model generates continuations. **GPT-3 (2020):** 175B parameters (100x GPT-2), 45TB text, in-context learning enabled (few-shot and zero-shot without fine-tuning). Key differences from BERT: (1) **Autoregressive vs Bidirectional:** GPT predicts next token from previous tokens (left-to-right causal masking), BERT uses full context. (2) **Training objective:** GPT uses next-token prediction, BERT uses MLM. (3) **Usage:** BERT requires fine-tuning (add task-specific head), GPT enables prompting (describe task in text, model responds). (4) **Scaling:** autoregressive models scale better (loss decreases predictably with scale; Chinchilla laws). Impact: GPT demonstrated that scaling language models (data + parameters) leads to emergent abilities—few-shot learning, reasoning, instruction following. This paradigm shifted NLP from supervised fine-tuning to prompting. Modern LLMs (GPT-4, Claude, Llama) follow GPT's approach. BERT used in retrieval/understanding tasks; GPT used for generation/reasoning.

**Interview Tip:** Explain causal masking's role (left-to-right constraint enables autoregressive generation). Discuss emergent abilities with scale—show you understand why scaling matters. Mention few-shot learning as key capability GPT unlocked.

---

### Q7: Explain T5 and its unified text-to-text framework. What's the advantage?

**A:** **T5 (Text-to-Text Transfer Transformer):** treats all NLP tasks as text-to-text generation. Instead of task-specific architectures, T5 takes text input and generates text output for any task. Examples: "summarize: [text]" → summary, "translate English to French: [text]" → translation, "question: [text] context: [passage]" → answer. Architecture: encoder-decoder transformer (like original "Attention is All You Need"), not encoder-only (BERT) or decoder-only (GPT). Trained with span corruption objective (similar to MLM): randomly remove/replace spans, predict removed text. Key advantage: unified framework—one model handles classification, generation, translation, QA, etc. Just prepend task prefix to input. Parameters: T5-small (60M) to T5-11B. Training: 750GB text (C4 corpus), unsupervised pretraining + supervised fine-tuning. Results: T5-11B matches or beats task-specific SOTA. Tradeoffs: (1) Encoder-decoder slower than decoder-only at inference (must encode full input, then decode token-by-token). (2) Requires task prefix (manual specification or learned). (3) Less scaling laws known vs pure decoder models. Modern impact: T5 influenced instruction-tuned models (T5 + task prefixes ≈ instruction tuning). However, decoder-only models (GPT-3, LLaMA) now dominate due to better scaling and ease of prompting (no architecture change). T5 still strong for seq2seq tasks (summarization, translation).

**Interview Tip:** Explain the text-to-text unification conceptually—all tasks reduced to seq2seq. Discuss why encoder-decoder architecture matters for this. Mention instruction tuning as a modern variant of this idea.

---

### Q8: Explain subword tokenization (BPE, WordPiece, SentencePiece). Why is it crucial for transformers?

**A:** Subword tokenization breaks words into smaller units (subwords), enabling models to handle OOV words, morphology, and rare words. Methods: (1) **BPE (Byte-Pair Encoding):** iteratively merge most frequent adjacent tokens. Start with characters ("t", "h", "e"), merge top pair 100K times. Example: "lower" → ["low", "er"]. (2) **WordPiece:** similar to BPE but merges pairs that maximize likelihood under a language model (used by BERT). (3) **SentencePiece:** operates on bytes, language-agnostic, handles spaces as tokens (popular for multilingual models). Advantages: (1) **Finite vocabulary:** model has fixed vocabulary size (30K-50K common in BERT, 128K in GPT-3), can represent any text. (2) **Morphology:** "running" → ["run", "ning"] captures shared root. (3) **Efficiency:** fewer tokens than character-level, faster than word-level with large vocabulary. (4) **Multilingual:** handles languages without clear word boundaries. Tradeoff: tokens aren't interpretable ("unk" token or multiple subword tokens per word), more tokens than word-level (longer sequences). Implementation: pre-tokenize with regex (split on whitespace/punctuation), then apply BPE. Example output: "don't" → ["don", "'", "t"] or ["don't"] depending on frequency. Modern practice: virtually all transformers use subword tokenization (BERT uses WordPiece, GPT uses BPE, LLaMA uses SentencePiece). This is non-negotiable for production models.

**Interview Tip:** Explain how BPE solves the vocabulary coverage problem. Mention that token length varies (important for sequence length calculations). Show you understand why it matters for multilingual models.

---

### Q9: Explain RoBERTa and ALBERT. What improvements do they make over BERT?

**A:** **RoBERTa (Robustly Optimized BERT):** retrained BERT with better hyperparameters and training procedure. Improvements: (1) **Removed NSP objective** (showed marginal benefit). (2) **Longer training:** trained 10 epochs vs BERT's 1 epoch on 10x larger corpus. (3) **Better masking:** dynamic masking (mask changes each epoch) vs static (BERT masks fixed). (4) **Better pretraining data:** trained on CommonCrawl (less Wikipedia bias). Results: RoBERTa-large beats BERT-large by 2-3% on most benchmarks. Key insight: careful hyperparameter tuning and longer training matter as much as architecture changes. **ALBERT (A Lite BERT):** focuses on parameter efficiency. Techniques: (1) **Parameter sharing:** same weights across layers (reduces parameters from 340M to ~12M for base variant). (2) **Factorized embeddings:** embedding dimension ≠ hidden dimension (embeddings compressed, saves memory). (3) **Sentence ordering prediction** (replacing NSP): more challenging auxiliary task. Results: ALBERT-xxl (223M params) matches BERT-large with fewer parameters. Tradeoff: parameter sharing reduces capacity (slightly lower accuracy), slower inference per layer (dependencies between layers). Modern impact: RoBERTa showed training details matter enormously (reproducibility lesson). ALBERT influenced mobile/edge models (DistilBERT). However, pure scaling (GPT-3 style) now preferred over compression—larger models + distillation > smaller models from scratch. Still, ALBERT principles (factorization, sharing) used in efficient architectures.

**Interview Tip:** Mention RoBERTa's recipe (dynamic masking, longer training, more data) as a lesson in experimental rigor. Explain ALBERT's parameter sharing trade-off (smaller, slower). Discuss why scaling now preferred.

---

### Q10: Explain DistilBERT. How does knowledge distillation work?

**A:** **DistilBERT:** smaller BERT (40% smaller, 60% faster) via knowledge distillation. Distillation transfers knowledge from teacher (BERT-base) to student (DistilBERT). Training: student learns to mimic teacher's output distribution. Loss = `α * CE(student_logits, gold_labels) + (1-α) * CE_soft(student_logits, teacher_logits / T)` where CE_soft is cross-entropy with temperature-scaled logits, T is temperature (typically 3). Temperature softens teacher outputs: `student_probs = softmax(student_logits / T)` (higher T increases softness, spreading probability mass to negative examples). Low T (near 0): soft targets near one-hot (hard targets). High T: soft targets near uniform (many negative examples contribute to learning). Why effective: (1) Teacher outputs encode richer information than hard labels (what alternatives teacher considered). (2) Soft targets reduce noise (mislabeled examples cause smaller gradient). (3) Student learns faster (fewer epochs). Results: DistilBERT matches BERT-base at 40% smaller size, 60% faster (40% parameter reduction from smaller hidden dimension + fewer layers). Tradeoff: slight accuracy drop (~2-3%), increased inference latency (per-token slower, though total latency lower due to fewer layers). Modern practice: knowledge distillation widely used in production (mobile models, latency-critical applications). However, efficiency gains from architecture changes (attention pruning, quantization) now match/exceed distillation.

**Interview Tip:** Explain temperature's role (higher T = softer targets). Show the loss balance between gold labels and soft targets. Mention why soft targets help (richer signal than hard labels).

---

### Q11: What are sentence embeddings? Explain Sentence-BERT (SBERT) and why it matters.

**A:** Sentence embeddings represent entire sentences as fixed vectors, enabling efficient similarity search and clustering. Naive approach: average word embeddings (word2vec) or average contextualized embeddings (BERT). Problem: averaging loses information (order-independent). **Sentence-BERT (SBERT):** fine-tunes BERT for sentence-level tasks. Architecture: (1) Pass sentence through BERT encoder, take [CLS] token (beginning of sequence token that BERT learns to aggregate sentence meaning). (2) Mean-pool all tokens (some variants). (3) Fine-tune on sentence pair tasks: (a) **NLI (Natural Language Inference):** given premise-hypothesis pairs, predict entailment/contradiction/neutral. (b) **STS (Semantic Textual Similarity):** predict similarity score (0-5) for sentence pairs. Training via siamese networks: twin encoders with shared weights, compute distance between pairs, optimize to match ground-truth similarity. Results: SBERT embeddings capture semantic similarity (sentence similarity scores correlate with cosine similarity of embeddings). Advantages: (1) **Fast similarity:** cosine similarity between fixed vectors (vs running BERT on both sentences). (2) **Interpretable:** 384-dim vectors (SBERT-base) are smaller than BERT's 768, enabling visualization. (3) **Transferable:** pretrained SBERT works out-of-the-box for semantic search, clustering, recommendation. Impact: SBERT enabled efficient semantic search and became standard for retrieval-augmented generation (RAG). Sentence embeddings also used in clustering (KMeans on embeddings), duplicate detection, recommendation systems.

**Interview Tip:** Explain why [CLS] token works (BERT learns to aggregate sentence meaning). Show how siamese networks enable sentence similarity optimization. Mention RAG applications (retrieval).

---

### Q12: Explain fine-tuning strategies: full fine-tuning vs feature extraction. When use each?

**A:** **Full fine-tuning:** update all pretrained weights during downstream training. Process: (1) Load pretrained model (BERT, GPT). (2) Add task-specific head (linear classifier for classification, span selector for QA). (3) Train entire model on labeled data. Learning rate: typically 1e-5 to 1e-4 (lower than normal to avoid catastrophic forgetting). Epochs: 2-4 (more data overfits). Advantages: best accuracy (model adapts to task), handles domain shift (medical text differs from news). Disadvantages: slow (update billions of parameters), requires more data (overfitting on small datasets), risk of catastrophic forgetting (erasing pretraining knowledge). **Feature extraction:** freeze pretrained weights, train only task-specific head. Process: (1) Encode data with frozen pretrained model. (2) Train classifier on frozen features. Advantages: fast (no backprop through encoder), requires less data (fewer parameters to fit), stable (no forgetting). Disadvantages: lower accuracy (features not adapted to task), doesn't help with domain shift. When to use: (1) **Full fine-tuning:** large labeled dataset (>10K examples), computational resources available, new domain distinct from pretraining. (2) **Feature extraction:** small dataset (<1K), limited compute, domain similar to pretraining (e.g., Wikipedia pretrained BERT for Wikipedia-like tasks). Hybrid: train with low learning rate (reduces updates) + early stopping (prevent overfitting). Modern practice: full fine-tuning with large models is preferred; smaller models use feature extraction or LoRA (efficient fine-tuning).

**Interview Tip:** Discuss data requirements for each approach. Mention that low learning rate is crucial for fine-tuning (prevents forgetting). Show understanding of catastrophic forgetting problem.

---

### Q13: Explain intrinsic vs extrinsic embedding evaluation. What are word analogy tasks?

**A:** **Intrinsic evaluation:** measure embedding quality directly (without downstream task). Metrics: (1) **Word analogy:** given "A is to B as C is to ?", solve with `embedding(D) ≈ embedding(C) - embedding(A) + embedding(B)`. Examples: "king is to queen as man is to woman" → embeddings satisfy `king - man + woman ≈ queen` (linear relationships). Standard datasets: Google analogy (capital-common, capital-world, currency, family, gram1-9). Pearson correlation with human judgments. (2) **Word similarity:** correlate embedding cosine similarity with human similarity ratings (SimLex-999, WordSim-353). (3) **Clustering:** K-Means on embeddings, purity/ARI vs gold clusters. Advantages: fast, interpretable, reveals what embeddings encode. Limitations: don't predict downstream performance (good analogy ≠ good downstream accuracy). **Extrinsic evaluation:** measure impact on downstream tasks. Examples: sentiment classification (embed reviews, train classifier), NER (use embeddings as features), semantic search (embed queries and docs, rank by similarity). Advantages: directly measure practical utility. Disadvantages: slower, multiple tasks needed for comprehensive assessment, downstream performance depends on task, not just embeddings. Best practice: combine both. Use intrinsic to understand embeddings, extrinsic to validate utility. Example: word2vec has strong intrinsic (analogy) but mediocre extrinsic (downstream); contextual embeddings (BERT) have mediocre intrinsic (contextual, not amenable to analogy) but strong extrinsic (SOTA on most tasks).

**Interview Tip:** Explain why word analogy captures linear structure in embeddings. Mention limitations of intrinsic metrics (not predictive of downstream). Show you use both for comprehensive assessment.

---

### Q14: How are embeddings evaluated? Discuss challenges and best practices.

**A:** Embedding evaluation faces several challenges: (1) **Task variance:** embeddings for sentiment may not transfer to NER (different semantic needs). (2) **Noise in human judgments:** SimLex-999 ratings subjective (humans disagree on similarity). (3) **Domain shift:** embeddings trained on Wikipedia fail on medical text. (4) **Intrinsic-extrinsic mismatch:** strong analogy performance doesn't guarantee downstream accuracy. Best practices: (1) **Multiple datasets:** don't rely on one benchmark (SimLex-999 alone insufficient). (2) **Downstream tasks:** extrinsic evaluation on actual tasks (sentiment, NER, QA). (3) **Statistical significance:** compute confidence intervals, not just point estimates. (4) **Ablations:** compare against baselines (random embeddings, frequency baselines). (5) **Domain diversity:** test on Wikipedia, news, social media, domain-specific text. (6) **Human evaluation:** for generation tasks (summarization), human judges score quality. (7) **Scaling analysis:** report how performance changes with embedding dimension (more dims ≈ higher quality but higher storage/compute). Practical metrics: (1) **Retrieval:** precision@k, NDCG@k on semantic search. (2) **Classification:** accuracy/F1 on sentiment/intent classification. (3) **Clustering:** purity, silhouette score. Red flags: single metric claimed as "best", no confidence intervals, no comparison to strong baselines. Modern practice: benchmark on standard leaderboards (MTEB—Massive Text Embedding Benchmark—evaluates embeddings on 56 tasks).

**Interview Tip:** Mention MTEB leaderboard as standard evaluation. Discuss why multiple metrics matter. Show skepticism of single-metric claims (engineering red flag).

---

### Q15: Explain how scaling laws apply to language models. What are implications for model size/data/compute?

**A:** Scaling laws characterize how loss decreases with model size N, data size D, compute C. Empirical power laws (Kaplan et al., Hoffman et al.): `Loss(N) = A*N^{-α}` (typically α ≈ 0.07), `Loss(D) = B*D^{-β}` (typically β ≈ 0.2). Both matter: loss doesn't converge with infinite parameters (data-limited), nor with infinite data (parameter-limited). **Chinchilla optimal allocation:** for fixed compute budget C, allocate equal compute to parameters and tokens: N ≈ D/20 (train on 20 tokens per parameter). Example: 70B parameter model train on ~1.4T tokens. Implications: (1) **Scale matters enormously:** 10x parameters → ~20% loss improvement. (2) **Data as important as parameters:** doubling D ≈ doubling N for loss. (3) **Emergent abilities:** larger models show qualitatively different behaviors (few-shot learning, reasoning). (4) **Cost:** compute scales as C ∝ N*D; doubling parameters → 3-4x training cost (quadratic in scaling exponents). Practical applications: (1) Deciding model size: tradeoff inference speed vs accuracy. (2) Data collection: if data scarce, smaller model better (avoid overfitting). (3) Resource allocation: if budget fixed, balance compute across training vs inference. (4) Predicting improvements: estimate how much accuracy improves with 2x scale. Modern trend: models scale to 100B+ parameters (LLaMA, GPT-4, Claude), limited by compute availability. Scaling laws hold remarkably well across architectures, data sources, languages (transferable).

**Interview Tip:** Explain the power law and Chinchilla allocation. Discuss emergent abilities as motivation for scaling. Mention that scaling laws enable prediction of improvements before training (useful for planning).

---

## Interview Cheatsheet

**Key Terms:**
- **Word2Vec (Skip-gram):** Predicts context from word via negative sampling, learns word embeddings efficiently
- **GloVe:** Global co-occurrence matrix factorization, balances global statistics with local context
- **FastText:** Character n-gram embeddings, handles OOV words and morphology via composition
- **ELMo:** Bidirectional LSTM language model, contextual embeddings (different per context)
- **BERT:** Bidirectional encoder with MLM and NSP pretraining, fine-tuned for downstream tasks
- **MLM (Masked Language Model):** Predicts randomly masked tokens from bidirectional context
- **RoBERTa:** Improved BERT via longer training, dynamic masking, and more data
- **SBERT (Sentence-BERT):** Fine-tuned for sentence embeddings via NLI and STS tasks
- **Fine-tuning:** Update all pretrained weights on downstream task; requires careful LR, avoids catastrophic forgetting
- **Feature Extraction:** Freeze pretrained weights, train only task-specific classifier; faster, needs less data
- **Scaling Laws:** Loss ∝ N^{-α}, allocate compute equally to parameters and tokens (Chinchilla)

**Rapid-Fire Q&A:**
- **Q: Skip-gram vs CBOW?** **A:** Skip-gram predicts context from word (better), CBOW averages context to predict word (faster)
- **Q: Why negative sampling in Word2Vec?** **A:** Reduces softmax from O(vocab) to O(k), enables training on billions of words
- **Q: GloVe advantage over Word2Vec?** **A:** Captures global co-occurrence, trains faster, better on rare words
- **Q: FastText innovation?** **A:** Character n-grams handle OOV and morphology, shares representations across similar words
- **Q: ELMo vs Word2Vec?** **A:** ELMo contextual (changes per context), Word2Vec static; ELMo bidirectional
- **Q: MLM objective?** **A:** Masks 15% of tokens, predicts from bidirectional context; enables pretraining on unlabeled data
- **Q: SBERT setup?** **A:** Fine-tune BERT on NLI/STS via siamese networks, use [CLS] token for fixed sentence embedding
- **Q: When full fine-tune vs feature extract?** **A:** Full: large dataset and compute; Feature: small dataset and limited compute
- **Q: Sentence embedding challenge?** **A:** Averaging loses information; SBERT fine-tunes to preserve sentence meaning in [CLS]
- **Q: Scaling law implication?** **A:** Allocate compute 50-50 to parameters and tokens; loss predictable via power law

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

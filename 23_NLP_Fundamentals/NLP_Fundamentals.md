# NLP Fundamentals

📺 **Video Lecture:** https://youtu.be/956n9PUi-zg


## Interview Anchor
- **Tokenization:** Process of splitting text into meaningful units (words, subwords, characters) for model processing
- **TF-IDF:** Statistical measure ranking term importance: term frequency in document × inverse frequency across corpus
- **Named Entity Recognition (NER):** Task of identifying and classifying named entities (persons, locations, organizations) in text

## Key Concepts Overview

Natural Language Processing fundamentals form the foundation for modern NLP systems. Before deep learning, rule-based and statistical approaches dominated: regex for pattern matching, TF-IDF for document retrieval, hand-crafted linguistic features (part-of-speech tags, dependency trees). While transformers have superseded many classical methods, understanding these fundamentals is critical for interviews because: (1) They reveal how information is encoded in text. (2) Many production systems still use TF-IDF, BM25, or regex preprocessing. (3) Interviewers test whether you understand why neural methods work—often by asking classical alternatives. (4) Hybrid approaches combining classical retrieval (BM25) with neural ranking are state-of-the-art. Mastering fundamentals demonstrates depth beyond just knowing transformer architectures.

---

### Q1: Explain the text preprocessing pipeline. What does each step accomplish?

**A:** Text preprocessing typically includes: (1) **Lowercasing:** converts "Hello" and "hello" to same token, reducing vocabulary. (2) **Tokenization:** splits into words/subwords; "don't" becomes ["do", "n't"] or similar depending on tokenizer. (3) **Removing punctuation:** strips non-alphanumeric except where semantically important. (4) **Stop word removal:** filters common words (the, a, is) that carry little meaning; reduces noise in IR but can hurt downstream tasks (negations: "not bad" loses meaning). (5) **Stemming/Lemmatization:** converts "running", "runs" to "run" (stemming uses heuristics, lemmatization uses dictionary—lemmatization is more accurate but slower). (6) **Normalization:** handles accents, special characters. The pipeline depends on task: sentiment analysis benefits from keeping negations (skip stop word removal), information retrieval benefits from stemming (improves recall), BERT tokenization is learned (skip manual tokenization). A common mistake: over-preprocessing, removing information the model could use. Modern practice for deep learning: minimal preprocessing, let the model learn.

**Interview Tip:** Tailor preprocessing to the task. Mention tradeoffs—lowercasing helps generalization but loses information. Show you think critically about each step.

---

### Q2: What is the bag-of-words (BoW) model? What are its limitations?

**A:** Bag-of-words represents text as an unordered collection of word frequencies, ignoring grammar and word order. For document d with vocabulary V, BoW creates a vector `x ∈ R^|V|` where `x_i` = count (or binary indicator) of word i in d. For example, "the cat sat on the mat" becomes a vector with counts [the: 2, cat: 1, sat: 1, on: 1, mat: 1, ...]. Advantages: simple, interpretable, works reasonably well for classification and IR. Limitations: (1) **Loses word order:** "good bad movie" and "bad good movie" are identical. (2) **High dimensionality:** vocabulary size can be 10K-100K, creating sparse vectors. (3) **No semantic understanding:** "dog" and "puppy" are completely different dimensions despite similarity. (4) **Poor on short text:** a single-word review loses context. (5) **Doesn't capture negation:** "not good" treated as two separate words, not as negative sentiment. Despite simplicity, BoW is still used in baseline models, spam filtering, and lightweight production systems. It scales well and is interpretable—useful when you need a fast baseline or explainable model.

**Interview Tip:** Mention that BoW is still useful as a baseline. Show you understand when it fails (examples: short text, negation, synonymy) and how neural methods address these.

---

### Q3: Explain TF-IDF and how it improves on BoW for information retrieval.

**A:** TF-IDF (Term Frequency-Inverse Document Frequency) weighs term importance: `TF-IDF(term, doc) = TF(term, doc) × IDF(term)` where `TF = count(term, doc) / total_words(doc)` (normalized term frequency) and `IDF = log(total_docs / docs_containing_term)`. The intuition: common words across many documents are less informative (low IDF: "the" appears everywhere), while rare discriminative words have high IDF (e.g., "covid" in 2020 articles). TF-IDF improves BoW by addressing the common word problem—stop words get downweighted automatically. For IR, TF-IDF documents are ranked by cosine similarity to query. Advantages: interpretable, computationally efficient (O(vocab_size)), works well for domain-specific vocabulary. Limitations: still ignores word order/semantics, assumes term independence, brittle to misspellings/synonyms. Empirically, TF-IDF is surpassed by dense embeddings (word2vec, BERT) for semantic search, but hybrid IR systems combine TF-IDF (sparse, exact match) with dense vectors (semantic match), leveraging both. BM25 (Okapi BM25) improves TF-IDF by modeling term saturation (diminishing returns of term frequency) and document length normalization—still widely used in production search systems.

**Interview Tip:** Explain the sparse vs dense tradeoff. TF-IDF is fast and interpretable; embeddings are semantic but require more compute. Hybrid search is increasingly popular.

---

### Q4: What are n-grams? Give examples and explain their use in language modeling.

**A:** n-grams are sequences of n consecutive tokens. 1-grams (unigrams): individual words. 2-grams (bigrams): "the cat", "cat sat". 3-grams (trigrams): "the cat sat". For language modeling, n-gram models compute probability using Markov assumption: `P(word_t | history) ≈ P(word_t | word_{t-n+1}, ..., word_{t-1})` (depends only on previous n-1 words). Example: bigram model predicts next word from previous word—"the" often precedes "dog" or "cat". Training: count n-gram frequencies in corpus, smooth (add-one smoothing, backoff) to handle unseen n-grams. Advantages: interpretable, fast, works for small vocabularies. Limitations: doesn't capture long-range dependencies (sentence meaning depends on context > 5 words), high sparsity for large n (5-grams rarely observed), curse of dimensionality. Modern language models (transformers, RNNs) replaced n-gram models by learning to condition on full history efficiently. However, n-grams are still used in: spell correction (edit distance + language model), text generation baseline, lightweight systems. Character-level n-grams are useful for morphologically rich languages and handle OOV (out-of-vocabulary) words naturally.

**Interview Tip:** Explain the Markov assumption and its limitation. Show you understand why neural models outperform n-grams (can model long-range dependencies). Mention smoothing techniques briefly.

---

### Q5: What is Named Entity Recognition (NER)? Explain a pipeline approach (rule-based, feature-based, neural).

**A:** NER identifies and classifies named entities (persons, locations, organizations, dates, money, etc.) in text. Example: "Apple CEO Tim Cook announced revenue of $100M" → [Apple: ORG, Tim Cook: PERSON, $100M: MONEY]. Approaches: (1) **Rule-based (regex):** patterns like "capital letter + lowercase" → likely person. Fast, interpretable, but brittle (misses "McDonald's"). (2) **Feature-based (CRF):** extract hand-crafted features (word capitalization, POS tags, character n-grams, gazetteer lists), use conditional random fields (CRF) to model label dependencies. More flexible than regex, still interpretable. (3) **Neural (BiLSTM-CRF, transformers):** BiLSTM encodes context, CRF layer models label transitions (avoid invalid sequences), or use transformer + classification head. Modern approaches fine-tune BERT on NER, achieving >90% F1 on standard benchmarks. Evaluation: precision (correct / predicted), recall (correct / actual), F1 (harmonic mean). Challenges: domain shift (entities in news vs medical text differ), ambiguity (is "Apple" a company or fruit?), long entities (multi-word names). Production systems often combine approaches: weak supervision (distant labeling) + transformer fine-tuning.

**Interview Tip:** Describe the progression from regex → CRF → neural, showing understanding of when each is appropriate. Mention recent advances: zero-shot NER with prompting (e.g., GPT-based).

---

### Q6: Explain part-of-speech (POS) tagging. Why is it useful despite modern neural methods?

**A:** POS tagging assigns grammatical categories (noun, verb, adjective, etc.) to words. "The cat sat on the mat" → [The: DET, cat: NOUN, sat: VERB, on: ADP, the: DET, mat: NOUN]. Approaches: (1) **Rule-based:** context patterns (word after "the" is often noun). (2) **HMM / Markov model:** treat as sequence labeling, tags depend on previous tag and word. (3) **Neural:** BiLSTM over word embeddings, or fine-tune transformer. Modern transformers incorporate POS implicitly (BERT's contextualized embeddings encode syntactic information). Yet POS tagging remains useful because: (1) **Interpretability:** directly tells you grammar, useful for rule-based NLP. (2) **Low-resource languages:** POS taggers exist for more languages than full transformers. (3) **Feature engineering:** POS tags serve as features for downstream tasks (coreference, relation extraction). (4) **Error analysis:** if your semantic task fails, check POS—maybe the model misunderstood syntax. (5) **Linguistic insight:** analyzing POS patterns reveals corpus properties (Shakespeare has more adjectives than news text). Modern practice: use transformer outputs directly rather than explicit POS tagging, but understand POS conceptually for debugging and linguistically-motivated architectures.

**Interview Tip:** Show you know POS is still relevant for interpretability and low-resource settings. Mention that transformer embeddings encode POS implicitly.

---

### Q7: Explain dependency parsing and its relationship to semantic understanding.

**A:** Dependency parsing extracts grammatical structure: directed graph where words point to their syntactic head. Example: "The quick brown fox jumps" → [quick → fox (adjective modifier), brown → fox, The → fox (determiner), fox → jumps (nsubj = nominal subject), jumps → ROOT]. Relations include: nsubj (nominal subject), dobj (direct object), prep (preposition), mod (modifier), etc. Parsing reveals structure independent of word order—"fox jumps" vs "jumps fox" have same structure but different meaning (one is grammatical). Approaches: (1) **Transition-based:** use stack/buffer, make decisions (shift/reduce/left-arc/right-arc) greedily or with beam search. (2) **Graph-based:** score all possible edges, find maximum spanning tree (Chu-Liu-Edmonds algorithm). (3) **Neural:** BiLSTM + arc scoring network, or transformer fine-tuned on parsing. Dependency trees are useful for: extracting relations ("subject verb object"), identifying predicates, linguistic analysis, semantic role labeling (SRL: "who did what to whom"). Semantic parsing goes further—converting text to logical forms or abstract meaning representation (AMR). Modern neural methods often skip explicit parsing, learning end-to-end from text → prediction, but dependency structure is useful for interpretability and structured prediction.

**Interview Tip:** Explain the graph structure clearly with examples. Mention that while neural models often skip explicit parsing, understanding syntax helps interpret model decisions.

---

### Q8: Explain sentiment analysis. Compare rule-based, statistical, and neural approaches.

**A:** Sentiment analysis classifies text as positive, negative, or neutral. Approaches: (1) **Lexicon-based (rule):** use sentiment dictionary ("good" → +1, "bad" → -1), aggregate scores. Simple, interpretable ("The movie is bad but the acting is good" = mixed), generalizes to new domains without training. Limitations: missing context (negation: "not good" should be negative), sarcasm, domain shift (medical "positive" ≠ sentiment). (2) **Statistical (Naive Bayes, SVM):** count positive/negative word frequencies, train classifier. Better than lexicon at learning what features matter for a dataset, but still relies on word-level features, missing compositionality. (3) **Neural (CNN, RNN, Transformer):** learn rich representations. BERT fine-tuned on sentiment achieves >95% accuracy. Captures negation ("not good" clearly negative from learned embeddings), sarcasm, long-range context. Limitations: requires labeled data, less interpretable, slower inference. Hybrid approaches combine lexicon (interpretability) + neural (accuracy): use lexicon scores as features for transformer, or use transformer for data annotation then lexicon for efficiency. Modern production: transformers for high-accuracy applications, lexicon for zero-shot (no labeled data) or interpretability requirements.

**Interview Tip:** Show understanding of tradeoffs: lexicon (fast, interpretable) vs neural (accurate, data-hungry). Mention negation as a key challenge that lexicon-based methods struggle with.

---

### Q9: What is text classification? Describe end-to-end approaches.

**A:** Text classification assigns documents to categories (spam/not-spam, sentiment, topic, intent). Classical pipeline: (1) Extract features (BoW, TF-IDF, hand-crafted linguistic features). (2) Train classifier (Naive Bayes, SVM, logistic regression). Modern: (1) Pretrain embedding model (BERT, GPT). (2) Fine-tune on labeled data: frozen embeddings + linear classifier, or fine-tune entire model. For small labeled datasets (~100 samples), fixed embeddings + simple classifier work well and avoid overfitting. For large datasets (>10K samples), end-to-end fine-tuning outperforms. Few-shot approaches: prompt LLMs ("classify as positive/negative: ...") achieve reasonable accuracy without fine-tuning, though full fine-tuning remains best. Multi-class vs multi-label: multi-class assigns one label (spam or not), multi-label assigns multiple (article can be [tech, science, health]). Evaluation: accuracy (overall), precision/recall/F1 per class (handle imbalance). Challenges: class imbalance (99% negative, 1% positive—accuracy useless), domain shift (model trained on movie reviews fails on product reviews), ambiguous labels (some reviews genuinely mixed). Modern practice: balance classes via sampling or loss weighting, use pretrained models for generalization.

**Interview Tip:** Describe the full pipeline end-to-end. Mention why pretrained models help (transfer learning from massive unlabeled text). Address class imbalance—shows practical understanding.

---

### Q10: Explain regular expressions in NLP. Give examples of their use and limitations.

**A:** Regular expressions (regex) match text patterns. Examples: `\d+` matches digits ("2024"), `\b\w+\b` matches words, `[A-Z]\w+` matches capitalized words. NLP uses regex for: (1) **Preprocessing:** extract emails `\w+@\w+\.\w+`, URLs `https?://\S+`, phone numbers. (2) **Tokenization:** split on whitespace/punctuation using `\s+` or `[.,;]`. (3) **Entity extraction:** simple NER for dates `\d{1,2}/\d{1,2}/\d{4}`, money `\$\d+`, IDs. (4) **Validation:** check if text matches expected format. (5) **Pattern-based rules:** detect negation with `\b(not|never)\s+\w+` (fragile but interpretable). Advantages: deterministic, fast, interpretable, no training data needed. Limitations: (1) Brittle to typos ("2024a" doesn't match `\d+`). (2) Over-matching or under-matching (getting boundaries right is hard). (3) Can't capture semantics ("sad" and "unhappy" are synonymous but different patterns). (4) Maintenance burden (complex regex is unreadable). Modern NLP couples regex with ML: regex extracts candidate patterns, neural model scores them. Hybrid approach: use regex for high-precision rules (invoke human for edge cases), neural for high-recall (catch variations). Regex remains essential for production systems despite neural advances—useful for structured data extraction and anomaly detection.

**Interview Tip:** Show practical regex knowledge with examples. Mention that regex + neural hybrid is common in production. Don't oversell neural methods—some tasks are better solved with regex.

---

### Q11: Explain language modeling basics. Define perplexity and why it matters.

**A:** Language modeling estimates probability distribution over text: `P(word_1, word_2, ..., word_n)`. Applications: speech recognition (score hypothesis), machine translation (score translations), text generation. N-gram models use conditional probability: `P(word_t | word_{t-n+1:t-1})`. Neural language models (RNN, transformer) learn `P(word_t | word_1:t-1)` (full history). Perplexity measures model quality: `Perplexity = exp(-1/N * Σ_i log P(word_i | context_i))` where N is number of words. Interpretation: average branching factor—if perplexity = 50, the model is as confused as uniformly choosing from 50 words at each step. Lower perplexity = better model. For English text, humans achieve ~1-2 perplexity (very predictable: "the cat sat on the ..."), basic n-gram models ~100-200, modern BERT-like models ~20-40. Perplexity decreases with better models and larger training data (Chinchilla scaling laws show predictable relationships). Evaluation: use held-out test set to avoid overfitting. Perplexity is useful for ranking models but imperfect—high perplexity doesn't always mean worse downstream performance, and perplexity on Penn Treebank doesn't predict real-world applicability (distribution mismatch). Still, it's a standard baseline metric.

**Interview Tip:** Explain perplexity as geometric mean of 1/probabilities. Mention that it's device-independent (log base e gives nats; conversion to bits is common in papers). Discuss its limitations.

---

### Q12: Explain edit distance (Levenshtein distance). How is it used in NLP?

**A:** Edit distance counts minimum edits (insertion, deletion, substitution) to transform string A to string B. Example: "cat" to "car" requires 1 substitution (t→r), distance = 1. Computed via dynamic programming: `DP[i][j] = minimum edits to transform A[0:i] to B[0:j]`. Recurrence: `DP[i][j] = min(DP[i-1][j]+1, DP[i][j-1]+1, DP[i-1][j-1] + (A[i]≠B[j]))` (delete, insert, or substitute). Time: O(|A|×|B|), space: O(min(|A|,|B|)) with optimization. Uses: (1) **Spell correction:** find similar words in dictionary (distance < 2). (2) **Fuzzy matching:** match entities despite typos (address matching). (3) **Sequence alignment:** compare DNA sequences (bioinformatics uses variants like alignment distance). (4) **Similarity metric:** normalize by max length: similarity = 1 - distance/max(|A|,|B|). Limitations: treats all edits equally (typo "teh" → "the" has same cost as "dog" → "cat", though former is obviously a typo). Variants: Damerau-Levenshtein allows transpositions (t-h swap), Hamming distance for equal-length strings (substitutions only). Modern NLP uses character-level n-grams or learned embeddings instead of edit distance for most tasks, but it remains useful for spell correction and exact string matching problems.

**Interview Tip:** Write out the DP recurrence clearly. Mention space optimization (only need previous row). Explain when to use (spell correction) vs alternatives (learned similarity for semantic).

---

### Q13: Explain BM25 and why it's still used in modern information retrieval despite neural methods.

**A:** BM25 (Okapi BM25) is a probabilistic IR model ranking documents by relevance to query. It improves TF-IDF with: (1) **Term saturation:** diminishing returns of term frequency—a term appearing 10 times in a document isn't 10x more relevant than 1 time. Modeled as `TF / (k1 * ((1-b) + b * doclen/avglen) + TF)` where k1, b are hyperparameters (typically k1=1.5, b=0.75). (2) **Document length normalization:** prevents long documents from dominating (they naturally have more term matches). (3) **IDF weighting:** `log((N - df + 0.5) / (df + 0.5))` where df is document frequency, N is corpus size. BM25 is more principled than TF-IDF and works better empirically. Why still used: (1) **Efficiency:** O(df) computation (only iterate documents containing query terms), much faster than dense embeddings. (2) **Exact match:** captures keyword relevance (dense embeddings miss exact phrases). (3) **Explainability:** can show why a document ranked high (which terms matched). (4) **Robustness:** works well without parameter tuning or training. Modern systems use hybrid search: BM25 for recall (fast, catches obvious matches) + dense embeddings (semantic reranking). Research shows BM25 + BERT reranker outperforms either alone. BM25 is now commodity (Elasticsearch, Solr implement it), making it standard in production.

**Interview Tip:** Explain term saturation and length normalization—these are the key innovations over TF-IDF. Mention hybrid search as state-of-the-art. Show you know when to use BM25 vs neural methods.

---

### Q14: What is information retrieval? Describe the evaluation metrics (precision, recall, MRR, NDCG).

**A:** Information retrieval (IR) finds relevant documents for a query from a large corpus. Metrics: (1) **Precision@k:** fraction of top-k results that are relevant = `|relevant ∩ retrieved| / k`. (2) **Recall@k:** fraction of all relevant documents in top-k = `|relevant ∩ retrieved| / |relevant|`. (3) **Mean Reciprocal Rank (MRR):** average of 1/rank of first relevant result = `1/N * Σ 1/rank_i`. Useful when only first result matters (search engine click-through). (4) **NDCG@k (Normalized Discounted Cumulative Gain):** weights results by relevance score and position. `DCG@k = Σ_{i=1}^k (2^{rel_i} - 1) / log_2(i+1)` (higher relevance and earlier position score higher). NDCG = DCG / IDCG (ideal DCG of perfect ranking). Useful for ranking with multiple relevance levels (0/1/2 stars). Choosing metrics depends on task: top-1 accuracy for search (MRR), top-10 for recommenders (NDCG@10), or balanced precision-recall. Pitfall: optimizing for one metric can hurt others (high precision, low recall). IR systems are evaluated offline (test set) and online (A/B tests with users). Modern IR: combine lexical (BM25) + semantic (embeddings) + learning-to-rank models.

**Interview Tip:** Explain why precision-recall tradeoff matters. Show you understand when to use NDCG vs MRR. Mention that online metrics (CTR, dwell time) matter more than offline metrics for real systems.

---

### Q15: Explain text similarity measures. When would you use cosine similarity vs other measures?

**A:** Text similarity measures quantify how similar two texts are. Key measures: (1) **Cosine similarity:** `cos(A, B) = (A·B) / (||A|| ||B||)` for vectors (embeddings). Ranges [0,1] (or [-1,1] with signed embeddings). Invariant to magnitude; only cares about direction. (2) **Jaccard similarity:** `|A ∩ B| / |A ∪ B|` for sets (tokens). Ranges [0,1]. Good for exact word matching. (3) **Euclidean distance:** `||A - B||_2`. Sensitive to magnitude; high distance means dissimilar. (4) **Edit distance:** token-level or character-level differences. (5) **Semantic similarity:** learned from embeddings (word2vec, BERT), captures meaning beyond surface form. Cosine similarity is standard for embeddings because: (1) Computationally efficient (dot product after normalization). (2) Semantically meaningful (direction in embedding space encodes meaning). (3) Invariant to length (doesn't matter if embedding is [1,0] or [10,0]). Use Jaccard for set overlap (tags, keywords, exact matching). Use edit distance for typo correction. Use learned similarity (BERT embeddings + cosine) for semantic search. Hybrid: combine cosine (semantic) + Jaccard (exact match) to boost recall. Common mistake: using raw cosine without embedding normalization (works but less stable); always normalize before cosine.

**Interview Tip:** Explain why cosine is standard for embeddings (computational + semantic). Show you know when alternatives are better (Jaccard for sets, edit distance for typos). Mention L2 normalization as a best practice.

---

## Interview Cheatsheet

**Key Terms:**
- **Tokenization:** Breaking text into meaningful units (words, subwords, characters)
- **TF-IDF:** Term frequency × inverse document frequency, weights terms by importance in corpus
- **NER:** Named Entity Recognition, identifies and classifies named entities (PERSON, ORG, LOC, etc.)
- **POS Tagging:** Assigns grammatical categories (NOUN, VERB, ADJECTIVE, etc.) to words
- **Dependency Parsing:** Extracts syntactic structure as directed graph of head-dependent relationships
- **Perplexity:** Inverse probability of test set averaged over words; lower = better language model
- **Edit Distance (Levenshtein):** Minimum insertions/deletions/substitutions to transform one string to another
- **BM25:** Probabilistic IR model with term saturation and length normalization, beats TF-IDF
- **NDCG:** Normalized Discounted Cumulative Gain, evaluation metric for ranking with relevance scores
- **Cosine Similarity:** Dot product of normalized vectors; standard for embedding similarity

**Rapid-Fire Q&A:**
- **Q: Why lowercase in preprocessing?** **A:** Reduces vocabulary, treats "Hello" and "hello" identically, improves generalization
- **Q: When would you skip stop word removal?** **A:** Sentiment analysis (negations like "not" matter), NER, any task where common words have meaning
- **Q: What's the advantage of lemmatization over stemming?** **A:** Lemmatization uses dictionary, more accurate; stemming is heuristic but faster
- **Q: Bag-of-words main limitation?** **A:** Ignores word order; "good bad movie" identical to "bad good movie"
- **Q: Why is IDF in TF-IDF important?** **A:** Downweights common words (the, a, is) that appear everywhere, upweights discriminative words
- **Q: Perplexity formula?** **A:** exp(-1/N * Σ log P(word_i | context)), geometric mean of 1/probabilities
- **Q: Edit distance algorithm?** **A:** Dynamic programming, DP[i][j] = min(delete, insert, substitute), O(|A|×|B|)
- **Q: Why BM25 beats TF-IDF?** **A:** Term saturation (diminishing returns) and length normalization
- **Q: When to use Jaccard over cosine?** **A:** Set overlap (tags, keywords); cosine for semantic embeddings
- **Q: What does NDCG handle that precision doesn't?** **A:** Multiple relevance levels (0/1/2 stars) and position-based discounting

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

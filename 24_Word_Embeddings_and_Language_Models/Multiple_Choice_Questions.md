# Multiple Choice Questions: Word Embeddings and Language Models

📺 **Video Lecture:** https://youtu.be/jBVeerEkcSk


Test your understanding of word embeddings and language models for AI/ML interviews.

---

**Q1. Word2Vec's Skip-gram model learns embeddings by:**

A) Predicting the center word from its context words
B) Predicting context words from a given center word
C) Counting word co-occurrences in a matrix
D) Applying PCA to TF-IDF vectors

---

**Q2. The famous Word2Vec analogy "king − man + woman ≈ queen" demonstrates that:**

A) Word2Vec memorizes word pairs
B) Word embeddings capture semantic relationships as linear vector operations
C) All word analogies are perfectly solved
D) Embeddings are random vectors

---

**Q3. GloVe differs from Word2Vec primarily because it:**

A) Uses only local context windows
B) Combines global co-occurrence statistics with local context prediction
C) Cannot handle rare words
D) Requires labeled training data

---

**Q4. FastText handles out-of-vocabulary (OOV) words by:**

A) Assigning them a zero vector
B) Representing words as sums of character n-gram embeddings, allowing OOV words to be constructed from known subwords
C) Ignoring them during training
D) Using a fixed random embedding

---

**Q5. Contextual embeddings (like ELMo and BERT) differ from static embeddings (like Word2Vec) because:**

A) They assign one fixed vector per word regardless of context
B) They compute different word representations depending on the surrounding context
C) They use smaller vocabularies
D) They cannot be used for downstream tasks

---

**Q6. BERT's masked language modeling (MLM) objective masks tokens with the strategy:**

A) 100% replaced with [MASK]
B) 80% [MASK], 10% random token, 10% unchanged
C) 50% [MASK], 50% unchanged
D) All tokens are masked simultaneously

---

**Q7. The key advantage of BERT's bidirectional context over GPT's left-to-right context is:**

A) BERT trains faster
B) BERT can attend to both left and right context simultaneously, giving richer representations for understanding tasks
C) BERT generates text better than GPT
D) GPT cannot be fine-tuned

---

**Q8. Negative sampling in Word2Vec training:**

A) Removes negative sentiment words from the corpus
B) Replaces the expensive full softmax with a simplified objective using k randomly sampled negative examples
C) Only trains on words with positive sentiment
D) Increases the vocabulary size

---

**Q9. Subword tokenization (BPE, WordPiece) is crucial for transformers because it:**

A) Eliminates the need for embeddings
B) Provides a finite vocabulary that can represent any text while handling rare and OOV words via subword decomposition
C) Always produces single-token words
D) Is only needed for non-English languages

---

**Q10. RoBERTa improved upon BERT by:**

A) Adding a decoder for generation
B) Removing NSP, using dynamic masking, training longer on more data, and tuning hyperparameters carefully
C) Reducing the model size to half
D) Switching from transformers to RNNs

---

**Q11. The GPT series (GPT-1 → GPT-2 → GPT-3) primarily demonstrated the power of:**

A) Reducing model size for efficiency
B) Scaling parameters and data, leading to emergent few-shot and zero-shot capabilities
C) Using encoder-only architectures
D) Training exclusively on labeled data

---

**Q12. ELMo generates contextual embeddings by:**

A) Using a transformer encoder
B) Combining outputs from a bidirectional LSTM language model at multiple layers
C) Looking up pre-computed word vectors
D) Training a GAN on word pairs

---

**Q13. T5 treats all NLP tasks as text-to-text by:**

A) Using only a decoder without any encoder
B) Prepending task-specific prefixes (e.g., "translate English to French:") and generating text outputs for all tasks
C) Removing the attention mechanism
D) Training separate models for each task

---

**Q14. ALBERT reduces BERT's parameter count primarily through:**

A) Removing attention heads
B) Cross-layer parameter sharing and factorized embedding parameterization
C) Using character-level instead of subword tokenization
D) Reducing the training corpus size

---

**Q15. The main limitation of static word embeddings (Word2Vec, GloVe) is:**

A) They require too much memory
B) They assign the same vector to a word regardless of context, failing to capture polysemy (e.g., "bank" as financial vs. river)
C) They cannot represent common words
D) They require labeled training data

---

## Answer Key

**Q1. Answer: B**
Skip-gram takes a center word as input and predicts surrounding context words. This generates multiple training pairs per word occurrence, making it effective for learning rare word representations.

**Q2. Answer: B**
Vector arithmetic on embeddings reveals that semantic relationships (gender, royalty) are encoded as consistent directional differences in vector space, a key property of well-trained embeddings.

**Q3. Answer: B**
GloVe explicitly factorizes the global word co-occurrence matrix, combining corpus-level statistics with local context learning. Word2Vec only uses local context windows during training.

**Q4. Answer: B**
FastText represents each word as a bag of character n-grams. An OOV word shares n-grams with known words, allowing a reasonable embedding to be computed from its subword components.

**Q5. Answer: B**
Contextual embeddings produce different vectors for the same word in different contexts (e.g., "bank" in "river bank" vs. "bank account"), capturing polysemy that static embeddings miss entirely.

**Q6. Answer: B**
The 80/10/10 strategy prevents the model from only learning to predict [MASK] tokens. Random replacement and unchanged tokens force the model to maintain good representations for all positions.

**Q7. Answer: B**
BERT's bidirectional attention sees both preceding and following context for each token, providing more complete representations for understanding tasks like NER, QA, and classification.

**Q8. Answer: B**
Instead of computing softmax over the entire vocabulary (expensive), negative sampling trains the model to distinguish the true context word from k randomly sampled "negative" words, reducing complexity from O(V) to O(k).

**Q9. Answer: B**
Subword tokenization (e.g., "unhappiness" → "un", "happiness") handles any text with a fixed vocabulary of 30K-50K tokens, gracefully decomposing rare or unseen words into known subword units.

**Q10. Answer: B**
RoBERTa showed that careful training choices (removing the NSP objective, dynamic masking, more data, longer training) significantly improve performance without any architectural changes to BERT.

**Q11. Answer: B**
GPT-3 (175B parameters) demonstrated that scaling model size and training data enables emergent capabilities like few-shot learning, instruction following, and reasoning that smaller models lack.

**Q12. Answer: B**
ELMo uses a two-layer bidirectional LSTM. The final embedding is a learned weighted combination of the character-level embedding and both LSTM layer outputs, capturing different levels of linguistic information.

**Q13. Answer: B**
T5's unified text-to-text format uses task prefixes so a single encoder-decoder model handles classification, translation, summarization, and QA without task-specific architectures.

**Q14. Answer: B**
ALBERT shares transformer weights across all layers and factorizes the embedding matrix into two smaller matrices (embedding dim ≠ hidden dim), dramatically reducing parameters while maintaining performance.

**Q15. Answer: B**
Static embeddings produce one vector per word, so polysemous words like "bank" (financial/river) get a single averaged representation. Contextual embeddings solve this by conditioning on surrounding text.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

# Multiple Choice Questions: NLP Fundamentals

📺 **Video Lecture:** https://youtu.be/956n9PUi-zg


Test your understanding of natural language processing fundamentals for AI/ML interviews.

---

**Q1. In the text preprocessing pipeline, stop word removal should be used with caution because:**

A) It always improves model accuracy  
B) It increases vocabulary size  
C) It can remove semantically important words like negations ("not") that change meaning  
D) It is only applicable to English text

---

**Q2. The bag-of-words (BoW) model represents text as:**

A) A sequence of word embeddings preserving order  
B) A parse tree of grammatical relationships  
C) A probability distribution over topics  
D) An unordered collection of word frequencies, ignoring grammar and word order

---

**Q3. TF-IDF improves upon raw term frequency by:**

A) Adding word order information  
B) Downweighting terms that appear frequently across many documents (high document frequency)  
C) Using neural networks for feature extraction  
D) Removing all stop words automatically

---

**Q4. In the IDF component of TF-IDF, a word that appears in nearly every document will have:**

A) A very low IDF score (close to 0)  
B) An IDF score of exactly 1  
C) A very high IDF score  
D) A negative IDF score

---

**Q5. N-gram language models suffer from the "curse of dimensionality" because:**

A) They cannot handle unigrams  
B) They require too much memory for any value of n  
C) The number of possible n-grams grows exponentially with n, making most n-grams unobserved in training  
D) They require GPU computation

---

**Q6. Named Entity Recognition (NER) is the task of:**

A) Identifying and classifying named entities (persons, organizations, locations) in text  
B) Translating text between languages  
C) Generating new text from a prompt  
D) Computing word similarity scores

---

**Q7. The key assumption in Naive Bayes text classification is:**

A) Words are conditionally independent given the class label  
B) The vocabulary must be fixed beforehand  
C) All words have equal importance  
D) The document must contain at least 100 words

---

**Q8. Lemmatization differs from stemming in that:**

A) Lemmatization only works for English  
B) Lemmatization is always faster  
C) Stemming always produces better results  
D) Lemmatization produces valid dictionary words using linguistic rules, while stemming applies heuristic suffix removal

---

**Q9. BM25 improves upon TF-IDF by incorporating:**

A) Neural embeddings for each term  
B) Word order through n-grams  
C) Part-of-speech tags for each word  
D) Term frequency saturation (diminishing returns) and document length normalization

---

**Q10. Sentiment analysis using a lexicon-based approach fails most notably on:**

A) Long documents  
B) Text with negation ("not good"), sarcasm, or domain-specific language  
C) Text written in all lowercase  
D) Documents with many stop words

---

**Q11. Part-of-speech (POS) tagging assigns:**

A) Translation pairs to word alignments  
B) Sentiment labels to sentences  
C) Topic labels to documents  
D) Grammatical categories (noun, verb, adjective, etc.) to each word

---

**Q12. Dependency parsing produces:**

A) A flat list of word frequencies  
B) A probability distribution over topics  
C) A directed graph showing grammatical relationships between words (subject, object, modifier)  
D) A sequence of POS tags

---

**Q13. For multi-label text classification (one document can belong to multiple categories), the appropriate output layer is:**

A) Independent sigmoid activations per label  
B) Argmax over all classes  
C) Linear regression  
D) Softmax (single label)

---

**Q14. In hybrid search systems, sparse retrieval (TF-IDF/BM25) is combined with dense retrieval (embeddings) because:**

A) Sparse methods are the only ones that scale  
B) Dense methods cannot handle short queries  
C) Sparse handles exact keyword matching while dense captures semantic similarity, complementing each other  
D) Sparse methods are always more accurate

---

**Q15. The F1 score for NER evaluation is preferred over accuracy because:**

A) F1 only considers the majority class  
B) F1 is always higher than accuracy  
C) Most tokens are non-entities (class imbalance), making accuracy misleadingly high  
D) Accuracy cannot be computed for sequence labeling

---

## Answer Key

**Q1. Answer: C**
Removing stop words like "not" from "not good" changes the sentiment from negative to positive. Preprocessing decisions should be task-specific; for sentiment analysis, keeping negations is critical.

**Q2. Answer: D**
BoW creates a vector of word counts ignoring order and grammar. "Dog bites man" and "man bites dog" produce identical BoW vectors, losing the crucial difference in meaning.

**Q3. Answer: B**
IDF = log(total_docs / docs_containing_term) gives low weight to ubiquitous terms like "the" and high weight to discriminative terms, improving document representation for retrieval.

**Q4. Answer: A**
If a term appears in most documents, docs_containing_term ≈ total_docs, so IDF = log(1) ≈ 0. The term is considered uninformative for distinguishing between documents.

**Q5. Answer: C**
For vocabulary size V, there are Vⁿ possible n-grams. Most are never observed in training, causing severe sparsity. Smoothing techniques partially address this but the fundamental problem limits n-gram models.

**Q6. Answer: A**
NER identifies spans of text referring to named entities and classifies them into categories such as PERSON, ORGANIZATION, LOCATION, DATE, and MONEY.

**Q7. Answer: A**
Naive Bayes assumes P(x₁,...,xₙ|y) = ∏P(xᵢ|y), treating features as conditionally independent given the class. Despite being unrealistic, this assumption makes computation tractable and often works well.

**Q8. Answer: D**
Stemming applies rules like removing "-ing" or "-ed" (e.g., "running" → "run"), sometimes producing non-words ("studies" → "studi"). Lemmatization uses morphological analysis to produce valid words ("studies" → "study").

**Q9. Answer: D**
BM25 applies a logarithmic saturation function to term frequency (additional occurrences have diminishing impact) and normalizes by document length, making it more robust than raw TF-IDF.

**Q10. Answer: B**
Lexicon-based methods assign fixed sentiment to individual words, missing contextual modifiers like negation ("not good" = negative, not positive), sarcasm ("oh great, another delay"), and domain-specific meaning.

**Q11. Answer: D**
POS tagging labels each word with its grammatical role in context. The same word can have different POS tags in different sentences (e.g., "run" as a verb vs. noun).

**Q12. Answer: C**
Dependency parsing creates a tree structure showing how words relate grammatically. For example, in "The cat sat on the mat," "cat" is the subject of "sat" and "mat" is the object of the preposition "on."

**Q13. Answer: A**
Multi-label classification uses independent sigmoids so each label gets its own probability. Softmax forces probabilities to sum to 1 (mutually exclusive classes), which is incorrect for multi-label settings.

**Q14. Answer: C**
Sparse retrieval excels at exact keyword matching (e.g., product IDs, technical terms), while dense retrieval captures semantic similarity (e.g., "car" matching "automobile"). Combining them yields superior results.

**Q15. Answer: C**
In NER, most tokens are "O" (not entities). A model predicting "O" for everything would achieve >90% accuracy but 0% entity detection. F1 measures precision and recall on actual entities.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

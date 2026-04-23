# Multiple Choice Questions: Recommendation Systems

Test your understanding of recommendation system concepts for AI/ML interviews.

---

**Q1. Collaborative filtering recommends items based on:**

A) Item features and descriptions
B) Patterns of similar user behavior — users who agreed in the past will likely agree in the future
C) Random selection
D) Alphabetical ordering

---

**Q2. Item-based collaborative filtering is often preferred over user-based because:**

A) It requires more computation
B) Item similarities are more stable than user similarities and can be precomputed for scalability
C) It ignores all user data
D) It only works for movies

---

**Q3. Matrix factorization decomposes the user-item rating matrix into:**

A) A single vector
B) Two lower-rank matrices (user factors and item factors) whose product approximates the original ratings
C) A decision tree
D) A neural network

---

**Q4. The cold start problem refers to:**

A) The system running too slowly
B) Difficulty making recommendations for new users (no history) or new items (no ratings)
C) The model overfitting
D) The database being empty

---

**Q5. Content-based filtering recommends items by:**

A) Matching users with similar behavior
B) Comparing item features to a user's preference profile built from their past interactions
C) Random sampling
D) Popularity ranking only

---

**Q6. Implicit feedback (clicks, views, time spent) differs from explicit feedback (ratings) because:**

A) Implicit has clearer preference signals
B) Implicit lacks true negative signals — non-interaction may mean disinterest OR unawareness
C) Explicit is always more abundant
D) They are identical

---

**Q7. NDCG (Normalized Discounted Cumulative Gain) is preferred over Precision@K because:**

A) NDCG is simpler to compute
B) NDCG accounts for the position of relevant items in the ranking, giving more credit to top-ranked correct items
C) Precision@K considers position
D) NDCG ignores ranking order

---

**Q8. The exploration-exploitation trade-off in recommendations refers to:**

A) Showing only popular items
B) Balancing recommending items the system is confident the user will like (exploit) vs. items that help learn user preferences (explore)
C) Removing all personalization
D) Only showing new items

---

**Q9. Hybrid recommendation systems combine:**

A) Only collaborative filtering variants
B) Multiple approaches (collaborative, content-based, knowledge-based) to leverage their complementary strengths
C) Only rule-based methods
D) Only deep learning methods

---

**Q10. Deep learning approaches like Neural Collaborative Filtering (NCF) improve over matrix factorization by:**

A) Using simpler linear models
B) Learning nonlinear user-item interactions through neural network layers instead of only dot products
C) Removing all latent factors
D) Using only item features

---

**Q11. The Wide & Deep model combines:**

A) Only wide features
B) A wide (linear) component for memorization of feature combinations and a deep (neural) component for generalization
C) Only deep features
D) Random forests and SVMs

---

**Q12. MRR (Mean Reciprocal Rank) measures:**

A) The total number of recommendations
B) On average, how highly ranked the first relevant item is (1/rank of first correct result)
C) The total number of users
D) Only recall

---

**Q13. A two-tower architecture for recommendations uses:**

A) A single model for everything
B) Separate neural networks for users and items that produce embeddings, with recommendations based on embedding similarity
C) Only collaborative filtering
D) Only content-based methods

---

**Q14. Popularity bias in recommendations means:**

A) Popular items are never recommended
B) The system disproportionately recommends popular items, reducing exposure for niche items (the "long tail")
C) All items get equal exposure
D) Unpopular items are most recommended

---

**Q15. ALS (Alternating Least Squares) for matrix factorization handles sparsity by:**

A) Filling in all missing values with zeros
B) Alternately fixing one factor matrix and solving for the other via least squares, gracefully handling unobserved ratings
C) Ignoring all observed ratings
D) Using gradient descent on dense matrices only

---

## Answer Key

**Q1. Answer: B**
Collaborative filtering finds users with similar rating patterns and recommends items liked by similar users but not yet seen by the target user.

**Q2. Answer: B**
Item-item similarities change less frequently than user-user similarities (users' tastes evolve). Pre-computing item similarities enables efficient real-time recommendations at scale.

**Q3. Answer: B**
Matrix factorization learns latent factors: R ≈ U × Vᵀ, where U captures user preferences and V captures item characteristics in a low-dimensional space (k dimensions).

**Q4. Answer: B**
New users lack rating history for collaborative filtering, and new items have no ratings. Solutions include content-based fallback, demographic recommendations, and active learning.

**Q5. Answer: B**
Content-based filtering builds a profile from item features the user has liked and recommends items with similar features, working even for new items with known features.

**Q6. Answer: B**
With implicit feedback, a user not clicking an item could mean they don't like it or simply haven't seen it. This ambiguity requires specialized algorithms like weighted matrix factorization.

**Q7. Answer: B**
NDCG discounts relevance by position: DCG = Σ relevance/log₂(rank+1). A relevant item at rank 1 contributes more than one at rank 10, reflecting that users primarily see top results.

**Q8. Answer: B**
Pure exploitation recommends "safe" items but misses preferences. Exploration shows diverse items to learn more. Balancing this is critical for long-term recommendation quality.

**Q9. Answer: B**
Hybrid systems use collaborative filtering for established users, content-based for new items, and knowledge-based when explicit preferences exist, combining strengths and mitigating weaknesses.

**Q10. Answer: B**
NCF replaces the dot product of matrix factorization with neural network layers, capturing complex nonlinear interactions between user and item embeddings.

**Q11. Answer: B**
The wide component memorizes specific feature crosses (e.g., "user X liked item Y"), while the deep component generalizes to unseen combinations through learned embeddings.

**Q12. Answer: B**
MRR averages 1/rank of the first relevant result. If the first correct recommendation is at position 3, its reciprocal rank is 1/3. Higher MRR means relevant items appear earlier.

**Q13. Answer: B**
Two-tower models separately encode users and items into embeddings. At inference, candidate items are ranked by embedding similarity (dot product) with the user embedding, enabling efficient retrieval.

**Q14. Answer: B**
Systems trained on interaction data naturally favor popular items (more training signal). This reduces discovery of niche items, hurting diversity and potentially user satisfaction.

**Q15. Answer: B**
ALS iteratively fixes U and solves for V, then fixes V and solves for U. Each step is a least squares problem that only involves observed ratings, naturally handling sparsity.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

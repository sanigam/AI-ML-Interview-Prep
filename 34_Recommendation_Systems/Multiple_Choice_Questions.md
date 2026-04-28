# Multiple Choice Questions: Recommendation Systems

📺 **Video Lecture:** https://youtu.be/8-Y8X7ZOD2s


Test your understanding of recommendation system concepts for AI/ML interviews.

---

**Q1. Collaborative filtering recommends items based on:**

A) Item features and descriptions  
B) Alphabetical ordering  
C) Patterns of similar user behavior — users who agreed in the past will likely agree in the future  
D) Random selection

---

**Q2. Item-based collaborative filtering is often preferred over user-based because:**

A) It ignores all user data  
B) It requires more computation  
C) It only works for movies  
D) Item similarities are more stable than user similarities and can be precomputed for scalability

---

**Q3. Matrix factorization decomposes the user-item rating matrix into:**

A) A decision tree  
B) Two lower-rank matrices (user factors and item factors) whose product approximates the original ratings  
C) A single vector  
D) A neural network

---

**Q4. The cold start problem refers to:**

A) The system running too slowly  
B) The database being empty  
C) Difficulty making recommendations for new users (no history) or new items (no ratings)  
D) The model overfitting

---

**Q5. Content-based filtering recommends items by:**

A) Random sampling  
B) Popularity ranking only  
C) Matching users with similar behavior  
D) Comparing item features to a user's preference profile built from their past interactions

---

**Q6. Implicit feedback (clicks, views, time spent) differs from explicit feedback (ratings) because:**

A) They are identical  
B) Implicit lacks true negative signals — non-interaction may mean disinterest OR unawareness  
C) Implicit has clearer preference signals  
D) Explicit is always more abundant

---

**Q7. NDCG (Normalized Discounted Cumulative Gain) is preferred over Precision@K because:**

A) Precision@K considers position  
B) NDCG accounts for the position of relevant items in the ranking, giving more credit to top-ranked correct items  
C) NDCG ignores ranking order  
D) NDCG is simpler to compute

---

**Q8. The exploration-exploitation trade-off in recommendations refers to:**

A) Balancing recommending items the system is confident the user will like (exploit) vs. items that help learn user preferences (explore)  
B) Showing only popular items  
C) Removing all personalization  
D) Only showing new items

---

**Q9. Hybrid recommendation systems combine:**

A) Only collaborative filtering variants  
B) Multiple approaches (collaborative, content-based, knowledge-based) to leverage their complementary strengths  
C) Only deep learning methods  
D) Only rule-based methods

---

**Q10. Deep learning approaches like Neural Collaborative Filtering (NCF) improve over matrix factorization by:**

A) Removing all latent factors  
B) Learning nonlinear user-item interactions through neural network layers instead of only dot products  
C) Using only item features  
D) Using simpler linear models

---

**Q11. The Wide & Deep model combines:**

A) Random forests and SVMs  
B) Only deep features  
C) Only wide features  
D) A wide (linear) component for memorization of feature combinations and a deep (neural) component for generalization

---

**Q12. MRR (Mean Reciprocal Rank) measures:**

A) Only recall  
B) On average, how highly ranked the first relevant item is (1/rank of first correct result)  
C) The total number of users  
D) The total number of recommendations

---

**Q13. A two-tower architecture for recommendations uses:**

A) Separate neural networks for users and items that produce embeddings, with recommendations based on embedding similarity  
B) Only content-based methods  
C) Only collaborative filtering  
D) A single model for everything

---

**Q14. Popularity bias in recommendations means:**

A) The system disproportionately recommends popular items, reducing exposure for niche items (the "long tail")  
B) All items get equal exposure  
C) Unpopular items are most recommended  
D) Popular items are never recommended

---

**Q15. ALS (Alternating Least Squares) for matrix factorization handles sparsity by:**

A) Ignoring all observed ratings  
B) Filling in all missing values with zeros  
C) Alternately fixing one factor matrix and solving for the other via least squares, gracefully handling unobserved ratings  
D) Using gradient descent on dense matrices only

---

## Answer Key

**Q1. Answer: C**
Collaborative filtering finds users with similar rating patterns and recommends items liked by similar users but not yet seen by the target user.

**Q2. Answer: D**
Item-item similarities change less frequently than user-user similarities (users' tastes evolve). Pre-computing item similarities enables efficient real-time recommendations at scale.

**Q3. Answer: B**
Matrix factorization learns latent factors: R ≈ U × Vᵀ, where U captures user preferences and V captures item characteristics in a low-dimensional space (k dimensions).

**Q4. Answer: C**
New users lack rating history for collaborative filtering, and new items have no ratings. Solutions include content-based fallback, demographic recommendations, and active learning.

**Q5. Answer: D**
Content-based filtering builds a profile from item features the user has liked and recommends items with similar features, working even for new items with known features.

**Q6. Answer: B**
With implicit feedback, a user not clicking an item could mean they don't like it or simply haven't seen it. This ambiguity requires specialized algorithms like weighted matrix factorization.

**Q7. Answer: B**
NDCG discounts relevance by position: DCG = Σ relevance/log₂(rank+1). A relevant item at rank 1 contributes more than one at rank 10, reflecting that users primarily see top results.

**Q8. Answer: A**
Pure exploitation recommends "safe" items but misses preferences. Exploration shows diverse items to learn more. Balancing this is critical for long-term recommendation quality.

**Q9. Answer: B**
Hybrid systems use collaborative filtering for established users, content-based for new items, and knowledge-based when explicit preferences exist, combining strengths and mitigating weaknesses.

**Q10. Answer: B**
NCF replaces the dot product of matrix factorization with neural network layers, capturing complex nonlinear interactions between user and item embeddings.

**Q11. Answer: D**
The wide component memorizes specific feature crosses (e.g., "user X liked item Y"), while the deep component generalizes to unseen combinations through learned embeddings.

**Q12. Answer: B**
MRR averages 1/rank of the first relevant result. If the first correct recommendation is at position 3, its reciprocal rank is 1/3. Higher MRR means relevant items appear earlier.

**Q13. Answer: A**
Two-tower models separately encode users and items into embeddings. At inference, candidate items are ranked by embedding similarity (dot product) with the user embedding, enabling efficient retrieval.

**Q14. Answer: A**
Systems trained on interaction data naturally favor popular items (more training signal). This reduces discovery of niche items, hurting diversity and potentially user satisfaction.

**Q15. Answer: C**
ALS iteratively fixes U and solves for V, then fixes V and solves for U. Each step is a least squares problem that only involves observed ratings, naturally handling sparsity.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

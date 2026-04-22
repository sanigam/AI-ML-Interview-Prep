# Multiple Choice Questions: Dimensionality Reduction

📺 **Video Lecture:** https://youtu.be/n3xBHnBuZHQ


Test your understanding of PCA, t-SNE, and other dimensionality reduction techniques.

---

**Q1. The primary goal of dimensionality reduction is to:**

A) Increase the number of features for better accuracy
B) Reduce the number of features while preserving important information or structure
C) Remove all features except one
D) Convert numerical features to categorical

---

**Q2. In PCA, the first principal component captures:**

A) The direction of least variance in the data
B) The direction of maximum variance in the data
C) The mean of all features
D) A random direction

---

**Q3. PCA components are orthogonal (perpendicular) to each other because:**

A) This is required by the algorithm's random initialization
B) They are eigenvectors of the symmetric covariance matrix, which are always orthogonal
C) The user specifies them to be orthogonal
D) They represent the same direction

---

**Q4. The explained variance ratio in PCA tells you:**

A) The total number of features
B) The proportion of total variance captured by each principal component
C) The accuracy of the model
D) The number of outliers

---

**Q5. t-SNE is primarily used for:**

A) Feature selection for linear models
B) High-dimensional data visualization in 2D or 3D, preserving local structure
C) Training neural networks
D) Computing exact distances between all points

---

**Q6. A key limitation of t-SNE compared to PCA is:**

A) t-SNE is linear
B) t-SNE is non-deterministic, computationally expensive, and cannot transform new data points
C) t-SNE always produces the same result
D) t-SNE only works with 2 features

---

**Q7. UMAP differs from t-SNE by:**

A) Being a linear method
B) Being faster, preserving more global structure, and supporting transformation of new points
C) Requiring labeled data
D) Only working with categorical features

---

**Q8. Before applying PCA, data should be:**

A) Log-transformed
B) Standardized (zero mean, unit variance) so all features contribute equally
C) Converted to categorical
D) Sorted by value

---

**Q9. The "scree plot" in PCA shows:**

A) The original data distribution
B) Eigenvalues (or explained variance) for each component, helping choose how many to keep
C) The correlation between features
D) Training vs. test accuracy

---

**Q10. PCA can be used for noise reduction because:**

A) It removes all features
B) Lower-variance components often capture noise, so keeping only top components removes noise
C) It adds smoothing to the data
D) It replaces all values with their mean

---

**Q11. Linear Discriminant Analysis (LDA) differs from PCA because LDA:**

A) Is unsupervised
B) Maximizes class separability (uses labels) rather than total variance
C) Cannot reduce dimensions
D) Only works with 2 features

---

**Q12. The perplexity parameter in t-SNE roughly controls:**

A) The number of output dimensions
B) The effective number of neighbors considered for each point
C) The learning rate
D) The total variance explained

---

**Q13. Kernel PCA extends standard PCA by:**

A) Using fewer components
B) Applying the kernel trick to capture non-linear relationships in the data
C) Removing the need for eigenvalue decomposition
D) Only working with categorical data

---

**Q14. When choosing the number of PCA components, a common rule is to keep enough to explain:**

A) Exactly 50% of variance
B) 90-95% of total variance
C) 100% of variance (keep all components)
D) Less than 10% of variance

---

**Q15. Autoencoders can be viewed as a non-linear generalization of PCA because they:**

A) Use the exact same algorithm as PCA
B) Learn compressed representations (bottleneck) that capture important structure, using neural networks
C) Only reduce to exactly 2 dimensions
D) Require principal components as input

---

## Answer Key

**Q1. Answer: B**
Dimensionality reduction projects data to a lower-dimensional space while retaining meaningful patterns. This combats the curse of dimensionality, reduces noise, and speeds up downstream algorithms.

**Q2. Answer: B**
The first PC is the direction of maximum variance — the axis along which data is most spread out. Each subsequent PC captures the maximum remaining variance orthogonal to previous components.

**Q3. Answer: B**
PCA computes eigenvectors of the covariance matrix (which is symmetric). Eigenvectors of symmetric matrices are guaranteed to be orthogonal, ensuring PCA components are uncorrelated.

**Q4. Answer: B**
Explained variance ratio = eigenvalue_k / sum(all eigenvalues). It tells you what fraction of total information each component carries. The first few components typically capture most variance.

**Q5. Answer: B**
t-SNE excels at visualization by preserving local neighborhoods — similar points stay close in the 2D/3D embedding. It's not suitable for general dimensionality reduction for modeling.

**Q6. Answer: B**
t-SNE is stochastic (different runs give different results), O(n²) complexity, and has no parametric mapping for new points. PCA is deterministic, fast, and easily transforms new data.

**Q7. Answer: B**
UMAP is faster than t-SNE, preserves more global structure (not just local), and provides a parametric version that can transform unseen data. It has become the preferred alternative.

**Q8. Answer: B**
Without standardization, features with larger scales dominate the variance. Standardization ensures each feature contributes proportionally, so PCA captures meaningful directions.

**Q9. Answer: B**
A scree plot shows eigenvalues in descending order. The "elbow" (where values flatten) suggests how many components to retain — components before the elbow carry signal, those after carry mostly noise.

**Q10. Answer: B**
The top PCA components capture systematic variance (signal), while bottom components capture random fluctuation (noise). Reconstructing from only top components effectively denoises the data.

**Q11. Answer: B**
LDA is supervised — it finds projections that maximize between-class variance relative to within-class variance. PCA is unsupervised and maximizes total variance regardless of class labels.

**Q12. Answer: B**
Perplexity (typically 5-50) is related to the effective number of neighbors. Low perplexity emphasizes very local structure; high perplexity captures more global patterns.

**Q13. Answer: B**
Kernel PCA applies the kernel trick to compute PCA in a high-dimensional feature space, enabling discovery of non-linear structures that standard PCA misses.

**Q14. Answer: B**
Keeping components that explain 90-95% of variance retains most information while achieving meaningful dimensionality reduction. The exact threshold depends on the application.

**Q15. Answer: B**
Autoencoders learn encoder (compression) and decoder (reconstruction) networks. The bottleneck layer is a non-linear low-dimensional representation, generalizing PCA's linear projection.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

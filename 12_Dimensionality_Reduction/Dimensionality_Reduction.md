# Dimensionality Reduction

📺 **Video Lecture:** https://youtu.be/n3xBHnBuZHQ


## Interview Anchor
- **PCA (Principal Component Analysis):** Linear projection maximizing variance; finds orthogonal directions; unsupervised.
- **Manifold Learning:** Assumes data lies on a low-dimensional manifold; t-SNE, UMAP reveal structure; useful for visualization.
- **Feature Selection vs. Extraction:** Selection keeps original features; extraction creates new ones; choice depends on interpretability needs.

## Key Concepts Overview

Dimensionality reduction is a fundamental preprocessing technique addressing the curse of dimensionality—high-dimensional data leads to sparsity, overfitting, and computational inefficiency. This topic tests both mathematical understanding (PCA eigendecomposition, variance explained) and practical intuition (when to apply PCA, when to use t-SNE, scalability considerations). Interviewers assess whether you understand the difference between linear methods (PCA, LDA) and nonlinear ones (t-SNE, UMAP), and the trade-offs between preservation of global structure (PCA) versus local structure (t-SNE). Additionally, this topic touches on feature selection—a sometimes overlooked alternative to extraction that preserves interpretability. Demonstrating knowledge of modern methods like UMAP and understanding of manifold learning shows current awareness.

---

### Q1: Explain PCA (Principal Component Analysis) and its mathematical foundation.

**A:** PCA finds a set of orthogonal directions (principal components) that maximize data variance. Mathematically, given data X ∈ ℝ^{n×d} (n samples, d features), center the data and compute the covariance matrix Σ = X^T X / (n-1). The first principal component is the eigenvector of Σ with the largest eigenvalue λ₁; it defines the direction of maximum variance. Subsequent components are eigenvectors with decreasing eigenvalues, orthogonal to previous components. If X = UΣV^T is the SVD, the principal components are the columns of V (right singular vectors). PCA projects data onto the first k components: X_reduced = XV_k, where V_k contains the first k eigenvectors. Advantages: (1) unsupervised (no labels needed), (2) interpretable (components are linear combinations of original features), (3) fast O(d² × n) or O(d × n²) depending on algorithm, (4) optimal for Gaussian data under mean-squared-error sense, (5) reduces storage and computation. Disadvantages: (1) linear only (fails on nonlinear structure), (2) sensitive to scaling (must standardize features), (3) less interpretable if d is large (component is mixture of many features). PCA is a standard baseline; if it performs well, the problem may not need complex nonlinear methods. In interviews, explain the variance maximization intuition clearly—larger variance ↔ more information.

---

### Q2: Explain explained variance ratio, scree plot, and how to choose number of components.

**A:** Explained variance ratio for component k is λ_k / ∑_i λ_i, the proportion of total variance captured. Cumulative explained variance is the sum up to component k; e.g., if first 3 components explain 95%, they capture 95% of data variance. Scree plot displays variance (or cumulative variance) vs. component number; useful for choosing k. The "elbow" where variance gain flattens suggests the optimal k—additional components contribute little. Rule of thumb: keep 95-99% cumulative variance (more for downstream ML tasks, less for visualization). For visualization, 2-3 components are used regardless of variance explained, to plot data on screen. Mathematically, keeping k components reduces error: ||X - X_{reduced}||² = ∑_{i=k+1}^d λ_i (sum of discarded eigenvalues). Choosing k is a trade-off: larger k preserves more information but requires higher dimension. In practice: (1) plot scree curve, visually identify elbow, (2) compute cumulative variance and choose k for desired threshold (e.g., 95%), (3) try k values spanning the range and evaluate downstream task performance. For classification, less variance may suffice (80-90%); for unsupervised tasks, preserve more (95%+). Scree plots are often subjective—combine with downstream validation. In interviews, mention both methods and discuss domain-specific thresholds rather than claiming 95% is always optimal.

---

### Q3: Compare linear (PCA, LDA) and nonlinear (t-SNE, UMAP) dimensionality reduction.

**A:** Linear methods (PCA, LDA) find linear projections onto lower-dimensional subspaces. PCA maximizes variance (unsupervised); LDA maximizes class separability (supervised). Linear methods are fast O(d² × n), interpretable (components are feature combinations), and preserve global structure. Disadvantages: they fail on nonlinear manifolds—if data lie on a curved surface, linear projections distort geometry. Nonlinear methods (t-SNE, UMAP, kernel PCA) handle curved manifolds by learning nonlinear embeddings. t-SNE (t-Distributed Stochastic Neighbor Embedding) preserves local structure—neighbors in original space remain neighbors in embedding; excellent for visualization but computationally expensive O(n²) and doesn't preserve global structure. UMAP (Uniform Manifold Approximation and Projection) is faster O(n) with approximations, preserves both local and global structure better than t-SNE, and is more suitable for downstream ML tasks. Practical choice: (1) exploratory visualization: t-SNE (best visuals but slow), (2) fast visualization + downstream ML: UMAP, (3) preprocessing for ML: PCA (fast, interpretable) or UMAP if nonlinearity is suspected, (4) classification with interpretability: LDA. Combining methods: apply PCA first to ~50 dimensions (fast), then t-SNE/UMAP (reduces computation). Nonlinear methods aren't always better—if data is already mostly linear, PCA suffices. In interviews, emphasize trade-offs: linear fast/interpretable vs. nonlinear flexible but slower/harder to interpret.

---

### Q4: Explain t-SNE: algorithm, intuition, and when to use.

**A:** t-SNE converts high-dimensional Euclidean distances to probabilities reflecting similarity: p_ij ∝ exp(-||x_i - x_j||² / σ_i²), where σ_i is adapted per sample (perplexity parameter controls effective neighborhood size). In the embedding space, probabilities are computed using Student-t distribution: q_ij ∝ (1 + ||y_i - y_j||²)^(-1), which has heavier tails (preserves distant points). The cost function is KL divergence between p and q; minimizing it via gradient descent produces the embedding. Intuition: samples close in original space should be close in embedding (preserve local neighborhood); samples far apart should remain far. t-SNE excels at visualization—clusters separate visually, revealing structure invisible in original space. Disadvantages: (1) O(n²) time and memory, impractical for n > 100k, (2) non-convex optimization, sensitive to random seed (results vary), (3) perplexity tuning (typically 5-50; higher values preserve more global structure), (4) doesn't preserve global distances (cluster separation is artifact, not meaningful), (5) no straightforward way to embed new test data (must rerun on full data). Use t-SNE: (1) exploratory visualization, (2) understanding cluster structure, (3) checking for outliers. Don't use t-SNE: (1) as preprocessing for ML models (distances not preserved), (2) for large datasets (slow), (3) when interpretable dimensions are needed. In interviews, mention t-SNE's visualization superiority but emphasize its limitations—many practitioners misuse it as ML preprocessing, which is incorrect.

---

### Q5: Explain UMAP (Uniform Manifold Approximation and Projection) and its advantages over t-SNE.

**A:** UMAP is a manifold learning technique that constructs a graph representation of high-dimensional data and optimizes a low-dimensional embedding to preserve graph structure. Algorithm: (1) build k-nearest neighbor graph in original space, (2) convert to weighted fuzzy graph using membership strengths, (3) optimize low-dimensional embedding to preserve graph structure via cross-entropy loss. UMAP is faster O(n log n) with approximations, scales to larger datasets (n > 100k), and preserves both local and global structure better than t-SNE. Key advantages: (1) faster than t-SNE (minutes vs. hours for large data), (2) better preservation of global structure (cluster positions are meaningful, not artifacts), (3) more stable (less random seed sensitivity), (4) hyperparameters more intuitive (n_neighbors controls locality, min_dist controls minimum spread), (5) supports custom metrics (not just Euclidean). Disadvantages: (1) still O(n²) worst-case for graph construction (though approximations reduce this), (2) less visually striking than t-SNE (less separation), (3) less theoretical justification than PCA (manifold learning is heuristic). Use UMAP: (1) exploratory visualization when t-SNE is too slow, (2) preprocessing for downstream ML (preserves more information than t-SNE), (3) handling large datasets. UMAP is increasingly the default for visualization—combines speed and quality. In interviews, positioning UMAP as a modern improvement over t-SNE shows current knowledge. Mention that UMAP can be used for both visualization and preprocessing, unlike t-SNE.

---

### Q6: Explain Linear Discriminant Analysis (LDA) and its relationship to PCA.

**A:** LDA is a supervised dimensionality reduction finding projections that maximize class separability. Unlike PCA (maximizing variance), LDA maximizes the ratio of between-class variance to within-class variance: J(w) = (w^T S_B w) / (w^T S_W w), where S_B is between-class scatter and S_W is within-class scatter. The optimal projection directions are generalized eigenvectors of S_B and S_W. For K classes, LDA yields at most K-1 discriminant components (one fewer than classes). Advantages: (1) supervised—uses class information to find discriminative directions, (2) often better for classification than unsupervised PCA, (3) interpretable (linear projections), (4) fast. Disadvantages: (1) assumes Gaussian class distributions (similar covariance structure), (2) limited to K-1 dimensions (ineffective if many classes), (3) not applicable if more classes than samples (singular S_W). LDA vs. PCA: PCA is unsupervised, maximizes total variance; LDA is supervised, maximizes class separability. PCA useful for general dimensionality reduction; LDA for classification. In practice: try PCA first (unsupervised, no class labels needed), then LDA if class info improves performance. LDA is less popular today (neural networks dominate), but valuable for interpretability. In interviews, mentioning LDA as a supervised alternative to PCA demonstrates understanding of both methods' trade-offs.

---

### Q7: Explain autoencoders for dimensionality reduction and their advantages.

**A:** Autoencoders are neural networks learning compressed representations. Architecture: encoder compresses input x to latent code z = f_enc(x) (usually low-dimensional), decoder reconstructs x̂ = f_dec(z). Training minimizes reconstruction error ||x - x̂||², forcing the latent code to capture essential information. Advantages: (1) nonlinear (handles complex structure), (2) flexible architecture (can specify exact bottleneck dimension), (3) can incorporate constraints (e.g., variational autoencoder adds KL regularization for smooth latent space), (4) scalable with SGD and GPUs, (5) can be fine-tuned for downstream tasks. Disadvantages: (1) requires training (unlike PCA's closed-form solution), (2) hyperparameter-heavy (architecture, learning rate, regularization), (3) less interpretable than PCA (learned representations are opaque), (4) can memorize input (undercomplete autoencoders still reconstruct perfectly if bottleneck allows). Variants: (1) Variational Autoencoder (VAE) adds KL divergence regularizing latent distribution, useful for generative modeling, (2) Denoising Autoencoder adds noise to input, improving robustness. Autoencoders are used when: (1) nonlinearity is essential, (2) data is complex (images, audio), (3) interpretability isn't critical, (4) you have substantial data for training. For tabular data with linear structure, PCA is often preferable (simpler, faster, interpretable). In interviews, autoencoders show awareness of deep learning for representation learning. Distinguish them from PCA: PCA is linear and closed-form, autoencoders are nonlinear and learned—different tools for different problems.

---

### Q8: Explain the curse of dimensionality and why dimensionality reduction helps.

**A:** The curse of dimensionality describes problems arising in high-dimensional spaces. Key issues: (1) Volume scales exponentially: ℝ^d has exponentially larger volume, making data sparser—neighborhoods become huge, distance metrics lose meaning, (2) Overfitting: high-dimensional feature spaces allow models to memorize training data; VC dimension grows, requiring exponentially more training samples to achieve same generalization, (3) Computational cost: algorithms scale with d (or d²); training/inference slow, memory intensive, (4) Noise dominates: in high dimensions, noise features become proportionally more important relative to signal. Dimensionality reduction helps by: (1) reducing noise (assuming signal lies in low-dimensional subspace), (2) improving generalization (fewer features ↔ lower VC dimension), (3) reducing computation, (4) enabling visualization, (5) concentrating information in fewer dimensions. Practical example: text data (d = 10k+ words) has only ~100-1000 relevant dimensions; applying dimensionality reduction (PCA, embeddings) drastically improves downstream models. Johnson-Lindenstrauss lemma formalizes this: n points in ℝ^d can be embedded in ℝ^k (k = O(log n / ε²)) preserving distances within ε. In interviews, explaining curse of dimensionality and how dimensionality reduction addresses it shows understanding of a fundamental challenge. Mention that not all high-dimensional problems suffer equally—sparse data (text) and dense data (images) have different challenges.

---

### Q9: Explain kernel PCA and its advantages over linear PCA.

**A:** Kernel PCA (KPCA) extends PCA to nonlinear structure by implicitly mapping data to a high-dimensional space φ(x) via a kernel k, then applying PCA. Algorithm: (1) compute Gram matrix K where K_ij = k(x_i, x_j), (2) center K in feature space (algebraically involved), (3) eigen-decompose centered K, (4) first k eigenvectors give principal components in feature space. KPCA can discover nonlinear structure that linear PCA misses—e.g., concentric circles, S-curves. Advantages: (1) nonlinear (captures curved structure), (2) kernel trick (efficient, no explicit φ computation), (3) interpretable via kernel choice (RBF for locality, polynomial for interactions). Disadvantages: (1) O(n²) memory for Gram matrix (prohibitive for large n), (2) hyperparameter tuning (kernel, γ for RBF), (3) projection of test data is expensive (requires kernel evaluation with all training data), (4) less interpretable than linear PCA (no feature combinations). Compared to PCA: PCA fast and interpretable; KPCA captures nonlinearity but slower. Modern alternatives: t-SNE, UMAP, autoencoders all capture nonlinearity more effectively. KPCA is elegant theoretically but rarely used in practice due to O(n²) scaling. In interviews, KPCA shows awareness of kernel methods extending linear techniques. Mention it as a historical approach; modern practitioners prefer neural networks or UMAP for nonlinear reduction.

---

### Q10: Explain factor analysis and its relationship to PCA.

**A:** Factor analysis (FA) is a probabilistic model assuming data is generated from latent factors z: x = Wz + μ + ε, where W are factor loadings, z ∈ ℝ^k are latent factors, ε is noise ~ N(0, Σ). Unlike PCA (deterministic projection), FA is probabilistic: x ~ N(μ, WW^T + Σ). FA models data covariance via factors and noise, enabling likelihood-based inference. Fitting uses EM algorithm, maximizing likelihood. Advantages: (1) probabilistic framework (likelihood-based model selection, uncertainty quantification), (2) noise model (ε explicit, unlike PCA), (3) interpretable (factors are latent sources of variation), (4) handles missing data (EM naturally accommodates). Disadvantages: (1) assumes linear generative model (similar limitation to PCA), (2) more complex than PCA (requires iterative EM fitting), (3) hyperparameter tuning (number of factors k, noise model), (4) slower than PCA. FA vs. PCA: PCA deterministic, fast, maximizes variance; FA probabilistic, slower, explains covariance via latent factors. PCA finds best projection; FA finds best latent variable model. Use FA when: (1) probabilistic framework is needed, (2) explicit noise modeling is valuable, (3) missing data must be handled, (4) likelihood comparison for model selection (AIC/BIC). For large-scale dimensionality reduction, PCA is preferred (faster, simpler). In interviews, FA demonstrates awareness of probabilistic approaches to dimensionality reduction. Mention it as an alternative when the data-generation perspective is valuable.

---

### Q11: Explain Independent Component Analysis (ICA) and its applications.

**A:** ICA assumes data x = As + n, where A is unknown mixing matrix, s are independent components (latent sources), n is noise. Unlike PCA (finds uncorrelated directions), ICA finds statistically independent directions. ICA applies when: (1) sources are non-Gaussian (Gaussian distributions are rotationally symmetric; independence can't be inferred), (2) you suspect independent underlying factors (e.g., independent sound sources, independent price drivers). Algorithm: fit x = As by maximizing non-Gaussianity of estimated s (via kurtosis, negentropy, mutual information). Applications: (1) blind source separation (cocktail party problem—extract individual speakers from mixed audio), (2) brain imaging (fMRI to identify independent brain networks), (3) financial data (independent price movements). Advantages: (1) finds independent sources (stronger assumption than uncorrelated), (2) applicable when PCA fails (non-Gaussian data). Disadvantages: (1) requires non-Gaussianity (doesn't work if s_i ~ Gaussian), (2) slower than PCA, (3) ambiguity in factor order and scaling (multiple valid solutions), (4) sensitive to noise. ICA vs. PCA: PCA finds uncorrelated directions (second-moment), ICA finds independent directions (higher-moment information). ICA is specialized—use only when independent sources are plausible. In practice, ICA is less common than PCA; mention it for signal processing applications. In interviews, knowing ICA differentiates you from basic practitioners; it's rarely asked but signals depth if mentioned appropriately.

---

### Q12: Explain feature selection vs. feature extraction and when to use each.

**A:** Feature selection keeps original features, removing irrelevant/redundant ones. Methods: (1) filter (univariate): rank features by correlation/information gain with target, select top k, (2) wrapper: evaluate feature subsets via model performance, select best subset, (3) embedded: regularization (L1/Lasso) automatically selects features. Feature extraction creates new features (linear: PCA; nonlinear: autoencoders). Trade-offs: (1) Selection preserves interpretability (original features); extraction loses it (new features are combinations), (2) Selection is fast (no learning); extraction requires fitting, (3) Selection is prone to discarding information; extraction captures structure more fully, (4) Selection handles categorical features naturally; extraction often requires encoding. When to use: (1) Selection: tabular data, high-dimensional sparse data (text), need interpretability, features have clear meaning, (2) Extraction: dense data (images), nonlinear structure, don't need interpretability, data is unlabeled. Practical strategy: (1) start with selection (fast baseline), (2) if performance plateaus, try extraction (more flexible). Combined approach: selection to reduce to ~100 features, then PCA/extraction on those. In interviews, discussing both options shows awareness; many practitioners use only PCA. Mention that for text (high-dimensional sparse), selection may outperform extraction due to high dimensionality curse and sparsity—extraction doesn't help sparse data much. Context-dependent decision-making impresses interviewers.

---

### Q13: What is the Johnson-Lindenstrauss lemma and its implications for dimensionality reduction?

**A:** Johnson-Lindenstrauss lemma states: for any set of n points in ℝ^d and ε > 0, there exists a linear projection to ℝ^k with k = O(log n / ε²) such that all pairwise distances are preserved within factor (1±ε). Implications: (1) any d-dimensional data can be reduced to k = O(log n / ε²) dimensions with small distortion, (2) number of target dimensions depends only on n (number of points), not d (original dimension), (3) surprising: even if d is huge, k is small for moderate n. Example: n = 1000 points, ε = 0.1 → k ≈ 50 dimensions suffice to preserve distances. This provides theoretical justification for random projection (random k × d matrix projects data while preserving distances), a fast approximation to PCA. Practical implications: (1) dimensionality reduction is fundamentally achievable—no information loss if k is chosen wisely, (2) curse of dimensionality isn't absolute; structure (low intrinsic dimension) enables reduction, (3) random projections scale better than PCA for very large d. In interviews, mentioning JL lemma shows theoretical grounding. It's not commonly asked but demonstrates deep understanding. Use it to justify why dimensionality reduction works: "Johnson-Lindenstrauss lemma guarantees that if intrinsic dimension is low, we can reduce to ~log(n) dimensions."

---

### Q14: Explain random projections and Gaussian random projection for efficient dimensionality reduction.

**A:** Random projection is a simple, fast approximation to PCA. Algorithm: generate a random k × d matrix R (often Gaussian or sparse random) and project: X_reduced = X @ R. Despite randomness, Johnson-Lindenstrauss guarantees distances are preserved if k = O(log n / ε²). Advantages: (1) extremely fast O(kd), no eigendecomposition, (2) scalable to large d (sparse random matrices), (3) memory-efficient, (4) parallelizable, (5) theoretical guarantees on distance preservation. Disadvantages: (1) doesn't maximize variance like PCA (may waste dimensions), (2) random variance—different random matrices yield different results, (3) less interpretable (features are random combinations), (4) requires k to be fairly large (O(log n) can still be large for very large n). Variants: (1) Gaussian random projection: R ~ N(0, 1), densest but most general, (2) sparse random projection: R has few non-zero entries, faster for sparse data, (3) structured random projection (Hadamard, discrete cosine): enable FFT-like fast computation. When to use: (1) very large d (d > 100k), (2) need speed over optimality, (3) incremental/streaming data (no need to recompute on new data), (4) random baseline. Comparison: PCA O(d²n) computation, better preserved variance; random projection O(kd) computation, approximate variance preservation. In practice, for moderate d, PCA is better; for extreme d, random projection wins. In interviews, random projection is underappreciated; mentioning it for ultra-high-dimensional problems shows practical knowledge.

---

### Q15: How would you decide which dimensionality reduction method to use for a given problem?

**A:** Decision factors: (1) **Problem type**: Unsupervised (PCA, UMAP) vs. supervised (LDA), visualization (t-SNE, UMAP) vs. preprocessing (PCA, random projection), (2) **Data structure**: Linear (PCA), nonlinear (t-SNE, UMAP, autoencoders), independent components (ICA), (3) **Dataset size**: Small (all methods), large n (avoid O(n²) methods; use PCA, random projection, UMAP with approximations), large d (random projection, sparse PCA), (4) **Interpretability**: Need interpretable features (selection, PCA), don't care (t-SNE, autoencoders), (5) **Downstream task**: Visualize clusters (t-SNE, UMAP), improve ML model (PCA, feature selection, autoencoders), (6) **Computational budget**: Fast (random projection, PCA), slow acceptable (t-SNE, ICA, autoencoders). Practical workflow: (1) **Baseline**: Apply PCA, check if 95% variance in 10-50 dimensions—if yes, problem may be nearly linear, (2) **Visualization**: Use UMAP (fast, preserves structure), or t-SNE if speed isn't critical, (3) **Classification preprocessing**: Use PCA or LDA if supervised; if nonlinear, try autoencoders, (4) **Anomaly detection**: UMAP or PCA; examine reconstruction error, (5) **If PCA underperforms**: Try UMAP (nonlinear) or feature engineering, (6) **Very large d (text, genomics)**: Feature selection (filter/L1) first, then PCA. Red flags: (1) t-SNE used for preprocessing (wrong—only for visualization), (2) ignoring explained variance (must check), (3) no scaling before PCA. In interviews, this decision-tree approach impresses. Avoid saying "always use PCA" or "UMAP is best"—context-dependent reasoning demonstrates expertise. Mention combining methods: selection → PCA → downstream model, or embedding → UMAP → clustering.

---

## Interview Cheatsheet

**Key Terms:**

- **PCA:** Finds orthogonal directions maximizing variance; linear, unsupervised, fast, closed-form solution.
- **Explained Variance Ratio:** λ_k / ∑λ_i; cumulative variance guides choosing number of components.
- **Scree Plot:** Variance vs. component number; elbow suggests optimal k; visual tool.
- **LDA (Linear Discriminant Analysis):** Supervised; maximizes between-class variance relative to within-class variance.
- **t-SNE:** Preserves local neighborhood structure; excellent visualization; O(n²) computation, nonconvex.
- **UMAP:** Fast nonlinear reduction; preserves local+global structure; scalable to large datasets.
- **Autoencoder:** Neural network learning compressed latent representation via reconstruction loss.
- **Kernel PCA:** Nonlinear PCA via kernel trick; requires O(n²) Gram matrix; captures manifold structure.
- **Factor Analysis:** Probabilistic model: x = Wz + ε; unsupervised, handles missing data, EM-fitted.
- **ICA (Independent Component Analysis):** Finds statistically independent sources; non-Gaussian assumption.
- **Feature Selection:** Removes irrelevant/redundant original features; preserves interpretability.
- **Feature Extraction:** Creates new features via transformation; captures structure, less interpretable.
- **Curse of Dimensionality:** High-dimensional spaces are sparse, overfitting-prone, computationally expensive.
- **Johnson-Lindenstrauss Lemma:** n points reducible to O(log n / ε²) dimensions with distance preservation.
- **Random Projection:** Fast O(kd) approximation to PCA; distances preserved by JL lemma.
- **Manifold Learning:** Assumes data lies on low-dimensional curved surface; nonlinear methods reveal it.

**Rapid-Fire Q&A:**

- **Q:** PCA sensitive to what? **A:** Feature scaling—features with large magnitude dominate; always standardize.
- **Q:** PCA preserves what property? **A:** Variance; first components capture directions of maximum variance.
- **Q:** How many components in PCA for 95% variance? **A:** Depends on data; check scree plot or cumulative variance.
- **Q:** t-SNE vs. UMAP: which is faster? **A:** UMAP O(n log n) with approximations; t-SNE O(n²); UMAP much faster for large n.
- **Q:** t-SNE cluster positions meaningful? **A:** No; distances/separation are artifacts of optimization, not real; for visualization only.
- **Q:** LDA limits on components? **A:** At most K-1 components for K classes; ineffective if many classes.
- **Q:** Autoencoder vs. PCA: nonlinear? **A:** Autoencoder is nonlinear; PCA is linear; trade-offs: speed/interpretability vs. flexibility.
- **Q:** Feature selection or extraction for sparse text? **A:** Selection (L1, filter); extraction wastes computation on high-dimensional sparse space.
- **Q:** Kernel PCA scalability? **A:** O(n²) memory/time; impractical for large n; UMAP or random projection better.
- **Q:** ICA when useful? **A:** When independent sources suspected (audio, finance); requires non-Gaussian sources.
- **Q:** Random projection vs. PCA? **A:** Random faster O(kd), PCA optimal variance O(d²n); trade speed vs. optimality.
- **Q:** Factor analysis assumes? **A:** Linear generative model x = Wz + ε; probabilistic, Gaussian latent/noise.
- **Q:** Johnson-Lindenstrauss implication? **A:** Data reducible to O(log n) dimensions with small distance distortion; justifies reduction.
- **Q:** Curse of dimensionality main issue? **A:** Sparsity, overfitting, computational cost; solved by reduction or focusing on intrinsic dimension.
- **Q:** Always scale before dimensionality reduction? **A:** Essentials for PCA, LDA, distance-based methods; not for tree-based models.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

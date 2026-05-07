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

**A:** PCA finds a set of orthogonal directions — **principal components** — that maximize the variance of the projected data.

**Mathematical setup.** Given data X ∈ ℝⁿˣᵈ with n samples and d features, first center the data, then compute the sample covariance matrix:

```
Σ = Xᵀ·X / (n − 1)
```

The first principal component is the eigenvector of Σ with the largest eigenvalue λ₁ — the direction of maximum variance. Subsequent components are eigenvectors with decreasing eigenvalues, each orthogonal to the previous ones.

**Equivalently via SVD.** If the centered data has SVD X = U·Σ·Vᵀ, the principal components are the columns of V (the right singular vectors).

**Projection to k dimensions:**

```
X_reduced = X · V_k
```

where V_k contains the top k eigenvectors.

**Advantages:**

- Unsupervised — no labels needed.
- Interpretable — components are linear combinations of original features.
- Fast — closed-form solution, O(d²·n) or O(d·n²) depending on algorithm.
- Optimal in MSE sense for Gaussian data.
- Reduces storage and downstream computation.

**Disadvantages:**

- Linear — fails on nonlinear structure.
- Sensitive to feature scaling — must standardize first.
- Components become harder to interpret as d grows (each is a mixture of many features).

PCA is the natural baseline — if it works well, the problem may not need anything nonlinear. In interviews, the headline intuition is "directions of maximum variance," with the mantra *more variance = more information*.

---

### Q2: Explain explained variance ratio, scree plot, and how to choose number of components.

**A:** The **explained variance ratio** for component k is the fraction of total variance it captures:

```
explained_ratio(k) = λ_k / Σᵢ λᵢ
```

The **cumulative explained variance** is the sum up to component k:

```
cumulative(k) = (λ_1 + λ_2 + ... + λ_k) / Σᵢ λᵢ
```

So if the first 3 components have cumulative ratio 0.95, they capture 95% of the data's variance.

**Scree plot.** Variance (or cumulative variance) plotted against component index. The "elbow" — where the curve flattens — suggests where additional components stop helping much.

**Reconstruction error.** Keeping k components leaves a residual equal to the sum of the discarded eigenvalues:

```
|| X − X_reduced ||² = Σ_{i=k+1}^{d} λ_i
```

So picking k is a tradeoff: larger k preserves more information but uses more dimensions.

**Practical guidance:**

- Plot the scree curve and visually identify the elbow.
- Pick the smallest k that crosses a desired cumulative-variance threshold (typically 95%).
- For classification preprocessing, 80–90% variance is often enough.
- For visualization, k = 2 or 3 regardless of variance, just to fit on screen.

Scree-plot reading is somewhat subjective — combine it with downstream validation. In interviews, discuss the domain-specific threshold rather than claiming "95% always" — context-dependent reasoning is what stands out.

---

### Q3: Compare linear (PCA, LDA) and nonlinear (t-SNE, UMAP) dimensionality reduction.

**A:** **Linear methods** (PCA, LDA) find linear projections to lower-dimensional subspaces.

- *PCA* maximizes variance (unsupervised).
- *LDA* maximizes class separability (supervised).
- Fast — typically O(d² · n).
- Interpretable — each new dimension is a linear combination of original features.
- Preserve global structure but fail on nonlinear manifolds (curved data gets distorted).

**Nonlinear methods** (t-SNE, UMAP, kernel PCA, autoencoders) handle curved manifolds.

- **t-SNE** preserves *local* structure — points that are neighbors stay neighbors. Excellent for visualization but O(n²) and doesn't preserve global structure (cluster *positions* are not meaningful).
- **UMAP** is faster (close to O(n log n) with approximations), preserves both local and global structure better than t-SNE, and is more suitable as ML preprocessing.

**Practical choice:**

- *Exploratory visualization:* t-SNE (best visuals, slower).
- *Fast visualization + downstream ML:* UMAP.
- *ML preprocessing:* PCA first (fast, interpretable); UMAP if you suspect nonlinearity.
- *Supervised dimensionality reduction with interpretability:* LDA.

A common workflow: apply PCA first to reduce to ~50 dimensions, then t-SNE/UMAP for the final 2D/3D embedding.

Nonlinear methods aren't always better — if data is mostly linear, PCA suffices. In interviews, frame the tradeoff as *linear (fast, interpretable, global)* vs *nonlinear (flexible, slower, harder to interpret)*.

---

### Q4: Explain t-SNE: algorithm, intuition, and when to use.

**A:** **t-SNE** turns high-dimensional Euclidean distances into similarity probabilities, then matches them in a low-dimensional embedding.

**In the original space**, define conditional probabilities using Gaussian kernels:

```
p_{j|i}  ∝  exp( − || x_i − x_j ||² / (2 · σ_i²) )
```

The bandwidth σ_i is adapted per sample — controlled implicitly by the **perplexity** hyperparameter, which sets the effective neighborhood size. The symmetric joint probability is then defined as `p_{ij} = (p_{j|i} + p_{i|j}) / (2n)`.

**In the embedding space**, use a Student-t distribution (heavy tails) for the corresponding probabilities:

```
q_{ij}  ∝  ( 1 + || y_i − y_j ||² )⁻¹
```

The heavy tails are what preserve cluster separation in the embedding.

**Loss.** The KL divergence between the two distributions:

```
loss = KL( p || q ) = Σ_{ij} p_{ij} · log( p_{ij} / q_{ij} )
```

Minimized via gradient descent.

**Intuition.** Samples that are close in the original space should remain close in the embedding (preserve local neighborhoods); samples that are far should stay far. The result is excellent visual cluster separation.

**Limitations:**

- O(n²) time and memory — impractical for n > 100K (Barnes-Hut speedup helps but doesn't eliminate this).
- Non-convex; results vary with random seed.
- Perplexity is a tuning hyperparameter (typical range 5–50).
- **Cluster *positions* and *separations* are not meaningful** — only neighborhoods are.
- No clean way to embed new points without rerunning.

**Use t-SNE for:** exploratory visualization, understanding cluster structure, outlier detection.

**Don't use t-SNE for:** ML preprocessing (distances aren't preserved), large datasets, or when you need interpretable axes.

In interviews, the most valuable point is the misuse warning — many practitioners (incorrectly) treat t-SNE coordinates as ML features.

---

### Q5: Explain UMAP (Uniform Manifold Approximation and Projection) and its advantages over t-SNE.

**A:** **UMAP** is a manifold-learning technique that builds a graph in the high-dimensional space and optimizes a low-dimensional embedding to preserve that graph structure.

**Algorithm sketch:**

1. Build a k-nearest-neighbor graph in the original space.
2. Convert it to a weighted fuzzy graph using membership strengths.
3. Optimize a low-dimensional embedding to preserve the graph via cross-entropy loss.

UMAP runs in close to O(n log n) with approximations, scaling well to n > 100K.

**Advantages over t-SNE:**

- **Faster** — minutes instead of hours on large datasets.
- **Preserves global structure** — cluster positions are more meaningful, not just neighborhoods.
- **More stable** — less sensitive to random seed.
- **Intuitive hyperparameters** — `n_neighbors` controls locality, `min_dist` controls minimum embedding spread.
- **Supports custom metrics** — works with any distance function, not just Euclidean.

**Disadvantages:**

- Worst-case graph construction is still O(n²); approximations help but aren't free.
- Visually less striking than t-SNE — clusters are less aggressively separated.
- Less theoretical grounding than PCA (manifold learning is partly heuristic).

**Use UMAP for:**

- Exploratory visualization when t-SNE is too slow.
- ML preprocessing (it preserves more usable information than t-SNE).
- Larger datasets where t-SNE's O(n²) is prohibitive.

UMAP is increasingly the default for both visualization and preprocessing. In interviews, positioning it as the modern improvement over t-SNE — *and* one that can serve as ML preprocessing (unlike t-SNE) — shows current practical knowledge.

---

### Q6: Explain Linear Discriminant Analysis (LDA) and its relationship to PCA.

**A:** **LDA** is a supervised dimensionality-reduction method that finds projections maximizing class separability. Unlike PCA (which maximizes total variance), LDA maximizes the ratio of *between-class* to *within-class* scatter:

```
J(w) = (wᵀ · S_B · w) / (wᵀ · S_W · w)
```

where:

- S_B is the between-class scatter matrix (variance of class means around the grand mean).
- S_W is the within-class scatter matrix (within-class spread, summed across classes).

**Optimal projections** are the generalized eigenvectors of S_B and S_W. For K classes, LDA yields at most **K − 1** discriminant components.

**Advantages:**

- Supervised — uses class labels to find truly discriminative directions.
- Often beats PCA for classification preprocessing.
- Linear and interpretable.
- Fast — closed-form solution.

**Disadvantages:**

- Assumes Gaussian class distributions with similar covariance structure.
- Limited to K − 1 dimensions, which can be too few for many-class problems.
- S_W can be singular when there are more features than samples per class.

**LDA vs PCA:** PCA is unsupervised and maximizes *total* variance. LDA is supervised and maximizes *class separability*. PCA is the right choice for general dimensionality reduction; LDA when class info is available and classification is the goal.

In practice, try PCA first (no labels needed) and switch to LDA if labels improve downstream performance. LDA is less popular today as deep models dominate, but it remains useful when interpretability matters. In interviews, framing LDA as the supervised counterpart of PCA demonstrates understanding of both methods.

---

### Q7: Explain autoencoders for dimensionality reduction and their advantages.

**A:** **Autoencoders** are neural networks that learn a compressed representation by encoding and reconstructing the input.

**Architecture:**

```
encoder:    z = f_enc(x)        # low-dimensional latent
decoder:    x̂ = f_dec(z)        # reconstruction
loss   :    || x − x̂ ||²
```

The bottleneck z is forced to capture the essential information needed to reconstruct x.

**Advantages:**

- Nonlinear — handles complex structure that PCA misses.
- Flexible — choose any bottleneck dimension and architecture.
- Can incorporate constraints (e.g., VAE adds KL regularization for a smooth latent space).
- Scales well with SGD and GPUs.
- Can be fine-tuned end-to-end for downstream tasks.

**Disadvantages:**

- Requires training — no closed-form solution like PCA.
- Hyperparameter-heavy (architecture, learning rate, regularization).
- Less interpretable than PCA — latent dimensions are opaque.
- Can memorize without learning structure if the bottleneck is too generous.

**Variants:**

- **Variational Autoencoder (VAE)** — adds KL regularization toward a prior, useful for generative modeling.
- **Denoising Autoencoder** — adds noise to the input, encouraging robustness.

**When to use autoencoders:** complex data (images, audio), nonlinearity is essential, interpretability isn't critical, and there's enough data to train. For tabular data with mostly linear structure, PCA is usually a better starting point — simpler, faster, and interpretable.

In interviews, the framing is "PCA is linear and closed-form; autoencoders are nonlinear and learned — different tools for different problems."

---

### Q8: Explain the curse of dimensionality and why dimensionality reduction helps.

**A:** The **curse of dimensionality** is the collection of problems that emerge in high-dimensional spaces.

**Key issues:**

- **Volume explodes** — the volume of ℝᵈ grows exponentially with d, so data becomes increasingly sparse. Neighborhoods become huge and Euclidean distances lose discriminative meaning.
- **Sample requirements grow** — VC-style bounds suggest you need exponentially more samples to reach the same generalization quality as features increase.
- **Computational cost** — algorithms scale with d or d², making training and inference slow and memory-intensive.
- **Noise dominates** — irrelevant or noisy features compete with signal more aggressively as d grows.

**How dimensionality reduction helps:**

- Removes noise dimensions (signal often lives in a low-dimensional subspace).
- Improves generalization (fewer effective parameters).
- Reduces computation.
- Enables visualization.
- Concentrates information into fewer dimensions.

**Practical example:** text with d = 10K+ word features often has only ~100–1000 relevant dimensions. Reducing via PCA or learned embeddings dramatically improves downstream models.

**Theoretical anchor:** the **Johnson-Lindenstrauss lemma** says n points in ℝᵈ can be embedded in ℝᵏ with k = O(log n / ε²) while preserving pairwise distances within a factor of (1 ± ε). The number of needed dimensions depends on n, not d.

A nuance worth mentioning in interviews: not all high-dimensional problems suffer equally. Sparse data (text) and dense data (images) have different versions of the curse and benefit differently from reduction.

---

### Q9: Explain kernel PCA and its advantages over linear PCA.

**A:** **Kernel PCA (KPCA)** extends PCA to nonlinear structure by implicitly mapping data into a high-dimensional feature space via a kernel k, then doing PCA there.

**Algorithm:**

1. Compute the Gram matrix:

   ```
   K_{ij} = k(x_i, x_j)
   ```

2. Center K in feature space.
3. Eigendecompose the centered K.
4. The top eigenvectors give principal components in the implicit feature space.

The "kernel trick" means we never compute the explicit feature map φ(x) — we only need K. This lets KPCA discover structure linear PCA misses (concentric circles, S-curves, etc.).

**Advantages:**

- Captures nonlinear structure.
- Kernel trick — efficient, no explicit feature map needed.
- Choice of kernel encodes assumptions (RBF for locality, polynomial for feature interactions).

**Disadvantages:**

- O(n²) memory for the Gram matrix — prohibitive for large n.
- Extra hyperparameters (kernel choice, γ for RBF).
- Embedding new points requires kernel evaluations against all training points.
- Less interpretable than linear PCA — components are not simple feature combinations.

**Vs PCA:** PCA is fast and interpretable; KPCA is more flexible but slower and harder to interpret. Modern alternatives (t-SNE, UMAP, autoencoders) typically capture nonlinearity more effectively at large scale, so KPCA is theoretically elegant but rarely the practical choice.

In interviews, mention KPCA as a kernel-method extension of PCA, with the caveat that modern practitioners reach for UMAP or neural networks for nonlinear reduction.

---

### Q10: Explain factor analysis and its relationship to PCA.

**A:** **Factor analysis (FA)** is a probabilistic latent-variable model: data is generated from a small number of latent factors plus noise.

```
x = W·z + μ + ε,    z ∈ ℝᵏ,    ε ~ Normal(0, Σ)
```

where W is the factor loading matrix, z are the latent factors, and ε is independent per-feature noise. Marginalizing z gives:

```
x ~ Normal( μ, W·Wᵀ + Σ )
```

So FA models the data covariance as a low-rank structure (W·Wᵀ) plus noise (Σ). Fitting is typically via EM, maximizing the marginal likelihood.

**Advantages:**

- Probabilistic — allows likelihood-based model selection and uncertainty quantification.
- Explicit noise model (Σ), unlike PCA.
- Latent factors can be interpretable as "underlying causes."
- Handles missing data naturally via EM.

**Disadvantages:**

- Still assumes a linear generative model.
- More complex than PCA (iterative fit instead of closed form).
- Extra hyperparameters (number of factors k, noise structure).
- Slower than PCA.

**FA vs PCA:** PCA finds the best deterministic projection that maximizes variance; FA finds the best probabilistic latent-variable model that explains the covariance. They often produce similar components when the noise is small and isotropic, but their philosophies differ.

**Use FA when:** you need a probabilistic framework, explicit noise modeling, missing-data support, or want to compare models by likelihood (AIC/BIC). For straightforward large-scale dimensionality reduction, PCA is usually preferred.

In interviews, FA is a useful name to drop when the data-generation perspective matters — it shows awareness of probabilistic alternatives to PCA.

---

### Q11: Explain Independent Component Analysis (ICA) and its applications.

**A:** **ICA** assumes data is a linear mixture of statistically *independent* latent sources:

```
x = A·s  +  n
```

where A is an unknown mixing matrix, s are independent latent sources, and n is noise. PCA finds *uncorrelated* directions (a second-moment property); ICA finds *independent* directions (which uses higher moments).

**Key requirement:** sources must be **non-Gaussian**. Gaussian distributions are rotationally symmetric, so independence is indistinguishable from uncorrelatedness — ICA degenerates to PCA in that case.

**Algorithm.** Fit x = A·s by maximizing the non-Gaussianity of the estimated sources, using measures like kurtosis, negentropy, or mutual information.

**Applications:**

- **Blind source separation** — the cocktail party problem (separate individual speakers from mixed audio).
- **Brain imaging** — identifying independent functional networks in fMRI.
- **Financial data** — extracting independent price drivers.

**Advantages:**

- Finds *independent* sources, which is a stronger and often more meaningful condition than uncorrelated.
- Useful in settings where PCA misses the structure (non-Gaussian sources).

**Disadvantages:**

- Requires non-Gaussian sources.
- Slower than PCA.
- Solutions are ambiguous up to permutation and scale.
- Sensitive to noise.

**ICA vs PCA:** PCA decorrelates (second-moment); ICA seeks independence (higher-moment information). ICA is specialized — reach for it when independent sources are physically plausible.

In interviews, knowing ICA signals depth — it's not commonly asked, but mentioning it appropriately (e.g., for signal-separation contexts) sets you apart.

---

### Q12: Explain feature selection vs. feature extraction and when to use each.

**A:** **Feature selection** keeps a subset of the original features, dropping the irrelevant or redundant ones. Three flavors:

- **Filter (univariate)** — rank features by correlation, mutual information, or information gain with the target; keep the top k.
- **Wrapper** — evaluate feature subsets by model performance and pick the best subset.
- **Embedded** — let the model do the selection (e.g., L1 / Lasso regularization automatically zeros out irrelevant coefficients).

**Feature extraction** creates *new* features as transformations of the originals — PCA (linear), autoencoders (nonlinear), etc.

**Tradeoffs:**

- *Interpretability:* selection preserves original features; extraction creates combinations that are harder to name.
- *Compute:* selection is fast; extraction requires fitting.
- *Information capture:* selection can miss interactions; extraction captures structured combinations.
- *Mixed types:* selection handles categorical features naturally; extraction often needs encoding.

**When to use which:**

- **Selection** — tabular data, high-dimensional sparse data (text), settings where features have clear meaning and interpretability matters.
- **Extraction** — dense data (images), nonlinear structure, unlabeled data, interpretability not critical.

**Practical strategy:**

1. Start with selection — fast baseline that keeps the original features intact.
2. If performance plateaus, try extraction.
3. For very high d, combine: selection to ~100 features, then PCA on those.

A useful nuance for interviews — for high-dimensional sparse data like text, selection often *beats* PCA-style extraction because PCA on sparse high-dim data wastes effort capturing irrelevant variance. Context-dependent answers impress more than blanket "use PCA."

---

### Q13: What is the Johnson-Lindenstrauss lemma and its implications for dimensionality reduction?

**A:** The **Johnson-Lindenstrauss (JL) lemma** says: for any set of n points in ℝᵈ and any ε > 0, there exists a linear projection to ℝᵏ with

```
k = O( log n / ε² )
```

such that all pairwise distances are preserved within a factor of (1 ± ε).

**Implications:**

- Any d-dimensional dataset can be reduced to k = O(log n / ε²) dimensions with bounded distortion.
- The target dimension k depends only on n and ε — *not on d*. Even if d is huge, k stays small for moderate n.

**Example.** n = 1000 points, ε = 0.1 → k ≈ 50 dimensions are enough to preserve distances within 10%.

**Why this matters in practice:**

- Provides theoretical justification for **random projection** — multiplying by a random k × d matrix is a fast approximation to PCA, with guaranteed distance preservation.
- Tells us that the curse of dimensionality isn't absolute — when the data has low intrinsic dimension, drastic reduction is feasible.
- Random projections scale much better than PCA for very large d.

In interviews, citing the JL lemma signals theoretical grounding. It's rarely asked outright, but it's a great anchor when justifying *why* dimensionality reduction works: "JL guarantees that if intrinsic dimension is low, we can preserve distances with only O(log n) target dimensions."

---

### Q14: Explain random projections and Gaussian random projection for efficient dimensionality reduction.

**A:** **Random projection** is a fast, theoretically justified approximation to PCA. Generate a random k × d matrix R (Gaussian or sparse) and project:

```
X_reduced = X · R
```

Despite the randomness, the Johnson-Lindenstrauss lemma guarantees pairwise distances are preserved when k = O(log n / ε²).

**Advantages:**

- Very fast — O(k · d) per row, no eigendecomposition required.
- Scales to huge d, especially with sparse random matrices.
- Memory-efficient and embarrassingly parallel.
- Theoretical guarantees on distance preservation.

**Disadvantages:**

- Doesn't maximize variance like PCA — may "waste" some target dimensions.
- Different random seeds give different embeddings.
- Components aren't interpretable (random combinations of features).
- O(log n) target dimensions can still be large for very large n.

**Variants:**

- **Gaussian random projection** — entries of R drawn i.i.d. from a standard normal. Most general, densest.
- **Sparse random projection** — R has few non-zero entries; very fast on sparse data.
- **Structured random projection** (Hadamard, DCT-based) — enables FFT-style fast multiplication.

**When to use:** very large d (≫ 100K), settings that need speed over optimality, streaming/incremental data, or as a fast baseline.

**Compared to PCA:** PCA is O(d² · n) and optimally preserves variance; random projection is O(k · d) and approximately preserves distances. For moderate d, PCA is usually better; for extreme d, random projection wins on speed and memory.

In interviews, random projection is an underappreciated tool — mentioning it for ultra-high-dimensional problems demonstrates practical knowledge.

---

### Q15: How would you decide which dimensionality reduction method to use for a given problem?

**A:** Pick based on a few axes:

- **Problem type** — unsupervised (PCA, UMAP) vs supervised (LDA); visualization (t-SNE, UMAP) vs preprocessing (PCA, random projection).
- **Data structure** — linear (PCA), nonlinear (t-SNE, UMAP, autoencoders), independent sources (ICA).
- **Dataset size** — small (anything works); large n (avoid O(n²) methods like t-SNE, prefer PCA, random projection, or UMAP with approximations); large d (random projection, sparse PCA).
- **Interpretability** — need it (selection, PCA, LDA); don't care (t-SNE, autoencoders).
- **Downstream task** — visualize clusters (t-SNE, UMAP); preprocess for an ML model (PCA, feature selection, autoencoders).
- **Compute budget** — tight (random projection, PCA); generous (t-SNE, ICA, autoencoders).

**Practical workflow:**

1. **Baseline:** apply PCA. If ~95% of variance lives in the first 10–50 components, the problem is mostly linear and you may not need anything else.
2. **Visualization:** UMAP first (fast, preserves more structure than t-SNE); fall back to t-SNE if speed isn't critical.
3. **Classification preprocessing:** PCA if unsupervised, LDA if supervised. If results suggest nonlinearity, try an autoencoder.
4. **Anomaly detection:** UMAP or PCA, examining reconstruction error.
5. **PCA underperforms:** try UMAP, autoencoders, or hand-engineered features.
6. **Very large d (text, genomics):** feature selection (filter or L1) first to ~hundreds of features, then PCA on those.

**Red flags to call out:**

- Using t-SNE coordinates as ML features (wrong — t-SNE distances aren't preserved).
- Skipping the explained-variance check.
- Forgetting to standardize features before PCA.

In interviews, this decision-tree style answer beats blanket recommendations. Mentioning hybrid pipelines — like selection → PCA → downstream model, or PCA → UMAP for visualization — shows real practitioner judgment.

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

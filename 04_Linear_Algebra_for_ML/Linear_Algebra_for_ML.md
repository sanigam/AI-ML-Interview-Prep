# Linear Algebra for Machine Learning

📺 **Video Lecture:** https://youtu.be/YpckH7F5vj0

## Interview Anchor
- **Eigenvalues and Eigenvectors:** For matrix A, non-zero vector v where Av = λv; λ is eigenvalue, v is eigenvector
- **Matrix Decomposition:** Factoring A into simpler matrices (SVD, QR, LU); reveals structure and enables efficient computation
- **Rank:** Maximum number of linearly independent rows or columns; determines invertibility and solution properties

## Key Concepts Overview
Linear algebra is the mathematical language of machine learning—almost every algorithm, from linear regression to neural networks, is fundamentally a linear algebra computation. Understanding eigenvalues helps you comprehend dimensionality reduction (PCA), interpret stability in neural networks, and recognize when systems are ill-conditioned. Matrix decompositions aren't just computational tools; they reveal the geometric structure hidden in data and algorithms. For instance, SVD reveals which directions in data have maximum variance, QR decomposition enables numerically stable least-squares solving, and eigendecomposition explains why certain optimization techniques converge quickly.

Interviewers test linear algebra knowledge because it separates practitioners who copy-paste code from those who understand what their models do mathematically. You'll encounter these concepts everywhere: understanding trace and determinant helps interpret loss landscapes, orthogonality appears in neural network regularization, and matrix calculus is essential for deriving gradients in deep learning.

---

### Q1: Define vectors and matrices and explain their geometric interpretation.

**A:** A **vector** is a 1D array of numbers — either a column vector x ∈ ℝⁿ or a row vector xᵀ ∈ ℝ¹ˣⁿ. Geometrically, it represents either a point in n-dimensional space or a direction from the origin.

A **matrix** A ∈ ℝᵐˣⁿ is a 2D array of numbers. Geometrically, it represents a linear transformation from ℝⁿ to ℝᵐ — multiplying Av transforms the vector v. The interpretation of rows and columns:

- The i-th row specifies coefficients of a linear constraint or linear combination.
- The j-th column specifies how the j-th standard basis vector gets transformed.

Visually, A stretches, rotates, or reflects vectors, and can change dimensionality (e.g., a 10×3 matrix maps 3D points into 10D space).

**Basic operations:**

- *Scalar multiplication:* λA scales all entries.
- *Addition:* A + B combines transformations (only if dimensions match).
- *Matrix multiplication:* AB means "apply transformation B first, then A."

In ML, data matrices X ∈ ℝⁿˣᵈ store n observations as rows and d features as columns, and linear models compute Xw where w is the weight vector. Reasoning about the geometry helps you predict algorithm behavior without actually computing anything.

---

### Q2: Explain eigenvalues and eigenvectors with intuition and applications.

**A:** For a square matrix A, an **eigenvalue** λ and **eigenvector** v (nonzero) satisfy:

```
A·v = λ·v
```

This means A acts on v just like scalar multiplication — it stretches v by factor λ without changing its direction. Eigenvectors are the "preferred directions" of A.

**Finding them:** solve the characteristic polynomial for λ, then solve for each eigenvector:

```
det(A − λI) = 0      → gives the eigenvalues
(A − λI)·v = 0       → gives the eigenvector for each λ
```

**Interpreting eigenvalues:**

- λ > 1 — A stretches along v
- 0 < λ < 1 — A contracts along v
- λ < 0 — direction is reversed

Real symmetric matrices (like covariance matrices) have real eigenvalues and orthogonal eigenvectors, which makes them especially nice to work with.

**ML applications:**

- **PCA** uses eigenvectors of the covariance matrix as the directions of maximum variance.
- **Power iteration** finds the largest eigenvalue, which underlies spectral clustering.
- **Condition number** is the ratio of largest to smallest eigenvalue, and predicts optimization difficulty.
- **Neural network training dynamics** depend on the eigenvalues of the Hessian at convergence.
- **Graph algorithms** use eigenvectors of adjacency or Laplacian matrices.

When eigenvalues are close to zero, the matrix is nearly singular. When they span a wide range, the condition number is high — meaning the system is ill-conditioned and numerically unstable.

---

### Q3: What is Singular Value Decomposition and why is it fundamental in ML?

**A:** **SVD** factorizes any m×n matrix A into three matrices:

```
A = U · Σ · Vᵀ
```

where:

- U is m×m orthogonal — its columns are the *left singular vectors*.
- Σ is m×n diagonal with non-negative entries σ₁ ≥ σ₂ ≥ ... ≥ 0 — the *singular values*.
- V is n×n orthogonal — its columns are the *right singular vectors*.

Singular values relate to eigenvalues: σᵢ² are the eigenvalues of both AᵀA and AAᵀ.

SVD is **universal** (works for any matrix, not just square), **numerically stable** (preferred over eigendecomposition for general matrices), and **reveals rank** (the number of nonzero singular values).

**ML applications:**

- **Low-rank approximation:** keep the top k singular values and vectors — this gives the best rank-k reconstruction in Frobenius norm.
- **PCA:** SVD on centered data gives the principal components directly.
- **Image compression:** apply SVD to an image matrix and keep the top singular values.
- **Least-squares via pseudoinverse:** A⁺ = V·Σ⁺·Uᵀ.
- **Noise reduction:** truncate small singular values.

**Interpretation of the three factors:** Uᵀ projects data onto the left singular vectors, Σ scales by the singular values, and Vᵀ captures the directions in the original space. When singular values decay rapidly, the data has low *intrinsic dimensionality* — many features are redundant.

---

### Q4: Explain QR decomposition and its role in solving least-squares problems.

**A:** **QR decomposition** factorizes an m×n matrix A as:

```
A = Q · R
```

where Q is m×n with orthonormal columns (so QᵀQ = Iₙ in "thin" QR) and R is n×n upper triangular. The decomposition isn't unique — sign flips on columns of Q with matching row negations in R give equivalent decompositions — but QR is numerically very stable.

**Why QR for least-squares.** For an overdetermined system Ax ≈ b, the normal equations would be:

```
AᵀA·x = Aᵀb
```

This is fast but squares the condition number, causing numerical issues. Using QR instead avoids that:

```
A·x = b   →   QR·x = b   →   R·x = Qᵀb
```

The final triangular system is well-conditioned and solved by back-substitution.

This approach is numerically superior especially when A is ill-conditioned.

**Other uses of QR:**

- **Gram-Schmidt orthogonalization:** QR effectively orthogonalizes the columns of A.
- **Multiple right-hand sides:** compute QR once and reuse it for many different b.
- **Eigenvalue algorithms:** the classic QR iteration for finding eigenvalues.
- **Reduced-rank fitting.**

Computational complexity O(mn²) is higher than the normal equations' O(n³) for small overdetermined systems, but the stability gain is well worth it. ML regression libraries use QR by default; understanding this helps you trust numerical results and debug ill-conditioned problems.

---

### Q5: What is LU decomposition and when is it preferred?

**A:** **LU decomposition** factors a square matrix A as:

```
A = L · U
```

where L is lower triangular (with ones on the diagonal) and U is upper triangular.

To solve A·x = b, do two cheap triangular solves:

```
L·y = b    →   forward substitution gives y
U·x = y    →   back-substitution gives x
```

LU is fast (O(n³) with a small constant) and reusable — compute it once, then solve many right-hand sides quickly. It needs no orthogonality like QR, so it's simpler computationally. The downside is that without pivoting, LU is less numerically stable than QR. **Partial pivoting** (reorder rows to avoid small divisors) fixes most of that, and LU with partial pivoting is the default in libraries like LAPACK.

**Cholesky decomposition** is a related trick for symmetric positive-definite A:

```
A = L · Lᵀ
```

It costs about half the work of LU and is more stable.

**When to use which:**

- **LU:** general square matrices where stability is adequate; solving many systems with the same A.
- **QR:** least-squares with rectangular A, or whenever numerical stability is paramount.
- **Cholesky:** covariance matrices and other symmetric positive-definite systems.

In ML, knowing which decomposition fits your problem affects numerical stability, especially with high-dimensional data or ill-conditioned covariance matrices.

---

### Q6: Define rank and explain its significance in linear systems.

**A:** The **rank** of a matrix A has three equivalent definitions, all useful:

- The dimension of its column space (which equals the dimension of its row space).
- The number of linearly independent columns (or rows).
- The number of nonzero singular values in the SVD.

For an m×n matrix, rank(A) ≤ min(m, n). The matrix has *full column rank* if rank = n, *full row rank* if rank = m, and *full rank* if rank = min(m, n).

**What rank tells you about A·x = b:**

- If rank(A) = n and m ≥ n → unique least-squares solution.
- If rank(A) < n → infinitely many solutions (underdetermined system).
- If b is not in the column space of A → no solution exists (inconsistent system).

**Practical implications:**

- Underdetermined systems have many solutions — adding regularization (like L2) picks out the one with smallest norm.
- Inconsistent systems require solving least-squares to minimize residual error.

**Rank in ML:**

- In feature design, rank < number of features signals **multicollinearity** (perfectly correlated features).
- In neural networks, the width of hidden layers controls rank — if a hidden layer is narrower than the input dimension, it creates a bottleneck (dimensionality reduction).

Understanding rank helps you diagnose why models fail to train (singular weight matrices, rank-deficient design matrices) and choose architectures with the right capacity.

---

### Q7: Explain the null space and column space of a matrix.

**A:** For a matrix A ∈ ℝᵐˣⁿ, two subspaces matter most:

- **Column space (range):** col(A) = { A·x : x ∈ ℝⁿ }. This is all the vectors A can produce — a subspace of ℝᵐ. Its dimension equals rank(A).
- **Null space (kernel):** null(A) = { x : A·x = 0 }. This is all the vectors A collapses to zero — a subspace of ℝⁿ. Its dimension is n − rank(A) (the **rank-nullity theorem**).

The null space and the row space are orthogonal complements: null(A) ⊥ row(A). Every vector decomposes uniquely as x = x_col + x_null, where A acts only on the x_col part.

**Implications for A·x = b:**

- A solution exists if and only if b lies in col(A).
- If it exists, the general solution is one particular solution plus any vector from null(A).

**Geometric picture:** the column space is the *observable* part of the input — what A can produce. The null space is the *invisible* part — directions A collapses to zero.

**ML relevance:** the null space corresponds to **unidentifiable parameters**. If a parameter direction lies in the null space, changing the parameter along that direction doesn't change predictions at all, so you cannot distinguish those parameter values from data. This is exactly what happens with perfectly collinear features — parameters along the collinear direction are unidentifiable without regularization. Covariate shift or data leakage that aligns with the null space can make learning impossible.

---

### Q8: Define positive definite matrices and explain their role in ML.

**A:** A symmetric matrix A is **positive definite (PD)** if its quadratic form is strictly positive for any nonzero x:

```
xᵀ·A·x > 0    for all x ≠ 0
```

It is **positive semidefinite (PSD)** if the inequality is non-strict (≥ 0).

**Equivalent characterizations** (any one implies the others):

- All eigenvalues are positive (PD) or non-negative (PSD).
- A = BᵀB for some full-rank matrix B (PD case).
- All leading principal minors are positive (Sylvester's criterion).

**Useful properties:**

- Always invertible (PD), or possibly singular with zero eigenvalues (PSD).
- Admit a fast and stable Cholesky decomposition A = L·Lᵀ.
- The quadratic form xᵀAx measures the "size" of x weighted by A.
- A function has positive local curvature wherever its Hessian is PD — a local minimum.
- Covariance matrices are always PSD, and PD when non-singular.

**Where this shows up in ML:**

- The regularization term λ·wᵀw adds a PD penalty.
- A PD Hessian at a critical point confirms a local minimum.
- Second-order optimization methods (Newton, quasi-Newton) require a PD Hessian for the step direction to be well-defined.
- In neural networks, checking Hessian positive-definiteness validates convergence to a minimum.
- Covariance matrices used in Gaussian processes, Bayesian inference, and whitening are PSD; whether they're invertible depends on having enough data.

---

### Q9: Explain orthogonality and orthonormal bases and their utility.

**A:** Two vectors u and v are **orthogonal** if their inner product is zero:

```
uᵀ·v = 0    (geometrically: perpendicular)
```

A set of vectors is **orthonormal** if it's pairwise orthogonal and each vector has unit norm. With an orthonormal basis {u₁, ..., uₙ} of ℝⁿ, any vector x decomposes simply:

```
x = Σᵢ (uᵢᵀ·x)·uᵢ
```

so coefficients are just inner products. Stacking the basis vectors as columns of U gives UᵀU = I.

**Orthogonal matrices** (square matrices with orthonormal columns) have two especially convenient properties:

```
QᵀQ = I        and    Qᵀ = Q⁻¹
```

That is, the inverse is just the transpose — trivial to compute. They are numerically stable (condition number = 1) and norm-preserving:

```
||Q·x|| = ||x||
```

which makes orthogonal transformations ideal for numerical computation.

**Practical advantages:**

- Coefficients are inner products — fast to compute.
- Norm-preserving — minimal rounding error when solving.
- Invertibility is automatic.
- Changes of basis don't distort geometry.

**ML uses of orthogonality:**

- **Whitening** transforms features into a decorrelated, unit-variance space.
- **Principal components** are orthonormal eigenvectors of the covariance matrix.
- Layers with orthogonal weight matrices have better conditioning (related to batch normalization and spectral normalization).
- Attention mechanisms in transformers compute orthogonal-style projections.

The **Gram-Schmidt** algorithm converts any basis into an orthonormal one, and **QR decomposition** produces an orthonormal basis of the column space.

---

### Q10: What is matrix calculus and how does it apply to ML optimization?

**A:** Matrix calculus extends single-variable calculus to vectors and matrices. The **gradient** of a scalar function f(X) with respect to a matrix X is itself a matrix of partial derivatives, ∂f/∂X.

A few common gradient formulas worth memorizing:

```
f(x) = aᵀ·x         →   ∇f = a
f(x) = xᵀ·A·x       →   ∇f = (A + Aᵀ)·x   = 2A·x  (if A is symmetric)
d(A·x)/dx = Aᵀ      (numerator-layout convention)
```

The **Jacobian** of a vector-valued function f: ℝⁿ → ℝᵐ is the m×n matrix:

```
[J]ᵢⱼ = ∂fᵢ / ∂xⱼ
```

The **Hessian** of f: ℝⁿ → ℝ is the matrix of second partials:

```
[H]ᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)
```

The **chain rule** generalizes naturally — if z = f(g(x)), then dz/dx = (dz/dg)·(dg/dx). And the **trace trick** is helpful when rearranging matrix derivatives, since the trace is invariant under cyclic permutation:

```
tr(A·B·C) = tr(C·A·B) = tr(B·C·A)
```

**In ML this directly drives optimization:**

- Computing ∇loss with respect to weights is what gradient descent uses.
- Backpropagation in deep networks is just the chain rule applied through layers.
- The Hessian determines the convergence rate of Newton's method.

Understanding matrix calculus helps you derive correct gradient formulas, implement autodiff correctly, and understand why certain parameterizations optimize better than others (for example, softmax parameterization avoids singular Hessians).

---

### Q11: How does SVD relate to PCA and how do you use it for dimensionality reduction?

**A:** **PCA** finds the directions of maximum variance in data. Given centered data X ∈ ℝⁿˣᵈ (n samples, d features), compute its SVD:

```
X = U · Σ · Vᵀ
```

The three pieces map onto PCA quantities:

- **Columns of V** are the principal components — equivalently, the eigenvectors of the sample covariance XᵀX / (n−1).
- **Singular values in Σ** relate to the variance along each component, with σᵢ² / (n−1) approximating the i-th component's variance.
- **U** contains the projections of the data onto the principal components.

**To reduce to k dimensions**, keep the top k columns of V (and corresponding parts of U and Σ):

```
X_reduced = X · V[:, 1:k]  =  U[:, 1:k] · Σ[1:k, :]
```

To reconstruct an approximation in the original feature space:

```
X̂ = X_reduced · V[:, 1:k]ᵀ
```

**Choosing k:** plot cumulative variance explained,

```
explained(k) = (σ₁² + σ₂² + ... + σₖ²) / (σ₁² + ... + σ_d²)
```

and pick the smallest k that hits roughly 95% of the variance.

**Why SVD is preferred over eigendecomposition of XᵀX:**

- Numerically more stable.
- Gives both components and projections directly.
- Works for rectangular matrices.
- Reveals effective dimensionality — a steep singular value decay means low intrinsic dimension.

SVD also gives the best rank-k approximation in Frobenius norm:

```
A ≈ U[:, 1:k] · Σ[1:k, 1:k] · V[:, 1:k]ᵀ
```

In production, precompute V once and apply V[:, 1:k]ᵀ to new samples for fast projection.

---

### Q12: Explain the role of matrix rank in neural networks and deep learning.

**A:** In neural networks, the rank of a weight matrix determines **expressive capacity**. If a weight matrix W ∈ ℝᵐˣⁿ has rank below min(m, n), it maps ℝⁿ into a lower-dimensional subspace and creates a bottleneck. When the hidden layer width h is less than the input dimension d, the layer reduces dimensionality and rank ≤ h.

Full expressiveness requires rank = min(input_dim, hidden_dim). Networks with insufficient rank simply cannot represent complex functions. **Overparameterization** (width far larger than needed) provides redundancy that helps optimization — gradient descent finds the minimum-norm solution (an implicit bias), which tends to generalize better.

**Rank shows up in several modern ML techniques:**

- **Low-rank adaptation (LoRA):** approximate weight updates as W ≈ W₀ + A·Bᵀ where A and B are low-rank, drastically reducing the parameters that need fine-tuning.
- **Neural collapse:** at convergence, class-wise means in the final hidden layer exhibit a structured high-rank pattern.
- **Deep matrix factorization:** representing weights as products of matrices can recover latent structure in the data.

**Singular value distribution as a diagnostic:** the distribution of singular values of a weight matrix indicates layer conditioning. A uniform distribution (all singular values roughly equal) is well-conditioned; a heavily skewed distribution often signals numerical issues. Batch normalization implicitly keeps the singular value spectrum reasonable, which is part of why it helps optimization.

Understanding rank helps explain several deep learning phenomena: why certain widths matter, why low-rank approximations work so well, and how overparameterization aids learning.

---

### Q13: How do you detect and handle ill-conditioning in linear systems?

**A:** A matrix is **ill-conditioned** if small changes in the data cause large changes in the solution. The **condition number** quantifies this:

```
κ(A) = σ_max / σ_min        (ratio of largest to smallest singular value)
```

κ = 1 is perfectly conditioned; κ ≫ 1 is ill-conditioned. Ill-conditioning typically arises when A is nearly singular (very small singular values) or when its columns are nearly linearly dependent.

**Detecting ill-conditioning:**

- Compute the SVD and look for a steep drop-off in singular values toward zero.
- Compute κ directly — values above ~10¹⁰ are very ill-conditioned.
- Solve A·x = b twice with slightly different b and see whether the solutions differ drastically.
- Look at the spread of eigenvalues of AᵀA.

**Handling ill-conditioning:**

- **Regularization:** add λ·I to AᵀA (ridge regression). This shifts small singular values upward and improves conditioning.
- **Feature scaling:** normalize features to similar magnitudes — a simple but powerful fix.
- **QR instead of normal equations:** better numerical stability.
- **SVD truncation:** drop small singular values, using the pseudoinverse implicitly.
- **Reformulate the problem:** sometimes a reparameterization is naturally better-conditioned.

**In ML specifically:**

- Multicollinear features cause ill-conditioning. L2 regularization adds λ to eigenvalues and stabilizes the solution.
- Covariance matrices with a wide range of eigenvalues are ill-conditioned, usually because features are on very different scales.

Preprocessing with **feature standardization** (zero mean, unit variance) is a simple step that often dramatically improves conditioning.

---

### Q14: Explain trace and determinant and their interpretations.

**A:** Two scalar summaries of a matrix come up constantly in ML:

```
tr(A) = Σᵢ Aᵢᵢ  =  Σᵢ λᵢ           (sum of diagonal = sum of eigenvalues)
det(A) = ∏ᵢ λᵢ                       (product of eigenvalues)
```

**Properties:**

- Trace is **linear:** tr(A + B) = tr(A) + tr(B), and **invariant under similarity:** tr(A) = tr(P⁻¹·A·P).
- Determinant is **multiplicative:** det(A·B) = det(A)·det(B), and **zero iff A is singular**.

**Geometric interpretation:** det(A) is the volume-scaling factor when A transforms a unit volume — positive det means orientation is preserved, negative means flipped. The trace can be thought of as a sense of "total curvature" — for example, the trace of a Hessian equals the Laplacian (sum of second derivatives).

**A few useful identities:**

```
tr(Xᵀ·Y) = Σᵢⱼ Xᵢⱼ·Yᵢⱼ              (Frobenius inner product)
∂ tr(A·B) / ∂A = Bᵀ
```

**In ML:**

- `log det(Σ)` appears in Gaussian probability (its negative is differential entropy).
- The trace is much cheaper to compute than the determinant, so it's preferred in approximate inference.
- tr(H) of the Hessian is used as a generalization estimate in overparameterized networks.
- Trace regularization on tr(WᵀW) is equivalent to L2 regularization of the weights.

Understanding trace and determinant helps you interpret loss functions and debug optimization — for example, a negative determinant signals a flipped orientation, which sometimes indicates non-convex behavior.

---

### Q15: How do you solve linear systems Ax = b using matrix decompositions?

**A:** **Direct methods** solve A·x = b via a decomposition, then a triangular solve. The three main choices:

**LU with partial pivoting** (most general, O(n³)):

```
A = P · L · U
solve  L·y = Pᵀ·b        (forward substitution)
then   U·x = y           (back-substitution)
```

**QR** (best for least-squares with rectangular A):

```
A = Q · R
solve  R·x = Qᵀ·b         (back-substitution)
```

This is numerically more stable than the normal equations.

**Cholesky** (fast for symmetric positive definite A — about half the cost of LU):

```
A = L · Lᵀ
solve  L·y = b           (forward substitution)
then   Lᵀ·x = y          (back-substitution)
```

**For least-squares (Aᵀ·A·x = Aᵀ·b):** two approaches:

- Solve the normal equations directly — fast but the condition number squares, hurting stability.
- Use QR of A and solve R·x = Qᵀ·b — more stable, condition number is unchanged.

**Iterative methods** (Conjugate Gradient, MINRES, GMRES) are preferred for sparse or very large matrices. They start from a guess and iteratively improve it, never forming a full decomposition.

**In ML:**

- Linear regression normal equations — QR is preferred for stability.
- Ridge regression solves (AᵀA + λI)·x = Aᵀb — Cholesky works if AᵀA is positive definite.
- For huge datasets, iterative solvers are the way to go.

Choosing a solver depends on matrix properties (size, sparsity, conditioning), accuracy requirements, and compute budget. Libraries like NumPy and SciPy pick appropriate solvers automatically based on matrix structure.

---

## Interview Cheatsheet

**Key Terms:**
- **Vector:** 1D array; geometrically a point or direction in n-dimensional space
- **Matrix:** 2D array; geometrically a linear transformation
- **Eigenvalue/Eigenvector:** λ, v where Av = λv; preferred directions of stretching
- **Rank:** Dimension of column space; number of linearly independent columns
- **SVD:** A = UΣV^T; universal decomposition revealing structure via singular values
- **QR:** A = QR; orthogonal (Q) and upper triangular (R), stable for least-squares
- **LU:** A = LU; lower and upper triangular, efficient for solving systems
- **Cholesky:** A = LL^T for positive definite A; fastest triangular decomposition
- **Column Space:** col(A) = {Ax : x ∈ ℝⁿ}; subspace of outputs A can produce
- **Null Space:** null(A) = {x : Ax = 0}; vectors mapped to zero by A
- **Orthogonal:** Vectors with zero dot product; orthogonal matrices preserve norms
- **Positive Definite:** x^T Ax > 0 for all x ≠ 0; has positive eigenvalues
- **Condition Number:** κ(A) = σ_max/σ_min; determines numerical stability
- **Trace:** tr(A) = Σ diagonal entries = Σ eigenvalues; sum of curvatures
- **Determinant:** det(A) = ∏ eigenvalues; volume scaling factor of transformation
- **Matrix Calculus:** Gradients and Jacobians of matrix functions; enables backpropagation

**Rapid-Fire Q&A:**
- **Q: Why is SVD preferred over eigendecomposition?** **A:** Works for any matrix (not just square), numerically stable, directly reveals rank and low-rank approximations
- **Q: What does rank-nullity theorem say?** **A:** rank(A) + nullity(A) = n; rank + null space dimension equals dimension of input space
- **Q: Why is orthogonality important numerically?** **A:** Orthogonal transformations preserve norms, have condition number 1, avoid rounding errors
- **Q: How does ridge regression change conditioning?** **A:** Adds λ to diagonal of A^T A, shifts small eigenvalues upward, improves condition number
- **Q: What's the relationship between SVD and PCA?** **A:** SVD on centered data directly gives PCA: columns of V are principal components, singular values relate to variance

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

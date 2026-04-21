# Linear Algebra for Machine Learning

## Interview Anchor
- **Eigenvalues and Eigenvectors:** For matrix A, non-zero vector v where Av = λv; λ is eigenvalue, v is eigenvector
- **Matrix Decomposition:** Factoring A into simpler matrices (SVD, QR, LU); reveals structure and enables efficient computation
- **Rank:** Maximum number of linearly independent rows or columns; determines invertibility and solution properties

## Key Concepts Overview
Linear algebra is the mathematical language of machine learning—almost every algorithm, from linear regression to neural networks, is fundamentally a linear algebra computation. Understanding eigenvalues helps you comprehend dimensionality reduction (PCA), interpret stability in neural networks, and recognize when systems are ill-conditioned. Matrix decompositions aren't just computational tools; they reveal the geometric structure hidden in data and algorithms. For instance, SVD reveals which directions in data have maximum variance, QR decomposition enables numerically stable least-squares solving, and eigendecomposition explains why certain optimization techniques converge quickly.

Interviewers test linear algebra knowledge because it separates practitioners who copy-paste code from those who understand what their models do mathematically. You'll encounter these concepts everywhere: understanding trace and determinant helps interpret loss landscapes, orthogonality appears in neural network regularization, and matrix calculus is essential for deriving gradients in deep learning.

---

### Q1: Define vectors and matrices and explain their geometric interpretation.

**A:** A vector is a 1D array of numbers (column vector x ∈ ℝⁿ or row vector x^T ∈ ℝ^(1×n)) representing a point in n-dimensional space or a direction from origin. A matrix A ∈ ℝ^(m×n) is a 2D array of numbers, geometrically representing a linear transformation from ℝⁿ to ℝᵐ: when you multiply Av, you're transforming the vector v. The i-th row of A specifies coefficients of a linear constraint or linear combination, while j-th column specifies how the j-th standard basis vector is transformed. Visually, matrix A stretches/rotates/reflects vectors and can change dimensionality (e.g., 10×3 matrix maps 3D points to 10D space). Operations: (1) scalar multiplication λA scales all entries, (2) addition A+B combines transformations (only if dimensions match), (3) matrix multiplication AB means "apply transformation B first, then A". In ML, data matrices X ∈ ℝ^(n×d) have n observations as rows and d features as columns; linear models compute Xw where w is weight vector. Understanding geometry helps you reason about algorithm behavior without computing.

---

### Q2: Explain eigenvalues and eigenvectors with intuition and applications.

**A:** For a square matrix A, eigenvalue λ and eigenvector v (nonzero) satisfy Av = λv, meaning A rotates v only by scaling factor λ (no direction change). Finding them: solve det(A - λI) = 0 (characteristic polynomial) for λ, then solve (A - λI)v = 0 for each λ. Geometrically: eigenvectors are "preferred directions" that A stretches along. If λ > 1, A stretches along v; 0 < λ < 1 contracts; λ < 0 reverses. Real symmetric matrices (like covariance matrices) have real eigenvalues and orthogonal eigenvectors. In ML applications: (1) PCA uses eigenvectors of covariance matrix (directions of maximum variance) to reduce dimensions, (2) power iteration finds largest eigenvalue (spectral clustering), (3) condition number = largest eigenvalue / smallest eigenvalue (determines optimization difficulty), (4) neural network training dynamics depend on eigenvalues of Hessian at convergence, (5) graph algorithms use eigenvectors of adjacency/Laplacian matrices. When eigenvalues are close to zero, matrix is nearly singular; when they're widely spread, condition number is high (ill-conditioned, numerically unstable).

---

### Q3: What is Singular Value Decomposition and why is it fundamental in ML?

**A:** SVD factorizes any m×n matrix A as A = UΣV^T where U is m×m orthogonal (columns are left singular vectors), Σ is m×n diagonal with σ₁ ≥ σ₂ ≥ ... ≥ 0 (singular values), and V is n×n orthogonal (columns are right singular vectors). Singular values are non-negative and relate to eigenvalues: σᵢ² are eigenvalues of A^T A and AA^T. SVD is universal (works for any matrix, not just square), numerically stable (preferred over eigendecomposition for general matrices), and reveals rank (number of non-zero singular values). Applications in ML: (1) low-rank approximation: keep top k singular values/vectors, best rank-k reconstruction with minimum error, (2) PCA: SVD on centered data gives principal components, (3) image compression: SVD of image matrix, keep top singular values, (4) solving least-squares: A = UΣV^T gives A⁺ = VΣ⁺U^T (pseudoinverse), (5) noise reduction: truncate small singular values. Interpretation: U^T x projects data onto left singular vectors, Σ scales by singular values, V^T captures directions in original space. When singular values decay rapidly, data has low intrinsic dimensionality (many features are redundant).

---

### Q4: Explain QR decomposition and its role in solving least-squares problems.

**A:** QR decomposition factorizes m×n matrix A as A = QR where Q is m×n orthogonal (Q^T Q = I_n if "thin" QR) and R is n×n upper triangular. The decomposition is not unique—adding sign flips to columns of Q and row negations to R gives equivalent decompositions, but QR is numerically stable. For solving least-squares Ax = b (overdetermined system), instead of normal equations A^T Ax = A^T b (which squares condition number, causing numerical issues), compute QR decomposition: Ax = QRx = b → Rx = Q^T b (well-conditioned triangular system, solved via back-substitution). This approach is numerically superior especially when A is ill-conditioned. QR also appears in: (1) Gram-Schmidt orthogonalization (orthogonalize columns of A), (2) least-squares with multiple right-hand sides (compute QR once, apply to many b), (3) eigenvalue algorithms (QR iteration), (4) reduced-rank fitting. Computational complexity O(mn²) is higher than normal equations O(n³) for small overdetermined systems, but the numerical stability gain is worth it in practice. In ML, regression libraries use QR for stability; understanding this helps you trust numerical results and debug ill-conditioned problems.

---

### Q5: What is LU decomposition and when is it preferred?

**A:** LU decomposition factors square matrix A as A = LU where L is lower triangular (with ones on diagonal) and U is upper triangular. Solve Ax = b via forward-substitution on Ly = b to get y, then back-substitution on Ux = y to get x. LU is fast (O(n³) but with small constant) and reusable: compute LU once, solve multiple right-hand sides quickly. Requires no orthogonality like QR, so simpler computationally. However, LU is less numerically stable than QR without pivoting; partial pivoting (reorder rows to avoid small divisors) improves stability. LU with partial pivoting is often the default in numerical libraries (like LAPACK) because it balances speed and stability. Related: Cholesky decomposition A = LL^T for positive definite A (symmetric matrix with positive eigenvalues) is even faster (about half computations of LU) and more stable. Use LU for: solving multiple systems with same A, general square matrices where stability is adequate. Use QR for: least-squares problems (rectangular A), when numerical stability is paramount. Use Cholesky for: covariance matrices, positive definite systems. In ML, understanding which decomposition to use affects numerical stability of your algorithms, especially with high-dimensional data or ill-conditioned covariance matrices.

---

### Q6: Define rank and explain its significance in linear systems.

**A:** Rank of matrix A (rank(A)) is the dimension of its column space (or row space—they're equal), equivalently the number of linearly independent columns (or rows), equivalently the number of non-zero singular values in SVD. For m×n matrix, rank(A) ≤ min(m, n). Full column rank (rank = n) means columns are linearly independent; full row rank (rank = m) means rows are linearly independent; full rank (rank = min(m,n)) means both. In solving Ax = b: (1) if rank(A) = n and m ≥ n, unique least-squares solution; (2) if rank(A) < n, infinitely many solutions (underdetermined); (3) if b is not in column space of A, no solution exists (inconsistent). Practical implications: underdetermined system has many solutions—adding regularization (like L2) picks out one with small norm; inconsistent system requires solving least-squares to minimize residual error. Rank reveals degeneracy: features in ML with rank < number of features indicates multicollinearity (perfectly correlated features). In neural networks, width of hidden layers relates to rank: if layer has width < input dimension, it creates bottleneck (dimensionality reduction). Understanding rank helps diagnose why models fail to train (singular weight matrices, rank-deficient design matrices) and design architectures with appropriate capacity.

---

### Q7: Explain the null space and column space of a matrix.

**A:** For matrix A ∈ ℝ^(m×n), the column space (or range) col(A) = {Ax : x ∈ ℝⁿ} is all linear combinations of columns, a subspace of ℝᵐ with dimension = rank(A). The null space (or kernel) null(A) = {x : Ax = 0} is all vectors that A maps to zero, a subspace of ℝⁿ with dimension = n - rank(A) (rank-nullity theorem). These are orthogonal complements: null(A) ⊥ row(A), and any vector decomposes as x = x_col + x_null where Ax_col = Ax and Ax_null = 0. In solving Ax = b: solution exists iff b ∈ col(A); if exists, general solution is (particular solution) + (null space vectors). Geometrically: column space is the "observable" subspace (what A can produce), null space is the "invisible" subspace (what A collapses to zero). In ML: null space represents unidentifiable parameters (changing parameters in null space direction doesn't change predictions, so you can't distinguish them from data). Covariate shift or data leakage creates alignment with null space, making learning impossible. Understanding these spaces helps diagnose models: if you're predicting from perfectly collinear features, the corresponding directions lie in null space, making parameters unidentifiable without regularization.

---

### Q8: Define positive definite matrices and explain their role in ML.

**A:** A symmetric matrix A is positive definite (PD) if x^T Ax > 0 for all nonzero x ∈ ℝⁿ, positive semidefinite (PSD) if x^T Ax ≥ 0. Equivalent definitions: all eigenvalues are positive (PD) or non-negative (PSD), or A = B^T B for some full-rank B (PD), or all leading principal minors are positive (Sylvester criterion). PD/PSD matrices have special properties: (1) always invertible (PD) or have zero eigenvalues (PSD), (2) admit Cholesky decomposition A = LL^T (fast and stable), (3) A = B^T B means quadratic form x^T Ax measures "size" weighted by A, (4) curvature of function f is positive if Hessian is PD (local minimum), (5) covariance matrices are always PSD (and PD if non-singular). In ML: regularization term λw^T w adds PD penalty; Hessian at optimum being PD confirms local minimum. Second-order optimization methods (Newton, quasi-Newton) assume PD Hessian for well-defined step direction. In neural networks, checking Hessian positive definiteness validates convergence to minimum. Covariance matrices used in Gaussian processes, Bayesian inference, and whitening transformations are PSD; invertibility depends on having enough data (non-singular).

---

### Q9: Explain orthogonality and orthonormal bases and their utility.

**A:** Vectors u, v are orthogonal if u^T v = 0 (perpendicular in geometric sense). A set of vectors is orthonormal if all pairwise orthogonal and each has unit norm (||u|| = 1). Orthonormal basis {u₁, ..., uₙ} of ℝⁿ means any x = Σᵢ (u^T_i x) uᵢ (easy coefficient computation), and U^T U = I for matrix U with columns uᵢ. Orthogonal matrices (square with orthonormal columns) satisfy Q^T Q = I and Q^T = Q⁻¹ (inverse is transpose, trivial to compute). Orthogonality is numerically stable (condition number = 1) and preserves norms (||Qx|| = ||x||), making orthogonal transformations ideal for numerical computation. Advantages: (1) coefficients are inner products (fast to compute), (2) no rounding error in solving (norm-preserving), (3) invertibility automatic (Q^T is inverse), (4) basis change via orthogonal matrix doesn't distort geometry. In ML: whitening transformation uses orthogonal matrix to decorrelate features; principal components are orthonormal eigenvectors; neural network layers with orthogonal weight matrices have better conditioning (batch normalization, spectral normalization); attention mechanisms in transformers compute orthogonal projections. Gram-Schmidt algorithm converts any basis to orthonormal basis; QR decomposition produces orthonormal basis of column space.

---

### Q10: What is matrix calculus and how does it apply to ML optimization?

**A:** Matrix calculus extends single-variable calculus to matrices. Gradient of scalar f(X) w.r.t. matrix X is ∂f/∂X (matrix of partial derivatives). For quadratic form f(x) = x^T A x, ∇f = (A + A^T)x = 2Ax if A symmetric. For linear form f(x) = a^T x, ∇f = a. Jacobian of vector-valued function f: ℝⁿ → ℝᵐ is J ∈ ℝ^(m×n) with [J]ᵢⱼ = ∂fᵢ/∂xⱼ. Hessian of f: ℝⁿ → ℝ is H = ∇²f, second derivatives [H]ᵢⱼ = ∂²f/∂xᵢ∂xⱼ (symmetric). Chain rule: if z = f(g(x)), then dz/dx = (dz/dg)(dg/dx). Common rule: d(Ax)/dx = A^T (or A if computing denom-layout), d(x^T Ax)/dx = (A + A^T)x. Trace trick: tr(ABC) = tr(CAB) = tr(BCA), useful for rewriting matrix derivatives. In ML: computing ∇loss w.r.t. weights drives gradient descent (backpropagation in deep networks is chain rule applied), Hessian determines convergence rate of Newton's method. Understanding matrix calculus helps you: derive correct gradient formulas, implement autodiff correctly, understand why certain parameterizations optimize better than others (e.g., softmax parameterization avoids singular Hessians).

---

### Q11: How does SVD relate to PCA and how do you use it for dimensionality reduction?

**A:** Principal Component Analysis (PCA) finds directions of maximum variance in data. Given centered data X ∈ ℝ^(n×d) (n samples, d features), compute SVD: X = UΣV^T. Columns of V are principal components (eigenvectors of X^T X / (n-1), the sample covariance), singular values Σ relate to variance along each component (variance_i ≈ σ_i² / (n-1)), and U contains projections onto principal components. To reduce to k dimensions: keep top k columns of U and top k rows/columns of V, project X_reduced = X V_{:,1:k} = U_{:,1:k} Σ_{1:k,:}. Reconstruction: X̂ = X_reduced V_{:,1:k}^T = U_{:,1:k} Σ_{1:k,:} V_{:,1:k}^T. Choosing k: plot cumulative variance explained (Σᵢ₌₁^k σᵢ² / Σᵢ₌₁^d σᵢ²), pick k where ~95% variance is explained. Why SVD is preferred: (1) numerically stable vs. eigendecomposition of X^T X, (2) directly gives both components and projections, (3) works for rectangular matrices, (4) reveals effective dimensionality (steep singular value decay = low intrinsic dimension). SVD also enables low-rank approximation: A ≈ U_{:,1:k} Σ_{1:k} V_{:,1:k}^T with minimum Frobenius norm loss (best rank-k approximation). In production, precompute V and apply V_{:,1:k}^T to new samples for fast projection.

---

### Q12: Explain the role of matrix rank in neural networks and deep learning.

**A:** In neural networks, weight matrix rank determines expressive capacity: if weight W ∈ ℝ^(m×n) has rank < min(m,n), it maps ℝⁿ into a lower-dimensional subspace, creating a bottleneck. When hidden layer width h < input dimension d, that layer reduces dimensionality (rank ≤ h). Expressiveness requires rank = min(input_dim, hidden_dim); networks with insufficient rank can't represent complex functions. Over-parameterization (width >> required) provides redundancy that aids optimization: gradient descent finds minimum-norm solution (Implicit bias), which is more generalizable. Rank also appears in: (1) low-rank adaptation (LoRA): approximate weight updates as W ≈ W₀ + AB^T (low-rank) for parameter efficiency, (2) neural collapse: at convergence, class-wise means in hidden layer have high rank structure, (3) deep matrix factorization: representing weights as products recovers structure in data. Singular value distribution of weight matrices indicates layer conditioning: uniform distribution (all singular values ~same) is well-conditioned, while skewed distribution indicates numerical issues. Batch normalization implicitly maintains reasonable singular value spectrum, improving optimization. Understanding rank helps explain neural network phenomena: why certain widths matter, why low-rank approximations work, and how over-parameterization aids learning.

---

### Q13: How do you detect and handle ill-conditioning in linear systems?

**A:** A matrix is ill-conditioned if small changes in data cause large changes in solution. Condition number κ(A) = σ_max / σ_min (ratio of largest to smallest singular values) quantifies this: κ >> 1 means ill-conditioned (κ = 1 is perfectly conditioned). Solutions are ill-conditioned when A is nearly singular (small singular values) or columns are nearly linearly dependent. Detecting ill-conditioning: (1) compute SVD and examine singular values (steep dropoff near zero), (2) compute condition number (if κ > 10^10, very ill-conditioned), (3) solve Ax = b twice with slightly different b, see if solutions differ drastically, (4) eigenvalue distribution of A^T A is spread out. Handling ill-conditioning: (1) regularization: add λI to A^T A (ridge regression), shifts small singular values upward, (2) feature scaling: normalize features to similar magnitudes (condition number improves if data is scaled uniformly), (3) QR decomposition instead of normal equations (better numerical stability), (4) SVD with truncation: discard small singular values (pseudoinverse), (5) reformulate problem if possible (sometimes reparameterization makes it better-conditioned). In ML: multicollinear features cause ill-conditioning (regularization with L2 penalty adds λ to eigenvalues); covariance matrices with huge range of eigenvalues are ill-conditioned (data has features with very different scales). Preprocessing with feature standardization is a simple fix that improves conditioning dramatically.

---

### Q14: Explain trace and determinant and their interpretations.

**A:** Trace tr(A) = Σᵢ Aᵢᵢ (sum of diagonal) = Σᵢ λᵢ (sum of eigenvalues). Determinant det(A) = ∏ᵢ λᵢ (product of eigenvalues). Trace is linear: tr(A+B) = tr(A)+tr(B), and invariant under similarity: tr(A) = tr(P⁻¹AP). Determinant is multiplicative: det(AB) = det(A)det(B), and zero iff A is singular. Geometric interpretation: det(A) is the volume scaling factor when A transforms a unit volume (for positive determinant, orientation preserved; negative means flipped). Trace interpretation: tr(A) = total curvature in some sense (sum of all diagonal entries), appears in: trace of Hessian gives Laplacian (sum of second derivatives). In loss functions, tr(X^T Y) = Σᵢⱼ Xᵢⱼ Yᵢⱼ (Frobenius inner product); matrix calculus: ∂tr(AB)/∂A = B^T. In ML: (1) logdet(Σ) appears in Gaussian probability (negative of log-determinant is differential entropy), (2) trace is easier to compute than determinant, used in approximate inference, (3) tr(H) estimates generalization in overparameterized networks, (4) trace regularization: small tr(W^T W) prevents weights from growing (equivalent to L2 regularization). Understanding trace and determinant helps interpret loss functions and debug optimization (negative determinant means flipped orientation, sometimes signals non-convexity).

---

### Q15: How do you solve linear systems Ax = b using matrix decompositions?

**A:** Direct methods solve Ax = b exactly (up to numerical precision) via decomposition: (1) LU with partial pivoting (most general, O(n³)), compute A = PLU via Gaussian elimination with row swaps, solve Ly = P^T b via forward substitution, then Ux = y via back-substitution, (2) QR (best for least-squares with rectangular A), A = QR, solve Rx = Q^T b via back-substitution (numerically more stable than normal equations), (3) Cholesky (fast for symmetric positive definite A), A = LL^T, solve Ly = b and L^T x = y via forward/back substitution (cheapest, ~half cost of LU). For normal equations (least-squares Ax ≈ b), two approaches: (i) compute A^T Ax = A^T b directly (fast but less stable, condition number squared), (ii) compute QR of A, solve Rx = Q^T b (more stable, condition number unchanged). Iterative methods (Conjugate gradient, MINRES, GMRES) are preferred for sparse/large matrices: start with guess, iteratively improve, don't require full decomposition. In ML: solve normal equations for regression (QR preferred for stability), solve regularized system (A^T A + λI)x = A^T b for ridge regression (Cholesky if A^T A is PD), use iterative solvers for huge datasets. Choosing solver depends on: matrix properties (size, sparsity, conditioning), accuracy requirements, computation budget. Libraries like NumPy, SciPy use appropriate solvers automatically based on matrix structure.

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

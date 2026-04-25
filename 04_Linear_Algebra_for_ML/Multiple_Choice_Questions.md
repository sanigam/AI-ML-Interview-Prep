# Multiple Choice Questions: Linear Algebra for ML

📺 **Video Lecture:** https://youtu.be/YpckH7F5vj0


Test your understanding of linear algebra concepts essential for machine learning.

---

**Q1. The eigenvalues of a matrix A represent:**

A) The dimensions of the matrix  
B) The factors by which eigenvectors are scaled when multiplied by A  
C) The sum of each row of A  
D) The inverse of A

---

**Q2. In Principal Component Analysis (PCA), the principal components are:**

A) The rows of the original data matrix  
B) The eigenvectors of the data's covariance matrix, ordered by eigenvalue magnitude  
C) Randomly chosen directions in feature space  
D) The mean values of each feature

---

**Q3. A matrix is singular (non-invertible) when:**

A) All its eigenvalues are positive  
B) Its determinant is zero  
C) It is symmetric  
D) It has more rows than columns

---

**Q4. The rank of a matrix tells you:**

A) The total number of elements in the matrix  
B) The number of linearly independent rows or columns  
C) The largest element in the matrix  
D) The trace of the matrix

---

**Q5. In the Singular Value Decomposition (SVD) A = UΣV^T, the diagonal entries of Σ are:**

A) The eigenvalues of A  
B) The singular values of A (square roots of eigenvalues of A^T A)  
C) The determinant of A  
D) The column means of A

---

**Q6. Two vectors are orthogonal when:**

A) Their dot product is 1  
B) Their dot product is 0  
C) They have the same magnitude  
D) They point in the same direction

---

**Q7. The condition number of a matrix is important because:**

A) It measures the number of non-zero elements  
B) It indicates how sensitive the matrix inverse is to small perturbations in the input  
C) It determines whether the matrix is symmetric  
D) It equals the rank of the matrix

---

**Q8. What does it mean for a matrix to be positive semi-definite?**

A) All elements of the matrix are positive  
B) All eigenvalues are non-negative (≥ 0)  
C) The matrix is invertible  
D) The determinant is positive

---

**Q9. In linear regression, the normal equation X^T X β = X^T y can fail to produce a unique solution when:**

A) The dataset has too many samples  
B) X^T X is singular (features are linearly dependent)  
C) y contains only positive values  
D) X is a square matrix

---

**Q10. The trace of a matrix (sum of diagonal elements) equals:**

A) The determinant of the matrix  
B) The sum of its eigenvalues  
C) The rank of the matrix  
D) The number of rows

---

**Q11. The dot product of two vectors u and v can be interpreted geometrically as:**

A) The area of the parallelogram formed by u and v  
B) ||u|| × ||v|| × cos(θ), where θ is the angle between them  
C) The cross product of u and v  
D) The sum of u and v

---

**Q12. Adding a small value λ to the diagonal of X^T X (as in Ridge regression) helps because:**

A) It increases the rank of X  
B) It makes X^T X + λI invertible and improves numerical stability  
C) It removes all features from the model  
D) It guarantees zero training error

---

**Q13. In matrix multiplication AB, the result is defined only when:**

A) A and B have the same dimensions  
B) The number of columns of A equals the number of rows of B  
C) Both matrices are square  
D) A is the transpose of B

---

**Q14. The determinant of a 2×2 matrix [[a, b], [c, d]] is:**

A) a + d  
B) ad + bc  
C) ad − bc  
D) ac − bd

---

**Q15. Low-rank matrix approximation (keeping top-k singular values in SVD) is useful in ML for:**

A) Increasing the dimensionality of data  
B) Dimensionality reduction, noise reduction, and compression  
C) Making all eigenvalues equal  
D) Converting sparse matrices to dense matrices

---

## Answer Key

**Q1. Answer: B**
When Av = λv, λ is the eigenvalue and v is the eigenvector. The eigenvalue λ represents the scaling factor applied to the eigenvector when the matrix transformation is applied.

**Q2. Answer: B**
PCA finds directions of maximum variance by computing eigenvectors of the covariance matrix. The eigenvectors corresponding to the largest eigenvalues capture the most variance and become the principal components.

**Q3. Answer: B**
A singular matrix has determinant = 0, meaning it has at least one zero eigenvalue and its rows/columns are linearly dependent. Such matrices cannot be inverted.

**Q4. Answer: B**
The rank equals the number of linearly independent rows (or equivalently, columns). It tells you the effective dimensionality of the information in the matrix.

**Q5. Answer: B**
Singular values are the square roots of the eigenvalues of A^T A (or AA^T). They represent the "stretching factors" of the matrix transformation along its principal axes.

**Q6. Answer: B**
Orthogonal vectors have a dot product of zero, meaning they are perpendicular. In ML, orthogonal features carry independent information, which is desirable for model stability.

**Q7. Answer: B**
The condition number (ratio of largest to smallest singular value) measures numerical sensitivity. A high condition number means small input changes cause large output changes, leading to unstable computations.

**Q8. Answer: B**
A positive semi-definite matrix has all eigenvalues ≥ 0. Covariance matrices are always positive semi-definite. x^T A x ≥ 0 for all vectors x.

**Q9. Answer: B**
When features are linearly dependent (multicollinearity), X^T X becomes singular and cannot be inverted. This is why regularization (Ridge/Lasso) or feature selection is needed.

**Q10. Answer: B**
The trace equals the sum of eigenvalues: tr(A) = Σ λᵢ. This property is used extensively in matrix calculus and optimization for ML.

**Q11. Answer: B**
The dot product u · v = ||u|| × ||v|| × cos(θ) measures the projection of one vector onto another. When θ = 90°, cos(θ) = 0, confirming orthogonality means zero dot product.

**Q12. Answer: B**
Adding λI to X^T X ensures all eigenvalues are at least λ > 0, making the matrix invertible. This is the mathematical basis of Ridge regression and Tikhonov regularization.

**Q13. Answer: B**
For AB to be defined, the inner dimensions must match: if A is m×n, B must be n×p, producing an m×p result. This is a fundamental rule of matrix multiplication.

**Q14. Answer: C**
The determinant of [[a, b], [c, d]] is ad − bc. If this equals zero, the matrix is singular. The determinant represents the signed area scaling factor of the linear transformation.

**Q15. Answer: B**
Truncated SVD (keeping top-k singular values) provides the best rank-k approximation (Eckart-Young theorem). This is used for dimensionality reduction, denoising, and compression in recommender systems and NLP.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

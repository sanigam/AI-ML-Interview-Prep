# Multiple Choice Questions: SVMs and Kernel Methods

📺 **Video Lecture:** https://youtu.be/TU5ljcc07eA


Test your understanding of Support Vector Machines and kernel-based learning.

---

**Q1. The goal of a linear SVM is to find:**

A) A curve that overfits the training data  
B) The maximum-margin hyperplane that separates classes with the largest gap  
C) The hyperplane that passes through the most data points  
D) The hyperplane closest to the mean of the data

---

**Q2. Support vectors are:**

A) The features with highest importance  
B) All data points in the training set  
C) The data points closest to the decision boundary that define the margin  
D) The outliers in the dataset

---

**Q3. The C parameter in a soft-margin SVM controls:**

A) The dimensionality of the feature space  
B) The kernel type  
C) The number of support vectors  
D) The tradeoff between maximizing the margin and minimizing classification errors

---

**Q4. A large C value in SVM produces:**

A) A wider margin with more misclassifications allowed (more regularization)  
B) A linear decision boundary regardless of data  
C) Identical results to C = 0  
D) A narrower margin with fewer misclassifications (less regularization, risk of overfitting)

---

**Q5. The kernel trick allows SVMs to:**

A) Remove the need for support vectors  
B) Implicitly compute dot products in a high-dimensional feature space without explicitly transforming the data  
C) Always find a linear decision boundary  
D) Reduce the number of training samples

---

**Q6. The RBF (Gaussian) kernel K(x, y) = exp(−γ||x−y||²) maps data to:**

A) A finite-dimensional space with exactly 10 dimensions  
B) An infinite-dimensional feature space  
C) A 2-dimensional space  
D) The same space as the original data

---

**Q7. The γ (gamma) parameter in an RBF kernel controls:**

A) How far the influence of a single training example reaches (inverse of radius)  
B) The number of classes  
C) The bias term  
D) The margin width

---

**Q8. SVMs are particularly effective when:**

A) The data is entirely categorical  
B) The number of features is large relative to the number of samples  
C) The dataset has millions of samples  
D) No kernel is available

---

**Q9. In the dual formulation of SVM, the optimization depends on the data only through:**

A) The mean of each feature  
B) The variance of the target variable  
C) Dot products (or kernel evaluations) between pairs of data points  
D) The number of features

---

**Q10. Multi-class SVMs are typically implemented using:**

A) Softmax output on top of a single SVM  
B) One-vs-Rest (OVR) or One-vs-One (OVO) strategies combining binary classifiers  
C) Clustering algorithms  
D) A single hyperplane that separates all classes simultaneously

---

**Q11. Compared to logistic regression, SVMs:**

A) Require fewer hyperparameters  
B) Focus on the margin (boundary-defining points) rather than modeling probability for all points  
C) Cannot handle non-linear boundaries  
D) Always achieve higher accuracy

---

**Q12. The hinge loss used in SVMs is defined as:**

A) (y − f(x))²  
B) max(0, 1 − y × f(x)), penalizing points within the margin or misclassified  
C) |y − f(x)|  
D) −y × log(f(x))

---

**Q13. Feature scaling is important for SVMs because:**

A) SVMs internally normalize all features  
B) SVMs only work with integer features  
C) Feature scaling changes the number of support vectors to exactly 3  
D) The distance calculations in the kernel are affected by feature scales

---

**Q14. A polynomial kernel of degree d allows the SVM to learn:**

A) Only circular boundaries  
B) Decision boundaries that are polynomials of degree d in the original features  
C) Only linear boundaries  
D) Infinite-dimensional representations

---

**Q15. The margin in SVM is defined as:**

A) The total number of misclassified points  
B) The value of the bias term  
C) The number of support vectors  
D) The distance between the decision boundary and the nearest data point(s) from either class (2/||w|| for the full margin)

---

## Answer Key

**Q1. Answer: B**
SVM seeks the hyperplane that maximizes the margin — the distance between the boundary and the nearest points from each class. This maximum-margin principle leads to better generalization.

**Q2. Answer: C**
Support vectors are the critical training points that lie on the margin boundaries. They alone determine the decision boundary — removing non-support-vector points doesn't change the model.

**Q3. Answer: D**
C balances margin width vs. classification accuracy. Small C = wide margin (more errors allowed, less overfitting). Large C = narrow margin (fewer errors, risk of overfitting).

**Q4. Answer: D**
Large C heavily penalizes misclassifications, forcing a tight boundary that classifies nearly all training points correctly. This can overfit by being too sensitive to noise.

**Q5. Answer: B**
The kernel trick computes K(x,y) = φ(x)·φ(y) without explicitly computing the (potentially infinite-dimensional) mapping φ. This makes non-linear SVM computationally feasible.

**Q6. Answer: B**
The RBF kernel corresponds to mapping data to an infinite-dimensional Hilbert space. Despite this, computations remain tractable because we only need kernel evaluations, not explicit coordinates.

**Q7. Answer: A**
High γ means each point has only local influence (complex boundary, risk of overfitting). Low γ means broader influence (smoother boundary, risk of underfitting). γ = 1/(2σ²).

**Q8. Answer: B**
SVMs work well in high-dimensional spaces (text classification, genomics) where the number of features exceeds samples. The kernel trick enables effective learning without the curse of dimensionality.

**Q9. Answer: C**
The dual SVM formulation involves only inner products xᵢ·xⱼ between training points. Replacing these with kernel values K(xᵢ, xⱼ) enables non-linear classification — this is the kernel trick.

**Q10. Answer: B**
Since SVMs are inherently binary classifiers, multi-class is handled by OVR (k binary classifiers, one per class vs rest) or OVO (k(k-1)/2 classifiers, one for each pair).

**Q11. Answer: B**
SVMs care only about points near the boundary (support vectors), optimizing the margin. Logistic regression considers all points and models the full probability. This makes SVMs efficient but they don't output calibrated probabilities directly.

**Q12. Answer: B**
Hinge loss = max(0, 1 − y·f(x)). It's zero when a point is correctly classified with sufficient margin, and increases linearly for violations. This drives the maximum-margin property.

**Q13. Answer: D**
Kernels (especially RBF) use distances between points. If one feature ranges 0-1000 and another 0-1, the first dominates the distance calculation. Scaling ensures all features contribute proportionally.

**Q14. Answer: B**
A polynomial kernel K(x,y) = (x·y + c)^d implicitly creates all polynomial feature combinations up to degree d, allowing the SVM to learn polynomial decision boundaries.

**Q15. Answer: D**
The margin is the perpendicular distance from the decision boundary to the closest point(s). For the full margin (both sides): margin = 2/||w||. Maximizing the margin is the SVM objective.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

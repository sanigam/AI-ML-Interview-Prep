# Multiple Choice Questions: Bias-Variance Tradeoff and Regularization

📺 **Video Lecture:** https://youtu.be/VsvnNYoMDZY


Test your understanding of overfitting, underfitting, and regularization techniques.

---

**Q1. The bias-variance decomposition states that total prediction error equals:**

A) Only variance  
B) Bias × Variance  
C) Bias² + Variance + Irreducible Noise  
D) Bias + Variance

---

**Q2. A model with high bias and low variance is:**

A) Overfitting  
B) Always the best choice  
C) Perfectly fitted  
D) Underfitting — too simple to capture the underlying pattern

---

**Q3. A model with low bias and high variance is:**

A) Underfitting  
B) Ignoring the data  
C) Overfitting — too sensitive to specific training data  
D) Always the worst choice

---

**Q4. Increasing model complexity (e.g., more polynomial degree, deeper tree) typically:**

A) Decreases bias but increases variance  
B) Increases both bias and variance  
C) Decreases both bias and variance  
D) Has no effect

---

**Q5. L2 regularization (Ridge) adds which penalty to the loss function?**

A) λ × Σ|βⱼ|  
B) λ × max(βⱼ)  
C) λ × Σβⱼ²  
D) λ × Σβⱼ

---

**Q6. L1 regularization (Lasso) produces sparse models because:**

A) It always selects exactly half the features  
B) The absolute value penalty has a sharp corner at zero, allowing coefficients to become exactly zero  
C) It removes data points  
D) It squares all coefficients

---

**Q7. Increasing the regularization parameter λ:**

A) Increases model simplicity (stronger regularization), increasing bias but decreasing variance  
B) Makes the model more complex  
C) Always improves test accuracy  
D) Has no effect on the model

---

**Q8. Dropout in neural networks acts as regularization by:**

A) Increasing the learning rate  
B) Adding extra layers  
C) Removing layers permanently  
D) Randomly setting a fraction of neurons to zero during training, forcing the network to be redundant

---

**Q9. Early stopping prevents overfitting by:**

A) Training for a fixed number of epochs regardless of performance  
B) Removing the validation set  
C) Using only the first 10% of training data  
D) Halting training when validation performance stops improving

---

**Q10. Cross-validation helps detect overfitting because:**

A) It trains and tests on the same data  
B) It eliminates all variance  
C) It increases the training set size  
D) It evaluates the model on data not used for training, revealing the generalization gap

---

**Q11. Adding more training data typically reduces:**

A) Variance (more data stabilizes parameter estimates)  
B) Bias  
C) The number of features needed  
D) Irreducible noise

---

**Q12. Data augmentation (e.g., flipping images, adding noise) helps reduce overfitting by:**

A) Removing samples from the training set  
B) Making the model more complex  
C) Reducing the number of parameters  
D) Effectively increasing training set diversity without collecting new data

---

**Q13. Batch normalization helps with regularization because:**

A) It removes all features  
B) It fixes the learning rate  
C) It eliminates the need for a validation set  
D) Each mini-batch introduces slight noise in the mean/variance estimates, acting as a regularizer

---

**Q14. The validation set is used for:**

A) Replacing cross-validation entirely  
B) Hyperparameter tuning and model selection (separate from the test set used for final evaluation)  
C) Training the model  
D) Final performance reporting

---

**Q15. Weight decay in neural networks is mathematically equivalent to:**

A) Dropout  
B) L2 regularization (adding λ||w||² to the loss)  
C) L1 regularization  
D) Increasing the learning rate

---

## Answer Key

**Q1. Answer: C**
Error = Bias² + Variance + σ² (irreducible noise). Bias measures systematic error from wrong assumptions; variance measures sensitivity to training data; noise is inherent randomness.

**Q2. Answer: D**
High bias = the model is too simple (e.g., fitting a line to curved data). It consistently misses the pattern regardless of training data — this is underfitting.

**Q3. Answer: C**
High variance = the model is too complex, fitting noise in the training data. It performs well on training data but poorly on new data — this is overfitting.

**Q4. Answer: A**
More complexity lets the model fit training data better (lower bias) but makes it more sensitive to particular training samples (higher variance). The goal is finding the sweet spot.

**Q5. Answer: C**
Ridge penalty = λ × Σβⱼ² pushes all coefficients toward zero but rarely to exactly zero. It's equivalent to a Gaussian prior on coefficients in Bayesian terms.

**Q6. Answer: B**
L1's absolute value penalty creates a diamond-shaped constraint region that touches coordinate axes, making it geometrically possible for solutions to land exactly at zero for some coefficients.

**Q7. Answer: A**
Larger λ means stronger regularization: coefficients are shrunk more aggressively. This increases bias (may miss patterns) but decreases variance (more stable predictions).

**Q8. Answer: D**
Dropout randomly zeros out neurons during training, preventing co-adaptation. The network learns redundant representations, making it more robust. At test time, all neurons are used with scaled weights.

**Q9. Answer: D**
Early stopping monitors validation loss and stops when it starts increasing (even if training loss continues decreasing). This prevents the model from memorizing training noise.

**Q10. Answer: D**
CV evaluates on held-out folds, revealing how well the model generalizes. A big gap between training and CV scores indicates overfitting that wouldn't be visible from training metrics alone.

**Q11. Answer: A**
More data better constrains parameter estimates, reducing their sensitivity to any particular sample. The model converges to a more stable solution, reducing variance.

**Q12. Answer: D**
Augmentation creates synthetic variations of existing data, exposing the model to more diverse examples. This improves generalization without the cost of collecting new real data.

**Q13. Answer: D**
Since batch normalization computes statistics per mini-batch (not the full dataset), each batch introduces small noise, providing a regularization effect similar to dropout.

**Q14. Answer: B**
The validation set tunes hyperparameters (learning rate, regularization, model architecture). The test set is only used once at the end for unbiased final performance estimation.

**Q15. Answer: B**
Weight decay subtracts λ×w from weights each step, which is equivalent to adding λ||w||² to the loss function — this is L2 regularization.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

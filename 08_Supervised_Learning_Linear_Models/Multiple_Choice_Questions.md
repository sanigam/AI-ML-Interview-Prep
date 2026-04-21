# Multiple Choice Questions: Supervised Learning — Linear Models

Test your understanding of linear regression, logistic regression, and related models.

---

**Q1. In ordinary least squares (OLS) linear regression, the objective is to minimize:**

A) The sum of absolute residuals
B) The sum of squared residuals (differences between predicted and actual values)
C) The maximum residual
D) The number of non-zero coefficients

---

**Q2. The coefficient β₁ in a simple linear regression y = β₀ + β₁x represents:**

A) The predicted value when x = 0
B) The expected change in y for a one-unit increase in x
C) The correlation between x and y
D) The variance of y

---

**Q3. Logistic regression models the probability of a binary outcome using:**

A) A linear function with output between −∞ and +∞
B) The sigmoid (logistic) function, mapping linear combination to [0, 1]
C) A step function
D) A polynomial of degree 3

---

**Q4. The loss function used in logistic regression is:**

A) Mean Squared Error
B) Binary Cross-Entropy (log loss)
C) Hinge loss
D) Huber loss

---

**Q5. Multicollinearity in linear regression causes:**

A) The model to always overfit
B) Unstable and unreliable coefficient estimates with high variance
C) The R² to become zero
D) The model to become non-linear

---

**Q6. The R² (coefficient of determination) value of 0.85 means:**

A) The model is 85% accurate
B) 85% of the variance in the target variable is explained by the model
C) 85% of predictions are exactly correct
D) The model has 85 features

---

**Q7. Ridge regression (L2 regularization) addresses overfitting by:**

A) Removing features from the model entirely
B) Adding a penalty proportional to the sum of squared coefficients, shrinking them toward zero
C) Increasing the number of features
D) Using a non-linear kernel

---

**Q8. Lasso regression (L1 regularization) differs from Ridge in that Lasso:**

A) Never shrinks any coefficients
B) Can shrink coefficients exactly to zero, performing automatic feature selection
C) Always outperforms Ridge
D) Does not have a regularization parameter

---

**Q9. The assumptions of OLS linear regression include all EXCEPT:**

A) Linearity of the relationship between features and target
B) Independence of residuals
C) The target variable must be categorical
D) Homoscedasticity (constant variance of residuals)

---

**Q10. In logistic regression, the odds ratio exp(β₁) represents:**

A) The probability of the positive class
B) The multiplicative change in odds for a one-unit increase in the corresponding feature
C) The number of correct predictions
D) The threshold for classification

---

**Q11. Elastic Net regularization combines:**

A) L1 and L2 penalties, offering a balance between Lasso and Ridge
B) Gradient descent and Newton's method
C) Linear and polynomial regression
D) Cross-validation and grid search

---

**Q12. Adjusted R² is preferred over R² when comparing models because:**

A) It is always higher than R²
B) It penalizes adding features that don't improve the model, preventing over-counting from added complexity
C) It only works for logistic regression
D) It requires less computation

---

**Q13. Heteroscedasticity (non-constant variance of residuals) in linear regression leads to:**

A) Biased coefficient estimates
B) Unreliable standard errors and confidence intervals for coefficients
C) R² becoming negative
D) The model becoming non-linear

---

**Q14. The softmax function extends logistic regression to multi-class by:**

A) Running binary logistic regression multiple times independently
B) Converting a vector of raw scores into a probability distribution over k classes (summing to 1)
C) Using separate thresholds for each class
D) Reducing the problem to binary classification

---

**Q15. In linear regression, the residuals should ideally be:**

A) Correlated with the predicted values
B) Independently and identically distributed with mean zero (normally distributed for inference)
C) All exactly zero
D) Increasing with the predicted values

---

## Answer Key

**Q1. Answer: B**
OLS minimizes Σ(yᵢ − ŷᵢ)², the sum of squared residuals. This has a closed-form solution via the normal equations: β = (X^TX)⁻¹X^Ty.

**Q2. Answer: B**
β₁ is the slope — it represents the expected change in y per unit increase in x, holding all other variables constant (in multiple regression).

**Q3. Answer: B**
Logistic regression applies the sigmoid σ(z) = 1/(1+e⁻ᶻ) to the linear combination z = β₀ + β₁x₁ + ..., ensuring output is a valid probability in [0, 1].

**Q4. Answer: B**
Binary cross-entropy L = −[y log(p) + (1−y) log(1−p)] is the natural loss for logistic regression. It is derived from maximum likelihood estimation of Bernoulli outcomes.

**Q5. Answer: B**
When features are highly correlated, small data changes cause large swings in coefficients. Individual coefficients become unreliable, though overall predictions may still be acceptable.

**Q6. Answer: B**
R² = 1 − SS_res/SS_tot measures the proportion of target variance explained by the model. R² = 0.85 means the model explains 85% of variance; the remaining 15% is unexplained.

**Q7. Answer: B**
Ridge adds λΣβⱼ² to the loss, penalizing large coefficients. This shrinks all coefficients toward (but not exactly to) zero, reducing overfitting when features are correlated.

**Q8. Answer: B**
L1 penalty (λΣ|βⱼ|) has a diamond-shaped constraint region that touches axes, allowing coefficients to become exactly zero. This makes Lasso useful for feature selection.

**Q9. Answer: C**
OLS assumes: linearity, independence of errors, homoscedasticity, normality of residuals (for inference). The target must be continuous, not categorical — categorical targets use logistic regression.

**Q10. Answer: B**
exp(β₁) is the odds ratio: a one-unit increase in the feature multiplies the odds of the positive class by exp(β₁). If β₁ = 0.5, the odds increase by a factor of e^0.5 ≈ 1.65.

**Q11. Answer: A**
Elastic Net penalty = α×L1 + (1−α)×L2, combining Lasso's sparsity with Ridge's stability for correlated features. The mixing parameter α controls the balance.

**Q12. Answer: B**
Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1), which decreases when adding uninformative features. Unlike R² which never decreases with added features, adjusted R² penalizes unnecessary complexity.

**Q13. Answer: B**
Heteroscedasticity doesn't bias coefficients but makes standard errors incorrect, leading to invalid confidence intervals and hypothesis tests. Weighted least squares or robust standard errors fix this.

**Q14. Answer: B**
Softmax converts logits z₁,...,zₖ into probabilities: P(class j) = e^zⱼ / Σe^zᵢ. All probabilities are positive and sum to 1, generalizing the sigmoid to multiple classes.

**Q15. Answer: B**
Well-behaved residuals are independent, homoscedastic, and normally distributed around zero. Patterns in residuals (e.g., funnel shape, curvature) indicate model misspecification.

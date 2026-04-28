# Multiple Choice Questions: Supervised Learning — Linear Models

📺 **Video Lecture:** https://youtu.be/v9OmF4GFaqw


Test your understanding of linear regression, logistic regression, and related models.

---

**Q1. In ordinary least squares (OLS) linear regression, the objective is to minimize:**

A) The sum of absolute residuals  
B) The maximum residual  
C) The sum of squared residuals (differences between predicted and actual values)  
D) The number of non-zero coefficients

---

**Q2. The coefficient β₁ in a simple linear regression y = β₀ + β₁x represents:**

A) The predicted value when x = 0  
B) The variance of y  
C) The correlation between x and y  
D) The expected change in y for a one-unit increase in x

---

**Q3. Logistic regression models the probability of a binary outcome using:**

A) A polynomial of degree 3  
B) The sigmoid (logistic) function, mapping linear combination to [0, 1]  
C) A linear function with output between −∞ and +∞  
D) A step function

---

**Q4. The loss function used in logistic regression is:**

A) Huber loss  
B) Mean Squared Error  
C) Hinge loss  
D) Binary Cross-Entropy (log loss)

---

**Q5. Multicollinearity in linear regression causes:**

A) The R² to become zero  
B) The model to always overfit  
C) The model to become non-linear  
D) Unstable and unreliable coefficient estimates with high variance

---

**Q6. The R² (coefficient of determination) value of 0.85 means:**

A) The model has 85 features  
B) The model is 85% accurate  
C) 85% of the variance in the target variable is explained by the model  
D) 85% of predictions are exactly correct

---

**Q7. Ridge regression (L2 regularization) addresses overfitting by:**

A) Increasing the number of features  
B) Using a non-linear kernel  
C) Removing features from the model entirely  
D) Adding a penalty proportional to the sum of squared coefficients, shrinking them toward zero

---

**Q8. Lasso regression (L1 regularization) differs from Ridge in that Lasso:**

A) Can shrink coefficients exactly to zero, performing automatic feature selection  
B) Never shrinks any coefficients  
C) Always outperforms Ridge  
D) Does not have a regularization parameter

---

**Q9. The assumptions of OLS linear regression include all EXCEPT:**

A) Homoscedasticity (constant variance of residuals)  
B) Independence of residuals  
C) The target variable must be categorical  
D) Linearity of the relationship between features and target

---

**Q10. In logistic regression, the odds ratio exp(β₁) represents:**

A) The multiplicative change in odds for a one-unit increase in the corresponding feature  
B) The number of correct predictions  
C) The probability of the positive class  
D) The threshold for classification

---

**Q11. Elastic Net regularization combines:**

A) L1 and L2 penalties, offering a balance between Lasso and Ridge  
B) Cross-validation and grid search  
C) Gradient descent and Newton's method  
D) Linear and polynomial regression

---

**Q12. Adjusted R² is preferred over R² when comparing models because:**

A) It only works for logistic regression  
B) It penalizes adding features that don't improve the model, preventing over-counting from added complexity  
C) It requires less computation  
D) It is always higher than R²

---

**Q13. Heteroscedasticity (non-constant variance of residuals) in linear regression leads to:**

A) Biased coefficient estimates  
B) Unreliable standard errors and confidence intervals for coefficients  
C) The model becoming non-linear  
D) R² becoming negative

---

**Q14. The softmax function extends logistic regression to multi-class by:**

A) Using separate thresholds for each class  
B) Running binary logistic regression multiple times independently  
C) Reducing the problem to binary classification  
D) Converting a vector of raw scores into a probability distribution over k classes (summing to 1)

---

**Q15. In linear regression, the residuals should ideally be:**

A) All exactly zero  
B) Increasing with the predicted values  
C) Independently and identically distributed with mean zero (normally distributed for inference)  
D) Correlated with the predicted values

---

## Answer Key

**Q1. Answer: C**
OLS minimizes Σ(yᵢ − ŷᵢ)², the sum of squared residuals. This has a closed-form solution via the normal equations: β = (X^TX)⁻¹X^Ty.

**Q2. Answer: D**
β₁ is the slope — it represents the expected change in y per unit increase in x, holding all other variables constant (in multiple regression).

**Q3. Answer: B**
Logistic regression applies the sigmoid σ(z) = 1/(1+e⁻ᶻ) to the linear combination z = β₀ + β₁x₁ + ..., ensuring output is a valid probability in [0, 1].

**Q4. Answer: D**
Binary cross-entropy L = −[y log(p) + (1−y) log(1−p)] is the natural loss for logistic regression. It is derived from maximum likelihood estimation of Bernoulli outcomes.

**Q5. Answer: D**
When features are highly correlated, small data changes cause large swings in coefficients. Individual coefficients become unreliable, though overall predictions may still be acceptable.

**Q6. Answer: C**
R² = 1 − SS_res/SS_tot measures the proportion of target variance explained by the model. R² = 0.85 means the model explains 85% of variance; the remaining 15% is unexplained.

**Q7. Answer: D**
Ridge adds λΣβⱼ² to the loss, penalizing large coefficients. This shrinks all coefficients toward (but not exactly to) zero, reducing overfitting when features are correlated.

**Q8. Answer: A**
L1 penalty (λΣ|βⱼ|) has a diamond-shaped constraint region that touches axes, allowing coefficients to become exactly zero. This makes Lasso useful for feature selection.

**Q9. Answer: C**
OLS assumes: linearity, independence of errors, homoscedasticity, normality of residuals (for inference). The target must be continuous, not categorical — categorical targets use logistic regression.

**Q10. Answer: A**
exp(β₁) is the odds ratio: a one-unit increase in the feature multiplies the odds of the positive class by exp(β₁). If β₁ = 0.5, the odds increase by a factor of e^0.5 ≈ 1.65.

**Q11. Answer: A**
Elastic Net penalty = α×L1 + (1−α)×L2, combining Lasso's sparsity with Ridge's stability for correlated features. The mixing parameter α controls the balance.

**Q12. Answer: B**
Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1), which decreases when adding uninformative features. Unlike R² which never decreases with added features, adjusted R² penalizes unnecessary complexity.

**Q13. Answer: B**
Heteroscedasticity doesn't bias coefficients but makes standard errors incorrect, leading to invalid confidence intervals and hypothesis tests. Weighted least squares or robust standard errors fix this.

**Q14. Answer: D**
Softmax converts logits z₁,...,zₖ into probabilities: P(class j) = e^zⱼ / Σe^zᵢ. All probabilities are positive and sum to 1, generalizing the sigmoid to multiple classes.

**Q15. Answer: C**
Well-behaved residuals are independent, homoscedastic, and normally distributed around zero. Patterns in residuals (e.g., funnel shape, curvature) indicate model misspecification.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

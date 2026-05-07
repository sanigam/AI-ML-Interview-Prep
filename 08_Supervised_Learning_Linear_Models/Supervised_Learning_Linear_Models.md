# Supervised Learning: Linear Models

📺 **Video Lecture:** https://youtu.be/v9OmF4GFaqw

## Interview Anchor
- **Linear Regression:** Predicts continuous values using a linear relationship between input features and output.
- **Logistic Regression:** Binary/multiclass classifier that models probability using the logistic sigmoid function.
- **Regularization:** Techniques (Ridge, Lasso, Elastic Net) that penalize model complexity to prevent overfitting.

## Key Concepts Overview

Linear models form the foundation of machine learning and are frequently tested in interviews because they're interpretable, fast, and theoretically grounded. Interviewers assess whether you understand the assumptions underlying these models, how regularization addresses overfitting, and the mathematical details of fitting procedures. Linear models are also a critical baseline—any complex model should outperform them meaningfully. This topic tests both your mathematical depth (normal equation, maximum likelihood) and practical understanding (multicollinearity, coefficient interpretation, when linear models succeed or fail).

---

### Q1: Explain linear regression and its underlying assumptions.

**A:** **Linear regression** models a continuous output as a linear function of input features:

```
y = X·β + ε,    ε ~ Normal(0, σ²·I)
```

where β are the coefficients and ε is normally distributed noise.

**Five classical assumptions:**

- **Linearity** — the relationship between features and target is genuinely linear.
- **Independence** — observations are independent of each other.
- **Homoscedasticity** — error variance is constant across all input values.
- **Normality of residuals** — errors follow a normal distribution.
- **No multicollinearity** — features are not highly correlated with each other.

**Detecting violations:** residual plots are the workhorse — systematic patterns in residuals reveal nonlinearity, a funnel shape suggests heteroscedasticity, and Q-Q plots check normality.

**Fixes:** apply transformations (log, sqrt), add polynomial or interaction features, use weighted least squares for heteroscedasticity, or move to a non-linear model when assumptions are badly broken.

Residual analysis is a critical, often-skipped part of model validation — it's where you find out whether linear regression is the right tool for the job.

---

### Q2: What is the closed-form solution (normal equation) for linear regression, and when might you use it versus gradient descent?

**A:** The **normal equation** gives the exact least-squares solution in one shot:

```
β = (Xᵀ·X)⁻¹ · Xᵀ·y
```

No iteration, no learning rate — just one matrix computation.

**When the normal equation works well:** small to moderate problems (a few thousand features), where you want an exact solution and interpretability matters.

**Drawbacks:**

- Computing the inverse is O(d³) in the number of features.
- Numerically unstable when XᵀX is ill-conditioned (near-singular due to multicollinearity).
- All data has to fit in memory.

**When to prefer gradient descent (or its variants):**

- Very large datasets where the matrix inverse is impractical.
- Streaming or online data.
- When you need a parallelizable training loop.

In practice, production ML libraries usually solve linear regression via **QR decomposition** rather than computing the inverse explicitly — same answer, much better numerical stability. Use the normal equation as the textbook default for small problems; reach for gradient descent (or SGD) when scale demands it.

---

### Q3: Explain multicollinearity: what causes it, why it's problematic, and how to detect and address it.

**A:** **Multicollinearity** is high correlation among input features, violating the implicit independence assumption in linear regression. The consequences:

- **Inflated coefficient variance** — estimates become unstable and unreliable.
- **Unpredictable coefficient swings** — small data changes can flip signs or shift magnitudes drastically.
- **Lost interpretability** — you can't isolate individual feature effects when multiple features carry the same information.

**Detection methods:**

- **Correlation matrix** — flag pairs with |r| > 0.8.
- **Variance Inflation Factor (VIF)** — values above 5–10 indicate problematic multicollinearity. VIF for feature i is computed by regressing it on all other features and using 1/(1 − R²ᵢ).
- **Eigenvalues of XᵀX** — very small eigenvalues signal multicollinearity (a near-singular design matrix).

**Fixes:**

- Drop redundant features.
- Combine correlated features via PCA or averaging.
- Use regularization (Ridge or Lasso).
- Collect more diverse data.

**Regularization is often preferred** because it doesn't throw out information — it shrinks coefficients proportionally. Ridge handles correlated features especially well; Lasso may arbitrarily pick one of several correlated features and zero the rest.

---

### Q4: Describe Ridge, Lasso, and Elastic Net regularization. What are their differences and use cases?

**A:** All three add a penalty on the coefficients to the squared-error loss, but the choice of penalty changes the behavior dramatically.

**Ridge (L2):**

```
loss = || y − X·β ||²  +  λ · Σᵢ βᵢ²
```

Shrinks all coefficients proportionally toward zero but never exactly to zero. Best when most features are relevant and multicollinearity is severe.

**Lasso (L1):**

```
loss = || y − X·β ||²  +  λ · Σᵢ |βᵢ|
```

Drives some coefficients exactly to zero, performing automatic feature selection. Ideal when you suspect many features are irrelevant.

**Elastic Net:**

```
loss = || y − X·β ||²  +  λ₁ · Σᵢ |βᵢ|  +  λ₂ · Σᵢ βᵢ²
```

Combines L1 and L2 — gets Lasso's sparsity with Ridge's stability under correlation. Useful when you want both feature selection and graceful handling of correlated features.

**Tuning:** the regularization strength λ is chosen by cross-validation. Larger λ increases bias but reduces variance — the classic bias-variance tradeoff knob.

**Choice rule of thumb:**

- Ridge — many small effects, severe multicollinearity.
- Lasso — sparse signal, interpretability matters, many irrelevant features.
- Elastic Net — sparse signal *and* correlated features, or when in doubt.

All three reduce overfitting by controlling effective model complexity.

---

### Q5: What is the difference between R-squared and adjusted R-squared?

**A:** **R²** is the proportion of variance in y that the model explains:

```
R² = 1  −  (SS_res / SS_tot)
```

It ranges from 0 to 1, with higher being better. The catch: R² is *monotonically non-decreasing* in the number of features — adding any feature can only keep R² the same or increase it, even if the feature is pure noise.

**Adjusted R²** corrects for this by penalizing model complexity:

```
Adj R² = 1  −  (1 − R²) · (n − 1) / (n − p − 1)
```

where n is sample size and p is the number of predictors. Adjusted R² *decreases* if a new feature doesn't improve the fit enough to justify its complexity. Adding a random feature will typically push R² slightly up but adjusted R² down.

**Practical guidance:** adjusted R² is the better metric for comparing models with different numbers of features, but for honest performance assessment, **test-set performance** (or cross-validated metrics) is always preferable. Both R² metrics are limited if your real concern is generalization rather than in-sample fit.

---

### Q6: Explain how logistic regression works and the role of the sigmoid function.

**A:** **Logistic regression** models the probability of a binary outcome using the sigmoid (logistic) function:

```
z       = X · β
P(y=1 | X) = σ(z) = 1 / ( 1 + e^(−z) )
```

The sigmoid maps any real number z to a probability in (0, 1), so outputs are always valid probabilities. The decision boundary is where the probability is 0.5, equivalently where z = 0 — a linear hyperplane in feature space.

Unlike linear regression, logistic regression is fit with **maximum likelihood estimation (MLE)** rather than least squares (more on that in Q8).

**Multiclass extension — softmax regression.** For K classes:

```
P(y = k | X) = e^(z_k) / Σⱼ e^(z_j),    z_k = X · β_k
```

Each class gets its own coefficient vector. When K = 2, this reduces to ordinary logistic regression.

**Interpretability.** Each exp(βᵢ) is an odds ratio — a unit increase in feature i multiplies the odds of class 1 by exp(βᵢ). This is why logistic regression remains a workhorse in regulated and interpretability-sensitive domains (medical, finance) — it's simple, fast, and outputs calibrated probabilities by design.

---

### Q7: What is the odds ratio in logistic regression, and how do you interpret coefficients?

**A:** In logistic regression, the **odds** of class 1 are:

```
odds = P(y=1) / P(y=0) = exp(z) = exp(X · β)
```

A one-unit increase in feature i multiplies the odds by:

```
odds ratio for feature i = exp(βᵢ)
```

So if βᵢ = 0.2, then exp(0.2) ≈ 1.22 — a 22% increase in odds per unit increase in that feature. Positive βᵢ raises the odds of class 1; negative lowers them.

**Categorical features.** Encode as one-hot (dummy) variables with one baseline category dropped. Each remaining coefficient is a log-odds ratio relative to the baseline.

**Confidence intervals.** Standard errors of coefficients come from the inverse Hessian at the MLE solution. Significance tests typically use the **Wald statistic** (β̂ / SE).

**Worked example.** If βᵢ = 0.5 with SE = 0.1:

- 95% CI for β is roughly [0.304, 0.696].
- Exponentiating gives the odds-ratio CI: [exp(0.304), exp(0.696)] ≈ [1.36, 2.00].

That interpretability — speaking in terms of odds ratios rather than raw probabilities — is why logistic regression remains popular for explainability-critical applications.

---

### Q8: Explain maximum likelihood estimation (MLE) for logistic regression. Why not use least squares?

**A:** **MLE for logistic regression** maximizes the likelihood of the observed labels under the Bernoulli model:

```
L(β) = Πᵢ Pᵢ^(yᵢ) · (1 − Pᵢ)^(1 − yᵢ),    where Pᵢ = σ(Xᵢ · β)
```

Equivalently, minimize the negative log-likelihood — the **cross-entropy loss**:

```
loss(β) = − Σᵢ [ yᵢ · log(Pᵢ)  +  (1 − yᵢ) · log(1 − Pᵢ) ]
```

**Why not least squares?**

- Binary outcomes are not normally distributed — least squares assumes Gaussian errors.
- Linear least squares can predict probabilities outside [0, 1], which is meaningless.
- Cross-entropy + sigmoid produces a *convex* optimization problem with calibrated probabilities; squared error on σ(Xβ) is non-convex and harder to optimize.

**Optimization.** No closed-form solution — solved iteratively via:

- **Newton-Raphson / IRLS** (second-order, fast for moderate problems).
- **Gradient descent / SGD** (first-order, scales to large problems).

The Hessian at the optimum gives standard errors and confidence intervals for free.

**Regularization.** L1, L2, and Elastic Net naturally extend MLE — just add the penalty to the negative log-likelihood. This is the principled framework behind regularized logistic regression.

In interviews, framing MLE as "the right loss for the data distribution" rather than "an alternative to least squares" shows theoretical maturity.

---

### Q9: What is softmax regression and how does it generalize logistic regression to multiclass problems?

**A:** **Softmax regression** (a.k.a. multinomial logistic regression) extends logistic regression to K ≥ 2 classes. Each class has its own coefficient vector β_k:

```
z_k        = X · β_k
P(y=k | X) = exp(z_k) / Σⱼ exp(z_j)
```

This gives a proper probability distribution over K classes. The prediction is `argmax_k P(y=k | X)`.

When K = 2, softmax reduces to ordinary logistic regression (the redundant parameters are dropped or one class fixed as reference).

**Training via MLE.** Minimize **categorical cross-entropy** with one-hot labels:

```
loss = − Σᵢ Σ_k  y_{i,k} · log P(y=k | Xᵢ)
```

For each example, only the term for the true class contributes — so this is just the negative log-probability assigned to the correct class.

**Softmax vs one-vs-rest.** Softmax models the *joint* probability over all classes, which is theoretically cleaner and produces calibrated probabilities. One-vs-rest trains K independent binary classifiers; the resulting per-class probabilities don't naturally sum to 1.

**Important assumption:** softmax assumes **mutually exclusive** classes — exactly one is correct. For *multilabel* problems where multiple classes can be true simultaneously, use independent sigmoids per class instead, with binary cross-entropy on each.

---

### Q10: Describe polynomial regression and when you'd use it instead of linear regression.

**A:** **Polynomial regression** adds polynomial features (X², X³, ...) to linear regression. The model becomes:

```
y = β₀  +  β₁·X  +  β₂·X²  +  ...  +  β_d·X^d  +  ε
```

It can fit nonlinear relationships while remaining *linear in the parameters* — so it's still trained with the same machinery as ordinary linear regression.

**When to use it.** Residual plots that show systematic curvature (U-shape, oscillation) suggest the relationship is nonlinear and a polynomial term might help.

**Caveats:**

- High-degree polynomials (d > 3) risk severe overfitting and **Runge's phenomenon** — wild oscillations between data points.
- **Extrapolation is dangerous** — predictions outside the data range can swing extremely.
- Coefficient interpretation gets harder as degree grows.

**Practical guidance:** validate with cross-validation, regularize (Ridge/Lasso) when using polynomial features, and keep the degree low. For genuinely complex nonlinear relationships, modern practitioners reach for tree-based models or neural networks rather than high-degree polynomials — they generalize better and don't require manual feature engineering.

Reserve polynomial regression for exploratory analysis or when domain knowledge suggests a true polynomial relationship.

---

### Q11: Explain residual analysis and what patterns indicate model violations.

**A:** **Residuals** are the prediction errors, εᵢ = yᵢ − ŷᵢ. Patterns in the residuals reveal whether regression assumptions hold.

**Three diagnostic plots:**

- **Residuals vs fitted values** — should be a random cloud. Curved patterns indicate nonlinearity.
- **Q-Q plot** (residuals vs theoretical normal quantiles) — should fall on a straight line. Deviations indicate non-normality.
- **Scale-location plot** (√|residuals| vs fitted values) — should be flat random scatter. A trend indicates heteroscedasticity.

**Common patterns and what they mean:**

- **Funnel shape** in residuals vs fitted → heteroscedasticity. Try a variance-stabilizing transformation (log, sqrt) or weighted least squares.
- **Curvature / U-shape** → missing nonlinearity. Add polynomial or interaction terms, or move to a non-linear model.
- **Outliers far from the Q-Q line** → potentially influential observations to investigate (Cook's distance helps identify them).
- **Time-correlated residuals** → independence violated. Check with the Durbin-Watson statistic; consider time-series models.

**Fixes summarized:**

- Transformations (log, sqrt, Box-Cox) for heteroscedasticity or skew.
- Additional features (polynomials, interactions) for missing structure.
- Robust regression for heavy-tailed errors.
- A different model class when the linearity assumption fundamentally fails.

Residual analysis is underrated in practice but essential for confirming a linear model is appropriate.

---

### Q12: What is gradient descent and how is it applied to linear regression? Explain batch, stochastic, and mini-batch variants.

**A:** **Gradient descent** iteratively updates the coefficients in the direction of steepest decrease of the loss:

```
β ← β  −  η · ∇L(β)
```

where η is the learning rate. For linear regression with squared-error loss, the gradient is:

```
∇L = (−2/n) · Xᵀ · (y − X·β)
```

**Three variants** by batch size:

- **Batch gradient descent** — use all n samples per update. Stable, exact gradient, but slow on large data and one epoch = one update.
- **Stochastic gradient descent (SGD)** — one sample per update. Fast and very noisy; noise can help escape local minima and is essential for streaming data.
- **Mini-batch gradient descent** — small batches (typically 32–256 samples). Standard in practice — balances stability and speed, vectorizes well on GPUs.

**Learning rate matters a lot:**

- Too large → divergence or oscillation.
- Too small → painfully slow convergence.
- Adaptive methods like Adam and RMSprop adjust the effective rate per parameter and reduce tuning burden.

**Convergence.** Linear regression's loss is convex with a single global minimum, so all variants converge given a reasonable learning rate.

**When to use which:**

- Mini-batch — the practical default.
- Pure SGD — online or streaming learning.
- Batch — small datasets or offline scenarios where compute isn't a constraint.

Modern libraries (scikit-learn, PyTorch) handle the implementation details automatically.

---

### Q13: What are generalized linear models (GLMs) and how do they extend linear regression?

**A:** **GLMs** extend linear regression by introducing a **link function** g that connects the expected response to the linear predictor:

```
g( E[y | X] )  =  X · β
```

Different choices of distribution and link function give different familiar models:

- **Normal + identity link** — ordinary linear regression. E[y | X] = X·β.
- **Binomial + logit link** — logistic regression. logit(p) = X·β.
- **Poisson + log link** — count regression. log(E[y | X]) = X·β, so E[y | X] = exp(X·β).
- **Gamma + log link** — positive continuous data (insurance claims, durations).

**Common properties:**

- The response is from an **exponential family** distribution (Normal, Binomial, Poisson, Gamma, etc.).
- The link function is chosen for interpretability or mathematical convenience.
- Fitting is via maximum likelihood (often iteratively reweighted least squares).
- Goodness of fit is measured by **deviance** rather than R².

**Why GLMs matter.** They unify a wide range of models under one framework, making it easier to reason about assumptions and choose the right model for the response variable.

**Modern relatives.** **GAMs (Generalized Additive Models)** relax the linearity assumption by replacing each X·β term with a smooth function f(X), keeping much of the interpretability while handling nonlinearity gracefully.

In interviews, knowing GLMs shows conceptual depth — even if the specific instantiations (logistic, Poisson) come up more often.

---

### Q14: Explain how to handle categorical features in linear regression and logistic regression.

**A:** Categorical features must be numerically encoded before fitting a linear model.

**One-hot encoding.** Create a binary column for each category. For a feature with categories {A, B, C}, create columns [is_A, is_B, is_C].

**Important: drop one column to avoid multicollinearity.** If you keep all three, they sum to 1 for every row, making XᵀX singular. The dropped category becomes the **baseline** — remaining coefficients are interpreted relative to it.

In logistic regression with one-hot encoding, each remaining coefficient is a log-odds ratio relative to the baseline category.

**Ordinal categories** (education level, ratings) can sometimes be encoded as integers — but only if the spacing between levels is meaningful and effects are roughly linear in that ordering.

**Plain label encoding** (mapping categories to 0, 1, 2, ...) implicitly treats categories as ordinal even when they aren't — usually a bad idea for linear models with non-ordinal categories.

**High-cardinality features** (zipcode with thousands of levels) need extra care:

- Group rare categories into an "Other" bucket.
- Use **target encoding** — replace each category with its mean target value (with care to avoid leakage).
- Use regularization to handle the high-dimensional encoding.

**Always scale continuous features before regularized linear models** so the L1 / L2 penalty applies fairly across features regardless of their original units.

---

### Q15: When would you choose linear regression or logistic regression over more complex models, and what are the tradeoffs?

**A:** Linear models still earn their keep in many situations.

**Strengths:**

- **Interpretability** — coefficients directly indicate feature importance and direction of effect. Crucial for regulatory and explainability-sensitive domains (loan decisions, clinical risk scores).
- **Fast to train and score** — closed-form or quickly converging.
- **Minimal tuning** — typically just a regularization strength.
- **Sample-efficient** — perform well with small datasets where complex models would overfit.

**Tradeoffs:**

- Assume linear relationships, which often don't hold.
- Don't naturally capture nonlinear patterns or feature interactions (unless manually engineered).
- Lack the rich, embedded feature selection of tree-based models (Lasso is helpful but not equivalent).

**When to reach for which family:**

- **Linear / logistic** — high-dimensional sparse data (text, genomics), small samples, or regulated/explainability-critical settings.
- **Tree-based (gradient boosting, random forest)** — tabular data with complex interactions.
- **Neural networks** — unstructured data (images, audio, text embeddings).

**Strong interview answer:** "I'd start with a linear model to establish a baseline, understand the data, and confirm what's already explainable by linear effects. If a more complex model meaningfully outperforms it, that gap tells me how much nonlinearity or interaction the problem has."

---

## Interview Cheatsheet

**Key Terms:**

- **Linear Regression:** y = Xβ + ε; minimizes squared error via normal equation or gradient descent.
- **Logistic Regression:** Models P(y=1|X) = sigmoid(Xβ); uses MLE, outputs probabilities in [0,1].
- **Multicollinearity:** High correlation among features; detected via correlation matrix or VIF; addressed by Lasso/Ridge.
- **Ridge (L2):** λ∑β²; shrinks coefficients proportionally; handles multicollinearity.
- **Lasso (L1):** λ∑|β|; shrinks coefficients to exactly zero; performs feature selection.
- **Elastic Net:** Combines L1 + L2; balances Ridge stability with Lasso sparsity.
- **R²:** Proportion of variance explained; always increases with features.
- **Adjusted R²:** Penalizes model complexity; better for model comparison.
- **Sigmoid:** σ(z) = 1/(1 + e^-z); maps (-∞, +∞) to (0,1); core of logistic regression.
- **Odds Ratio:** exp(β); multiplicative change in odds per unit increase in feature.
- **MLE:** Maximizes likelihood; preferred for categorical outputs over least squares.
- **Softmax:** Generalizes sigmoid to K classes; models P(y=k|X) ∝ e^(z_k).
- **Residuals:** y - ŷ; should be random, normal, homoscedastic; violations indicate model issues.
- **Gradient Descent:** β ← β - η∇L; batch (stable, slow), SGD (fast, noisy), mini-batch (practical).
- **Normal Equation:** β = (X^T X)^-1 X^T y; closed-form solution; O(n³) but exact.
- **One-hot Encoding:** Convert categorical to binary columns; drop one to avoid multicollinearity.

**Rapid-Fire Q&A:**

- **Q:** What's the main assumption violated if residuals show a funnel pattern? **A:** Heteroscedasticity—error variance increases with fitted values.
- **Q:** How do you choose between Ridge and Lasso? **A:** Ridge if all features matter; Lasso if many are irrelevant; Elastic Net for both.
- **Q:** Why is MLE better than least squares for logistic regression? **A:** Binary outcomes aren't normally distributed; MLE produces calibrated probabilities; LSE can predict outside [0,1].
- **Q:** What does exp(β) = 1.5 mean in logistic regression? **A:** A unit increase in that feature multiplies odds by 1.5 (50% increase).
- **Q:** How do you detect multicollinearity? **A:** Correlation matrix (|r| > 0.8), VIF > 5, or eigenvalue analysis of X^T X.
- **Q:** What's the difference between gradient descent and normal equation? **A:** Gradient descent is iterative (scales to big data); normal equation is one-shot (O(n³), for small data).
- **Q:** Should you regularize before or after scaling features? **A:** Regularize **after** scaling; otherwise features with large magnitude are penalized more.
- **Q:** When would you use polynomial regression? **A:** When residual plots show systematic nonlinear patterns; avoid high degrees due to overfitting.
- **Q:** How does softmax differ from one-vs-rest logistic regression? **A:** Softmax models joint distribution over K classes; one-vs-rest trains K binary classifiers independently.
- **Q:** What does adjusted R² = 0.92 vs. R² = 0.93 suggest? **A:** The extra feature(s) slightly hurt adjusted performance; likely overfitting to noise.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

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

**A:** Linear regression models the relationship between input features and a continuous output as y = Xβ + ε, where β represents coefficients and ε is normally distributed noise. The key assumptions are: (1) linearity—the relationship is truly linear, (2) independence—observations are independent, (3) homoscedasticity—error variance is constant, (4) normality—residuals follow a normal distribution, and (5) no multicollinearity—features are not highly correlated. Violations of these assumptions (e.g., nonlinear relationships detected via residual plots) require transformations, polynomial features, or entirely different models. Testing assumptions through residual analysis (checking for heteroscedasticity patterns or non-normality) is a critical part of model validation.

---

### Q2: What is the closed-form solution (normal equation) for linear regression, and when might you use it versus gradient descent?

**A:** The normal equation directly computes optimal coefficients as β = (X^T X)^(-1) X^T y without iteration. This is computationally efficient for small to medium datasets (n < 100,000 features), providing an exact solution in one step. However, the approach requires computing the matrix inverse, which is O(n³) and numerically unstable if X^T X is ill-conditioned (near-singular). For large datasets or streaming data, gradient descent is preferable because it's iterative and parallelizable. In practice, you'd use the normal equation for interpretability and exact solutions when feasible, but default to gradient descent or stochastic variants (SGD) for scalability. Some libraries use QR decomposition instead of explicit inversion to improve numerical stability.

---

### Q3: Explain multicollinearity: what causes it, why it's problematic, and how to detect and address it.

**A:** Multicollinearity occurs when input features are highly correlated, violating the linear regression assumption of independence. This inflates coefficient variance, making estimates unstable and unreliable—small data changes cause large coefficient shifts. It's problematic for interpretation because you cannot isolate individual feature effects. Detection methods include: computing the correlation matrix (look for |r| > 0.8), calculating Variance Inflation Factor (VIF, values > 5-10 suggest problems), or examining eigenvalues of X^T X (small eigenvalues indicate multicollinearity). To address it, you can: drop redundant features, combine correlated features via PCA or averaging, use regularization (Ridge/Lasso), or collect more diverse data. Regularization is often preferred because it doesn't discard information entirely—it shrinks coefficients proportionally.

---

### Q4: Describe Ridge, Lasso, and Elastic Net regularization. What are their differences and use cases?

**A:** Ridge (L2) regularization adds a penalty λ∑β² to the loss function, shrinking all coefficients proportionally but never to zero—this helps when all features are relevant. Lasso (L1) adds λ∑|β|, which can shrink coefficients exactly to zero, performing feature selection automatically—ideal when you suspect many features are irrelevant. Elastic Net combines both (λ₁∑β² + λ₂∑|β|), balancing Ridge's stability with Lasso's feature selection. Ridge is best when multicollinearity is severe and features are truly correlated; Lasso when you want interpretability and sparse models; Elastic Net when you want both properties. The regularization parameter λ is tuned via cross-validation—larger λ increases bias but reduces variance, following the bias-variance tradeoff. All three shrink coefficients toward zero, preventing overfitting by controlling model complexity.

---

### Q5: What is the difference between R-squared and adjusted R-squared?

**A:** R² measures the proportion of variance explained by the model: R² = 1 - (SS_res / SS_tot), ranging from 0 to 1, where higher is better. However, R² always increases (or stays same) when adding features, regardless of whether they improve generalization. Adjusted R² = 1 - ((1 - R²)(n - 1)/(n - p - 1)) penalizes model complexity by accounting for the number of features p and sample size n. It decreases if added features don't sufficiently improve fit, making it a better metric for model selection. For example, adding a random feature will increase R² but decrease adjusted R². In interviews, mention that adjusted R² is more trustworthy for comparing models with different numbers of features, and external validation (test set performance) is always preferable. Both metrics are less critical in production if you're using proper cross-validation.

---

### Q6: Explain how logistic regression works and the role of the sigmoid function.

**A:** Logistic regression models the probability of a binary outcome using P(y=1|X) = 1 / (1 + e^(-z)), where z = Xβ. The sigmoid function maps any z ∈ (-∞, +∞) to probability ∈ (0, 1), ensuring outputs are valid probabilities. Unlike linear regression, logistic regression uses maximum likelihood estimation (MLE) for fitting, not least squares. The decision boundary is where P(y=1) = 0.5, or equivalently z = 0. Logistic regression extends to multiclass via softmax regression (a generalization of sigmoid for K classes), where P(y=k|X) = e^(z_k) / ∑_j e^(z_j). This model is highly interpretable: exp(β_i) represents the odds ratio for feature i, showing how a unit increase changes odds of class 1. It's widely used because it's simple, fast, interpretable, and provides calibrated probabilities.

---

### Q7: What is the odds ratio in logistic regression, and how do you interpret coefficients?

**A:** In logistic regression, the odds of class 1 are P(y=1) / P(y=0) = e^(z) = e^(Xβ). For a single feature increase by 1 unit, the odds multiply by e^(β_i). If β_i = 0.2, then e^0.2 ≈ 1.22, meaning odds increase by 22%. This makes coefficients directly interpretable: positive β_i increases odds of class 1, negative decreases them. For a feature with categorical values, encode as dummy variables and interpret relative to the baseline category. Confidence intervals on coefficients come from the Hessian (inverse second derivative) during MLE, and statistical significance tests use Wald statistics. For example, if β_i = 0.5 with SE = 0.1, the 95% CI is [0.304, 0.696], giving [e^0.304, e^0.696] ≈ [1.36, 2.00] as the odds ratio CI. This interpretability is why logistic regression remains popular for explainability-critical applications.

---

### Q8: Explain maximum likelihood estimation (MLE) for logistic regression. Why not use least squares?

**A:** MLE maximizes the likelihood L(β) = ∏ P(y_i|X_i)^{y_i} (1 - P(y_i|X_i))^{1-y_i}, equivalent to minimizing cross-entropy loss: -∑ [y_i log(P_i) + (1-y_i) log(1-P_i)]. This is more appropriate than least squares (which assumes normally distributed errors) because y ∈ {0,1} is categorical—predictions outside [0,1] are meaningless. Least squares on binary targets would produce suboptimal fits and underestimate uncertainty. MLE is solved iteratively via Newton-Raphson (second-order) or gradient descent (first-order), using the Hessian to obtain uncertainty estimates. The resulting probability estimates are calibrated by design—they reflect true probabilities under the model. MLE also provides the principled framework for regularized logistic regression (L2, L1 penalties), which reduces to a penalized MLE problem. Many ML practitioners overlook this, but understanding MLE versus least squares demonstrates theoretical maturity.

---

### Q9: What is softmax regression and how does it generalize logistic regression to multiclass problems?

**A:** Softmax regression (multinomial logistic regression) extends binary logistic regression to K ≥ 2 classes by modeling P(y=k|X) = e^(z_k) / ∑_{j=1}^K e^(z_j), where z_k = X β_k. Each class has its own coefficient vector β_k, making this K times more parameterized than binary logistic regression. When K = 2, softmax reduces to logistic regression (redundant class parameters are dropped). The decision rule is argmax_k P(y=k|X). Softmax is trained via MLE, minimizing categorical cross-entropy: -∑ y_i log(P_{i,y_i}), where y_i is one-hot encoded. Unlike one-vs-rest approaches, softmax models joint probability over all classes, making it theoretically cleaner. It's computationally efficient and highly interpretable. However, softmax assumes mutual exclusivity (one true class per sample); for multilabel problems, use independent sigmoid on each class.

---

### Q10: Describe polynomial regression and when you'd use it instead of linear regression.

**A:** Polynomial regression adds polynomial features (X², X³, etc.) to linear regression, allowing it to fit nonlinear relationships while remaining "linear in parameters." For degree d, the model is y = β₀ + β₁X + β₂X² + ... + β_d X^d + ε. You detect the need for polynomial regression via residual plots—if residuals show systematic patterns (U-shape, oscillation), the relationship is likely nonlinear. Polynomial regression is useful when data truly exhibits polynomial structure, but be cautious: (1) high-degree polynomials (d > 3) risk severe overfitting due to Runge's phenomenon, (2) extrapolation beyond data range produces wild predictions, (3) interpretation becomes harder. Always validate via cross-validation and use regularization. For highly complex nonlinear relationships, modern ML practitioners prefer tree-based models or neural networks instead of high-degree polynomials, since they generalize better and don't require manual feature engineering. Reserve polynomial regression for exploratory analysis or when domain knowledge suggests polynomial structure.

---

### Q11: Explain residual analysis and what patterns indicate model violations.

**A:** Residuals ε_i = y_i - ŷ_i represent prediction errors and reveal whether regression assumptions hold. Create plots: (1) residuals vs. fitted values should show random scatter (no patterns)—curved patterns indicate nonlinearity, (2) Q-Q plot (residuals vs. normal quantiles) should follow a straight line—deviations indicate non-normality, (3) scale-location plot (√|residuals| vs. fitted values) should be randomly scattered—trends indicate heteroscedasticity. Specific patterns: (1) heteroscedasticity (variance increases with fitted values, forming a "funnel") suggests weighting or transformation, (2) nonlinear trends indicate missing polynomial terms or wrong model class, (3) outliers far from the Q-Q line might be influential observations needing investigation, (4) autocorrelation (residuals correlated across time) violates independence—test via Durbin-Watson statistic. Addressing violations: apply transformations (log, sqrt), add features, use robust regression, or switch models. Residual analysis is underrated in practice but essential for model validation.

---

### Q12: What is gradient descent and how is it applied to linear regression? Explain batch, stochastic, and mini-batch variants.

**A:** Gradient descent iteratively updates coefficients β ← β - η ∇L(β), where ∇L is the gradient of loss and η is the learning rate. For linear regression, the gradient is ∇L = -2 X^T(y - Xβ) / n. Batch gradient descent uses all n samples to compute each gradient—stable but slow on large data. Stochastic gradient descent (SGD) uses one sample per update—fast and noisy, but noise helps escape local minima and scales to huge datasets. Mini-batch gradient descent (typical in practice) uses small batches (32-256 samples)—balances stability and speed. Learning rate choice is critical: too large causes divergence, too small is slow; adaptive methods (Adam, RMSprop) adjust per parameter. Convergence depends on loss surface shape—linear regression has a single global minimum, so all variants eventually converge. In interviews, discuss that SGD is standard for online learning, batch for small data or offline scenarios, and mini-batch for deep learning. Modern libraries (scikit-learn, PyTorch) handle this automatically.

---

### Q13: What are generalized linear models (GLMs) and how do they extend linear regression?

**A:** GLMs extend linear regression by modeling E[y|X] via a link function g: g(E[y|X]) = Xβ. Linear regression assumes y ~ Normal with identity link (g = identity), but GLMs allow different distributions and links. For example: (1) Binomial with logit link gives logistic regression, (2) Poisson with log link models count data with E[y] = e^(Xβ), (3) Gamma with log link for positive continuous data. The general form uses maximum likelihood estimation on the chosen distribution. GLMs unify many models under one framework, sharing common properties: exponential family distributions, link functions chosen for interpretation or mathematical convenience, and inference via deviance (generalized R²). In interviews, GLMs are less common than specific instantiations (logistic regression, Poisson regression), but demonstrating knowledge of the framework shows conceptual depth. Modern alternatives like generalized additive models (GAMs) relax the linearity assumption while keeping interpretability.

---

### Q14: Explain how to handle categorical features in linear regression and logistic regression.

**A:** Categorical features must be encoded before fitting linear models. One-hot encoding creates binary columns for each category, allowing interpretation: if a feature has 3 categories {A, B, C}, create columns [is_A, is_B, is_C]. Drop one column (usually baseline) to avoid multicollinearity—with all 3, the sum is always 1, causing singular X^T X. In logistic regression with one-hot encoding, the dropped category (baseline) becomes the reference; coefficients represent log-odds relative to baseline. Ordinal categories (e.g., education level: high school < bachelor < masters) can be encoded as integers, assuming linear effects. Label encoding (mapping to 0, 1, 2...) treats categories as ordinal even if they're not—use cautiously. For many categories (e.g., zipcode with 1000+ levels), one-hot creates high-dimensional sparse data; consider grouping rare categories or using target encoding (encode each category as its mean target value). Regularization helps when high-dimensional encoding is unavoidable. Always scale continuous features before regularized linear models to ensure fair penalty strength across features.

---

### Q15: When would you choose linear regression or logistic regression over more complex models, and what are the tradeoffs?

**A:** Linear models excel in interpretability—coefficients directly show feature importance and direction of effects, critical for regulatory compliance and explainability (e.g., loan decisions). They're fast to train and score, need little tuning, and perform well with small sample sizes. Tradeoffs: they assume linear relationships, which often don't hold; they struggle with nonlinear patterns or feature interactions (unless manually engineered); they lack embedded feature selection (Lasso helps but isn't as powerful as tree-based). Start with linear models as a baseline—if performance is acceptable, use them; if gap to complex models is large, the nonlinearity is significant. In practice: linear for high-dimensional sparse data (text, genomics), small samples, or regulatory settings; tree-based for tabular data with complex interactions; neural nets for unstructured data (images, text embeddings). A strong interviewer approach is: "I'd fit a linear model first to establish a baseline and ensure I understand the problem, then escalate complexity if needed."

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

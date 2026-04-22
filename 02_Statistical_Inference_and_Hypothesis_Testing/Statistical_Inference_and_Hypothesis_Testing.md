# Statistical Inference and Hypothesis Testing

📺 **Video Lecture:** https://youtu.be/4NlWsOKGBLc


## Interview Anchor
- **Point Estimation:** Using sample data to estimate an unknown population parameter with a single value
- **Hypothesis Testing:** A statistical procedure to decide between competing hypotheses about a parameter using evidence from data
- **Type I and Type II Errors:** False positives (rejecting true null) and false negatives (failing to reject false null), respectively

## Key Concepts Overview
Statistical inference is the bridge between observed data and unobserved populations—it's how you make conclusions about reality given imperfect information. In machine learning interviews, testing knowledge here reveals whether candidates understand the limits of their models, how to validate assumptions, and how to draw reliable conclusions from experiments. This is especially critical for A/B testing, model evaluation, and determining statistical significance of improvements.

Understanding the relationship between p-values, confidence intervals, and effect sizes helps you interpret results correctly and avoid common pitfalls like p-hacking or misinterpreting confidence intervals as probability statements about parameters. Modern ML applications increasingly require this knowledge: designing fair experiments requires understanding power, sample size calculations drive data collection requirements, and multiple testing corrections prevent false discoveries in high-dimensional settings.

---

### Q1: Explain point estimation and distinguish between biased and unbiased estimators.

**A:** Point estimation uses sample data to produce a single estimate θ̂ of an unknown parameter θ. An estimator is unbiased if E[θ̂] = θ, meaning on average across repeated samples, it estimates the true value correctly; biased if E[θ̂] ≠ θ. For example, the sample mean is an unbiased estimator of the population mean, but the sample variance (using divisor n) is biased, while dividing by (n-1) makes it unbiased. Bias = E[θ̂] - θ, and even biased estimators can be useful if they have lower variance (bias-variance tradeoff). In ML, regularized regression estimators (like ridge regression) are biased but have lower variance than OLS, often giving better predictions. When evaluating an estimator, you care about both bias and variance—mean squared error MSE(θ̂) = Bias² + Var(θ̂), so sometimes accepting small bias trades off for substantial variance reduction.

---

### Q2: What are confidence intervals and how do they relate to hypothesis testing?

**A:** A confidence interval (CI) is a range [L, U] constructed from sample data such that if you repeated the sampling procedure many times, approximately (1-α)×100% of the intervals would contain the true parameter (e.g., 95% CI means 95% coverage in repeated samples). This is NOT a probability statement about the parameter itself—the parameter is fixed, either in [L, U] or it isn't. A 95% CI for a mean obtained via sample mean ± 1.96×SE is closely related to hypothesis testing: if a null value (like zero) falls outside the 95% CI, you'd reject the null at α=0.05 significance level. Conversely, the set of values not rejected by a test forms a CI. In ML reporting, confidence intervals are more informative than point estimates alone because they quantify uncertainty. For instance, reporting "model accuracy = 0.85 ± 0.03" immediately tells stakeholders the range of plausible values, whereas "accuracy = 0.85" alone is ambiguous.

---

### Q3: Define p-values and explain common misconceptions about their interpretation.

**A:** The p-value is the probability of observing a test statistic as extreme as or more extreme than what you observed, assuming the null hypothesis is true: P(test stat | H₀). A small p-value (typically <0.05) means the observed data is unlikely under H₀, suggesting evidence against H₀. Critical misconceptions: (1) p-value is NOT the probability H₀ is true—H₀ is either true or false, not probabilistic; (2) p-value is NOT the probability the result is due to chance—all results involve randomness; (3) small p-value doesn't prove large effect size; (4) p-value depends on sample size—with huge n, tiny effects become "significant." The correct interpretation: "If H₀ were true, we'd see data this extreme or more extreme 5% of the time." In practice, overreliance on p-values without considering effect sizes or practical significance leads to spurious discoveries, especially in high-dimensional settings where multiple comparisons inflate false discovery rates.

---

### Q4: Explain Type I error, Type II error, and statistical power.

**A:** Type I error (false positive) is rejecting H₀ when it's actually true; its probability is the significance level α. Type II error (false negative) is failing to reject H₀ when it's actually false; its probability is β. Statistical power = 1 - β, the probability of correctly detecting a true effect when one exists. These three quantities (α, β, effect size, sample size) are deeply linked: fix any three and you can solve for the fourth. For example, to achieve 80% power detecting a medium effect at α=0.05 requires roughly n=64 per group in a two-sample t-test. In ML, Type I errors (false positives, like flagging benign users as fraudsters) and Type II errors (false negatives, like missing actual fraud) have different costs depending on context. Higher power requires larger sample sizes or stronger true effects. When designing A/B tests, you specify desired power (often 80%) and acceptable α (often 0.05), then calculate required sample size to achieve meaningful detection.

---

### Q5: Compare parametric and non-parametric hypothesis tests and when to use each.

**A:** Parametric tests (t-test, ANOVA, linear regression) assume a specific distribution (usually normal) and estimate parameters of that distribution. Non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Spearman correlation) don't assume a particular distribution and work on ranks instead of raw values. Parametric tests are more powerful (better at detecting true effects) when their assumptions hold, but non-parametric tests are robust when assumptions are violated. Use parametric tests when data is approximately normal or sample size is large (CLT makes them robust); use non-parametric when data is clearly non-normal, heavily skewed, contains outliers, or sample size is very small. In practice, you often try both—if conclusions agree, you're confident; if they differ, non-parametric results are more trustworthy. For ML practitioners, understanding that rank-based methods are robust helps you design stable evaluation pipelines: using median (non-parametric) instead of mean (parametric) for aggregating across runs is more robust to outlier runs.

---

### Q6: Explain the t-test: assumptions, variants, and when to apply each.

**A:** The t-test compares means and assumes normality and equal variances (classical form). Variants include: one-sample t-test (does sample mean differ from a value?), two-sample independent t-test (do two group means differ?), and paired t-test (do paired observations differ?). Welch's t-test relaxes the equal variance assumption. All assume independence and approximate normality (robust with larger samples). The test statistic t = (M₁ - M₂) / SE_{diff}, where SE_{diff} is the standard error of the difference; under H₀, t follows a t-distribution with degrees of freedom depending on sample sizes. Degrees of freedom slightly above normal: t-distributions have heavier tails than normal, making tests slightly more conservative. In ML, paired t-tests compare two models on the same test instances (e.g., baseline vs. new model on same 5 datasets), which is more powerful than independent tests. Always check assumptions via Shapiro-Wilk test (normality) and Levene's test (equal variances) before reporting t-tests.

---

### Q7: What is ANOVA and how does it extend the t-test?

**A:** ANOVA (Analysis of Variance) tests whether means differ across k≥3 groups by partitioning total variance into between-group variance (explained by group membership) and within-group variance (residual). The F-statistic = MS_{between} / MS_{within}, where MS are mean squares; large F suggests groups differ. One-way ANOVA tests one categorical factor; two-way ANOVA tests two factors and their interaction. ANOVA assumes normality, equal variances across groups (Levene's test), and independence. If ANOVA rejects H₀ (not all means equal), post-hoc tests (Tukey HSD, Bonferroni) identify which pairs differ. Kruskal-Wallis is the non-parametric alternative. In ML, ANOVA compares multiple models or hyperparameter settings: fit models across k settings, compute ANOVA on held-out test results to see if setting significantly affects performance. The extension from t-test (2 groups) to ANOVA (k groups) avoids repeatedly running pairwise tests, which would inflate Type I error.

---

### Q8: Explain the chi-square test and its applications.

**A:** The chi-square test compares observed frequencies in categorical data to expected frequencies under a null hypothesis. The test statistic χ² = Σ (O_i - E_i)² / E_i, where O_i is observed count and E_i is expected count under H₀; under H₀, χ² approximately follows a chi-square distribution with k degrees of freedom (k = number of categories - 1, minus parameters estimated). Applications: (1) goodness-of-fit: does data follow a specific distribution? (2) independence: are two categorical variables independent? (3) homogeneity: do k populations have the same distribution? Assumptions: observations are independent, categories are mutually exclusive, expected frequency ≥5 in each cell (merge small categories if violated). In ML, chi-square tests evaluate whether predicted class distributions match observed (e.g., classifier outputting wrong class proportions), or whether feature distribution differs significantly between training and deployed data (indicator of data drift).

---

### Q9: What is maximum likelihood estimation and why is it powerful?

**A:** Maximum likelihood estimation (MLE) finds the parameter value θ that maximizes the likelihood function L(θ; x) = P(x | θ) (or log-likelihood ℓ(θ) = log L(θ)). Intuitively, MLE finds the parameter value making the observed data most probable. For a sample of independent observations, L(θ; x₁, ..., xₙ) = ∏ᵢ P(xᵢ | θ), and we maximize by taking derivatives: dℓ/dθ = 0. MLEs have desirable asymptotic properties (consistency, asymptotic normality, efficiency), so with large samples, MLE is approximately optimal. In practice, you maximize log-likelihood (numerically stable, easier to differentiate) using gradient descent or closed-form solutions. In ML, logistic regression finds MLE of class probabilities, Gaussian mixture models use EM algorithm to compute MLE's for mixture parameters, and neural networks trained with cross-entropy loss are implicitly finding MLE. Understanding MLE helps you see why particular loss functions (like cross-entropy) are natural choices for different problems.

---

### Q10: Explain the method of moments and compare it to MLE.

**A:** Method of moments equates sample moments (like sample mean, sample variance) to theoretical moments under the distributional model, then solves for parameters. For example, for a normal distribution, E[X] = μ and Var(X) = σ², so the method of moments estimators are μ̂ = sample mean and σ̂² = sample variance. Method of moments is computationally simpler than MLE (just solve equations rather than optimize) and often provides good initial guesses for numerical optimization. However, method of moments estimators are generally less efficient than MLEs—they converge more slowly to the true value and have higher variance. Both are consistent and asymptotically normal. Method of moments is useful when likelihood is difficult to specify (e.g., mixture models) or when quick estimates suffice, but for formal inference with small samples, MLE is preferred. Interviewers appreciate understanding the tradeoff: computational simplicity vs. statistical efficiency.

---

### Q11: What is Fisher information and what does it tell you about an estimator?

**A:** Fisher information I(θ) = -E[d²ℓ/dθ² | θ] quantifies how much information the data carries about parameter θ. Large Fisher information means the likelihood is sharply peaked at the true θ (data constrains θ well), while small information means likelihood is flat (data tells you little about θ). The Cramér-Rao lower bound states that for unbiased estimator θ̂ of θ, Var(θ̂) ≥ 1/I(θ), meaning the inverse Fisher information is the minimum possible variance. MLEs achieve this lower bound asymptotically (efficient estimators). In practice, Fisher information drives standard errors of parameter estimates: SE(θ̂) ≈ 1/√I(θ) (in large samples), so you can evaluate how precisely you can estimate a parameter before collecting data. In ML, Fisher information matrix appears in optimization (preconditioners in second-order methods) and in uncertainty quantification for neural network weights. Understanding Fisher information helps you reason about what sample sizes suffice for achieving desired estimation precision.

---

### Q12: Define sufficient statistics and explain their role in inference.

**A:** A sufficient statistic T(X) captures all information in the data relevant to parameter θ—the distribution of X given T(X) doesn't depend on θ. For normal distribution with unknown mean and variance, the pair (sample mean, sample variance) is sufficient; for Poisson with unknown rate λ, the sample sum Σxᵢ is sufficient. Factorization criterion: T is sufficient if likelihood factors as L(θ; x) = g(T(x), θ) h(x), where h doesn't depend on θ. Practical importance: once you compute T(x), the original data x is irrelevant for likelihood-based inference—you can discard x and work with T(x). This enables data compression: for Poisson data, you only need to track the sum and sample size, not all individual observations. In Bayesian inference, sufficient statistics determine the posterior, so identifying them helps you understand which aspects of data matter. For example, in linear regression, the sufficient statistic involves X^T X and X^T Y, explaining why these appear in normal equations.

---

### Q13: What is the Neyman-Pearson Lemma and why is it important?

**A:** The Neyman-Pearson Lemma states that for testing H₀: θ = θ₀ vs. H₁: θ = θ₁, the most powerful test (highest power for given α) rejects H₀ when likelihood ratio L(θ₁; x) / L(θ₀; x) exceeds a threshold. This test is uniformly most powerful (UMP) among all tests with that significance level. The lemma provides a principled way to construct optimal tests: find the likelihood ratio and threshold it. Many classical tests can be derived this way—t-test, F-test, chi-square test all emerge as likelihood ratio tests with simple structures. In practice, likelihood ratio tests generalize beyond simple vs. simple hypotheses (where we compare nested models). The lemma justifies why statistical tests often take the form "reject if test statistic > threshold"—that threshold is derived to optimize power. Understanding this helps you appreciate that classical hypothesis tests aren't arbitrary but derived from principled optimality criteria.

---

### Q14: Explain multiple testing correction and its relevance to ML model selection.

**A:** When you perform many hypothesis tests, the family-wise error rate (FWER, probability of ≥1 false positive) exceeds the per-test significance level. Bonferroni correction sets individual test significance to α/m (for m tests), guaranteeing FWER ≤ α; this is conservative but simple. False discovery rate (FDR) controls the expected fraction of false discoveries among rejected hypotheses; Benjamini-Hochberg procedure controls FDR and is less stringent than Bonferroni, often preferred for exploratory analysis. In ML hyperparameter tuning, trying many settings inflates false discovery: a setting appears "best" just by chance if you test enough settings. Cross-validation partially addresses this by holding out test data. When doing feature selection (testing which of 1000 features matter), without correction, ~50 will appear significant at p<0.05 by chance alone. Interviewers expect you to know that reporting results from exploratory analysis without correction overstates evidence, and that validation on held-out data is essential to verify true improvement vs. overfitting to the test set.

---

### Q15: How do you design A/B tests and determine sample size requirements?

**A:** A/B testing compares two variants (control A and treatment B) on randomly assigned users, measuring a primary metric (e.g., conversion rate). Design steps: (1) specify primary metric and hypothesis (one-sided or two-sided), (2) specify minimum detectable effect size (e.g., 10% relative lift), (3) choose significance level α (often 0.05) and power (often 0.80), (4) calculate sample size n per variant, (5) run test with proper randomization and blinding, (6) analyze via t-test or proportion test at specified α level. Sample size formula for comparing proportions: n ≈ 2(z_α + z_β)²p(1-p) / Δ², where p is baseline rate and Δ is effect size. Practical considerations: avoid peeking at results during test (inflates α), use sequential testing for early stopping, account for multiple comparisons if testing multiple metrics, and pre-register analysis plan to prevent p-hacking. In ML contexts, A/B tests validate that model improvements generalize (online metrics matter more than offline metrics), and understanding this framework helps you propose tests that convince stakeholders of real value.

---

## Interview Cheatsheet

**Key Terms:**
- **Point Estimation:** Using sample data to estimate unknown population parameter with single value
- **Unbiased Estimator:** E[θ̂] = θ; on average, it estimates the true parameter correctly
- **Confidence Interval:** Range [L, U] such that (1-α)×100% of repeated samples' intervals contain true parameter
- **p-value:** P(test statistic | H₀); probability of observing data as extreme under null hypothesis
- **Type I Error:** α; probability of rejecting H₀ when it's true (false positive)
- **Type II Error:** β; probability of failing to reject H₀ when it's false (false negative)
- **Power:** 1 - β; probability of correctly detecting true effect
- **t-test:** Compares means assuming normal distribution and independence
- **ANOVA:** Compares means across k≥3 groups using variance partitioning
- **Chi-square Test:** Tests goodness-of-fit, independence, or homogeneity for categorical data
- **MLE:** Maximum likelihood estimation; finds parameter maximizing probability of observed data
- **Method of Moments:** Equates sample moments to theoretical moments to estimate parameters
- **Fisher Information:** Quantifies how much data constrains parameter; inverse is lower bound on variance
- **Sufficient Statistic:** T(X) captures all information in data relevant to parameter θ
- **Neyman-Pearson Lemma:** Most powerful test uses likelihood ratio threshold
- **Multiple Testing Correction:** Bonferroni (conservative) or FDR controls false discoveries in many tests

**Rapid-Fire Q&A:**
- **Q: What does a 95% confidence interval mean?** **A:** In repeated samples, ~95% of intervals contain the true parameter; NOT "95% chance parameter is in this interval"
- **Q: Why do we use log-likelihood instead of likelihood in optimization?** **A:** Numerical stability and easier differentiation; products become sums when logged
- **Q: How does sample size affect p-value?** **A:** Larger n makes p-values smaller for fixed effect size; same effect is "more significant" with more data
- **Q: What's the difference between FWER and FDR?** **A:** FWER controls probability of any false positive; FDR controls expected proportion of false positives
- **Q: How do you choose effect size for sample size calculation?** **A:** Use prior research, pilot data, or minimum practically meaningful difference (e.g., 10% improvement)

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

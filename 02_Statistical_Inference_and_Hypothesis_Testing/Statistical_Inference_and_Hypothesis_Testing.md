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

**A:** **Point estimation** uses sample data to produce a single estimate θ̂ of an unknown population parameter θ.

An estimator is **unbiased** if its expected value equals the true parameter:

```
E[θ̂] = θ            (unbiased)
E[θ̂] ≠ θ            (biased — bias = E[θ̂] − θ)
```

A biased estimator over- or under-estimates θ on average across repeated samples.

**Common examples:**

- The sample mean is an unbiased estimator of the population mean.
- The sample variance with divisor n is biased; using divisor (n − 1) makes it unbiased.

A biased estimator can still be useful if it has much lower variance — that's the bias–variance tradeoff. The total error of an estimator is captured by **mean squared error**:

```
MSE(θ̂) = Bias(θ̂)² + Var(θ̂)
```

In ML, regularized regression estimators like ridge regression are deliberately biased but have lower variance than OLS, often producing better predictions. When evaluating an estimator, both bias and variance matter — sometimes a little bias is well worth a lot of variance reduction.

---

### Q2: What are confidence intervals and how do they relate to hypothesis testing?

**A:** A **confidence interval (CI)** is a range [L, U] computed from sample data with the property that if you repeated the sampling procedure many times, about (1 − α) × 100% of the intervals would contain the true parameter. So a 95% CI corresponds to ~95% coverage in repeated samples.

**Critical interpretation:** this is *not* a probability statement about the parameter. The true parameter is fixed — it's either in [L, U] or it isn't. The "95%" describes the long-run behavior of the procedure, not a particular interval.

A standard form for a mean is:

```
CI = sample_mean ± 1.96 · SE        (95% CI under normal approximation)
```

**Tight link with hypothesis testing:** if a null value (like zero) falls *outside* the 95% CI, you'd reject the null at α = 0.05. Conversely, the set of values *not* rejected by a test (at level α) forms a (1 − α) CI.

In ML reporting, confidence intervals are far more informative than bare point estimates because they quantify uncertainty. "Model accuracy = 0.85 ± 0.03" immediately conveys the range of plausible values, whereas just "0.85" leaves stakeholders guessing.

---

### Q3: Define p-values and explain common misconceptions about their interpretation.

**A:** The **p-value** is the probability of observing a test statistic at least as extreme as the one you observed, assuming the null hypothesis is true:

```
p-value = P(test statistic at least as extreme as observed | H₀)
```

A small p-value (typically below 0.05) means the observed data would be unlikely under H₀, which is evidence against H₀.

**Critical misconceptions to avoid:**

- A p-value is **NOT** the probability that H₀ is true. H₀ is either true or false; it isn't probabilistic.
- A p-value is **NOT** the probability the result is due to chance. All results involve randomness.
- A small p-value does **NOT** imply a large effect size — it's a statement about evidence, not magnitude.
- A p-value depends on sample size. With huge n, even trivially small effects become "statistically significant."

**The correct interpretation:** "If H₀ were true, we would see data at least this extreme about p × 100% of the time."

In practice, over-relying on p-values without also reporting effect sizes or considering practical significance leads to spurious discoveries — especially in high-dimensional settings where multiple comparisons inflate false discovery rates.

---

### Q4: Explain Type I error, Type II error, and statistical power.

**A:** Two ways a hypothesis test can be wrong:

- **Type I error (false positive):** rejecting H₀ when it's actually true. Probability = α (the significance level).
- **Type II error (false negative):** failing to reject H₀ when it's actually false. Probability = β.

**Statistical power** is the complement of Type II error:

```
power = 1 − β
```

It's the probability of correctly detecting a true effect.

**Four-way relationship:** α, β, effect size, and sample size are tightly linked — fix any three and the fourth is determined. For example, achieving 80% power to detect a medium effect at α = 0.05 needs roughly n = 64 per group in a two-sample t-test.

**In ML:** Type I errors (flagging benign users as fraudsters) and Type II errors (missing actual fraud) have very different costs depending on context. Higher power demands larger samples or stronger true effects.

When designing an A/B test, you specify desired power (typically 0.80) and acceptable α (typically 0.05), then calculate the required sample size for the minimum detectable effect.

---

### Q5: Compare parametric and non-parametric hypothesis tests and when to use each.

**A:** Two big families of hypothesis tests:

- **Parametric tests** (t-test, ANOVA, linear regression): assume a specific distribution (usually normal) and estimate its parameters.
- **Non-parametric tests** (Mann-Whitney U, Kruskal-Wallis, Spearman correlation): make no distributional assumptions and operate on ranks rather than raw values.

**Tradeoffs:**

- Parametric tests are *more powerful* when their assumptions hold — they'll detect smaller true effects.
- Non-parametric tests are *more robust* when assumptions are violated.

**When to use each:**

- Parametric — data is approximately normal, or the sample size is large enough that the CLT kicks in.
- Non-parametric — data is clearly non-normal, heavily skewed, contains outliers, or the sample size is very small.

In practice, running both is a useful sanity check: if conclusions agree, you're confident; if they disagree, non-parametric results are usually more trustworthy.

For ML practitioners, understanding that rank-based methods are robust helps you design stable evaluation pipelines — for example, using the median (non-parametric) instead of the mean (parametric) when aggregating across runs is more robust to outlier runs.

---

### Q6: Explain the t-test: assumptions, variants, and when to apply each.

**A:** The **t-test** compares means under the assumption of normality and (in classical form) equal variances. The main variants:

- **One-sample t-test:** does a sample mean differ from a specified value?
- **Two-sample independent t-test:** do two group means differ?
- **Paired t-test:** do paired observations differ (e.g., before vs. after on the same units)?
- **Welch's t-test:** like the two-sample test but relaxes the equal-variance assumption.

All variants assume independence and approximate normality (which becomes mild for larger samples thanks to the CLT).

The test statistic for a two-sample test is:

```
t = (M₁ − M₂) / SE_diff
```

where SE_diff is the standard error of the difference between sample means. Under H₀, this follows a t-distribution whose degrees of freedom depend on the sample sizes.

**About the t-distribution:** with small degrees of freedom it has heavier tails than the normal, making tests slightly more conservative. As df grows, it converges to the normal.

**In ML:** paired t-tests compare two models on the same test instances (baseline vs. new model on the same 5 datasets, for instance) and are more powerful than independent tests because they remove between-instance variance.

Always check the assumptions before reporting — Shapiro-Wilk for normality and Levene's test for equal variances.

---

### Q7: What is ANOVA and how does it extend the t-test?

**A:** **ANOVA (Analysis of Variance)** tests whether the means differ across k ≥ 3 groups by partitioning total variability into two pieces:

- **Between-group variance** — variability explained by group membership.
- **Within-group variance** — residual variability inside each group.

The test statistic is the **F-statistic**, the ratio of the corresponding mean squares:

```
F = MS_between / MS_within
```

A large F suggests the group means differ.

**Variants:**

- **One-way ANOVA** — one categorical factor.
- **Two-way ANOVA** — two factors plus their interaction.

**Assumptions:** normality, equal variances across groups (test with Levene's), and independence. If ANOVA rejects the null (not all means are equal), follow up with **post-hoc tests** like Tukey HSD or Bonferroni-corrected pairwise tests to identify which specific pairs differ. The non-parametric alternative is the **Kruskal-Wallis** test.

**In ML:** ANOVA compares multiple models or hyperparameter settings — fit k settings, run ANOVA on the held-out test metric, and check whether the setting significantly affects performance. The extension from t-test (2 groups) to ANOVA (k groups) is important because running many pairwise t-tests would inflate Type I error.

---

### Q8: Explain the chi-square test and its applications.

**A:** The **chi-square test** compares observed frequencies in categorical data to the frequencies expected under a null hypothesis. The test statistic is:

```
χ² = Σᵢ (Oᵢ − Eᵢ)² / Eᵢ
```

where Oᵢ is the observed count and Eᵢ is the expected count under H₀. Under H₀, χ² approximately follows a chi-square distribution with k degrees of freedom (k = number of categories − 1, minus any estimated parameters).

**Three common applications:**

- **Goodness-of-fit:** does the data follow a specific distribution?
- **Independence:** are two categorical variables independent?
- **Homogeneity:** do k populations share the same distribution?

**Assumptions:** observations are independent, categories are mutually exclusive, and expected frequency ≥ 5 in each cell. If a cell is too small, merge categories.

**In ML:** chi-square tests check whether predicted class distributions match observed counts (a classifier that outputs the wrong class proportions), or whether a feature's distribution has shifted significantly between training and deployed data — a signal of data drift.

---

### Q9: What is maximum likelihood estimation and why is it powerful?

**A:** **Maximum likelihood estimation (MLE)** finds the parameter θ that makes the observed data most probable. The likelihood function is:

```
L(θ ; x) = P(x | θ)
```

In practice we maximize the **log-likelihood** because it's numerically stable and turns products into sums:

```
ℓ(θ) = log L(θ)
```

For a sample of independent observations, the likelihood factors:

```
L(θ ; x₁, ..., xₙ) = Πᵢ P(xᵢ | θ)
ℓ(θ)              = Σᵢ log P(xᵢ | θ)
```

Maximization typically proceeds by setting the derivative to zero:

```
dℓ/dθ = 0
```

solved either in closed form or numerically (gradient descent, Newton's method).

**Why MLE is so widely used:** it has excellent asymptotic properties — consistency, asymptotic normality, and efficiency — so with enough data, MLE is approximately the best estimator possible.

**In ML it shows up everywhere:**

- Logistic regression finds the MLE of class probabilities.
- Gaussian mixture models use the EM algorithm to compute MLE of mixture parameters.
- Neural networks trained with cross-entropy loss are implicitly performing MLE.

Understanding MLE helps you see why particular loss functions (like cross-entropy) are natural choices for different problems.

---

### Q10: Explain the method of moments and compare it to MLE.

**A:** The **method of moments** equates sample moments to theoretical moments under the distributional model, then solves for the parameters.

Example for the normal distribution: theory says E[X] = μ and Var(X) = σ², so the method-of-moments estimators are simply:

```
μ̂  = sample mean
σ̂² = sample variance
```

**Compared to MLE:**

- *Method of moments* — easier to compute (just solve equations), works when the likelihood is hard to specify, often provides good starting points for numerical optimization. Less statistically efficient — slower convergence to truth and higher variance.
- *MLE* — typically more efficient, with the asymptotic optimality properties from Q9. More work to compute, especially when no closed form exists.

Both are consistent and asymptotically normal. The method of moments is useful when likelihood is difficult to specify (mixture models, for example) or when quick estimates suffice, but for formal inference with small samples, MLE is preferred. Interviewers appreciate the framing as *computational simplicity vs. statistical efficiency*.

---

### Q11: What is Fisher information and what does it tell you about an estimator?

**A:** **Fisher information** I(θ) measures how much information the data carries about a parameter θ. It's defined as the negative expected second derivative of the log-likelihood:

```
I(θ) = − E[ d²ℓ/dθ² | θ ]
```

Intuitively, large Fisher information means the likelihood is sharply peaked at the true θ — the data constrains θ tightly. Small information means the likelihood is flat — the data tells you little about θ.

The **Cramér-Rao lower bound** says that for any unbiased estimator θ̂:

```
Var(θ̂) ≥ 1 / I(θ)
```

That is, the inverse Fisher information is the *minimum possible variance* of any unbiased estimator. MLEs reach this bound asymptotically — they're efficient estimators.

The link to standard errors:

```
SE(θ̂) ≈ 1 / √I(θ)        (in large samples)
```

so you can predict how precisely you can estimate a parameter *before* collecting data — useful for sample-size planning.

**In ML:** the Fisher information matrix shows up as a preconditioner in second-order optimization methods (natural gradient descent) and in uncertainty quantification for neural network weights. Understanding Fisher information helps you reason about what sample sizes suffice for achieving desired estimation precision.

---

### Q12: Define sufficient statistics and explain their role in inference.

**A:** A **sufficient statistic** T(X) captures all the information in the data that's relevant to a parameter θ — the distribution of X given T(X) doesn't depend on θ. Once you know T(X), the raw data x adds nothing new for inference about θ.

**Examples:**

- Normal distribution with unknown mean and variance — the pair (sample mean, sample variance) is sufficient.
- Poisson distribution with unknown rate λ — the sample sum Σ xᵢ is sufficient.

**Factorization criterion** — a quick way to check sufficiency. T is sufficient if the likelihood factors as:

```
L(θ ; x) = g(T(x), θ) · h(x)
```

where h(x) doesn't depend on θ.

**Why it matters in practice:**

- **Data compression** — for Poisson data, you only need the sum and sample size, not every observation.
- **Bayesian inference** — sufficient statistics determine the posterior, so identifying them tells you which aspects of the data actually matter.
- **Linear regression** — the sufficient statistic involves XᵀX and XᵀY, which is exactly why those terms appear in the normal equations.

---

### Q13: What is the Neyman-Pearson Lemma and why is it important?

**A:** The **Neyman-Pearson Lemma** says: for testing a simple null versus a simple alternative,

```
H₀ : θ = θ₀     vs.     H₁ : θ = θ₁
```

the **most powerful test** at significance level α rejects H₀ whenever the likelihood ratio exceeds a threshold:

```
L(θ₁ ; x) / L(θ₀ ; x)  >  c
```

This test is uniformly most powerful (UMP) among all tests with that significance level.

**Why it matters:**

- It gives a principled recipe for constructing optimal tests: form the likelihood ratio and threshold it.
- Many classical tests are likelihood ratio tests in disguise — t-test, F-test, chi-square test all emerge from this framework.
- Likelihood ratio tests generalize beyond simple-vs-simple hypotheses to comparisons of nested models.

The lemma is also why so many statistical tests have the form "reject if test statistic > threshold." That threshold isn't arbitrary — it's derived to optimize power against the alternative. Understanding this helps you see classical hypothesis tests as principled, not ad hoc.

---

### Q14: Explain multiple testing correction and its relevance to ML model selection.

**A:** When you perform many hypothesis tests, the **family-wise error rate (FWER)** — the probability of at least one false positive — climbs well above the per-test significance level.

Two main approaches to controlling for this:

- **Bonferroni correction:** test each hypothesis at α/m (for m tests). Guarantees FWER ≤ α — simple and safe but conservative.
- **False discovery rate (FDR):** controls the expected *fraction* of false discoveries among rejected hypotheses. The **Benjamini-Hochberg** procedure achieves this and is less stringent than Bonferroni, often preferred for exploratory analysis.

**Why this matters in ML:**

- **Hyperparameter tuning:** trying many settings inflates false discovery — a setting can appear "best" just by chance if you test enough alternatives. Cross-validation only partially addresses this.
- **Feature selection:** if you test 1000 features, about 50 will look significant at p < 0.05 *by chance alone*, even if none are really useful.

Interviewers expect you to know that reporting results from exploratory analysis without correction overstates evidence, and that validation on held-out data is essential for distinguishing true improvement from overfitting to the test set.

---

### Q15: How do you design A/B tests and determine sample size requirements?

**A:** **A/B testing** compares two variants (control A and treatment B) on randomly assigned users, measuring a primary metric like conversion rate.

**Design steps:**

1. Specify the primary metric and hypothesis (one-sided or two-sided).
2. Specify the minimum detectable effect size (e.g., 10% relative lift).
3. Choose significance level α (typically 0.05) and power (typically 0.80).
4. Calculate the required sample size n per variant.
5. Run the test with proper randomization and blinding.
6. Analyze via t-test or proportion test at the specified α level.

**Sample size formula for comparing proportions** (rule of thumb, per arm):

```
n ≈ 2 · (z_α + z_β)² · p · (1 − p) / Δ²
```

where p is the (assumed common) baseline rate and Δ is the minimum detectable effect. This is an approximation that assumes the two groups have similar baseline proportions; for materially different p₁ and p₂, use the more general formula `n ≈ (z_α + z_β)² · [ p₁(1−p₁) + p₂(1−p₂) ] / Δ²` or a power-analysis tool (e.g., `statsmodels.stats.power`).

**Practical considerations:**

- **Don't peek** at results mid-test — repeated checks inflate α. Use sequential testing if you want early stopping.
- Apply multiple-comparison corrections if testing multiple metrics.
- **Pre-register** the analysis plan to prevent p-hacking.

**In ML:** A/B tests are how you validate that model improvements actually generalize — online metrics matter more than offline metrics, and a clean A/B test is what convinces stakeholders the change is real.

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

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

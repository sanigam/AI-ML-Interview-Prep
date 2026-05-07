# Probability and Statistics Fundamentals

📺 **Video Lecture:** https://youtu.be/T2v-1SwoTZQ

## Interview Anchor
- **Probability Axioms:** Foundational rules (non-negativity, unitarity, additivity) that all probability models must satisfy
- **Conditional Probability:** The probability of an event occurring given that another event has already occurred, denoted P(A|B)
- **Bayes' Theorem:** A formula relating conditional probabilities that enables updating beliefs when new evidence arrives

## Key Concepts Overview
Probability and statistics form the mathematical foundation for machine learning, enabling us to model uncertainty, make predictions with confidence, and derive meaningful insights from data. Understanding these fundamentals is essential for ML practitioners because almost every algorithm—from logistic regression to Bayesian neural networks—relies on probabilistic reasoning. Interviewers frequently test this knowledge because it reveals whether candidates understand how ML models handle uncertainty and how to interpret their outputs statistically.

In practice, you'll encounter probability distributions constantly: normal distributions in regression residuals, Poisson distributions for count data, exponential distributions in time-to-event analysis. Mastering the relationships between different ways of characterizing distributions (PDFs, CDFs, moments) allows you to quickly assess model assumptions and troubleshoot issues when assumptions are violated.

---

### Q1: What are the three axioms of probability and why are they important?

**A:** Probability theory rests on three axioms that any valid probability model must satisfy:

- **Non-negativity:** P(A) ≥ 0 for every event A.
- **Unitarity:** P(Ω) = 1, where Ω is the sample space (something must happen).
- **Countable additivity:** for mutually exclusive events A₁, A₂, ..., the probability of their union is the sum of their probabilities:

  ```
  P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...
  ```

These three axioms ensure mathematical consistency and let us derive every other rule of probability from first principles — for example, P(A) + P(Aᶜ) = 1 follows directly.

In interviews, citing these axioms demonstrates you know probability isn't arbitrary but is built on rigorous mathematical foundations.

---

### Q2: Explain conditional probability and how it relates to independence.

**A:** **Conditional probability** is the probability of A given that B has already occurred:

```
P(A | B) = P(A ∩ B) / P(B)
```

Two events are **independent** if conditioning on one tells you nothing about the other:

```
P(A | B) = P(A)        (equivalently: P(A ∩ B) = P(A)·P(B))
```

This is practically useful in ML: if features are independent, you can model them separately (as in naive Bayes); if they're dependent, you need to account for the relationship.

A common interview mistake is confusing independence with **mutual exclusivity**, which means P(A ∩ B) = 0. Independent events can co-occur — knowing one just doesn't change the probability of the other. Rolling two dice produces independent events; getting heads and getting tails on the same toss are mutually exclusive but dependent.

---

### Q3: State Bayes' Theorem and provide a machine learning interpretation.

**A:** **Bayes' Theorem** relates four probabilities — prior, likelihood, evidence, and posterior:

```
P(A | B) = P(B | A) · P(A) / P(B)
```

The four terms have specific names:

- **Prior** P(A) — your belief in A before seeing data.
- **Likelihood** P(B | A) — how probable the observed data is under hypothesis A.
- **Evidence** P(B) — the marginal probability of B (a normalizer).
- **Posterior** P(A | B) — updated belief in A after seeing the data.

In ML terms, if A is a hypothesis ("this email is spam") and B is observed data, Bayes' Theorem says how to update belief in the hypothesis after seeing the data. The numerator P(B | A) · P(A) is the likelihood weighted by prior belief; the denominator P(B) just normalizes across all possible hypotheses.

This framework underlies Bayesian neural networks, Bayesian optimization, and probabilistic models in production. Interviewers appreciate the framing that Bayes' Theorem is a principled way to incorporate prior knowledge and quantify uncertainty.

---

### Q4: What is the difference between a probability distribution and a probability density function?

**A:** A **probability distribution** is the complete specification of how probability is assigned to outcomes — applicable to either discrete or continuous random variables. The **probability density function (PDF)** specifically describes density for *continuous* random variables.

A subtle but critical point about PDFs: the probability at any single point is zero. The PDF instead gives probability *per unit width*, and the area under the PDF curve over an interval gives the probability that X falls in that interval.

For discrete distributions, the analog of a PDF is the **probability mass function (PMF)**, which directly gives probabilities at each outcome. The **cumulative distribution function (CDF)** works for both — F(x) = P(X ≤ x) gives the probability that X is at most x.

In practice, the PDF tells you where values are likely to concentrate. A normal distribution's PDF peaks at the mean, so samples concentrate near the mean. When working with continuous ML variables (weights, activations), you're implicitly working with PDFs.

---

### Q5: Describe the normal (Gaussian) distribution and explain why it's central to statistics and ML.

**A:** The **normal distribution** with mean μ and standard deviation σ has the bell-shaped PDF:

```
            1            (x − μ)²
f(x) = ───────────  · exp( − ──────── )
        σ · √(2π)            2σ²
```

It's symmetric around μ and fully characterized by just its mean and variance.

**Why it's everywhere:** the **central limit theorem** guarantees that sums (or averages) of independent random variables approach normality regardless of their original distribution. This is the default assumption in many statistical methods.

**Where it appears in ML:**

- Residuals in linear regression are assumed normal.
- Prediction uncertainties in Gaussian processes are normal.
- Neural network weights are typically initialized from normal distributions.

The **68–95–99.7 rule** is a fast sanity check — about 68% of values lie within 1σ of the mean, 95% within 2σ, and 99.7% within 3σ. This makes it easy to gauge how unusual an observation is.

Before applying methods that assume normality (like t-tests), check the assumption with a Q-Q plot.

---

### Q6: What are the binomial, Poisson, and exponential distributions and when would you use each?

**A:** Three discrete/continuous distributions that come up constantly in ML:

- **Binomial B(n, p):** number of successes in n independent trials with success probability p each. Use it for binary classification accuracy estimation, counting defects, or click-through counts.
- **Poisson P(λ):** counts of rare events occurring randomly over time or space at average rate λ. Use it for website traffic modeling, fraud detection, or predicting customer complaints.
- **Exponential E(λ):** waiting time between Poisson events. Use it for customer lifetime value, time-to-failure analysis, or session duration modeling.

A useful relationship to remember: **if events arrive as a Poisson process, the time between them is exponential.**

In practice, Poisson distributions appear in GLMs for count regression, and recognizing when your data is Poisson-distributed helps you select an appropriate model rather than incorrectly assuming normality.

---

### Q7: Define expectation (mean) and variance, and explain their importance in ML.

**A:** **Expectation** E[X] is the center of mass — the average value of a random variable. The formula depends on whether X is discrete or continuous:

```
discrete:    E[X] = Σ x · P(x)
continuous:  E[X] = ∫ x · f(x) dx
```

**Variance** measures spread around the mean. Two equivalent forms:

```
Var(X) = E[(X − E[X])²]
       = E[X²] − (E[X])²
```

The **standard deviation** σ = √Var(X) puts the spread in the original units of X.

**Why this matters in ML:** the mean is the typical prediction or feature value; the variance quantifies uncertainty. High-variance models overfit (predictions swing widely with different training sets); high-bias models underfit. This is the foundation of the **bias–variance tradeoff**:

```
total error ≈ bias² + variance + irreducible noise
```

When you normalize features to zero mean and unit variance, you're standardizing these statistics to make optimization easier. Note that variance in the probabilistic sense (spread of a distribution) and statistical variance (a sample-based estimate) are different — both matter for understanding model behavior.

---

### Q8: Explain covariance and correlation and how they differ.

**A:** **Covariance** measures how two variables move together:

```
Cov(X, Y) = E[ (X − E[X]) · (Y − E[Y]) ]
```

- Positive covariance — X and Y tend to increase together.
- Negative covariance — they move in opposite directions.
- Zero covariance — no linear relationship.

**Correlation** is covariance normalized by the standard deviations:

```
ρ = Cov(X, Y) / (σ_X · σ_Y)
```

This rescales the value into [−1, +1], making it unit-free and comparable across datasets.

**Key difference:** covariance's magnitude depends on the scales of X and Y, so it isn't directly comparable across problems. Correlation is scale-invariant. A value near ±1 indicates a strong linear relationship; near 0 indicates a weak or nonexistent linear relationship.

Two important caveats:

- **Correlation doesn't imply causation.**
- Correlation only captures *linear* relationships — two variables can be perfectly nonlinearly dependent yet have zero correlation.

In feature engineering, high correlations between features signal multicollinearity, which can destabilize regression coefficients.

---

### Q9: What is the law of large numbers and how does it apply to ML?

**A:** The **law of large numbers (LLN)** says that as sample size n grows, the sample mean converges to the population mean:

```
(1/n) · Σᵢ Xᵢ  →  E[X]    as n → ∞
```

(With probability 1 in the strong LLN, or in probability in the weak LLN.) This is what justifies using empirical averages — sample mean, sample variance — as estimates of population parameters.

**In ML:** LLN guarantees that as you collect more training data, your empirical loss (average error on the training set) becomes a better approximation of true expected loss on the population. This is one reason bigger datasets tend to give better models.

**The key caveat:** LLN only guarantees convergence to the *population* mean. If your population is itself biased — selection bias, label noise — larger samples won't fix it. This is why data quality matters as much as quantity, and why testing on held-out data from the same population is essential.

---

### Q10: State the central limit theorem and explain its significance for inference and ML.

**A:** The **central limit theorem (CLT)** says that for i.i.d. random variables X₁, X₂, ..., Xₙ with finite mean μ and finite variance σ², the sample mean is approximately normal for large n, regardless of the original distribution:

```
M̄ₙ = (1/n) · Σᵢ Xᵢ   ~  Normal(μ, σ²/n)    as n → ∞
```

This is extraordinary: even if the data is uniform, exponential, or some other non-normal distribution, the *average* is approximately normal for large enough n.

**Why this matters in ML:**

- It justifies normal-based confidence intervals for sample means (e.g., test accuracy).
- Mini-batch gradient estimates in SGD are approximately normal because they're averages.
- Many statistical tests assume normality of means even when the underlying data isn't normal.

It also explains the famous **1/√n rule**: the standard error of the sample mean is σ/√n, so doubling sample size reduces standard error by about 30%, but you need 4× the data to halve it.

---

### Q11: Explain joint, marginal, and conditional distributions and their relationships.

**A:** Three related views of a multivariate distribution:

- **Joint distribution** P(X, Y) — probability of every combination of X and Y values.
- **Marginal distribution** — sum out (or integrate out) the other variable:

  ```
  P(X) = Σ_y P(X, Y = y)
  ```

- **Conditional distribution** — restrict and renormalize:

  ```
  P(X | Y) = P(X, Y) / P(Y)
  ```

These three are tied together by the **chain rule**:

```
P(X, Y) = P(X | Y) · P(Y) = P(Y | X) · P(X)
```

If you picture a 2D joint distribution table, the marginals are row/column totals and the conditionals are normalized rows or columns.

**ML uses:** factoring models efficiently. The Naive Bayes independence assumption,

```
P(X₁, ..., Xₙ | Y) = Πᵢ P(Xᵢ | Y)
```

drastically simplifies computation. **Graphical models** (Bayesian networks) are built by specifying conditional independence relationships among variables, so being able to read these from joint distributions is essential.

---

### Q12: How do you distinguish between PDF and CDF? What are their practical uses?

**A:** The **PDF** f(x) gives the probability *density* (probability per unit width). The **CDF** F(x) gives the cumulative probability up to x:

```
F(x) = P(X ≤ x)
```

The two are linked by integration and differentiation:

```
F(x) = ∫_{−∞}^x f(u) du           (CDF as integral of PDF)
f(x) = dF(x)/dx                    (PDF as derivative of CDF)
```

For a discrete distribution, the PMF takes the place of the PDF, and the CDF still gives cumulative probabilities.

**When to use which:**

- **PDFs** are for visualizing where data concentrates and for evaluating likelihoods in probabilistic models.
- **CDFs** are for percentiles (e.g., F⁻¹(0.95) is the 95th percentile), tail probabilities (P(X > threshold) = 1 − F(threshold)), and many hypothesis tests.

The Kolmogorov-Smirnov test compares empirical CDFs across samples. When reporting model uncertainties, you often quote CDF-based intervals to quantify the range of plausible values.

---

### Q13: What are moments and moment-generating functions, and why do they matter?

**A:** The **k-th moment** of a distribution is the expected value of Xᵏ:

```
k-th moment = E[Xᵏ]
```

The first few moments correspond to familiar quantities:

- **1st moment** — the mean.
- **2nd central moment** — the variance.
- **3rd central moment** — skewness (asymmetry of the distribution).
- **4th central moment** — kurtosis (heaviness of tails).

The **moment-generating function (MGF)** is a single function that encodes all moments:

```
M(t) = E[ e^(t·X) ]
```

You can recover any moment by differentiating M(t) at t = 0:

```
E[Xᵏ] = M^(k)(0)
```

The MGF **uniquely characterizes** a distribution — if two random variables have the same MGF, they have the same distribution. This is powerful because you can derive moments without doing integrals, just by differentiating M(t).

**Why it matters in interviews:** different distributions have characteristic shapes (skewness, kurtosis). Recognizing them goes beyond mean/variance and shows you understand the data. For example, income distributions are right-skewed (long tail of high earners), so the median is more informative than the mean — that's practical knowledge that distinguishes thoughtful analysts from those who blindly assume normality.

---

### Q14: What does it mean for a statistic to be sufficient, and when is this important?

**A:** A statistic T(X) is **sufficient** for parameter θ if the conditional distribution of the data X given T(X) doesn't depend on θ. Intuitively: once you know T(X), the raw data gives you no additional information about θ.

**Examples:**

- For a normal distribution with unknown mean μ, the sample mean is sufficient for μ.
- For a Poisson distribution with unknown rate λ, the sample sum Σxᵢ is sufficient (equivalently, the sample mean when n is fixed).

The **factorization criterion** is a practical test: T is sufficient if the likelihood factors as

```
L(θ ; x) = g(T(x), θ) · h(x)
```

where h(x) doesn't depend on θ.

**Why this matters in practice:** sufficiency means you can compress data down to a sufficient statistic without losing any information about θ. That's useful for summarization, understanding which statistics actually drive inference, and for engineering reasons like A/B testing and monitoring systems — you can track just the sufficient statistics rather than all raw events.

---

### Q15: How do independence and conditional independence affect model design and inference?

**A:** Two variables are **independent** if knowing one tells you nothing about the other:

```
P(X, Y) = P(X) · P(Y)
```

**Conditional independence** means independence holds *given* a third variable. Written X ⊥ Y | Z:

```
P(X, Y | Z) = P(X | Z) · P(Y | Z)
```

**Where this shows up:**

- **Naive Bayes** assumes features are conditionally independent given the class label:

  ```
  P(X₁, ..., Xₙ | Y) = Πᵢ P(Xᵢ | Y)
  ```

  This is rarely literally true but makes computation tractable and often works well in practice.

- **Wrong independence assumptions** lead to trouble: assuming independence when variables are correlated underestimates uncertainty and produces overconfident predictions.

- **Graphical models** encode independence assumptions structurally — in a directed acyclic graph, a variable is independent of its non-descendants given its parents (d-separation).

Understanding which variables must be independent, conditionally independent, or dependent is critical for building interpretable models and avoiding bugs where two features inappropriately influence each other.

---

## Interview Cheatsheet

**Key Terms:**
- **Probability Axioms:** Non-negativity, unitarity, additivity; foundation of all probability models
- **Conditional Probability:** P(A|B) = P(A∩B)/P(B); updated probability given an event occurred
- **Bayes' Theorem:** P(A|B) = P(B|A)·P(A)/P(B); relates prior, likelihood, and posterior
- **PDF vs PMF vs CDF:** PDF for continuous (density), PMF for discrete (probability), CDF for cumulative probability
- **Expectation:** E[X] = average value, center of mass of distribution
- **Variance:** Var(X) = E[(X-E[X])²]; measures spread around mean
- **Covariance:** Cov(X,Y) measures joint variation (scale-dependent)
- **Correlation:** ρ = Cov(X,Y)/(σ_X·σ_Y); normalized covariance in [-1, 1]
- **Law of Large Numbers:** Sample mean converges to population mean as n increases
- **Central Limit Theorem:** Sample means are approximately normal regardless of original distribution
- **Joint Distribution:** P(X, Y) specifies probability over multiple variables
- **Marginal Distribution:** P(X) = Σ_y P(X, Y=y); distribution of single variable
- **Conditional Distribution:** P(X|Y) = P(X,Y)/P(Y); distribution given another variable
- **Sufficient Statistic:** T(X) captures all information relevant to parameter θ
- **Independence:** X ⊥ Y means P(X,Y) = P(X)·P(Y)
- **Conditional Independence:** X ⊥ Y | Z means P(X,Y|Z) = P(X|Z)·P(Y|Z)

**Rapid-Fire Q&A:**
- **Q: Why is the normal distribution so important in statistics?** **A:** Central limit theorem ensures sample means are approximately normal; default assumption in many tests and models
- **Q: What's the difference between Poisson and exponential distributions?** **A:** Poisson models count of events; exponential models time between events
- **Q: How do you interpret P(A|B)?** **A:** Probability of A occurring given B has occurred; updated probability incorporating new information
- **Q: What does it mean if Cov(X,Y) = 0?** **A:** No linear relationship, but they could still be nonlinearly dependent
- **Q: Why does variance decrease with √n?** **A:** Standard error of sample mean is σ/√n; larger samples give more stable estimates

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

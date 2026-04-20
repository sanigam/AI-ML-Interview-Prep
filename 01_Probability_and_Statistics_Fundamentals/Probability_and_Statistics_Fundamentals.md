# Probability and Statistics Fundamentals

## Interview Anchor
- **Probability Axioms:** Foundational rules (non-negativity, unitarity, additivity) that all probability models must satisfy
- **Conditional Probability:** The probability of an event occurring given that another event has already occurred, denoted P(A|B)
- **Bayes' Theorem:** A formula relating conditional probabilities that enables updating beliefs when new evidence arrives

## Key Concepts Overview
Probability and statistics form the mathematical foundation for machine learning, enabling us to model uncertainty, make predictions with confidence, and derive meaningful insights from data. Understanding these fundamentals is essential for ML practitioners because almost every algorithm—from logistic regression to Bayesian neural networks—relies on probabilistic reasoning. Interviewers frequently test this knowledge because it reveals whether candidates understand how ML models handle uncertainty and how to interpret their outputs statistically.

In practice, you'll encounter probability distributions constantly: normal distributions in regression residuals, Poisson distributions for count data, exponential distributions in time-to-event analysis. Mastering the relationships between different ways of characterizing distributions (PDFs, CDFs, moments) allows you to quickly assess model assumptions and troubleshoot issues when assumptions are violated.

---

### Q1: What are the three axioms of probability and why are they important?

**A:** The three axioms are: (1) Non-negativity: P(A) ≥ 0 for any event A, (2) Unitarity: P(Ω) = 1 where Ω is the sample space, (3) Additivity (countable): for mutually exclusive events A₁, A₂, ..., P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + .... These axioms are the foundation of probability theory—any valid probability model must satisfy them. They ensure mathematical consistency and allow us to derive all other probability rules (like P(A) + P(Aᶜ) = 1) from first principles. In interviews, mentioning that you understand these axioms demonstrates you know probability isn't arbitrary but built on rigorous mathematical foundations.

---

### Q2: Explain conditional probability and how it relates to independence.

**A:** Conditional probability P(A|B) = P(A ∩ B) / P(B) represents the probability of A given that B has occurred. Two events are independent if P(A|B) = P(A), meaning B provides no information about A. This is practical in ML: if features are independent, you can model them separately (as in naive Bayes); if they're dependent, you need to account for their relationship. A common interview mistake is confusing independence with mutual exclusivity (which means P(A ∩ B) = 0). Independent events can co-occur, but knowing one doesn't change the probability of the other. For example, rolling two dice are independent events, while getting heads and getting tails on the same coin toss are mutually exclusive but dependent.

---

### Q3: State Bayes' Theorem and provide a machine learning interpretation.

**A:** Bayes' Theorem states: P(A|B) = P(B|A) × P(A) / P(B), where P(A) is the prior, P(B|A) is the likelihood, P(B) is the evidence, and P(A|B) is the posterior. In ML terms, if A represents a hypothesis (like "this email is spam") and B represents observed data, Bayes' Theorem tells us how to update our belief in the hypothesis given the data. The numerator P(B|A) × P(A) represents how likely we are to see this data under our hypothesis weighted by prior belief, while P(B) normalizes across all possible hypotheses. This framework underlies Bayesian neural networks, Bayesian optimization, and probabilistic models used in production systems. Interviewers appreciate when you explain that Bayes' Theorem provides a principled way to incorporate prior knowledge and quantify uncertainty.

---

### Q4: What is the difference between a probability distribution and a probability density function?

**A:** A probability distribution is the complete specification of how probability is assigned to outcomes (discrete or continuous), while a probability density function (PDF) specifically describes the probability density for continuous random variables—the actual probability at any single point is zero, but the area under the PDF curve over an interval gives the probability. For discrete distributions, we use probability mass functions (PMFs) instead, which directly give probabilities. The cumulative distribution function (CDF) F(x) = P(X ≤ x) works for both, giving the probability that X is at most x. In practice, the PDF tells you where values are likely to concentrate: a normal distribution's PDF peaks at the mean, so you'd expect values near the mean when sampling. When working with continuous ML variables (weights, activations), you're implicitly working with PDFs.

---

### Q5: Describe the normal (Gaussian) distribution and explain why it's central to statistics and ML.

**A:** The normal distribution with parameters μ (mean) and σ (standard deviation) has PDF f(x) = (1/(σ√(2π))) × exp(-(x-μ)²/(2σ²)). It's symmetric around μ and fully characterized by its mean and variance. The central limit theorem guarantees that sums of independent random variables approach normality regardless of their original distribution, making it the default assumption in many statistical methods. In ML, residuals in linear regression are assumed normal, prediction uncertainties in Gaussian processes are normally distributed, and neural network weights are often initialized from normal distributions. The 68-95-99.7 rule (roughly 68% within 1σ, 95% within 2σ) helps you quickly assess how unusual an observation is. Always check if your data violates normality assumptions using Q-Q plots before applying methods like t-tests.

---

### Q6: What are the binomial, Poisson, and exponential distributions and when would you use each?

**A:** The binomial distribution B(n, p) models the number of successes in n independent trials with success probability p each; use it for binary classification accuracy estimation or counting defects. The Poisson distribution P(λ) models counts of rare events occurring randomly over time/space at average rate λ; use it for website traffic modeling, fraud detection, or predicting customer complaints. The exponential distribution E(λ) models waiting times between Poisson events with parameter λ; use it for customer lifetime value, time-to-failure analysis, or session duration modeling. Key relationship: if events follow Poisson, the time between them follows exponential. In practice, Poisson is often used in GLMs for count regression, and recognizing when your data is Poisson-distributed helps you select appropriate models rather than incorrectly assuming normality.

---

### Q7: Define expectation (mean) and variance, and explain their importance in ML.

**A:** Expectation E[X] = Σ x·P(x) (discrete) or ∫ x·f(x)dx (continuous) represents the center of mass or average value of a random variable. Variance Var(X) = E[(X - E[X])²] = E[X²] - (E[X])² measures spread around the mean. Standard deviation σ = √Var(X) is variance in the original units. In ML, the mean tells you the typical prediction or feature value, while variance quantifies uncertainty—high variance models overfit (high Var on training data, low on test), while high-bias models underfit. The bias-variance tradeoff is foundational: total error ≈ bias² + variance + noise. When you normalize features to zero mean and unit variance, you're standardizing these statistics to make optimization easier. Interviewers expect you to understand that variance in the probabilistic sense (spread of a distribution) differs from statistical variance (sample-based estimate), and both matter for understanding model behavior.

---

### Q8: Explain covariance and correlation and how they differ.

**A:** Covariance Cov(X,Y) = E[(X - E[X])(Y - E[Y])] measures how two variables move together—positive means they co-increase, negative means they move opposite, zero means no linear relationship. Correlation ρ = Cov(X,Y) / (σ_X × σ_Y) normalizes covariance to [-1, 1], making it unit-free and comparable across different scales. Key difference: covariance's magnitude depends on the scales of X and Y, so you can't compare covariances across datasets, but correlation is scale-invariant. A correlation near ±1 indicates strong linear relationship; near 0 indicates weak or no linear relationship. Important: correlation doesn't imply causation and only captures linear relationships (two variables can be perfectly dependent nonlinearly with zero correlation). In feature engineering, high correlations between features signal multicollinearity that can destabilize regression coefficients.

---

### Q9: What is the law of large numbers and how does it apply to ML?

**A:** The law of large numbers (LLN) states that as sample size n increases, the sample mean converges to the population mean: (1/n)Σᵢ₌₁ⁿ Xᵢ → E[X] with probability 1 (strong LLN) or in probability (weak LLN). This justifies using empirical averages (sample mean, sample variance) as estimates of population parameters. In ML, LLN guarantees that as you collect more training data, your empirical loss (average error on training set) better approximates true expected loss on the population, which is why bigger datasets generally lead to better models. However, LLN only guarantees convergence to the population mean—if your population itself is biased (selection bias, label noise), larger samples won't help. This is why data quality matters as much as quantity, and why testing on held-out data from the same population is essential.

---

### Q10: State the central limit theorem and explain its significance for inference and ML.

**A:** The central limit theorem (CLT) states that for i.i.d. random variables X₁, X₂, ..., Xₙ with finite mean μ and variance σ², the sample mean M̄ₙ = (1/n)Σᵢ₌₁ⁿ Xᵢ is approximately normally distributed with mean μ and variance σ²/n as n → ∞, regardless of the original distribution of the Xᵢ. This is extraordinary: even if your data comes from a uniform, exponential, or other non-normal distribution, the average is approximately normal for large enough n. In ML, CLT justifies why we can use normal-based confidence intervals for sample means (like test accuracy), why minibatch gradient estimates in SGD are approximately normal-distributed, and why many statistical tests assume normality of means even if underlying data isn't normal. This explains why adding more samples (increasing n) reduces the standard error of your estimates by √n.

---

### Q11: Explain joint, marginal, and conditional distributions and their relationships.

**A:** For a joint distribution P(X, Y) over two variables, the marginal distribution P(X) = Σ_y P(X, Y=y) sums over the other variable, and the conditional distribution P(X|Y) = P(X, Y) / P(Y). These relate via the chain rule: P(X, Y) = P(X|Y)·P(Y) = P(Y|X)·P(X). Visually, if you have a 2D joint distribution table, marginals are row/column totals, and conditionals are normalized rows/columns. In ML, whenever you have multiple variables, understanding these relationships helps you factor models efficiently: the independence assumption in Naive Bayes (P(X₁, ..., Xₙ | Y) = Πᵢ P(Xᵢ | Y)) dramatically simplifies computation. Graphical models (Bayesian networks) are built by specifying conditional independence relationships among variables, so understanding how to read these from joint distributions is essential.

---

### Q12: How do you distinguish between PDF and CDF? What are their practical uses?

**A:** The PDF f(x) describes the probability density (probability per unit width), while the CDF F(x) = P(X ≤ x) is the cumulative probability up to point x. Mathematically, CDF is the integral of PDF: F(x) = ∫_{-∞}^x f(u)du, and conversely f(x) = dF(x)/dx. For a discrete distribution, the PMF gives probabilities and the CDF still gives cumulative probabilities. Practically, you use PDFs to visualize where data concentrates and to evaluate likelihood in probabilistic models, while you use CDFs for calculating percentiles (P(X ≤ value) = 0.95 tells you the 95th percentile), computing tail probabilities (P(X > threshold)), and hypothesis testing. In Kolmogorov-Smirnov tests, you compare empirical CDFs across samples. When reporting model uncertainties, you often specify CDF-based confidence intervals: "95% CI means there's a 95% probability the true value is between these bounds."

---

### Q13: What are moments and moment-generating functions, and why do they matter?

**A:** The k-th moment of a distribution is E[Xᵏ], with the 1st moment being the mean and the 2nd central moment being variance. Higher moments capture skewness (3rd moment) and kurtosis (4th moment). The moment-generating function (MGF) is M(t) = E[e^{tX}], and all moments can be recovered as M^(k)(0) = E[Xᵏ]. The MGF uniquely characterizes a distribution—if two random variables have the same MGF, they have the same distribution. This is powerful: you can derive moments without integration by differentiating the MGF. In interviews, knowing that different distributions have characteristic shapes (skewness, kurtosis) shows you understand data beyond just mean/variance. For example, income distributions are right-skewed (long tail of high earners), so median is more informative than mean—this is practical knowledge that differentiates thoughtful analysts from those who blindly assume normality.

---

### Q14: What does it mean for a distribution to be a sufficient statistic, and when is this important?

**A:** A statistic T(X) is sufficient for parameter θ if the conditional distribution of data X given T(X) doesn't depend on θ. Intuitively, a sufficient statistic captures all the information in the data relevant to θ—the raw data gives no additional information once you know T(X). For a normal distribution with unknown mean μ, the sample mean is sufficient for μ; for Poisson with unknown rate λ, the sample mean is sufficient. Factorization criterion: T is sufficient if you can write the likelihood as L(θ; x) = g(T(x), θ) × h(x) where h doesn't depend on θ. In practice, sufficiency matters because you can compress data down to a sufficient statistic without losing information—this is useful for summarization and understanding what statistics actually matter for inference. When designing A/B tests or building monitoring systems, identifying sufficient statistics helps you track just enough information to make decisions.

---

### Q15: How do independence and conditional independence affect model design and inference?

**A:** Two variables are independent if P(X, Y) = P(X)·P(Y), meaning knowing one tells you nothing about the other. Conditional independence, written X ⊥ Y | Z, means P(X, Y | Z) = P(X|Z)·P(Y|Z)—they're independent given a third variable. Naive Bayes assumes features are conditionally independent given the class label: P(X₁, ..., Xₙ | Y) = Πᵢ P(Xᵢ | Y), which is rarely true but makes computation tractable and often works well in practice. Conversely, assuming independence when variables are correlated leads to underestimated uncertainties and overconfident predictions. Graphical models encode independence assumptions: in a directed acyclic graph, variables are independent of their non-descendants given their parents (d-separation criterion). Understanding which variables must be independent, conditionally independent, or dependent is critical for building interpretable models and avoiding bugs where two features inappropriately influence each other when they shouldn't.

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

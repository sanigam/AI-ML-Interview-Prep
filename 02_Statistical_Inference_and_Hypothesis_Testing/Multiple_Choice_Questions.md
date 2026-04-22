# Multiple Choice Questions: Statistical Inference and Hypothesis Testing

📺 **Video Lecture:** https://youtu.be/4NlWsOKGBLc


Test your understanding of statistical inference, hypothesis testing, and experimental design for AI/ML interviews.

---

**Q1. A 95% confidence interval means:**

A) There is a 95% probability that the true parameter lies within this specific interval
B) If we repeated the sampling procedure many times, approximately 95% of the constructed intervals would contain the true parameter
C) 95% of the sample data falls within this interval
D) The parameter has a 95% chance of being equal to the point estimate

---

**Q2. A researcher conducts an A/B test and obtains a p-value of 0.03. What is the correct interpretation?**

A) There is a 3% probability that the null hypothesis is true
B) There is a 97% probability that the treatment works
C) If the null hypothesis were true, there is a 3% probability of observing results this extreme or more extreme
D) The effect size is 3%

---

**Q3. In hypothesis testing, a Type II error occurs when:**

A) You reject the null hypothesis when it is actually true
B) You fail to reject the null hypothesis when it is actually false
C) You accept the alternative hypothesis correctly
D) The p-value is less than the significance level

---

**Q4. Statistical power is defined as:**

A) The probability of rejecting the null hypothesis when it is true
B) The probability of correctly failing to reject a true null hypothesis
C) The probability of correctly rejecting a false null hypothesis
D) The probability of obtaining a p-value less than 0.05

---

**Q5. Which of the following increases statistical power?**

A) Decreasing the sample size
B) Increasing the significance level (α) from 0.01 to 0.05
C) Decreasing the effect size
D) Increasing the number of hypothesis tests performed

---

**Q6. An estimator θ̂ is said to be unbiased if:**

A) It always equals the true parameter value
B) Its variance is zero
C) E[θ̂] = θ (its expected value equals the true parameter)
D) It has the smallest possible mean squared error

---

**Q7. Why do we divide by (n−1) instead of n when computing sample variance?**

A) To make the computation simpler
B) To correct for the bias introduced by using the sample mean instead of the population mean
C) Because the sample always has one fewer observation than the population
D) To ensure the variance is always positive

---

**Q8. Maximum Likelihood Estimation (MLE) finds the parameter value that:**

A) Minimizes the sum of squared errors
B) Maximizes the probability of the observed data given the parameter
C) Minimizes the prior probability of the parameter
D) Maximizes the posterior probability of the parameter given the data

---

**Q9. When comparing means across 4 groups, which test is most appropriate?**

A) Paired t-test
B) Chi-square test
C) One-way ANOVA
D) Two-sample t-test repeated 6 times

---

**Q10. A researcher tests 20 hypotheses at α = 0.05 with no correction. Approximately how many false positives are expected if all null hypotheses are true?**

A) 0
B) 1
C) 5
D) 20

---

**Q11. The Bonferroni correction for m = 10 tests at family-wise α = 0.05 sets each individual test significance at:**

A) 0.05
B) 0.01
C) 0.005
D) 0.50

---

**Q12. Which of the following is a non-parametric alternative to the independent two-sample t-test?**

A) Paired t-test
B) Mann-Whitney U test
C) Chi-square test
D) F-test

---

**Q13. The Fisher Information I(θ) is important because:**

A) It determines the maximum possible bias of an estimator
B) Its inverse provides a lower bound on the variance of any unbiased estimator (Cramér-Rao bound)
C) It equals the sample size required for the test
D) It measures the Type I error rate

---

**Q14. In the context of A/B testing, what is the primary reason you should NOT repeatedly check results and stop early when significance is reached?**

A) It increases the required sample size
B) It inflates the actual Type I error rate above the nominal α level
C) It makes the test non-parametric
D) It decreases the effect size

---

**Q15. A chi-square test for independence is used to determine:**

A) Whether a continuous variable follows a normal distribution
B) Whether the means of two groups are equal
C) Whether two categorical variables are independent of each other
D) Whether the variance of a sample equals a hypothesized value

---

## Answer Key

**Q1. Answer: B**
A confidence interval is a frequentist concept about the procedure, not about a specific interval. If repeated many times, 95% of intervals would capture the true parameter. The parameter is fixed — it's either in this interval or not.

**Q2. Answer: C**
A p-value is the probability of observing data as extreme or more extreme than what was observed, assuming H₀ is true. It is NOT the probability that H₀ is true, nor the probability the treatment works.

**Q3. Answer: B**
Type II error (false negative) means failing to detect a real effect — you don't reject H₀ when it should be rejected. Type I error is rejecting a true H₀. The probability of Type II error is denoted β.

**Q4. Answer: C**
Power = 1 − β = probability of correctly rejecting a false null hypothesis. It measures the test's ability to detect a true effect when one exists.

**Q5. Answer: B**
Increasing α (e.g., from 0.01 to 0.05) makes it easier to reject H₀, thus increasing power. Other ways to increase power include increasing sample size and increasing effect size. Decreasing sample size or effect size reduces power.

**Q6. Answer: C**
An unbiased estimator has E[θ̂] = θ, meaning its average value across many samples equals the true parameter. It can still vary around θ in any single sample — unbiasedness is about the long-run average.

**Q7. Answer: B**
Dividing by (n−1) applies Bessel's correction. Since we estimate the mean from the same sample, we lose one degree of freedom. Using n would systematically underestimate the population variance.

**Q8. Answer: B**
MLE finds θ that maximizes L(θ; x) = P(data | θ), the likelihood of observed data. This differs from MAP (Maximum A Posteriori), which maximizes P(θ | data) by incorporating a prior.

**Q9. Answer: C**
One-way ANOVA is designed to compare means across 3 or more groups simultaneously. Running multiple t-tests inflates the Type I error rate. ANOVA uses an F-test to determine if any group means differ.

**Q10. Answer: B**
With 20 tests at α = 0.05 and all nulls true, expected false positives = 20 × 0.05 = 1. This illustrates why multiple testing corrections are needed when conducting many simultaneous tests.

**Q11. Answer: C**
Bonferroni correction divides α by m: 0.05 / 10 = 0.005. Each individual test must achieve p < 0.005 to be declared significant, controlling the family-wise error rate at 0.05.

**Q12. Answer: B**
The Mann-Whitney U test (also called Wilcoxon rank-sum test) is the non-parametric equivalent of the independent two-sample t-test. It compares distributions using ranks rather than assuming normality.

**Q13. Answer: B**
The Cramér-Rao lower bound states that Var(θ̂) ≥ 1/I(θ) for any unbiased estimator. Fisher Information quantifies how much information the data carries about θ — higher information means tighter possible estimation.

**Q14. Answer: B**
Repeated peeking ("optional stopping") inflates the true Type I error rate well above the nominal α because each peek is an additional opportunity to falsely reject H₀. Sequential testing methods (like group sequential designs) are needed for valid early stopping.

**Q15. Answer: C**
The chi-square test for independence tests whether two categorical variables are associated by comparing observed cell frequencies in a contingency table to expected frequencies under independence.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

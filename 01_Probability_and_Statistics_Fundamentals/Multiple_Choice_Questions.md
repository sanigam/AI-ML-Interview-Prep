# Multiple Choice Questions: Probability and Statistics Fundamentals

📺 **Video Lecture:** https://youtu.be/T2v-1SwoTZQ

Test your understanding of probability and statistics concepts essential for AI/ML interviews.

---

**Q1. Which of the following is NOT one of the three axioms of probability?**

A) P(A) ≥ 0 for any event A  
B) P(Ω) = 1 where Ω is the sample space  
C) P(A) + P(B) = 1 for any two events A and B  
D) For mutually exclusive events A and B, P(A ∪ B) = P(A) + P(B)

---

**Q2. Two events A and B are independent. Which statement is true?**

A) P(A ∩ B) = 0  
B) P(A | B) = P(A)  
C) A and B cannot occur at the same time  
D) P(A ∪ B) = P(A) + P(B)

---

**Q3. In Bayes' Theorem P(A|B) = P(B|A) × P(A) / P(B), what does P(A) represent?**

A) The likelihood  
B) The posterior probability  
C) The prior probability  
D) The evidence

---

**Q4. A medical test has a 99% sensitivity (true positive rate) and 95% specificity (true negative rate). If the disease prevalence is 1%, what is the approximate probability that a person who tests positive actually has the disease?**

A) About 99%  
B) About 95%  
C) About 17%  
D) About 50%

---

**Q5. Which distribution would you use to model the number of website visitors per hour?**

A) Normal distribution  
B) Exponential distribution  
C) Poisson distribution  
D) Binomial distribution

---

**Q6. The Central Limit Theorem (CLT) states that:**

A) All data in nature follows a normal distribution  
B) The sample mean approaches a normal distribution as sample size increases, regardless of the original distribution  
C) The variance of a sample always equals the population variance  
D) Larger samples always produce more accurate predictions

---

**Q7. If Cov(X, Y) = 0, which of the following is true?**

A) X and Y are independent  
B) There is no linear relationship between X and Y  
C) X and Y are mutually exclusive  
D) X and Y have the same distribution

---

**Q8. What is the relationship between the PDF and the CDF of a continuous random variable?**

A) The CDF is the derivative of the PDF  
B) The PDF is the integral of the CDF  
C) The CDF is the integral of the PDF from −∞ to x  
D) The PDF and CDF are always equal

---

**Q9. The standard error of the sample mean is σ/√n. What happens when you quadruple the sample size?**

A) The standard error is halved  
B) The standard error is quartered  
C) The standard error is doubled  
D) The standard error remains unchanged

---

**Q10. Which of the following correctly describes the 68-95-99.7 rule for a normal distribution?**

A) 68% of data falls within 2 standard deviations of the mean  
B) 95% of data falls within 1 standard deviation of the mean  
C) 99.7% of data falls within 3 standard deviations of the mean  
D) 68% of data falls within 3 standard deviations of the mean

---

**Q11. In the context of Naive Bayes classification, what key assumption is made about features?**

A) All features follow a normal distribution  
B) Features are mutually exclusive  
C) Features are conditionally independent given the class label  
D) Features must have zero correlation with each other

---

**Q12. Which distribution models the waiting time between events that follow a Poisson process?**

A) Normal distribution  
B) Binomial distribution  
C) Exponential distribution  
D) Uniform distribution

---

**Q13. Variance of a random variable X is defined as:**

A) E[X] − E[X²]  
B) E[X²] − (E[X])²  
C) (E[X])² − E[X²]  
D) E[X] × E[X]

---

**Q14. What does a sufficient statistic T(X) for a parameter θ guarantee?**

A) T(X) always equals θ  
B) The raw data provides no additional information about θ beyond what T(X) provides  
C) T(X) is always the sample mean  
D) The distribution of T(X) is always normal

---

**Q15. Correlation ρ between two variables ranges from −1 to +1. A value of ρ = −0.95 indicates:**

A) A weak negative linear relationship  
B) A strong positive linear relationship  
C) No relationship between the variables  
D) A strong negative linear relationship

---

## Answer Key

**Q1. Answer: C**
P(A) + P(B) = 1 is not an axiom. The additivity axiom applies only to mutually exclusive events: P(A ∪ B) = P(A) + P(B). The rule P(A) + P(Aᶜ) = 1 applies to an event and its complement, not any two arbitrary events.

**Q2. Answer: B**
Independence means P(A|B) = P(A) — knowing B occurred doesn't change the probability of A. Independence is often confused with mutual exclusivity (P(A ∩ B) = 0), which is a different concept entirely. Independent events can and do co-occur.

**Q3. Answer: C**
P(A) is the prior probability — our initial belief about A before observing evidence B. P(B|A) is the likelihood, P(A|B) is the posterior, and P(B) is the evidence (normalizing constant).

**Q4. Answer: C**
Using Bayes' Theorem: P(disease|positive) = (0.99 × 0.01) / ((0.99 × 0.01) + (0.05 × 0.99)) ≈ 0.0099 / 0.0594 ≈ 16.7%. This classic result shows that even accurate tests produce many false positives when the base rate (prevalence) is low.

**Q5. Answer: C**
The Poisson distribution models the count of events occurring randomly in a fixed interval (time/space) at a known average rate. Website visitors per hour is a classic count-of-events scenario.

**Q6. Answer: B**
The CLT states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the original distribution of the data (provided variance is finite). It does not claim that all data is normal.

**Q7. Answer: B**
Zero covariance means there is no linear relationship between X and Y. However, they could still have a strong nonlinear relationship (e.g., Y = X²). Zero covariance does not imply independence.

**Q8. Answer: C**
The CDF F(x) = ∫ from −∞ to x of f(u) du, meaning the CDF is the cumulative integral of the PDF. Conversely, the PDF is the derivative of the CDF: f(x) = dF(x)/dx.

**Q9. Answer: A**
Standard error = σ/√n. If n is quadrupled (4n), the new standard error = σ/√(4n) = σ/(2√n), which is half the original. This is why doubling precision requires quadrupling the sample size.

**Q10. Answer: C**
The 68-95-99.7 rule states: approximately 68% of data falls within 1σ, 95% within 2σ, and 99.7% within 3σ of the mean in a normal distribution.

**Q11. Answer: C**
Naive Bayes assumes that features are conditionally independent given the class label, i.e., P(X₁, ..., Xₙ | Y) = Π P(Xᵢ | Y). This simplifying assumption makes computation tractable and often works surprisingly well in practice.

**Q12. Answer: C**
The exponential distribution models the time between consecutive events in a Poisson process. If events occur at a Poisson rate λ, the waiting time between events follows Exp(λ).

**Q13. Answer: B**
Var(X) = E[X²] − (E[X])², which is the expected value of the squared variable minus the square of the expected value. This is a commonly used alternative form of Var(X) = E[(X − E[X])²].

**Q14. Answer: B**
A sufficient statistic T(X) captures all the information in the data relevant to parameter θ. Once you know T(X), the raw data provides no additional information about θ. For example, the sample mean is sufficient for the mean of a normal distribution.

**Q15. Answer: D**
A correlation of −0.95 indicates a strong negative linear relationship — as one variable increases, the other decreases in a nearly linear fashion. Values near ±1 indicate strong linearity; values near 0 indicate weak or no linear relationship.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

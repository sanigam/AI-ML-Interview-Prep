# Multiple Choice Questions: Probability and Statistics Fundamentals

📺 **Video Lecture:** https://youtu.be/T2v-1SwoTZQ

Test your understanding of probability and statistics concepts essential for AI/ML interviews.

---

**Q1. Which of the following is NOT one of the three axioms of probability?**

A) P(A) + P(B) = 1 for any two events A and B  
B) For mutually exclusive events A and B, P(A ∪ B) = P(A) + P(B)  
C) P(Ω) = 1 where Ω is the sample space  
D) P(A) ≥ 0 for any event A

---

**Q2. Two events A and B are independent. Which statement is true?**

A) A and B cannot occur at the same time  
B) P(A ∩ B) = 0  
C) P(A | B) = P(A)  
D) P(A ∪ B) = P(A) + P(B)

---

**Q3. In Bayes' Theorem P(A|B) = P(B|A) × P(A) / P(B), what does P(A) represent?**

A) The likelihood  
B) The posterior probability  
C) The prior probability  
D) The evidence

---

**Q4. A medical test has a 99% sensitivity (true positive rate) and 95% specificity (true negative rate). If the disease prevalence is 1%, what is the approximate probability that a person who tests positive actually has the disease?**

A) About 50%  
B) About 95%  
C) About 17%  
D) About 99%

---

**Q5. Which distribution would you use to model the number of website visitors per hour?**

A) Normal distribution  
B) Exponential distribution  
C) Binomial distribution  
D) Poisson distribution

---

**Q6. The Central Limit Theorem (CLT) states that:**

A) The sample mean approaches a normal distribution as sample size increases, regardless of the original distribution  
B) The variance of a sample always equals the population variance  
C) All data in nature follows a normal distribution  
D) Larger samples always produce more accurate predictions

---

**Q7. If Cov(X, Y) = 0, which of the following is true?**

A) X and Y are mutually exclusive  
B) There is no linear relationship between X and Y  
C) X and Y have the same distribution  
D) X and Y are independent

---

**Q8. What is the relationship between the PDF and the CDF of a continuous random variable?**

A) The PDF and CDF are always equal  
B) The CDF is the integral of the PDF from −∞ to x  
C) The CDF is the derivative of the PDF  
D) The PDF is the integral of the CDF

---

**Q9. The standard error of the sample mean is σ/√n. What happens when you quadruple the sample size?**

A) The standard error is doubled  
B) The standard error is quartered  
C) The standard error is halved  
D) The standard error remains unchanged

---

**Q10. Which of the following correctly describes the 68-95-99.7 rule for a normal distribution?**

A) 95% of data falls within 1 standard deviation of the mean  
B) 68% of data falls within 2 standard deviations of the mean  
C) 68% of data falls within 3 standard deviations of the mean  
D) 99.7% of data falls within 3 standard deviations of the mean

---

**Q11. In the context of Naive Bayes classification, what key assumption is made about features?**

A) Features are mutually exclusive  
B) Features are conditionally independent given the class label  
C) All features follow a normal distribution  
D) Features must have zero correlation with each other

---

**Q12. Which distribution models the waiting time between events that follow a Poisson process?**

A) Binomial distribution  
B) Exponential distribution  
C) Uniform distribution  
D) Normal distribution

---

**Q13. Variance of a random variable X is defined as:**

A) E[X] − E[X²]  
B) (E[X])² − E[X²]  
C) E[X] × E[X]  
D) E[X²] − (E[X])²

---

**Q14. What does a sufficient statistic T(X) for a parameter θ guarantee?**

A) The raw data provides no additional information about θ beyond what T(X) provides  
B) The distribution of T(X) is always normal  
C) T(X) is always the sample mean  
D) T(X) always equals θ

---

**Q15. Correlation ρ between two variables ranges from −1 to +1. A value of ρ = −0.95 indicates:**

A) No relationship between the variables  
B) A strong positive linear relationship  
C) A strong negative linear relationship  
D) A weak negative linear relationship

---

## Answer Key

**Q1. Answer: A**
P(A) + P(B) = 1 is not an axiom. The additivity axiom applies only to mutually exclusive events: P(A ∪ B) = P(A) + P(B). The rule P(A) + P(Aᶜ) = 1 applies to an event and its complement, not any two arbitrary events.

**Q2. Answer: C**
Independence means P(A|B) = P(A) — knowing B occurred doesn't change the probability of A. Independence is often confused with mutual exclusivity (P(A ∩ B) = 0), which is a different concept entirely. Independent events can and do co-occur.

**Q3. Answer: C**
P(A) is the prior probability — our initial belief about A before observing evidence B. P(B|A) is the likelihood, P(A|B) is the posterior, and P(B) is the evidence (normalizing constant).

**Q4. Answer: C**
Using Bayes' Theorem: P(disease|positive) = (0.99 × 0.01) / ((0.99 × 0.01) + (0.05 × 0.99)) ≈ 0.0099 / 0.0594 ≈ 16.7%. This classic result shows that even accurate tests produce many false positives when the base rate (prevalence) is low.

**Q5. Answer: D**
The Poisson distribution models the count of events occurring randomly in a fixed interval (time/space) at a known average rate. Website visitors per hour is a classic count-of-events scenario.

**Q6. Answer: A**
The CLT states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the original distribution of the data (provided variance is finite). It does not claim that all data is normal.

**Q7. Answer: B**
Zero covariance means there is no linear relationship between X and Y. However, they could still have a strong nonlinear relationship (e.g., Y = X²). Zero covariance does not imply independence.

**Q8. Answer: B**
The CDF F(x) = ∫ from −∞ to x of f(u) du, meaning the CDF is the cumulative integral of the PDF. Conversely, the PDF is the derivative of the CDF: f(x) = dF(x)/dx.

**Q9. Answer: C**
Standard error = σ/√n. If n is quadrupled (4n), the new standard error = σ/√(4n) = σ/(2√n), which is half the original. This is why doubling precision requires quadrupling the sample size.

**Q10. Answer: D**
The 68-95-99.7 rule states: approximately 68% of data falls within 1σ, 95% within 2σ, and 99.7% within 3σ of the mean in a normal distribution.

**Q11. Answer: B**
Naive Bayes assumes that features are conditionally independent given the class label, i.e., P(X₁, ..., Xₙ | Y) = Π P(Xᵢ | Y). This simplifying assumption makes computation tractable and often works surprisingly well in practice.

**Q12. Answer: B**
The exponential distribution models the time between consecutive events in a Poisson process. If events occur at a Poisson rate λ, the waiting time between events follows Exp(λ).

**Q13. Answer: D**
Var(X) = E[X²] − (E[X])², which is the expected value of the squared variable minus the square of the expected value. This is a commonly used alternative form of Var(X) = E[(X − E[X])²].

**Q14. Answer: A**
A sufficient statistic T(X) captures all the information in the data relevant to parameter θ. Once you know T(X), the raw data provides no additional information about θ. For example, the sample mean is sufficient for the mean of a normal distribution.

**Q15. Answer: C**
A correlation of −0.95 indicates a strong negative linear relationship — as one variable increases, the other decreases in a nearly linear fashion. Values near ±1 indicate strong linearity; values near 0 indicate weak or no linear relationship.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

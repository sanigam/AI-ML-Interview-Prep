# Multiple Choice Questions: Bayesian Statistics

📺 **Video Lecture:** https://youtu.be/YpckH7F5vj0


Test your understanding of Bayesian inference, priors, posteriors, and computational methods for AI/ML interviews.

---

**Q1. In Bayes' Theorem, P(θ | data) ∝ P(data | θ) × P(θ), the term P(θ) represents:**

A) The likelihood  
B) The marginal likelihood  
C) The posterior distribution  
D) The prior distribution

---

**Q2. A conjugate prior is one where:**

A) The prior eliminates the need for data  
B) The prior is always uniform  
C) The prior and posterior belong to the same family of distributions  
D) The posterior is always a normal distribution

---

**Q3. In the Beta-Binomial model, if the prior is Beta(2, 3) and we observe 4 successes in 10 trials, the posterior is:**

A) Beta(4, 6)  
B) Beta(6, 7)  
C) Beta(6, 9)  
D) Beta(2, 3)

---

**Q4. MAP estimation differs from MLE in that MAP:**

A) Incorporates a prior distribution over the parameters  
B) Always produces larger parameter estimates than MLE  
C) Does not use the likelihood function  
D) Requires MCMC sampling

---

**Q5. A Gaussian prior on regression weights in MAP estimation is equivalent to:**

A) No regularization  
B) L2 (Ridge) regularization  
C) L1 (Lasso) regularization  
D) Dropout regularization

---

**Q6. A Bayesian credible interval differs from a frequentist confidence interval because:**

A) It directly gives the probability that the parameter lies within the interval given the observed data  
B) It is only valid for large samples  
C) It does not require any assumptions  
D) It is always narrower

---

**Q7. MCMC (Markov Chain Monte Carlo) is used in Bayesian inference primarily to:**

A) Optimize the loss function in neural networks  
B) Compute the exact posterior distribution analytically  
C) Select the best model among candidates  
D) Generate samples from the posterior when it cannot be computed in closed form

---

**Q8. The Metropolis-Hastings algorithm accepts a proposed new parameter value θ' based on:**

A) Whether it increases the posterior probability  
B) A fixed probability of 0.5  
C) Whether it minimizes the squared error  
D) An acceptance ratio involving the posterior and proposal distributions, with randomization

---

**Q9. When would you choose variational inference over MCMC?**

A) When the posterior has only one parameter  
B) When the dataset is very small  
C) When the dataset is large and speed is important  
D) When exact posterior samples are required

---

**Q10. In hierarchical Bayesian models, partial pooling refers to:**

A) Using only the largest group's data for estimation  
B) Ignoring group-level differences entirely  
C) Shrinking group-level estimates toward the overall mean, borrowing strength across groups  
D) Estimating each group independently with no shared information

---

**Q11. Empirical Bayes differs from full Bayesian analysis because it:**

A) Always produces unbiased estimates  
B) Estimates hyperparameters from the data rather than placing priors on them  
C) Requires MCMC for computation  
D) Does not use Bayes' Theorem at all

---

**Q12. A Bayes factor of 15 comparing Model 1 to Model 2 means:**

A) Model 2 is 15 times more likely than Model 1  
B) The p-value is 1/15  
C) The data is 15 times more probable under Model 1 than Model 2  
D) Model 1 has 15 more parameters than Model 2

---

**Q13. The posterior predictive distribution P(y_new | y_obs) accounts for uncertainty by:**

A) Ignoring parameter uncertainty and using the prior  
B) Selecting the parameter with highest likelihood  
C) Averaging predictions over all plausible parameter values weighted by the posterior  
D) Using only the MAP estimate for prediction

---

**Q14. As the amount of observed data increases, the Bayesian posterior typically:**

A) Stays constant regardless of data  
B) Becomes more diffuse and uncertain  
C) Becomes dominated by the likelihood, with diminishing influence from the prior  
D) Becomes identical to the prior

---

**Q15. Which statement about uninformative priors is correct?**

A) They guarantee the posterior equals the likelihood  
B) They attempt to express minimal prior knowledge, letting the data dominate inference  
C) They are always uniform distributions  
D) They make Bayesian and frequentist results identical in all cases

---

## Answer Key

**Q1. Answer: D**
P(θ) is the prior distribution — it represents our beliefs about the parameter before observing any data. The likelihood is P(data | θ), the posterior is P(θ | data), and the marginal likelihood is P(data).

**Q2. Answer: C**
A conjugate prior ensures the posterior belongs to the same distributional family as the prior (e.g., Beta prior + Binomial likelihood = Beta posterior), enabling closed-form computation without MCMC.

**Q3. Answer: C**
For Beta-Binomial conjugacy: posterior = Beta(α + successes, β + failures) = Beta(2 + 4, 3 + 6) = Beta(6, 9). The 10 trials with 4 successes means 6 failures.

**Q4. Answer: A**
MAP = argmax P(θ | data) = argmax [P(data | θ) × P(θ)]. Unlike MLE which maximizes only the likelihood, MAP includes the prior P(θ), which acts as a regularizer pulling estimates toward prior beliefs.

**Q5. Answer: B**
A Gaussian (Normal) prior on weights corresponds to L2/Ridge regularization, which penalizes the sum of squared weights. A Laplace prior corresponds to L1/Lasso regularization.

**Q6. Answer: A**
A 95% Bayesian credible interval means P(L ≤ θ ≤ U | data) = 0.95 — there is a 95% posterior probability the parameter is in this range. Frequentist CIs have a different interpretation about repeated sampling procedures.

**Q7. Answer: D**
MCMC generates samples from the posterior distribution when it cannot be computed analytically. The samples approximate the posterior and can be used to estimate means, variances, and credible intervals.

**Q8. Answer: D**
Metropolis-Hastings computes an acceptance ratio α = min(1, P(θ'|data) × q(θ|θ') / [P(θ|data) × q(θ'|θ)]) and accepts with probability α. This randomized acceptance ensures the chain converges to the target posterior.

**Q9. Answer: C**
Variational inference converts sampling into optimization, making it faster than MCMC for large datasets. It sacrifices some accuracy for speed, which is a worthwhile tradeoff in large-scale applications.

**Q10. Answer: C**
Partial pooling is a hallmark of hierarchical models: each group's estimate is a weighted combination of its own data and the overall group-level mean, effectively borrowing strength from other groups. This improves estimates for groups with limited data.

**Q11. Answer: B**
Empirical Bayes estimates hyperparameters from the marginal distribution of the data rather than placing priors on them. It's a practical hybrid of Bayesian and frequentist approaches.

**Q12. Answer: C**
A Bayes factor BF₁₂ = 15 means the observed data is 15 times more probable under Model 1 than under Model 2. It quantifies relative evidence for one model over another.

**Q13. Answer: C**
The posterior predictive integrates over all parameter values: P(y_new | y_obs) = ∫ P(y_new | θ) P(θ | y_obs) dθ, properly accounting for parameter uncertainty rather than relying on a single point estimate.

**Q14. Answer: C**
With increasing data, the likelihood dominates and the prior's influence diminishes. The posterior concentrates around the true parameter value — this is the Bayesian consistency property.

**Q15. Answer: B**
Uninformative (or weakly informative) priors attempt to express minimal prior knowledge so the data drives the inference. Truly uninformative priors don't technically exist, and they are not always uniform.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

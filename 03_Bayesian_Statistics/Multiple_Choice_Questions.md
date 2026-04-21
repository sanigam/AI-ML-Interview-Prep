# Multiple Choice Questions: Bayesian Statistics

Test your understanding of Bayesian inference, priors, posteriors, and computational methods for AI/ML interviews.

---

**Q1. In Bayes' Theorem, P(θ | data) ∝ P(data | θ) × P(θ), the term P(θ) represents:**

A) The likelihood
B) The posterior distribution
C) The prior distribution
D) The marginal likelihood

---

**Q2. A conjugate prior is one where:**

A) The prior and posterior belong to the same family of distributions
B) The prior is always uniform
C) The posterior is always a normal distribution
D) The prior eliminates the need for data

---

**Q3. In the Beta-Binomial model, if the prior is Beta(2, 3) and we observe 4 successes in 10 trials, the posterior is:**

A) Beta(6, 9)
B) Beta(4, 6)
C) Beta(2, 3)
D) Beta(6, 7)

---

**Q4. MAP estimation differs from MLE in that MAP:**

A) Does not use the likelihood function
B) Incorporates a prior distribution over the parameters
C) Always produces larger parameter estimates than MLE
D) Requires MCMC sampling

---

**Q5. A Gaussian prior on regression weights in MAP estimation is equivalent to:**

A) L1 (Lasso) regularization
B) L2 (Ridge) regularization
C) Dropout regularization
D) No regularization

---

**Q6. A Bayesian credible interval differs from a frequentist confidence interval because:**

A) It is always narrower
B) It directly gives the probability that the parameter lies within the interval given the observed data
C) It does not require any assumptions
D) It is only valid for large samples

---

**Q7. MCMC (Markov Chain Monte Carlo) is used in Bayesian inference primarily to:**

A) Compute the exact posterior distribution analytically
B) Generate samples from the posterior when it cannot be computed in closed form
C) Optimize the loss function in neural networks
D) Select the best model among candidates

---

**Q8. The Metropolis-Hastings algorithm accepts a proposed new parameter value θ' based on:**

A) Whether it increases the posterior probability
B) An acceptance ratio involving the posterior and proposal distributions, with randomization
C) Whether it minimizes the squared error
D) A fixed probability of 0.5

---

**Q9. When would you choose variational inference over MCMC?**

A) When exact posterior samples are required
B) When the dataset is very small
C) When the dataset is large and speed is important
D) When the posterior has only one parameter

---

**Q10. In hierarchical Bayesian models, partial pooling refers to:**

A) Ignoring group-level differences entirely
B) Estimating each group independently with no shared information
C) Shrinking group-level estimates toward the overall mean, borrowing strength across groups
D) Using only the largest group's data for estimation

---

**Q11. Empirical Bayes differs from full Bayesian analysis because it:**

A) Does not use Bayes' Theorem at all
B) Estimates hyperparameters from the data rather than placing priors on them
C) Always produces unbiased estimates
D) Requires MCMC for computation

---

**Q12. A Bayes factor of 15 comparing Model 1 to Model 2 means:**

A) Model 2 is 15 times more likely than Model 1
B) The data is 15 times more probable under Model 1 than Model 2
C) Model 1 has 15 more parameters than Model 2
D) The p-value is 1/15

---

**Q13. The posterior predictive distribution P(y_new | y_obs) accounts for uncertainty by:**

A) Using only the MAP estimate for prediction
B) Averaging predictions over all plausible parameter values weighted by the posterior
C) Ignoring parameter uncertainty and using the prior
D) Selecting the parameter with highest likelihood

---

**Q14. As the amount of observed data increases, the Bayesian posterior typically:**

A) Becomes identical to the prior
B) Becomes dominated by the likelihood, with diminishing influence from the prior
C) Becomes more diffuse and uncertain
D) Stays constant regardless of data

---

**Q15. Which statement about uninformative priors is correct?**

A) They guarantee the posterior equals the likelihood
B) They attempt to express minimal prior knowledge, letting the data dominate inference
C) They are always uniform distributions
D) They make Bayesian and frequentist results identical in all cases

---

## Answer Key

**Q1. Answer: C**
P(θ) is the prior distribution — it represents our beliefs about the parameter before observing any data. The likelihood is P(data | θ), the posterior is P(θ | data), and the marginal likelihood is P(data).

**Q2. Answer: A**
A conjugate prior ensures the posterior belongs to the same distributional family as the prior (e.g., Beta prior + Binomial likelihood = Beta posterior), enabling closed-form computation without MCMC.

**Q3. Answer: A**
For Beta-Binomial conjugacy: posterior = Beta(α + successes, β + failures) = Beta(2 + 4, 3 + 6) = Beta(6, 9). The 10 trials with 4 successes means 6 failures.

**Q4. Answer: B**
MAP = argmax P(θ | data) = argmax [P(data | θ) × P(θ)]. Unlike MLE which maximizes only the likelihood, MAP includes the prior P(θ), which acts as a regularizer pulling estimates toward prior beliefs.

**Q5. Answer: B**
A Gaussian (Normal) prior on weights corresponds to L2/Ridge regularization, which penalizes the sum of squared weights. A Laplace prior corresponds to L1/Lasso regularization.

**Q6. Answer: B**
A 95% Bayesian credible interval means P(L ≤ θ ≤ U | data) = 0.95 — there is a 95% posterior probability the parameter is in this range. Frequentist CIs have a different interpretation about repeated sampling procedures.

**Q7. Answer: B**
MCMC generates samples from the posterior distribution when it cannot be computed analytically. The samples approximate the posterior and can be used to estimate means, variances, and credible intervals.

**Q8. Answer: B**
Metropolis-Hastings computes an acceptance ratio α = min(1, P(θ'|data) × q(θ|θ') / [P(θ|data) × q(θ'|θ)]) and accepts with probability α. This randomized acceptance ensures the chain converges to the target posterior.

**Q9. Answer: C**
Variational inference converts sampling into optimization, making it faster than MCMC for large datasets. It sacrifices some accuracy for speed, which is a worthwhile tradeoff in large-scale applications.

**Q10. Answer: C**
Partial pooling is a hallmark of hierarchical models: each group's estimate is a weighted combination of its own data and the overall group-level mean, effectively borrowing strength from other groups. This improves estimates for groups with limited data.

**Q11. Answer: B**
Empirical Bayes estimates hyperparameters from the marginal distribution of the data rather than placing priors on them. It's a practical hybrid of Bayesian and frequentist approaches.

**Q12. Answer: B**
A Bayes factor BF₁₂ = 15 means the observed data is 15 times more probable under Model 1 than under Model 2. It quantifies relative evidence for one model over another.

**Q13. Answer: B**
The posterior predictive integrates over all parameter values: P(y_new | y_obs) = ∫ P(y_new | θ) P(θ | y_obs) dθ, properly accounting for parameter uncertainty rather than relying on a single point estimate.

**Q14. Answer: B**
With increasing data, the likelihood dominates and the prior's influence diminishes. The posterior concentrates around the true parameter value — this is the Bayesian consistency property.

**Q15. Answer: B**
Uninformative (or weakly informative) priors attempt to express minimal prior knowledge so the data drives the inference. Truly uninformative priors don't technically exist, and they are not always uniform.

# Multiple Choice Questions: Causal Inference and Experimentation

📺 **Video Lecture:** https://youtu.be/J4_3yDvDvL0


## Question 1
A retail company observes that customers who download their mobile app spend 40% more than those who don't. The company executives conclude the app drives spending and plan to invest heavily in app promotion. What is the primary causal inference problem with this conclusion?

A) The company failed to use randomization to establish causation  
B) The correlation observed could be confounded by customer engagement—more engaged customers are both more likely to download the app and more likely to spend money regardless  
C) The sample size is too small to detect a true effect  
D) The company should have used instrumental variables instead of observational data

**Correct Answer: B**

---

## Question 2
In the potential outcomes framework, what does the "fundamental problem of causal inference" refer to?

A) The inability to collect enough data for statistical significance  
B) The impossibility of observing both potential outcomes Y(1) and Y(0) for the same unit simultaneously  
C) The difficulty of designing experiments in real-world business settings  
D) The challenge of controlling for all possible confounding variables

**Correct Answer: B**

---

## Question 3
A company runs an A/B test comparing a new checkout flow (treatment) to the current one (control). The treatment group shows a 5% higher conversion rate. In randomized experiments, this difference directly estimates which causal quantity?

A) Average Treatment Effect on the Treated (ATT)  
B) Average Treatment Effect (ATE)  
C) Heterogeneous Treatment Effect (HTE)  
D) Average Treatment Effect on the Control (ATC)

**Correct Answer: B**

---

## Question 4
Which of the following is an example of a confounding variable in a study of whether a new productivity software increases employee output?

A) The time employees have used the software (mediator)  
B) Employee motivation level, which affects both whether they adopt the software and their output  
C) The operating system employees use (independent of both treatment and outcome)  
D) Total hours worked, which is directly caused by the software

**Correct Answer: B**

---

## Question 5
Using a directed acyclic graph (DAG), you identify that Variable X affects treatment T, outcome Y, and also affects a mediator M, where T → M → Y. To estimate the direct causal effect of T on Y (not through M), what should you do?

A) Condition on M to block the indirect path  
B) Condition on X to control for confounding  
C) Don't condition on any variables; just compare T and Y  
D) Condition on both X and M

**Correct Answer: B**

---

## Question 6
In propensity score matching, what is the primary assumption required for unbiased causal effect estimation?

A) Common support (overlap) so that no propensity score groups are excluded  
B) The propensity score model must have higher predictive accuracy than the outcome model  
C) Unconfoundedness—all variables that confound treatment and outcome are observed and included  
D) All units within propensity score quintiles are perfectly homogeneous

**Correct Answer: C**

---

## Question 7
Compared to propensity score matching, inverse probability weighting (IPW) has which advantage?

A) It doesn't require the unconfoundedness assumption  
B) It uses all observed data instead of discarding unmatched units, making it more statistically efficient  
C) It eliminates the need for checking covariate balance  
D) It automatically handles unmeasured confounding

**Correct Answer: B**

---

## Question 8
When would you use instrumental variables (IV) instead of propensity score methods?

A) When you have a large sample size (n > 10,000)  
B) When you believe there are unmeasured confounders affecting treatment and outcome  
C) When treatment is randomly assigned in the data  
D) When you want to estimate heterogeneous treatment effects by subgroup

**Correct Answer: B**

---

## Question 9
In a difference-in-differences (DiD) analysis comparing treatment stores to control stores over time, the "parallel trends" assumption means:

A) Treatment and control stores must have identical sales levels before the treatment begins  
B) Absent treatment, the treatment group would follow the same trajectory as the control group  
C) Treatment and control stores must be in the same geographic region  
D) The treatment effect must be constant for all stores

**Correct Answer: B**

---

## Question 10
In regression discontinuity design (RDD), why is being just above vs. just below a treatment threshold (e.g., test score 80.5 vs. 79.5) nearly a randomized comparison?

A) Because the running variable is truly random near the threshold  
B) Because units on either side of the threshold are expected to be similar in all characteristics except treatment assignment, assuming continuity of the outcome function  
C) Because the RDD method automatically removes all confounding variables  
D) Because larger sample sizes near the threshold ensure statistical power

**Correct Answer: B**

---

## Question 11
Simpson's Paradox illustrates which causal inference principle?

A) That larger sample sizes always lead to more accurate causal estimates  
B) That associations can reverse direction depending on which variables you condition on, and proper conditioning requires causal reasoning  
C) That randomization is always inferior to observational methods  
D) That correlation never implies causation under any circumstances

**Correct Answer: B**

---

## Question 12
What is the key difference between P(Y | T=1) and P(Y | do(T=1)) in Pearl's do-calculus?

A) P(Y | T=1) is causal while P(Y | do(T=1)) includes bias from selection effects  
B) P(Y | T=1) includes selection bias from confounding while P(Y | do(T=1)) isolates the causal effect of forcibly setting T=1  
C) They are mathematically identical; the notation is just stylistic  
D) P(Y | T=1) requires a randomized experiment while P(Y | do(T=1)) works with observational data only

**Correct Answer: B**

---

## Question 13
In uplift modeling for targeted marketing, why is it valuable to distinguish between customers with high, zero, and negative treatment effects?

A) Because it allows marketing teams to target those most likely to respond (high uplift) and avoid those likely to be harmed (negative uplift), improving ROI  
B) Because it eliminates the need for any control group in experiments  
C) Because it automatically increases the sample size in experiments  
D) Because it makes the unconfoundedness assumption unnecessary

**Correct Answer: A**

---

## Question 14
When an A/B test is "peeked" at (results checked before the pre-specified sample size is reached, followed by early stopping if significant), what happens to the false positive rate?

A) It stays at the nominal α = 0.05  
B) It decreases below α = 0.05 due to additional data analysis  
C) It increases above α = 0.05 due to multiple testing implicit in sequential checking  
D) It becomes impossible to calculate

**Correct Answer: C**

---

## Question 15
A social media platform runs an A/B test but finds that treated users in a friend group influence untreated users in the same group, biasing the results. Which pitfall does this exemplify and how should it be addressed?

A) Novelty effects; solve by running the test longer  
B) Sample ratio mismatch; solve by checking if randomization broke  
C) Network effects / interference; solve by randomizing at the cluster level (entire friend group) rather than individual level  
D) Peeking; solve by pre-registering the analysis plan

**Correct Answer: C**

---

# Answer Key

**Q1: B** - The fundamental issue is that more engaged customers self-select into downloading the app AND naturally spend more. This is confounding, not causal effect of the app itself. Randomization (A) would solve it but wasn't used. This illustrates correlation vs. causation.

**Q2: B** - The core challenge in causal inference is that we observe only one potential outcome per unit (either Y(1) if treated or Y(0) if untreated), never both. This is what makes counterfactual inference necessary and why randomization is so powerful—it allows us to estimate population-level counterfactuals through group comparisons.

**Q3: B** - Randomized experiments directly estimate ATE because random assignment makes treatment independent of potential outcomes. The simple difference in means between treatment and control groups is an unbiased estimate of E[Y(1)] - E[Y(0)]. ATT (A) would apply in observational studies; HTE (C) requires individual-level predictions.

**Q4: B** - A confounder must affect both treatment (whether employees adopt) and outcome (their output). Motivation satisfies this—motivated employees are more likely to try new tools AND naturally produce more. Mediators (A) are downstream of treatment, not confounders. The OS (C) is independent. Hours worked (D) is caused by treatment, making it a mediator or outcome.

**Q5: B** - To estimate the direct effect of T on Y (not through M), block the indirect path T → M → Y by conditioning on X (the confounder) but NOT on M (that would block the causal pathway too). Conditioning on M (A, D) would remove part of the treatment's true effect. Just comparing T and Y (C) leaves X's confounding unblocked.

**Q6: C** - Propensity score matching requires unconfoundedness: all confounders are observed and included in the propensity score model. Without this, matching fails to control hidden confounders. Common support (A) is also necessary but not sufficient. Predictive accuracy (B) is less important than covariate balance. Perfect homogeneity (D) is impossible.

**Q7: B** - IPW uses all data points by reweighting rather than discarding unmatched units as matching does. This improves statistical efficiency and power. It does NOT relax unconfoundedness (A), automatically handle unmeasured confounding (D), or eliminate balance checks (C). All require the same core assumptions.

**Q8: B** - IV handles unmeasured confounding by using an exogenous source of variation (the instrument) to extract treatment effect, unlike propensity score methods which require measured confounders. Large sample size (A) alone doesn't necessitate IV. Random assignment (C) makes IV unnecessary. HTE estimation (D) is orthogonal to the IV vs. propensity score choice.

**Q9: B** - Parallel trends assumes that absent treatment, both groups would follow the same trajectory over time. This allows DiD to isolate treatment effect by comparing within-group changes. Identical pre-treatment levels (A) are not required—different levels are fine if trends are parallel. Geographic proximity (C) is not the assumption. Time-varying effects (D) are possible in extended DiD.

**Q10: B** - Units just above and below the threshold are nearly identical in all confounders due to continuity of the outcome relationship, creating a natural comparison group. The running variable itself (A) isn't random, but the threshold exploits the cutoff structure. RDD doesn't automatically remove confounding (C); it exploits local continuity. Sample size (D) affects power, not the validity of comparison.

**Q11: B** - Simpson's Paradox demonstrates that conditioning (or not conditioning) on variables matters for causal inference and can reverse conclusions. This shows why proper causal reasoning—identifying confounders via DAGs and conditioning appropriately—is essential. It doesn't invalidate causal inference (A, C, D); it highlights the importance of doing it correctly.

**Q12: B** - P(Y | T=1) is observational/associational and includes confounding bias; P(Y | do(T=1)) represents the causal effect of an intervention and removes selection mechanisms. They differ precisely because conditioning includes the selection bias in how treatment was assigned, while do(·) breaks that mechanism.

**Q13: A** - Uplift modeling is valuable because it targets heterogeneous effects: spend marketing on high-uplift customers (who respond), save on zero-uplift customers (who'd buy anyway), and avoid negative-uplift customers (who'd churn). This improves ROI by precision targeting. It doesn't eliminate control groups (B), increase sample size (C), or relax assumptions (D).

**Q14: C** - Peeking inflates false positive rate because sequential checking of significance without correction inflates the probability of observing statistical significance under the null. The nominal α = 0.05 no longer holds. Fix via pre-specifying sample size, pre-registering analyses, or using sequential testing methods that correct for repeated looks.

**Q15: C** - Network effects (interference) occur when treated units affect untreated units through spillover. This violates the assumption that units are independent, biasing estimates. Solution: randomize at the cluster level (entire friend group) so spillover happens within treatment groups. Novelty effects (A) are addressed by longer run duration, not cluster randomization. Sample ratio (B) is a different issue.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

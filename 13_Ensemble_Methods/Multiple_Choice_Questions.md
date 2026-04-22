# Multiple Choice Questions: Ensemble Methods

Test your understanding of bagging, boosting, stacking, and ensemble techniques.

---

**Q1. The fundamental idea behind ensemble methods is:**

A) Using a single strong model
B) Combining multiple models to produce better predictions than any individual model
C) Reducing the training data size
D) Using only linear models

---

**Q2. Bagging (Bootstrap Aggregating) primarily reduces:**

A) Bias
B) Variance
C) Both equally
D) Neither

---

**Q3. Boosting primarily reduces:**

A) Variance only
B) Bias by iteratively focusing on hard-to-classify examples
C) The number of features
D) The training set size

---

**Q4. In AdaBoost, misclassified samples from the previous iteration:**

A) Are removed from the dataset
B) Receive higher weights so the next classifier focuses on them
C) Are always correctly classified in the next round
D) Have no effect on subsequent models

---

**Q5. Stacking (stacked generalization) differs from bagging and boosting by:**

A) Using only one base model
B) Training a meta-learner on the predictions of diverse base models
C) Being identical to random forests
D) Not using any base models

---

**Q6. For an ensemble to be effective, the individual models should be:**

A) Identical to each other
B) Diverse (making different errors) while still being reasonably accurate
C) As complex as possible regardless of accuracy
D) Trained on the same exact data in the same way

---

**Q7. Voting ensembles use which strategy for classification?**

A) Only the first model's prediction
B) Majority vote (hard voting) or averaged probabilities (soft voting) across models
C) Random selection of one model's prediction
D) The prediction with highest confidence from any model

---

**Q8. Random Forest is an example of:**

A) Boosting
B) Bagging with additional random feature selection
C) Stacking
D) A single decision tree

---

**Q9. Gradient Boosting is an example of:**

A) Bagging
B) Sequential ensemble where each model corrects residual errors of the previous ensemble
C) Stacking
D) Random sampling without replacement

---

**Q10. Why does boosting have a higher risk of overfitting than bagging?**

A) Boosting uses fewer models
B) Boosting sequentially fits residuals, which can include noise, leading to overfitting with too many iterations
C) Boosting never uses regularization
D) Boosting uses random subsets of data

---

**Q11. The learning rate (shrinkage) in gradient boosting:**

A) Controls the step size in gradient descent for each base model, requiring more trees but improving generalization
B) Determines the depth of each tree
C) Sets the number of features
D) Has no effect on performance

---

**Q12. In a weighted majority vote, models with better performance should receive:**

A) Equal weights to all other models
B) Higher weights in the final prediction
C) Zero weight
D) Negative weights

---

**Q13. Out-of-fold predictions in stacking are used to:**

A) Speed up training
B) Generate training data for the meta-learner without data leakage
C) Remove outliers
D) Select features

---

**Q14. Which statement about ensemble size is generally true?**

A) More models always significantly improve performance
B) Performance improves with more models but with diminishing returns after a certain point
C) Using exactly 3 models is always optimal
D) Fewer models always produce better results

---

**Q15. Blending differs from stacking in that blending:**

A) Uses cross-validation to generate meta-features
B) Uses a simple holdout set for meta-feature generation instead of cross-validation
C) Does not use a meta-learner
D) Only works with neural networks

---

## Answer Key

**Q1. Answer: B**
Ensembles exploit the "wisdom of crowds" — combining diverse models reduces individual model weaknesses. The combination typically outperforms any single model.

**Q2. Answer: B**
Bagging trains models on bootstrap samples and averages predictions. This averaging reduces variance (sensitivity to specific training data) while keeping bias roughly constant.

**Q3. Answer: B**
Boosting sequentially trains models to correct predecessors' errors, effectively reducing bias. Each new model targets the residual error, improving the ensemble's fit to the true function.

**Q4. Answer: B**
AdaBoost increases the weight of misclassified samples so the next weak learner focuses on the difficult cases. Over iterations, the ensemble masters progressively harder examples.

**Q5. Answer: B**
Stacking uses base model predictions as input features for a meta-learner (e.g., logistic regression), learning the optimal way to combine diverse models rather than simple averaging.

**Q6. Answer: B**
Diversity is key — if all models make the same errors, combining them doesn't help. Models should be accurate individually but make different mistakes (uncorrelated errors).

**Q7. Answer: B**
Hard voting takes the majority class prediction. Soft voting averages predicted probabilities and selects the class with highest average probability, often performing better.

**Q8. Answer: B**
Random Forest = bagging (bootstrap samples) + random feature subsets at each split. The double randomness creates diversity among trees, making the ensemble effective.

**Q9. Answer: B**
Gradient Boosting fits each new tree to the negative gradient of the loss (residuals for MSE). Trees are added sequentially, each correcting the ensemble's remaining errors.

**Q10. Answer: B**
Boosting's sequential nature means later trees fit noise in the residuals if training continues too long. Regularization (learning rate, early stopping, tree constraints) mitigates this.

**Q11. Answer: A**
The learning rate (0 < η ≤ 1) shrinks each tree's contribution: F(x) += η × tree(x). Smaller η requires more trees but produces smoother, better-generalizing ensembles.

**Q12. Answer: B**
In weighted voting, better-performing models get higher weights so their predictions contribute more. This is more principled than equal-weight voting when model quality varies.

**Q13. Answer: B**
Out-of-fold predictions (from k-fold CV) give unbiased meta-features for the second-level model. Using in-sample predictions would leak information and cause the meta-learner to overfit.

**Q14. Answer: B**
Performance typically improves rapidly with initial models, then plateaus. For bagging, variance reduction follows 1/n law. Beyond a point, computational cost outweighs marginal gains.

**Q15. Answer: B**
Blending uses a simple train/validation split to generate meta-features (simpler, faster), while stacking uses full cross-validation (more data-efficient, less biased). Both train a meta-learner.

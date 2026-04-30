# Multiple Choice Questions: Tree-Based Models

📺 **Video Lecture:** https://youtu.be/tmCPYgbE-vk


Test your understanding of decision trees, random forests, and gradient boosting.

---

**Q1. A decision tree splits data based on:**

A) The mean of all features  
B) Random chance  
C) The alphabetical order of feature names  
D) The feature and threshold that best separates the target classes (maximizes information gain or minimizes impurity)

---

**Q2. Gini impurity of a node with 50% class A and 50% class B equals:**

A) 0.5  
B) 1.0  
C) 0.0  
D) 0.25

---

**Q3. A decision tree with no pruning or depth limits will likely:**

A) Underfit the training data  
B) Overfit the training data by memorizing it  
C) Always achieve optimal test accuracy  
D) Have high bias and low variance

---

**Q4. Random Forest reduces overfitting compared to a single decision tree by:**

A) Using a shallower tree  
B) Training multiple trees on bootstrapped samples and random feature subsets, then averaging predictions  
C) Removing all outliers from the data  
D) Using only one feature

---

**Q5. In Random Forest, the "random" aspect refers to:**

A) Random initialization of weights  
B) Random selection of the target variable  
C) Random sampling of training data (bagging) AND random selection of features at each split  
D) Random depth for each tree

---

**Q6. Gradient Boosting builds trees sequentially where each new tree:**

A) Fits the residuals (errors) of the previous ensemble  
B) Uses a completely different dataset  
C) Has more features than the previous tree  
D) Is identical to the previous tree

---

**Q7. XGBoost improves on basic gradient boosting by incorporating:**

A) Neural network layers  
B) Regularization, efficient handling of sparse data, and parallel computation  
C) Only deeper trees  
D) Only random feature sampling

---

**Q8. The max_depth hyperparameter in a decision tree controls:**

A) The maximum number of features used  
B) The maximum number of training samples  
C) The maximum number of trees in an ensemble  
D) The maximum levels of splits from root to leaf

---

**Q9. Feature importance in a random forest is typically measured by:**

A) The number of missing values in the feature  
B) The order in which features appear alphabetically  
C) The total reduction in impurity (Gini or entropy) contributed by that feature across all trees  
D) The correlation of the feature with the target

---

**Q10. Bagging (Bootstrap Aggregating) reduces:**

A) Both bias and variance equally  
B) The number of features  
C) Variance of the model by averaging multiple models trained on different subsets  
D) Bias of the model

---

**Q11. Out-of-bag (OOB) error in Random Forest is:**

A) An estimate of test error using samples not included in each tree's bootstrap sample  
B) The error after removing outliers  
C) The error on the training set  
D) Always lower than cross-validation error

---

**Q12. Entropy of a node with 100% of one class equals:**

A) 0.5  
B) Infinity  
C) 0.0 (pure node — no uncertainty)  
D) 1.0

---

**Q13. LightGBM differs from XGBoost primarily by:**

A) Using leaf-wise tree growth instead of level-wise, making it faster on large datasets  
B) Building trees in parallel  
C) Only working with categorical features  
D) Not using gradient boosting

---

**Q14. In gradient boosting, a smaller learning rate (shrinkage) typically requires:**

A) Larger tree depth  
B) More trees to achieve the same performance, but often generalizes better  
C) No regularization  
D) Fewer trees for good performance

---

**Q15. Decision trees are invariant to feature scaling because:**

A) They always normalize features internally  
B) Splits are based on threshold comparisons (greater/less than), not on magnitudes  
C) They use gradient descent  
D) They can only handle categorical features

---

## Answer Key

**Q1. Answer: D**
Decision trees select the feature and split point that maximizes information gain (or minimizes Gini impurity/entropy) at each node, creating the most homogeneous child nodes.

**Q2. Answer: A**
Gini impurity = 1 − Σpᵢ² = 1 − (0.5² + 0.5²) = 1 − 0.5 = 0.5. This is the maximum impurity for binary classification, representing complete uncertainty.

**Q3. Answer: B**
An unrestricted tree grows until each leaf is pure, essentially memorizing the training data. This gives zero training error but poor generalization (high variance, low bias).

**Q4. Answer: B**
Random Forest combines bagging (bootstrap sampling) with random feature selection at each split. Averaging many decorrelated trees reduces variance while maintaining low bias.

**Q5. Answer: C**
Random Forest uses two sources of randomness: (1) bootstrap sampling of training data for each tree, and (2) random subset of features considered at each split point.

**Q6. Answer: A**
Each new tree in gradient boosting fits the negative gradient of the loss (residuals for MSE). The ensemble progressively corrects the mistakes of previous trees.

**Q7. Answer: B**
XGBoost adds L1/L2 regularization on leaf weights, efficient sparse data handling, column subsampling, and system optimizations like parallel tree construction and cache-aware computation.

**Q8. Answer: D**
max_depth limits how deep a tree can grow. Deeper trees capture more complex patterns but risk overfitting. Common values range from 3-10 for boosted ensembles.

**Q9. Answer: C**
Feature importance sums the impurity reduction from all splits using that feature across all trees. Features used frequently at high levels with large impurity reductions are deemed most important.

**Q10. Answer: C**
Bagging reduces variance by averaging predictions from multiple models trained on different bootstrap samples. Each model has similar bias, but their errors partially cancel when averaged.

**Q11. Answer: A**
Each tree in RF is trained on ~63% of samples (bootstrap). The OOB error uses the remaining ~37% as a validation set for each tree, providing a free estimate of generalization error.

**Q12. Answer: C**
Entropy = −Σpᵢ log₂(pᵢ). For a pure node (100% one class), entropy = −1×log₂(1) = 0. Zero entropy means no uncertainty — the node perfectly classifies its samples.

**Q13. Answer: A**
LightGBM grows trees leaf-wise (choosing the leaf with largest loss reduction), while XGBoost grows level-wise. Leaf-wise is more efficient but can overfit with small data.

**Q14. Answer: B**
A smaller learning rate shrinks each tree's contribution, requiring more trees to reach the same training loss. This slower, more careful approach typically produces better generalization.

**Q15. Answer: B**
Trees make binary decisions (is feature > threshold?), which is unaffected by monotonic transformations of features. Multiplying a feature by 100 changes the threshold but not the split's effect.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# Multiple Choice Questions: Model Evaluation and Selection

📺 **Video Lecture:** https://youtu.be/F-JCSIv_gDo


Test your understanding of metrics, cross-validation, and model selection techniques.

---

**Q1. Accuracy is a misleading metric when:**

A) The model is very complex
B) The classes are highly imbalanced (e.g., 99% negative, 1% positive)
C) The features are standardized
D) Cross-validation is used

---

**Q2. Precision is defined as:**

A) TP / (TP + FN)
B) TP / (TP + FP) — the fraction of positive predictions that are actually positive
C) (TP + TN) / Total
D) TN / (TN + FP)

---

**Q3. Recall (Sensitivity) is defined as:**

A) TP / (TP + FP)
B) TP / (TP + FN) — the fraction of actual positives correctly identified
C) TN / (TN + FP)
D) (TP + TN) / Total

---

**Q4. The F1 score is:**

A) The average of accuracy and precision
B) The harmonic mean of precision and recall
C) The geometric mean of specificity and sensitivity
D) Always higher than accuracy

---

**Q5. AUC-ROC measures:**

A) The model's accuracy at a single threshold
B) The model's ability to discriminate between classes across all possible thresholds
C) The number of features used
D) The training time

---

**Q6. K-fold cross-validation works by:**

A) Training on 50% of data and testing on 50%
B) Splitting data into k folds, training on k−1 folds and testing on the remaining fold, rotating k times
C) Randomly selecting k features
D) Using the entire dataset for both training and testing

---

**Q7. Stratified k-fold cross-validation preserves:**

A) The temporal order of data
B) The class distribution (proportion of each class) in each fold
C) The feature scaling across folds
D) The exact same samples in each fold

---

**Q8. When comparing two models, a model with lower training error but higher test error likely:**

A) Is underfitting
B) Is overfitting (has high variance)
C) Is perfectly fit
D) Needs more features

---

**Q9. The ROC curve plots:**

A) Precision vs. Recall
B) True Positive Rate vs. False Positive Rate at various thresholds
C) Accuracy vs. Number of features
D) Loss vs. Epochs

---

**Q10. In a medical screening test where missing a disease is costly, you should optimize for:**

A) High precision (few false positives)
B) High recall (few false negatives — catch most actual positives)
C) Low AUC
D) High specificity only

---

**Q11. Leave-one-out cross-validation (LOOCV) has:**

A) High bias, low variance in error estimate
B) Low bias but high variance in error estimate, and is computationally expensive
C) No bias and no variance
D) Always lower error than k-fold

---

**Q12. The Precision-Recall curve is preferred over ROC when:**

A) Classes are balanced
B) Classes are highly imbalanced (ROC can be overly optimistic with many true negatives)
C) The model is linear
D) Only binary features are present

---

**Q13. Mean Squared Error (MSE) is used for regression and is sensitive to:**

A) The number of features
B) Outliers (large errors are squared, amplifying their effect)
C) The sign of predictions
D) Categorical targets

---

**Q14. The log loss (cross-entropy) metric penalizes:**

A) Only incorrect predictions
B) Confident wrong predictions more heavily than uncertain wrong predictions
C) All predictions equally
D) Only predictions above 0.5

---

**Q15. Nested cross-validation is used to:**

A) Speed up model training
B) Provide an unbiased estimate of model performance when hyperparameters are also tuned via CV
C) Reduce the dataset size
D) Eliminate the need for a test set

---

## Answer Key

**Q1. Answer: B**
With 99% negatives, a model predicting "always negative" achieves 99% accuracy but is useless. Precision, recall, F1, and AUC are more informative for imbalanced datasets.

**Q2. Answer: B**
Precision = TP/(TP+FP): "Of all predicted positives, how many are truly positive?" High precision means few false alarms.

**Q3. Answer: B**
Recall = TP/(TP+FN): "Of all actual positives, how many did we catch?" High recall means few missed positives.

**Q4. Answer: B**
F1 = 2×(Precision×Recall)/(Precision+Recall). The harmonic mean penalizes models where one metric is high but the other is very low, encouraging balance.

**Q5. Answer: B**
AUC-ROC summarizes classification performance across all thresholds. AUC = 1.0 is perfect, 0.5 is random. It measures ranking quality — how well the model separates positives from negatives.

**Q6. Answer: B**
K-fold CV uses all data for both training and validation. Each fold serves as the test set exactly once. The final metric is the average across all k folds, giving a robust performance estimate.

**Q7. Answer: B**
Stratified k-fold ensures each fold has approximately the same class proportions as the full dataset. This is especially important for imbalanced datasets to avoid folds with no minority class samples.

**Q8. Answer: B**
A large gap between training and test error indicates overfitting — the model memorizes training data but fails to generalize. Solutions include regularization, more data, or simpler models.

**Q9. Answer: B**
The ROC curve plots TPR (recall) vs. FPR (1−specificity) as the classification threshold varies from 0 to 1. The curve shows the tradeoff between catching positives and creating false alarms.

**Q10. Answer: B**
When false negatives are costly (missing a disease), prioritize recall. You'd rather have some false positives (unnecessary follow-up tests) than miss actual cases.

**Q11. Answer: B**
LOOCV uses n−1 training points per fold (very close to full data → low bias), but each fold's test set is one point, creating high variance in the error estimate. Also requires n model fits.

**Q12. Answer: B**
With imbalanced data, TN dominates, making FPR artificially low and ROC overly optimistic. PR curves focus on the minority (positive) class, providing a more honest assessment.

**Q13. Answer: B**
MSE squares errors, so a single large error (outlier) disproportionately inflates MSE. MAE (mean absolute error) is more robust to outliers but less mathematically convenient.

**Q14. Answer: B**
Log loss = −[y log(p) + (1−y)log(1−p)]. A confident wrong prediction (e.g., predicting 0.99 when true label is 0) receives extremely high penalty, encouraging well-calibrated probabilities.

**Q15. Answer: B**
Nested CV has an outer loop for performance estimation and inner loop for hyperparameter tuning. This prevents the optimistic bias that occurs when the same CV is used for both tuning and evaluation.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

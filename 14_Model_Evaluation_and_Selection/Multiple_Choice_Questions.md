# Multiple Choice Questions: Model Evaluation and Selection

📺 **Video Lecture:** https://youtu.be/F-JCSIv_gDo


Test your understanding of metrics, cross-validation, and model selection techniques.

---

**Q1. Accuracy is a misleading metric when:**

A) Cross-validation is used  
B) The features are standardized  
C) The classes are highly imbalanced (e.g., 99% negative, 1% positive)  
D) The model is very complex

---

**Q2. Precision is defined as:**

A) TP / (TP + FP) — the fraction of positive predictions that are actually positive  
B) TP / (TP + FN)  
C) (TP + TN) / Total  
D) TN / (TN + FP)

---

**Q3. Recall (Sensitivity) is defined as:**

A) TN / (TN + FP)  
B) (TP + TN) / Total  
C) TP / (TP + FN) — the fraction of actual positives correctly identified  
D) TP / (TP + FP)

---

**Q4. The F1 score is:**

A) The average of accuracy and precision  
B) The harmonic mean of precision and recall  
C) Always higher than accuracy  
D) The geometric mean of specificity and sensitivity

---

**Q5. AUC-ROC measures:**

A) The training time  
B) The number of features used  
C) The model's accuracy at a single threshold  
D) The model's ability to discriminate between classes across all possible thresholds

---

**Q6. K-fold cross-validation works by:**

A) Splitting data into k folds, training on k−1 folds and testing on the remaining fold, rotating k times  
B) Using the entire dataset for both training and testing  
C) Randomly selecting k features  
D) Training on 50% of data and testing on 50%

---

**Q7. Stratified k-fold cross-validation preserves:**

A) The class distribution (proportion of each class) in each fold  
B) The temporal order of data  
C) The feature scaling across folds  
D) The exact same samples in each fold

---

**Q8. When comparing two models, a model with lower training error but higher test error likely:**

A) Is perfectly fit  
B) Is underfitting  
C) Needs more features  
D) Is overfitting (has high variance)

---

**Q9. The ROC curve plots:**

A) Loss vs. Epochs  
B) Accuracy vs. Number of features  
C) Precision vs. Recall  
D) True Positive Rate vs. False Positive Rate at various thresholds

---

**Q10. In a medical screening test where missing a disease is costly, you should optimize for:**

A) Low AUC  
B) High precision (few false positives)  
C) High specificity only  
D) High recall (few false negatives — catch most actual positives)

---

**Q11. Leave-one-out cross-validation (LOOCV) has:**

A) Low bias but high variance in error estimate, and is computationally expensive  
B) No bias and no variance  
C) High bias, low variance in error estimate  
D) Always lower error than k-fold

---

**Q12. The Precision-Recall curve is preferred over ROC when:**

A) Only binary features are present  
B) Classes are balanced  
C) The model is linear  
D) Classes are highly imbalanced (ROC can be overly optimistic with many true negatives)

---

**Q13. Mean Squared Error (MSE) is used for regression and is sensitive to:**

A) Categorical targets  
B) The sign of predictions  
C) The number of features  
D) Outliers (large errors are squared, amplifying their effect)

---

**Q14. The log loss (cross-entropy) metric penalizes:**

A) Only incorrect predictions  
B) Confident wrong predictions more heavily than uncertain wrong predictions  
C) All predictions equally  
D) Only predictions above 0.5

---

**Q15. Nested cross-validation is used to:**

A) Provide an unbiased estimate of model performance when hyperparameters are also tuned via CV  
B) Eliminate the need for a test set  
C) Reduce the dataset size  
D) Speed up model training

---

## Answer Key

**Q1. Answer: C**
With 99% negatives, a model predicting "always negative" achieves 99% accuracy but is useless. Precision, recall, F1, and AUC are more informative for imbalanced datasets.

**Q2. Answer: A**
Precision = TP/(TP+FP): "Of all predicted positives, how many are truly positive?" High precision means few false alarms.

**Q3. Answer: C**
Recall = TP/(TP+FN): "Of all actual positives, how many did we catch?" High recall means few missed positives.

**Q4. Answer: B**
F1 = 2×(Precision×Recall)/(Precision+Recall). The harmonic mean penalizes models where one metric is high but the other is very low, encouraging balance.

**Q5. Answer: D**
AUC-ROC summarizes classification performance across all thresholds. AUC = 1.0 is perfect, 0.5 is random. It measures ranking quality — how well the model separates positives from negatives.

**Q6. Answer: A**
K-fold CV uses all data for both training and validation. Each fold serves as the test set exactly once. The final metric is the average across all k folds, giving a robust performance estimate.

**Q7. Answer: A**
Stratified k-fold ensures each fold has approximately the same class proportions as the full dataset. This is especially important for imbalanced datasets to avoid folds with no minority class samples.

**Q8. Answer: D**
A large gap between training and test error indicates overfitting — the model memorizes training data but fails to generalize. Solutions include regularization, more data, or simpler models.

**Q9. Answer: D**
The ROC curve plots TPR (recall) vs. FPR (1−specificity) as the classification threshold varies from 0 to 1. The curve shows the tradeoff between catching positives and creating false alarms.

**Q10. Answer: D**
When false negatives are costly (missing a disease), prioritize recall. You'd rather have some false positives (unnecessary follow-up tests) than miss actual cases.

**Q11. Answer: A**
LOOCV uses n−1 training points per fold (very close to full data → low bias), but each fold's test set is one point, creating high variance in the error estimate. Also requires n model fits.

**Q12. Answer: D**
With imbalanced data, TN dominates, making FPR artificially low and ROC overly optimistic. PR curves focus on the minority (positive) class, providing a more honest assessment.

**Q13. Answer: D**
MSE squares errors, so a single large error (outlier) disproportionately inflates MSE. MAE (mean absolute error) is more robust to outliers but less mathematically convenient.

**Q14. Answer: B**
Log loss = −[y log(p) + (1−y)log(1−p)]. A confident wrong prediction (e.g., predicting 0.99 when true label is 0) receives extremely high penalty, encouraging well-calibrated probabilities.

**Q15. Answer: A**
Nested CV has an outer loop for performance estimation and inner loop for hyperparameter tuning. This prevents the optimistic bias that occurs when the same CV is used for both tuning and evaluation.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

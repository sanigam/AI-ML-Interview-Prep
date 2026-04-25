# Model Evaluation and Selection

📺 **Video Lecture:** https://youtu.be/F-JCSIv_gDo


## Interview Anchor
- **Classification Metrics:** Accuracy, precision, recall, F1; choose based on class distribution and cost asymmetry.
- **ROC Curve & AUC:** Plots TPR vs. FPR; AUC summarizes classifier quality; threshold-independent.
- **Cross-Validation:** K-fold, stratified, time-series CV; prevents optimistic bias; essential for model selection.

## Key Concepts Overview

Model evaluation and selection are fundamental to ML practice, yet frequently misunderstood or done incorrectly. This topic tests both technical knowledge—computing precision/recall, understanding ROC curves, interpreting AUC—and practical wisdom about when metrics fail (class imbalance, cost asymmetry) and how to avoid pitfalls (test set leakage, optimistic reporting). Interviewers assess whether you understand the difference between various classification metrics, why cross-validation matters, how to tune hyperparameters without overfitting to the validation set, and when a single metric is insufficient. Additionally, this topic touches on calibration (probability estimates), learning curves (detecting bias-variance), and model selection criteria (AIC, BIC). Strong candidates discuss metric limitations contextually and validate multiple perspectives before declaring a "best" model.

---

### Q1: Explain key classification metrics: accuracy, precision, recall, F1-score.

**A:** Accuracy = (TP + TN) / (TP + TN + FP + FN), the fraction of correct predictions. Simple, but misleading on imbalanced data: 99% negative class accuracy is worthless if your goal is detecting rare positives. Precision = TP / (TP + FP), the fraction of positive predictions that are correct; answers "when model predicts positive, how often is it right?" Recall (sensitivity) = TP / (TP + FN), the fraction of actual positives detected; answers "of all true positives, how many did we find?" F1 = 2 × (Precision × Recall) / (Precision + Recall), the harmonic mean, balancing both. Trade-off: increasing recall often decreases precision (lower decision threshold → more positives, more false positives). Choice depends on cost structure: (1) False negatives costly (disease detection)? Maximize recall—catch all positives even with false alarms, (2) False positives costly (spam filtering)? Maximize precision—avoid innocent users marked as spam. (3) Both equal? Use F1 or accuracy (on balanced data). Macro-averaging: compute metric per class, average (treats classes equally). Micro-averaging: pool all TP/FP/FN, compute metric (weights by class frequency). On imbalanced data, macro-average is often preferred. In interviews, explain metrics in context—accuracy alone is insufficient for imbalanced data; precision/recall reveal trade-offs.

---

### Q2: Explain ROC curve and AUC-ROC, and when to use them.

**A:** ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR = Recall = TP / (TP + FN)) vs. False Positive Rate (FPR = FP / (FP + TN)) as the decision threshold varies. Starting at (0, 0) (high threshold, predict few positives), moving to (1, 1) (low threshold, predict many positives). A diagonal line from (0,0) to (1,1) represents random guessing (50% probability). Good classifiers curve toward top-left (high TPR, low FPR). AUC (Area Under Curve) summarizes ROC curve as a scalar (0 to 1); AUC = 0.5 random, 1.0 perfect. AUC = probability that classifier ranks a random positive sample higher than a random negative sample (ranking interpretation). Advantages: (1) threshold-independent (single number for any decision rule), (2) handles class imbalance well (uses rates, not absolute counts), (3) intuitive interpretation (probability of correct ranking). Disadvantages: (1) assumes all thresholds equally important (reality: one operating point matters), (2) AUC can be misleading with extreme class imbalance (99:1, AUC may look good despite poor precision), (3) focuses on ranking, not calibration (probabilities). Use ROC/AUC: imbalanced data, when threshold is flexible (don't yet know operating point), evaluation across multiple thresholds. Don't use ROC/AUC: heavily imbalanced data (precision-recall curve better), when one specific threshold is fixed (use confusion matrix metrics). In interviews, explain both strengths and limitations; many practitioners over-rely on AUC.

---

### Q3: Explain AUC-PR (Precision-Recall Curve) and when it's better than AUC-ROC.

**A:** Precision-Recall curve plots Recall (x-axis) vs. Precision (y-axis) as threshold varies. Area Under PR Curve (AUC-PR) summarizes; ranges [0,1]. Advantage: focuses on positive class performance, ignoring true negatives—crucial for imbalanced data. Example: 99% negative class. ROC includes TN in FPR denominator, so FPR stays low even if FP is high (TN is huge); PR curve ignores TN, directly showing precision (did we correctly identify positives?). With 99:1 imbalance, ROC-AUC might be 0.95 (looks good) while AUC-PR is 0.50 (bad). PR curve is more honest on imbalanced data. Disadvantages: less intuitive than ROC (Precision = TP/(TP+FP) depends on both TP and FP), less commonly used (fewer tools), random baseline is not 0.5 but P(positive) (positive class rate). Practical choice: (1) balanced data (≈50-50): ROC and PR curves similar, use ROC (simpler), (2) moderately imbalanced (70-30 to 90-10): ROC still useful, complement with PR, (3) extremely imbalanced (99-1): use PR curve primarily. In interviews, mentioning PR curve for imbalanced data shows nuanced understanding. Many practitioners default to AUC-ROC unaware of limitations; awareness differentiates you. Recommendation: always report both on imbalanced data.

---

### Q4: Explain confusion matrix and its interpretation in multiclass scenarios.

**A:** Confusion matrix is an n×n table (for n classes) where entry (i,j) = number of samples truly class i but predicted as class j. Diagonal = correct predictions, off-diagonal = errors. Example (binary): [[TP, FP], [FN, TN]]. From confusion matrix, derive: Accuracy, Precision per class, Recall per class. For multiclass, compute macro-average (per-class metrics averaged) or micro-average (pooled counts). Interpretation: (1) high diagonal values ↔ good classifier, (2) off-diagonal patterns reveal specific confusions—e.g., if class A often confused with class B, they may be intrinsically similar; investigate. (3) imbalanced confusion matrix (one class dominates) may indicate class imbalance or poor training. Metrics per class: Precision_i = CM[i,i] / ∑_j CM[j,i] (what fraction of predicted class i are truly class i), Recall_i = CM[i,i] / ∑_j CM[i,j] (what fraction of true class i are correctly predicted). In multiclass, macro-average treats all classes equally (useful for balanced data, or when all errors equal), micro-average weights by class frequency (useful for imbalanced data, or when rare class errors matter less). In interviews, analyzing a confusion matrix shows practical understanding. Discuss that off-diagonal patterns can guide feature engineering or data collection (improve data for confused classes). Mention that raw counts (non-normalized) confusion matrix is less informative than percentages or normalized matrix.

---

### Q5: Explain log loss (cross-entropy loss) and its relationship to probability calibration.

**A:** Log loss (cross-entropy): L = -(1/n) ∑_i [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)] for binary classification, where ŷ_i ∈ (0,1) is predicted probability. For multiclass: L = -(1/n) ∑_i ∑_k y_ik log(ŷ_ik), where y_ik is one-hot encoded, ŷ_ik is predicted probability for class k. Log loss penalizes confident wrong predictions heavily (log(0) → ∞ as ŷ → 0 for true label). Key property: minimizing log loss produces calibrated probability estimates—predicted probability matches empirical frequency. Example: if model predicts 70% probability on 100 samples, ~70 should be positive. Well-calibrated models enable downstream decision-making (set thresholds based on business cost). Comparison to accuracy: accuracy is 0-1 (wrong/right), log loss is continuous (penalizes degree of wrongness). Log loss is more sensitive: improving from 99% to 100% accuracy (10 more correct out of 1000) changes accuracy slightly but log loss significantly (infinity penalty removed). Use log loss: (1) need probability estimates, (2) care about calibration, (3) continuous performance measure (learning curves). Use accuracy: simpler, less sensitive to outliers, interpretable. In interviews, explain calibration connection—many practitioners see log loss as black-box metric, unaware it drives probability quality. Mention Platt scaling or temperature scaling as post-hoc calibration if raw model probabilities are poorly calibrated.

---

### Q6: Explain regression metrics: MSE, MAE, RMSE, R-squared.

**A:** Mean Squared Error (MSE) = (1/n) ∑(y_i - ŷ_i)², sensitive to outliers (large errors squared). Mean Absolute Error (MAE) = (1/n) ∑|y_i - ŷ_i|, robust to outliers (linear penalty). RMSE = √MSE, in same units as y (more interpretable than MSE). Choice: (1) outliers rare/important? Use MSE, (2) outliers common/noise? Use MAE, (3) need interpretability in original units? Use RMSE or MAE. R² (coefficient of determination) = 1 - (SS_res / SS_tot), measures fraction of variance explained. R² ∈ (-∞, 1]; 1 = perfect fit, 0 = predicting mean, < 0 = worse than mean. Advantages: unit-independent (0-1 scale), interpretable (% variance explained), enables model comparison. Disadvantages: assumes linear relationship between y and ŷ, can mislead with nonlinear fits, penalizes complexity implicitly. Adjusted R² = 1 - ((1-R²)(n-1)/(n-p-1)) accounts for number of features p; decreases if features don't improve enough (model selection tool). In interviews, discuss metric choice contextually: outliers present? MAE. Interpretability critical? RMSE/MAE. Variance explanation relevant? R². Always report multiple metrics—single metric insufficient. Mention that RMSE/MSE sensitive to rare large errors; MAE more robust. For regression, learning curves (train vs. validation error) are valuable complementary views.

---

### Q7: Explain K-fold cross-validation and its advantages over a single train-test split.

**A:** K-fold cross-validation splits data into K equal parts (folds), performs K iterations: in iteration k, use fold k as test, remaining K-1 as training. Average performance across K iterations gives overall CV score. Advantages: (1) uses all data for training and testing (no wasted validation set), (2) reduces variance in performance estimate (K independent measurements, average more stable), (3) detects overfitting (if train >> test error, model is overfitting), (4) stable for small datasets (K-1 folds still substantial training data). Typical K = 5-10; k-fold = leave-one-out CV, leave-one-out is unbiased but computationally expensive O(n) training runs. Stratified K-fold: ensures each fold has same class distribution as full data; critical for imbalanced classification (random split might create folds with zero positives). Time-series CV: for temporal data, don't shuffle; use growing windows (train on past, test on future in chronological order; prevents future leakage). Disadvantages: (1) k-fold slower than single split (K times training), (2) reports average performance (doesn't show variance across folds), (3) hyperparameter tuning on CV results can still overfit (need separate test set). Best practice: (1) train-validation-test split (train on train, tune hyperparameters on validation), (2) final evaluation on held-out test set never seen during tuning, (3) report CV performance on validation set (honest estimate), (4) report test set performance (unbiased). In interviews, emphasizing proper train-val-test split and CV shows rigor. Many practitioners only do train-test split, missing CV's benefits.

---

### Q8: Explain stratified K-fold cross-validation and when it's essential.

**A:** Stratified K-fold ensures each fold maintains class distribution of full dataset. For imbalanced data (99% negative, 1% positive), random K-fold might create a fold with zero positives—training on that fold would be meaningless. Stratified ensures ~99% negative in each fold, maintaining data statistics. Implementation: shuffle, sort by class label, cyclically assign folds (fold 1 gets samples 1, K+1, 2K+1, ..., fold 2 gets samples 2, K+2, 2K+2, ...). Guarantees class balance. Essential for: (1) imbalanced classification, (2) small datasets (each fold is small; stratification ensures each has representative samples), (3) multiclass (ensure each class present in each fold). Less critical: (1) large balanced datasets (random splits naturally represent each class), (2) regression (no class distribution). In practice, always use stratified K-fold for classification unless data is very balanced and large—it's safer and never hurts. For multiclass with K classes, stratified K-fold preserves all K classes in each fold simultaneously (more complex than binary, but most libraries handle it). In interviews, mentioning stratified K-fold for imbalanced data shows awareness—many practitioners use plain K-fold unaware of class imbalance consequences. Example impact: with 1% positive rate, random 5-fold might create a fold with 0% positives; metrics on that fold are undefined (divide by zero). Stratified prevents this.

---

### Q9: Explain hyperparameter tuning: grid search, random search, and Bayesian optimization.

**A:** Hyperparameter tuning finds settings minimizing validation error. Grid search: exhaustively evaluate all combinations from a discrete grid. Example: max_depth ∈ [3,5,7,10], learning_rate ∈ [0.01, 0.1, 1.0]—9 combinations tested. Advantages: simple, exhaustive (guaranteed to find best in grid), parallelizable. Disadvantages: exponential cost (d hyperparameters with k values each = O(k^d) combinations), inefficient (wastes computation on unpromising regions). Random search: randomly sample hyperparameters from distributions. Advantages: (1) faster than grid (sample fixed number, e.g., 100 combinations, not all), (2) better coverage (especially for high-dimensional spaces), (3) more likely to find good regions outside grid. Disadvantages: may miss exact grid optima, requires more iterations for same confidence. Bayesian optimization: use probabilistic model (Gaussian process) to model performance surface, select next hyperparameters to maximize information gain. Advantages: (1) most sample-efficient (focuses on promising regions), (2) adapts to landscape, (3) fewer iterations needed. Disadvantages: more complex to implement, longer per-iteration computation (fitting GP). Practical recommendation: (1) coarse grid search (large steps, few values) to understand landscape, (2) fine random search or Bayesian around best region, (3) use cross-validation for robustness (not just single validation set). Tools: GridSearchCV, RandomizedSearchCV (scikit-learn), Hyperopt, Optuna (Bayesian). In interviews, discussing trade-offs (computational cost vs. sample efficiency) shows practical wisdom. Avoid endless tuning—returns diminish; 80% of gain often from first 20% of tuning effort.

---

### Q10: Explain Bayesian optimization and its advantages for hyperparameter tuning.

**A:** Bayesian optimization uses a probabilistic surrogate model (typically Gaussian Process) to model the relationship between hyperparameters and performance. Algorithm: (1) Define prior distribution over function space, (2) compute posterior given observed (hyperparameter, performance) pairs, (3) use acquisition function (e.g., Expected Improvement, Upper Confidence Bound) to select next hyperparameters balancing exploration (try uncertain regions) and exploitation (refine promising regions), (4) train model with selected hyperparameters, update posterior, repeat. Advantages: (1) sample-efficient (exploits landscape structure, finds optimum with fewer iterations), (2) adaptive (focuses on promising regions, skips obviously bad ones), (3) handles continuous hyperparameters naturally (no discretization), (4) probabilistic (quantifies uncertainty, enables principled decisions). Disadvantages: (1) computationally expensive (fitting GP per iteration), (2) more complex (requires GPL/Bayesian expertise), (3) hyperparameter optimization itself has hyperparameters (kernel, acquisition function). Comparison: grid search O(k^d) exhaustive, random search O(N) fixed samples, Bayesian optimization O(N × computation) but often N << k^d. Example: 5 hyperparameters, 10 values each → grid search 100k combinations; Bayesian optimization finds optimum in ~20-50 iterations. Applications: deep learning (expensive to train), AutoML (automatic model selection). Libraries: Hyperopt, Optuna, Ax, Spearmint. In interviews, demonstrating knowledge of Bayesian optimization shows advanced understanding. Mention that it's increasingly standard in industry (AutoML platforms use it); mention computational trade-offs (feasible for ~5-10 hyperparameters, harder for 50+).

---

### Q11: Explain learning curves and what they reveal about model bias/variance.

**A:** Learning curve plots training and validation error vs. training set size. Reveals model behavior: (1) High bias (underfitting): both train and validation curves plateau at high error; adding more data doesn't help. Solution: more complex model, better features. (2) High variance (overfitting): large gap between train and validation curves; train error low, validation error high. Solution: regularization, more data, simpler model. (3) Good fit: curves close together, both low (ideal). (4) High variance + low bias: train error keeps decreasing with more data, validation error decreases but lags. (5) High bias + low variance: both curves plateau early, high error. Practical use: (1) diagnose the problem (is it bias or variance?), (2) guide next steps (add data for variance, complex model for bias), (3) determine diminishing returns (if curves flatten, more data won't help). Curve interpretation details: with more training data, training error typically increases (larger, more diverse training set is harder to fit perfectly), validation error typically decreases (more data ↔ better model estimate). If validation error diverges strongly from training as data increases, high variance. If both converge to high error, high bias. In interviews, generating and interpreting learning curves shows practical ML understanding. Mention that learning curves are underused—many practitioners don't inspect them but they provide invaluable diagnostics. Always visualize learning curves when model performance is underwhelming.

---

### Q12: Explain model selection criteria: AIC, BIC, and their relationship to cross-validation.

**A:** AIC (Akaike Information Criterion) = 2k - 2log(L), BIC (Bayesian Information Criterion) = k log(n) - 2log(L), where k = number of parameters, n = sample size, L = maximum likelihood. Both balance fit quality (log-likelihood, higher is better) and complexity (k). Lower AIC/BIC is better. BIC penalizes complexity more (multiplies k by log(n)) than AIC, favoring simpler models when n is large. Interpretation: AIC/BIC are proxies for generalization error (under certain assumptions). Minimize AIC/BIC to select best model among candidates. Advantages: (1) single number (no separate validation set needed), (2) theoretically grounded (AIC from Kullback-Leibler divergence, BIC from Bayesian perspective), (3) applicable to non-nested models. Disadvantages: (1) assume model is correct (Gaussian errors, etc.), (2) less intuitive than CV, (3) not always reliable when assumptions violated. Relationship to CV: cross-validation directly estimates generalization error (empirical); AIC/BIC estimate it via theory (analytical). CV is often preferred in practice (no assumptions), but AIC/BIC are fast (no retraining). Practical recommendation: (1) use cross-validation for model selection (most robust), (2) use AIC/BIC for feature selection within model (quick screening), (3) compare multiple criteria (if they disagree, investigate). In interviews, discussing both theoretical (AIC/BIC) and empirical (CV) approaches shows balance. Mention that CV is increasingly preferred as computational cost drops; AIC/BIC remain useful for interpretability (parameter penalties are explicit).

---

### Q13: Explain calibration and how to assess and improve probability estimates.

**A:** Calibration measures if predicted probabilities match empirical frequencies—if model predicts 70% probability on 100 samples, ~70 should be positive. Well-calibrated probabilities enable principled decision-making (set threshold based on business cost). Assessing calibration: (1) Calibration plot: bin predictions by probability (e.g., [0-0.1], [0.1-0.2], ...), compute empirical frequency in each bin, plot against predicted probability; perfect calibration is diagonal line, (2) Expected Calibration Error (ECE): average absolute difference between predicted and empirical probability, weighted by bin frequency; lower is better, (3) Brier score: B = (1/n) ∑(ŷ_i - y_i)² (MSE of probabilities), penalizes miscalibration and misclassification. Improving calibration: (1) Platt scaling: fit logistic regression P(y=1) = sigmoid(a×f(x) + b) on predicted scores f(x); re-calibrates probabilities, (2) Temperature scaling: scale logits by scalar τ, then apply sigmoid; T < 1 sharpens predictions (high confidence), T > 1 softens (lower confidence), (3) Isotonic regression: fit non-parametric monotonic mapping from predictions to calibrated probabilities. Some models naturally well-calibrated (logistic regression with log loss); others poorly calibrated (neural networks, random forests). In practice: (1) check calibration on validation set, (2) if poorly calibrated, apply post-hoc calibration, (3) report both accuracy and calibration (e.g., AUC and ECE). In interviews, mentioning calibration shows awareness beyond standard metrics. Many practitioners ignore it, missing the value of probability estimates (actionable thresholds, cost-based decisions).

---

### Q14: Explain Brier score and its interpretation.

**A:** Brier score: B = (1/n) ∑(ŷ_i - y_i)², the mean squared error of probabilities. Ranges [0, 1]; 0 = perfect (all probabilities correct), 1 = worst (predicting opposite probabilities). Example: 100 samples, 60 positive; model predicts 0.7 for all → B = (0.7-1)² × 60 + (0.7-0)² × 40 = 0.09 × 60 + 0.49 × 40 = 5.4 + 19.6 = 25 / 100 = 0.25. Interpretation: (1) decomposes into calibration error (difference between predicted and empirical probability) and refinement (ability to rank-order samples), (2) sensitive to both confidence and accuracy, (3) comparable across models (same scale). Comparison to log loss: log loss penalizes confident wrong predictions exponentially (log(0.01) = -4.6), Brier score penalizes quadratically; Brier score is more interpretable (squared error in probability space). When to use: (1) need interpretable probability quality metric, (2) calibration matters, (3) comparing probabilistic predictions. Disadvantages: (1) less sensitive to extreme errors (quadratic vs. exponential), (2) less used than log loss (non-standard in some libraries), (3) scale-dependent (random guessing Brier = 0.25 for balanced binary, 0.5 for extreme imbalance). In interviews, explaining Brier score decomposition (calibration + refinement) demonstrates understanding of probability quality. Mention that ignoring Brier score is common; assessing it alongside accuracy provides fuller picture.

---

### Q15: How would you design a comprehensive model evaluation strategy, and how do you avoid pitfalls?

**A:** Comprehensive evaluation strategy: (1) **Data splitting**: train (60%), validation (20%), test (20%), with stratification for imbalance, time-based splits for temporal data. Validation for hyperparameter tuning, test for final unbiased estimate. (2) **Cross-validation**: run K-fold CV on train+validation to get robust performance estimate; monitor variance across folds (high variance = unstable model), (3) **Metrics selection**: use multiple metrics aligned with business goals—accuracy, precision, recall, F1 for classification; MAE, RMSE for regression; always check calibration (Brier score, calibration plots). (4) **Learning curves**: diagnose bias-variance; inspect as sample size increases, (5) **Ablation studies**: remove features/components, measure impact; validates feature importance, (6) **Error analysis**: visualize misclassifications, identify patterns (certain classes confused, certain features important), (7) **Sanity checks**: baseline comparison (random, dummy model), compare multiple algorithms, ensure improvements are real (not noise), (8) **External validation** (if possible): test on new data or different domain; catches distribution shift. Pitfalls to avoid: (1) **Test set leakage**: don't touch test set during tuning; don't generate features using test data, (2) **Reporting only accuracy**: insufficient on imbalanced data; report precision/recall/F1, (3) **Hyperparameter overfitting**: tuning on CV results can still overfit; use separate validation set, (4) **Ignoring calibration**: probability estimates may be miscalibrated despite good accuracy, (5) **Not comparing baselines**: always have simple baselines (linear, dummy); complex models should beat them significantly, (6) **Selective reporting**: report all metrics, not cherry-picking favorable ones, (7) **Assuming one metric sufficient**: use multiple angles; confusion matrix + ROC + learning curves > single AUC. In interviews, this structured thinking impresses. Discuss trade-offs (computation vs. reliability) and practical constraints (time, data). Strong answer: "I'd start simple (baseline + single split), iterate (CV, multiple metrics), then diagnose issues (learning curves, error analysis) before final test evaluation."

---

## Interview Cheatsheet

**Key Terms:**

- **Accuracy:** (TP+TN)/(TP+TN+FP+FN); overall correctness; misleading on imbalanced data.
- **Precision:** TP/(TP+FP); of predicted positives, how many are correct?
- **Recall (Sensitivity):** TP/(TP+FN); of true positives, how many detected?
- **F1-score:** 2 × (Precision × Recall) / (Precision + Recall); harmonic mean balancing both.
- **Specificity:** TN/(TN+FP); correctly identified negatives; recall's counterpart.
- **ROC Curve:** TPR vs. FPR as threshold varies; threshold-independent summary.
- **AUC-ROC:** Area under ROC; 0.5 random, 1.0 perfect; probability of correct ranking.
- **AUC-PR:** Area under precision-recall curve; better for imbalanced data than AUC-ROC.
- **Log Loss (Cross-Entropy):** -(1/n) ∑ y_i log(ŷ_i); penalizes confident wrong predictions; drives calibration.
- **MSE/RMSE:** Mean squared error / root; sensitive to outliers; interpretable in original units.
- **MAE:** Mean absolute error; robust to outliers; linear penalty.
- **R²:** Coefficient of determination; fraction of variance explained; 1 = perfect, 0 = mean prediction.
- **Confusion Matrix:** n×n table; diagonal = correct, off-diagonal = errors; reveals specific confusions.
- **K-fold CV:** Split data K ways, train on K-1, test on 1; average K scores; all data used.
- **Stratified K-fold:** Maintains class distribution in each fold; essential for imbalanced data.
- **Calibration:** Predicted probability matches empirical frequency; assessed via calibration plots, ECE.
- **Brier Score:** (1/n) ∑(ŷ_i - y_i)²; MSE of probabilities; 0 = perfect, 1 = worst.
- **Learning Curves:** Train vs. validation error vs. sample size; diagnoses bias-variance.
- **AIC/BIC:** Information criteria balancing fit and complexity; lower is better.
- **Grid Search:** Exhaustively evaluate all hyperparameter combinations; slow but complete.
- **Random Search:** Randomly sample hyperparameters; faster than grid, better exploration.
- **Bayesian Optimization:** Probabilistic model (GP) for hyperparameter tuning; sample-efficient.

**Rapid-Fire Q&A:**

- **Q:** Accuracy misleading when? **A:** Imbalanced data (99:1 split); 99% accuracy achieved by predicting majority class only.
- **Q:** Precision vs. recall trade-off? **A:** Lower threshold ↔ higher recall (catch more positives), lower precision (more false alarms).
- **Q:** Which metric for disease detection? **A:** Recall (maximize true positives); false negatives costly (miss disease).
- **Q:** AUC-ROC sensitive to class imbalance? **A:** Yes, can be misleading on extreme imbalance; AUC-PR better.
- **Q:** Log loss vs. accuracy: difference? **A:** Log loss continuous (degree of wrongness), accuracy 0-1 (binary); log loss more sensitive.
- **Q:** Calibration plot: perfect is? **A:** Diagonal line (predicted probability = empirical frequency in each bin).
- **Q:** K-fold or single train-test split? **A:** K-fold more robust (all data used, variance estimate); single faster.
- **Q:** Stratified vs. regular K-fold: when critical? **A:** Imbalanced classification; regular K-fold may create folds missing rare class.
- **Q:** Learning curve high bias: fix? **A:** More complex model, more features; adding data won't help.
- **Q:** Learning curve high variance: fix? **A:** Regularization, more data, simpler model; variance decreases with more samples.
- **Q:** Grid or random search? **A:** Grid for ~2-3 hyperparameters, random for 4+; random usually more efficient.
- **Q:** Bayesian optimization advantage? **A:** Sample-efficient; focuses on promising regions, skips obviously bad ones.
- **Q:** AIC vs. BIC: difference? **A:** BIC penalizes complexity more (log(n) factor); BIC favors simpler models.
- **Q:** Test set leakage: what? **A:** Using test data during training/tuning; invalidates performance estimates; must keep test set separate.
- **Q:** Multiple metrics or just AUC? **A:** Always multiple; confusion matrix + precision + recall + calibration gives complete picture.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

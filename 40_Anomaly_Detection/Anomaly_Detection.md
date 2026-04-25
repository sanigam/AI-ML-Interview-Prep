# Anomaly Detection

📺 **Video Lecture:** https://youtu.be/wOb5Il6zMqI


## Interview Anchor
- **Point Anomalies:** Individual data points that deviate significantly from the overall pattern or expected behavior.
- **Contextual Anomalies:** Data points that are anomalous within a specific context or subsequence, but may be normal in other contexts.
- **Collective Anomalies:** Collections of related data points that are individually normal but form an unusual pattern together.

## Key Concepts Overview
Anomaly detection is a critical capability in machine learning that identifies unusual patterns, outliers, or deviations from normal behavior in data. This is foundational for numerous real-world applications including fraud detection in financial systems, network intrusion detection, manufacturing quality control, and healthcare monitoring. The challenge lies in defining what "normal" means, handling extreme class imbalance where anomalies are rare, and adapting to evolving patterns in streaming environments. Understanding both statistical baselines and modern deep learning approaches positions candidates to handle diverse anomaly scenarios effectively.

---

### Q1: What is the difference between point anomalies, contextual anomalies, and collective anomalies? Provide a practical example for each.

**A:** Point anomalies are individual data instances that deviate from the overall dataset pattern—for example, a single credit card transaction of $10,000 when typical transactions are $50-200. Contextual anomalies are anomalous only within a specific context; a $200 transaction at 3 AM might be anomalous for a user who typically shops during business hours, yet normal in other contexts. Collective anomalies involve groups of related data points that are individually normal but collectively unusual; for instance, a series of small transactions totaling $50,000 across many different vendors in one day could indicate coordinated fraud despite each transaction appearing individually benign. Understanding these distinctions allows you to choose appropriate detection methods—point anomalies often respond to statistical tests, contextual anomalies require sequential or time-based features, and collective anomalies need to analyze relationships between multiple observations.

---

### Q2: Compare supervised, unsupervised, and semi-supervised approaches to anomaly detection. When would you use each?

**A:** Supervised anomaly detection trains a classifier on labeled normal and anomalous examples, achieving high accuracy when quality labeled data exists but requiring expensive labeling effort. Unsupervised approaches like Isolation Forest or One-Class SVM require no labels and assume anomalies are rare or distinct, making them practical for real-world scenarios where labeled data is scarce or expensive. Semi-supervised methods train on predominantly normal examples with a small set of known anomalies, leveraging both assumptions that anomalies are rare and some ground truth knowledge. Use supervised methods in mature, well-instrumented systems with historical anomaly records (fraud teams with years of labeled cases); unsupervised when anomalies are genuinely novel and cannot be pre-labeled (zero-day network attacks); and semi-supervised as a practical middle ground where you have a few known examples to guide the model.

---

### Q3: Explain how Z-score and Grubbs test work for anomaly detection. What are their limitations?

**A:** The Z-score method identifies anomalies by computing z = (x - μ) / σ, where points with |z| > 3 (or other threshold) are flagged as anomalous, assuming data follows a normal distribution. Grubbs test is a statistical hypothesis test that formally tests whether the extreme value in a dataset is an outlier by comparing a test statistic to a critical value, removing flagged outliers iteratively. Both methods are simple, interpretable, and computationally efficient, making them suitable for univariate or low-dimensional data with approximately normal distributions. However, they fail on multivariate data, skewed distributions, or when anomalies cluster together; they also assume a fixed normal distribution rather than adapting to local density variations. In practice, these statistical baselines work well as preprocessing steps but are often outperformed by modern techniques on complex, high-dimensional, or non-Gaussian data.

---

### Q4: What is the Mahalanobis distance and how does it improve upon Euclidean distance for anomaly detection?

**A:** Mahalanobis distance accounts for correlations between variables and different scales by computing d = √((x - μ)^T Σ^-1 (x - μ)), where Σ is the covariance matrix, rather than treating all dimensions equally as Euclidean distance does. This is particularly valuable in anomaly detection because it recognizes that deviations along correlated directions may be normal while orthogonal deviations are unusual. For example, in a dataset where height and weight are naturally correlated, a person with unusual height-to-weight ratio would be detected as anomalous even if both individual measurements are within normal ranges. However, Mahalanobis distance assumes multivariate normality, requires computing and inverting a covariance matrix (which becomes numerically unstable in high dimensions), and is sensitive to outliers in the training set used to estimate Σ. It works best on lower-dimensional data with clear covariance structure; for high-dimensional or non-Gaussian data, modern methods like Isolation Forest are more robust.

---

### Q5: Explain the Isolation Forest algorithm. How does it differ from density-based approaches?

**A:** Isolation Forest builds an ensemble of random binary trees by recursively partitioning the feature space with random splits, with the key insight that anomalies are isolated in fewer splits than normal points (shorter path lengths to isolation). The anomaly score is computed as the average path length across trees: anomalies have shorter paths and thus higher scores. Unlike density-based approaches (LOF, DBSCAN) that explicitly compute local density and flag low-density regions, Isolation Forest implicitly uses the principle that anomalies are more easily separated, making it inherently scalable and not requiring density estimation. Isolation Forest has O(n log n) complexity and handles high-dimensional data naturally without distance metric computation. However, it assumes anomalies are sparse and isolated; it can underperform on datasets where anomalies cluster together or occupy similar regions in the feature space, and its hyperparameter tuning (number of trees, subsampling size) requires careful consideration.

---

### Q6: What are the key hyperparameters of Isolation Forest and how do you tune them?

**A:** The main hyperparameters are n_estimators (number of trees; typically 100-1000, higher values improve stability at computational cost), max_samples (subsample size for each tree; smaller values increase isolation speed and work well for large datasets), and contamination (expected proportion of anomalies; directly affects the decision threshold). n_estimators should be increased until performance plateaus; max_samples is often set to min(256, n) as a practical default since random subsampling inherently helps isolation. Contamination is the most impactful: if you overestimate contamination, normal points get flagged; underestimation misses anomalies. Tune it via validation on labeled holdout data if available, or use domain knowledge about anomaly prevalence. Additionally, max_features and max_depth can be tuned: smaller max_features reduce feature dimensionality per split, improving efficiency on high-dimensional data, while max_depth controls tree growth. In practice, start with defaults and adjust based on cross-validation recall/precision tradeoffs or business metrics like fraud catch rate.

---

### Q7: Explain Local Outlier Factor (LOF). What does it detect that other methods might miss?

**A:** Local Outlier Factor computes an anomaly score for each point as the ratio of the average local density of its k-nearest neighbors to its own local density, with scores near 1 indicating normal points and scores >> 1 indicating anomalies. Unlike global methods that compute a single density threshold, LOF detects local density anomalies—regions of lower density relative to their neighborhoods, making it effective at identifying contextual anomalies where global density varies. For example, in geographical data, a city location might appear normal globally but anomalous relative to neighboring rural areas. LOF naturally handles variable-density clusters and is robust to outliers in distance computation. However, it is computationally expensive (O(n²) in worst case due to k-nearest neighbor queries), sensitive to the choice of k, and can be noisy in very high dimensions where distances become less informative. It works best on lower-dimensional data with clear local structure and when you expect anomalies to deviate from local patterns rather than global patterns.

---

### Q8: What is One-Class SVM and how is it different from binary SVM?

**A:** One-Class SVM learns a boundary around normal data by finding the smallest hypersphere (or hyperplane in a transformed space) that contains most training examples, treating all training data as a single "normal" class. It solves the optimization problem of maximizing margin while minimizing the volume of the boundary, controlled by a parameter ν that bounds the expected fraction of support vectors and anomalies. Unlike binary SVM which learns a decision boundary between two classes, One-Class SVM assumes only normal examples are available and aims to create a tight boundary; points falling outside this boundary are flagged as anomalies. One-Class SVM handles high-dimensional data well, supports non-linear kernels (RBF, polynomial) for flexible boundaries, and is theoretically grounded in margin maximization. However, it can be slow on large datasets (quadratic complexity), sensitive to kernel choice and scale parameters (gamma, C/nu), and may struggle if normal data itself has complex, multi-modal structure. It's particularly useful when anomalies are fundamentally different from normal data and you want a theoretically justified approach.

---

### Q9: How do autoencoders detect anomalies using reconstruction error? What are the assumptions and limitations?

**A:** Autoencoders are neural networks trained to reconstruct input data through a bottleneck, with the assumption that normal data can be reconstructed with low error while anomalies (unseen during training) will have high reconstruction error. The anomaly score is simply the reconstruction loss (MSE, MAE, or other metric) between input and output; points with error exceeding a threshold are flagged as anomalies. This approach naturally handles high-dimensional data, learns nonlinear patterns, and requires no labels, making it practical for complex domains like images or time series. The key assumption is that the model has seen diverse examples of normal data and generalizes to reconstruct new normal instances well. Limitations include: (1) if anomalies occur frequently during training, the model may learn to reconstruct them well, reducing detection power; (2) threshold selection is non-trivial and often requires labeled validation data; (3) models may overfit on small datasets, giving artificially low reconstruction errors; and (4) the method struggles with high-variance normal data (legitimate variation that isn't anomalous) which also produces high reconstruction error. In practice, careful architecture design and threshold calibration on holdout data are essential.

---

### Q10: Explain Variational Autoencoders (VAEs) for anomaly detection. How do they differ from standard autoencoders?

**A:** VAEs add a probabilistic interpretation by learning to generate a distribution over latent representations rather than point encodings, optimizing a loss combining reconstruction error and KL divergence from a prior distribution N(0, I). This regularization forces the latent space to be smooth and follow a known distribution, so anomalies that don't fit the learned manifold have high reconstruction error. VAEs explicitly model data distribution, enabling principled density estimation; anomalies are samples with low probability under the learned model. Unlike standard autoencoders which may produce arbitrary latent encodings, VAEs learn a continuous latent space where nearby points generate similar outputs, improving generalization. Anomaly detection uses the ELBO (Evidence Lower Bound) or reconstruction error as the anomaly score. However, VAEs are more complex to train (requiring careful hyperparameter tuning of KL weight), may still fail if anomalies occur frequently in training data, and the smooth latent space sometimes reconstructs anomalies well if they lie on the data manifold. VAEs shine when you need both anomaly detection and generative modeling capabilities, or when a principled probabilistic framework is important for downstream decision-making.

---

### Q11: Describe time series anomaly detection methods including STL decomposition, Prophet, and LSTM-based approaches.

**A:** STL (Seasonal and Trend decomposition using Loess) separates time series into trend, seasonal, and residual components; anomalies appear as large residuals, allowing you to detect deviations from expected seasonal and trend patterns. Facebook's Prophet models time series with trend and seasonality components plus additive regressors, providing uncertainty intervals—points falling far outside intervals are flagged as anomalies, handling seasonality and changepoints naturally. LSTM-based approaches train a sequence-to-sequence model to predict the next value given historical context, with high prediction error indicating anomalies; this captures complex temporal dependencies better than linear methods. STL is simple and interpretable, working well on clearly seasonal data with known periodicity, but struggles with abrupt changes or multiple seasonalities. Prophet is robust to missing data and changepoints, requiring minimal tuning but being less flexible for non-standard patterns. LSTM methods are flexible and can detect complex temporal anomalies but require substantial training data, are computationally expensive, and need careful threshold tuning. Choice depends on data characteristics: use STL for clean periodic data, Prophet for business time series with known seasonality, and LSTMs for complex temporal patterns in sufficient data.

---

### Q12: How do you evaluate anomaly detection models? Discuss precision-recall tradeoff, F1 at different thresholds, and AUC-PR.

**A:** Unlike balanced classification, anomaly detection evaluation emphasizes precision and recall tradeoffs because the cost of false positives and false negatives differs dramatically—false positive fraud alerts annoy customers, while false negatives cause financial loss. Precision (TP / (TP + FP)) measures what fraction of flagged anomalies are truly anomalous; recall (TP / (TP + FN)) measures what fraction of true anomalies are detected. F1 score balances precision and recall as 2 * (precision * recall) / (precision + recall), but it weights both equally; you can adjust via F-beta scores to prioritize recall or precision based on business costs. Precision-Recall curves plot recall vs precision across different decision thresholds (probability cutoffs), providing insight into the tradeoff space; AUC-PR (Area Under the Precision-Recall curve) summarizes this into a single metric, handling class imbalance much better than ROC-AUC which becomes uninformative when anomalies are rare. In practice, choose threshold by: (1) using domain knowledge of tolerable false positive/negative rates, (2) computing cost-benefit analysis (cost_FP * FPR + cost_FN * FNR), or (3) optimizing F-beta for appropriate beta. Always validate on held-out test data and report metrics at your chosen operating point, not just overall AUC.

---

### Q13: How do you handle extreme class imbalance in anomaly detection? What techniques can help?

**A:** Extreme class imbalance (anomalies << 1% of data) poses several challenges: simple classifiers achieve high accuracy by predicting all as normal, standard cross-validation can be misleading, and many algorithms fail. Techniques include: (1) adjusting decision thresholds or class weights (weight_anomaly >> weight_normal) rather than optimizing for accuracy; (2) using anomaly detection methods (Isolation Forest, LOF, autoencoders) designed for imbalance rather than balanced classifiers; (3) stratified sampling or separate validation on anomaly-enriched subsets to properly estimate model performance; (4) resampling strategies like oversampling anomalies (random duplication or SMOTE) or undersampling normal data, though these modify data distribution; and (5) cost-sensitive learning where misclassifying anomalies incurs higher loss. Prefer anomaly-aware evaluation metrics (precision-recall, F-beta, AUC-PR, cost curves) over accuracy. In production, combine multiple signals via ensemble methods and implement human-in-the-loop workflows where low-confidence predictions are reviewed by domain experts. The key insight is that standard machine learning assumptions break down—focus on detecting rare anomalies, not overall accuracy.

---

### Q14: What feature engineering techniques improve anomaly detection? Provide examples for different data types.

**A:** Feature engineering for anomaly detection aims to make anomalies more distinct and easier to separate from normal patterns. For time series, compute features like rolling mean, volatility, rate of change, and autocorrelation differences to capture temporal patterns; a sudden jump in volatility or break in autocorrelation can signal anomalies. For tabular data, create interaction features (e.g., transaction_amount / account_balance) and domain-specific ratios (fraud score indicators like transaction_frequency * transaction_amount), normalize features to comparable scales, and include temporal context (time of day, day of week effects). For graph/network data, compute node centrality, clustering coefficients, and community detection features; unusual connection patterns become visible through degree anomalies or bridge node behavior. Domain knowledge is critical—involve subject matter experts to understand what constitutes anomalous behavior. Dimensionality reduction (PCA, autoencoders) can also improve anomaly detection by denoising or highlighting principal variation patterns. The general principle: design features that either make anomalies stand out (high variance in anomalies vs. low variance in normal) or emphasize relationships that break down for anomalies (high correlation among features for normal, decorrelated for anomalies).

---

### Q15: Describe ensemble approaches for anomaly detection and real-world applications (fraud, intrusion, manufacturing).

**A:** Ensemble anomaly detection combines multiple diverse models (Isolation Forest, LOF, One-Class SVM, autoencoders) via voting, averaging scores, or stacking, improving robustness and catching anomalies different methods might miss. For fraud detection, combine rule-based detection (known fraud patterns), statistical methods (transaction velocity), and neural networks (behavior patterns); this catches both signature fraud and novel behavior deviations. Network intrusion detection uses ensemble of density-based methods (LOF for unusual traffic patterns), time series models (protocol behavior over time), and graph-based methods (unusual connection graphs), coordinated to flag multi-stage attacks. Manufacturing defects combine point anomalies (sensor readings out of spec) via statistical methods, collective anomalies (patterns of slow degradation) via LSTM/STL, and contextual anomalies (normal sensor readings but wrong relationships) via multivariate methods like Mahalanobis distance. Streaming/online variants use evolving ensembles where models are continuously updated as new normal patterns emerge, with drift detection to retrain when distributions shift. In production, layer multiple signals: statistical anomaly scores feed alerts, human review confirms true positives, and confirmed anomalies improve ground truth for model retraining, creating a continuous feedback loop.

---

## Interview Cheatsheet

**Key Terms:**
- **Point Anomaly:** A single instance that deviates from the overall dataset.
- **Contextual Anomaly:** An instance anomalous within a specific context but normal elsewhere.
- **Collective Anomaly:** A group of instances individually normal but collectively unusual.
- **Reconstruction Error:** Difference between original input and autoencoder output, used as anomaly score.
- **Local Density:** Density of points in a local neighborhood; low density indicates anomalies in LOF.
- **Contamination:** Expected proportion of anomalies in the dataset; affects decision thresholds.
- **AUC-PR:** Area under the Precision-Recall curve; preferred metric for imbalanced data.
- **Isolation Path Length:** Number of splits to isolate a point in Isolation Forest; shorter paths = higher anomaly scores.

**Rapid-Fire Q&A:**
- **Q:** Why is Euclidean distance problematic for anomaly detection in correlated data? **A:** It treats all dimensions equally and ignores correlations; Mahalanobis distance corrects this by weighting deviations inversely by variance and covariance.
- **Q:** When would you choose Isolation Forest over LOF? **A:** For high-dimensional data, large datasets, or when anomalies are sparse and isolated; LOF is better for detecting local density anomalies.
- **Q:** What threshold should you use for autoencoder anomaly scores? **A:** Validate on held-out labeled data by computing precision-recall across thresholds, then choose based on business cost of false positives vs. false negatives.
- **Q:** How do you handle seasonal normal patterns in time series anomaly detection? **A:** Use STL decomposition to remove seasonality, or employ Prophet which models seasonal components explicitly.
- **Q:** Why do class weights matter in One-Class SVM? **A:** The ν parameter directly controls the fraction of support vectors and tolerated anomalies; smaller ν creates tighter boundaries but risks false positives on normal variations.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# Multiple Choice Questions: Anomaly Detection

📺 **Video Lecture:** https://youtu.be/wOb5Il6zMqI


## Question 1
A bank detects that a customer's credit card was used for a single transaction of $15,000 when their typical transactions range from $50-300. Which type of anomaly is this?

A) Contextual anomaly  
B) Point anomaly  
C) Collective anomaly  
D) Behavioral anomaly

---

## Question 2
You have a dataset of network traffic where a series of small data transfers (each individually normal) from multiple internal IPs to a single external server in one hour is detected as suspicious. This represents:

A) Point anomaly  
B) Contextual anomaly  
C) Collective anomaly  
D) Statistical anomaly

---

## Question 3
Which anomaly detection approach would you choose when you have 10,000 labeled examples of fraud and 2,000,000 labeled examples of legitimate transactions?

A) Unsupervised methods like Isolation Forest  
B) Supervised classification methods  
C) Semi-supervised methods  
D) Statistical methods like Z-score

---

## Question 4
The Z-score method identifies anomalies by computing z = (x - μ) / σ. Which of the following is a key limitation of this approach?

A) It is too computationally expensive  
B) It assumes a normal distribution and fails on multivariate data with skewed distributions  
C) It requires labeled training data  
D) It cannot be applied to real-time data

---

## Question 5
How does Mahalanobis distance improve upon Euclidean distance for anomaly detection?

A) It is faster to compute in high dimensions  
B) It accounts for correlations between variables and different scales  
C) It eliminates the need for covariance matrix estimation  
D) It requires no assumptions about data distribution

---

## Question 6
In Isolation Forest, what is the key insight that makes it effective for anomaly detection?

A) Anomalies have higher local density than normal points  
B) Anomalies require more splits to be isolated than normal points  
C) Anomalies can be isolated with fewer random splits than normal points  
D) Anomalies form dense clusters that are easy to identify

---

## Question 7
You are tuning an Isolation Forest model and need to set the `contamination` parameter. What does this parameter represent?

A) The number of trees in the ensemble  
B) The expected proportion of anomalies in the dataset  
C) The maximum depth of each tree  
D) The threshold for reconstruction error

---

## Question 8
Local Outlier Factor (LOF) is particularly effective at detecting which type of anomaly?

A) Global point anomalies where anomalous points deviate from the overall mean  
B) Collective anomalies in time series data  
C) Contextual anomalies where points are normal globally but anomalous relative to their local neighborhood  
D) Anomalies in high-dimensional data with >100 dimensions

---

## Question 9
What is the computational complexity of Local Outlier Factor in the worst case?

A) O(n log n)  
B) O(n)  
C) O(n²)  
D) O(n³)

---

## Question 10
One-Class SVM learns a boundary around normal data by finding the smallest hypersphere that contains most training examples. Which parameter controls the fraction of support vectors and tolerated anomalies?

A) gamma  
B) C  
C) ν (nu)  
D) epsilon

---

## Question 11
An autoencoder is trained on historical normal data for anomaly detection. What is the core assumption behind using reconstruction error as an anomaly score?

A) Normal data points will have high reconstruction error  
B) Anomalies are more compressible than normal data  
C) Normal data can be reconstructed with low error; anomalies (unseen during training) will have high reconstruction error  
D) The model will perfectly reconstruct all data points

---

## Question 12
How do Variational Autoencoders (VAEs) differ from standard autoencoders for anomaly detection?

A) VAEs are faster to train  
B) VAEs add probabilistic interpretation by learning a distribution over latent representations and regularizing with KL divergence  
C) VAEs eliminate the need for threshold selection  
D) VAEs work only on time series data

---

## Question 13
For time series anomaly detection, which method explicitly models trend, seasonality, and residual components?

A) One-Class SVM  
B) Isolation Forest  
C) STL (Seasonal and Trend decomposition using Loess)  
D) LSTM autoencoders

---

## Question 14
In anomaly detection evaluation, why is AUC-PR (Area Under the Precision-Recall curve) preferred over ROC-AUC when anomalies are rare?

A) It is faster to compute  
B) It handles extreme class imbalance better and is more informative when anomalies represent <1% of data  
C) It does not require labeled test data  
D) It automatically selects the optimal threshold

---

## Question 15
Which combination of techniques would be most effective for handling extreme class imbalance in anomaly detection?

A) Use standard binary classification with balanced datasets  
B) Adjust class weights, use anomaly-specific algorithms, employ cost-sensitive learning, and validate with stratified sampling  
C) Oversample all anomalies using SMOTE regardless of training distribution changes  
D) Report accuracy as the primary evaluation metric

---

## Answer Key

**Question 1: B) Point anomaly**
A single transaction that deviates significantly from the customer's normal pattern is a point anomaly. This is the classic example of an individual data instance that is anomalous.

**Question 2: C) Collective anomaly**
Individual transactions are normal, but the collective pattern of multiple transfers to a single external server is anomalous. This defines a collective anomaly where the group exhibits unusual patterns.

**Question 3: B) Supervised classification methods**
With abundant labeled examples of both classes, supervised methods are appropriate and will likely outperform unsupervised approaches. This is a mature, well-instrumented system scenario.

**Question 4: B) It assumes a normal distribution and fails on multivariate data with skewed distributions**
Z-score is fundamentally limited to univariate data with normal distributions. It cannot handle multivariate data effectively or non-Gaussian distributions.

**Question 5: B) It accounts for correlations between variables and different scales**
Mahalanobis distance uses the covariance matrix to account for correlations and scale differences, making it superior for detecting anomalies in correlated multivariate data.

**Question 6: C) Anomalies can be isolated with fewer random splits than normal points**
The core insight of Isolation Forest is that anomalies, being rare and distinct, are easier to isolate and thus require fewer splits in random binary trees.

**Question 7: B) The expected proportion of anomalies in the dataset**
The contamination parameter directly specifies the expected fraction of anomalies and determines the decision threshold for flagging anomalies.

**Question 8: C) Contextual anomalies where points are normal globally but anomalous relative to their local neighborhood**
LOF explicitly computes local density ratios, making it highly effective at detecting contextual anomalies that depend on local neighborhood patterns rather than global statistics.

**Question 9: C) O(n²)**
LOF requires computing k-nearest neighbors for all points, which is O(n²) in the worst case, making it computationally expensive on large datasets.

**Question 10: C) ν (nu)**
The ν parameter in One-Class SVM directly bounds the expected fraction of support vectors and anomalies, controlling the tightness of the learned boundary.

**Question 11: C) Normal data can be reconstructed with low error; anomalies (unseen during training) will have high reconstruction error**
This is the fundamental assumption: the model learns the manifold of normal data and reconstructs it well, while novel anomalies deviate from this learned manifold and produce high reconstruction error.

**Question 12: B) VAEs add probabilistic interpretation by learning a distribution over latent representations and regularizing with KL divergence**
VAEs differ by adding probabilistic structure with KL divergence regularization, enabling principled density estimation and smoother latent spaces compared to standard autoencoders.

**Question 13: C) STL (Seasonal and Trend decomposition using Loess)**
STL explicitly decomposes time series into trend, seasonal, and residual components, with anomalies appearing as large residuals. This is its defining characteristic.

**Question 14: B) It handles extreme class imbalance better and is more informative when anomalies represent <1% of data**
When anomalies are rare, ROC-AUC becomes uninformative because the false positive rate dominates. Precision-Recall curves and AUC-PR directly address the imbalance problem.

**Question 15: B) Adjust class weights, use anomaly-specific algorithms, employ cost-sensitive learning, and validate with stratified sampling**
This combination addresses class imbalance comprehensively: algorithm selection (Isolation Forest, LOF), hyperparameter adjustment (weights), evaluation methodology (stratified sampling), and business-aligned optimization (cost-sensitive learning).

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

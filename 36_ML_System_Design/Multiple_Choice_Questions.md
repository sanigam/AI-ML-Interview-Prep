# Multiple Choice Questions: ML System Design

📺 **Video Lecture:** https://youtu.be/7bjOx8U1AyY


---

## Question 1

What is the primary purpose of a feature store in an ML system?

A) To store raw datasets before preprocessing  
B) To prevent training-serving skew by managing feature computation for both training and serving  
C) To cache prediction results for faster inference  
D) To track model performance metrics over time

---

## Question 2

In an ML system, when would you use offline features instead of online features?

A) Offline features are never used; online features are always preferable  
B) Offline features are used only during model evaluation, not training  
C) Offline features are computed from historical data during training with high latency tolerance  
D) Offline features are used exclusively for real-time serving to reduce memory usage

---

## Question 3

You're designing a fraud detection system that must make decisions in <500ms. Which latency-critical components are most important?

A) Training the model as accurately as possible, even if it takes days  
B) Real-time feature computation, efficient model serving, and precomputed historical features  
C) Storing all historical data in memory for instant access  
D) Using the most complex ensemble model for maximum accuracy

---

## Question 4

What is the key difference between batch serving and real-time serving?

A) Batch serving is always more accurate than real-time serving  
B) Batch serving precomputes predictions and stores them; real-time serving answers requests synchronously  
C) Real-time serving is cheaper because it requires less computational infrastructure  
D) There is no practical difference between them in modern ML systems

---

## Question 5

In a recommendation system with millions of users and items, why is a two-stage architecture (candidate generation + ranking) used instead of directly ranking all items?

A) To ensure all items are considered regardless of computational cost  
B) To retrieve thousands of plausible items efficiently first, then rank them with a complex model  
C) Two-stage architecture is outdated; modern systems rank all items simultaneously  
D) To avoid using feature stores or vector databases

---

## Question 6

How does data distribution shift differ from concept drift?

A) They are the same thing with different names  
B) Data distribution shift is when feature distributions change; concept drift is when the relationship between features and labels changes  
C) Data distribution shift only affects training; concept drift only affects serving  
D) Concept drift refers to model versions, while data distribution shift refers to user behavior

---

## Question 7

What makes A/B testing critical in production ML systems even if a model has better offline metrics?

A) Offline metrics always perfectly predict real-world impact  
B) A/B testing is only necessary for research purposes  
C) Offline metrics like accuracy may not correlate with business metrics like engagement or revenue  
D) A/B testing is used only to validate that the system is running, not to measure improvements

---

## Question 8

In a training pipeline handling billions of examples, what does model parallelism address?

A) Distributing data across multiple machines for faster training  
B) Reducing the number of epochs needed for convergence  
C) Splitting model layers across machines when the model is too large for a single device  
D) Ensuring all GPUs have identical model weights at all times

---

## Question 9

You notice that a model's performance has degraded in production even though the training code hasn't changed. What should you investigate first?

A) Replace the entire model immediately  
B) Monitor feature statistics and prediction distributions for drift, indicating data distribution shift or concept drift  
C) Assume the data quality team made a mistake and request a full data audit  
D) Increase model complexity to compensate for accuracy loss

---

## Question 10

In designing an end-to-end ML system, what is the correct sequence of steps?

A) Select a model architecture, train on all available data, deploy to production  
B) Clarify business requirements, understand data, design features, train models, evaluate, design serving, implement monitoring  
C) Collect as much data as possible, then figure out what problem to solve  
D) Optimize model accuracy first, then worry about latency and cost

---

## Question 11

How would you address a true positive rate of only 30% in a fraud detection system while maintaining low false positives?

A) Ignore the issue; fraud detection systems are inherently inaccurate  
B) Use weighted loss to increase fraud case importance, collect more labeled fraud data, and adjust decision thresholds  
C) Replace the model with a simpler one that catches fewer frauds but with fewer false positives  
D) Ensure the model only flags transactions of extremely high value as fraudulent

---

## Question 12

Pipeline parallelism differs from data and model parallelism in which key way?

A) It only works with recurrent neural networks  
B) It interleaves data parallelism with model parallelism, allowing overlapped computation across pipeline stages  
C) It's an older technique that has been entirely replaced by other methods  
D) It requires more GPUs than either data or model parallelism alone

---

## Question 13

When designing a search ranking system with <100ms latency, why would you precompute expensive features offline rather than computing them during serving?

A) Offline computation produces more accurate results  
B) Online serving requests have strict latency budgets, so expensive features must be precomputed and retrieved from memory or fast storage  
C) Precomputation is always cheaper than online computation  
D) Modern hardware makes online computation fast enough, so precomputation is unnecessary

---

## Question 14

In a content moderation system, why is a two-stage architecture (automated filtering + human review) preferable to pure automated or pure human moderation?

A) It eliminates the need for human review entirely  
B) Automated filtering catches obvious violations cheaply and quickly, while human review handles uncertain cases and provides labels for retraining  
C) It's more expensive than pure automation but guarantees zero false positives  
D) The two stages process independent content types with no feedback loop

---

## Question 15

How do feedback loops enable continuous improvement in ML systems?

A) They eliminate the need for manual retraining and model updates  
B) They collect ground truth labels on predictions and periodically retrain models, improving performance over time as the system learns from real-world outcomes  
C) They only apply to recommender systems, not other ML applications  
D) Feedback loops are optional features that provide marginal improvements

---

# Answer Key

**Question 1: B**  
A feature store solves the critical problem of training-serving skew by managing feature computation in a centralized way for both offline (training) and online (serving) use cases. This ensures consistency across the pipeline.

**Question 2: C**  
Offline features are computed from historical data during training and can afford high latency because training happens periodically in batch. They enable expensive aggregations like "user's entire purchase history over 2 years" which wouldn't be feasible in real-time serving.

**Question 3: B**  
With a strict 500ms latency budget, the system needs precomputed historical features (stored in a feature store), efficient model serving infrastructure, and real-time feature computation. Training accuracy is important but not the bottleneck—serving speed is.

**Question 4: B**  
Batch serving precomputes predictions periodically and stores results for lookup, offering high throughput and simplicity but high staleness. Real-time serving answers requests synchronously with low staleness but higher complexity and infrastructure needs.

**Question 5: B**  
With millions of items, ranking all of them with a complex model would be too slow. Candidate generation efficiently retrieves thousands of plausible items (using embedding similarity), then ranking applies expensive models to only those candidates.

**Question 6: B**  
Data distribution shift means the distribution of features or labels has changed (e.g., users have different spending patterns). Concept drift means the relationship between features and labels has changed (e.g., the same behavior predicts fraud in one context but not another). Both degrade model performance but are different phenomena.

**Question 7: C**  
Offline metrics like accuracy and AUC don't guarantee business impact. A model with higher accuracy might recommend diverse items that users skip, or have higher latency that reduces engagement. A/B testing validates that improvements in offline metrics translate to real business gains.

**Question 8: C**  
Model parallelism splits the neural network across devices when it's too large for one device. Data parallelism distributes data; model parallelism distributes the model architecture itself.

**Question 9: B**  
When model performance degrades without code changes, the most likely cause is data distribution shift or concept drift. Monitor feature statistics and prediction distributions to detect these changes, then trigger retraining.

**Question 10: B**  
The correct ML system design flow is: clarify requirements (target metric, latency, throughput) → understand data (quality, volume, labels) → feature engineering → model training → offline evaluation → serving design → monitoring and feedback loops.

**Question 11: B**  
With low true positive rate and class imbalance, use weighted loss (fraud cases weighted higher), collect more labeled fraud data, and adjust decision thresholds. This improves recall for fraud detection while controlling false positive rate.

**Question 12: B**  
Pipeline parallelism interleaves data and model parallelism: different transformer layers (model) run on different GPUs while processing different batches (data), enabling better hardware utilization than pure data or model parallelism.

**Question 13: B**  
Online serving requests have strict latency budgets (typically <100ms total). Expensive features like PageRank or user quality scores must be precomputed offline and retrieved from memory during serving. Computing them online would exceed latency budgets.

**Question 14: B**  
Automated filtering quickly catches obvious violations and is cheap to run at scale. Uncertain cases go to human review, which provides high-quality labels for retraining the automated model. This hybrid approach balances cost, speed, accuracy, and feedback.

**Question 15: B**  
Feedback loops collect ground truth labels (implicit from user behavior or explicit from human review) and periodically retrain models on fresh data. Over time, the system learns from real-world outcomes and improves without manual intervention, though retraining still requires human scheduling or automation.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

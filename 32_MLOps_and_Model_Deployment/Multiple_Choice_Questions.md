# Multiple Choice Questions: MLOps and Model Deployment

Test your understanding of MLOps and model deployment for AI/ML interviews.

---

**Q1. The ML lifecycle includes all of the following EXCEPT:**

A) Data collection and cleaning
B) Model training and evaluation
C) Deployment and monitoring
D) Manufacturing hardware chips

---

**Q2. Experiment tracking tools like MLflow help by:**

A) Automatically improving model accuracy
B) Recording hyperparameters, metrics, code versions, and artifacts for reproducibility and comparison
C) Replacing the need for training data
D) Only storing final model weights

---

**Q3. A model registry differs from simple model versioning by providing:**

A) Only file storage
B) Centralized management with stage transitions (staging → production), metadata, lineage, and governance
C) Automatic model improvement
D) Only version numbers

---

**Q4. Data drift in production occurs when:**

A) The model gets better over time
B) The distribution of incoming data shifts from the training data distribution, potentially degrading model performance
C) The model is retrained daily
D) Users stop using the model

---

**Q5. Concept drift differs from data drift because:**

A) They are identical
B) Concept drift means the relationship between features and target changes, not just the feature distribution
C) Concept drift only affects images
D) Data drift is always worse

---

**Q6. A/B testing for model deployment involves:**

A) Deploying only the new model
B) Running old and new models simultaneously on different user segments to compare real-world performance
C) Only using offline metrics
D) Deploying to zero users

---

**Q7. Canary deployment for ML models means:**

A) Deploying to all users at once
B) Gradually rolling out the new model to a small percentage of traffic first, monitoring for issues before full rollout
C) Deploying only to internal users
D) Rolling back automatically without monitoring

---

**Q8. Feature stores address the problem of:**

A) Storing model weights
B) Providing consistent, reusable feature definitions across training and serving to prevent train-serve skew
C) Only generating new features
D) Replacing the model registry

---

**Q9. CI/CD for ML (MLOps pipelines) automates:**

A) Only code testing
B) The entire pipeline from data validation through training, evaluation, and deployment with automated quality gates
C) Only model deployment
D) Only data collection

---

**Q10. Model monitoring in production should track:**

A) Only inference latency
B) Prediction quality metrics, data distribution, latency, error rates, and business KPIs
C) Only the number of API calls
D) Only model size

---

**Q11. Shadow deployment runs the new model:**

A) Instead of the old model
B) Alongside the old model on real traffic but only the old model's predictions are served to users
C) Only on synthetic data
D) Without any logging

---

**Q12. Train-serve skew refers to:**

A) The model training being too slow
B) Differences between the training environment and serving environment that cause the model to perform differently in production
C) The serving infrastructure being too expensive
D) Users preferring older models

---

**Q13. Model quantization for deployment:**

A) Increases model accuracy
B) Reduces model size and inference time by converting weights from float32 to lower precision (int8, float16)
C) Removes all model parameters
D) Only works for NLP models

---

**Q14. DVC (Data Version Control) helps ML teams by:**

A) Only tracking code changes
B) Versioning datasets and model artifacts alongside code, enabling reproducible experiments
C) Replacing Git entirely
D) Only storing images

---

**Q15. When should you retrain a model in production?**

A) Never after initial deployment
B) When monitoring detects significant data drift, concept drift, or performance degradation on business metrics
C) Every hour regardless of performance
D) Only when the code changes

---

## Answer Key

**Q1. Answer: D**
The ML lifecycle covers problem definition, data preparation, training, evaluation, deployment, monitoring, and retraining — all software and data processes, not hardware manufacturing.

**Q2. Answer: B**
Experiment tracking records all metadata about training runs, enabling teams to compare experiments, reproduce results, and understand which hyperparameters led to the best performance.

**Q3. Answer: B**
A model registry adds governance: stage management (dev → staging → production), metadata, lineage tracking, annotations, and API access, going beyond simple file versioning.

**Q4. Answer: B**
Data drift means the production data distribution differs from training data. For example, a model trained on summer data may perform poorly in winter if seasonal patterns weren't captured.

**Q5. Answer: B**
Concept drift means the mapping from inputs to outputs changes (e.g., customer preferences shift). Data drift means input distribution changes. Both degrade performance but require different responses.

**Q6. Answer: B**
A/B testing splits traffic between models, measuring real-world metrics (conversion, engagement) to determine which model performs better before committing to the new one.

**Q7. Answer: B**
Canary deployment exposes the new model to a small fraction of traffic (e.g., 5%), monitoring key metrics. If metrics look good, traffic is gradually increased; if not, the canary is rolled back.

**Q8. Answer: B**
Feature stores serve the same feature definitions during training and inference, ensuring features computed offline (training) match those computed in real-time (serving), preventing train-serve skew.

**Q9. Answer: B**
ML CI/CD automates data validation, model training, evaluation against thresholds, and deployment — with automated gates that prevent deploying models that don't meet quality standards.

**Q10. Answer: B**
Comprehensive monitoring tracks prediction quality (accuracy, drift), system health (latency, errors), and business impact (revenue, engagement), enabling early detection of degradation.

**Q11. Answer: B**
Shadow deployment logs new model predictions alongside old model predictions on real traffic. Operators compare results without risk, since only the old model's predictions affect users.

**Q12. Answer: B**
Train-serve skew arises from differences in feature computation, preprocessing, or data access between training and serving environments, causing models to behave differently in production.

**Q13. Answer: B**
Quantization converts model weights to lower precision (e.g., float32 → int8), reducing memory by 2-4x and often improving inference speed with minimal accuracy loss.

**Q14. Answer: B**
DVC integrates with Git to version large datasets and model files, storing pointers in Git while keeping actual data in cloud storage, enabling reproducible ML experiments.

**Q15. Answer: B**
Retraining should be triggered by monitoring signals: significant data drift, performance degradation, or when business metrics indicate the model is no longer meeting requirements.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

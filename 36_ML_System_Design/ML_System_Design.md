# ML System Design

📺 **Video Lecture:** https://youtu.be/7bjOx8U1AyY


## Interview Anchor
- **End-to-End ML Lifecycle:** Complete workflow from requirements through deployment including data, features, model training, evaluation, and monitoring
- **Distributed Systems at Scale:** Handling large-scale data and model serving across multiple machines, managing latency and throughput tradeoffs
- **Production Constraints:** Real-world considerations like latency budgets, cost optimization, reliability, and continuous learning from feedback

## Key Concepts Overview
ML system design is fundamentally different from model optimization because it requires balancing multiple competing objectives: accuracy, latency, cost, and operational complexity. Interviewers test whether you understand the full pipeline beyond just training models—how data flows through the system, how features are computed efficiently, how models are served at scale, and how you monitor and improve systems in production. Strong candidates distinguish between online and offline requirements, discuss specific architectural patterns like feature stores and training pipelines, and explain practical decisions like batch vs real-time serving based on business constraints.

---

### Q1: Walk me through the end-to-end process of designing an ML system from scratch.

**A:** Start by clarifying business requirements: What problem are we solving? What's the target metric? What are latency and throughput constraints? Then move to data collection and understanding—what data do we have, what's its quality and volume, is it labeled? Next, design features: What raw signals matter? How do we compute them efficiently for both training and serving? Then select and train models, evaluating on offline metrics that correlate with business goals. Establish an evaluation strategy including holdout test sets and simulation. Finally, design the serving infrastructure (batch vs real-time), implement monitoring for data drift and model performance, and plan for feedback loops. Example: For a recommendation system, you'd clarify that low latency (<100ms) is critical, design feature computation that works both in training and at serving time, choose a model that balances accuracy with inference speed, and instrument monitoring to catch when recommendation diversity drops.

---

### Q2: Explain the difference between online and offline features and why this distinction matters.

**A:** Offline features are computed from historical data during training and stored for model building—they have high latency tolerance and can be expensive to compute (e.g., user's entire purchase history aggregated over 2 years). Online features are computed in real-time during serving with strict latency budgets (typically <100ms)—they must be fast and use only current or recent data (e.g., user's purchase count in the last 7 days). This distinction matters because what's feasible for training may be impossible for serving: aggregating all of a user's historical behavior works offline but would timeout during a serving request. Good systems ensure consistency between offline and online feature computation using tools like feature stores, documenting which features can be computed where. Example: In a fraud detection system, offline features might include "average transaction amount over past year," but the online serving pipeline can only access "transaction frequency in last hour" computed from a real-time stream.

---

### Q3: What is a feature store and how does it solve common challenges in ML systems?

**A:** A feature store is a centralized repository that manages feature computation, storage, and retrieval for both training and serving, ensuring consistency and reducing duplication. It addresses several critical problems: preventing training-serving skew (where features are computed differently offline vs online), avoiding duplicate feature engineering logic across teams, and enabling fast feature retrieval during serving. Feature stores manage both batch features (computed periodically and stored) and streaming features (computed in real-time), provide APIs for accessing features at serving time, and track feature lineage and versions. Example: With Tecton or Feast, you define a feature once as code, it automatically handles computation for both training (joining historical data) and serving (retrieving latest values), prevents accidental data leakage by managing time-based joins correctly, and logs which features were used in which models. Without a feature store, teams often rewrite feature logic separately for training pipelines and serving code, leading to subtle inconsistencies that degrade production model performance.

---

### Q4: Design a training pipeline architecture that can handle billions of training examples.

**A:** Use distributed batch processing: split data into shards, process each shard in parallel, aggregate statistics and gradients, update the model. Tools like Apache Spark or TensorFlow distributed training enable data parallelism where different machines process different data batches. For model parallelism, split the model across machines when it's too large for a single GPU. Implement checkpointing to save intermediate model states, allowing recovery from failures. Design the pipeline to separate data preprocessing (which can be cached) from model training. Use a scheduler (Kubernetes, Ray) to orchestrate these distributed jobs, and implement monitoring to detect stragglers (slow machines) that would otherwise block completion. For data engineering: use efficient file formats (Parquet, TFRecord), compress features, and cache preprocessed data. Example: TensorFlow's tf.distribute.Strategy automatically handles distributed training across GPUs/TPUs, and you write the same training loop once—the framework handles gradient aggregation across machines. A typical pipeline for billions of examples might use hourly batch jobs processing compressed Parquet files on a 100-node Spark cluster, with checkpoints saved every 10 minutes.

---

### Q5: Explain the architecture for serving ML models at scale (batch vs real-time serving).

**A:** Batch serving is appropriate for offline use cases where predictions can be computed periodically and stored (e.g., daily personalization scores for email campaigns). Models run on all data once per day, predictions are cached in a database, and applications simply look up precomputed results—this is high-throughput, low-latency at serving time, but has high staleness. Real-time serving answers prediction requests synchronously: a request comes in, features are computed, the model runs, and the result is returned within milliseconds. Real-time requires low-latency feature computation, efficient model inference (often using optimized frameworks like ONNX or TensorRT), and horizontal scaling via load balancing. Hybrid approaches use batch precomputation for some features and real-time computation for others. Example: Recommendation systems often combine batch serving for base recommendations (computed daily) with real-time serving for ranking (reorders recommendations based on current context). Netflix precomputes thousands of personalized lists overnight, but ranks them in real-time based on what a user is currently watching. Trade-off considerations: batch serves high throughput and is simpler operationally, while real-time enables freshness and personalization but adds latency and complexity.

---

### Q6: How do you design a recommendation system that scales to millions of users and items?

**A:** Start with collaborative filtering or content-based approaches, then add business logic. The architecture layers into: (1) candidate generation—retrieve thousands of possible items from a large pool using efficient approximation (e.g., embedding-based retrieval with approximate nearest neighbors), (2) ranking—score candidates using a more complex model to select top K items. For candidate generation, store user and item embeddings in a vector database (Pinecone, Weaviate, or FAISS) and query k-nearest neighbors; embeddings are precomputed offline from user behavior data. For ranking, use a neural network that combines multiple features (user's recent clicks, item popularity, diversity penalties), and optimize for business metrics beyond accuracy (e.g., diversity, serendipity, freshness). Implement real-time personalization by incorporating user's current session activity, but handle cold-start problems (new users or items without embeddings) with content-based or popularity-based fallbacks. Example: YouTube's recommendation system uses collaborative filtering for candidate generation (finding videos similar to user's history), then ranks with a deep neural network trained on watch time and engagement. Cache precomputed embeddings in memory for <10ms candidate retrieval, and implement A/B testing to validate that ranking improvements actually increase watch time.

---

### Q7: Design a search ranking system that serves results in <100ms latency.

**A:** A search ranking system typically has two stages: retrieval (find relevant documents) and ranking (order them by relevance). For retrieval, use an inverted index (built with tools like Elasticsearch or Solr) that maps terms to documents, enabling sub-millisecond matching against billions of documents. For ranking, first use retrieval scores as a baseline, then apply learned-to-rank models on top 100-1000 candidates to identify the best results. To meet strict latency budgets: (1) precompute expensive features offline (document quality scores, link-based importance like PageRank), (2) compute query-dependent features in real-time (term frequency, TF-IDF scores), (3) use feature engineering to keep inference fast (linear models or shallow neural networks rather than deep models), (4) cache results for common queries. Implement early exit strategies where you evaluate features in order of cost, and stop early if confidence in ranking is high enough. Example: Google Search might retrieve 10 million documents matching query terms in 5ms using the inverted index, then apply a learned-to-rank model to score top 1000 documents in another 20ms, considering factors like relevance, freshness, domain authority, and personalization signals. Precompute page quality scores once per day, store in memory for instant access, and continuously monitor latency percentiles (aim for <50ms p50, <100ms p99).

---

### Q8: How would you design a fraud detection system for a financial platform?

**A:** Fraud detection requires low false positives (to avoid blocking legitimate users), real-time decisions, and handling severe class imbalance (fraud is typically <0.1% of transactions). Architecture: (1) Real-time features computed during the transaction—transaction amount, merchant category, user location, time of day, device fingerprint. (2) User-specific features computed in real-time from recent history—average transaction amount, typical merchant categories, typical transaction times, country of transaction (flag if unusual). (3) Historical precomputed features—user's fraud history, account age, verification status. (4) Model ensemble: use a fast rule-based model (blocks obvious cases like transactions in two countries within 10 minutes) plus a machine learning model (gradient boosting or neural network trained on historical fraud patterns). (5) Feedback loop: human review queue for uncertain cases, investigation outcomes feed back into retraining. Handle class imbalance using weighted loss or SMOTE. Minimize latency (<500ms decision time) by precomputing features in feature store and using efficient model serving. Example: For a credit card transaction, compute user's typical merchant category and location in <50ms using cached features, run through logistic regression model that outputs fraud probability, compare against dynamic threshold based on transaction amount (higher threshold for small amounts, lower for large ones), and flag for review if probability exceeds threshold.

---

### Q9: Explain how to handle data distribution shift and concept drift in production ML systems.

**A:** Data distribution shift occurs when the feature distribution or label distribution changes from training data (e.g., fraudsters adapt tactics, user behavior changes seasonally, model receives inputs from new sources). Concept drift happens when the relationship between features and labels changes—e.g., the same user behavior might mean different things at different times. Detect both by monitoring key metrics: track feature statistics (mean, std, percentiles) and compare against baseline, use statistical tests (Kolmogorov-Smirnov test) to flag when distributions diverge significantly, monitor model performance metrics (accuracy, AUC on held-out validation data). When drift is detected, trigger retraining: either full retraining on fresh data, or fine-tuning on recent data while keeping learned representations. Some systems use adaptive learning rates that increase when drift is detected. Example: In a recommendation system, if user engagement patterns change seasonally (summer users engage differently than winter users), precomputed embeddings become stale. Detect this by monitoring embedding similarity (measure how different are embeddings trained on recent data vs baseline), and retrain embeddings monthly instead of yearly. For a credit scoring model, monitor feature distributions (income, credit utilization) and retrain weekly if they shift significantly, using only recent data which better represents current customer base.

---

### Q10: What is A/B testing in production ML systems and why is it critical?

**A:** A/B testing in ML systems means deploying a new model to a subset of users (treatment group) while keeping the old model for another subset (control group), measuring whether business metrics improve. This is critical because offline metrics (accuracy, AUC) often don't predict real-world impact—a model with slightly better accuracy might reduce user engagement if it optimizes for the wrong thing, or might be offset by serving latency increases. Set up infrastructure where traffic routing is configurable (e.g., 90% old model, 10% new model), implement comprehensive metric tracking (primary metrics like conversion rate, secondary metrics like latency, guardrail metrics to catch regressions), and run tests for sufficient duration to capture user behavior variation. Use statistical tests to determine if observed differences are significant or due to random variation. Example: In recommendation systems, A/B testing reveals that a model with 2% higher offline recall might hurt engagement because it recommends more diverse items that users skip—you'd switch back to the original model or retrain optimizing for engagement rather than recall. Uber's ML systems run hundreds of A/B tests simultaneously, measuring impact on ride acceptance rate, driver earnings, and system latency, enabling them to validate that new models actually improve the business before full rollout.

---

### Q11: How do you design a content moderation system for user-generated content at scale?

**A:** Content moderation typically uses a two-stage architecture: (1) automated filtering to catch obvious violations quickly and cheaply, (2) human review queue for uncertain cases. For automated filtering, use text classifiers (BERT-based models) trained on labeled harmful content, image classifiers (CNNs) for detecting violating images, and heuristic-based rules (keyword blocklists, account age filters). Implement multiple classifiers targeting different violation types (hate speech, sexual content, violence, spam), running in parallel. Keep latency low (<2s to make moderation decision) by using efficient models or distilled versions. Design a feedback loop: human moderators review flagged content, their decisions retrain classifiers, and high-confidence predictions bypass human review to speed up the process. Handle false positives carefully—incorrectly flagging legitimate content harms users, so design for high precision. Example: Facebook's moderation pipeline uses a combination of Deeptext (their text understanding model) to catch text-based violations, computer vision models for image violations, and enforces a feedback loop where human moderation decisions are logged and used to retrain models weekly. A typical system might flag 10% of content for human review, and human moderators' decisions on that 10% feed back to improve the 90% that were auto-accepted or rejected.

---

### Q12: Explain model parallelism, data parallelism, and pipeline parallelism for training large models.

**A:** Data parallelism splits the dataset across machines—each machine processes a different batch of data, computes gradients, and a central parameter server aggregates gradients before updating the model. This works well when the model fits on one machine but the dataset is huge. Model parallelism splits the model across machines—different parts of the neural network run on different GPUs/TPUs, useful for enormous models (like GPT-3 with 175B parameters) that don't fit on a single device. The trade-off is that model parallelism introduces communication overhead between layers. Pipeline parallelism interleaves data parallelism with model parallelism: split the model into stages (e.g., transformer layers 1-6 on GPU A, layers 7-12 on GPU B), and while GPU B processes output from GPU A, GPU A can start processing the next batch. Example: Training BERT uses data parallelism across 8 GPUs, where each GPU processes 32 examples per batch, computes gradients for all 250M parameters, and a central aggregator sums gradients across the 8 GPUs before each update. Training GPT-3 (175B parameters) requires model parallelism because it's too large to fit on any single GPU, so transformer layers are distributed across thousands of GPUs. Choosing between these: data parallelism is simplest and most efficient if the model fits on one device; use model parallelism only when necessary due to model size.

---

### Q13: How do you decide between latency and accuracy tradeoffs in model serving?

**A:** Latency-accuracy tradeoffs emerge because more accurate models often require more computation (deeper neural networks, ensemble methods, complex feature engineering). First, understand your latency budget from business requirements: user-facing systems might need <100ms total latency, backend systems have more flexibility. Then profile your current model: measure inference time across different devices/batch sizes, identify bottlenecks. Explore techniques to reduce latency while maintaining accuracy: model distillation (train a smaller student model to mimic a larger teacher), quantization (reduce precision from float32 to int8, typically 1-5% accuracy loss but 4x faster), pruning (remove unimportant weights), caching predictions for common inputs. Use ensemble methods wisely—averaging 5 models improves accuracy but increases latency proportionally. Implement adaptive serving: use a fast model for most requests, and only run a slower, more accurate model for uncertain cases. Example: In fraud detection, you might use a logistic regression model (<5ms inference) for 80% of transactions that are clearly legitimate or fraudulent, and only run a slower gradient boosting model (100ms inference) on the 20% with intermediate scores. A/B test latency changes carefully—serving in 50ms instead of 10ms might reduce conversion by more than the accuracy improvement justifies.

---

### Q14: Design monitoring and alerting for ML systems in production.

**A:** Comprehensive monitoring tracks multiple layers: input data (distribution of features), model predictions (predicted probabilities, predicted labels), user outcomes (clicks, conversions, fraud cases), and system health (latency, error rate, throughput). For input data monitoring, track feature statistics (mean, std, percentiles, null rates) and alert if they deviate significantly from baseline using statistical tests or anomaly detection. For predictions, monitor prediction distribution (if class distribution shifts unexpectedly, it suggests concept drift), and compare predictions to actual labels using held-out validation sets. For business metrics, track primary metrics (engagement, conversion) and guardrail metrics to catch regressions. Implement logging at scale: sample predictions and reasons for critical decisions, store in a data warehouse, use for diagnosis and model retraining decisions. Set up alert thresholds that are meaningful (alert when AUC drops by >2%, not >1%) to reduce false alarms. Example: LinkedIn's recommendation system monitors that click-through-rate on recommendations stays within expected range, feature distributions don't shift, latency stays <200ms p99, and embedding similarity stays consistent. If embedding similarity drops (indicating embeddings diverged), it automatically triggers full retraining. Use dashboards for visibility and alerting rules for automated incident response—if latency spikes, automatically trigger traffic rerouting to fallback model.

---

### Q15: How do you implement feedback loops and continuously improve ML systems?

**A:** Feedback loops close the gap between model predictions and real-world outcomes: collect labels on predictions (whether a recommendation was clicked, whether fraud was actually committed), use these labels to retrain and improve the model. Explicit feedback is direct labels from users (thumbs up/down on recommendations, reported spam), but is sparse. Implicit feedback infers labels from behavior (clicks indicate relevance, time spent indicates satisfaction). Implement the loop: predictions → user sees result → user provides feedback (implicit or explicit) → feedback is logged → periodically retrain model on fresh data including new feedback. Key challenges: addressing feedback bias (users only interact with items they see, creating selection bias), and handling labeling delays (in fraud detection, true labels come days later after human review). Use techniques like importance weighting to correct for selection bias. Implement efficient retraining: use online learning to update models incrementally on new data, or batch retraining daily/weekly. Example: For recommendations, when a user clicks a recommended item, log that signal, and weekly retrain embeddings and ranking models on accumulated clicks. For fraud detection, human moderators label suspicious transactions, and model is retrained weekly on confirmed fraud cases, continuously improving precision. Without feedback loops, model performance degrades over time as the world changes; with them, systems improve automatically.

---

## Interview Cheatsheet

**Key Terms:**
- **Data Parallelism:** Distributing data across machines, each computes gradients on different batches; central aggregator sums gradients before model update
- **Model Parallelism:** Splitting model layers across machines; used for models too large for single device; introduces inter-device communication overhead
- **Pipeline Parallelism:** Combining data and model parallelism; different model stages run on different GPUs while processing different batches
- **Feature Store:** Centralized system managing feature computation for both training and serving, preventing training-serving skew
- **Online Features:** Computed in real-time during serving with strict latency budget; use current/recent data
- **Offline Features:** Computed from historical data during training; high latency tolerance, can be expensive
- **A/B Testing:** Comparing new model against baseline on subset of traffic, measuring business metrics to validate improvements
- **Data Distribution Shift:** Feature or label distribution changes from training data (e.g., user behavior seasonality)
- **Concept Drift:** Relationship between features and labels changes over time
- **Batch Serving:** Models run periodically on all data, results cached; high throughput, high staleness
- **Real-Time Serving:** Models answer requests synchronously; low latency at serving time, requires efficient inference
- **Feedback Loop:** Collecting ground truth labels on predictions and retraining models with new data

**Rapid-Fire Q&A:**
- **Q: What's the first step in designing an ML system?** **A:** Clarify business requirements: target metric, latency constraints, throughput needs, what problem solves.
- **Q: Why does training-serving skew happen?** **A:** Features computed differently offline (for training) vs online (for serving); prevents consistent predictions.
- **Q: How do you detect model drift?** **A:** Monitor feature statistics, prediction distribution, and model performance on held-out validation set; alert on significant deviations.
- **Q: Batch or real-time serving for email personalization?** **A:** Batch—email is sent once per day, so precomputing personalization scores offline works well.
- **Q: How do you handle fraud class imbalance?** **A:** Use weighted loss (fraud cases weighted higher), SMOTE oversampling, or change decision threshold to optimize precision.
- **Q: What's the difference between accuracy and business metrics?** **A:** Accuracy doesn't guarantee business impact; recommendations with higher accuracy might reduce engagement if they're less diverse.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# MLOps and Model Deployment

## Interview Anchor
- **ML Lifecycle:** The complete process from problem definition through data collection, training, evaluation, deployment, monitoring, and iteration
- **Model Versioning and Registry:** Systems for tracking model artifacts, metadata, and lineage to enable reproducibility and rollback
- **MLOps:** Practices and tools automating the ML lifecycle, similar to DevOps but addressing unique ML challenges

## Key Concepts Overview
MLOps bridges the gap between machine learning research and production systems. While research focuses on model accuracy, MLOps ensures models are reproducible, deployable, monitorable, and maintainable at scale. Key challenges that MLOps addresses include: tracking which data/code/hyperparameters created which model, detecting when models degrade in production, managing multiple model versions, automating retraining, and ensuring consistency across environments. Understanding MLOps is critical for production ML engineers because models in production face real-world data drift, adversarial scenarios, and resource constraints that training doesn't encounter, requiring continuous monitoring and iteration.

---

### Q1: Describe the complete ML lifecycle and why each phase matters.

**A:** The ML lifecycle consists of: (1) **Problem Definition** - clarify business goals, success metrics, constraints, (2) **Data Collection** - gather raw data from various sources, (3) **Data Exploration & Cleaning** - understand distributions, handle missing values, outliers, (4) **Feature Engineering** - create predictive features from raw data, (5) **Train/Validation Split** - divide data, avoiding leakage, (6) **Model Selection & Training** - choose architecture, optimize hyperparameters, (7) **Evaluation** - measure performance on held-out test set, (8) **Deployment** - move model to production, integrate with applications, (9) **Monitoring** - track performance, detect issues, (10) **Retraining** - periodically retrain on new data. Each phase is critical: skipping problem definition leads to optimizing the wrong objective, poor data collection/cleaning causes garbage-in-garbage-out, bad train/test splits cause overfitting estimates, and neglecting monitoring causes silent model degradation. The lifecycle is iterative: monitoring reveals data drift, prompting retraining with new data, which flows back through feature engineering and model selection. Most model failures occur not during training (which works well) but during deployment and monitoring phases where real-world complexity emerges.

---

### Q2: What is experiment tracking and how do tools like MLflow help?

**A:** Experiment tracking records metadata about training runs: hyperparameters, metrics, code versions, data versions, and artifacts (model files). Without tracking, it's impossible to reproduce results or understand which hyperparameters led to best performance. MLflow provides: (1) **parameter logging** - save hyperparameters, (2) **metric logging** - record metrics at each epoch, (3) **artifact storage** - save model weights and plots, (4) **comparison UI** - visualize which experiments were best, (5) **reproducibility** - re-run experiments with recorded parameters. Example:
```python
import mlflow
mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
for epoch in range(10):
    loss = train_epoch()
    mlflow.log_metric("loss", loss, step=epoch)
mlflow.log_artifact("model.pkl")
mlflow.end_run()
```
MLflow then provides a UI showing all runs, allowing filtering (best runs, specific hyperparameters) and comparison. Alternatives include Weights & Biases (excellent visualizations), Neptune (good collaboration), Comet (comprehensive). Experiment tracking is essential because: (1) hyperparameter tuning runs hundreds of experiments; tracking keeps them organized, (2) production models are typically best runs from large searches, (3) debugging requires understanding what worked, (4) collaboration requires shared visibility. Without tracking, teams often re-run experiments or forget what was tried.

---

### Q3: What is model versioning and why is it critical?

**A:** Model versioning tracks different versions of trained models with their metadata: training data version, code version, hyperparameters, performance metrics, and training date. Without versioning, it's unclear which model file corresponds to which experiment, making debugging or rollback impossible. A model version might be: `model_v2.3_trained_2024_q4_lr0.001_val_acc_0.94`. Versioning enables: (1) **reproducibility** - given a model version, you can identify exactly what data and code created it, (2) **comparison** - systematically compare v2.2 (94% accuracy) vs. v2.3 (95% accuracy) to decide if upgrade is worth risks, (3) **rollback** - if a new model performs poorly in production, revert to previous version, (4) **experimentation** - run A/B tests comparing models. Implementation: store models with descriptive names/metadata, use semantic versioning (major.minor.patch), and document changes between versions. Best practice: never overwrite model files; instead, create new versions. Tools like DVC (Data Version Control) and MLflow provide versioning as a service, maintaining a registry of all models with metadata and tracking lineage (which data/code/params created model v2.3).

---

### Q4: What is a model registry and how does it differ from model versioning?

**A:** A **model registry** is a centralized system that stores, manages, and tracks multiple model versions, enabling governance and controlled promotion. A registry typically provides: (1) **storage** - centralized repository for model artifacts (weights, configs), (2) **metadata** - hyperparameters, training data version, performance metrics, (3) **versioning** - track multiple versions, (4) **stage management** - categorize models (Staging, Production, Archived), (5) **annotations** - add notes, issues, or requirements, (6) **lineage** - track which data/code/params created each model, (7) **API access** - retrieve model by name/version for prediction. Example workflow:
1. Train and log model to registry: `mlflow.models.log("sales_model", model_artifact, v1)`
2. Register in staging: `registry.transition_model_version("sales_model", "1", "Staging")`
3. Evaluate in staging environment
4. Promote to production: `registry.transition_model_version("sales_model", "1", "Production")`
5. Monitor in production; if it degrades, promote v2: `transition_model_version(..., "2", "Production")`

Model versioning (tracking models) is different from a registry (managing, governing, promoting models). A registry is crucial for controlled deployments: preventing accidentally deploying unvalidated models, tracking which model is in production, and enabling rollbacks. MLflow Model Registry and SageMaker Model Registry are common implementations.

---

### Q5: Explain CI/CD for ML and how it differs from traditional CI/CD.

**A:** Traditional CI/CD (Continuous Integration/Continuous Deployment) for software: commit code → run tests → build artifact → deploy. ML CI/CD adds complexity because model quality depends on data and hyperparameters, not just code. ML CI/CD pipeline: (1) **Data validation** - check that new data has expected schema, distributions, (2) **Train** - retrain model on new data, (3) **Evaluate** - measure performance, compare to baseline (production model), (4) **Validate** - check that new model exceeds performance threshold (e.g., > 0.95 accuracy), (5) **Register** - log model to registry if it passes, (6) **Deploy** - promote to staging/production if approved. Example:
```
Code push → 
  Run data validation (schema, nulls, distributions) →
  Retrain model →
  Evaluate against test set →
  Compare to production baseline →
  If new_model_accuracy > baseline_accuracy + threshold:
    Register model
    (optional) Deploy to staging
    (manual approval) → Deploy to production
  Else: Reject
```
Challenges unique to ML: (1) **non-deterministic** - retraining with same code/data may yield slightly different results due to randomness, (2) **data-dependent** - pipeline requires fresh data; can't just run tests, (3) **slow** - training takes hours; can't run on every commit, (4) **evaluation is subjective** - what accuracy threshold justifies deployment? Traditional CI/CD for software is deterministic (tests pass or fail); ML CI/CD is probabilistic (accuracy improved by 0.5%). This requires different practices: data validation is critical, A/B testing is essential, and staged rollouts are preferred over binary deploy/no-deploy.

---

### Q6: What are feature stores and how do they improve ML systems?

**A:** A feature store is a centralized system that manages feature definitions, storage, and retrieval, enabling consistency across training and serving. Without a feature store, teams define features redundantly: one in training code, another in serving code, leading to training-serving skew. A feature store provides: (1) **feature definitions** - describe features (raw or computed) with metadata, (2) **offline storage** - historical features for training, (3) **online storage** - low-latency retrieval for inference, (4) **feature computation** - batch compute or real-time compute, (5) **versioning** - track feature definitions over time, (6) **discovery** - catalog features for team visibility. Example workflow:
```
Define features:
  user_age, user_recent_purchases, user_purchase_frequency
Store in offline warehouse (BigQuery)
Compute daily: user_recent_purchases = count(purchases in last 30 days)
For training: fetch features at training time (historical)
For serving: copy features to online store (Redis), fetch at prediction time
Result: same features, same definitions, in both training and serving
```
Popular feature stores: Tecton, Feast, Databricks Feature Store. Benefits: (1) **consistency** - no skew, (2) **efficiency** - feature computation is shared, (3) **discovery** - teams see what features exist, (4) **governance** - track which models use which features. Challenges: (1) **complexity** - requires infrastructure, (2) **latency** - computing features may add latency to serving, (3) **maintenance** - feature definitions must be kept up-to-date. Feature stores are valuable for teams with many models and complex features; they reduce duplication and bugs.

---

### Q7: What is data versioning with DVC and why does it matter?

**A:** DVC (Data Version Control) tracks data files and their versions, solving the problem of reproducibility when data changes. Unlike Git (which tracks code), DVC tracks data: store data on remote storage (S3, GCS), track metadata locally. Example:
```bash
dvc add data/train.csv  # Creates data/train.csv.dvc (metadata file)
git add data/train.csv.dvc  # Track metadata in Git
```
When data changes:
```bash
dvc update data/train.csv  # Recompute or fetch latest
dvc checkout  # Retrieve specific version
```
DVC enables: (1) **reproducibility** - given a Git commit, checkout the exact data used for training, (2) **collaboration** - share data versions across team, (3) **efficiency** - don't replicate large files in Git, (4) **workflows** - define data pipelines (transform raw data → train → evaluate). Benefits: (1) **debugging** - if a model fails, you can recreate exact training data, (2) **auditing** - track what data produced what model, (3) **efficiency** - don't duplicate data; store once, reference everywhere. DVC integrates with Git: Git tracks code and data metadata, DVC tracks data itself. Together, they enable reproducible, versioned ML: check out commit, DVC retrieves corresponding data, retrain model, get identical results. This is crucial for production systems where auditing and reproducibility are requirements.

---

### Q8: Compare batch vs real-time model serving and their trade-offs.

**A:** **Batch serving** generates predictions for all observations at once, usually scheduled (e.g., nightly). Process: load data, generate predictions, store results, serve from cache. Example: movie recommendations computed nightly, served from a database at prediction time. Benefits: (1) **efficiency** - leverage GPU batching, amortize load times, (2) **cost** - run on cheaper, off-peak resources, (3) **simplicity** - deterministic, easy to monitor. Drawbacks: (1) **latency** - recommendations are 24 hours old, (2) **scalability** - if new users arrive, batch hasn't computed for them yet (cold start), (3) **resource** - requires storage proportional to observation count. **Real-time serving** generates predictions on-demand, milliseconds after request. Process: receive request → run inference → return response. Benefits: (1) **freshness** - predictions use current data, (2) **scalability** - compute only for users requesting, (3) **dynamic** - adapt to new data immediately. Drawbacks: (1) **latency-critical** - P99 latency matters, (2) **cost** - inference on every request is expensive, (3) **complexity** - requires low-latency infrastructure, caching. Hybrid approaches: pre-compute for common cases (batch), fall back to real-time for uncommon cases. Example: Netflix precomputes recommendations (batch), but also generates real-time recommendations based on current behavior. Choose batch for: internal reporting, asynchronous tasks, stable patterns. Choose real-time for: user-facing recommendations, trading, fraud detection where latency matters.

---

### Q9: Explain containerization with Docker and its role in ML deployment.

**A:** Containerization packages models, code, dependencies, and runtime into a self-contained unit (Docker image) that runs identically across environments. Benefits: (1) **reproducibility** - Docker ensures code runs the same on laptop, staging, production, (2) **isolation** - container doesn't interfere with host system, (3) **dependency management** - specify exact Python/library versions, (4) **scalability** - easily replicate containers across servers. Docker workflow:
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pkl .
COPY app.py .
EXPOSE 8000
CMD ["python", "app.py"]
```
Build: `docker build -t my_model:v1 .`
Run: `docker run -p 8000:8000 my_model:v1`

Without Docker: "works on my machine but not in production" is common because dependency versions differ. With Docker, the same image runs everywhere. Docker is essential for ML because: (1) **models have complex dependencies** (specific PyTorch versions, CUDA versions), (2) **serving code may have different dependencies than training**, (3) **scaling requires deploying many copies; Docker makes this easy.** Production ML systems almost always use Docker. Combined with orchestration (Kubernetes), Docker enables deploying hundreds of model instances elastically.

---

### Q10: What is Kubernetes and how does it support ML deployments?

**A:** Kubernetes (K8s) is an orchestration platform that automates deployment, scaling, and management of containerized applications. For ML: deploy Docker containers running model servers, automatically scale up/down based on load, handle failures. Key features: (1) **pods** - smallest unit, run one or more containers, (2) **deployments** - specify desired replicas and updates, (3) **services** - expose pods as network endpoints, (4) **auto-scaling** - increase replicas if CPU/memory exceed thresholds, (5) **load balancing** - distribute requests across replicas, (6) **rolling updates** - update models without downtime. Example:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: model
        image: my_model:v1
        ports:
        - containerPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    name: model-server
  minReplicas: 3
  maxReplicas: 100
  targetCPUUtilizationPercentage: 80
```
This creates 3 replicas initially, auto-scales if CPU exceeds 80%, up to 100 replicas. Benefits: (1) **elasticity** - handle traffic spikes, (2) **high availability** - if a pod crashes, K8s restarts it, (3) **cost efficiency** - scale down during low traffic, (4) **multi-environment** - same config runs on local, staging, production. Challenges: (1) **steep learning curve**, (2) **infrastructure overhead**, (3) **troubleshooting can be complex**. K8s is the industry standard for deploying ML at scale; startups may use simpler platforms (AWS Lambda, Heroku), but enterprise systems almost always use K8s.

---

### Q11: Explain model monitoring, data drift, and concept drift.

**A:** **Model monitoring** tracks model performance in production to detect degradation. Key metrics: accuracy, precision, recall, latency, inference count. But monitoring accuracy is tricky in production because true labels arrive late (for credit approval, true label arrives when loan is repaid). **Data drift** (covariate shift) occurs when input data distribution changes. Example: a model trained on urban traffic patterns is deployed in rural areas with different traffic patterns. Features look similar (speed, road type), but distribution is different. Detection: compare training data distribution to production data distribution using statistical tests (Kolmogorov-Smirnov test). **Concept drift** (label shift) occurs when the relationship between features and labels changes. Example: a model predicts house prices based on location, square footage, trained when housing market was stable. After a market crash, the relationship changes (location matters less). Both drifts cause performance degradation. Monitoring strategies: (1) **hold-out validation set** - periodically compare production performance to validation set (if drift-free, should match), (2) **proxy metrics** - track features (e.g., if average house price drops 30%, likely concept drift), (3) **shadow models** - train new models on recent data, compare to production model, (4) **human feedback** - if possible, sample predictions and check true labels. Response: retrain on recent data, adjust hyperparameters, or roll back to previous model.

---

### Q12: What is A/B testing in production and why is it essential for models?

**A:** A/B testing compares two models in production: split traffic, run model A for 50% of users, model B for 50%, measure outcomes. A/B testing is crucial for ML because offline metrics (validation set accuracy) may not translate to real-world success. Example: model B has 1% higher accuracy on validation set, but increases inference latency by 100ms, causing users to leave. A/B tests reveal this. Implementation: (1) **split traffic** - route 50% of requests to model A, 50% to model B, (2) **measure outcomes** - track business metrics (CTR, conversion, engagement, revenue), (3) **run long enough** - need sufficient samples for statistical significance, (4) **analyze** - perform statistical test (t-test) to check if B is better than A. Example: model A (current): 2.5% click-through rate, model B (new): 2.6% CTR. Run for 1M users per group. Statistical test: is 2.6% significantly better than 2.5%? If p-value < 0.05, yes, roll out B. A/B testing is superior to offline evaluation because: (1) **business metrics** - accuracy is proxy; CTR is what matters, (2) **long-term effects** - online reveals effects that don't appear in offline testing (user satisfaction, retention), (3) **discovers issues** - latency, fairness, user experience issues only appear in production. Best practices: plan duration upfront (longer = higher precision), account for multiple comparisons (if testing 5 models, adjust significance threshold), document all A/B tests.

---

### Q13: Explain canary deployments and shadow mode for safe rollouts.

**A:** **Canary deployment** rolls out new models to a small percentage of traffic first, monitoring for issues before full rollout. Process: (1) Deploy model v2 alongside v1, (2) Route 1% of traffic to v2, 99% to v1, (3) Monitor metrics (accuracy, latency, errors), (4) If v2 looks good, increase to 5%, then 20%, then 100%, (5) If issues detected, rollback to v1. Canary deployments are safer than big-bang rollouts (100% deployment at once) because: (1) **limit blast radius** - if v2 has bugs, only 1% of users affected, (2) **catch issues early** - real-world traffic reveals problems, (3) **gradual confidence building** - roll out slowly as confidence increases. Example configuration:
```yaml
# Canary deployment with Istio/Kubernetes
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-service
spec:
  hosts:
  - model.example.com
  http:
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: model-v1
      weight: 99
    - destination:
        host: model-v2
      weight: 1
```

**Shadow mode** runs new model in parallel without affecting predictions, purely for testing. Process: (1) For each prediction request, run both v1 and v2, (2) Return v1's prediction to user (no change), (3) Log v2's prediction and compare to v1, (4) No impact on users but reveals how v2 would perform. Shadow mode is ideal for: validating model changes before canary, ensuring output format compatibility, stress-testing without risk. Canary is preferred for rolling out production changes; shadow mode is preferred for offline validation before canary.

---

### Q14: Explain model compression (quantization, pruning, distillation) and their use cases.

**A:** Model compression reduces model size and latency for deployment to edge/mobile devices or reducing serving costs. Techniques: (1) **Quantization** - reduce precision (float32 → int8), shrinking model by 4x, slightly reducing accuracy. Example: 200MB model becomes 50MB. Trade-off: faster inference, lower memory, but slightly lower accuracy. (2) **Pruning** - remove unimportant weights (set to 0), reducing computations. Example: remove 50% of weights, model is 2x faster, maintaining accuracy. Requires retraining to recover accuracy. (3) **Knowledge distillation** - train a small "student" model to mimic large "teacher" model. Process: use teacher's soft targets (probability distributions), not just hard labels, to train student. Result: small model approximates large model's behavior. Example: 500MB teacher model is distilled into 50MB student, maintaining 95% of accuracy. Trade-off: requires careful tuning, but small model can be deployed to mobile/edge. Use cases: (1) **mobile inference** - compress for phones (limited memory/battery), (2) **latency constraints** - financial trading, autonomous driving need fast inference, (3) **cost reduction** - serving costs scale with model size; compression reduces costs, (4) **edge deployment** - IoT devices, embedded systems have limited resources. Quantization is simplest; distillation achieves best results but requires more work. Most production systems use combinations: distill to smaller architecture, then quantize.

---

### Q15: What is ONNX and how does it enable cross-platform deployment?

**A:** ONNX (Open Neural Network Exchange) is a standard format for representing trained models, enabling portability across frameworks and platforms. Without ONNX: models trained in PyTorch are .pt files (PyTorch-specific), can't easily run in TensorFlow, Rust, JavaScript. With ONNX: export model to .onnx file, run anywhere. Process: train in PyTorch → export to ONNX → load in any ONNX runtime (TensorFlow, ONNX Runtime, TensorRT, etc.). Benefits: (1) **framework agnostic** - train in PyTorch, serve in Java/C++, (2) **hardware optimization** - ONNX runtimes optimize for CPUs, GPUs, TPUs, (3) **performance** - ONNX Runtime often faster than native frameworks, (4) **portability** - same model runs on cloud, edge, mobile. Example:
```python
import torch
import onnx

model = torch.load("model.pt")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")

# Now load in ONNX Runtime (any language)
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
output = sess.run(None, {"input": dummy_input.numpy()})
```

ONNX also enables optimizations: **TensorRT** (NVIDIA) optimizes ONNX for GPUs, achieving 5-10x speedup through layer fusion, mixed precision (float32 → float16). This is valuable for latency-critical serving. ONNX is becoming industry standard; most production systems export models to ONNX for serving.

---

## Interview Cheatsheet

**Key Terms:**
- **ML Lifecycle:** Problem definition → Data → Training → Evaluation → Deployment → Monitoring → Retraining
- **Experiment Tracking:** Recording hyperparameters, metrics, artifacts for reproducibility
- **Model Registry:** Centralized system for storing, versioning, and promoting models
- **Feature Store:** Centralized management of feature definitions and retrieval
- **Data Drift:** Distribution shift in input data
- **Concept Drift:** Shift in relationship between features and labels
- **Canary Deployment:** Rolling out models to small traffic percentage before full rollout
- **Knowledge Distillation:** Training small student model to mimic large teacher model

**Rapid-Fire Q&A:**
- **Q: What's the most common MLOps mistake?** **A:** Neglecting monitoring; models degrade silently in production.
- **Q: Why use feature stores?** **A:** Prevent training-serving skew, share computations, enable discovery.
- **Q: When should I use batch vs real-time serving?** **A:** Batch for internal/stable patterns; real-time for user-facing/changing requirements.
- **Q: How do I know if data drifted?** **A:** Compare training/production data distributions (KS test), or monitor proxy metrics (feature statistics).
- **Q: What's the first step in MLOps?** **A:** Versioning: code (Git), data (DVC), models (MLflow registry).

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

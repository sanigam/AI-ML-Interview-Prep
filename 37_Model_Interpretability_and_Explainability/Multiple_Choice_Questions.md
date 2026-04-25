# Multiple Choice Questions: Model Interpretability and Explainability

📺 **Video Lecture:** https://youtu.be/lLWq7gd0w5g


## Question 1
What is the primary difference between interpretability and explainability in machine learning?

A) Interpretability means the model is transparent and its structure can be understood directly, while explainability refers to post-hoc methods that explain black-box model predictions
  
B) Interpretability and explainability are interchangeable terms referring to the same concept of understanding model decisions
  
C) Explainability is stricter than interpretability and always requires regulatory compliance
  
D) Interpretability applies only to neural networks, while explainability applies to traditional models

---

## Question 2
Which of the following intrinsically interpretable models can capture nonlinear relationships while maintaining separate univariate interpretability?

A) Linear regression
  
B) Logistic regression
  
C) Generalized Additive Models (GAMs)
  
D) Standard decision trees

---

## Question 3
What makes Shapley values unique among feature attribution methods?

A) They are the fastest method to compute across all model types
  
B) They are the only attribution method satisfying local accuracy, symmetry, and dummy properties simultaneously
  
C) They automatically handle categorical features without preprocessing
  
D) They require no knowledge of the underlying model architecture

---

## Question 4
TreeSHAP makes Shapley value computation practical for tree models. What is its computational complexity improvement compared to exact coalitional computation?

A) Linear in the number of features instead of exponential
  
B) O(leaf_count * num_features) instead of exponential
  
C) Logarithmic instead of polynomial
  
D) No improvement; it only provides approximate values faster

---

## Question 5
Which statement about LIME (Local Interpretable Model-agnostic Explanations) is most accurate?

A) LIME produces stable explanations that remain consistent across multiple runs on the same instance
  
B) LIME fits a weighted linear model locally around a prediction to explain it, but this approach assumes the model behaves linearly locally
  
C) LIME is guaranteed to provide accurate explanations for any model regardless of complexity
  
D) LIME eliminates the need for Shapley values or other feature attribution methods

---

## Question 6
When using KernelSHAP to approximate Shapley values, what is the primary computational tradeoff compared to TreeSHAP?

A) KernelSHAP requires fewer model evaluations but produces lower-quality approximations
  
B) KernelSHAP is model-agnostic and works for any model, but is slower and produces approximate (not exact) Shapley values
  
C) KernelSHAP is faster but only works with tree-based models
  
D) KernelSHAP produces exact values but requires retraining the model

---

## Question 7
For convolutional neural networks, Grad-CAM is an improvement over simple saliency maps primarily because:

A) Grad-CAM computes pixel-level gradients more accurately than saliency maps
  
B) Grad-CAM uses information from deep convolutional layers to preserve spatial structure, whereas saliency maps only use gradients with respect to input pixels
  
C) Grad-CAM works faster and is easier to implement than saliency maps
  
D) Grad-CAM can handle any type of neural network architecture, not just CNNs

---

## Question 8
What is a critical limitation of using attention weights in transformer models as an interpretability tool?

A) Attention weights are too expensive to compute for large models
  
B) Attention weights show what the model attends to, which always correlates with feature importance for the final prediction
  
C) Attention weights don't directly measure feature importance; high attention doesn't necessarily mean the token is important for the decision
  
D) Transformer models don't actually use attention mechanisms

---

## Question 9
How does permutation importance differ from drop-column importance in its approach to measuring feature significance?

A) Permutation importance retrains the model without the feature, while drop-column importance shuffles the feature values
  
B) Permutation importance shuffles feature values to measure performance drop, while drop-column importance retrains the model entirely without the feature
  
C) They are identical methods with different names
  
D) Drop-column importance is specific to tree models, while permutation importance works for all models

---

## Question 10
Partial Dependence Plots (PDP) have an important assumption that limits their applicability. What is it?

A) PDPs only work for classification tasks, not regression
  
B) PDPs assume that the feature of interest is independent of other features in the dataset
  
C) PDPs require the model to be a linear model
  
D) PDPs cannot be used with categorical features

---

## Question 11
How do Individual Conditional Expectation (ICE) curves complement Partial Dependence Plots (PDP)?

A) ICE curves provide faster computation than PDPs
  
B) ICE curves show how predictions change with a feature for each individual instance, revealing heterogeneity that PDP's average might obscure
  
C) ICE curves replace the need for PDPs in modern interpretability practices
  
D) ICE curves are only applicable to tree-based models

---

## Question 12
What makes counterfactual explanations particularly valuable in high-stakes decision-making contexts?

A) They are the most accurate method for computing feature importance
  
B) They answer "what needs to change for a different decision?" which is actionable and intuitive, unlike abstract importance scores
  
C) They completely eliminate the need for regulatory compliance documentation
  
D) They work equally well for all types of models without any additional considerations

---

## Question 13
When should you prioritize global explanations over local explanations in model interpretability?

A) Global explanations should always be used instead of local explanations
  
B) Global explanations help identify systematic issues across the dataset like bias patterns, while local explanations justify individual decisions
  
C) Local explanations are used for debugging, global explanations are only for regulatory compliance
  
D) The choice between global and local explanations has no practical difference

---

## Question 14
What does the GDPR "right to explanation" require for automated decision-making systems?

A) Companies must always use only interpretable models like linear regression
  
B) Meaningful information about the logic and significance of the decision; this can be satisfied through interpretable models, explainability methods, or human review
  
C) Companies must explain every single model prediction in real-time to users
  
D) The right to explanation only applies to recommendation systems, not credit or hiring decisions

---

## Question 15
In balancing model complexity and interpretability, what is a practical strategy for high-stakes domains like healthcare or finance?

A) Always choose the model with the highest accuracy regardless of interpretability
  
B) Start with interpretable baseline models and add complexity only if accuracy gains justify it; use explainability methods as a fallback
  
C) Ignore interpretability completely and focus solely on predictive performance
  
D) Use only neural networks since they provide the best interpretability through attention mechanisms

---

## Answer Key

**Q1: A**
Interpretability means the model structure is transparent and understandable directly (linear models, decision trees), while explainability refers to post-hoc methods (SHAP, LIME) that explain black-box model predictions after training. This is the fundamental distinction in the field.

**Q2: C**
Generalized Additive Models (GAMs) maintain interpretability through univariate relationships (Y = intercept + f1(X1) + f2(X2) + ...) while allowing nonlinear functions fi. Linear and logistic regression are linear-only, and decision trees don't explicitly separate features this way.

**Q3: B**
Shapley values are the only attribution method satisfying three key properties: local accuracy (contributions sum to prediction), symmetry (identical features get identical values), and dummy (irrelevant features get zero). This theoretical grounding makes them unique.

**Q4: B**
TreeSHAP exploits tree structure with dynamic programming, achieving O(leaf_count * num_features) complexity instead of exponential coalitional enumeration. This makes exact Shapley values practical for tree-based models like random forests and gradient boosting.

**Q5: B**
LIME fits a weighted linear model locally to explain individual predictions. Its key limitation is that this approach assumes the model behaves linearly near the instance, which fails at complex decision boundaries. LIME is also unstable—different runs can produce different explanations.

**Q6: B**
KernelSHAP is model-agnostic (works for any model) but slower than TreeSHAP and produces approximate values. It fits a weighted linear regression locally instead of exact Shapley computation, trading off accuracy for speed and generality.

**Q7: B**
Grad-CAM weights feature maps using gradients of the target class score, using information from deep convolutional layers to preserve spatial structure. Saliency maps only compute gradients with respect to input pixels, making Grad-CAM more robust.

**Q8: C**
A critical misunderstanding is that attention weights directly measure importance. In reality, high attention doesn't necessarily mean the token is important for the final decision. Attention is optimized for training loss, not interpretability, and different layers/heads have different roles.

**Q9: B**
Permutation importance shuffles feature values and measures performance drop while keeping the model fixed. Drop-column importance retrains the model entirely without the feature. Drop-column is more direct but expensive; permutation importance is model-agnostic.

**Q10: B**
PDPs assume the feature is independent of other features. If income and age are correlated, PDPs that replace income with arbitrary values may produce unrealistic scenarios, making the plot misleading.

**Q11: B**
While PDPs show the average relationship across the dataset, ICE curves show the relationship for individual instances separately. ICE reveals heterogeneity—some instances may be sensitive to a feature while others aren't.

**Q12: B**
Counterfactual explanations are actionable ("increase income by $10k to get approved") rather than abstract. They directly answer what needs to change, making them particularly valuable for high-stakes decisions where justification matters.

**Q13: B**
Global explanations (permutation importance, PDP) reveal systematic patterns across the dataset and are essential for fairness audits and model debugging. Local explanations (LIME, SHAP) justify individual decisions. Both are needed: global for systematic issues, local for individual fairness.

**Q14: B**
GDPR Article 22 requires "meaningful information about the logic" of automated decisions. This can be satisfied through interpretable models, explainability methods like SHAP, or human review. The regulation doesn't mandate a specific approach but requires understandability.

**Q15: B**
The practical strategy is to start with interpretable baselines (logistic regression, decision trees) and add complexity only if accuracy gains justify it. For high-stakes domains, explainability methods can augment complex models, but interpretability should be treated as a first-class metric alongside accuracy.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

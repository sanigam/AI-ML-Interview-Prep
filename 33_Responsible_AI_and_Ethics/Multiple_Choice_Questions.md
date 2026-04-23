# Multiple Choice Questions: Responsible AI and Ethics

Test your understanding of responsible AI concepts for AI/ML interviews.

---

**Q1. Demographic parity as a fairness criterion requires that:**

A) The model has equal accuracy for all groups
B) Model predictions are independent of the protected attribute (e.g., equal approval rates across genders)
C) All groups have the same features
D) The model never makes errors

---

**Q2. Historical bias in ML systems occurs when:**

A) The algorithm is poorly coded
B) Training data reflects past societal discrimination, causing the model to learn and perpetuate those biases
C) The model is too small
D) The test set is too large

---

**Q3. SHAP (SHapley Additive exPlanations) explains predictions by:**

A) Removing the model entirely
B) Using game-theoretic Shapley values to assign each feature a contribution score for a specific prediction
C) Only explaining linear models
D) Randomly assigning feature importance

---

**Q4. LIME (Local Interpretable Model-agnostic Explanations) works by:**

A) Globally retraining the model
B) Perturbing inputs around a prediction and fitting a simple interpretable model locally to approximate the black-box behavior
C) Only using decision trees
D) Replacing the model with a rule-based system

---

**Q5. The tension between fairness criteria means that:**

A) All fairness criteria can always be satisfied simultaneously
B) Optimizing for one criterion (e.g., demographic parity) may worsen another (e.g., equalized odds), requiring context-dependent choices
C) Fairness is irrelevant in ML
D) Only one fairness metric exists

---

**Q6. Post-processing bias mitigation adjusts:**

A) The training data before training
B) Model predictions after training, such as using different decision thresholds per group
C) The model architecture
D) The loss function during training

---

**Q7. A model card documents:**

A) Only the model's accuracy
B) Model details, intended use, performance across groups, limitations, ethical considerations, and training data characteristics
C) Only the training code
D) Only the model's file size

---

**Q8. Equalized odds requires that the model has:**

A) The same prediction rate for all groups
B) Equal true positive rates AND equal false positive rates across protected groups
C) Perfect accuracy
D) Zero false positives only

---

**Q9. Representation bias in training data occurs when:**

A) All groups are equally represented
B) Certain groups are underrepresented, causing the model to perform worse for those groups
C) The model is too complex
D) Features are perfectly correlated

---

**Q10. Grad-CAM explains image classifier decisions by:**

A) Retraining the model
B) Visualizing which image regions most influenced the prediction using gradient-weighted activation maps
C) Only working with text models
D) Randomly highlighting pixels

---

**Q11. The EU AI Act classifies AI systems into risk categories requiring:**

A) No regulations for any AI
B) Different levels of transparency, documentation, and oversight based on the risk level (unacceptable, high, limited, minimal)
C) All AI to be banned
D) Only voluntary guidelines

---

**Q12. Adversarial debiasing is an in-processing technique that:**

A) Adds more biased data
B) Uses an adversarial network to prevent the model from being able to predict the protected attribute from its representations
C) Removes all features
D) Only works with GANs

---

**Q13. Differential privacy in ML ensures that:**

A) The model's weights are encrypted
B) The presence or absence of any single individual's data has minimal impact on the model's outputs, protecting individual privacy
C) All data is public
D) Only large datasets can be used

---

**Q14. The right to explanation under GDPR gives individuals:**

A) Access to all training data
B) The right to meaningful information about the logic behind automated decisions that significantly affect them
C) The right to retrain the model
D) The right to delete the model

---

**Q15. Calibration as a fairness criterion means:**

A) The model always predicts 50%
B) When the model predicts a probability (e.g., 80%), the actual positive rate should be approximately 80% across all groups
C) All groups have the same features
D) The model is perfectly accurate

---

## Answer Key

**Q1. Answer: B**
Demographic parity requires P(ŷ=1|A=a) = P(ŷ=1|A=b) for all groups a, b, meaning the prediction rate is independent of the protected attribute, regardless of actual outcome rates.

**Q2. Answer: B**
If historical data shows men in leadership roles due to past discrimination, a model trained on this data learns to associate leadership with being male, perpetuating the bias.

**Q3. Answer: B**
SHAP assigns each feature a Shapley value quantifying its marginal contribution to the prediction. These values satisfy theoretical fairness axioms and provide both local and global explanations.

**Q4. Answer: B**
LIME creates perturbed versions of the input, gets predictions for each, and fits a simple linear model to approximate the decision boundary locally, revealing which features drove that specific prediction.

**Q5. Answer: B**
It's mathematically proven that demographic parity, equalized odds, and calibration cannot all be simultaneously satisfied except in trivial cases. The choice depends on the application context and values.

**Q6. Answer: B**
Post-processing adjusts outputs without retraining — for example, using a lower confidence threshold for disadvantaged groups to achieve equal approval rates across groups.

**Q7. Answer: B**
Model cards are standardized documentation covering model purpose, training data, performance disaggregated by group, known limitations, and ethical considerations, promoting transparency.

**Q8. Answer: B**
Equalized odds ensures both TPR and FPR are equal across groups, meaning the model's errors are distributed fairly — no group disproportionately bears false positives or false negatives.

**Q9. Answer: B**
If a facial recognition system is trained primarily on light-skinned faces, it performs poorly on dark-skinned faces due to underrepresentation, a well-documented real-world bias.

**Q10. Answer: B**
Grad-CAM backpropagates the target class gradient to the final convolutional layer, creating a heatmap showing which spatial regions were most important for the classification decision.

**Q11. Answer: B**
The EU AI Act categorizes AI by risk: unacceptable (banned), high-risk (strict requirements), limited risk (transparency obligations), and minimal risk (no requirements).

**Q12. Answer: B**
An adversary tries to predict the protected attribute from model representations. The main model is trained to minimize this predictability, reducing bias in learned representations.

**Q13. Answer: B**
Differential privacy adds calibrated noise during training so the model's behavior is nearly identical whether any individual's data is included or not, providing formal privacy guarantees.

**Q14. Answer: B**
GDPR Article 22 provides rights related to automated decision-making, including the right to obtain meaningful information about the logic involved in decisions that significantly affect individuals.

**Q15. Answer: B**
Calibration means predicted probabilities match actual frequencies across all groups. If a model is calibrated, a 70% confidence prediction should be correct approximately 70% of the time for every group.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

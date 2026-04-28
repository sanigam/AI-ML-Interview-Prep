# Multiple Choice Questions: Data Preprocessing and Feature Engineering

📺 **Video Lecture:** https://youtu.be/cDMa4ukiDF0


Test your understanding of data preparation and feature engineering for ML pipelines.

---

**Q1. Standardization (z-score normalization) transforms features to have:**

A) Equal minimum and maximum values  
B) Mean of 0 and standard deviation of 1  
C) Values between 0 and 1  
D) All positive values

---

**Q2. Min-Max scaling is preferred over standardization when:**

A) The data contains many outliers  
B) The dataset is very large  
C) The features follow a normal distribution  
D) You need values in a bounded range (e.g., [0, 1]) and the data has no extreme outliers

---

**Q3. One-hot encoding converts a categorical variable with k categories into:**

A) A single column of probabilities  
B) k−1 continuous columns  
C) A single numerical column with values 1 to k  
D) k binary columns, each indicating presence/absence of a category

---

**Q4. The "dummy variable trap" occurs when:**

A) You include all k one-hot encoded columns for a k-category variable, creating perfect multicollinearity  
B) You forget to encode categorical variables  
C) You use label encoding instead of one-hot encoding  
D) The categorical variable has missing values

---

**Q5. Which imputation strategy is most appropriate for a feature with many outliers?**

A) Deleting the entire feature  
B) Mean imputation  
C) Replacing missing values with zero  
D) Median imputation

---

**Q6. Feature scaling is particularly important for which type of algorithm?**

A) Random forests  
B) Gradient descent-based algorithms (e.g., logistic regression, SVMs, neural networks)  
C) Decision trees  
D) Rule-based systems

---

**Q7. Log transformation is commonly applied to features that are:**

A) Uniformly distributed  
B) Binary  
C) Right-skewed with a long positive tail  
D) Normally distributed

---

**Q8. Target encoding replaces each category with:**

A) The category's frequency count  
B) A random number  
C) The one-hot encoded vector  
D) The mean of the target variable for that category

---

**Q9. What is data leakage?**

A) When features have missing values  
B) When the model is too complex  
C) When data is stored in an insecure database  
D) When information from the test set or future data inadvertently influences the training process

---

**Q10. Polynomial feature engineering (adding x², x₁x₂, etc.) helps when:**

A) The relationship between features and target is strictly linear  
B) There are nonlinear relationships that a linear model cannot capture  
C) All features are categorical  
D) The dataset has too many features

---

**Q11. When handling missing data, Multiple Imputation is preferred over single imputation because:**

A) It always produces the same result  
B) It accounts for the uncertainty in the imputed values by creating multiple plausible datasets  
C) It removes the need for feature engineering  
D) It is computationally faster

---

**Q12. Binning (discretization) of a continuous variable is useful when:**

A) The relationship with the target is non-monotonic or you want to reduce the effect of outliers  
B) You want to preserve the exact values of the variable  
C) You have very few data points  
D) The variable is already categorical

---

**Q13. Which technique helps detect outliers in multivariate data?**

A) Mahalanobis distance, which accounts for correlations between features  
B) Checking if values exceed 3 standard deviations (works only for univariate)  
C) Sorting the data by index  
D) Counting missing values

---

**Q14. Feature engineering should be applied:**

A) Only to the training set, then the same transformations (fitted on training) applied to test set  
B) Jointly on the combined training and test sets  
C) Differently for training and test sets  
D) Only to the test set

---

**Q15. Interaction features (e.g., x₁ × x₂) capture:**

A) The correlation between features  
B) The missing value pattern  
C) The individual effect of each feature  
D) The combined effect where the influence of one feature depends on the value of another

---

## Answer Key

**Q1. Answer: B**
Standardization computes z = (x − μ) / σ, resulting in mean = 0 and standard deviation = 1. This is different from Min-Max scaling which maps to [0, 1].

**Q2. Answer: D**
Min-Max scaling is best when you need bounded outputs (e.g., for neural network inputs or image pixel values) and when outliers are not a concern, since outliers compress the rest of the data into a narrow range.

**Q3. Answer: D**
One-hot encoding creates k binary columns (one per category), where exactly one column is 1 and the rest are 0 for each observation. This avoids implying ordinal relationships.

**Q4. Answer: A**
Including all k dummy columns creates perfect multicollinearity (any one column can be derived from the others). The fix is to drop one column (k−1 encoding) for linear models.

**Q5. Answer: D**
The median is robust to outliers, unlike the mean which gets pulled by extreme values. For heavily skewed data with outliers, median imputation preserves a more representative central tendency.

**Q6. Answer: B**
Gradient descent-based algorithms are sensitive to feature scales because unscaled features create elongated loss surfaces, causing slow convergence. Tree-based methods are scale-invariant.

**Q7. Answer: C**
Log transformation compresses the right tail and spreads out the left, making right-skewed distributions more symmetric and closer to normal. Common for income, prices, and counts.

**Q8. Answer: D**
Target encoding replaces each category with the mean target value for that category. It can be powerful but requires regularization (smoothing) to avoid overfitting on rare categories.

**Q9. Answer: D**
Data leakage occurs when the model has access to information during training that it wouldn't have at prediction time — such as fitting a scaler on the full dataset including the test set, or using future values as features.

**Q10. Answer: B**
Polynomial features allow linear models to capture nonlinear patterns. For example, adding x² lets a linear regression fit a parabola. However, this increases dimensionality and risk of overfitting.

**Q11. Answer: B**
Multiple imputation creates several imputed datasets, analyzes each, and pools results, properly reflecting the uncertainty from missingness. Single imputation treats imputed values as known, underestimating variance.

**Q12. Answer: A**
Binning is useful when the relationship between a continuous feature and target is non-linear/non-monotonic (e.g., U-shaped), or to reduce sensitivity to outliers and noise.

**Q13. Answer: A**
Mahalanobis distance accounts for the covariance structure of the data, identifying multivariate outliers that might appear normal when examining each feature individually.

**Q14. Answer: A**
Feature transformations must be fit on the training set only, then applied to the test set using the same parameters. Fitting on the full dataset causes data leakage and overestimates model performance.

**Q15. Answer: D**
Interaction features model situations where the effect of one feature depends on another. For example, the effect of education on income may depend on years of experience.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# Multiple Choice Questions: Data Preprocessing and Feature Engineering

Test your understanding of data preparation and feature engineering for ML pipelines.

---

**Q1. Standardization (z-score normalization) transforms features to have:**

A) Values between 0 and 1
B) Mean of 0 and standard deviation of 1
C) All positive values
D) Equal minimum and maximum values

---

**Q2. Min-Max scaling is preferred over standardization when:**

A) The data contains many outliers
B) You need values in a bounded range (e.g., [0, 1]) and the data has no extreme outliers
C) The features follow a normal distribution
D) The dataset is very large

---

**Q3. One-hot encoding converts a categorical variable with k categories into:**

A) A single numerical column with values 1 to k
B) k binary columns, each indicating presence/absence of a category
C) k−1 continuous columns
D) A single column of probabilities

---

**Q4. The "dummy variable trap" occurs when:**

A) You forget to encode categorical variables
B) You include all k one-hot encoded columns for a k-category variable, creating perfect multicollinearity
C) You use label encoding instead of one-hot encoding
D) The categorical variable has missing values

---

**Q5. Which imputation strategy is most appropriate for a feature with many outliers?**

A) Mean imputation
B) Median imputation
C) Replacing missing values with zero
D) Deleting the entire feature

---

**Q6. Feature scaling is particularly important for which type of algorithm?**

A) Decision trees
B) Random forests
C) Gradient descent-based algorithms (e.g., logistic regression, SVMs, neural networks)
D) Rule-based systems

---

**Q7. Log transformation is commonly applied to features that are:**

A) Normally distributed
B) Right-skewed with a long positive tail
C) Uniformly distributed
D) Binary

---

**Q8. Target encoding replaces each category with:**

A) A random number
B) The mean of the target variable for that category
C) The category's frequency count
D) The one-hot encoded vector

---

**Q9. What is data leakage?**

A) When data is stored in an insecure database
B) When information from the test set or future data inadvertently influences the training process
C) When features have missing values
D) When the model is too complex

---

**Q10. Polynomial feature engineering (adding x², x₁x₂, etc.) helps when:**

A) The relationship between features and target is strictly linear
B) There are nonlinear relationships that a linear model cannot capture
C) The dataset has too many features
D) All features are categorical

---

**Q11. When handling missing data, Multiple Imputation is preferred over single imputation because:**

A) It is computationally faster
B) It accounts for the uncertainty in the imputed values by creating multiple plausible datasets
C) It always produces the same result
D) It removes the need for feature engineering

---

**Q12. Binning (discretization) of a continuous variable is useful when:**

A) You want to preserve the exact values of the variable
B) The relationship with the target is non-monotonic or you want to reduce the effect of outliers
C) The variable is already categorical
D) You have very few data points

---

**Q13. Which technique helps detect outliers in multivariate data?**

A) Checking if values exceed 3 standard deviations (works only for univariate)
B) Mahalanobis distance, which accounts for correlations between features
C) Counting missing values
D) Sorting the data by index

---

**Q14. Feature engineering should be applied:**

A) Only to the training set, then the same transformations (fitted on training) applied to test set
B) Jointly on the combined training and test sets
C) Only to the test set
D) Differently for training and test sets

---

**Q15. Interaction features (e.g., x₁ × x₂) capture:**

A) The individual effect of each feature
B) The combined effect where the influence of one feature depends on the value of another
C) The correlation between features
D) The missing value pattern

---

## Answer Key

**Q1. Answer: B**
Standardization computes z = (x − μ) / σ, resulting in mean = 0 and standard deviation = 1. This is different from Min-Max scaling which maps to [0, 1].

**Q2. Answer: B**
Min-Max scaling is best when you need bounded outputs (e.g., for neural network inputs or image pixel values) and when outliers are not a concern, since outliers compress the rest of the data into a narrow range.

**Q3. Answer: B**
One-hot encoding creates k binary columns (one per category), where exactly one column is 1 and the rest are 0 for each observation. This avoids implying ordinal relationships.

**Q4. Answer: B**
Including all k dummy columns creates perfect multicollinearity (any one column can be derived from the others). The fix is to drop one column (k−1 encoding) for linear models.

**Q5. Answer: B**
The median is robust to outliers, unlike the mean which gets pulled by extreme values. For heavily skewed data with outliers, median imputation preserves a more representative central tendency.

**Q6. Answer: C**
Gradient descent-based algorithms are sensitive to feature scales because unscaled features create elongated loss surfaces, causing slow convergence. Tree-based methods are scale-invariant.

**Q7. Answer: B**
Log transformation compresses the right tail and spreads out the left, making right-skewed distributions more symmetric and closer to normal. Common for income, prices, and counts.

**Q8. Answer: B**
Target encoding replaces each category with the mean target value for that category. It can be powerful but requires regularization (smoothing) to avoid overfitting on rare categories.

**Q9. Answer: B**
Data leakage occurs when the model has access to information during training that it wouldn't have at prediction time — such as fitting a scaler on the full dataset including the test set, or using future values as features.

**Q10. Answer: B**
Polynomial features allow linear models to capture nonlinear patterns. For example, adding x² lets a linear regression fit a parabola. However, this increases dimensionality and risk of overfitting.

**Q11. Answer: B**
Multiple imputation creates several imputed datasets, analyzes each, and pools results, properly reflecting the uncertainty from missingness. Single imputation treats imputed values as known, underestimating variance.

**Q12. Answer: B**
Binning is useful when the relationship between a continuous feature and target is non-linear/non-monotonic (e.g., U-shaped), or to reduce sensitivity to outliers and noise.

**Q13. Answer: B**
Mahalanobis distance accounts for the covariance structure of the data, identifying multivariate outliers that might appear normal when examining each feature individually.

**Q14. Answer: A**
Feature transformations must be fit on the training set only, then applied to the test set using the same parameters. Fitting on the full dataset causes data leakage and overestimates model performance.

**Q15. Answer: B**
Interaction features model situations where the effect of one feature depends on another. For example, the effect of education on income may depend on years of experience.

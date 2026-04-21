# Multiple Choice Questions: Exploratory Data Analysis

Test your understanding of EDA techniques for understanding and visualizing data.

---

**Q1. The primary goal of Exploratory Data Analysis (EDA) is to:**

A) Build the final production model
B) Understand data distributions, relationships, and anomalies before modeling
C) Deploy the model to production
D) Optimize hyperparameters

---

**Q2. A box plot displays all of the following EXCEPT:**

A) Median
B) Interquartile range (Q1 to Q3)
C) Mean
D) Potential outliers (points beyond 1.5×IQR)

---

**Q3. A correlation heatmap is most useful for identifying:**

A) Causal relationships between variables
B) Linear relationships and multicollinearity among numerical features
C) The best model to use
D) Missing value patterns

---

**Q4. Skewness measures:**

A) The spread of a distribution
B) The asymmetry of a distribution around its mean
C) The number of peaks in a distribution
D) The correlation between two variables

---

**Q5. A Q-Q plot (quantile-quantile plot) is used to:**

A) Compare the quantiles of a dataset against a theoretical distribution (typically normal)
B) Display the frequency of each category
C) Show the relationship between two continuous variables
D) Compute p-values for hypothesis tests

---

**Q6. When you discover high multicollinearity between two features during EDA, you should:**

A) Always remove both features
B) Consider removing one, combining them, or using regularization
C) Add more features to compensate
D) Ignore it — it never affects model performance

---

**Q7. A histogram with a very long right tail suggests:**

A) The data is normally distributed
B) The data is right-skewed (positively skewed)
C) The data is left-skewed
D) The data has no variance

---

**Q8. The value of a Pearson correlation coefficient r = 0 means:**

A) The two variables are completely independent
B) There is no linear relationship (but there may be a nonlinear one)
C) The two variables are identical
D) There is a perfect negative relationship

---

**Q9. During EDA, discovering that the target variable is highly imbalanced (e.g., 95% class 0, 5% class 1) suggests you should:**

A) Proceed with default model settings — imbalance never matters
B) Consider resampling techniques, class weights, or appropriate evaluation metrics like F1/AUC
C) Remove the minority class
D) Convert the problem to regression

---

**Q10. A scatter plot matrix (pair plot) is useful for:**

A) Showing only categorical variable distributions
B) Visualizing pairwise relationships between all numerical features simultaneously
C) Computing exact p-values
D) Replacing all other EDA techniques

---

**Q11. When you observe a bimodal distribution in a feature, this likely indicates:**

A) The data is normally distributed
B) The data may contain two distinct sub-populations or groups
C) There are no outliers
D) The feature is irrelevant

---

**Q12. Simpson's Paradox refers to:**

A) A pattern that appears in data only during visualization
B) A trend that appears in aggregated data but reverses when data is divided into subgroups
C) The phenomenon of data always being normally distributed
D) The impossibility of finding correlations in small datasets

---

**Q13. Value counts and frequency tables are most appropriate for analyzing:**

A) Continuous numerical variables
B) Categorical variables or discrete variables with few unique values
C) Time series data
D) High-dimensional data

---

**Q14. A violin plot combines information from:**

A) A bar chart and a pie chart
B) A box plot and a kernel density estimation (showing the distribution shape)
C) A scatter plot and a line chart
D) A histogram and a Q-Q plot

---

**Q15. The best practice when performing EDA is to:**

A) Only look at summary statistics without any visualizations
B) Combine statistical summaries with visualizations and investigate unexpected patterns before modeling
C) Skip EDA when the dataset is large
D) Only use EDA for small datasets

---

## Answer Key

**Q1. Answer: B**
EDA's purpose is to understand data characteristics — distributions, missing values, outliers, correlations, and patterns — before building models. It informs feature engineering and model selection.

**Q2. Answer: C**
A standard box plot shows the median (line), Q1-Q3 (box), whiskers (1.5×IQR), and outliers. The mean is NOT shown by default, though some variants add it as a separate marker.

**Q3. Answer: B**
Correlation heatmaps show pairwise linear relationships (Pearson r) among features. High correlations signal multicollinearity. Note: correlation does not imply causation.

**Q4. Answer: B**
Skewness quantifies asymmetry. Positive skew means a long right tail; negative skew means a long left tail. Zero skewness indicates symmetry (like a normal distribution).

**Q5. Answer: A**
Q-Q plots compare ordered sample quantiles against theoretical quantiles. Points on a straight diagonal line indicate the data matches the theoretical distribution.

**Q6. Answer: B**
High multicollinearity destabilizes coefficient estimates in linear models. Options include removing one redundant feature, creating a combined feature, or using regularization (Ridge/Lasso).

**Q7. Answer: B**
A long right tail indicates positive/right skew, where most values are concentrated on the left with some extreme high values. Common in income, house prices, and count data.

**Q8. Answer: B**
Pearson r = 0 means no LINEAR relationship. The variables could still have a strong nonlinear relationship (e.g., quadratic: y = x²). Always visualize data to check for nonlinear patterns.

**Q9. Answer: B**
Class imbalance causes models to favor the majority class. Solutions include SMOTE, undersampling, class weights, or using metrics like F1, precision-recall AUC that are sensitive to minority class performance.

**Q10. Answer: B**
Pair plots display scatter plots for all pairs of numerical features plus diagonal distributions, enabling quick identification of correlations, clusters, and nonlinear relationships.

**Q11. Answer: B**
Bimodal distributions typically indicate a mixture of two groups (e.g., male and female heights). This suggests the feature might benefit from being analyzed per subgroup.

**Q12. Answer: B**
Simpson's Paradox occurs when a trend present in aggregated data reverses in subgroups due to a confounding variable. It highlights the importance of stratified analysis.

**Q13. Answer: B**
Value counts show the frequency of each unique value, making them ideal for categorical variables or discrete variables. For continuous variables, histograms or density plots are more appropriate.

**Q14. Answer: B**
Violin plots show both the box plot summary statistics and the kernel density estimate of the distribution shape, giving a richer view of the data distribution than either alone.

**Q15. Answer: B**
Good EDA combines descriptive statistics (mean, median, std) with visualizations (histograms, scatter plots, box plots) and investigates anomalies. Skipping EDA risks building models on misunderstood data.

# Multiple Choice Questions: Exploratory Data Analysis

📺 **Video Lecture:** https://youtu.be/Io0lkeQqUXY


Test your understanding of EDA techniques for understanding and visualizing data.

---

**Q1. The primary goal of Exploratory Data Analysis (EDA) is to:**

A) Optimize hyperparameters  
B) Deploy the model to production  
C) Understand data distributions, relationships, and anomalies before modeling  
D) Build the final production model

---

**Q2. A box plot displays all of the following EXCEPT:**

A) Mean  
B) Potential outliers (points beyond 1.5×IQR)  
C) Interquartile range (Q1 to Q3)  
D) Median

---

**Q3. A correlation heatmap is most useful for identifying:**

A) Missing value patterns  
B) Linear relationships and multicollinearity among numerical features  
C) The best model to use  
D) Causal relationships between variables

---

**Q4. Skewness measures:**

A) The number of peaks in a distribution  
B) The spread of a distribution  
C) The correlation between two variables  
D) The asymmetry of a distribution around its mean

---

**Q5. A Q-Q plot (quantile-quantile plot) is used to:**

A) Compare the quantiles of a dataset against a theoretical distribution (typically normal)  
B) Compute p-values for hypothesis tests  
C) Show the relationship between two continuous variables  
D) Display the frequency of each category

---

**Q6. When you discover high multicollinearity between two features during EDA, you should:**

A) Ignore it — it never affects model performance  
B) Always remove both features  
C) Consider removing one, combining them, or using regularization  
D) Add more features to compensate

---

**Q7. A histogram with a very long right tail suggests:**

A) The data has no variance  
B) The data is normally distributed  
C) The data is left-skewed  
D) The data is right-skewed (positively skewed)

---

**Q8. The value of a Pearson correlation coefficient r = 0 means:**

A) There is a perfect negative relationship  
B) The two variables are identical  
C) There is no linear relationship (but there may be a nonlinear one)  
D) The two variables are completely independent

---

**Q9. During EDA, discovering that the target variable is highly imbalanced (e.g., 95% class 0, 5% class 1) suggests you should:**

A) Proceed with default model settings — imbalance never matters  
B) Convert the problem to regression  
C) Consider resampling techniques, class weights, or appropriate evaluation metrics like F1/AUC  
D) Remove the minority class

---

**Q10. A scatter plot matrix (pair plot) is useful for:**

A) Visualizing pairwise relationships between all numerical features simultaneously  
B) Showing only categorical variable distributions  
C) Computing exact p-values  
D) Replacing all other EDA techniques

---

**Q11. When you observe a bimodal distribution in a feature, this likely indicates:**

A) The data may contain two distinct sub-populations or groups  
B) The data is normally distributed  
C) There are no outliers  
D) The feature is irrelevant

---

**Q12. Simpson's Paradox refers to:**

A) The impossibility of finding correlations in small datasets  
B) A trend that appears in aggregated data but reverses when data is divided into subgroups  
C) The phenomenon of data always being normally distributed  
D) A pattern that appears in data only during visualization

---

**Q13. Value counts and frequency tables are most appropriate for analyzing:**

A) High-dimensional data  
B) Categorical variables or discrete variables with few unique values  
C) Continuous numerical variables  
D) Time series data

---

**Q14. A violin plot combines information from:**

A) A bar chart and a pie chart  
B) A scatter plot and a line chart  
C) A histogram and a Q-Q plot  
D) A box plot and a kernel density estimation (showing the distribution shape)

---

**Q15. The best practice when performing EDA is to:**

A) Only look at summary statistics without any visualizations  
B) Combine statistical summaries with visualizations and investigate unexpected patterns before modeling  
C) Skip EDA when the dataset is large  
D) Only use EDA for small datasets

---

## Answer Key

**Q1. Answer: C**
EDA's purpose is to understand data characteristics — distributions, missing values, outliers, correlations, and patterns — before building models. It informs feature engineering and model selection.

**Q2. Answer: A**
A standard box plot shows the median (line), Q1-Q3 (box), whiskers (1.5×IQR), and outliers. The mean is NOT shown by default, though some variants add it as a separate marker.

**Q3. Answer: B**
Correlation heatmaps show pairwise linear relationships (Pearson r) among features. High correlations signal multicollinearity. Note: correlation does not imply causation.

**Q4. Answer: D**
Skewness quantifies asymmetry. Positive skew means a long right tail; negative skew means a long left tail. Zero skewness indicates symmetry (like a normal distribution).

**Q5. Answer: A**
Q-Q plots compare ordered sample quantiles against theoretical quantiles. Points on a straight diagonal line indicate the data matches the theoretical distribution.

**Q6. Answer: C**
High multicollinearity destabilizes coefficient estimates in linear models. Options include removing one redundant feature, creating a combined feature, or using regularization (Ridge/Lasso).

**Q7. Answer: D**
A long right tail indicates positive/right skew, where most values are concentrated on the left with some extreme high values. Common in income, house prices, and count data.

**Q8. Answer: C**
Pearson r = 0 means no LINEAR relationship. The variables could still have a strong nonlinear relationship (e.g., quadratic: y = x²). Always visualize data to check for nonlinear patterns.

**Q9. Answer: C**
Class imbalance causes models to favor the majority class. Solutions include SMOTE, undersampling, class weights, or using metrics like F1, precision-recall AUC that are sensitive to minority class performance.

**Q10. Answer: A**
Pair plots display scatter plots for all pairs of numerical features plus diagonal distributions, enabling quick identification of correlations, clusters, and nonlinear relationships.

**Q11. Answer: A**
Bimodal distributions typically indicate a mixture of two groups (e.g., male and female heights). This suggests the feature might benefit from being analyzed per subgroup.

**Q12. Answer: B**
Simpson's Paradox occurs when a trend present in aggregated data reverses in subgroups due to a confounding variable. It highlights the importance of stratified analysis.

**Q13. Answer: B**
Value counts show the frequency of each unique value, making them ideal for categorical variables or discrete variables. For continuous variables, histograms or density plots are more appropriate.

**Q14. Answer: D**
Violin plots show both the box plot summary statistics and the kernel density estimate of the distribution shape, giving a richer view of the data distribution than either alone.

**Q15. Answer: B**
Good EDA combines descriptive statistics (mean, median, std) with visualizations (histograms, scatter plots, box plots) and investigates anomalies. Skipping EDA risks building models on misunderstood data.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

# Multiple Choice Questions: Python Coding for ML

📺 **Video Lecture:** https://youtu.be/RhmUGXcQSUc

## Question 1
You have a NumPy array `X` of shape (1000, 10) and a 1D array `means` of shape (10,). You want to center each feature by subtracting its mean. What will be the result of `X - means`?

A) Shape error; broadcasting fails because shapes (1000, 10) and (10,) are incompatible  
B) The means array is broadcasted to shape (1000, 10), and each row has the means subtracted element-wise  
C) Only the first row of X gets the means subtracted  
D) NumPy raises a "shapes could not be broadcast together" exception

**Answer: B**

---

## Question 2
Which of the following is the FASTEST way to compute a new column in a Pandas DataFrame where each value equals the sum of two existing columns?

A) `df['new_col'] = [df.loc[i, 'A'] + df.loc[i, 'B'] for i in range(len(df))]`  
B) `df['new_col'] = df['A'] + df['B']`  
C) `for idx, row in df.iterrows(): df.loc[idx, 'new_col'] = row['A'] + row['B']`  
D) `df.apply(lambda row: row['A'] + row['B'], axis=1)` assigned to `df['new_col']`

**Answer: B**

---

## Question 3
What is the primary issue with this code for computing user lifetime statistics?

```python
user_lifetime_spend = df.groupby('user_id')['amount'].sum()
train_data = df[df['date'] < '2024-01-01']
train_data['lifetime_spend'] = train_data['user_id'].map(user_lifetime_spend)
```

A) The groupby operation is inefficient; use `.agg()` instead  
B) Data leakage occurs because `user_lifetime_spend` includes transactions after the cutoff date  
C) The `.map()` function is deprecated; use `.merge()` instead  
D) Categorical dtype should be applied to user_id before grouping

**Answer: B**

---

## Question 4
You have a dictionary of feature engineering functions that are slow to compute. How would you cache results to avoid recomputation while maintaining reproducibility?

A) Use `@lru_cache` on the function; it automatically handles all edge cases  
B) Use `@lru_cache` for pure functions, or `joblib.Memory` for filesystem-backed caching that persists across sessions  
C) Store results in a global dictionary; it's faster than caching  
D) Don't cache; just re-run computations; caching introduces bugs

**Answer: B**

---

## Question 5
What does `np.random.permutation(len(X))` do, and why is it useful in ML?

A) Returns a sorted array of indices; used to verify data is sorted  
B) Returns random values between 0 and 1; used to initialize model weights  
C) Returns a shuffled array of indices [0, 1, ..., n-1]; useful for randomizing data order during training (e.g., SGD)  
D) Removes duplicate rows from X; used for deduplication

**Answer: C**

---

## Question 6
Which scikit-learn class helps prevent data leakage by ensuring preprocessing is fit only on training data?

A) `GridSearchCV` with `cv=5`  
B) `Pipeline`, which fits transformers only on training data and applies them consistently  
C) `StandardScaler` applied directly to all data before splitting  
D) `StratifiedKFold` with manual scaling in each fold

**Answer: B**

---

## Question 7
You want to compute the distance from each of n_samples points to k_centroids without explicit loops. Which is the correct vectorized approach?

A) `distances = np.zeros((n_samples, k_centroids)); for i in range(n_samples): for j in range(k_centroids): distances[i, j] = np.sum((X[i] - centroids[j]) ** 2)`  
B) `distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)`  
C) `distances = X @ centroids.T` (simple matrix multiplication)  
D) `distances = [np.linalg.norm(X[i] - centroids[j]) for i in range(n_samples) for j in range(k_centroids)]`

**Answer: B**

---

## Question 8
Your DataFrame has 1 billion rows and you want to reduce memory usage. You notice a column `country` has only 100 unique string values. What optimization should you apply?

A) Convert to `int64` dtype; integers are more efficient than strings  
B) Convert to `category` dtype; it stores indices instead of repeated strings, reducing memory significantly  
C) Use `.drop()` to remove the column; it's not useful for ML  
D) Apply `pd.cut()` to create bins; categorical binning always reduces memory

**Answer: B**

---

## Question 9
When should you use `multiprocessing` or `joblib.Parallel` for ML code?

A) For all computations; parallelization always makes code faster  
B) For I/O-bound tasks like network requests; threading is more efficient  
C) For CPU-bound tasks like feature engineering loops or cross-validation grid search; avoid parallelizing very fast computations where overhead dominates  
D) Only for data loading; parallelization doesn't help with model training

**Answer: C**

---

## Question 10
What is the purpose of storing computed statistics with a trailing underscore (e.g., `self.median_`, `self.scaler_`)?

A) It's a Python syntax requirement for sklearn transformers  
B) It signals that these are fit-time attributes computed during `.fit()`, following sklearn convention  
C) It indicates the attribute is private and should not be accessed  
D) It has no special meaning; it's just a naming preference

**Answer: B**

---

## Question 11
You have a custom transformer that computes a log transformation. To integrate it into a scikit-learn Pipeline, what must it inherit from?

A) `object` and implement `fit()` and `transform()` methods  
B) `BaseEstimator` and `TransformerMixin`, implementing `fit()` and `transform()` methods  
C) `Pipeline` class directly  
D) Only `TransformerMixin`; inheritance from `BaseEstimator` is optional

**Answer: B**

---

## Question 12
When using k-fold cross-validation with imbalanced classes, what should you use to prevent class distribution shifts between folds?

A) Random shuffle with `shuffle=True` and high `random_state`  
B) `StratifiedKFold` to preserve class proportions in each fold  
C) Manual train/test split with `sklearn.model_selection.train_test_split`  
D) Time series split; it automatically handles imbalance

**Answer: B**

---

## Question 13
You want to measure which line of code is taking the most time in your model training function. Which tool is most appropriate?

A) `cProfile` for function-level analysis  
B) `timeit` for repeating a function many times  
C) `line_profiler` with `@profile` decorator for line-level analysis  
D) `time.time()` manually around each line

**Answer: C**

---

## Question 14
What does the `with` statement do in Python context managers, and why is it important for ML pipelines?

A) It's equivalent to an `if` statement; no special behavior  
B) It ensures setup and cleanup operations are executed reliably, preventing resource leaks (e.g., temp files, database connections)  
C) It automatically parallelizes code across multiple cores  
D) It eliminates the need for try/except blocks

**Answer: B**

---

## Question 15
Type hints like `def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]` have what benefit for ML teams?

A) They enable runtime type checking and prevent all bugs automatically  
B) They document function signatures, allow static type checkers (mypy, pyright) to catch errors, and improve IDE autocomplete support  
C) They require zero additional documentation because types replace written descriptions  
D) They slow down code execution; use only for documentation, not in production

**Answer: B**

---

## Answer Key

**Q1: B** - NumPy broadcasting aligns shapes by adding dimensions on the left. Shape (10,) becomes (1, 10), then expands to (1000, 10), subtracting from each row. This is fundamental to vectorized ML operations.

**Q2: B** - Vectorized operations on Series are 100-1000x faster than loops or `.iterrows()`. Always prefer `df['col_A'] + df['col_B']` over explicit iteration.

**Q3: B** - Computing statistics on the entire dataset before splitting causes temporal leakage. User lifetime spend computed on all data includes future transactions. Fix: compute statistics only on data before the cutoff date.

**Q4: B** - `@lru_cache` caches in memory; `joblib.Memory` persists to disk and supports cross-session reproducibility. Choose based on whether you need persistence and dataset size.

**Q5: C** - Shuffling data order is critical for SGD convergence and prevents the model from learning batch patterns. `np.random.permutation()` returns indices, allowing `X[shuffled_indices]` to randomize.

**Q6: B** - Pipelines fit preprocessing only on training data, preventing leakage. `StandardScaler` applied to all data before splitting would include test statistics in the scaler.

**Q7: B** - Broadcasting creates (n_samples, 1, n_features) - (1, k_centroids, n_features), producing (n_samples, k_centroids, n_features), then `.sum(axis=2)` collapses to distances. Explicit loops are ~1000x slower.

**Q8: B** - `category` dtype stores integer indices instead of strings, reducing memory per column from 8*n_rows to ~log2(n_unique)*n_rows. Critical optimization for billion-row tables.

**Q9: C** - CPU-bound tasks like feature engineering and grid search benefit from multiprocessing (bypasses GIL). I/O-bound tasks (network, file) use threading/async. Parallelization overhead only matters for fast operations.

**Q10: B** - Trailing underscore signals fit-time attributes in sklearn convention. Users know `self.median_` exists only after `.fit()`, preventing accidental use of unfitted transformers.

**Q11: B** - Both `BaseEstimator` (provides `get_params()`, `set_params()` for sklearn integration) and `TransformerMixin` (provides `fit_transform()` shorthand) are required for seamless Pipeline integration.

**Q12: B** - `StratifiedKFold` preserves class ratios in each fold, preventing training on 90% positive data while validating on 10% (or vice versa). Critical for imbalanced classification.

**Q13: C** - `line_profiler` shows time per line, pinpointing exact bottlenecks. `cProfile` is too high-level; `timeit` measures repetitions, not line-level breakdown.

**Q14: B** - Context managers ensure cleanup (files closed, resources freed) even if exceptions occur. Essential for ML pipelines handling temp files, database connections, or GPU memory.

**Q15: B** - Type hints enable static checkers to catch bugs before runtime and help teams understand expected inputs/outputs without reading function bodies. Zero runtime cost; pure documentation and IDE support.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

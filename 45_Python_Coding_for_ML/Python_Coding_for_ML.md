# Python Coding for ML

📺 **Video Lecture:** https://youtu.be/RhmUGXcQSUc

## Interview Anchor
- **NumPy Essentials:** Vectorization and broadcasting for efficient numerical operations without Python loops
- **Pandas Mastery:** Data manipulation (groupby, merge, pivot_table, apply) for preparing ML datasets
- **Memory and Performance:** Optimizing code with dtypes, chunked reading, and profiling for large-scale ML pipelines

## Key Concepts Overview
Python is the dominant language for ML, and writing efficient, correct code is non-negotiable in production systems. Interviewers assess your ability to write vectorized code (not loops), handle large datasets efficiently, and debug common ML pitfalls like data leakage and shape mismatches. Understanding NumPy's broadcasting, Pandas' groupby mechanics, and scikit-learn's API patterns shows you can build scalable pipelines that don't crash on real data. Strong Python fundamentals—memory optimization, testing, and type hints—separate junior practitioners from ML engineers who can ship reliable systems.

---

### Q1: Explain NumPy broadcasting and provide three examples of its utility in ML.

**A:** Broadcasting allows arrays of different shapes to be combined element-wise without explicit loops. When operating on arrays of shapes (m, n) and (n,), NumPy aligns them by adding dimensions on the left and repeating the smaller array: `X = np.random.randn(1000, 10); means = X.mean(axis=0)  # shape (10,); X_centered = X - means  # broadcasts means to (1000, 10)` centers each feature without looping. Example 1 (feature scaling): `X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)` normalizes all 1000 samples across 10 features in one operation. Example 2 (distance computation): `distances = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))` with shapes (n_samples, 1, n_features) and (1, n_centroids, n_features) creates an (n_samples, n_centroids) distance matrix for k-means without loops. Example 3 (batch normalization): `X_norm = (X - batch_mean[:, None]) / batch_std[:, None]` reshapes (n_features,) into (n_features, 1) to broadcast across batches. Broadcasting eliminates explicit loops, making code 100x faster and more readable.

---

### Q2: What is vectorization and why is it critical for ML performance? Write a slow and fast version.

**A:** Vectorization replaces Python loops with NumPy/Pandas operations on entire arrays, leveraging C-level optimizations. Slow (loop): `result = []; for x in X: result.append(x ** 2 + 2 * x + 1)` iterates row-by-row in pure Python. Fast (vectorized): `result = X ** 2 + 2 * X + 1` computes element-wise on the entire array at C speed. For a 1M-element array, vectorized code is 100-1000x faster. In Pandas, slow version `for idx, row in df.iterrows(): df.loc[idx, 'new_col'] = row['A'] + row['B']` uses Python iteration; fast version `df['new_col'] = df['A'] + df['B']` operates on Series. The performance gap widens with dataset size—on 1B rows, loop-based code becomes infeasible while vectorized code runs in seconds. ML practitioners must internalize this: always ask "can I avoid a loop?" before writing iteration code. Apply the mindset to pandas groupby (use `.agg()` with vectorized functions), NumPy operations (prefer `np.dot()` to explicit matrix multiplication), and scikit-learn (batching data through `.transform()`).

---

### Q3: Explain NumPy's advanced indexing and provide an ML use case.

**A:** Advanced indexing uses arrays or boolean masks to select elements non-sequentially. Boolean indexing: `mask = X[:, 0] > 5; X_filtered = X[mask]` selects rows where the first column exceeds 5. Integer array indexing: `indices = np.array([0, 3, 1, 2]); X_reordered = X[indices]` reorders rows by index. Fancy indexing combines both: `row_indices = np.where(y == 1)[0]; feature_indices = np.array([0, 2, 4]); X_subset = X[np.ix_(row_indices, feature_indices)]` selects specific rows and columns, useful for extracting positive samples and their relevant features. For ML: `top_k_indices = np.argsort(scores)[-10:]; top_predictions = predictions[top_k_indices]` retrieves the top 10 highest scores without sorting the entire array. Another use case: `X_train[np.random.permutation(len(X_train))]` shuffles data in-place for SGD training. Masking is critical for imbalanced learning: `X_minority = X[y == 1]; X_majority = X[y == 0]` separates classes for SMOTE or stratified sampling. Advanced indexing is faster and more expressive than loops, fundamental for efficient ML preprocessing.

---

### Q4: How do you optimize Pandas operations? Explain dtypes and chunked reading.

**A:** Dtypes determine memory usage; using the smallest sufficient type saves memory. For example, `df['user_id'] = df['user_id'].astype('uint32')` uses 4 bytes instead of 8 (int64), reducing a 1B-row table by 4GB. Categorical dtypes compress string columns: `df['country'] = df['country'].astype('category')` transforms a column with 100 repeated values from 8 bytes per string to 1 byte per index. Dates should use datetime64: `df['date'] = pd.to_datetime(df['date'])` enables fast date filtering and arithmetic. For a 1M-row table, strategic dtype selection reduces memory from 8GB to 2GB. Chunked reading loads data in batches: `for chunk in pd.read_csv('large.csv', chunksize=100000): process(chunk)` avoids loading 10B rows into memory simultaneously. In Pandas, avoid iterrows(): `df.iterrows()` is slow; instead use `.apply()`, `.groupby().agg()`, or vectorized operations. Profile memory with `df.memory_usage(deep=True).sum()`. For ML pipelines, combine chunked reading with feature engineering: compute aggregates per chunk (e.g., user statistics), then concatenate results, avoiding full-data materialization.

---

### Q5: Explain Pandas groupby and provide three aggregation patterns for feature engineering.

**A:** Groupby aggregates data by one or more columns, essential for creating group-level features. Basic syntax: `df.groupby('user_id')['amount'].sum()` sums amounts per user. Pattern 1 (multi-column aggregation): `user_stats = df.groupby('user_id').agg({'amount': ['sum', 'mean', 'std'], 'timestamp': 'count'}).reset_index()` computes multiple aggregates in one pass, creating a wide feature table. Column names become tuples; rename with `columns = ['_'.join(col).strip() for col in result.columns.values]; user_stats.columns = columns`. Pattern 2 (conditional aggregation): `df.groupby('user_id').apply(lambda g: pd.Series({'high_spend': (g['amount'] > g['amount'].quantile(0.75)).sum(), 'purchase_count': len(g)}))` counts high-value purchases per user. Pattern 3 (transform to broadcast aggregates back): `df['user_mean_spend'] = df.groupby('user_id')['amount'].transform('mean')` appends group statistics to original dataframe, enabling features like deviation from user average. Groupby is memory-efficient (processes one group at a time) and blazingly fast, making it the preferred method for aggregation in pandas-based pipelines. Use `.reset_index()` to convert group keys to columns for joining back to original data.

---

### Q6: What are common data leakage bugs and how do you avoid them?

**A:** Data leakage occurs when information from the future or test set contaminates training. Bug 1 (temporal leakage): computing user lifetime statistics before split—`user_lifetime_spend = df.groupby('user_id')['amount'].sum(); train = df[df['date'] < cutoff]; train['lifetime_spend'] = train['user_id'].map(user_lifetime_spend)` includes spending from after the cutoff. Fix: compute statistics only on data before the cutoff date. Bug 2 (group leakage): if user_id appears in both train and test, group-level features leak information from test into training. Fix: ensure no user_id appears in both splits (stratified split by user_id). Bug 3 (target leakage): using highly correlated feature unknowable at prediction time, like "purchase_was_shipped" when predicting purchase intent. Fix: only use features computable before the target is realized. Bug 4 (scaling leakage): fitting a scaler on all data before splitting, then applying to train—the scaler's statistics are influenced by test data. Fix: fit scaler on train, apply to test. Implement this rigorously: `scaler.fit(X_train); X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)`. Best practice: wrap preprocessing in a function that operates strictly on temporal or split boundaries, making leakage obvious.

---

### Q7: Explain the scikit-learn Pipeline and ColumnTransformer, and why they prevent leakage.

**A:** Pipeline chains preprocessing and modeling steps, fitting transformers only on training data: `pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())]); pipe.fit(X_train, y_train); pred = pipe.predict(X_test)` automatically fits the scaler on X_train (never touching X_test), then applies the fitted scaler to X_test, preventing leakage. ColumnTransformer applies different transformations to different feature subsets: `ct = ColumnTransformer([('scale_num', StandardScaler(), ['age', 'income']), ('onehot_cat', OneHotEncoder(), ['country'])])` scales numeric columns and one-hots categorical columns independently. Combining both: `full_pipe = Pipeline([('preprocess', ct), ('model', RandomForestClassifier())])` ensures the categorical encoder and scaler are fit only on training data, then consistently applied to test. The key benefit: `.fit()` is called once, `.transform()` is never called on training data during test evaluation, eliminating leakage. For custom preprocessing, write a sklearn-compatible transformer with `.fit()` and `.transform()` methods, inheriting from BaseEstimator and TransformerMixin—this integrates seamlessly into pipelines. Using pipelines is non-negotiable for production ML code; it forces clean data flow and makes leakage obvious.

---

### Q8: How do you write custom transformers for scikit-learn pipelines?

**A:** Inherit from BaseEstimator and TransformerMixin, implement `fit(X, y=None)` and `transform(X)`: `from sklearn.base import BaseEstimator, TransformerMixin; class LogTransformer(BaseEstimator, TransformerMixin): def fit(self, X, y=None): return self; def transform(self, X): return np.log1p(X)` creates a log transformer that safely handles zeros with log1p. For stateful transformers, store computed statistics in fit(): `class RobustScaler_Custom(BaseEstimator, TransformerMixin): def fit(self, X, y=None): self.median_ = np.median(X, axis=0); self.iqr_ = np.percentile(X, 75) - np.percentile(X, 25); return self; def transform(self, X): return (X - self.median_) / self.iqr_` computes median and IQR on train, applies them to test. The underscore suffix on `median_` and `iqr_` signals they're fit-time attributes (scikit-learn convention). For column-specific logic, combine with ColumnTransformer: `ct = ColumnTransformer([('my_transform', LogTransformer(), ['feature1'])])`. Custom transformers enable domain-specific preprocessing (e.g., handling timestamps, encoding cyclical features, denoising) while maintaining the clean train/test boundary that pipelines enforce.

---

### Q9: Explain multiprocessing and joblib for parallelizing ML code. When is parallelization beneficial?

**A:** Multiprocessing uses multiple OS processes to avoid Python's GIL, enabling true parallelism for CPU-bound tasks. Example: `from multiprocessing import Pool; def compute_feature(user_data): return expensive_function(user_data); with Pool(4) as p: results = p.map(compute_feature, user_list)` distributes computation across 4 processes. Joblib simplifies this with caching: `from joblib import Parallel, delayed; results = Parallel(n_jobs=4)(delayed(compute_feature)(user) for user in user_list)` auto-parallelizes and caches results, avoiding recomputation. Parallelization is beneficial for: (1) grid search in scikit-learn (`GridSearchCV(model, param_grid, n_jobs=-1)` uses all cores), (2) feature engineering on multiple groups (`.groupby().apply(func, n_jobs=-1)`), (3) cross-validation (split folds across cores). Parallelization is NOT beneficial for: (1) I/O-bound tasks (network requests) where async/threading is better, (2) very fast computations where overhead dominates, (3) shared memory operations (synchronization slows parallel code). Debugging parallel code is hard; use `n_jobs=1` to run serially for development. Profile before parallelizing: if your function is a Python loop, parallelize the loop, not the function. Joblib is preferred because it caches results and integrates with scikit-learn's API.

---

### Q10: How do you profile and optimize ML code? Walk through an example.

**A:** Use cProfile to identify bottlenecks: `import cProfile; cProfile.run('my_training_loop()', sort='cumtime')` lists functions by cumulative time. Then use line_profiler for line-level analysis: `@profile def train_model(X, y): features = [expensive_feature(x) for x in X]; model.fit(features, y)` annotates the function, run with `kernprof -l -v script.py` to see per-line time. Memory profiling uses memory_profiler: `@profile def load_data(): X = np.random.randn(100000, 1000); return X` shows memory growth per line with `python -m memory_profiler script.py`. Example optimization: you discover that `expensive_feature(x)` takes 70% of time and is called 1M times. First check: is it vectorizable? If yes: `features = vectorized_expensive_function(X)` (1000x speedup). If not: is it cacheable? Use `@lru_cache` or joblib.Memory for repeated calls. Last resort: parallelize with joblib. Always measure before and after: `timeit.timeit('expensive_function(x)', number=1000)` confirms speedup. For ML pipelines, profile on realistic data sizes; optimizations that matter on 1M rows might be irrelevant on 1B rows.

---

### Q11: Implement k-means clustering from scratch in NumPy (simplified version).

**A:** `import numpy as np; def kmeans(X, k, max_iter=100): n, d = X.shape; centroids = X[np.random.choice(n, k, replace=False)]; for _ in range(max_iter): distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2); labels = np.argmin(distances, axis=1); new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)]); if np.allclose(centroids, new_centroids): break; centroids = new_centroids; return centroids, labels` initializes k centroids from data, iteratively (1) assigns points to nearest centroid, (2) recomputes centroids as cluster means, (3) checks convergence. The key insight: `distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)` uses broadcasting to compute all pairwise distances in O(1) Python loops. Compared to `for i in range(n): for j in range(k): dist[i, j] = ((X[i] - centroids[j]) ** 2).sum()`, vectorized code is 1000x faster. This implementation is inefficient (recomputes distances every iteration; production k-means uses mini-batch); the goal is demonstrating numpy proficiency, not optimality.

---

### Q12: Implement logistic regression from scratch using NumPy and gradient descent.

**A:** `import numpy as np; class LogisticRegression: def fit(self, X, y, lr=0.01, epochs=100): self.w = np.zeros(X.shape[1]); self.b = 0; for _ in range(epochs): z = X @ self.w + self.b; probs = 1 / (1 + np.exp(-z)); dw = (X.T @ (probs - y)) / len(y); db = np.mean(probs - y); self.w -= lr * dw; self.b -= lr * db; return self; def predict(self, X): return 1 / (1 + np.exp(-(X @ self.w + self.b))) > 0.5` implements SGD-style gradient descent with vectorized loss computation. Key points: (1) `z = X @ self.w` uses matrix multiplication (n_samples, n_features) @ (n_features,) = (n_samples,), (2) sigmoid `1 / (1 + np.exp(-z))` computes probabilities, (3) gradients `dw = X.T @ (probs - y)` are vectorized (no loops), (4) weight updates use learning rate `lr`. This demonstrates understanding of: backprop, vectorization, numerical stability (be careful with `np.exp(-z)` for large z), and the logistic sigmoid. Production implementations (scikit-learn, TensorFlow) add regularization, better optimizers (Adam, LBFGS), and numerical stability tricks. Interviewers ask this to assess whether you understand ML fundamentals deeply, not just API usage.

---

### Q13: How do you implement cross-validation from scratch? When would you use it?

**A:** `def cross_val_score(X, y, model, k=5): fold_size = len(X) // k; scores = []; indices = np.arange(len(X)); np.random.shuffle(indices); for fold in range(k): test_idx = indices[fold * fold_size:(fold + 1) * fold_size]; train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]]); model.fit(X[train_idx], y[train_idx]); score = model.score(X[test_idx], y[test_idx]); scores.append(score); return np.mean(scores), np.std(scores)` splits data into k folds, trains on k-1 folds, evaluates on the held-out fold, repeating k times. Use cross-validation to: (1) estimate generalization error when data is limited (small datasets), (2) detect overfitting (training loss >> CV loss), (3) select hyperparameters (choose k that maximizes CV score). For large datasets (>100k samples), stratified k-fold preserves class distribution: `from sklearn.model_selection import StratifiedKFold; skf = StratifiedKFold(n_splits=5, shuffle=True); for train_idx, test_idx in skf.split(X, y)` ensures each fold has similar class proportions. Time series data requires time series split (no shuffling, forward-looking test set) to prevent temporal leakage. Cross-validation is computationally expensive (trains model k times), so for production, use it sparingly—e.g., during hyperparameter tuning, not per prediction.

---

### Q14: What are decorators and context managers? Provide ML pipeline examples.

**A:** Decorators wrap functions to add behavior without modifying the function body. Example: `def timing_decorator(func): import time; def wrapper(*args, **kwargs): start = time.time(); result = func(*args, **kwargs); print(f"Execution time: {time.time() - start}"); return result; return wrapper; @timing_decorator; def train_model(X, y): model.fit(X, y)` measures training time without cluttering the function. Context managers ensure setup/cleanup with `with` statements. Example: `from contextlib import contextmanager; @contextmanager; def model_timer(): import time; start = time.time(); yield; print(f"Elapsed: {time.time() - start}"); with model_timer(): model.fit(X_train, y_train)` automatically measures timing. For ML, use decorators for caching results: `from functools import lru_cache; @lru_cache(maxsize=128); def compute_feature(user_id): ...` avoids recomputing expensive features. Use context managers for resource management: `with tempfile.TemporaryDirectory() as tmpdir: model.save(f"{tmpdir}/model.pkl")` cleans up temporary files automatically. Another example: `with warnings.catch_warnings(): warnings.simplefilter("ignore"); sklearn_model.fit(X, y)` suppresses non-critical warnings during training. These patterns improve code readability and prevent bugs (forgotten cleanup, multiple timing calls).

---

### Q15: How do you add type hints to ML code and why does it matter?

**A:** Type hints document function signatures and enable static type checking. Example: `from typing import Tuple, List; import numpy as np; def train_model(X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> Tuple[np.ndarray, float]: w = np.zeros(X.shape[1]); loss = compute_loss(X, y, w); return w, loss` specifies X and y are ndarrays, lr is float, and the function returns a tuple of ndarray and float. Type hints prevent bugs: if you call `train_model(X, y, "0.01")` (string instead of float), a type checker (mypy, pyright) catches it before runtime. For complex types: `from typing import Dict, Optional; def process_data(config: Dict[str, float], cache: Optional[dict] = None) -> Dict[str, np.ndarray]` clarifies that config is a dict of strings to floats, cache is optional, and return is a dict of strings to arrays. Type hints are especially valuable in teams: they serve as live documentation that IDEs enforce, catching errors in IDE autocomplete. For ML pipelines: `def preprocess(X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]` makes the transformation explicit. Runtime checking with pydantic for data validation: `from pydantic import BaseModel; class ModelConfig(BaseModel): learning_rate: float; batch_size: int` ensures input data matches expected schema. Type hints have zero runtime cost and improve code quality dramatically, a standard in production ML code.

---

## Interview Cheatsheet

**Key Terms:**

- **Broadcasting:** Aligns arrays of different shapes for element-wise operations without explicit loops
- **Vectorization:** Replacing Python loops with array operations for 100-1000x speedup
- **Advanced Indexing:** Using boolean masks or arrays to select non-sequential elements
- **Dtype:** Data type specifying memory usage (int32 vs. int64, category, datetime64)
- **Chunked Reading:** Loading data in batches to avoid loading entire large datasets into memory
- **Groupby:** Aggregating data by one or more columns for feature engineering
- **Data Leakage:** Using future or test set information to contaminate training data
- **Pipeline:** Chain of preprocessing and modeling steps that prevents leakage
- **ColumnTransformer:** Applies different transformations to different feature subsets
- **Custom Transformer:** User-defined class with fit() and transform() for sklearn compatibility
- **Multiprocessing:** Using multiple OS processes to parallelize CPU-bound tasks
- **Profiling:** Identifying performance bottlenecks using cProfile, line_profiler, memory_profiler
- **Cross-Validation:** Splitting data into k folds to estimate generalization error
- **Decorator:** Function wrapper that adds behavior without modifying the original function
- **Context Manager:** Resource management with setup/cleanup using `with` statements
- **Type Hints:** Annotations documenting function signatures and enabling static type checking

**Rapid-Fire Q&A:**

- **Q:** How do you compute pairwise distances between n_samples and k_centroids without a loop? **A:** `distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)` uses broadcasting to create (n_samples, k_centroids) distance matrix.
- **Q:** What does `np.argsort()[-k:]` do? **A:** Returns indices of the top k largest elements, useful for selecting top predictions.
- **Q:** How do you find rows with NaN values? **A:** `df[df.isna().any(axis=1)]` returns rows with at least one NaN.
- **Q:** What's the difference between `.apply()` and `.transform()` in groupby? **A:** `.apply()` returns a scalar per group (one-to-one reduction); `.transform()` returns same-sized output (broadcasts back to original size).
- **Q:** How do you prevent overfitting during hyperparameter tuning? **A:** Use cross-validation to tune on training data, validate on unseen fold.
- **Q:** What is the GIL and how does it affect parallelization? **A:** Python's Global Interpreter Lock prevents multiple threads from running code simultaneously; use multiprocessing for CPU-bound tasks, not threading.
- **Q:** How do you cache function results in Python? **A:** Use `@functools.lru_cache()` for pure functions or `joblib.Memory()` for filesystem-backed caching.
- **Q:** How do you handle imbalanced classes during cross-validation? **A:** Use `StratifiedKFold` to preserve class distribution in each fold.
- **Q:** What's the `.fit_transform()` shorthand in sklearn? **A:** Equivalent to `.fit().transform()` but can be more efficient (single pass for some transformers).
- **Q:** How do you ensure reproducibility in ML code? **A:** Set random seeds: `np.random.seed(42); random.seed(42); sklearn.set_config(random_state=42)`.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

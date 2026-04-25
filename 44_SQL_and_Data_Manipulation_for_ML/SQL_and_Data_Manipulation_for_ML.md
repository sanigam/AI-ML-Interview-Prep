# SQL and Data Manipulation for ML

📺 **Video Lecture:** https://youtu.be/HLPBLmkYABE

## Interview Anchor
- **SQL Fundamentals:** Core queries (SELECT, WHERE, GROUP BY, HAVING, ORDER BY) that filter, aggregate, and sort data for ML feature engineering
- **Advanced Joins:** Different join types (INNER, LEFT, RIGHT, FULL OUTER, CROSS, self-joins) used to combine datasets from multiple tables for feature creation
- **Window Functions:** Analytics functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, running aggregates) that enable time-series and sequential feature engineering without self-joins

## Key Concepts Overview
SQL is the backbone of data manipulation in ML pipelines, enabling efficient feature engineering at scale. Unlike pandas operations on small datasets, SQL allows you to work directly with databases and create features on millions of rows without loading everything into memory. Understanding SQL window functions, joins, and query optimization is critical because many ML roles involve working with data engineers and building reproducible, efficient pipelines. Interviewers test your SQL skills to assess whether you can solve real-world data problems and collaborate effectively with data infrastructure teams.

---

### Q1: Explain the difference between WHERE and HAVING clauses. When would you use each?

**A:** WHERE filters rows before aggregation (operates on individual rows), while HAVING filters groups after aggregation (operates on aggregated results). For example, `WHERE salary > 50000` removes low-salary rows before grouping, but `HAVING AVG(salary) > 75000` removes departments where the average salary is below 75000. Use WHERE when filtering raw data or non-aggregated columns, and use HAVING when applying conditions to aggregate functions like COUNT(), SUM(), or AVG(). A common pattern: `SELECT department, AVG(salary) FROM employees WHERE hire_date > '2020-01-01' GROUP BY department HAVING COUNT(*) > 10` filters employees hired recently (WHERE), groups by department, then only shows departments with more than 10 recent hires (HAVING).

---

### Q2: Describe the four main types of JOINs (INNER, LEFT, RIGHT, FULL OUTER) and provide a real ML use case for each.

**A:** INNER JOIN returns rows where the key exists in both tables—useful for linking user events to verified user profiles where you only want users with complete data. LEFT JOIN keeps all rows from the left table and matches right table data where available—perfect for customer records (left) joined with purchase history (right), preserving customers with no purchases as NULLs, useful for churn prediction. RIGHT JOIN keeps all rows from the right table—less common but useful when you want all products (right) matched with sales data (left), showing products with zero sales. FULL OUTER JOIN keeps all rows from both tables with NULLs where keys don't match—valuable for detecting misalignment between two data sources, such as comparing expected vs. actual feature tables in a feature store. The choice depends on whether you want to keep unmatched records or focus only on matching data.

---

### Q3: What are window functions and how do they differ from GROUP BY? Provide an example.

**A:** Window functions operate on rows related to the current row without collapsing groups—you retain the original row count, whereas GROUP BY collapses rows. For example, `SELECT user_id, purchase_date, purchase_amount, SUM(purchase_amount) OVER (PARTITION BY user_id ORDER BY purchase_date) FROM orders` creates a running sum of purchases per user while keeping all order rows; GROUP BY would only show one row per user with an aggregate total. Window functions are essential for time-series features like LAG() to access the previous purchase, ROW_NUMBER() to rank events within a user, and NTILE() to create spending quintiles. The PARTITION BY defines the groups to window over, and ORDER BY determines the sequence within each group, making window functions more flexible for feature engineering than GROUP BY.

---

### Q4: How would you use ROW_NUMBER, RANK, and DENSE_RANK? When would each be appropriate?

**A:** ROW_NUMBER assigns a unique integer to each row within a partition—useful for sampling, such as `SELECT * FROM events WHERE ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) = 1` to grab each user's most recent event. RANK skips numbers after ties—if two users have the same score, RANK goes 1, 2, 2, 4, which is useful for percentile features but problematic when you need contiguous integers. DENSE_RANK doesn't skip numbers after ties—it returns 1, 2, 2, 3, making it ideal for creating cohort or bucket features without gaps. For churn prediction, you might use DENSE_RANK to create a feature showing "how many distinct engagement periods has this user had?", while ROW_NUMBER would let you select the nth event for feature engineering. The choice depends on whether ties should affect subsequent rankings.

---

### Q5: Explain LAG and LEAD functions with a time-series ML example.

**A:** LAG(column, offset) accesses the previous row's value within a partition, while LEAD(column, offset) accesses the future row's value—both ordered by a specified column (usually timestamp). For example, `SELECT timestamp, clicks, LAG(clicks, 1) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_clicks, clicks - LAG(clicks, 1) OVER(...) AS click_velocity FROM user_events` creates a feature measuring how the click rate changed between consecutive periods. LEAD is similarly useful: `LEAD(purchase_amount, 1) OVER (PARTITION BY user_id ORDER BY timestamp)` lets you create forward-looking features for churn prediction, like "how much did the user spend in the next 30 days?" (requires careful handling to avoid temporal leakage). These functions eliminate the need for expensive self-joins and are critical for building sequential features like transaction velocity, price momentum, or inter-event durations.

---

### Q6: What are Common Table Expressions (CTEs) and how do they improve query readability and reusability?

**A:** CTEs (WITH clauses) create temporary named result sets within a query, improving readability and enabling recursive queries. For example: `WITH recent_orders AS (SELECT * FROM orders WHERE order_date > '2024-01-01'), customer_aggregates AS (SELECT customer_id, COUNT(*) as order_count, SUM(amount) as total_spend FROM recent_orders GROUP BY customer_id) SELECT * FROM customer_aggregates WHERE order_count >= 5` breaks complex logic into named steps. CTEs make queries self-documenting and allow you to reference the same intermediate table multiple times without recomputing it. For ML pipelines, CTEs are essential for layering feature engineering: you can have a CTE for user-level aggregations, another for product-level aggregations, and a final CTE that combines them—much clearer than deeply nested subqueries. Modern databases optimize CTEs efficiently, and some support recursive CTEs for hierarchical data like organizational charts.

---

### Q7: How do you handle NULL values in SQL? What are common pitfalls?

**A:** NULL represents missing or unknown data and behaves specially: NULL = NULL evaluates to NULL (not TRUE), so you must use IS NULL or IS NOT NULL for comparisons. Common pitfalls include: (1) assuming NULL counts in COUNT(*) (it does; use COUNT(column) to exclude NULLs), (2) SUM(column) ignores NULLs, potentially masking missing data, (3) NOT IN (col1, col2, NULL) returns zero rows because of NULL behavior in NOT IN logic. For feature engineering, handle NULLs explicitly: `CASE WHEN value IS NULL THEN 0 ELSE value END` to impute with a default, or `COALESCE(column1, column2, 0)` to use the first non-NULL value. Understanding NULL propagation is critical—for instance, if you have `revenue / users WHERE users IS NULL`, the division fails for those rows, potentially breaking your pipeline. Different SQL dialects handle NULL slightly differently, so always test NULL behavior in your specific database.

---

### Q8: Write a query to create a training dataset with explicit temporal separation to avoid leakage.

**A:** Temporal leakage occurs when future information bleeds into past training data. Here's a correct approach: `SELECT user_id, DATE_TRUNC('day', event_date) as feature_date, SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) as clicks_7d, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases_7d, CASE WHEN DATE_TRUNC('day', purchase_date) > DATE_TRUNC('day', event_date) + INTERVAL '30 days' THEN 1 ELSE 0 END as churn_30d FROM events LEFT JOIN purchases USING (user_id) WHERE event_date BETWEEN '2023-01-01' AND '2023-12-01' GROUP BY 1, 2` computes features (clicks, purchases) in a 7-day window ending on feature_date, then labels the target as churn in the subsequent 30-day window. The key: use LEFT JOIN and explicit date comparisons to ensure features come strictly before the label window. Never mix events from the label period into feature aggregations, and consider how your query would work for a production pipeline predicting tomorrow's churn.

---

### Q9: Explain CASE statements and how you'd use them for feature engineering.

**A:** CASE statements create conditional logic to transform columns: `CASE WHEN condition1 THEN value1 WHEN condition2 THEN value2 ELSE default_value END` allows you to bucket continuous values into categorical features. For example, `CASE WHEN age < 18 THEN 'teen' WHEN age BETWEEN 18 AND 65 THEN 'adult' ELSE 'senior' END` creates age buckets. For ML, CASE is powerful for domain-driven features: `CASE WHEN order_total > (SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY order_total) FROM orders) THEN 'high_spender' ELSE 'regular' END` classifies customers into spending tiers relative to the 75th percentile. You can nest CASE statements for complex logic: `CASE WHEN region = 'US' AND segment = 'premium' THEN 1 ELSE 0 END` creates indicator variables. CASE is often more readable and performant than multiple JOINs when creating categorical features, and it avoids NULL propagation issues inherent in arithmetic operations.

---

### Q10: What are the performance implications of indexing, and how do you write query-efficient SQL?

**A:** Indexes speed up WHERE clause filtering and JOINs by allowing the database to locate rows without scanning every record—a query on an indexed column on a 1M-row table might scan 100 rows vs. 1M. However, indexes slow down INSERT/UPDATE operations because the index must be maintained. For ML pipelines, index frequently-filtered columns (user_id, date, event_type) and join keys, but avoid over-indexing. Query efficiency tips: (1) use EXPLAIN PLAN to check if your query uses indexes and avoids full table scans, (2) push filters down with WHERE before GROUP BY rather than filtering aggregates, (3) avoid functions in WHERE clauses (`WHERE YEAR(date) = 2024` prevents index use; prefer `WHERE date >= '2024-01-01'`), (4) SELECT only needed columns instead of SELECT *, (5) denormalize tables for read-heavy workloads, storing pre-aggregated features in a dedicated table. For large datasets, partition tables by date and filter partitions early to reduce data scanned, a crucial optimization in Snowflake/BigQuery.

---

### Q11: Explain self-joins and provide an ML use case.

**A:** A self-join queries a table against itself, usually to compare rows—for example, finding pairs of users with similar purchase patterns or identifying parent-child relationships in hierarchical data. A concrete example: `SELECT a.user_id, b.user_id, COUNT(*) as common_products FROM user_purchases a JOIN user_purchases b ON a.product_id = b.product_id AND a.user_id < b.user_id GROUP BY a.user_id, b.user_id HAVING COUNT(*) >= 5` finds pairs of users who bought at least 5 common products, useful for a collaborative filtering feature. Another use case: `SELECT emp.name, mgr.name FROM employees emp LEFT JOIN employees mgr ON emp.manager_id = mgr.employee_id` retrieves employee-manager pairs for org hierarchy features. Self-joins are expensive because they scan the table twice, so use them carefully and consider window functions (LAG/LEAD) or CTEs as alternatives when possible. In ML, self-joins often precede similarity computations that feed into clustering or recommendation models.

---

### Q12: How would you pivot and unpivot data in SQL for feature engineering?

**A:** Pivoting converts rows into columns (wide format), while unpivoting does the reverse (long format). Example pivot: `SELECT user_id, SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) as clicks, SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as views, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases FROM events GROUP BY user_id` transforms a long table (one row per event) into wide format with one row per user and three event-type columns. Some databases support PIVOT syntax: `SELECT * FROM (SELECT event_type, user_id FROM events) PIVOT (COUNT(*) FOR event_type IN ('click', 'view', 'purchase'))`. Pivoting is useful when you want features as columns for ML models (models expect columnar data), while unpivoting is useful when combining multiple tables with different granularities. The trade-off: pivoting creates sparse matrices if categories are many, while unpivoting is less efficient for wide formats.

---

### Q13: Describe sampling techniques in SQL and when to use each.

**A:** LIMIT with ORDER BY RANDOM() gives a uniformly random sample but is inefficient on large tables because it must randomize all rows. For big data, use modulo sampling: `SELECT * FROM orders WHERE user_id % 100 = 0` deterministically samples 1% of rows (good for reproducible splits). Stratified sampling preserves class distribution: `SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) as rn FROM data) WHERE rn <= 100` picks 100 samples per class. Reservoir sampling is ideal for streaming: maintain a fixed-size sample of rows seen so far, giving each new row a probability of replacement. For ML, use stratified sampling when classes are imbalanced, modulo sampling for reproducible train/test splits, and random sampling for exploratory analysis. The key consideration: large-scale ML pipelines often train on samples to reduce computation, so understanding which sampling method preserves your data's characteristics is critical.

---

### Q14: How do you validate data quality in SQL?

**A:** Write queries to check for common data quality issues: (1) nulls and cardinality—`SELECT column_name, COUNT(*), COUNT(DISTINCT column_name) FROM table` shows missing values and unique value counts, (2) duplicates—`SELECT column_name, COUNT(*) FROM table GROUP BY column_name HAVING COUNT(*) > 1` identifies duplicate rows, (3) outliers—`SELECT column_name, MIN(), MAX(), AVG(), STDDEV() FROM table` detects suspicious distributions, (4) referential integrity—`SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers)` finds orphaned rows, (5) freshness—`SELECT MAX(last_updated) FROM data_table` checks if data is stale. For ML pipelines, implement these checks as assertions before training: if duplicate rate exceeds 5%, halt the pipeline; if >20% of values are NULL, flag for investigation. Version control your data quality queries so the team knows what "good data" looks like for your specific domain.

---

### Q15: Write a query to build features from an events table with multiple aggregation windows (e.g., 7-day, 30-day, lifetime).

**A:** Use multiple window functions with different ORDER BY frames: `SELECT user_id, DATE_TRUNC('day', timestamp) as feature_date, SUM(amount) OVER (PARTITION BY user_id ORDER BY DATE_TRUNC('day', timestamp) RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW) as spend_7d, SUM(amount) OVER (PARTITION BY user_id ORDER BY DATE_TRUNC('day', timestamp) RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW) as spend_30d, SUM(amount) OVER (PARTITION BY user_id) as spend_lifetime, COUNT(*) OVER (PARTITION BY user_id ORDER BY DATE_TRUNC('day', timestamp) RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW) as transactions_7d FROM events` creates rolling 7/30-day and lifetime aggregates in a single pass. The RANGE BETWEEN clause defines the window; PRECEDING includes past rows, CURRENT ROW sets the endpoint. Aggregate at daily granularity to create one row per user per day, then join to features at prediction time. This approach is efficient (single table scan), handles sparse timelines (RANGE handles date gaps), and scales to billions of rows when computed incrementally and stored in a feature store.

---

## Interview Cheatsheet

**Key Terms:**

- **WHERE:** Filters rows before aggregation (operates on raw data)
- **HAVING:** Filters groups after aggregation (operates on aggregate functions)
- **INNER JOIN:** Returns only rows where keys match in both tables
- **LEFT JOIN:** Keeps all rows from left table, matches right table data
- **FULL OUTER JOIN:** Keeps all rows from both tables with NULLs where unmatched
- **Window Function:** Operates on rows related to current row without collapsing groups
- **ROW_NUMBER:** Assigns unique integer per row; skips numbers after ties
- **RANK:** Assigns rank with gaps after ties (1, 2, 2, 4)
- **DENSE_RANK:** Assigns rank without gaps after ties (1, 2, 2, 3)
- **LAG/LEAD:** Access previous/future row values within partition
- **CTE:** Named temporary result set within query (WITH clause)
- **CASE WHEN:** Conditional logic to create new columns
- **Temporal Leakage:** Using future information to predict the past (data contamination)
- **Pivot:** Transforms rows into columns (long to wide format)
- **Unpivot:** Transforms columns into rows (wide to long format)
- **Modulo Sampling:** Deterministic sampling using hash modulo
- **Stratified Sampling:** Preserves class distribution during sampling

**Rapid-Fire Q&A:**

- **Q:** When should you use COALESCE? **A:** To handle NULLs by returning the first non-NULL value, e.g., `COALESCE(phone, email, user_id)`.
- **Q:** What's the difference between COUNT(*) and COUNT(column)? **A:** COUNT(*) counts all rows including NULLs; COUNT(column) excludes NULLs.
- **Q:** How do you handle NULL in NOT IN? **A:** NOT IN fails with NULLs; use NOT IN (SELECT col FROM table WHERE col IS NOT NULL) or LEFT JOIN anti-pattern.
- **Q:** What is a self-join? **A:** Joining a table to itself to compare rows, e.g., finding user pairs with similar behaviors.
- **Q:** How do you detect duplicates in SQL? **A:** Use GROUP BY and HAVING COUNT(*) > 1 to find rows with duplicate values.
- **Q:** What's the advantage of CTEs? **A:** Improves readability by naming intermediate steps and allows recursive queries.
- **Q:** How do you prevent temporal leakage? **A:** Use explicit date filters to ensure features come strictly before labels; use LEFT JOIN to preserve temporal order.
- **Q:** What does PARTITION BY do in window functions? **A:** Divides rows into groups; window functions compute within each group separately.
- **Q:** When is indexing ineffective? **A:** When filtering on functions (YEAR(date) = 2024) or when selecting many columns.
- **Q:** How do you sample 1% of a 1B-row table efficiently? **A:** Use WHERE user_id % 100 = 0 for deterministic modulo sampling instead of RANDOM().

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

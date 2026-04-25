# Multiple Choice Questions: SQL and Data Manipulation for ML

📺 **Video Lecture:** https://youtu.be/HLPBLmkYABE

## Question 1
You're filtering employee data to find departments where the average salary is above $75,000. Which SQL clause should you use?

A) WHERE AVG(salary) > 75000  
B) HAVING AVG(salary) > 75000  
C) FILTER BY AVG(salary) > 75000  
D) GROUP BY AVG(salary) > 75000  

---

## Question 2
When building a churn prediction model, you need to preserve all customers in your dataset, even those with no purchase history. Which JOIN type is most appropriate?

A) INNER JOIN  
B) LEFT JOIN  
C) RIGHT JOIN  
D) FULL OUTER JOIN  

---

## Question 3
You need to create a feature that counts how many times each user appeared in a transaction table, without reducing the number of rows. Which approach is best?

A) GROUP BY user_id and use COUNT(*)  
B) Use a window function like COUNT(*) OVER (PARTITION BY user_id)  
C) Use a SELF JOIN on user_id  
D) Create a temporary table with pre-aggregated counts  

---

## Question 4
In the context of ranking users by spending within regions, you have two users tied for first place. Using RANK, the next rank would be 3. Using DENSE_RANK, what would the next rank be?

A) 1  
B) 2  
C) 3  
D) 4  

---

## Question 5
You're building a feature that calculates the change in click rates between consecutive days for each user. Which window function would be most useful?

A) ROW_NUMBER()  
B) LAG()  
C) NTILE()  
D) SUM() OVER (PARTITION BY...)  

---

## Question 6
Which of the following statements about Common Table Expressions (CTEs) is FALSE?

A) CTEs improve query readability by breaking complex logic into named steps  
B) CTEs can be referenced multiple times within a single query  
C) CTEs are automatically optimized and never require index support  
D) CTEs support recursive queries for hierarchical data  

---

## Question 7
Consider the following query:
```
SELECT product_id, SUM(amount)
FROM orders
WHERE amount IS NULL
GROUP BY product_id
```
What issue exists with this query?

A) The WHERE clause will filter out all rows, returning an empty result  
B) The SUM() function will fail because NULL values aren't numeric  
C) The GROUP BY is missing a HAVING clause  
D) The query is syntactically correct and will work as intended  

---

## Question 8
You're creating a training dataset where features are computed from events before a label date, and labels are events after that date. What common ML issue are you preventing?

A) Class imbalance  
B) Temporal leakage  
C) Overfitting  
D) Underfitting  

---

## Question 9
You need to segment users into spending tiers based on percentiles: high spenders (top 25%), regular (50-75%), and low (below 50%). Which SQL construct is most suitable for this multi-condition logic?

A) CASE WHEN statement  
B) Multiple INNER JOINs  
C) PIVOT operation  
D) Window function with NTILE()  

---

## Question 10
A query on a 100M-row table with `WHERE YEAR(date_column) = 2024` is running slowly. Why might an index on date_column not help?

A) The index only works on exact matches, not function results  
B) Indexes cannot be applied to date columns  
C) The WHERE clause is filtering too many rows  
D) The query is not using PARTITION BY  

---

## Question 11
You're joining a users table (1M rows) with a user_events table (100M rows) to add user demographic data to events. What performance consideration is most important?

A) Always use RIGHT JOIN to preserve all events  
B) Ensure the join key is indexed on both tables  
C) Use a CROSS JOIN to combine all rows  
D) Group by both table names before joining  

---

## Question 12
You have event data in long format (one row per event_type per user) and need to convert it to wide format (one row per user with columns for each event_type). What operation are you performing?

A) Unpivoting  
B) Pivoting  
C) Normalizing  
D) Stratifying  

---

## Question 13
To build a reproducible training/test split on a 1B-row table without computing expensive random numbers, which sampling approach is best?

A) SELECT * FROM orders LIMIT 10000000  
B) SELECT * FROM orders WHERE RANDOM() < 0.01  
C) SELECT * FROM orders WHERE user_id % 100 = 0  
D) SELECT * FROM orders ORDER BY RANDOM() LIMIT 10000000  

---

## Question 14
You want to validate that no customer_id values in an orders table reference non-existent customers. Which query correctly identifies orphaned rows?

A) `SELECT COUNT(*) FROM orders WHERE customer_id IN (SELECT customer_id FROM customers)`  
B) `SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers WHERE customer_id IS NOT NULL)`  
C) `SELECT COUNT(*) FROM orders LEFT JOIN customers USING (customer_id) WHERE customers.customer_id IS NULL`  
D) `SELECT COUNT(*) FROM orders CROSS JOIN customers WHERE orders.customer_id != customers.customer_id`  

---

## Question 15
You're creating rolling-window features for an events table with columns (user_id, timestamp, amount). You need 7-day, 30-day, and lifetime spend totals per user. What is the most efficient approach?

A) Three separate queries that compute each window, then JOIN results  
B) A single query with three SUM() OVER clauses using RANGE BETWEEN  
C) A self-join to connect events within 7, 30, and lifetime windows  
D) Three separate aggregations using GROUP BY with different date filters  

---

## Answer Key

**Q1: B**  
WHERE filters individual rows before aggregation, while HAVING filters groups after aggregation. To filter based on an aggregate function like AVG(), use HAVING.

**Q2: B**  
LEFT JOIN preserves all rows from the left table (customers) and matches data from the right table (purchases) where available, filling in NULLs for customers with no purchases. This is essential for churn prediction where you need the full customer universe.

**Q3: B**  
Window functions like COUNT(*) OVER (PARTITION BY user_id) compute aggregates without reducing row count, unlike GROUP BY which collapses to one row per user. This is ideal for retaining original transaction rows while adding aggregate features.

**Q4: B**  
DENSE_RANK skips no numbers after ties: 1, 2, 2, 3. RANK would be 1, 2, 2, 4. DENSE_RANK is useful for creating cohort features without gaps.

**Q5: B**  
LAG() accesses the previous row's value, allowing you to compute the difference (current_clicks - previous_clicks) to measure velocity changes. LEAD() would look forward instead.

**Q6: C**  
CTEs are NOT automatically optimized to avoid needing index support. Modern databases optimize CTEs efficiently, but they still benefit from proper indexing strategies. The other statements are true.

**Q7: A**  
WHERE amount IS NULL filters to rows where amount is NULL, then SUM(amount) on NULL values returns NULL per group. This is likely unintended; the query should use IS NOT NULL or COALESCE to handle NULLs properly.

**Q8: B**  
Temporal leakage occurs when future information contaminates historical training features. By ensuring features come strictly before labels in time, you prevent the model from learning impossible future information.

**Q9: D**  
While CASE WHEN could work, NTILE() is explicitly designed for percentile-based bucketing. NTILE(4) divides users into quartiles directly. CASE WHEN with PERCENT_RANK() or aggregate percentiles is more verbose.

**Q10: A**  
Applying a function like YEAR() to an indexed column prevents the index from being used because the database must evaluate the function for every row. Use `WHERE date_column >= '2024-01-01' AND date_column < '2025-01-01'` to leverage the index.

**Q11: B**  
Indexes on both join keys significantly speed up the join by allowing the database to locate matching rows efficiently. This is critical when one table is much larger than the other.

**Q12: B**  
Pivoting transforms data from long format (many rows, few columns) to wide format (few rows, many columns). Unpivoting does the reverse.

**Q13: C**  
Modulo sampling (user_id % 100 = 0) is deterministic, reproducible, and doesn't require computing RANDOM() on all 1B rows. It efficiently samples exactly 1% with no sorting overhead.

**Q14: C**  
A LEFT JOIN anti-pattern correctly identifies orphaned rows: join orders to customers and find rows where customers.customer_id IS NULL (meaning no match was found). Option B has the right idea but NOT IN fails with NULLs unless explicitly excluded.

**Q15: B**  
A single query with three SUM() OVER clauses using RANGE BETWEEN is the most efficient: it scans the table once and computes all windows in parallel. Multiple queries or self-joins would be much slower on large datasets.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

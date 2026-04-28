# Multiple Choice Questions: Unsupervised Learning — Clustering

📺 **Video Lecture:** https://youtu.be/h7cfnNGl9mU


Test your understanding of clustering algorithms and unsupervised learning concepts.

---

**Q1. K-Means clustering requires the user to specify in advance:**

A) The cluster shapes  
B) The labels for each data point  
C) The distance between all clusters  
D) The number of clusters k

---

**Q2. The K-Means algorithm converges when:**

A) All data points belong to one cluster  
B) Cluster assignments no longer change (or change below a threshold)  
C) The centroids move to the origin  
D) The number of iterations reaches exactly 100

---

**Q3. A major limitation of K-Means is that it:**

A) Assumes clusters are spherical and of similar size  
B) Cannot handle more than 2 features  
C) Requires labeled data  
D) Works only with categorical data

---

**Q4. The Elbow Method for choosing k plots:**

A) Within-cluster sum of squares (inertia) vs. number of clusters, looking for a bend  
B) Accuracy vs. number of features  
C) Silhouette score vs. training time  
D) Number of outliers vs. k

---

**Q5. DBSCAN differs from K-Means in that DBSCAN:**

A) Can find arbitrarily shaped clusters and identify noise points as outliers  
B) Requires specifying the number of clusters  
C) Requires labeled data  
D) Always produces spherical clusters

---

**Q6. In DBSCAN, the two main parameters are:**

A) Number of clusters and max iterations  
B) Variance threshold and correlation threshold  
C) k and learning rate  
D) eps (neighborhood radius) and min_samples (minimum points to form a dense region)

---

**Q7. The Silhouette Score measures:**

A) The total distance between all points  
B) The number of clusters  
C) The variance within each cluster  
D) How similar a point is to its own cluster compared to neighboring clusters (ranges from −1 to 1)

---

**Q8. Hierarchical clustering produces:**

A) A single centroid  
B) Only two clusters  
C) Exactly k clusters with no options  
D) A dendrogram showing nested cluster relationships at all levels

---

**Q9. Which linkage criterion in hierarchical clustering tends to produce compact, spherical clusters?**

A) Single linkage (minimum distance between clusters)  
B) Complete linkage (maximum distance between clusters)  
C) No linkage affects cluster shape  
D) Random linkage

---

**Q10. Gaussian Mixture Models (GMMs) differ from K-Means by:**

A) Not requiring any parameters  
B) Modeling each cluster as a Gaussian distribution and providing soft (probabilistic) cluster assignments  
C) Being deterministic  
D) Only working with 1-dimensional data

---

**Q11. The K-Means++ initialization strategy improves K-Means by:**

A) Choosing initial centroids that are spread far apart, reducing chance of poor convergence  
B) Using the same centroid for all clusters  
C) Using random centroid placement  
D) Fixing k to always be 3

---

**Q12. When clusters have very different densities, which algorithm is most appropriate?**

A) K-Means (assumes equal-size spherical clusters)  
B) PCA (dimensionality reduction, not clustering)  
C) DBSCAN (density-based, adapts to local density)  
D) Linear regression

---

**Q13. The "curse of dimensionality" affects clustering because:**

A) Labels become easier to assign  
B) Distance metrics become less meaningful as dimensionality increases  
C) All points become closer together  
D) Algorithms run faster in high dimensions

---

**Q14. In soft clustering (e.g., GMMs), each data point:**

A) Must be on the cluster boundary  
B) Is discarded if ambiguous  
C) Has a probability of belonging to each cluster  
D) Belongs to exactly one cluster

---

**Q15. Mini-batch K-Means is preferred over standard K-Means when:**

A) Perfect cluster assignments are required  
B) The dataset is very small  
C) The data has no structure  
D) The dataset is very large and computational efficiency is needed

---

## Answer Key

**Q1. Answer: D**
K-Means requires specifying k (number of clusters) before running. Choosing k is a model selection problem often addressed with the elbow method, silhouette scores, or gap statistic.

**Q2. Answer: B**
K-Means alternates between assigning points to nearest centroids and updating centroids. It converges when assignments stabilize. Convergence is guaranteed but may be to a local minimum.

**Q3. Answer: A**
K-Means uses Euclidean distance to nearest centroid, which inherently favors spherical, equal-variance clusters. It struggles with elongated, irregular, or differently-sized clusters.

**Q4. Answer: A**
The elbow method plots inertia (within-cluster sum of squares) vs. k. The "elbow" — where adding more clusters gives diminishing returns — suggests the appropriate k.

**Q5. Answer: A**
DBSCAN groups points in dense regions and labels sparse points as noise. It doesn't require specifying k and can discover clusters of arbitrary shape, unlike K-Means.

**Q6. Answer: D**
eps defines the neighborhood radius, and min_samples defines the minimum points needed within eps to form a core point. Together they define what constitutes a dense region.

**Q7. Answer: D**
Silhouette score s = (b−a)/max(a,b) where a = mean intra-cluster distance and b = mean nearest-cluster distance. Values near 1 indicate well-clustered points; near −1 indicates misassignment.

**Q8. Answer: D**
Hierarchical clustering builds a tree (dendrogram) of nested clusters. You can cut the dendrogram at any level to get a different number of clusters, providing flexibility without re-running.

**Q9. Answer: B**
Complete linkage uses the maximum distance between clusters, penalizing elongated shapes and producing compact clusters. Single linkage can produce "chaining" effects with elongated clusters.

**Q10. Answer: B**
GMMs model data as a mixture of Gaussians, providing P(cluster | point) for soft assignments. K-Means is a special case of GMM with equal, spherical covariances and hard assignments.

**Q11. Answer: A**
K-Means++ selects initial centroids with probability proportional to distance from existing centroids, ensuring good spread. This reduces the risk of poor local minima.

**Q12. Answer: C**
DBSCAN's density-based approach naturally handles varying cluster densities. K-Means assumes roughly equal density across clusters and would split dense clusters or merge sparse ones.

**Q13. Answer: B**
In high dimensions, distances between points converge (all points become equidistant), making it hard to distinguish neighbors from non-neighbors. Dimensionality reduction before clustering often helps.

**Q14. Answer: C**
Soft clustering assigns probability distributions over clusters for each point. A point might be 70% cluster A and 30% cluster B, which is more informative than a hard assignment for ambiguous points.

**Q15. Answer: D**
Mini-batch K-Means uses random subsets per iteration instead of the full dataset, dramatically reducing computation time for large datasets with minimal quality loss.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

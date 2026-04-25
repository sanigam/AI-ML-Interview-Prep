# Multiple Choice Questions: Unsupervised Learning — Clustering

📺 **Video Lecture:** https://youtu.be/h7cfnNGl9mU


Test your understanding of clustering algorithms and unsupervised learning concepts.

---

**Q1. K-Means clustering requires the user to specify in advance:**

A) The cluster shapes  
B) The number of clusters k  
C) The labels for each data point  
D) The distance between all clusters

---

**Q2. The K-Means algorithm converges when:**

A) All data points belong to one cluster  
B) Cluster assignments no longer change (or change below a threshold)  
C) The number of iterations reaches exactly 100  
D) The centroids move to the origin

---

**Q3. A major limitation of K-Means is that it:**

A) Works only with categorical data  
B) Assumes clusters are spherical and of similar size  
C) Cannot handle more than 2 features  
D) Requires labeled data

---

**Q4. The Elbow Method for choosing k plots:**

A) Accuracy vs. number of features  
B) Within-cluster sum of squares (inertia) vs. number of clusters, looking for a bend  
C) Number of outliers vs. k  
D) Silhouette score vs. training time

---

**Q5. DBSCAN differs from K-Means in that DBSCAN:**

A) Requires specifying the number of clusters  
B) Can find arbitrarily shaped clusters and identify noise points as outliers  
C) Always produces spherical clusters  
D) Requires labeled data

---

**Q6. In DBSCAN, the two main parameters are:**

A) k and learning rate  
B) eps (neighborhood radius) and min_samples (minimum points to form a dense region)  
C) Number of clusters and max iterations  
D) Variance threshold and correlation threshold

---

**Q7. The Silhouette Score measures:**

A) The number of clusters  
B) How similar a point is to its own cluster compared to neighboring clusters (ranges from −1 to 1)  
C) The total distance between all points  
D) The variance within each cluster

---

**Q8. Hierarchical clustering produces:**

A) Exactly k clusters with no options  
B) A dendrogram showing nested cluster relationships at all levels  
C) Only two clusters  
D) A single centroid

---

**Q9. Which linkage criterion in hierarchical clustering tends to produce compact, spherical clusters?**

A) Single linkage (minimum distance between clusters)  
B) Complete linkage (maximum distance between clusters)  
C) Random linkage  
D) No linkage affects cluster shape

---

**Q10. Gaussian Mixture Models (GMMs) differ from K-Means by:**

A) Being deterministic  
B) Modeling each cluster as a Gaussian distribution and providing soft (probabilistic) cluster assignments  
C) Not requiring any parameters  
D) Only working with 1-dimensional data

---

**Q11. The K-Means++ initialization strategy improves K-Means by:**

A) Using random centroid placement  
B) Choosing initial centroids that are spread far apart, reducing chance of poor convergence  
C) Fixing k to always be 3  
D) Using the same centroid for all clusters

---

**Q12. When clusters have very different densities, which algorithm is most appropriate?**

A) K-Means (assumes equal-size spherical clusters)  
B) DBSCAN (density-based, adapts to local density)  
C) PCA (dimensionality reduction, not clustering)  
D) Linear regression

---

**Q13. The "curse of dimensionality" affects clustering because:**

A) Distance metrics become less meaningful as dimensionality increases  
B) Algorithms run faster in high dimensions  
C) All points become closer together  
D) Labels become easier to assign

---

**Q14. In soft clustering (e.g., GMMs), each data point:**

A) Belongs to exactly one cluster  
B) Has a probability of belonging to each cluster  
C) Is discarded if ambiguous  
D) Must be on the cluster boundary

---

**Q15. Mini-batch K-Means is preferred over standard K-Means when:**

A) The dataset is very small  
B) The dataset is very large and computational efficiency is needed  
C) Perfect cluster assignments are required  
D) The data has no structure

---

## Answer Key

**Q1. Answer: B**
K-Means requires specifying k (number of clusters) before running. Choosing k is a model selection problem often addressed with the elbow method, silhouette scores, or gap statistic.

**Q2. Answer: B**
K-Means alternates between assigning points to nearest centroids and updating centroids. It converges when assignments stabilize. Convergence is guaranteed but may be to a local minimum.

**Q3. Answer: B**
K-Means uses Euclidean distance to nearest centroid, which inherently favors spherical, equal-variance clusters. It struggles with elongated, irregular, or differently-sized clusters.

**Q4. Answer: B**
The elbow method plots inertia (within-cluster sum of squares) vs. k. The "elbow" — where adding more clusters gives diminishing returns — suggests the appropriate k.

**Q5. Answer: B**
DBSCAN groups points in dense regions and labels sparse points as noise. It doesn't require specifying k and can discover clusters of arbitrary shape, unlike K-Means.

**Q6. Answer: B**
eps defines the neighborhood radius, and min_samples defines the minimum points needed within eps to form a core point. Together they define what constitutes a dense region.

**Q7. Answer: B**
Silhouette score s = (b−a)/max(a,b) where a = mean intra-cluster distance and b = mean nearest-cluster distance. Values near 1 indicate well-clustered points; near −1 indicates misassignment.

**Q8. Answer: B**
Hierarchical clustering builds a tree (dendrogram) of nested clusters. You can cut the dendrogram at any level to get a different number of clusters, providing flexibility without re-running.

**Q9. Answer: B**
Complete linkage uses the maximum distance between clusters, penalizing elongated shapes and producing compact clusters. Single linkage can produce "chaining" effects with elongated clusters.

**Q10. Answer: B**
GMMs model data as a mixture of Gaussians, providing P(cluster | point) for soft assignments. K-Means is a special case of GMM with equal, spherical covariances and hard assignments.

**Q11. Answer: B**
K-Means++ selects initial centroids with probability proportional to distance from existing centroids, ensuring good spread. This reduces the risk of poor local minima.

**Q12. Answer: B**
DBSCAN's density-based approach naturally handles varying cluster densities. K-Means assumes roughly equal density across clusters and would split dense clusters or merge sparse ones.

**Q13. Answer: A**
In high dimensions, distances between points converge (all points become equidistant), making it hard to distinguish neighbors from non-neighbors. Dimensionality reduction before clustering often helps.

**Q14. Answer: B**
Soft clustering assigns probability distributions over clusters for each point. A point might be 70% cluster A and 30% cluster B, which is more informative than a hard assignment for ambiguous points.

**Q15. Answer: B**
Mini-batch K-Means uses random subsets per iteration instead of the full dataset, dramatically reducing computation time for large datasets with minimal quality loss.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

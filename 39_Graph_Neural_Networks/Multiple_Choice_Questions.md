# Multiple Choice Questions: Graph Neural Networks

📺 **Video Lecture:** https://youtu.be/2wre5n0cCf8

## Question 1
Which of the following best explains why standard neural networks (MLPs, CNNs, RNNs) are insufficient for processing graph data?

A) Graphs have too many parameters that standard networks cannot handle  
B) Graphs are always too large to fit in GPU memory  
C) Standard networks cannot handle variable node degrees and require permutation-invariant operations on irregular structures  
D) Standard networks cannot represent numerical data

---

## Question 2
In the message passing framework, what is the primary purpose of the aggregation step?

A) To combine information from all neighbors into a single representation that will be used to update the node  
B) To compute the attention weights for each node  
C) To reduce the dimensionality of node features  
D) To remove noisy information from the graph

---

## Question 3
How does the Graph Convolutional Network (GCN) differ from spectral graph convolution approaches?

A) GCN requires labeling all nodes before training  
B) GCN uses eigendeposition of the Laplacian while spectral methods operate directly on graph structure  
C) GCN cannot handle graphs with varying node degrees  
D) GCN applies convolution directly on graph structure (spatial approach) without computing spectral transforms, making it more scalable

---

## Question 4
What is the key innovation of GraphSAGE that enables it to scale to billion-node graphs?

A) It uses matrix factorization instead of message passing  
B) It only trains on a subset of graph nodes  
C) It samples a fixed number of neighbors per node instead of aggregating all neighbors, enabling constant-sized mini-batches  
D) It reduces the embedding dimension by half

---

## Question 5
Which of the following is an advantage of Graph Attention Networks (GAT) compared to GCN?

A) GAT learns adaptive attention weights for each node, allowing it to emphasize important neighbors differently  
B) GAT has lower computational complexity  
C) GAT eliminates the need for message passing  
D) GAT uses fixed aggregation weights, making it more interpretable

---

## Question 6
In link prediction tasks, what is the primary goal?

A) To group similar nodes together  
B) To identify missing or future edges in a graph  
C) To compress the graph representation  
D) To predict node labels based on graph structure

---

## Question 7
What is the purpose of graph pooling and readout operations?

A) To speed up message passing by reducing the number of nodes  
B) To add skip connections between layers  
C) To normalize node features  
D) To aggregate node-level representations into graph-level representations for tasks like graph classification

---

## Question 8
What is the "over-smoothing" problem in deep Graph Neural Networks?

A) As the number of layers increases, node embeddings converge to similar values, limiting expressivity and preventing effective learning  
B) Node embeddings become too sparse in higher layers  
C) The graph becomes disconnected in deeper layers  
D) The gradient signal becomes too large during backpropagation

---

## Question 9
Which of the following is a valid solution to address the over-smoothing problem?

A) Adding skip connections that blend old and new representations, or using Jumping Knowledge networks  
B) Increasing the learning rate during training  
C) Using sigmoid activation instead of ReLU  
D) Removing edges from the graph to reduce connectivity

---

## Question 10
How do heterogeneous graphs differ from homogeneous graphs, and what architectural change does this require?

A) Heterogeneous graphs require converting to homogeneous graphs first  
B) Heterogeneous graphs have multiple node types and edge types, requiring separate learned functions for different (source type, edge type, target type) combinations  
C) Heterogeneous graphs cannot be processed by neural networks  
D) Heterogeneous graphs have weighted edges; weighted aggregation must be used

---

## Question 11
What are the main challenges in training GNNs on temporal/dynamic graphs?

A) Temporal information cannot be represented mathematically  
B) Standard GNNs already handle temporal information perfectly  
C) Storing all graph snapshots is memory-intensive, patterns change over time (concept drift), and new nodes arrive continuously  
D) Temporal graphs do not have edges

---

## Question 12
In knowledge graph embedding, what does the TransE model assume about the relationship between entities?

A) Entities are related through non-linear transformations  
B) The head entity plus the relation vector approximately equals the tail entity (h + r ≈ t), treating relations as translations in embedding space  
C) Entities can only have one relation between them  
D) Relations must be symmetric

---

## Question 13
What is the "cold-start problem" in GNN-based recommendation systems, and how can it be addressed?

A) New users or items with no interaction history cannot aggregate from neighbors; solutions include using content features to initialize embeddings or hybrid approaches combining GNN with content-based methods  
B) The graph becomes disconnected when new nodes are added  
C) Cold-start problems only exist in knowledge graph embeddings  
D) The system learns too slowly when training begins

---

## Question 14
How do graph transformers differ from message-passing GNNs in terms of their attention mechanism?

A) Graph transformers use only local neighborhood attention, while GNNs use global attention  
B) Graph transformers compute attention over all nodes (all-to-all attention) rather than only neighbors, enabling global context but with O(n²) complexity  
C) Graph transformers cannot handle sparse graphs  
D) Message-passing GNNs use attention while graph transformers do not

---

## Question 15
For a billion-node social network, which scalability approach would be most practical for training a GNN-based recommendation system?

A) Train only on a small subset of nodes (e.g., 1000 nodes) and ignore the rest  
B) Convert the graph to a lower-dimensional representation and discard edge information  
C) Store the entire adjacency matrix in memory and train on the full batch  
D) Use mini-batch sampling to select fixed-size neighborhoods per node, distributed training across multiple machines, and importance sampling to reduce variance

---

---

# Answer Key

**Q1: C** - Standard networks fail on graphs due to variable node degrees (irregular structure), the need for permutation-invariant operations, and the inability to directly apply concepts like convolution on non-Euclidean structures. GNNs solve this by aggregating information from neighbors.

**Q2: A** - The aggregation step combines messages from all neighbors into a single representation. This aggregated information is then used in the update step to compute the new node embedding. Aggregation functions (mean, sum, max) are permutation-invariant.

**Q3: D** - GCN uses a spatial approach (operating directly on graph structure) with degree normalization, avoiding expensive eigendecomposition. Spectral methods compute Laplacian eigenvectors (expensive for large graphs), making GCN more practical and scalable.

**Q4: C** - GraphSAGE samples a fixed number of neighbors (e.g., 10 per layer) instead of aggregating all neighbors. This keeps memory and computation constant per mini-batch regardless of node degree, enabling scalable training on billion-node graphs.

**Q5: A** - GAT learns attention weights per edge, allowing the model to dynamically emphasize important neighbors while downweighting less relevant ones. This adaptive mechanism improves expressivity compared to GCN's fixed mean aggregation, though at higher computational cost.

**Q6: B** - Link prediction predicts missing or future edges in a graph. Common approaches include feature-based methods (extracting pair features) and embedding-based methods (learning node embeddings such that connected nodes have similar embeddings).

**Q7: D** - Graph pooling and readout aggregate node-level representations into a single graph-level representation. This is essential for graph-level tasks like graph classification, where we need to make predictions about entire graphs rather than individual nodes.

**Q8: A** - Over-smoothing occurs when deep GNNs cause node embeddings to converge to nearly identical values across the graph. This happens because aggregation averages neighbor features, and after K hops, each node's embedding represents an average of K-hop neighborhoods, losing node-specific information.

**Q9: A** - Skip connections preserve node-specific information by adding the original representation to the updated representation, preventing convergence. Jumping Knowledge networks concatenate representations from all layers, enabling different layers to contribute separately. Both effectively mitigate over-smoothing.

**Q10: B** - Heterogeneous graphs have multiple node types and edge types with different semantics. This requires separate learned aggregation functions for each (source type, edge type, target type) combination, unlike homogeneous graphs where all nodes and edges are treated identically.

**Q11: C** - Temporal graphs evolve over time, creating challenges: (1) storing all snapshots is memory-intensive, (2) patterns change (concept drift), (3) new nodes appear continuously requiring inductive learning, and (4) feedback loops occur when predictions affect future graph structure.

**Q12: B** - TransE models relations as translations in embedding space: the score of a true triple (h, r, t) is high when h + r ≈ t. This is simple but limited for asymmetric relations; RotatE improves by modeling relations as rotations in complex vector space.

**Q13: A** - Cold-start occurs when new users/items have no interaction history and cannot aggregate information from neighbors. Solutions include: (1) using content features (demographics, item category) to initialize embeddings, (2) hybrid approaches combining GNN with content-based filtering, or (3) zero-shot embeddings from metadata.

**Q14: B** - Graph transformers apply self-attention to all node pairs (all-to-all), enabling each node to access global information without explicit message passing steps. However, this results in O(n²) complexity, making them impractical for very large graphs. Message-passing GNNs attend only to neighbors (O(E) complexity).

**Q15: D** - For billion-node graphs, the practical approach combines: (1) mini-batch sampling (fixed neighborhood size like GraphSAGE), (2) distributed training across machines to parallelize, (3) importance sampling to prioritize high-degree neighbors and reduce variance, and (4) GPU acceleration for faster computation.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

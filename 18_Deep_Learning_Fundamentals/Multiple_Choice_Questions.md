# Multiple Choice Questions: Deep Learning Fundamentals

📺 **Video Lecture:** https://youtu.be/96bvudFrdhg


Test your understanding of deep learning fundamentals for AI/ML interviews.

---

**Q1. The perceptron learning algorithm is guaranteed to converge if:**

A) Batch normalization is applied  
B) The learning rate is sufficiently large  
C) The data is linearly separable  
D) The data has more than two features

---

**Q2. The universal approximation theorem states that:**

A) All neural networks converge in finite time  
B) A single hidden layer network can approximate any continuous function given enough neurons  
C) Deep networks always generalize better than shallow networks  
D) Gradient descent always finds the global minimum

---

**Q3. Without nonlinear activation functions, a deep neural network is equivalent to:**

A) A support vector machine  
B) A decision tree  
C) A single linear transformation  
D) A random forest

---

**Q4. ReLU (Rectified Linear Unit) is preferred over sigmoid in hidden layers primarily because:**

A) ReLU avoids the vanishing gradient problem for positive inputs  
B) ReLU always produces non-zero gradients  
C) ReLU is a smoother function than sigmoid  
D) ReLU outputs are bounded between 0 and 1

---

**Q5. Vanishing gradients occur during backpropagation when:**

A) The batch size is too small  
B) The learning rate is too large  
C) The model has too few parameters  
D) Activation function derivatives are consistently less than 1, causing gradients to shrink exponentially through layers

---

**Q6. He initialization (w ~ N(0, √(2/nᵢₙ))) is specifically designed for networks using:**

A) Softmax output layers  
B) Sigmoid activation functions  
C) ReLU activation functions  
D) Tanh activation functions

---

**Q7. Batch normalization normalizes activations across:**

A) All samples in the minibatch for each feature  
B) All features within a single sample  
C) All layers in the network simultaneously  
D) Only the output layer

---

**Q8. Dropout during training randomly deactivates neurons with probability p. At test time:**

A) Only the output layer uses dropout  
B) Dropout is applied identically to training  
C) The dropout probability is doubled  
D) No dropout is applied; outputs are scaled by (1−p)

---

**Q9. The backpropagation algorithm computes gradients efficiently by:**

A) Reusing intermediate partial derivatives via the chain rule (dynamic programming)  
B) Using only first-order Taylor expansions  
C) Approximating gradients with finite differences  
D) Computing all gradients in a single matrix multiplication

---

**Q10. Which loss function is the standard choice for multi-class classification with softmax output?**

A) Hinge loss  
B) Huber loss  
C) Mean Squared Error  
D) Cross-entropy loss

---

**Q11. Exploding gradients are most commonly addressed by:**

A) Increasing the learning rate  
B) Using sigmoid activations in all layers  
C) Removing all hidden layers  
D) Gradient clipping (capping gradient norms)

---

**Q12. Layer normalization differs from batch normalization in that it normalizes:**

A) Across features within each individual sample  
B) Using population statistics rather than batch statistics  
C) Across the minibatch for each feature  
D) Only during the backward pass

---

**Q13. Learning rate warmup is used to:**

A) Train only the output layer first  
B) Replace the need for an optimizer  
C) Increase the learning rate throughout the entire training  
D) Start with a small learning rate and gradually increase it to stabilize early training

---

**Q14. The "dead neuron" problem with ReLU occurs when:**

A) The neuron's bias is initialized to zero  
B) The neuron's output is always 1  
C) The neuron's weights cause it to always output 0 for all inputs, stopping gradient flow  
D) The neuron receives only positive inputs

---

**Q15. Skip connections (as in ResNets) help deep networks train because:**

A) They force each layer to learn completely new features  
B) They provide a direct gradient path, preventing vanishing gradients via the identity shortcut  
C) They reduce the total number of parameters  
D) They eliminate the need for activation functions

---

## Answer Key

**Q1. Answer: C**
The perceptron convergence theorem guarantees convergence only when the training data is linearly separable. For non-separable data (like XOR), the algorithm will never converge.

**Q2. Answer: B**
The theorem proves that a single hidden layer with enough neurons can approximate any continuous function arbitrarily well. However, it doesn't specify how many neurons are needed or how to find the weights.

**Q3. Answer: C**
Without nonlinear activations, stacking multiple linear layers produces a composition of linear transformations, which is itself a single linear transformation. Nonlinearity is essential for learning complex functions.

**Q4. Answer: A**
ReLU has a gradient of 1 for positive inputs, avoiding the vanishing gradient problem that plagues sigmoid (max derivative 0.25). This enables effective training of deep networks.

**Q5. Answer: D**
When activation derivatives are consistently less than 1 (as with sigmoid, max 0.25), multiplying many such values through the chain rule causes gradients to decay exponentially toward zero.

**Q6. Answer: C**
He initialization accounts for the fact that ReLU zeros out negative inputs, so the variance needs to be larger (2/nᵢₙ) compared to Xavier initialization (1/nᵢₙ) to maintain signal through the network.

**Q7. Answer: A**
Batch normalization computes mean and variance across the minibatch dimension for each feature. Layer normalization normalizes across features within each sample.

**Q8. Answer: D**
At test time, all neurons are active but their outputs are scaled by (1−p) to compensate for the higher expected activation compared to training. This is equivalent to averaging over all possible dropout masks.

**Q9. Answer: A**
Backpropagation reuses partial derivatives computed during the backward pass, applying the chain rule layer by layer. This dynamic programming approach makes gradient computation O(parameters), not O(parameters²).

**Q10. Answer: D**
Cross-entropy loss naturally pairs with softmax output for multi-class classification. It is derived from maximum likelihood estimation of categorical distributions.

**Q11. Answer: D**
Gradient clipping caps the gradient norm to a threshold before the weight update, preventing extremely large updates that cause training instability, especially in RNNs.

**Q12. Answer: A**
Layer normalization computes statistics across all features for each individual sample, making it independent of batch size. This is why it's preferred in transformers and RNNs.

**Q13. Answer: D**
Warmup starts with a very small learning rate and gradually increases it, preventing large, potentially destructive weight updates during the initial unstable phase of training.

**Q14. Answer: C**
If weights push all inputs to negative values, ReLU always outputs 0 and its gradient is 0, so the neuron never updates. Leaky ReLU (small slope for negatives) mitigates this issue.

**Q15. Answer: B**
The skip connection y = F(x) + x creates a gradient term of (∂F/∂x + 1). The "+1" provides an unimpeded gradient path through the identity shortcut, preventing vanishing gradients in very deep networks.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

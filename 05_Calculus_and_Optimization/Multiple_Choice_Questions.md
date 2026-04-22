# Multiple Choice Questions: Calculus and Optimization

📺 **Video Lecture:** https://youtu.be/QSOVolxsaWg


Test your understanding of calculus and optimization concepts used in machine learning.

---

**Q1. The gradient of a function f(x₁, x₂, ..., xₙ) points in the direction of:**

A) Steepest decrease of f
B) Steepest increase of f
C) Zero change in f
D) The global minimum of f

---

**Q2. In gradient descent, the update rule is θ = θ − α∇f(θ). The learning rate α controls:**

A) The direction of each step
B) The size of each step toward the minimum
C) The number of features in the model
D) The shape of the loss function

---

**Q3. Stochastic Gradient Descent (SGD) differs from batch gradient descent in that:**

A) SGD always converges faster
B) SGD computes the gradient using a single sample (or mini-batch) instead of the entire dataset
C) SGD does not use a learning rate
D) SGD guarantees finding the global minimum

---

**Q4. A function is convex if:**

A) It has exactly one local minimum
B) Every local minimum is a global minimum, and a line segment between any two points lies on or above the function
C) Its second derivative is always negative
D) It has no critical points

---

**Q5. The Hessian matrix of a function f contains:**

A) The first-order partial derivatives
B) The second-order partial derivatives
C) The eigenvalues of the gradient
D) The function values at critical points

---

**Q6. At a saddle point of a function:**

A) The gradient is non-zero
B) The gradient is zero, but the point is neither a local minimum nor a local maximum
C) The function achieves its global minimum
D) All second derivatives are positive

---

**Q7. The chain rule is essential in deep learning because:**

A) It determines the number of layers in a network
B) It enables backpropagation by computing gradients through nested function compositions
C) It selects the best activation function
D) It determines the learning rate schedule

---

**Q8. Adam optimizer combines ideas from:**

A) SGD and L-BFGS
B) Momentum and RMSProp (adaptive learning rates)
C) Newton's method and gradient descent
D) Random search and grid search

---

**Q9. L2 regularization adds λ||θ||² to the loss function. In terms of optimization, this:**

A) Makes the loss function non-differentiable
B) Adds curvature to the loss surface, shrinking parameters toward zero
C) Removes the gradient entirely
D) Makes the problem non-convex

---

**Q10. Learning rate warmup is a technique where:**

A) The learning rate starts high and decreases linearly
B) The learning rate starts small and gradually increases before following a decay schedule
C) The learning rate is fixed throughout training
D) The learning rate is randomly sampled each epoch

---

**Q11. The vanishing gradient problem occurs when:**

A) The learning rate is too large
B) Gradients become extremely small during backpropagation, slowing or stopping learning in early layers
C) The model has too few parameters
D) The loss function is convex

---

**Q12. Newton's method uses second-order information (Hessian) and converges:**

A) Linearly, same as gradient descent
B) Quadratically near the optimum, but is computationally expensive
C) Slower than SGD in all cases
D) Only for non-convex functions

---

**Q13. What is the purpose of gradient clipping?**

A) To increase the learning rate automatically
B) To prevent exploding gradients by capping gradient magnitudes
C) To make the loss function convex
D) To remove features with small gradients

---

**Q14. In the context of neural networks, a loss function must be:**

A) Convex with respect to the inputs
B) Differentiable with respect to the model parameters (at least almost everywhere)
C) Constant across all training examples
D) Independent of the model architecture

---

**Q15. Mini-batch gradient descent (batch size between 1 and N) is preferred over full-batch because:**

A) It always produces lower loss values
B) It provides a good tradeoff between noisy gradient estimates (which help escape local minima) and computational efficiency
C) It eliminates the need for a learning rate
D) It guarantees convergence in fewer epochs

---

## Answer Key

**Q1. Answer: B**
The gradient ∇f points in the direction of steepest increase. That's why gradient descent uses the negative gradient (−∇f) to move toward the minimum.

**Q2. Answer: B**
The learning rate α determines the step size. Too large causes overshooting and divergence; too small causes slow convergence. Tuning α is critical for training.

**Q3. Answer: B**
SGD approximates the true gradient using one sample or a mini-batch, making each update much faster. The noise from sampling can actually help escape local minima.

**Q4. Answer: B**
A convex function has the property that any local minimum is also a global minimum, and the line segment between any two points on the graph lies on or above the graph. This guarantees gradient descent finds the optimum.

**Q5. Answer: B**
The Hessian H contains all second-order partial derivatives: H_ij = ∂²f/∂xᵢ∂xⱼ. It describes the curvature of the function and is used in second-order optimization methods.

**Q6. Answer: B**
At a saddle point, the gradient is zero but the point is a minimum along some directions and a maximum along others. Saddle points are common in high-dimensional neural network loss surfaces.

**Q7. Answer: B**
The chain rule computes df/dx = (df/dg)(dg/dx) for composed functions. Backpropagation applies this recursively through network layers to compute gradients for all parameters.

**Q8. Answer: B**
Adam combines momentum (exponential moving average of gradients) with RMSProp (adaptive per-parameter learning rates using second moment of gradients), making it effective for a wide range of problems.

**Q9. Answer: B**
L2 regularization adds a quadratic penalty that increases the curvature of the loss surface, pushing parameters toward zero. This reduces overfitting by preventing large weight values.

**Q10. Answer: B**
Learning rate warmup starts with a small learning rate that gradually increases, preventing large destabilizing updates early in training when gradients may be unreliable.

**Q11. Answer: B**
In deep networks, gradients can shrink exponentially as they propagate backward through many layers, especially with sigmoid/tanh activations. ReLU and skip connections help mitigate this.

**Q12. Answer: B**
Newton's method converges quadratically near the optimum (much faster than gradient descent's linear convergence), but computing and inverting the Hessian is O(n³), making it impractical for large models.

**Q13. Answer: B**
Gradient clipping caps the norm or value of gradients to prevent exploding gradients, which cause unstable training. It is commonly used in RNNs and transformer training.

**Q14. Answer: B**
The loss function must be differentiable with respect to model parameters so that gradients can be computed for optimization. Functions like ReLU are technically non-differentiable at a point but have subgradients that work in practice.

**Q15. Answer: B**
Mini-batches balance between the noisy but fast updates of SGD (batch=1) and the accurate but slow updates of full-batch. The noise acts as implicit regularization and helps escape sharp local minima.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*

# Deep Learning Fundamentals

📺 **Video Lecture:** https://youtu.be/96bvudFrdhg

## Interview Anchor
- **Perceptron to MLP:** Evolution from linear classifiers to universal function approximators via hidden layers and nonlinearity
- **Backpropagation:** Efficient gradient computation via chain rule; backbone of neural network training
- **Activation Functions & Optimization:** ReLU, normalization, learning rate scheduling; techniques managing vanishing gradients and enabling deep networks

## Key Concepts Overview
Deep learning's power comes from composing simple building blocks—neurons, layers, activation functions—into architectures learning hierarchical representations. Understanding the fundamentals is critical: why backpropagation works, how activation functions enable nonlinearity, and why gradient descent gets stuck in deep networks (vanishing/exploding gradients, poor initialization). Modern deep learning is engineering—choosing architectures, normalizing appropriately, and tuning learning rates—not just memorizing formulas. Interviewers test whether you understand why things work, not just how to call PyTorch functions. This section covers foundational concepts: perceptrons, MLPs, activation functions, backpropagation, gradient pathologies, weight initialization, and computational graphs. Master these and you can understand any modern architecture (CNNs, Transformers, etc.) as variations on core principles.

---

### Q1: Explain the perceptron algorithm and its limitations. How does it lead to MLPs?

**A:** A **perceptron** is a binary classifier that produces a prediction by thresholding a linear combination of inputs:

```
ŷ = sign( wᵀ·x + b )
```

The classic perceptron learning rule updates weights only when a prediction is wrong:

```
if ŷᵢ ≠ yᵢ:    w ← w + yᵢ · xᵢ
```

The perceptron converges if the data is **linearly separable** but fails entirely on data that isn't — the canonical example is XOR.

This limitation motivated the **multilayer perceptron (MLP)**: stack multiple layers with nonlinear activations between them. Each layer transforms the input into a new space, eventually one where a linear classifier suffices. A two-layer MLP is enough to approximate any continuous function — the universal approximation theorem (more in Q10).

**Example — XOR.** A hidden layer learns two features (each separating one pair of classes); the output layer linearly combines them.

Depth increases expressiveness without exponentially blowing up parameters, which is a big part of why deep networks are powerful. In interviews, the natural narrative is: perceptron → linearly inseparable failure → MLP with hidden layers → universal approximation.

---

### Q2: Describe the architecture of a multilayer perceptron (MLP). What does each layer do?

**A:** An **MLP** has three kinds of layers stacked in sequence:

- **Input layer** — features xᵢ (no computation, passive).
- **Hidden layers** — learn intermediate representations.
- **Output layer** — maps the final hidden state to predictions.

Each non-input layer applies a linear transformation followed by a nonlinear activation:

```
h_j = σ( W_j · h_{j−1}  +  b_j )
```

**Sketch of a 3-layer network:**

```
input x  ∈ ℝ^d
   ↓
hidden 1: h₁ = σ(W₁·x + b₁)              ∈ ℝ^{h₁}
   ↓
hidden 2: h₂ = σ(W₂·h₁ + b₂)             ∈ ℝ^{h₂}
   ↓
output  : y  = f(W_out·h₂ + b_out)        ∈ ℝ (or ℝ^k)
```

**Two architectural knobs:**

- **Depth** (number of layers) — primarily increases expressiveness.
- **Width** (hidden units per layer) — primarily increases capacity.

**Why nonlinear activations are non-negotiable:** without them, stacking layers collapses to a single linear transformation (composition of linear functions is linear). The activations are what give a deep MLP its power.

**Layer-wise interpretation.** In trained networks, early layers tend to learn simple, generic features (edges, textures in images) while later layers compose them into more abstract concepts (objects, categories). Choosing hidden layer sizes is itself a hyperparameter — too small and you underfit; too large and you overfit while wasting compute.

---

### Q3: Explain the activation function concept. Why are they essential?

**A:** **Activation functions** introduce nonlinearity. Without them, a stack of layers collapses to a single linear transformation, since the composition of linear functions is linear. Nonlinearity is what lets a deep network learn rich hierarchical representations.

**Common choices:**

- **Sigmoid:**

  ```
  σ(z) = 1 / (1 + e^(−z))
  ```

  Output in (0, 1). Nice probabilistic interpretation, but its derivative is at most 0.25, which causes severe vanishing gradients in deep networks.

- **Tanh:**

  ```
  tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))
  ```

  Output in (−1, 1). Similar issues to sigmoid, but zero-centered.

- **ReLU (Rectified Linear Unit) — modern default:**

  ```
  f(z) = max(0, z)
  ```

  Simple, efficient, gradient is 1 for active neurons (no vanishing). Failure mode: "dead neurons" — if a neuron is always negative, it never learns.

- **Leaky ReLU** (small leak for negatives):

  ```
  f(z) = max( α·z, z ),    α ≈ 0.01
  ```

  Prevents dead neurons.

- **GELU / Swish** — smoother, better-behaved alternatives common in transformers.

**Activation by layer role:**

- *Hidden layers* — ReLU (or GELU in transformers).
- *Binary classification output* — sigmoid.
- *Multi-class classification output* — softmax.
- *Regression output* — no activation (linear).

In interviews, the key insight is "activations are what make deep networks more than a single linear layer." ReLU's dominance comes from its combination of simplicity and clean gradient flow.

---

### Q4: What is the backpropagation algorithm? Explain how it computes gradients.

**A:** **Backpropagation** efficiently computes gradients of the loss with respect to every parameter using the chain rule, working from the output layer back to the input.

**Two passes:**

- **Forward pass:** compute predictions and the loss.
- **Backward pass:** propagate gradients backward through the network.

**Per-layer derivation.** For a layer computing

```
z_j = W_j · h_{j−1}  +  b_j
h_j = σ(z_j)
```

the chain rule gives:

```
∂L/∂W_j  =  (∂L/∂h_j)  ·  (∂h_j/∂z_j)  ·  (∂z_j/∂W_j)
```

Each factor is local to a single operation and easy to compute.

**The efficiency insight.** The quantity ∂L/∂h_j is shared across all weights in layer j, and it's computed once and reused via dynamic programming. This makes the total cost O(parameters), not O(parameters²) — without this structure, deep networks would be hopelessly expensive to train.

**In modern libraries.** PyTorch, TensorFlow, and JAX implement backprop via automatic differentiation (autograd) — you write the forward pass, the framework records the computation graph, and gradients are computed automatically by walking that graph in reverse.

In interviews, avoid deriving the full backprop equations from scratch. Instead, explain the concept: forward pass computes predictions, backward pass propagates gradients via the chain rule, reusing intermediate results for efficiency. If asked to derive, work through a simple two-layer example.

---

### Q5: What are vanishing and exploding gradients? How do they affect training?

**A:** Backprop multiplies a chain of derivatives across layers, so the magnitude of gradients depends on the *product* of those per-layer derivatives.

**Vanishing gradients.** If each per-layer derivative is < 1, the product shrinks exponentially with depth. With sigmoid (derivative ≤ 0.25):

```
after 10 sigmoid layers:  gradient  ≈  0.25^10  ≈  10⁻⁶
```

Early layers barely update. Symptom: deep networks train much more slowly than shallow ones.

**Exploding gradients.** If per-layer derivatives are > 1, the product grows exponentially. Weight updates become huge, causing oscillation, NaNs, and training divergence.

Both are why training very deep networks was hard before modern techniques. Sigmoid and tanh have small derivatives (≤ 0.25 for sigmoid), so they're prone to vanishing — one of the major reasons modern networks use ReLU, which has derivative 1 for active neurons.

**Mitigations:**

- **Better activations** — ReLU, Leaky ReLU, GELU all have gradients near 1 in the active regime.
- **Careful initialization** — Xavier or He initialization keeps activations and gradients in a reasonable range across layers (see next question).
- **Batch / layer normalization** — stabilizes the distribution of inputs to each layer.
- **Gradient clipping** — caps the gradient norm to prevent explosions, essential for RNNs.
- **Skip connections (ResNets)** — add identity paths so gradients flow directly through the network without depending on the multiplicative chain.

In interviews, explaining vanishing gradients tells the story of *why* deep learning was hard before modern architectural innovations (ReLU, batch norm, residual connections) and why those innovations matter.

---

### Q6: Explain weight initialization. Why is it important?

**A:** **Weight initialization** sets the starting parameters before training. Poor choices cause vanishing gradients, exploding activations, or painfully slow convergence. The goal: keep activations and gradients in reasonable ranges across all layers.

**The failure modes:**

- Weights too small → z ≈ 0 → activations stay near zero (dead ReLU neurons, sigmoid stuck at 0.5).
- Weights too large → z explodes → saturation, NaNs, vanishing gradients.

**Standard initialization schemes** (Normal(μ, σ) below denotes mean μ and *standard deviation* σ; equivalently, variance σ²):

- **Xavier (Glorot)** — for tanh / sigmoid layers. Keeps activation variance constant across layers:

  ```
  w ~ Uniform(  −√( 6 / (n_in + n_out) ),  +√( 6 / (n_in + n_out) ) )
  ```

  Equivalent normal form: variance 2 / (n_in + n_out).

- **He (Kaiming)** — for ReLU layers. Variance 2 / n_in compensates for the half of neurons that ReLU zeroes out:

  ```
  w ~ Normal( 0,  σ = √( 2 / n_in ) )       # variance = 2 / n_in
  ```

- **LeCun** — for SELU and similar:

  ```
  w ~ Normal( 0,  σ = √( 1 / n_in ) )       # variance = 1 / n_in
  ```

Biases are typically initialized to zero.

**Modern practice.** Rely on framework defaults (PyTorch and TensorFlow default to He for ReLU layers). Batch normalization reduces sensitivity to initialization but doesn't replace it.

In interviews, name-drop Xavier and He, explain the intuition (preserve activation variance across layers), and note that bad initialization can prevent learning entirely — not just slow it down.

---

### Q7: What is batch normalization and why does it help training?

**A:** **Batch normalization** normalizes the activations within each minibatch to have zero mean and unit variance, then applies a learned affine transform:

```
x_norm = (x − μ_batch) / √( σ²_batch + ε )

y      = γ · x_norm + β        # γ, β are learned per feature
```

**Benefits:**

- **Reduces internal covariate shift** — the distribution of inputs to each layer doesn't drift as much during training, which stabilizes optimization.
- **Allows higher learning rates** — without divergence.
- **Mild regularization** — minibatch statistics introduce useful noise.
- **Less sensitivity to initialization.**
- **Enables training very deep networks** — combined with skip connections, this unlocked architectures like ResNet.

**Train vs inference behavior:**

- *Training* — compute μ and σ from the current minibatch.
- *Inference* — use exponential moving averages of the training statistics, so predictions don't depend on batch composition.

**Where batch norm shines vs falters.** Batch norm is ubiquitous in CNNs and is a genuine workhorse there. It struggles in two settings: very small batch sizes (statistics become unreliable) and sequence models with variable-length inputs. Transformers usually use **layer normalization** instead (covered in Q8).

In interviews, frame batch norm as something that *fundamentally changes the optimization landscape* — not just a regularizer. Its impact on the trainability of deep networks is what made modern CNNs practical.

---

### Q8: What is layer normalization and when is it preferred over batch normalization?

**A:** **Layer normalization** normalizes across the *feature* dimension within a single sample, rather than across the *batch* dimension within a single feature like batch norm:

```
x_norm = (x − μ_sample) / √( σ²_sample + ε )
```

So each sample computes its own mean and variance from its own features.

**Why this matters:**

- **Works at any batch size** — no dependence on minibatch statistics, no problem at batch size 1.
- **Deterministic** — same computation at train and inference, no moving averages.
- **Handles variable-length sequences** — fits RNNs and transformers naturally.
- **Standard in transformers** — BERT, GPT, T5, and modern LLMs all use layer norm.

**Tradeoff:** the minibatch noise that batch norm provides as a side effect is gone, so layer norm gives slightly less regularization "for free."

**Family of normalization schemes:**

- *Batch norm* — across batch, per feature.
- *Layer norm* — across features, per sample.
- *Group norm* — across groups of features per sample (compromise between the two).
- *Instance norm* — per sample per feature (used in style transfer).

**When to use which:** batch norm for CNNs with reasonable batch sizes; layer norm for transformers and RNNs (or anywhere batch statistics are unreliable). In interviews, the headline is that batch norm's minibatch dependency is problematic for variable-length sequences, and layer norm solves that elegantly.

---

### Q9: Explain dropout and how it prevents overfitting in neural networks.

**A:** **Dropout** randomly zeros out neurons during training. For each neuron, flip a coin with probability p; if heads, zero its activation (and its outgoing connections); otherwise keep it. So only a fraction (1 − p) of neurons are active in any forward pass.

**At test time** dropout is turned off — but to keep the expected activations consistent, outputs are scaled by (1 − p) (or equivalently, "inverted dropout" scales by 1/(1 − p) at training time so no scaling is needed at inference).

**Why it regularizes:**

- **Breaks co-adaptation.** Neurons can't rely on specific partners always being present, so the network has to learn redundant, robust representations.
- **Ensemble interpretation.** Each forward pass uses a different randomly-thinned subnetwork. Training with dropout approximately averages over an exponentially large family of thinned subnetworks (≈ 2ⁿ for n neurons). This is why it acts like a model ensemble — and ensembles reduce variance.

**Variants:**

- **Spatial dropout** — same mask across all spatial positions of a CNN feature map (drops entire feature channels).
- **Variational dropout** — same mask across all time steps of an RNN, avoiding temporal leakage.

**Tuning p.** Typical starting point is 0.5 for fully-connected layers, 0.1–0.3 for convolutional layers. Too low → no regularization; too high → underfit.

**Combining with other regularizers.** Dropout is orthogonal to L1/L2 regularization and batch norm — using them together is normal. (Note: dropout and batch norm can interact awkwardly; in modern practice batch norm or layer norm often replace dropout in some architectures.)

In interviews, the ensemble interpretation is the gold-star explanation — much sharper than vague "dropout breaks co-adaptation" language.

---

### Q10: State the universal approximation theorem. What are its limitations?

**A:** The **universal approximation theorem** says that any continuous function on a compact domain can be approximated arbitrarily closely by a feedforward network with a *single* hidden layer of sufficiently many neurons.

Formally, for any continuous f: ℝⁿ → ℝᵐ and any ε > 0, there exists a width h such that:

```
sup_x  | f(x) − network(x) |  <  ε
```

This is profound — it proves neural networks are theoretically capable of learning any function, given enough neurons.

**Important limitations:**

- **No size guarantee.** The required width h can be exponentially large for some functions, which makes the theorem an existence result rather than a practical recipe.
- **Existence ≠ trainability.** The theorem doesn't say gradient descent will *find* the right weights.
- **Doesn't address generalization.** A network can fit the training data perfectly and still overfit.
- **Continuous functions only.** Real data has noise and discontinuities the theorem doesn't address.
- **Doesn't capture the value of depth.** Many functions require exponentially fewer neurons in deep networks than in shallow ones.

**Why we still go deep.** Shallow networks have universality, but deep networks are *much more parameter-efficient* — depth yields exponential gains in expressiveness for many problems, plus it learns hierarchical, compositional representations that match real-world structure.

In interviews, cite the theorem to justify that neural networks are sufficiently powerful in principle, but emphasize that depth matters in practice for sample efficiency and that the theorem is more philosophy than design guidance.

---

### Q11: Explain loss functions for regression and classification. When do you use each?

**A:** **Regression losses:**

- **Mean Squared Error (MSE):**

  ```
  L = (1/n) · Σᵢ (yᵢ − ŷᵢ)²
  ```

  Penalizes large errors quadratically. Smooth and easy to optimize.

- **Mean Absolute Error (MAE):**

  ```
  L = (1/n) · Σᵢ | yᵢ − ŷᵢ |
  ```

  Linear penalty — much more robust to outliers than MSE.

- **Huber loss:** quadratic near zero, linear for large errors. A practical hybrid of MSE and MAE.

**Classification losses:**

- **Binary cross-entropy:**

  ```
  L = − (1/n) · Σᵢ [ yᵢ · log(ŷᵢ)  +  (1 − yᵢ) · log(1 − ŷᵢ) ]
  ```

  Pairs naturally with sigmoid output.

- **Categorical cross-entropy** (multi-class):

  ```
  L = − (1/n) · Σᵢ Σ_k  y_{i,k} · log(ŷ_{i,k})
  ```

  Pairs naturally with softmax output.

- **Hinge loss** (used by SVMs):

  ```
  L = (1/n) · Σᵢ max( 0,  1 − yᵢ · ŷᵢ )
  ```

  Margin-based, not probabilistic.

- **Focal loss** for imbalanced classification:

  ```
  L = − (1/n) · Σᵢ ( 1 − p_t,ᵢ )^γ · log( p_t,ᵢ )
  ```

  where p_t is the predicted probability of the true class and γ ≥ 0 down-weights easy examples, focusing on hard ones.

**Choice by task:**

- Regression — MSE by default; Huber if outliers; MAE for full robustness.
- Multi-class classification — softmax + categorical cross-entropy.
- Binary classification — sigmoid + binary cross-entropy.
- Heavy class imbalance — focal loss.

In interviews, the key insight is matching the *output activation* to the *loss*: linear output ↔ MSE, sigmoid ↔ binary cross-entropy, softmax ↔ categorical cross-entropy. Understanding why those pairings exist separates competent practitioners.

---

### Q12: What is learning rate scheduling? Why is it important?

**A:** The **learning rate (LR)** α is the step size in gradient descent:

```
w ← w − α · ∇L
```

A fixed α is rarely optimal — too small means slow training, too large means oscillation or divergence. **Learning rate scheduling** varies α over training to get the best of both.

**Common schedules:**

- **Step decay** — reduce α by a factor λ every k epochs:

  ```
  α_new = α_old · λ
  ```

  Simple and surprisingly effective.

- **Exponential decay:**

  ```
  α(t) = α₀ · exp(−λ · t)
  ```

  Smooth, continuous decrease.

- **Cosine annealing** — smoothly anneals from α_initial down to α_final:

  ```
  α(t) = α_final + ½ · (α_initial − α_final) · (1 + cos(π·t/T))
  ```

  Often combined with warm restarts.

- **Warm-up** — linearly ramp α from a small value up to the target over the first few iterations. Stabilizes early training, especially for transformers.

**Adaptive optimizers** (Adam, AdamW, RMSprop) automatically adjust effective per-parameter rates using gradient history. They reduce the need for hand-crafted schedules, which is why "Adam with defaults" is a strong baseline.

**Modern practice for big models:** warmup + cosine decay (sometimes combined with Adam) is the standard recipe for training large transformers and other deep models.

In interviews, frame fixed learning rates as naive — scheduling or adaptive optimization is essential for stable, efficient training.

---

### Q13: Explain computational graphs and automatic differentiation.

**A:** A **computational graph** is a directed acyclic graph (DAG) that represents the forward pass — nodes are operations (multiply, add, activation), edges carry tensors.

**Example.** For z = w · x + b:

```
w, x, b  →  multiply (w × x)  →  add ( + b )  →  z
```

**Automatic differentiation** walks the graph in reverse and applies the chain rule at each node, computing gradients with respect to all inputs. For the example above, given ∂L/∂z:

```
∂L/∂w = (∂L/∂z) · x
∂L/∂b = (∂L/∂z) · 1
```

You never have to derive these by hand — the graph structure makes gradients automatic.

**Two AD modes:**

- **Reverse-mode AD** (backpropagation) — efficient for many inputs and a single scalar output (the typical loss case). This is the standard mode in deep learning.
- **Forward-mode AD** — efficient when there are few inputs and many outputs.

**Two graph styles:**

- **Define-by-run (dynamic)** — PyTorch, JAX. The graph is built fresh during each forward pass; supports control flow naturally.
- **Define-and-run (static)** — original TensorFlow 1.x, ONNX. The graph is constructed up front and then executed; better for compiler optimization.

**Practical benefits:** no error-prone gradient derivations, works for arbitrary architectures, fast iteration on new ideas. In PyTorch, graphs are freed after `backward()` to keep memory usage in check.

In interviews, the simple framing is "autograd lets you focus on architecture, not calculus" — frameworks do the chain rule for you.

---

### Q14: What is gradient clipping and when is it necessary?

**A:** **Gradient clipping** caps the magnitude of gradients during backprop to prevent exploding gradients from blowing up training.

**Two flavors:**

- **Clip by norm** (standard) — if the gradient's L2 norm exceeds threshold τ, scale the entire gradient down proportionally:

  ```
  if ||g|| > τ:    g ← g · ( τ / ||g|| )
  ```

  Direction is preserved; magnitude is bounded.

- **Clip by value** (less common) — cap each individual gradient component to [−τ, τ]. Can bias learning by changing the gradient direction.

**When it's important:**

- **RNNs** — gradients flow through long unrolled graphs in time, so they explode easily. Gradient clipping is essential for stable RNN/LSTM training.
- **Transformers with aggressive learning rates** — attention mechanisms can sometimes produce extreme gradients during training.
- **Any architecture where training divergence has been observed** — adding clipping is a cheap insurance policy.

**Modern practice.** Most deep feedforward networks with batch norm don't really need gradient clipping. RNNs almost always do (`clip_norm` in PyTorch / TF). Clipping by norm is the standard choice; clipping by value is rare.

In interviews, gradient clipping is most commonly cited as an RNN trick — mentioning it shows familiarity with the practical challenges of training recurrent networks.

---

### Q15: What is the relationship between network depth and expressiveness? Can shallow networks approximate any function?

**A:** Shallow networks (one hidden layer) have *universal approximation* in principle (Q10), but **depth gives exponential gains in efficiency**. Many functions that need exponentially many neurons in a single layer can be represented with only polynomially many in a deep network.

**Concrete example:** the parity function (output 1 if an even number of inputs are 1) requires exponential width in shallow networks but only *logarithmic depth* in deep ones.

**Why depth wins.** Deep networks learn **hierarchical representations** — early layers learn primitive features (edges, textures), deeper layers compose them into more abstract concepts (objects, scenes). Real-world functions are largely compositional, and depth exploits that compositional structure to reach high expressiveness with far fewer parameters than a shallow network would need.

**Empirical evidence.** Deep networks (ResNet, VGG, modern transformers) outperform shallow networks of the same total parameter count.

**The downside.** Deep networks are *harder to optimize* — vanishing gradients, sensitivity to initialization, and saddle points all become worse with depth. Modern mitigations (ReLU, batch / layer norm, skip connections, careful initialization) are what make 100+ layer networks trainable.

In interviews, the framing is: shallow networks have universality but deep networks have **sample efficiency and parameter efficiency** — and that's why we always go deep in practice.

---

## Interview Cheatsheet

**Key Terms:**
- **Perceptron:** Linear binary classifier; fails on linearly inseparable problems; motivates MLPs
- **MLP (Multilayer Perceptron):** Stack of layers with nonlinear activations; universal approximator
- **Activation Function:** Introduces nonlinearity; ReLU standard for hidden layers, softmax/sigmoid for outputs
- **ReLU:** f(z) = max(0,z); avoids vanishing gradients; default hidden activation
- **Sigmoid:** σ(z) = 1/(1+e^(-z)); bounded in [0,1]; suffers vanishing gradients
- **Tanh:** Bounded in [-1,1]; derivative larger than sigmoid but still vanishes
- **Backpropagation:** Computes gradients via chain rule in reverse; backbone of neural network training
- **Vanishing Gradient:** Gradients shrink through layers (sigmoid chains); early layers barely update
- **Exploding Gradient:** Gradients grow exponentially; weights update become huge; training unstable
- **Weight Initialization:** Sets starting parameters; Xavier/He initialization prevent vanishing gradients
- **Batch Normalization:** Normalizes layer inputs; reduces internal covariate shift; enables fast training and deep networks
- **Layer Normalization:** Normalizes per-sample over features; preferred in transformers
- **Dropout:** Randomly deactivates neurons during training; approximates ensemble; prevents overfitting
- **Universal Approximation:** Single hidden layer can approximate any continuous function (existence, not practical guide)
- **Loss Functions:** MSE/MAE for regression, cross-entropy for classification
- **Learning Rate Scheduling:** Adapts learning rate over training; improves convergence vs. fixed rate
- **Computational Graph:** DAG representing forward pass; enables automatic differentiation
- **Gradient Clipping:** Caps gradient magnitude; necessary for RNNs, prevents exploding gradients
- **Depth vs. Width:** Deep networks need fewer parameters than shallow for same expressiveness

**Rapid-Fire Q&A:**
- **Q: Why use activation functions?** **A:** Introduce nonlinearity; without them, stacked layers are equivalent to one linear layer
- **Q: Why ReLU over sigmoid?** **A:** Avoids vanishing gradients, simpler computation, empirically superior
- **Q: What does backprop compute?** **A:** Gradients of loss w.r.t. parameters via chain rule; basis for gradient descent
- **Q: How to prevent vanishing gradients?** **A:** Use ReLU, batch norm, careful initialization, skip connections
- **Q: When do gradients explode?** **A:** Deep unrolled graphs (RNNs), large weights, unlucky random seed
- **Q: How to initialize weights?** **A:** Xavier (constant variance) or He (for ReLU); avoid all zeros or extremes
- **Q: Does batch norm always help?** **A:** Usually, but adds computation; some modern architectures (transformers) prefer layer norm
- **Q: How does dropout regularize?** **A:** Breaks co-adaptation; approximates ensemble; noise acts like regularization
- **Q: Can shallow networks approximate everything?** **A:** Yes (universal approx), but deep networks more sample-efficient
- **Q: Learning rate too high?** **A:** Training diverges; use scheduling or adaptive optimizers

---

## Interview Tips
- **Draw computational graphs:** Show understanding by sketching how data flows and gradients backprop
- **Connect to architecture:** Explain that conv layers, attention, etc. are compositional layers; core principles same
- **Discuss trade-offs:** Depth improves expressiveness but complicates optimization; batch norm helps but adds cost
- **Master one activation well:** Deep knowledge of ReLU (when it helps, dead neurons, Leaky ReLU variants) impresses more than name-dropping
- **Relate to your experience:** Mention specific problems you solved (underfitting → deeper network, overfitting → dropout)
- **Prepare derivations:** Be ready to derive cross-entropy loss or backprop for simple 2-layer network; shows theoretical depth
- **Emphasize empirical validation:** Theory says universal approximation; practice says we validate hyperparameters via cross-validation
- **Discuss computational cost:** Training time matters; explain why batch norm and layer norm affect memory/speed

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

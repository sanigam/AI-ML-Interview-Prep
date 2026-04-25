# Multiple Choice Questions: Transfer Learning and Domain Adaptation

📺 **Video Lecture:** https://youtu.be/TqFEGBhRZbk

## Question 1
Early layers of a CNN trained on ImageNet learn features like edges and textures. Why does this make transfer learning effective for medical imaging tasks?
- A) Medical images contain the same objects as ImageNet (dogs, cats, etc.)
- B) Early layers learn universal, low-level features that apply across domains
- C) BERT was specifically designed for medical imaging transfer
- D) ImageNet pre-training prevents overfitting on small medical datasets

**Correct Answer: B**

---

## Question 2
You have a target dataset of 500 labeled images and limited GPU memory. Should you use feature extraction or fine-tuning?
- A) Fine-tuning, because it always produces better results
- B) Feature extraction, because the dataset is small and it's computationally efficient
- C) MAML, because it handles few-shot scenarios best
- D) Zero-shot learning, because you don't have enough data

**Correct Answer: B**

---

## Question 3
In PyTorch, what does setting `layer.requires_grad = False` accomplish during training?
- A) It prevents that layer from receiving gradient updates while gradients still flow through it
- B) It completely blocks gradients from flowing through that layer
- C) It makes the layer output random values
- D) It saves the layer weights to memory

**Correct Answer: A**

---

## Question 4
How does domain adversarial neural networks (DANN) prevent the domain classifier from correctly distinguishing source and target domains?
- A) By using a gradient reversal layer that inverts gradients flowing to the domain classifier
- B) By randomly freezing domain classifier weights
- C) By training only on source domain data
- D) By using a separate encoder for each domain

**Correct Answer: A**

---

## Question 5
Which type of distribution shift occurs when P(X) changes but P(Y|X) remains constant?
- A) Concept drift
- B) Catastrophic forgetting
- C) Covariate shift
- D) Negative transfer

**Correct Answer: C**

---

## Question 6
Maximum Mean Discrepancy (MMD) addresses domain adaptation by:
- A) Using a gradient reversal layer to fool a domain classifier
- B) Measuring and minimizing the distance between source and target feature distributions
- C) Generating synthetic target domain data
- D) Freezing early layers and fine-tuning only the classifier head

**Correct Answer: B**

---

## Question 7
In prototypical networks, how is a new query point classified in a few-shot scenario?
- A) By comparing its embedding to the prototype (mean embedding) of each class and choosing the nearest
- B) By fine-tuning the entire network on the support examples
- C) By using a domain classifier to determine its source
- D) By computing the cosine similarity to the average of all training examples

**Correct Answer: A**

---

## Question 8
What is the key difference between MAML (Model-Agnostic Meta-Learning) and prototypical networks?
- A) MAML is task-agnostic; prototypical networks are model-agnostic
- B) MAML learns weights that adapt quickly to new tasks via gradient steps; prototypical networks learn a fixed metric space
- C) MAML works only for vision; prototypical networks work only for NLP
- D) MAML requires fewer examples than prototypical networks

**Correct Answer: B**

---

## Question 9
Zero-shot learning differs from few-shot learning in that:
- A) Zero-shot uses no examples of the target class, relying on semantic descriptions or attributes
- B) Zero-shot requires more target domain data than few-shot
- C) Zero-shot always outperforms few-shot learning
- D) Zero-shot is only applicable to image classification

**Correct Answer: A**

---

## Question 10
Why does multi-task learning improve generalization on data-limited tasks?
- A) It requires larger batch sizes during training
- B) Gradients from other tasks act as regularization, and shared representations learn richer features
- C) It automatically prevents overfitting by training multiple models
- D) It bypasses the need for domain adaptation

**Correct Answer: B**

---

## Question 11
Catastrophic forgetting occurs when:
- A) A model's performance on previous tasks degrades after fine-tuning on a new task
- B) The pre-training dataset is too small
- C) The learning rate is set too low
- D) Domain adaptation is applied incorrectly

**Correct Answer: A**

---

## Question 12
When fine-tuning BERT for sentiment classification, why should the learning rate be much lower (e.g., 2e-5 to 5e-5) than standard Adam (1e-3)?
- A) To ensure the model memorizes the entire dataset
- B) To preserve pre-trained knowledge while adapting to the downstream task
- C) To speed up convergence
- D) To increase batch size limits

**Correct Answer: B**

---

## Question 13
Negative transfer is most likely to occur in which scenario?
- A) Training on large datasets with many epochs
- B) Source and target domains are very similar
- C) Source and target are highly dissimilar, or fine-tuning is done with too high a learning rate
- D) Using multiple GPUs for distributed training

**Correct Answer: C**

---

## Question 14
Gradual unfreezing during fine-tuning involves:
- A) Training only the task-specific head first, then progressively unfreezing earlier layers
- B) Freezing all layers except the very first layer
- C) Randomly freezing layers during each epoch
- D) Computing gradients but not updating weights for the first few epochs

**Correct Answer: A**

---

## Question 15
Elastic Weight Consolidation (EWC) addresses catastrophic forgetting by:
- A) Storing and replaying examples from previous tasks
- B) Using task-specific adapter modules for each task
- C) Adding a regularization penalty that keeps weights close to their pre-task values, weighted by task importance (Fisher information)
- D) Using a separate model for each task

**Correct Answer: C**

---

## Answer Key

**Q1: B** - Transfer learning works because CNNs learn hierarchical features; early layers learn universal patterns (edges, textures) applicable across domains, while later layers specialize. Medical imaging benefits from these pre-learned low-level features without needing massive domain-specific data.

**Q2: B** - With only 500 images and limited GPU memory, feature extraction is ideal. It's computationally cheap (train only the head), prevents overfitting on small datasets by keeping pre-trained features frozen, and avoids the need for expensive fine-tuning. Fine-tuning would risk overfitting and requires more compute.

**Q3: A** - Setting `requires_grad = False` prevents the optimizer from updating that layer's weights, but gradients still propagate through it for computing downstream activations. This enables efficient feature extraction—you reuse frozen pre-trained features while training task-specific layers.

**Q4: A** - DANN uses a gradient reversal layer (GRL) that multiplies gradients by -lambda before they reach the domain classifier. This reverses the loss signal, forcing the encoder to minimize domain classifier accuracy and learn domain-invariant features. The encoder is in a min-max game with the domain classifier.

**Q5: C** - Covariate shift is defined as P(X) changing while P(Y|X) remains constant—the input distribution shifts but the relationship between features and labels stays the same. This is relatively benign and can be addressed through reweighting, unlike concept drift where the relationship itself changes.

**Q6: B** - MMD directly measures the distance between feature distributions: MMD^2 = ||E_source[phi(X)] - E_target[phi(X)]||^2. Domain adaptation adds MMD as a regularizer to the task loss, forcing the model to learn representations where source and target align, reducing domain discrepancy.

**Q7: A** - Prototypical networks compute a prototype (mean embedding) for each class from the K support examples, then classify a query by measuring its distance to each prototype and selecting the nearest class. This leverages the learned metric space without requiring gradient-based adaptation.

**Q8: B** - MAML learns initial weights optimized for fast few-shot adaptation via inner-loop SGD steps. Prototypical networks learn a fixed metric space for classification. MAML works with any differentiable model (model-agnostic); prototypical networks are task-specific but can be applied to any classification problem (task-agnostic).

**Q9: A** - Zero-shot learning classifies on unseen classes using semantic descriptions or attributes (e.g., "has four legs, is furry") without seeing any examples. Few-shot uses a few examples (5 images of tiger). Zero-shot is harder but more practical for truly novel classes not seen during training.

**Q10: B** - Multi-task learning improves generalization because (1) gradients from unrelated tasks act as regularization, reducing overfitting on data-limited tasks, and (2) shared representations learn richer, more general features that transfer across tasks. It encodes the inductive bias that tasks are related.

**Q11: A** - Catastrophic forgetting (also called interference) occurs when fine-tuning on task2 overwrites weights learned for task1, degrading task1 performance. This is a central challenge in continual learning where models must learn sequences of tasks without forgetting previous knowledge.

**Q12: B** - BERT's pre-trained weights are already optimized on 3.3B words. High learning rates (e.g., 1e-3) cause large weight updates that destroy this knowledge. Low learning rates (2e-5 to 5e-5) make small, careful adjustments, preserving pre-trained representations while adapting to the downstream task.

**Q13: C** - Negative transfer occurs when fine-tuning hurts performance compared to training from scratch. This happens when source and target domains differ drastically (pre-trained knowledge doesn't apply), or when fine-tuning is done carelessly (too-high learning rate, too many epochs). Proper domain adaptation and careful tuning mitigate this.

**Q14: A** - Gradual unfreezing balances stability and adaptation: train the task head first (fast convergence), then progressively thaw earlier layers. This prevents catastrophic forgetting (early layers are frozen initially) while allowing the model to adapt. More nuanced than all-or-nothing fine-tuning or feature extraction.

**Q15: C** - Elastic Weight Consolidation adds a regularization term: loss = task_loss + lambda * sum((theta - theta_old)^2 * Fisher_info). The Fisher information identifies important weights for previous tasks; the penalty keeps those weights close to their pre-task values, preventing catastrophic forgetting while allowing new task learning.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*

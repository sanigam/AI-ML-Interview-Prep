# Multiple Choice Questions: AI Safety, Alignment, and Robustness

📺 **Video Lecture:** https://youtu.be/yvHFDGLkzws

## Question 1
Which of the following best describes the fundamental challenge of the AI alignment problem?

A) Machine learning models are incapable of understanding language semantics  
B) Human values are complex, context-dependent, and difficult to specify formally  
C) Neural networks require too much computational power to train  
D) AI systems are too slow to make real-time decisions

**Correct Answer: B**

---

## Question 2
What is the key distinction between outer alignment and inner alignment?

A) Outer alignment applies to supervised learning; inner alignment applies only to reinforcement learning  
B) Outer alignment concerns system architecture; inner alignment concerns hardware efficiency  
C) They are synonymous terms used interchangeably in AI safety literature  
D) Outer alignment is about specifying the correct objective; inner alignment is about the model actually pursuing that specified objective

**Correct Answer: D**

---

## Question 3
Which of the following is a classic example of specification gaming?

A) A reinforcement learning agent trained to maximize Tetris score learning to pause the game indefinitely  
B) A recommendation system suggesting popular content to users  
C) A language model generating coherent and grammatically correct text  
D) A neural network achieving 99% accuracy on the ImageNet dataset

**Correct Answer: A**

---

## Question 4
The Fast Gradient Sign Method (FGSM) adversarial attack can be best characterized as:

A) A slow but highly effective multi-step attack that requires solving an optimization problem  
B) An attack that works only on image classification tasks, not on NLP models  
C) A certified robustness technique that provides provable guarantees against adversarial examples  
D) A single-step gradient-based attack that is fast but relatively weak compared to iterative methods

**Correct Answer: D**

---

## Question 5
What is the primary advantage of Projected Gradient Descent (PGD) over FGSM?

A) PGD is computationally faster than FGSM  
B) PGD only works on text-based models, making it more versatile  
C) PGD iteratively applies perturbations, finding stronger adversarial examples within an epsilon-ball  
D) PGD requires no gradient information from the model

**Correct Answer: C**

---

## Question 6
Adversarial training improves model robustness but introduces which trade-off?

A) Reduced clean accuracy (models lose 5-10% performance on benign inputs)  
B) Increased training time with no change in accuracy  
C) Inability to classify adversarial examples of any strength  
D) Improved robustness at the cost of model interpretability only

**Correct Answer: A**

---

## Question 7
Certified robustness using randomized smoothing guarantees:

A) That the model cannot be fooled by adversarial examples in real-world deployments  
B) That adversarial training is unnecessary and redundant  
C) 100% immunity to all possible adversarial attacks regardless of perturbation magnitude  
D) Provable robustness to L2 perturbations up to a computed radius, with probabilistic guarantees

**Correct Answer: D**

---

## Question 8
What distinguishes a backdoor attack (BadNets) from regular data poisoning?

A) Regular poisoning is a more sophisticated attack than backdoor attacks  
B) Backdoor attacks use a secret trigger pattern that causes misclassification when inserted, while regular poisoning generally degrades performance across the board  
C) Backdoor attacks are only possible on convolutional neural networks  
D) Backdoor attacks modify all training data; regular poisoning modifies only labels

**Correct Answer: B**

---

## Question 9
Membership inference attacks pose a privacy threat because they can:

A) Modify model weights to extract training examples  
B) Determine whether a specific sample was in a model's training dataset with high accuracy  
C) Only work on small models with fewer than 1 million parameters  
D) Directly access a model's training data

**Correct Answer: B**

---

## Question 10
Which of the following is an example of direct prompt injection on an LLM?

A) A background process modifying the model's weights during inference  
B) A user including harmful instructions directly in their input prompt, attempting to override system instructions  
C) A malicious instruction hidden in a webpage that the model reads  
D) Poisoning the training data with adversarial examples

**Correct Answer: B**

---

## Question 11
Jailbreaking an LLM is effective because:

A) The base model has no knowledge of harmful content  
B) Safety training creates a completely separate model that can be easily bypassed  
C) Modern LLMs have no safety mechanisms whatsoever  
D) The base model contains knowledge to generate harmful content, and safety training only makes refusal more likely; sophisticated attacks can overwhelm this safety layer

**Correct Answer: D**

---

## Question 12
Red teaming for LLMs differs from automated adversarial attacks primarily in that:

A) Red teaming only tests for bias; automated attacks test for all safety categories  
B) Automated attacks are more effective at finding failure modes  
C) Red teaming is fully automated while adversarial attacks require manual input  
D) Red teaming involves creative manual probing by humans, while automated attacks use systematic methods like gradient-based optimization

**Correct Answer: D**

---

## Question 13
Constitutional AI improves upon RLHF by:

A) Making models more vulnerable to jailbreaks  
B) Requiring human raters to label even more examples than standard RLHF  
C) Removing the need for any training data whatsoever  
D) Using a learned critic to evaluate responses against explicit principles, replacing expensive and inconsistent human raters

**Correct Answer: D**

---

## Question 14
Scalable oversight enables human supervision at scale through techniques such as:

A) Sampling outputs, delegation via model-generated summaries, and automated checks for known failure modes  
B) Completely removing human involvement in the decision-making process  
C) Training larger models that require no human oversight  
D) Having humans manually review every single model prediction

**Correct Answer: A**

---

## Question 15
Which of the following represents a fundamental limitation of interpretability as a safety tool?

A) Interpretability guarantees that a model is safe and aligned with human values  
B) Even if individual model components are interpretable, understanding emergent behavior at scale is difficult; additionally, models may learn to "look interpretable" while remaining misaligned  
C) Interpretability is only useful for supervised learning, not reinforcement learning  
D) Interpretability works perfectly for understanding billion-parameter language models

**Correct Answer: B**

---

## Answer Key

**Q1: B** - The core challenge is value specification. Human values are multifaceted, context-dependent, and difficult to formalize into objective functions that AI systems can optimize.

**Q2: D** - This is the critical distinction in alignment research. Outer alignment asks "did we specify the right goal?" while inner alignment asks "will the model pursue the specified goal?" Both must be satisfied for true alignment.

**Q3: A** - The Tetris agent exemplifies specification gaming: it found a technical loophole (pausing indefinitely) that maximizes the specified objective without achieving the intended goal of playing well.

**Q4: D** - FGSM is a fast, single-step method that moves in the gradient direction. It's computationally efficient but produces weaker adversarial examples than iterative methods like PGD or C&W.

**Q5: C** - PGD's iterative nature allows it to find stronger adversarial examples within a perturbation budget by repeatedly optimizing the attack while projecting back into the constraint set.

**Q6: A** - Adversarial training trades clean accuracy for robustness. Models trained to resist adversarial perturbations typically lose 5-10% accuracy on benign inputs, a consistent finding across domains.

**Q7: D** - Randomized smoothing provides formal, provable robustness guarantees computed via the Neyman-Pearson lemma, but only for L2-bounded perturbations and with inherent accuracy-robustness trade-offs.

**Q8: B** - Backdoor attacks are stealthy: they maintain high clean accuracy while embedding a secret trigger, making detection difficult. Regular poisoning more generally degrades performance on all examples.

**Q9: B** - Membership inference exploits the fact that models have lower loss on training examples than test examples. Attackers can infer dataset membership by comparing loss values, raising serious privacy concerns.

**Q10: B** - Direct prompt injection is when a user embeds malicious instructions in their own input, asking the model to ignore prior instructions or perform harmful tasks—an immediate, visible attack.

**Q11: D** - Jailbreaks work because the base model has learned harmful capabilities; safety training only increases refusal probability without eliminating the underlying knowledge. Creative prompting can overcome this.

**Q12: D** - Red teaming is manual and creative (humans think adversarially); automated attacks are systematic and gradient-driven. Both are valuable; red teams often find conceptual failures automated attacks miss.

**Q13: D** - Constitutional AI replaces the human bottleneck of RLHF with a learned critic that evaluates responses against explicit principles, achieving better scalability and consistency.

**Q14: A** - Scalable oversight uses practical techniques: random sampling of outputs, having models summarize their reasoning for human review, and automated detection of known failure patterns.

**Q15: B** - This captures a real limitation: while interpretability is valuable for diagnosis, scaling it to massive models is challenging, and models may exploit interpretability tools by learning to appear aligned while remaining misaligned.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
